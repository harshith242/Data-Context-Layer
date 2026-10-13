"""Structured phases: each reads its inputs from the artifacts folder and writes only its own outputs."""
import contextlib
import json
from dataclasses import asdict

from ctxlayer.common.catalog import Entity, apply_entity_decisions, load_catalog, save_catalog, seed_catalog
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.decisions import Decisions, Kind, Verdict
from ctxlayer.common.files import atomic_file, read_json, read_yaml, write_jsonl, write_yaml
from ctxlayer.common.manifest import write_manifest
from ctxlayer.common.ontology import apply_type_decisions, draft_ontology
from ctxlayer.common.phase_context import PhaseContext
from ctxlayer.common.profile import profile_database
from ctxlayer.common.review import UNSTRUCTURED_MARKER
from ctxlayer.common.roles import apply_column_decisions, detect_roles
from ctxlayer.common.runlog import event
from ctxlayer.structured.moves import detect_moves
from ctxlayer.structured.semantic_draft import apply_rule_decisions, draft_semantic
from ctxlayer.structured.semantic_views import open_semantic, write_views

_STAGE = "structured"
_PROFILE_FILE = "profile.json"
_ROLES_FILE = "roles.yaml"
_ONTOLOGY_FILE = "ontology.yaml"
_ENTITIES_FILE = "entities.yaml"
_SEMANTIC_FILE = "semantic.yaml"
_MOVES_FILE = "structured/moves.jsonl"
_REVIEW_FILE = "REVIEW.md"
_HUMAN = "human"


def _draft(pc: PhaseContext, name: str, make):
    """The saved pre-decision draft when only the decisions changed since the phase ran (no model call), else make()."""
    path = pc.out_dir / name
    if pc.reuse_draft and path.is_file():
        return read_yaml(path)

    draft = make()
    write_yaml(path, draft)
    return draft


def run_profile(pc: PhaseContext) -> dict:
    event(pc.logger(), "build_structured.phase", phase="profile")
    with contextlib.closing(connect_readonly(pc.settings.path("database.path"))) as con:
        profile = profile_database(con, pc.settings)

    # Written in column order (not sorted): roles and the ontology follow it, and extraction prompts follow them.
    with atomic_file(pc.out_dir / _PROFILE_FILE) as temp:
        temp.write_text(json.dumps(profile, indent=2, default=str))
    return {"tables": len(profile["tables"])}


def run_roles(pc: PhaseContext) -> dict:
    event(pc.logger(), "build_structured.phase", phase="roles")
    profile = read_json(pc.out_dir / _PROFILE_FILE)
    draft = _draft(pc, "roles.draft.yaml", lambda: detect_roles(profile, pc.llm, pc.context, pc.settings))
    roles = apply_column_decisions(draft, pc.decisions)

    write_yaml(pc.out_dir / _ROLES_FILE, roles)
    _write_review(pc)
    return {"tables": len(roles["tables"]), "flagged_roles": len(_flagged_roles(roles))}


def run_ontology(pc: PhaseContext) -> dict:
    event(pc.logger(), "build_structured.phase", phase="ontology")
    roles = read_yaml(pc.out_dir / _ROLES_FILE)
    profile = read_json(pc.out_dir / _PROFILE_FILE)
    ontology = apply_type_decisions(draft_ontology(roles, profile, pc.llm, pc.context), pc.decisions)

    # Event types have their own phase and file.
    ontology.pop("event_types", None)
    write_yaml(pc.out_dir / _ONTOLOGY_FILE, ontology)
    return {"types": len(ontology["types"]), "relations": len(ontology["relations"])}


def run_catalog(pc: PhaseContext) -> dict:
    event(pc.logger(), "build_structured.phase", phase="catalog")
    roles = read_yaml(pc.out_dir / _ROLES_FILE)
    ontology = read_yaml(pc.out_dir / _ONTOLOGY_FILE)
    def seed() -> list[dict]:
        with contextlib.closing(connect_readonly(pc.settings.path("database.path"))) as con:
            return [asdict(entity) for entity in seed_catalog(con, roles, ontology, pc.llm, pc.settings, pc.context)]

    draft = _draft(pc, "entities.draft.yaml", seed)
    entities = apply_entity_decisions([Entity(**item) for item in draft], pc.decisions)

    save_catalog(entities, pc.out_dir / _ENTITIES_FILE)
    _write_review(pc)
    return {"entities": len(entities)}


def run_semantic(pc: PhaseContext) -> dict:
    event(pc.logger(), "build_structured.phase", phase="semantic")
    roles = read_yaml(pc.out_dir / _ROLES_FILE)
    profile = read_json(pc.out_dir / _PROFILE_FILE)
    def draft_it() -> dict:
        with contextlib.closing(connect_readonly(pc.settings.path("database.path"))) as con:
            return draft_semantic(roles, profile, con, pc.llm, pc.context)

    semantic = apply_rule_decisions(_draft(pc, "semantic.draft.yaml", draft_it), pc.decisions)

    write_yaml(pc.out_dir / _SEMANTIC_FILE, semantic)
    _write_review(pc)
    return {"semantic_tables": len(semantic["tables"])}


def run_views(pc: PhaseContext) -> dict:
    event(pc.logger(), "build_structured.phase", phase="views")
    semantic = read_yaml(pc.out_dir / _SEMANTIC_FILE)
    write_views(semantic, pc.settings)
    return {"views": len(semantic["tables"])}


def run_moves(pc: PhaseContext) -> dict:
    """Detect moves, refresh REVIEW.md and the structured manifest; returns the counts of the whole structured build."""
    event(pc.logger(), "build_structured.phase", phase="moves")
    semantic = read_yaml(pc.out_dir / _SEMANTIC_FILE)

    # Moves are read through the semantic views, so they need the views written first.
    with contextlib.closing(open_semantic(pc.settings)) as con_sem:
        moves = apply_move_decisions(detect_moves(con_sem, semantic, pc.settings), pc.decisions)
    write_jsonl(pc.out_dir / _MOVES_FILE, moves)

    review_count = _write_review(pc)
    event(pc.logger(), "build_structured.phase", phase="manifest")
    with contextlib.closing(connect_readonly(pc.settings.path("database.path"))) as con:
        write_manifest(
            pc.out_dir, _STAGE, settings=pc.settings, con=con,
            models={"llm": type(pc.llm).__name__, "embeddings": pc.settings.get("embeddings.model")},
        )

    roles = read_yaml(pc.out_dir / _ROLES_FILE)
    return {
        "tables": len(roles["tables"]), "entities": len(load_catalog(pc.out_dir / _ENTITIES_FILE)),
        "moves": len(moves), "needs_review": review_count,
    }


def apply_move_decisions(moves: list[dict], decisions: Decisions) -> list[dict]:
    """Drop the moves a `move` decision calls wrong (their drivers go with them); confirmed ones get source human."""
    kept = []
    for move in moves:
        decision = decisions.find(Kind.MOVE, id=move["id"])
        if decision is None:
            kept.append(move)
            continue

        decisions.mark_used(decision)
        if decision.verdict is Verdict.WRONG:
            continue
        kept.append({**move, "source": _HUMAN})
    return kept


def _flagged_roles(roles: dict) -> list[tuple[str, float]]:
    return [
        (f"{table}.{column}: {spec['role']}", spec["confidence"])
        for table, table_spec in roles["tables"].items()
        for column, spec in table_spec["columns"].items()
        if spec.get("needs_review")
    ]


def _flagged_entities(entities: list[Entity]) -> list[tuple[str, float]]:
    return [(f"{entity.id} ({entity.type})", entity.confidence) for entity in entities if entity.needs_review]


def _semantic_gaps(semantic: dict) -> list[tuple[str, float]]:
    """Semantic-level breakages that disable metrics, e.g. "tbl: no primary_measure" (confidence 0)."""
    gaps = [("now is null", 0.0)] if not semantic.get("now", {}).get("value") else []
    for table, spec in semantic["tables"].items():
        if spec["kind"] != "measures":
            continue
        missing = [key for key in ("entity", "time", "primary_measure") if not spec.get(key)]
        if "scenario" in spec and not spec["scenario"].get("values"):
            missing.append("scenario labels")
        gaps += [(f"{table}: no {key}", 0.0) for key in missing]
    return gaps


def _write_review(pc: PhaseContext) -> int:
    """Rewrite the structured sections of REVIEW.md from the files built so far; returns the flagged count.

    The unstructured sections after the marker are kept. A section whose file is not built yet is left out.
    """
    sections = []
    for title, name, flagged in (
        ("Roles", _ROLES_FILE, lambda path: _flagged_roles(read_yaml(path))),
        ("Entities", _ENTITIES_FILE, lambda path: _flagged_entities(load_catalog(path))),
        ("Semantic rules", _SEMANTIC_FILE, lambda path: _semantic_gaps(read_yaml(path))),
    ):
        if (pc.out_dir / name).is_file():
            sections.append((title, flagged(pc.out_dir / name)))

    # Ids and confidences only, lowest confidence first, e.g. "- tbl.col: dimension (0.55)".
    lines = ["# Review", ""]
    for title, items in sections:
        lines += [f"## {title}", ""]
        lines += [f"- {label} ({confidence:.2f})" for label, confidence in sorted(items, key=lambda item: item[1])]
        lines.append("" if items else "none\n")

    path = pc.out_dir / _REVIEW_FILE
    text = "\n".join(lines) + "\n"
    existing = path.read_text() if path.is_file() else ""
    if UNSTRUCTURED_MARKER in existing:
        text = text.rstrip("\n") + "\n\n" + UNSTRUCTURED_MARKER + existing.split(UNSTRUCTURED_MARKER, 1)[1]
    path.write_text(text)
    return sum(len(items) for _, items in sections)
