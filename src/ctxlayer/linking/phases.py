"""Links phase: move-to-catalog mapping, data-derived hierarchy edges and candidate drivers, written under links/."""
import contextlib

from ctxlayer.common.catalog import Entity, load_catalog
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.decisions import Decisions, Kind, Verdict
from ctxlayer.common.files import read_jsonl, read_yaml, write_jsonl
from ctxlayer.common.manifest import write_manifest
from ctxlayer.common.phase_context import PhaseContext
from ctxlayer.common.runlog import event
from ctxlayer.linking._series import id_columns, raw_ids
from ctxlayer.linking.bridge import candidate_drivers
from ctxlayer.structured.semantic_views import open_semantic
from ctxlayer.unstructured.graph_store import GraphStore

_STAGE = "links"
_LINKS_DIR = "links"
_MOVES_FILE = "structured/moves.jsonl"
_SOURCE_LINKING = "linking"
_SOURCE_DATA = "data"
_SOURCE_HUMAN = "human"
_LOCATION = "Location"
_EVENT = "Event"
_DOCUMENT = "Document"
_MOVE = "Move"
_HAS_PARENT = "HAS_PARENT"
_CANDIDATE_DRIVER_OF = "CANDIDATE_DRIVER_OF"


def run_links(pc: PhaseContext) -> dict:
    """Write links/candidate_drivers.jsonl, links/nodes.jsonl (Move nodes) and links/edges.jsonl; returns counts only.

    edges.jsonl holds CANDIDATE_DRIVER_OF (event -> move) and the HAS_PARENT edges the data implies
    (for related hierarchy levels that share measure rows), so parent-level events reach their series' moves.
    """
    out_dir = pc.out_dir
    links_dir = out_dir / _LINKS_DIR
    semantic = read_yaml(out_dir / "semantic.yaml")
    catalog = load_catalog(out_dir / "entities.yaml")
    moves = read_jsonl(out_dir / _MOVES_FILE)
    graph_files = [out_dir / "graph" / "nodes.jsonl", out_dir / "graph" / "edges.jsonl"]
    logger = pc.logger()

    event(logger, "build_links.phase", phase="hierarchy")
    with contextlib.closing(open_semantic(pc.settings)) as con_sem:
        parent_edges = _parent_edges(con_sem, semantic, catalog)
    write_jsonl(links_dir / "edges.jsonl", parent_edges)

    store = GraphStore.load([*graph_files, links_dir / "edges.jsonl"])
    events = [node["props"] for node in store.nodes(_EVENT)]
    days = [node["props"]["date"] for node in store.nodes(_DOCUMENT)]
    source_lookup = _catalog_source_lookup(semantic, catalog)
    raw_lookup = {key[2]: entity_id for key, entity_id in reversed(source_lookup.items())}
    for move in moves:
        move["catalog_id"] = raw_lookup.get(str(move["entity_id"]))

    event(logger, "build_links.phase", phase="drivers")
    drivers = candidate_drivers(moves, store, events, catalog, pc.settings, (min(days), max(days))) if days else []
    drivers = apply_driver_decisions(drivers, {item["event_id"]: item.get("keys", []) for item in events}, pc.decisions)
    write_jsonl(links_dir / "candidate_drivers.jsonl", drivers)
    write_jsonl(links_dir / "nodes.jsonl", [_move_node(move) for move in moves])
    write_jsonl(links_dir / "edges.jsonl", [*parent_edges, *(_driver_edge(row) for row in drivers)])

    counts = {
        "moves": len(moves), "drivers": len(drivers), "moves_with_drivers": len({row["move_id"] for row in drivers}),
        "parent_edges": len(parent_edges),
    }
    event(logger, "build_links.phase", phase="manifest")
    with contextlib.closing(connect_readonly(pc.settings.path("database.path"))) as con:
        write_manifest(out_dir, _STAGE, settings=pc.settings, con=con, models={}, extra=counts)
    return counts


def apply_driver_decisions(drivers: list[dict], key_of: dict[str, list[str]], decisions: Decisions) -> list[dict]:
    """Drop the drivers a `driver` decision calls wrong; confirmed ones get source human.

    `key_of` maps an event id to its member extraction keys: a decision written with any of them matches.
    """
    kept = []
    for driver in drivers:
        found = [
            d for key in key_of.get(driver["event_id"], [])
            if (d := decisions.find(Kind.DRIVER, move_id=driver["move_id"], key=key)) is not None
        ]
        for decision in found:
            decisions.mark_used(decision)
        if any(decision.verdict is Verdict.WRONG for decision in found):
            continue
        kept.append({**driver, "source": _SOURCE_HUMAN} if found else driver)
    return kept


def _catalog_source_lookup(semantic: dict, catalog: list[Entity]) -> dict[tuple[str, str, str], str]:
    """Map each measures source (table, id column, value) to its catalog id."""
    columns = id_columns(semantic)
    lookup: dict[tuple[str, str, str], str] = {}
    for entity in catalog:
        for source in entity.attributes.get("sources", []):
            key = (source["table"], source["column"], str(source["value"]))
            if columns.get(source["table"]) != source["column"]:
                continue
            lookup.setdefault(key, entity.id)
    return lookup


def _parent_edges(con_sem, semantic: dict, catalog: list[Entity]) -> list[dict]:
    """HAS_PARENT from each series entity to every non-place entity whose rows include all of its rows."""
    columns = id_columns(semantic)
    measured = {
        entity.id: set(raw_ids(con_sem, semantic, entity.attributes.get("sources", [])))
        for entity in catalog if entity.type != _LOCATION
    }

    edges = []
    for entity in catalog:
        is_series = any(columns.get(s["table"]) == s["column"] for s in entity.attributes.get("sources", []))
        if not is_series:
            continue
        if not measured[entity.id]:
            continue
        known = set(entity.attributes.get("parents", []))
        for other_id, ids in sorted(measured.items()):
            if other_id == entity.id or other_id in known or not measured[entity.id] <= ids:
                continue
            edges.append(_edge(entity.id, _HAS_PARENT, other_id, {}, 1.0, _SOURCE_DATA, "", ""))
    return edges


def _move_node(move: dict) -> dict:
    name = f"{move['entity_id']} {move['direction']} {move['change_pct']:.1f}% on {move['date']}"
    props = {key: move[key] for key in ("date", "entity_id", "catalog_id", "direction", "change_pct", "z")}
    return {"id": move["id"], "type": _MOVE, "name": name, "props": props}


def _driver_edge(row: dict) -> dict:
    props = {"lag_days": row["lag_days"], "relation": row["relation"], "direction_match": row["direction_match"]}
    return _edge(
        row["event_id"], _CANDIDATE_DRIVER_OF, row["move_id"], props, row["score"],
        row.get("source", _SOURCE_LINKING),
        row["evidence_doc"], row["evidence_quote"],
    )


def _edge(src: str, rel: str, dst: str, props: dict, confidence: float, source: str, doc_id: str, quote: str) -> dict:
    return {
        "src": src, "rel": rel, "dst": dst, "props": props, "confidence": confidence, "source": source,
        "evidence_doc": doc_id, "evidence_quote": quote, "needs_review": False,
    }

