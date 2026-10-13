"""Entity catalog: one record per real-world thing discovered in the data."""
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from ctxlayer.common.config import Settings
from ctxlayer.common.context_files import ContextFiles, Step
from ctxlayer.common.decisions import Decisions, Kind, Verdict
from ctxlayer.common.db import quote_ident
from ctxlayer.common.files import read_yaml, write_yaml
from ctxlayer.common.llm import LLM, LLMError, extract_json
from ctxlayer.common.parallel import parallel_map

_METADATA = "metadata"
_INFERRED = "inferred"
_HUMAN = "human"
_LOCATION = "Location"
_DENOMINATION_ROLES = ("currency", "unit")
_LLM_BATCH = 50

_PAREN = re.compile(r"\(([^()]*)\)")
_CODE = re.compile(r"\b[A-Z0-9]{2,}\b")

_ALIAS_PROMPT = """[task: aliases]
For each entity, list other names people use for it: abbreviations, spelling variants,
tags a document might carry. Do not repeat known aliases. Use [] when you know none.

Context from a person (may be empty):
{context}

Entities, one per line as `<id> | <type> | <name> | <known aliases>`:
{lines}

Reply with one JSON object and nothing else:
{{"aliases": {{"<id>": ["<alias>", ...]}}}}"""

_PARENT_PROMPT = """[task: location_parents]
For each location, name the larger place that contains it (a city's region, a region's
country or continent). Use null when it has no larger place or you are not sure.

Context from a person (may be empty):
{context}

Locations, one per line as `<id> | <name> | <known aliases>`:
{lines}

Reply with one JSON object and nothing else:
{{"parents": {{"<id>": {{"name": "<larger place>", "confidence": <0..1>}} | null}}}}"""


@dataclass
class Entity:
    id: str
    type: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    source: str = "metadata"
    confidence: float = 1.0
    needs_review: bool = False


@dataclass
class _Source:
    type: str
    table: str
    columns: list[str]  # [code, label] or [value]


class _AliasReply(BaseModel):
    aliases: dict[str, list[str]]


class _Parent(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)


class _ParentsReply(BaseModel):
    parents: dict[str, _Parent | None]


def parse_aliases(label: str) -> list[str]:
    """Names hidden in a label: "Widget (large) CUR1" -> ["large", "Widget", "CUR1"]."""
    found = [text.strip() for text in _PAREN.findall(label)]
    if "(" in label:
        found.append(label.split("(")[0].strip())
    found.extend(_CODE.findall(label))

    aliases = []
    for text in found:
        if text and text != label and text not in aliases:
            aliases.append(text)
    return aliases


def load_catalog(path: Path) -> list[Entity]:
    return [Entity(**item) for item in read_yaml(path)]


def save_catalog(entities: list[Entity], path: Path) -> None:
    write_yaml(path, [asdict(entity) for entity in entities])


def seed_catalog(
    con, roles: dict, ontology: dict, llm: LLM, settings: Settings, context: ContextFiles | None = None,
) -> list[Entity]:
    """Entities from the distinct values behind each ontology type, then LLM aliases and location parents."""
    sources = [
        _Source(item["name"], source["table"], source["columns"])
        for item in ontology["types"] for source in item.get("sources", [])
    ]

    entities: dict[str, Entity] = {}
    for source in sources:
        _seed(con, source, entities)
    for source in sources:
        if source.type != _LOCATION:
            _link(con, source, sources, ontology, roles, entities)

    workers = settings.get("llm.parallel_calls")
    names = context.for_step(Step.CATALOG) if context else ""
    _add_llm_aliases(list(entities.values()), llm, workers, names)
    _add_location_parents(entities, llm, settings.get("review.confidence_threshold"), workers, names)
    return list(entities.values())


# ----------------------------------------------------------------- decisions

def apply_entity_decisions(entities: list[Entity], decisions: Decisions) -> list[Entity]:
    """Apply `entity` decisions to a freshly seeded catalog; reviewed entities get source human and confidence 1.0.

    A wrong entity is removed. A merged one gives its names and sources to the target, and references to it follow.
    """
    by_id = {entity.id: entity for entity in entities}
    merges: dict[str, str] = {}
    removed: set[str] = set()
    for decision in decisions.of(Kind.ENTITY):
        entity = by_id.get(decision.target["id"])
        if entity is None:
            continue

        decisions.mark_used(decision)
        if decision.verdict is Verdict.WRONG:
            del by_id[entity.id]
            removed.add(entity.id)
            continue

        if decision.verdict is Verdict.FIX:
            _fix_entity(entity, decision.fix, by_id)
            if decision.fix.get("merge_into"):
                merges[entity.id] = decision.fix["merge_into"]
        entity.source, entity.confidence, entity.needs_review = _HUMAN, 1.0, False

    redirect = _merge_all(by_id, merges)
    for entity in by_id.values():
        _redirect_refs(entity, redirect, removed)
    return list(by_id.values())


def _fix_entity(entity: Entity, fix: dict, by_id: dict[str, Entity]) -> None:
    name = fix.get("canonical_name")
    if name and name != entity.canonical_name:
        old, entity.canonical_name = entity.canonical_name, name
        _add_aliases(entity, [old])
    _add_aliases(entity, fix.get("add_aliases", []))

    if "parents" in fix:
        entity.attributes["parents"] = list(fix["parents"])
    # A part_of that would close a loop is ignored, as when the catalog is seeded.
    if fix.get("part_of") and not _reaches(by_id, fix["part_of"], entity.id):
        entity.attributes["part_of"] = fix["part_of"]


def _merge_all(by_id: dict[str, Entity], merges: dict[str, str]) -> dict[str, str]:
    """Merge each entity into its target (names and sources move); returns {merged id: target id}."""
    redirect: dict[str, str] = {}
    for source_id, target_id in sorted(merges.items()):
        # The target may itself have been merged already: follow it.
        while target_id in redirect:
            target_id = redirect[target_id]
        source, target = by_id.get(source_id), by_id.get(target_id)
        if source is None or target is None or source is target:
            continue

        _add_aliases(target, [source.canonical_name, *source.aliases])
        for found_in in source.attributes.get("sources", []):
            if found_in not in target.attributes.setdefault("sources", []):
                target.attributes["sources"].append(found_in)
        del by_id[source_id]
        redirect[source_id] = target_id

    # a -> b -> c becomes a -> c.
    for source_id in redirect:
        while redirect[source_id] in redirect:
            redirect[source_id] = redirect[redirect[source_id]]
    return redirect


def _redirect_refs(entity: Entity, redirect: dict[str, str], removed: set[str]) -> None:
    """Point parents, locations and part_of at the merge target instead of a merged entity; drop removed ones."""
    for key in ("parents", "locations"):
        if key in entity.attributes:
            moved = [redirect.get(ref, ref) for ref in entity.attributes[key]]
            entity.attributes[key] = [ref for ref in dict.fromkeys(moved) if ref != entity.id and ref not in removed]
    part_of = entity.attributes.get("part_of")
    if part_of in redirect:
        entity.attributes["part_of"] = redirect[part_of]
    if entity.attributes.get("part_of") in removed:
        entity.attributes["part_of"] = None


# ------------------------------------------------------------------- seeding

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _distinct(con, table: str, columns: list[str]) -> list[tuple]:
    names = ", ".join(quote_ident(column) for column in columns)
    return con.execute(f"SELECT DISTINCT {names} FROM {quote_ident(table)} ORDER BY ALL").fetchall()


def _identify(source: _Source, values: tuple) -> tuple[str, str | None, int] | None:
    """(name, code, position of the name in the row); None when the row holds no usable value.

    The label is the name, the id a code: ("C1", "Widget") -> ("Widget", "C1", 1).
    """
    cleaned = [str(value).strip() if value is not None else "" for value in values]
    position = -1 if cleaned[-1] else 0
    name = cleaned[position]
    if not _slug(name):
        return None

    code = cleaned[0] if len(cleaned) == 2 and cleaned[0] != name else None
    return name, code, position


def _entity_id(type_name: str, name: str) -> str:
    return f"{_slug(type_name)}:{_slug(name)}"


def _own_id(source: _Source, found: tuple[str, str | None, int]) -> str:
    """Keyed by the code when there is one: two codes sharing a label stay two entities."""
    name, code, _ = found
    return _entity_id(source.type, code or name)


def _add_aliases(entity: Entity, candidates: list[str]) -> list[str]:
    """Append the candidates not yet known (case-insensitive); returns those added."""
    known = {entity.canonical_name.lower(), *(alias.lower() for alias in entity.aliases)}

    added = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate.lower() in known:
            continue
        known.add(candidate.lower())
        entity.aliases.append(candidate)
        added.append(candidate)
    return added


def _seed(con, source: _Source, entities: dict[str, Entity]) -> None:
    for values in _distinct(con, source.table, source.columns):
        found = _identify(source, values)
        if not found:
            continue

        name, code, position = found
        entity_id = _own_id(source, found)
        if entity_id not in entities:
            entities[entity_id] = Entity(entity_id, source.type, name, attributes={"sources": []})
            _add_aliases(entities[entity_id], parse_aliases(name))

        # Every table holding the value is kept, so query tools can filter the right view
        # (e.g. a region present in both a text and a measures table). Codes win over labels.
        key = 0 if code else position
        found_in = {"table": source.table, "column": source.columns[key], "value": values[key]}
        if found_in not in entities[entity_id].attributes["sources"]:
            entities[entity_id].attributes["sources"].append(found_in)

        # "Widget" and "widget" share an id: the second spelling becomes an alias.
        _add_aliases(entities[entity_id], [name, code or ""])


def _link(con, source: _Source, sources: list[_Source], ontology: dict, roles: dict, entities: dict) -> None:
    """Attributes seen on the rows of the entity: parents, locations and currency / unit values."""
    parent_types = {
        relation["to"] for relation in ontology["relations"]
        if relation["name"] == "HAS_PARENT" and relation["from"] == source.type and relation["table"] == source.table
    }

    for other in sources:
        if other.table != source.table:
            continue

        key = "parents" if other.type in parent_types else "locations" if other.type == _LOCATION else None
        if key:
            _attach(con, source, other.columns, lambda row, other=other: _linked_id(other, row), key, entities)

    for column, item in roles["tables"][source.table]["columns"].items():
        if item["role"] in _DENOMINATION_ROLES:
            _attach(con, source, [column], lambda row: str(row[0]), item["role"], entities)


def _linked_id(other: _Source, values: tuple) -> str | None:
    found = _identify(other, values)
    return _own_id(other, found) if found else None


def _attach(con, source: _Source, columns: list[str], to_value: Callable, key: str, entities: dict) -> None:
    width = len(source.columns)

    found: dict[str, set[str]] = {}
    for row in _distinct(con, source.table, source.columns + columns):
        own = _identify(source, row[:width])
        value = to_value(row[width:]) if any(cell is not None for cell in row[width:]) else None
        if own and value:
            found.setdefault(_own_id(source, own), set()).add(value)

    for entity_id, values in found.items():
        attrs = entities[entity_id].attributes
        attrs[key] = sorted(set(attrs.get(key, [])) | values)


# ----------------------------------------------------------------- LLM steps

def _ask(llm: LLM, prompt: str, model: type[BaseModel]):
    """The validated reply, or None when the LLM fails or answers badly."""
    try:
        data = extract_json(llm.complete(prompt).text)
        return model.model_validate(data)
    except (LLMError, ValidationError):
        return None


def _add_llm_aliases(entities: list[Entity], llm: LLM, workers: int, names: str) -> None:
    by_id = {entity.id: entity for entity in entities}

    # Batches are asked `workers` at a time; replies are applied in batch order.
    def ask(batch: list[Entity]):
        lines = [f"{e.id} | {e.type} | {e.canonical_name} | {e.aliases}" for e in batch]
        return _ask(llm, _ALIAS_PROMPT.format(lines="\n".join(lines), context=names), _AliasReply)

    batches = [entities[start:start + _LLM_BATCH] for start in range(0, len(entities), _LLM_BATCH)]
    for reply in parallel_map(ask, batches, workers):
        if not reply:
            continue

        for entity_id, aliases in reply.aliases.items():
            if entity_id not in by_id:
                continue
            added = _add_aliases(by_id[entity_id], aliases)
            if added:
                by_id[entity_id].attributes.setdefault("inferred_aliases", []).extend(added)


def _add_location_parents(
    entities: dict[str, Entity], llm: LLM, threshold: float, workers: int, names: str,
) -> None:
    """PART_OF as attribute `part_of`; a parent missing from the catalog becomes a new Location."""
    locations = [entity for entity in entities.values() if entity.type == _LOCATION]
    by_name = {_slug(name): entity.id for entity in locations for name in [entity.canonical_name, *entity.aliases]}

    def ask(batch: list[Entity]):
        lines = [f"{e.id} | {e.canonical_name} | {e.aliases}" for e in batch]
        return _ask(llm, _PARENT_PROMPT.format(lines="\n".join(lines), context=names), _ParentsReply)

    # Asked `workers` at a time; applied in batch order, since new parents are shared across batches.
    batches = [locations[start:start + _LLM_BATCH] for start in range(0, len(locations), _LLM_BATCH)]
    for reply in parallel_map(ask, batches, workers):
        if not reply:
            continue

        for child_id, parent in reply.parents.items():
            child = entities.get(child_id)
            if not parent or not child or child.type != _LOCATION or not _slug(parent.name):
                continue

            parent_id = by_name.get(_slug(parent.name))
            if parent_id is None:
                parent_id = _entity_id(_LOCATION, parent.name)
                entities[parent_id] = Entity(
                    parent_id, _LOCATION, parent.name.strip(), source=_INFERRED,
                    confidence=parent.confidence, needs_review=parent.confidence < threshold,
                )
                by_name[_slug(parent.name)] = parent_id

            # Skip a link that would close a loop (A part of B part of ... part of A).
            if _reaches(entities, parent_id, child_id):
                continue
            child.attributes["part_of"] = parent_id


def _reaches(entities: dict[str, Entity], start_id: str, target_id: str) -> bool:
    """True when walking part_of from `start_id` (itself included) arrives at `target_id`."""
    seen = set()
    current = start_id
    while current and current not in seen:
        if current == target_id:
            return True
        seen.add(current)
        current = entities[current].attributes.get("part_of") if current in entities else None
    return False
