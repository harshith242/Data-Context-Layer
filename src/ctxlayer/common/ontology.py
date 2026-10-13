"""Ontology draft: entity types from column roles, relations from functional dependencies."""
import hashlib
import json
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ctxlayer.common.context_files import ContextFiles, Step
from ctxlayer.common.decisions import Decisions, Kind, Verdict
from ctxlayer.common.files import read_yaml, write_yaml
from ctxlayer.common.llm import LLM, LLMError, extract_json

EVENT_TYPES_FILE = "event_types.yaml"
_ONTOLOGY_FILE = "ontology.yaml"

BUILTIN_TYPES = ["Document", "Event", "Organization", "Location", "Move"]

_BUILTIN_DESCRIPTIONS = {
    "Document": "A text document such as an article or report.",
    "Event": "Something that happened and may explain a move.",
    "Organization": "A company or institution.",
    "Location": "A place such as a region or country.",
    "Move": "A large change in a measured value.",
}
_LOCATION = "Location"
_DOCUMENT = "Document"
_EVENT = "Event"
_HUMAN = "human"
_OTHER = "other"

_ENTITY = "entity"
_HIERARCHY = "hierarchy level"
_LOCATION_KIND = "location"
_MEASURES = "measures"
_MAX_SAMPLES = 5

# Document and event relations of spec 6.2 that do not depend on the data: (name, from, to, properties).
_FIXED_RELATIONS = [
    ("PART_OF", _LOCATION, _LOCATION, []),
    ("REPORTS", _DOCUMENT, _EVENT, []),
    ("LOCATED_IN", _EVENT, _LOCATION, []),
    ("INVOLVES", _EVENT, "Organization", []),
    ("CANDIDATE_DRIVER_OF", _EVENT, "Move", ["score", "lag_days"]),
]

_PROMPT = """[task: ontology]
Name the entity types of a database. Each group below is one kind of real-world thing.

Reply with one JSON object and nothing else. Keys are the group keys below; names are
PascalCase singular nouns, unique, and none of {builtins}:
{{"types": {{"<group key>": {{"name": "<TypeName>", "description": "<one short sentence>"}}}}}}

Context from the data owner (may be empty):
{context}

Groups, one JSON object per line:
{groups}
"""


class _TypeReply(BaseModel):
    name: str = Field(pattern=r"^[A-Z][A-Za-z0-9]*$")
    description: str = ""


@dataclass
class _Group:
    kind: str
    table: str
    columns: list[str]

    @property
    def key(self) -> str:
        return f"{self.table}.{self.columns[0]}"


def draft_ontology(roles: dict, profile: dict, llm: LLM, context: ContextFiles) -> dict:
    """ontology.yaml content: types with their source columns, relations, empty event_types."""
    groups = _groups(roles)
    induced = [group for group in groups if group.kind != _LOCATION_KIND]
    names = _type_names(induced, profile, llm, _context_text(context))

    # Location is the only builtin that has columns behind it.
    location_sources = [_source(group) for group in groups if group.kind == _LOCATION_KIND]
    types = []
    for name in BUILTIN_TYPES:
        item = {"name": name, "builtin": True, "description": _BUILTIN_DESCRIPTIONS[name]}
        if name == _LOCATION and location_sources:
            item["sources"] = location_sources
        types.append(item)

    for group in induced:
        name, description = names[group.key]
        types.append({"name": name, "builtin": False, "description": description, "sources": [_source(group)]})

    type_of = {group.key: names[group.key][0] for group in induced}
    return {
        "version": _version(roles, profile, _context_text(context)),
        "types": types,
        "relations": _relations(roles, profile, groups, type_of),
        "event_types": [],
    }


def apply_type_decisions(ontology: dict, decisions: Decisions) -> dict:
    """Apply `entity_type` and `relation` decisions to a fresh draft; reviewed items get source human.

    A wrong type is dropped with the relations that touch it; a renamed one is renamed in its relations.
    Builtin types are not reviewable.
    """
    renames: dict[str, str] = {}
    dropped: set[str] = set()
    taken = {item["name"] for item in ontology["types"]}
    types = []
    for item in ontology["types"]:
        decision = None if item.get("builtin") else decisions.find(Kind.ENTITY_TYPE, name=item["name"])
        if decision is None or (decision.verdict is Verdict.FIX and not _renamable(decision.fix, taken)):
            types.append(item)
            continue

        decisions.mark_used(decision)
        if decision.verdict is Verdict.WRONG:
            dropped.add(item["name"])
            continue

        fixed = {**item, **decision.fix, "source": _HUMAN}
        if fixed["name"] != item["name"]:
            renames[item["name"]] = fixed["name"]
            taken.add(fixed["name"])
        types.append(fixed)

    relations = []
    for relation in ontology["relations"]:
        if {relation["from"], relation["to"]} & dropped:
            continue

        decision = decisions.find(Kind.RELATION, **{"from": relation["from"], "name": relation["name"], "to": relation["to"]})
        if decision is not None:
            decisions.mark_used(decision)
        if decision is not None and decision.verdict is Verdict.WRONG:
            continue

        fixed = {**relation, "from": renames.get(relation["from"], relation["from"]), "to": renames.get(relation["to"], relation["to"])}
        relations.append({**fixed, "source": _HUMAN} if decision is not None else fixed)
    return {**ontology, "types": types, "relations": relations}


def _renamable(fix: dict, taken: set[str]) -> bool:
    """A fix may not give a type a name another type already has."""
    return fix.get("name", "") not in taken


def apply_event_type_decisions(event_types: list[dict], decisions: Decisions) -> tuple[list[dict], dict[str, str]]:
    """(types for the extraction prompt, {induced name: final name}) after the `event_type` decisions.

    Renames and merges only relabel: the type keeps its name in the prompt, so no document is extracted again,
    and `relabels` renames its events later. A wrong type merges into "other". A description fix or a new type
    (a decision on an unknown name, with a description) does change the prompt.
    """
    types = [dict(item) for item in event_types]
    names = {item["name"] for item in types}
    relabels: dict[str, str] = {}
    for decision in decisions.of(Kind.EVENT_TYPE):
        name = decision.target["name"]
        item = next((candidate for candidate in types if candidate["name"] == name), None)
        if item is None:
            if _add_event_type(types, decision, names):
                decisions.mark_used(decision)
            continue
        if name == _OTHER:
            continue

        if decision.verdict is Verdict.WRONG:
            relabels[name] = _OTHER
        elif decision.verdict is Verdict.FIX and not _fix_event_type(item, decision.fix, relabels, names):
            continue
        item["source"] = _HUMAN
        decisions.mark_used(decision)

    return types, {name: _final_name(name, relabels) for name in relabels}


def _add_event_type(types: list[dict], decision, names: set[str]) -> bool:
    description = decision.fix.get("description")
    if decision.verdict is not Verdict.FIX or not description:
        return False

    added = {"name": decision.target["name"], "description": description, "source": _HUMAN}
    # "other" stays last.
    position = len(types) - 1 if types and types[-1]["name"] == _OTHER else len(types)
    types.insert(position, added)
    names.add(added["name"])
    return True


def _fix_event_type(item: dict, fix: dict, relabels: dict[str, str], names: set[str]) -> bool:
    """Apply a fix to `item`; False when it names a type that does not exist or one already taken."""
    target = fix.get("merge_into") or fix.get("name")
    if fix.get("merge_into") and fix["merge_into"] not in names:
        return False
    if fix.get("name") and not fix.get("merge_into") and fix["name"] in names:
        return False

    if "description" in fix:
        item["description"] = fix["description"]
    if target and target != item["name"]:
        relabels[item["name"]] = target
    return True


def _final_name(name: str, relabels: dict[str, str]) -> str:
    # A renamed type may itself be renamed or merged: follow the chain, e.g. a -> b -> other.
    seen = {name}
    while name in relabels and relabels[name] not in seen:
        name = relabels[name]
        seen.add(name)
    return name


def load_ontology(out_dir: Path, prompt: bool = False) -> dict:
    """ontology.yaml with `event_types` from event_types.yaml; a build made before that file keeps its own.

    `event_types` are the types events end up with (renames and merges applied), or with `prompt` the types the
    extraction prompt lists.
    """
    ontology = read_yaml(out_dir / _ONTOLOGY_FILE)
    saved = out_dir / EVENT_TYPES_FILE
    if saved.is_file():
        types = read_yaml(saved)
        ontology["event_types"] = types["event_types"] if prompt else types.get("final_types", types["event_types"])
    return {**ontology, "event_types": ontology.get("event_types", [])}


def final_event_types(types: list[dict], relabels: dict[str, str]) -> list[dict]:
    """`types` as events carry them: a renamed type under its new name, a merged-away one dropped."""
    names = {item["name"] for item in types}
    kept: dict[str, dict] = {}
    for item in types:
        name = relabels.get(item["name"], item["name"])
        if name == item["name"] or name not in names:
            kept.setdefault(name, {**item, "name": name})
    return list(kept.values())


def _context_text(context: ContextFiles) -> str:
    return context.for_step(Step.ONTOLOGY)


def _version(roles: dict, profile: dict, context_text: str) -> str:
    # Only the context this step reads: editing another heading (News) must not re-induce event types.
    # Review state (confidence, needs_review, source) is left out: confirming a role must not re-induce event types.
    reviewed = {"confidence", "needs_review", "source"}
    tables = {
        table: {
            **info,
            "columns": {
                column: {key: value for key, value in item.items() if key not in reviewed}
                for column, item in info["columns"].items()
            },
        }
        for table, info in roles["tables"].items()
    }
    return _digest({"roles": {**roles, "tables": tables}, "profile": _without_samples(profile), "context": context_text})


def _digest(inputs: dict) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()


def restamp_legacy_version(out_dir: Path, roles: dict, profile: dict, context: ContextFiles) -> None:
    """Give an ontology.yaml stamped by the pre-plan formula the current version (and its event_types.yaml too).

    The ontology and its event types are unchanged, so a later ontology run does not induce event types again.
    """
    path = out_dir / _ONTOLOGY_FILE
    ontology = read_yaml(path)
    legacy = _digest({"roles": roles, "profile": _without_samples(profile), "context": [None, {}]})
    if ontology.get("version") != legacy:
        return

    ontology["version"] = _version(roles, profile, _context_text(context))
    write_yaml(path, ontology)
    saved_path = out_dir / EVENT_TYPES_FILE
    if saved_path.is_file():
        saved = read_yaml(saved_path)
        saved["ontology_version"] = ontology["version"]
        write_yaml(saved_path, saved)


def _without_samples(value):
    if isinstance(value, dict):
        return {key: _without_samples(item) for key, item in value.items() if key != "samples"}
    if isinstance(value, list):
        return [_without_samples(item) for item in value]
    return value


def _source(group: _Group) -> dict:
    return {"table": group.table, "columns": group.columns}


def _groups(roles: dict) -> list[_Group]:
    """One group per entity id/label pair, per hierarchy level column and per location column."""
    groups = []
    for table, info in roles["tables"].items():
        columns = {role: [] for role in ("entity_id", "entity_label", "hierarchy_level", "location")}
        for column, item in info["columns"].items():
            if item["role"] in columns:
                columns[item["role"]].append(column)

        # (item_code, item_title) -> one entity group
        for pair in zip_longest(columns["entity_id"], columns["entity_label"]):
            groups.append(_Group(_ENTITY, table, [column for column in pair if column]))
        groups.extend(_Group(_HIERARCHY, table, [column]) for column in columns["hierarchy_level"])
        groups.extend(_Group(_LOCATION_KIND, table, [column]) for column in columns["location"])
    return groups


def _type_names(groups: list[_Group], profile: dict, llm: LLM, context_text: str) -> dict[str, tuple[str, str]]:
    """{group key: (type name, description)}: the LLM's valid names, else names from the column."""
    replies = _ask(groups, profile, llm, context_text)

    taken = set(BUILTIN_TYPES)
    names = {}
    for group in groups:
        reply = replies.get(group.key)
        name = reply.name if reply and reply.name not in taken else _unique(_pascal(group.columns[0]), taken)
        taken.add(name)
        names[group.key] = (name, reply.description if reply else "")
    return names


def _ask(groups: list[_Group], profile: dict, llm: LLM, context_text: str) -> dict[str, _TypeReply]:
    """Valid entries of the LLM's reply, keyed by group key; {} when the reply is unusable."""
    lines = [json.dumps(_describe(group, profile), default=str) for group in groups]
    prompt = _PROMPT.format(builtins=BUILTIN_TYPES, context=context_text or "(none)", groups="\n".join(lines))

    try:
        data = extract_json(llm.complete(prompt).text)
    except LLMError:
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("types"), dict):
        return {}

    replies = {}
    for key, raw in data["types"].items():
        try:
            replies[key] = _TypeReply.model_validate(raw)
        except ValidationError:
            continue
    return replies


def _describe(group: _Group, profile: dict) -> dict:
    stats = profile["tables"][group.table]["columns"]
    return {
        "key": group.key,
        "kind": group.kind,
        "columns": [
            {"name": column, "samples": stats[column].get("samples", [])[:_MAX_SAMPLES]}
            for column in group.columns
        ],
    }


def _pascal(column: str) -> str:
    words = "".join(char if char.isalnum() else " " for char in column).split()
    name = "".join(word.capitalize() for word in words)
    return name if name[:1].isalpha() else f"Type{name}"


def _unique(name: str, taken: set[str]) -> str:
    candidate, number = name, 1
    while candidate in taken:
        number += 1
        candidate = f"{name}{number}"
    return candidate


def _relations(roles: dict, profile: dict, groups: list[_Group], type_of: dict[str, str]) -> list[dict]:
    relations = []

    # Child -> parent type per table, from "a determines b" pairs: level_a -> level_b.
    for table in roles["tables"]:
        table_groups = [group for group in groups if group.table == table and group.kind != _LOCATION_KIND]
        edges = _parent_edges(profile["tables"][table]["fds"], table_groups, type_of)
        relations.extend({"name": "HAS_PARENT", "from": a, "to": b, "table": table} for a, b in edges)

        if not any(group.table == table and group.kind == _LOCATION_KIND for group in groups):
            continue
        relations.extend(
            {"name": "AT_LOCATION", "from": type_of[group.key], "to": _LOCATION, "table": table}
            for group in table_groups
        )

    targets = [*type_of.values(), _LOCATION]
    measured = _measured_types(roles, groups, type_of)
    relations.extend({"name": "TAGGED_WITH", "from": _DOCUMENT, "to": to} for to in targets)
    relations.extend({"name": "MENTIONS", "from": _DOCUMENT, "to": to} for to in targets)
    relations.extend({"name": "AFFECTS", "from": _EVENT, "to": to, "properties": ["direction", "confidence"]} for to in measured)

    for name, source, target, properties in _FIXED_RELATIONS:
        relation = {"name": name, "from": source, "to": target}
        if properties:
            relation["properties"] = properties
        relations.append(relation)
    return relations


def _parent_edges(fds: list, table_groups: list[_Group], type_of: dict[str, str]) -> list[tuple[str, str]]:
    """Direct (child type, parent type) pairs; a pair implied by two others is dropped."""
    owner = {column: type_of[group.key] for group in table_groups for column in group.columns}

    edges = []
    for child, parent in fds:
        edge = (owner.get(child), owner.get(parent))
        if None in edge or edge[0] == edge[1] or edge in edges:
            continue
        edges.append(edge)

    # sku -> brand -> category implies sku -> category: keep the two direct links only.
    return [
        (a, c) for a, c in edges
        if not any((a, b) in edges and (b, c) in edges for b in {x for _, x in edges})
    ]


def _measured_types(roles: dict, groups: list[_Group], type_of: dict[str, str]) -> list[str]:
    return [
        type_of[group.key] for group in groups
        if group.kind == _ENTITY and roles["tables"][group.table].get("kind") == _MEASURES
    ]
