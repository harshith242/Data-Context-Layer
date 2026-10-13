"""Resolve document tags and extracted events to catalog entities, and merge duplicate events across documents."""
import hashlib
import json
import re
from dataclasses import replace
from datetime import date

from ctxlayer.common.catalog import Entity
from ctxlayer.common.context_files import ContextFiles, Step
from ctxlayer.common.decisions import Decisions, Kind, Verdict
from ctxlayer.common.embeddings import Embedder
from ctxlayer.common.llm import LLM, extract_json
from ctxlayer.common.resolve import Method, Resolver, Status
from ctxlayer.unstructured._names import DOC_PREFIX
from ctxlayer.unstructured.documents import Document
from ctxlayer.unstructured.extract import MIN_AFFECT_CONFIDENCE, Extraction

_TAGGED_WITH = "TAGGED_WITH"
_SOURCE_METADATA = "metadata"
_SOURCE_INFERRED = "inferred"
_SOURCE_HUMAN = "human"
_NEW_ENTITY_CONFIDENCE = 0.5
# The tag value is on the document for certain; the doubt is the new entity, so the edge stays flagged.
_TAG_VALUE_CONFIDENCE = 1.0
_UNTYPED = "tag"
_LOCATION_TYPE = "Location"
_CONTEXT_CHARS = 200  # sentence / title text shown to the resolution judge per name
_DOC_TAG_ROLE = "doc_tag"
_TEXT_KIND = "text"
_EVENT_PREFIX = "event:"
_EVENT_ID_CHARS = 10
_DATE_STATED = "stated"
_UNKNOWN_DIRECTION = "unknown"

_TAG_TYPES_PROMPT = """[task: tag_types]
Each column below tags documents of table "{table}". For each column, name the entity type
its values belong to, or null when none of the types fits.

Entity types:
{types}

Context from a person (may be empty):
{context}

Tag columns:
{columns}

Reply with JSON only: {{"tag_types": {{"<column>": "<entity type or null>"}}}}"""


def resolve_tags(
    docs: list[Document], resolver: Resolver, roles: dict, ontology: dict, llm: LLM,
    context: ContextFiles | None = None,
) -> tuple[list[dict], list[Entity]]:
    """(TAGGED_WITH edges, new flagged entities) for every tag value of `docs`."""
    tag_types = _tag_types(roles, ontology, llm, context or ContextFiles())
    new_entities: dict[str, Entity] = {}
    edges = []

    # Every (tag, type) resolved in one batched pass before the per-document loop; the judge sees the tag's
    # column and the title of the first document carrying it, e.g. "tag column region, in: Strike hits mills".
    seen = {}
    for doc in docs:
        for column, values in doc.tags.items():
            for value in values:
                seen.setdefault(value, f"tag column {column}, in: {doc.title[:_CONTEXT_CHARS]}")
    resolver.prefetch([
        (value, tag_types.get(column)) for doc in docs for column, values in doc.tags.items() for value in values
    ], seen)

    for doc in docs:
        for column, values in doc.tags.items():
            for value in values:
                edges.append(_tag_edge(doc, column, value, tag_types.get(column), resolver, new_entities))

    return edges, list(new_entities.values())


def resolve_extractions(extractions: dict[str, Extraction], resolver: Resolver, ontology: dict | None = None) -> dict:
    """{doc_id: {"mentions": [...], "events": [...]}} with catalog ids; unmatched names keep `entity_id: None`."""
    affect_types = [
        relation["to"] for relation in (ontology or {}).get("relations", []) if relation["name"] == "AFFECTS"
    ]
    # Derived measurable entities can be absent from AFFECTS when the source has no direct FD for them.
    affect_types += [item["name"] for item in (ontology or {}).get("types", []) if not item.get("builtin", False)]
    # Every affect (once, whole catalog) and location resolved in one batched pass first; the judge sees the
    # sentence each name came from (first occurrence), e.g. "Storms cut widget output for two weeks."
    events = [event for extraction in extractions.values() for event in extraction.events]
    seen = {}
    for event in events:
        for affect in event.affects:
            seen.setdefault(affect.entity, (affect.quote or event.evidence)[:_CONTEXT_CHARS])
        for name in event.locations:
            seen.setdefault(name, event.evidence[:_CONTEXT_CHARS])
    resolver.prefetch([
        *((affect.entity, None) for event in events for affect in event.affects),
        *((name, _LOCATION_TYPE) for event in events for name in event.locations),
    ], seen)
    return {
        doc_id: _resolve_extraction(extraction, resolver, affect_types)
        for doc_id, extraction in extractions.items()
    }


def dedupe_events(
    events: list[dict], embedder: Embedder, days: int, similarity: float, decisions: Decisions | None = None,
) -> list[dict]:
    """Canonical events: resolved events (each with `doc_id`) of one type that share an entity or
    location, fall within `days` of each other and have summaries with cosine >= `similarity`.

    `decisions`: same_event ones regroup the events before they are merged; event ones fix or drop the results."""
    if not events:
        return []

    vectors = embedder.embed_passages([event["summary"] for event in events])
    cosines = vectors @ vectors.T

    # Union-find over mergeable pairs, so A~B and B~C put A, B and C in one event.
    parent = list(range(len(events)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in range(len(events)):
        for second in range(first + 1, len(events)):
            if _mergeable(events[first], events[second], float(cosines[first, second]), days, similarity):
                parent[find(second)] = find(first)

    member_positions: dict[str, int] = {}
    groups: dict[int, list[tuple[int, dict]]] = {}
    for index, event in enumerate(events):
        position = member_positions.get(event["doc_id"], 0)
        member_positions[event["doc_id"]] = position + 1
        groups.setdefault(find(index), []).append((position, event))

    if decisions is not None:
        positions = {id(event): position for group in groups.values() for position, event in group}
        regrouped = apply_same_event_decisions([[event for _, event in group] for group in groups.values()], decisions)
        groups = dict(enumerate([(positions[id(event)], event) for event in group] for group in regrouped))

    canonical = [_canonical(group) for group in groups.values()]
    if decisions is not None:
        canonical = apply_event_decisions(canonical, decisions)
    return sorted(canonical, key=lambda event: (event["date"] or "", event["event_id"]))


def apply_same_event_decisions(groups: list[list[dict]], decisions: Decisions) -> list[list[dict]]:
    """Regroup events (each with its extraction `key`): correct merges the two events' groups, wrong splits them.

    A decision matches when both keys exist. For a wrong one that shares a group, the second key as written in the
    decision leaves into a group of its own.
    """
    groups = [list(group) for group in groups]
    found = sorted(decisions.of(Kind.SAME_EVENT), key=lambda decision: decision.verdict is Verdict.WRONG)
    for decision in found:
        first, second = sorted(map(str, decision.target["keys"]))
        leaving = str(decision.target["keys"][1])
        at_first, at_second = _group_of(groups, first), _group_of(groups, second)
        if at_first is None or at_second is None or decision.verdict is Verdict.FIX:
            continue

        decisions.mark_used(decision)
        if decision.verdict is Verdict.CORRECT and at_first != at_second:
            groups[at_first].extend(groups.pop(at_second))
        if decision.verdict is Verdict.WRONG and at_first == at_second:
            event = next(item for item in groups[at_second] if item.get("key") == leaving)
            groups[at_second].remove(event)
            groups.append([event])
    return groups


def _group_of(groups: list[list[dict]], key: str) -> int | None:
    return next((index for index, group in enumerate(groups) if any(item.get("key") == key for item in group)), None)


def apply_event_decisions(events: list[dict], decisions: Decisions) -> list[dict]:
    """Apply `event` decisions to canonical events, found by any of their member `keys`.

    Wrong drops the event; correct and fix unflag it (confidence 1.0, source human); a fix sets type, date, status
    and the direction of the named entity's affects (or adds an affect).
    """
    kept = []
    for event in events:
        found = [d for key in event["keys"] if (d := decisions.find(Kind.EVENT, key=key)) is not None]
        for decision in found:
            decisions.mark_used(decision)
        if any(decision.verdict is Verdict.WRONG for decision in found):
            continue

        for decision in found:
            if decision.verdict is Verdict.FIX:
                _fix_event(event, decision.fix)
        if found:
            event.update(confidence=1.0, needs_review=False, source=_SOURCE_HUMAN)
        kept.append(event)
    return kept


def _fix_event(event: dict, fix: dict) -> None:
    for field in ("type", "status"):
        if field in fix:
            event[field] = fix[field]
    if "date" in fix:
        event["date"], event["date_source"] = fix["date"], _DATE_STATED

    for change in fix.get("affects", []):
        _fix_affect(event, change)


def _fix_affect(event: dict, change: dict) -> None:
    """Set the direction of the affects of `change["entity"]` (those of `change["aspect"]`, if given), or add one."""
    entity, aspect = change["entity"], change.get("aspect")
    same = [affect for affect in event["affects"] if affect["entity_id"] == entity and aspect in (None, affect["aspect"])]
    if not same:
        same = [{
            "entity_id": entity, "aspect": aspect or "", "direction": _UNKNOWN_DIRECTION, "quote": "",
            "doc_id": event["doc_ids"][0],
        }]
        event["affects"].append(same[0])
        event["entity_ids"] = sorted({*event["entity_ids"], entity})

    for affect in same:
        affect.update(confidence=1.0, match_score=1.0, source=_SOURCE_HUMAN)
        if "direction" in change:
            affect["direction"] = change["direction"]


def apply_new_entity_decisions(
    entities_text: list[Entity], tag_edges: list[dict], decisions: Decisions, known: frozenset[str] = frozenset(),
) -> tuple[list[Entity], list[dict]]:
    """Apply `new_entity` decisions to the tag-made entities and their tag edges.

    Correct unflags the entity and its edges, wrong drops them, a merge points the edges at the target (a new entity
    or one of `known`) and drops the entity. Reviewed items get source human.
    """
    new_ids = {entity.id for entity in entities_text}
    kept, dropped, merged, approved = [], set(), {}, set()
    for entity in entities_text:
        decision = decisions.find(Kind.NEW_ENTITY, id=entity.id)
        target = decision.fix.get("merge_into") if decision else None
        if decision is None or (decision.verdict is Verdict.FIX and target not in (known | new_ids) - {entity.id}):
            kept.append(entity)
            continue

        decisions.mark_used(decision)
        if decision.verdict is Verdict.WRONG:
            dropped.add(entity.id)
        elif decision.verdict is Verdict.FIX:
            merged[entity.id] = target
        else:
            approved.add(entity.id)
            kept.append(replace(entity, source=_SOURCE_HUMAN, confidence=1.0, needs_review=False))

    return kept, _follow_edges(tag_edges, dropped, merged, approved)


def _follow_edges(tag_edges: list[dict], dropped: set[str], merged: dict[str, str], approved: set[str]) -> list[dict]:
    edges = []
    for edge in tag_edges:
        target, seen = edge["dst"], set()
        while target in merged and target not in seen:
            seen.add(target)
            target = merged[target]
        if edge["dst"] in dropped or target in dropped:
            continue

        if target != edge["dst"] or target in approved:
            edge = {**edge, "dst": target, "needs_review": False, "source": _SOURCE_HUMAN}
        edges.append(edge)
    return edges


# --------------------------------------------------------------------- tags

def _tag_types(roles: dict, ontology: dict, llm: LLM, context: ContextFiles) -> dict[str, str | None]:
    """{tag column: ontology type name or None}, one LLM call per text table."""
    type_names = [item["name"] for item in ontology["types"]]
    result: dict[str, str | None] = {}

    for table, spec in roles["tables"].items():
        if spec.get("kind") != _TEXT_KIND:
            continue
        columns = {
            spec_column.get("logical_name", column): spec_column.get("description", "")
            for column, spec_column in spec["columns"].items()
            if spec_column["role"] == _DOC_TAG_ROLE
        }
        if not columns:
            continue

        prompt = _TAG_TYPES_PROMPT.format(
            table=table,
            types="\n".join(f"- {name}" for name in type_names),
            columns="\n".join(f"- {name}: {description}" for name, description in columns.items()),
            context=context.for_step(Step.TAG_TYPES, table),
        )
        reply = extract_json(llm.complete(prompt).text)
        mapping = reply.get("tag_types", {}) if isinstance(reply, dict) else {}

        # A type outside the ontology means no constraint, never a guess.
        for name in columns:
            chosen = mapping.get(name)
            result[name] = chosen if chosen in type_names else None

    return result


def _tag_edge(
    doc: Document, column: str, value: str, type: str | None, resolver: Resolver, new_entities: dict[str, Entity],
) -> dict:
    result = resolver.resolve(value, type)
    if result.status is Status.MATCHED:
        match = result.matches[0]
        source = _SOURCE_HUMAN if match.method is Method.HUMAN else _SOURCE_METADATA
        return _edge(doc.doc_id, match.id, match.score, match.needs_review, column, value, source)

    # Unknown or ambiguous tag: a flagged new entity, reused by later documents, e.g. "product:gadget".
    entity_id = f"{_slug(type) if type else _UNTYPED}:{_slug(value)}"
    if entity_id not in new_entities:
        new_entities[entity_id] = Entity(
            id=entity_id, type=type or _UNTYPED, canonical_name=value, attributes={"tag_column": column},
            source=_SOURCE_INFERRED, confidence=_NEW_ENTITY_CONFIDENCE, needs_review=True,
        )
    return _edge(doc.doc_id, entity_id, _TAG_VALUE_CONFIDENCE, True, column, value, _SOURCE_METADATA)


def _edge(
    doc_id: str, entity_id: str, confidence: float, needs_review: bool, column: str, text: str, source: str,
) -> dict:
    # `tag_column` and `text` (the raw tag value, e.g. "GP") show where the tag came from; the graph copies the
    # column into the evidence.
    return {
        "src": DOC_PREFIX + doc_id, "rel": _TAGGED_WITH, "dst": entity_id, "props": {}, "confidence": confidence,
        "source": source, "evidence_doc": doc_id, "evidence_quote": "", "needs_review": needs_review,
        "tag_column": column, "text": text,
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# -------------------------------------------------------------- extractions

def _resolve_extraction(extraction: Extraction, resolver: Resolver, affect_types: list[str]) -> dict:
    # Mentions come from catalog-name matching in the build (find_mentions), not from the LLM.
    return {
        "mentions": [],
        "events": [_resolve_event(event.model_dump(), resolver, affect_types) for event in extraction.events],
    }


def _resolve_event(event: dict, resolver: Resolver, affect_types: list[str]) -> dict:
    for affect in event["affects"]:
        affect["entity_id"], affect["match_score"] = _lookup_allowed(affect["entity"], affect_types, resolver)

    # Kept per place, so graph evidence can weigh a place by how surely its name was matched.
    location_scores: dict[str, float] = {}
    for name in event["locations"]:
        location_id, score = _lookup(name, _LOCATION_TYPE, resolver)
        if location_id is not None:
            location_scores[location_id] = max(score, location_scores.get(location_id, 0.0))

    entity_ids = [affect["entity_id"] for affect in event["affects"]]
    return {
        **event,
        "entity_ids": _unique(entity_ids),
        "location_ids": sorted(location_scores),
        "location_scores": location_scores,
        # An event that states no effect counts as the weakest claim extraction keeps (its "implied" floor).
        "confidence": max((affect["confidence"] for affect in event["affects"]), default=MIN_AFFECT_CONFIDENCE),
    }


def _lookup(name: str, type: str | None, resolver: Resolver) -> tuple[str | None, float]:
    """(entity id, score) of a unique match, else (None, 0.0)."""
    result = resolver.resolve(name, type)
    if result.status is not Status.MATCHED:
        return None, 0.0
    return result.matches[0].id, result.matches[0].score


def _lookup_allowed(name: str, types: list[str], resolver: Resolver) -> tuple[str | None, float]:
    # One lookup across the whole catalog (not one per type, which multiplied judge calls by the type count);
    # the match counts only when its type is allowed.
    result = resolver.resolve(name, None)
    if result.status is not Status.MATCHED:
        return None, 0.0
    match = result.matches[0]
    # A name_match fix is the person's call, whatever the relation says.
    if types and match.type not in types and match.method is not Method.HUMAN:
        return None, 0.0
    return match.id, match.score


def _unique(ids: list[str | None]) -> list[str]:
    return sorted({item for item in ids if item is not None})


# ------------------------------------------------------------------- dedupe

def _mergeable(first: dict, second: dict, cosine: float, days: int, similarity: float) -> bool:
    if first["type"] != second["type"] or cosine < similarity:
        return False
    if not _share_target(first, second):
        return False
    return _days_apart(first["date"], second["date"]) <= days


def _share_target(first: dict, second: dict) -> bool:
    return bool(
        set(first["entity_ids"]) & set(second["entity_ids"]) or set(first["location_ids"]) & set(second["location_ids"])
    )


def _days_apart(first: str | None, second: str | None) -> float:
    if not first or not second:
        return float("inf")
    return abs((date.fromisoformat(first) - date.fromisoformat(second)).days)


def _canonical(group: list[tuple[int, dict]]) -> dict:
    group = sorted(group, key=lambda item: (item[1]["date"] or "", item[1]["doc_id"], item[0]))
    first = group[0][1]
    events = [event for _, event in group]
    entity_ids = sorted({id_ for event in events for id_ in event["entity_ids"]})
    location_ids = sorted({id_ for event in events for id_ in event["location_ids"]})

    key = json.dumps(
        {
            "type": first["type"], "date": first["date"], "entity_ids": entity_ids, "location_ids": location_ids,
            "members": [(event["doc_id"], position) for position, event in group], "summary": first["summary"],
        },
        separators=(",", ":"), sort_keys=True,
    )
    quotes = [event["evidence"] for event in events if event.get("evidence")]
    return {
        "event_id": _EVENT_PREFIX + hashlib.sha1(key.encode()).hexdigest()[:_EVENT_ID_CHARS],
        "type": first["type"],
        "date": first["date"],
        "date_source": first.get("date_source", "stated"),
        "status": first.get("status", "reported"),
        "summary": first["summary"],
        "entity_ids": entity_ids,
        "location_ids": location_ids,
        "org_names": sorted({name for event in events for name in event["organizations"]}),
        "doc_ids": sorted({event["doc_id"] for event in events}),
        # Extraction keys "<doc_id>#<n>" of the members; they do not change when merging does.
        "keys": [event.get("key", f"{event['doc_id']}#{position}") for position, event in group],
        "quotes": list(dict.fromkeys(quotes)),
        "confidence": max(event["confidence"] for event in events),
        "affects": _union_affects(events),
        "needs_review": any(event["needs_review"] for event in events),
    }


def _union_affects(group: list[dict]) -> list[dict]:
    """Resolved affects of every member, each with its document; unmatched entities are skipped."""
    return [
        {
            "entity_id": affect["entity_id"], "aspect": affect.get("aspect", ""), "direction": affect["direction"],
            "confidence": affect["confidence"], "quote": affect["quote"], "doc_id": event["doc_id"],
            "match_score": affect.get("match_score", 1.0),
        }
        for event in group for affect in event["affects"] if affect["entity_id"] is not None
    ]
