"""Knowledge graph build: catalog, document and event nodes with their edges, written as nodes.jsonl / edges.jsonl."""
import math
import re
from pathlib import Path

from ctxlayer.common.catalog import Entity
from ctxlayer.common.files import write_jsonl
from ctxlayer.unstructured._names import DOC_PREFIX
from ctxlayer.unstructured.documents import Document

_ORG_PREFIX = "org:"
_SOURCE_CATALOG = "metadata"
_SOURCE_EXTRACTION = "extraction"
_NODES_FILE = "nodes.jsonl"
_EDGES_FILE = "edges.jsonl"
_DOCUMENT = "Document"
_EVENT = "Event"
_ORGANIZATION = "Organization"
_AFFECTS = "AFFECTS"
# Claims whose strength decides review; other edges are flagged only through their evidence.
_THRESHOLDED = (_AFFECTS, "MENTIONS")
# Several independent documents raise an extracted claim, but never to the 1.0 of a fact read from the data.
_COMBINED_CAP = 0.95
_OPPOSITE = ("up", "down")
_DECIMALS = 4
_TAG_COLUMN = "tag_column"


def build_graph(
    catalog: list[Entity], new_entities: list[Entity], docs: list[Document], tag_edges: list[dict],
    extractions_resolved: dict, events: list[dict], out_dir: Path, review_threshold: float,
) -> dict:
    """Write nodes.jsonl and edges.jsonl into `out_dir`; returns {"nodes": n, "edges": n}.

    Node ids: catalog ids, `doc:<doc_id>`, `<event_id>`, `org:<slug>`. An event node's props are the canonical event.
    Every piece of evidence for one edge is kept in its `evidence` list; see `_merged` for how they combine.
    """
    entities = [*catalog, *new_entities]
    nodes = [*_entity_nodes(entities), *_document_nodes(docs), *_event_nodes(events), *_org_nodes(events)]

    items = [
        *_catalog_edges(entities),
        *tag_edges,
        *_mention_edges(extractions_resolved),
        *_reports_edges(events, extractions_resolved),
        *_affects_edges(events),
        *_event_edges(events, extractions_resolved),
    ]
    groups: dict[tuple, list[dict]] = {}
    for item in items:
        groups.setdefault(_edge_key(item), []).append(item)
    edges = [_merged(group, review_threshold) for group in groups.values()]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / _NODES_FILE, nodes)
    write_jsonl(out_dir / _EDGES_FILE, edges)
    return {"nodes": len(nodes), "edges": len(edges)}


# -------------------------------------------------------------------- nodes

def _entity_nodes(entities: list[Entity]) -> list[dict]:
    return [
        {
            "id": entity.id, "type": entity.type, "name": entity.canonical_name,
            "props": {**entity.attributes, "needs_review": entity.needs_review},
        }
        for entity in entities
    ]


def _document_nodes(docs: list[Document]) -> list[dict]:
    return [
        {"id": DOC_PREFIX + doc.doc_id, "type": _DOCUMENT, "name": doc.title, "props": {"date": doc.date}}
        for doc in docs
    ]


def _event_nodes(events: list[dict]) -> list[dict]:
    # A missing date stays out of props so date filters leave the event alone.
    return [
        {
            "id": event["event_id"], "type": _EVENT, "name": event["summary"],
            "props": {key: value for key, value in event.items() if not (key == "date" and value is None)},
        }
        for event in events
    ]


def _org_nodes(events: list[dict]) -> list[dict]:
    names = {name for event in events for name in event["org_names"]}
    return [{"id": _org_id(name), "type": _ORGANIZATION, "name": name, "props": {}} for name in sorted(names)]


def _org_id(name: str) -> str:
    return _ORG_PREFIX + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# -------------------------------------------------------------------- edges

def _edge(
    src: str, rel: str, dst: str, confidence: float, source: str, doc_id: str, quote: str,
    needs_review: bool, props: dict | None = None,
) -> dict:
    return {
        "src": src, "rel": rel, "dst": dst, "props": props or {}, "confidence": confidence, "source": source,
        "evidence_doc": doc_id, "evidence_quote": quote, "needs_review": needs_review,
    }


def _catalog_edges(entities: list[Entity]) -> list[dict]:
    """HAS_PARENT / AT_LOCATION / PART_OF edges from entity attributes."""
    ids = {entity.id for entity in entities}
    edges = []

    for entity in entities:
        part_of = entity.attributes.get("part_of")
        links = [
            ("HAS_PARENT", entity.attributes.get("parents", [])),
            ("AT_LOCATION", entity.attributes.get("locations", [])),
            ("PART_OF", [part_of] if part_of else []),
        ]
        for rel, targets in links:
            edges += [
                _edge(entity.id, rel, target, entity.confidence, _SOURCE_CATALOG, "", "", entity.needs_review)
                for target in targets if target in ids
            ]
    return edges


def _mention_edges(extractions_resolved: dict) -> list[dict]:
    """One MENTIONS edge per (document, entity), at the best match score."""
    best: dict[tuple[str, str], float] = {}
    for doc_id, item in extractions_resolved.items():
        for mention in item["mentions"]:
            if mention["entity_id"] is None:
                continue
            key = (doc_id, mention["entity_id"])
            best[key] = max(best.get(key, 0.0), mention["score"])

    return [
        _edge(DOC_PREFIX + doc_id, "MENTIONS", entity_id, score, _SOURCE_EXTRACTION, doc_id, "", False)
        for (doc_id, entity_id), score in sorted(best.items())
    ]


def _members(event: dict, doc_id: str, extractions_resolved: dict) -> list[dict]:
    """The document's own extracted events that were merged into `event`: by extraction key, else (same type, quote
    among its quotes). The key survives a type that a decision or a relabel changed."""
    own = extractions_resolved.get(doc_id, {}).get("events", [])
    keys = set(event.get("keys", []))
    if keys and all("key" in item for item in own):
        return [item for item in own if item["key"] in keys]
    return [item for item in own if item["type"] == event["type"] and item["evidence"] in event["quotes"]]


def _reports_edges(events: list[dict], extractions_resolved: dict) -> list[dict]:
    """Document -> event, with the quote and strength that document gave for the event."""
    edges = []
    for event in events:
        for doc_id in event["doc_ids"]:
            member = next(iter(_members(event, doc_id, extractions_resolved)), None)
            edges.append(_edge(
                DOC_PREFIX + doc_id, "REPORTS", event["event_id"],
                member["confidence"] if member else event["confidence"], _SOURCE_EXTRACTION,
                doc_id, member["evidence"] if member else "", event["needs_review"],
            ))
    return edges


def _affects_edges(events: list[dict]) -> list[dict]:
    """Event -> entity, one evidence item per affect: its confidence x how surely the entity name was matched."""
    return [
        _edge(
            event["event_id"], _AFFECTS, affect["entity_id"], affect["confidence"] * affect.get("match_score", 1.0),
            affect.get("source", _SOURCE_EXTRACTION), affect["doc_id"], affect["quote"], event["needs_review"],
            {"aspect": affect.get("aspect", ""), "direction": affect["direction"]},
        )
        for event in events for affect in event["affects"]
    ]


def _event_edges(events: list[dict], extractions_resolved: dict) -> list[dict]:
    """LOCATED_IN to catalog locations and INVOLVES to organisations, each from the document that named it."""
    edges = []
    for event in events:
        for doc_id in event["doc_ids"]:
            for member in _members(event, doc_id, extractions_resolved):
                scores = member.get("location_scores", {})
                targets = [
                    ("LOCATED_IN", place, member["confidence"] * scores.get(place, 1.0))
                    for place in member["location_ids"] if place in event["location_ids"]
                ]
                targets += [("INVOLVES", _org_id(name), member["confidence"]) for name in member["organizations"]]
                edges += [
                    _edge(event["event_id"], rel, target, strength, _SOURCE_EXTRACTION, doc_id, member["evidence"],
                          event["needs_review"])
                    for rel, target, strength in targets
                ]
    return edges


# ----------------------------------------------------------------- evidence

def _edge_key(edge: dict) -> tuple:
    # Effects on different aspects of one thing (price up, output down) are separate facts.
    aspect = edge["props"].get("aspect") if edge["rel"] == _AFFECTS else None
    return edge["src"], edge["rel"], edge["dst"], aspect


def _merged(group: list[dict], threshold: float) -> dict:
    """One edge from all its evidence items.

    Strength: the best item per independent source, combined as 1 - prod(1 - s), never below the best single
    source and capped at 0.95 when several sources agree. Up and down for one aspect is a conflict: the edge keeps
    the better-supported side and is flagged; the evidence list keeps both sides.
    """
    sides = {item["props"].get("direction") for item in group} & set(_OPPOSITE)
    conflict = len(sides) > 1
    kept = group
    if conflict:
        by_side = {side: [item for item in group if item["props"].get("direction") == side] for side in sorted(sides)}
        kept = max(by_side.values(), key=_strength)

    best = max(kept, key=lambda item: item["confidence"])
    strength = _strength(kept)
    flagged = conflict or all(item["needs_review"] for item in group)
    if best["rel"] in _THRESHOLDED and strength < threshold:
        flagged = True

    evidence = sorted(group, key=lambda item: (-item["confidence"], item["evidence_doc"], item["evidence_quote"]))
    edge = {
        **best, "confidence": round(strength, _DECIMALS), "needs_review": flagged,
        "support": len(_sources(kept)), "evidence": [_evidence(item) for item in evidence],
    }
    # Tag provenance lives in the evidence; the raw tag value stays in docs/tag_edges.jsonl.
    for field in (_TAG_COLUMN, "text"):
        edge.pop(field, None)
    if conflict:
        edge["conflict"] = True
    return edge


def _strength(items: list[dict]) -> float:
    values = list(_sources(items).values())
    best = max(values)
    if len(values) == 1:
        return best
    combined = 1.0 - math.prod(1.0 - value for value in values)
    return max(best, min(_COMBINED_CAP, combined))


def _sources(items: list[dict]) -> dict[str, float]:
    """Best strength per independent source: one per document, and documents sharing a quote count once
    (syndicated copies of one story are not independent)."""
    best_by_doc: dict[str, float] = {}
    quotes_by_doc: dict[str, set[str]] = {}
    for item in items:
        doc = item["evidence_doc"]
        best_by_doc[doc] = max(best_by_doc.get(doc, 0.0), item["confidence"])
        if item["evidence_quote"]:
            quotes_by_doc.setdefault(doc, set()).add(_normal(item["evidence_quote"]))

    # Union of documents that share any quote; the source keeps the best strength of its documents.
    owner = {doc: doc for doc in best_by_doc}

    def root(doc: str) -> str:
        while owner[doc] != doc:
            doc = owner[doc]
        return doc

    first_with: dict[str, str] = {}
    for doc in sorted(quotes_by_doc):
        for quote in quotes_by_doc[doc]:
            if quote in first_with:
                owner[root(doc)] = root(first_with[quote])
            else:
                first_with[quote] = doc

    sources: dict[str, float] = {}
    for doc, value in best_by_doc.items():
        key = root(doc)
        sources[key] = max(sources.get(key, 0.0), value)
    return sources


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _evidence(item: dict) -> dict:
    found = {"doc": item["evidence_doc"], "quote": item["evidence_quote"], "confidence": round(item["confidence"], _DECIMALS)}
    if item["props"].get("direction"):
        found["direction"] = item["props"]["direction"]
    if item.get(_TAG_COLUMN):
        found["column"] = item[_TAG_COLUMN]
    return found
