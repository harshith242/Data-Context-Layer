"""Graph edges keep all their evidence: independent sources combine, repeats and syndicated copies do not,
conflicts are flagged, aspects stay apart, and paths keep each edge's direction."""
import json

import pytest

from ctxlayer.common.catalog import Entity
from ctxlayer.unstructured.documents import Document
from ctxlayer.unstructured.graph_build import build_graph
from ctxlayer.unstructured.graph_store import GraphStore

_THRESHOLD = 0.8
_CROP = "crop:x"
_PLACE = "location:p"
_EVENT = "event:e1"


def _affect(doc: str, quote: str, confidence: float, direction: str = "up", aspect: str = "price") -> dict:
    return {"entity_id": _CROP, "aspect": aspect, "direction": direction, "confidence": confidence,
            "quote": quote, "doc_id": doc, "match_score": 1.0}


def _member(quote: str, confidence: float, places: dict[str, float] | None = None, orgs: list[str] | None = None) -> dict:
    places = places or {}
    return {"type": "t", "evidence": quote, "confidence": confidence, "location_ids": sorted(places),
            "location_scores": places, "organizations": orgs or []}


def _build(tmp_path, affects: list[dict], members: dict[str, list[dict]], quotes: list[str]) -> list[dict]:
    """Edges of a graph with one canonical event built from `members` (doc -> its extracted events)."""
    docs = sorted(members)
    event = {
        "event_id": _EVENT, "type": "t", "date": "2025-01-02", "summary": "s", "doc_ids": docs, "quotes": quotes,
        "entity_ids": [_CROP], "location_ids": sorted({p for ms in members.values() for m in ms for p in m["location_ids"]}),
        "org_names": sorted({o for ms in members.values() for m in ms for o in m["organizations"]}),
        "affects": affects, "confidence": max(m["confidence"] for ms in members.values() for m in ms),
        "needs_review": False,
    }
    catalog = [Entity(_CROP, "Crop", "X"), Entity(_PLACE, "Location", "P")]
    documents = [Document(doc, "2025-01-01", "title", "", "") for doc in docs]
    resolved = {doc: {"mentions": [], "events": items} for doc, items in members.items()}
    build_graph(catalog, [], documents, [], resolved, [event], tmp_path, _THRESHOLD)
    return [json.loads(line) for line in (tmp_path / "edges.jsonl").read_text().splitlines()]


def _only(edges: list[dict], rel: str) -> list[dict]:
    return [edge for edge in edges if edge["rel"] == rel]


def test_two_independent_documents_combine_and_keep_both_evidence(tmp_path):
    edges = _build(tmp_path, [_affect("d1", "a", 0.6), _affect("d2", "b", 0.6)],
                   {"d1": [_member("a", 0.6)], "d2": [_member("b", 0.6)]}, ["a", "b"])
    (edge,) = _only(edges, "AFFECTS")

    # 1 - 0.4 x 0.4 = 0.84: two implied statements from different documents pass the 0.8 review line.
    assert edge["confidence"] == pytest.approx(0.84)
    assert edge["support"] == 2 and edge["needs_review"] is False
    assert [item["doc"] for item in edge["evidence"]] == ["d1", "d2"]


def test_repeats_in_one_document_do_not_inflate(tmp_path):
    edges = _build(tmp_path, [_affect("d1", "a", 0.6), _affect("d1", "b", 0.6)],
                   {"d1": [_member("a", 0.6), _member("b", 0.6)]}, ["a", "b"])
    (edge,) = _only(edges, "AFFECTS")

    assert edge["confidence"] == pytest.approx(0.6) and edge["support"] == 1
    assert edge["needs_review"] is True  # 0.6 is below the review line
    assert len(edge["evidence"]) == 2


def test_syndicated_copies_with_the_same_quote_count_once(tmp_path):
    edges = _build(tmp_path, [_affect("d1", "Same  sentence.", 0.6), _affect("d2", "same sentence.", 0.6)],
                   {"d1": [_member("Same  sentence.", 0.6)], "d2": [_member("same sentence.", 0.6)]},
                   ["Same  sentence.", "same sentence."])
    (edge,) = _only(edges, "AFFECTS")

    assert edge["support"] == 1 and edge["confidence"] == pytest.approx(0.6)


def test_up_and_down_for_one_aspect_is_a_flagged_conflict(tmp_path):
    edges = _build(tmp_path, [_affect("d1", "a", 0.9, "up"), _affect("d2", "b", 0.6, "down")],
                   {"d1": [_member("a", 0.9)], "d2": [_member("b", 0.6)]}, ["a", "b"])
    (edge,) = _only(edges, "AFFECTS")

    # The better-supported side is the edge; nothing is combined across sides; both stay as evidence.
    assert edge["conflict"] is True and edge["needs_review"] is True
    assert edge["props"]["direction"] == "up" and edge["confidence"] == pytest.approx(0.9)
    assert sorted(item["direction"] for item in edge["evidence"]) == ["down", "up"]


def test_different_aspects_of_one_thing_are_separate_edges(tmp_path):
    edges = _build(tmp_path, [_affect("d1", "a", 0.9, "up", "price"), _affect("d1", "a", 0.9, "down", "output")],
                   {"d1": [_member("a", 0.9)]}, ["a"])

    assert sorted((e["props"]["aspect"], e["props"]["direction"]) for e in _only(edges, "AFFECTS")) == [
        ("output", "down"), ("price", "up"),
    ]
    assert not any(edge.get("conflict") for edge in edges)


def test_a_place_is_cited_from_the_document_that_named_it(tmp_path):
    edges = _build(tmp_path, [_affect("d1", "a", 0.9)],
                   {"d1": [_member("a", 0.9)], "d2": [_member("b", 0.6, places={_PLACE: 0.9})]}, ["a", "b"])
    (located,) = _only(edges, "LOCATED_IN")

    # Strength = the naming event's 0.6 x the 0.9 name match.
    assert (located["evidence_doc"], located["evidence_quote"]) == ("d2", "b")
    assert located["confidence"] == pytest.approx(0.54)


def test_paths_keep_each_edges_own_direction_and_reciprocal_edges(tmp_path):
    nodes = [{"id": node, "type": "T", "name": node, "props": {}} for node in "abc"]
    edges = [_plain("a", "R", "b"), _plain("c", "S", "b"), _plain("b", "Q", "a")]
    (tmp_path / "nodes.jsonl").write_text("".join(json.dumps(n) + "\n" for n in nodes))
    (tmp_path / "edges.jsonl").write_text("".join(json.dumps(e) + "\n" for e in edges))
    store = GraphStore.load([tmp_path / "nodes.jsonl", tmp_path / "edges.jsonl"])

    walked = sorted(
        [(e["src"], e["rel"], e["dst"], e["traversal"]) for e in path] for path in store.paths("a", "c")
    )
    # a->b can be walked on R (forward) or Q (reverse): the opposite b->a edge is not merged away.
    assert walked == [
        [("a", "R", "b", "forward"), ("c", "S", "b", "reverse")],
        [("b", "Q", "a", "reverse"), ("c", "S", "b", "reverse")],
    ]


def _plain(src: str, rel: str, dst: str) -> dict:
    return {"src": src, "rel": rel, "dst": dst, "props": {}, "confidence": 1.0, "source": "test",
            "evidence_doc": "", "evidence_quote": "", "needs_review": False}
