"""Knowledge graph from the domain_a build: nodes, edges, GraphStore queries, graph_explore and the build outputs."""
import json
import os

import pytest
from fakes import FakeEmbedder
from fixtures.domain_a.extract_replies import extract_reply
from fixtures.domain_a.make_domain import PLANTED_BULLETIN_REF
from fixtures.domain_a.unstructured_build import copy_built, unstructured_llm

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.files import read_yaml
from ctxlayer.serving.context import coverage
from ctxlayer.serving.tools import open_context
from ctxlayer.unstructured.build import Start, build_unstructured
from ctxlayer.unstructured.graph_store import GraphStore, ReviewFilter
from ctxlayer.unstructured.tools import graph_explore, search_docs_tool

_SECOND_REF = "FB-20250427-01"
_CROP = "crop:milling_wheat"


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def built(tmp_path_factory, domain_a, domain_a_built):
    """(settings, counts, llm) of build_unstructured run on a copy of the structured build."""
    settings = copy_built(tmp_path_factory.mktemp("unstructured"), domain_a, domain_a_built)
    llm = unstructured_llm()
    counts = build_unstructured(settings, llm, FakeEmbedder())
    return settings, counts, llm


@pytest.fixture(scope="module")
def store(built) -> GraphStore:
    graph = artifacts_dir(built[0]) / "graph"
    return GraphStore.load([graph / "nodes.jsonl", graph / "edges.jsonl"])


def test_planted_event_affects_planted_crop_going_up(built):
    graph = artifacts_dir(built[0]) / "graph"
    edges = [e for e in _read_jsonl(graph / "edges.jsonl") if e["rel"] == "AFFECTS"]

    # Both bulletins state it at 0.8: 1 - 0.2 x 0.2 = 0.96, capped at 0.95 for extracted evidence.
    assert [(e["dst"], e["props"], e["confidence"], e["source"]) for e in edges] == [
        (_CROP, {"aspect": "price", "direction": "up"}, 0.95, "extraction")
    ]
    assert edges[0]["support"] == 2 and edges[0]["needs_review"] is False
    assert {item["doc"] for item in edges[0]["evidence"]} == {PLANTED_BULLETIN_REF, _SECOND_REF}
    assert edges[0]["evidence_doc"] in {PLANTED_BULLETIN_REF, _SECOND_REF} and edges[0]["evidence_quote"]


def test_both_bulletins_report_one_event(built):
    graph = artifacts_dir(built[0]) / "graph"
    reports = [e for e in _read_jsonl(graph / "edges.jsonl") if e["rel"] == "REPORTS"]

    assert sorted(e["src"] for e in reports) == [f"doc:{PLANTED_BULLETIN_REF}", f"doc:{_SECOND_REF}"]
    assert len({e["dst"] for e in reports}) == 1


def test_catalog_hierarchy_and_location_edges_are_in_the_graph(store):
    rels = {(e["src"], e["rel"], e["dst"]) for e in store.neighbors("location:great_plains")}

    assert ("location:great_plains", "PART_OF", "location:north_america") in rels
    assert ("series:wht_mill_gp", "AT_LOCATION", "location:great_plains") in rels


def test_neighbors_honour_min_confidence_and_dates(store):
    event = next(e["neighbor"] for e in store.neighbors(_CROP, rel="AFFECTS"))

    assert [e["rel"] for e in store.neighbors(event, rel="AFFECTS", min_confidence=0.95)] == ["AFFECTS"]
    assert store.neighbors(event, rel="AFFECTS", min_confidence=0.96) == []
    assert len(store.neighbors(event, rel="AFFECTS", start="2025-04-20", end="2025-04-30")) == 1
    assert store.neighbors(event, rel="AFFECTS", start="2025-05-01") == []


def test_docs_for_parent_region_include_docs_tagged_with_its_child(store):
    children = store.docs_for_entities(["location:great_plains"])
    parent = store.docs_for_entities(["location:north_america"])

    assert {PLANTED_BULLETIN_REF, _SECOND_REF} <= children
    assert children <= parent
    assert PLANTED_BULLETIN_REF in parent and _SECOND_REF in parent
    assert store.docs_for_entities(["location:rhine_valley"]).isdisjoint({PLANTED_BULLETIN_REF})


def test_docs_for_entities_ignores_review_flagged_hierarchy_edges(tmp_path):
    nodes = [
        {"id": "parent", "type": "T", "name": "parent", "props": {}},
        {"id": "child", "type": "T", "name": "child", "props": {}},
        {"id": "doc:d1", "type": "Document", "name": "d1", "props": {}},
    ]
    edges = [
        _edge("child", "HAS_PARENT", "parent", True),
        _edge("doc:d1", "TAGGED_WITH", "child", False),
    ]
    (tmp_path / "nodes.jsonl").write_text("".join(json.dumps(node) + "\n" for node in nodes))
    (tmp_path / "edges.jsonl").write_text("".join(json.dumps(edge) + "\n" for edge in edges))
    flagged_store = GraphStore.load([tmp_path / "nodes.jsonl", tmp_path / "edges.jsonl"])

    assert flagged_store.docs_for_entities(["parent"]) == set()


def test_paths_connect_document_to_crop(store):
    paths = store.paths(f"doc:{PLANTED_BULLETIN_REF}", _CROP, max_len=2)

    assert sorted(e["rel"] for path in paths if len(path) == 1 for e in path) == ["MENTIONS", "TAGGED_WITH"]


def test_graph_explore_depth_two_is_bounded_and_clamped(store):
    one = graph_explore(store, _CROP, depth=1, min_confidence=0.0)
    two = graph_explore(store, _CROP, depth=2, min_confidence=0.0)
    deep = graph_explore(store, _CROP, depth=9, min_confidence=0.0)

    assert (one["depth"], two["depth"], deep["depth"]) == (1, 2, 2)
    assert len(one["edges"]) < len(two["edges"]) == len(deep["edges"])
    assert {n["id"] for n in one["nodes"]} < {n["id"] for n in two["nodes"]}
    assert two["truncated"] is False


def test_graph_explore_hides_needs_review_edges_by_default(tmp_path):
    flagged_store = _store_with_flagged_edge(tmp_path)
    hidden = graph_explore(flagged_store, "a", min_confidence=0.0)
    shown = graph_explore(flagged_store, "a", min_confidence=0.0, review=ReviewFilter.INCLUDE)

    assert [e["dst"] for e in hidden["edges"]] == ["b"]
    assert sorted(e["dst"] for e in shown["edges"]) == ["b", "c"]


def test_stale_link_edges_without_events_are_skipped(tmp_path):
    (tmp_path / "nodes.jsonl").write_text(json.dumps({"id": "series:x", "type": "Series", "name": "x", "props": {}}))
    edge = {
        "src": "series:x", "rel": "CANDIDATE_DRIVER", "dst": "event:gone", "props": {}, "confidence": 1.0,
        "source": "x", "evidence_doc": "", "evidence_quote": "", "needs_review": False,
    }
    (tmp_path / "edges.jsonl").write_text(json.dumps(edge))

    assert GraphStore.load([tmp_path / "nodes.jsonl", tmp_path / "edges.jsonl"]).neighbors("series:x") == []


def test_search_docs_tool_limits_hits_to_entity_documents(built, store):
    settings = built[0]
    result = search_docs_tool(
        artifacts_dir(settings) / "docs", FakeEmbedder(), store, "drought", entity_ids=["location:rhine_valley"],
        k=20,
    )

    assert result["hits"]
    assert {PLANTED_BULLETIN_REF, _SECOND_REF}.isdisjoint({hit["doc_id"] for hit in result["hits"]})


def test_failed_extraction_still_keeps_document_tag_edges(tmp_path, domain_a, domain_a_built):
    settings = copy_built(tmp_path, domain_a, domain_a_built)

    def extract_or_fail(prompt, tools):
        return "not json" if PLANTED_BULLETIN_REF in prompt else extract_reply(prompt, tools)

    build_unstructured(settings, unstructured_llm(extract=extract_or_fail), FakeEmbedder())
    edges = _read_jsonl(artifacts_dir(settings) / "graph" / "edges.jsonl")

    assert any(edge["src"] == f"doc:{PLANTED_BULLETIN_REF}" and edge["rel"] == "TAGGED_WITH" for edge in edges)


def test_build_counts_and_artifacts(built):
    settings, counts, _ = built
    out = artifacts_dir(settings)
    extractions = _read_jsonl(out / "docs" / "extractions.jsonl")

    assert counts["documents"] == len(extractions) == len(_read_jsonl(out / "docs" / "documents.jsonl"))
    assert counts["events"] == 1 and counts["extraction_failures"] == 0
    planted = next(item for item in extractions if item["doc_id"] == PLANTED_BULLETIN_REF)
    assert [e["type"] for e in planted["events"]] == ["supply_disruption"]
    assert (out / "graph" / "entities_text.yaml").is_file() and (out / "docs" / "failures.jsonl").is_file()
    assert (out / "docs" / "chunks.parquet").is_file()


def test_serving_coverage_reads_the_unstructured_manifest(built):
    context = open_context(built[0], embedder=FakeEmbedder())
    try:
        assert coverage(context)["documents"] == ["2025-03-01", "2025-06-30"]
    finally:
        context.con.close()


def test_event_types_are_written_to_their_own_file_and_reused_on_rebuild(built):
    settings, _, _ = built
    names = [t["name"] for t in read_yaml(artifacts_dir(settings) / "event_types.yaml")["event_types"]]
    assert names[0] == "supply_disruption" and names[-1] == "other"

    rebuilt_llm = unstructured_llm(event_types="not json")
    build_unstructured(settings, rebuilt_llm, FakeEmbedder())
    tasks = [prompt.split("\n", 1)[0] for prompt, _ in rebuilt_llm.calls]

    assert "[task: event_types]" not in tasks and "[task: extract]" not in tasks


def test_review_file_gets_unstructured_sections_once(built):
    settings, _, _ = built
    build_unstructured(settings, unstructured_llm(), FakeEmbedder())
    text = (artifacts_dir(settings) / "REVIEW.md").read_text()

    assert text.count("## Extraction failures") == 1
    assert text.count("## New entities from documents") == 1
    assert "## Roles" in text


def _store_with_flagged_edge(tmp_path) -> GraphStore:
    """Node a with a clean edge to b and a needs_review edge to c."""
    def edge(dst: str, review: bool) -> dict:
        return {"src": "a", "rel": "R", "dst": dst, "props": {}, "confidence": 0.9, "source": "x",
                "evidence_doc": "", "evidence_quote": "", "needs_review": review}

    nodes = [{"id": node, "type": "T", "name": node, "props": {}} for node in "abc"]
    (tmp_path / "nodes.jsonl").write_text("".join(json.dumps(n) + "\n" for n in nodes))
    (tmp_path / "edges.jsonl").write_text(json.dumps(edge("b", False)) + "\n" + json.dumps(edge("c", True)) + "\n")
    return GraphStore.load([tmp_path / "nodes.jsonl", tmp_path / "edges.jsonl"])


def _edge(src: str, rel: str, dst: str, review: bool) -> dict:
    return {
        "src": src, "rel": rel, "dst": dst, "props": {}, "confidence": 0.9, "source": "test",
        "evidence_doc": "", "evidence_quote": "", "needs_review": review,
    }


def _no_call(prompt, tools):
    raise AssertionError("the shortcut must not extract or induce event types again")


def test_after_extract_reuses_saved_extractions_and_gives_the_same_graph(tmp_path, domain_a, domain_a_built):
    settings = copy_built(tmp_path, domain_a, domain_a_built)
    full = build_unstructured(settings, unstructured_llm(), FakeEmbedder())
    edges_path = artifacts_dir(settings) / "graph" / "edges.jsonl"
    full_edges = edges_path.read_text()

    resumed = build_unstructured(
        settings, unstructured_llm(extract=_no_call, event_types=_no_call), FakeEmbedder(), start=Start.AFTER_EXTRACT,
    )

    assert resumed == full
    assert edges_path.read_text() == full_edges


def test_after_extract_stops_when_an_input_changed_after_the_extractions(tmp_path, domain_a, domain_a_built):
    settings = copy_built(tmp_path, domain_a, domain_a_built)
    build_unstructured(settings, unstructured_llm(), FakeEmbedder())
    out = artifacts_dir(settings)
    later = (out / "docs" / "extractions.jsonl").stat().st_mtime + 10
    os.utime(out / "ontology.yaml", (later, later))

    with pytest.raises(ValueError, match="ontology.yaml"):
        build_unstructured(settings, unstructured_llm(), FakeEmbedder(), start=Start.AFTER_EXTRACT)


def test_after_extract_stops_when_no_extractions_were_saved(tmp_path, domain_a, domain_a_built):
    settings = copy_built(tmp_path, domain_a, domain_a_built)

    with pytest.raises(FileNotFoundError, match="extractions.jsonl"):
        build_unstructured(settings, unstructured_llm(), FakeEmbedder(), start=Start.AFTER_EXTRACT)
