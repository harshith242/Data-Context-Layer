"""The agent tool set over a full domain_a build (structured + documents + graph + links)."""
import json

import pytest
from fakes import FakeEmbedder, FakeLLM
from fixtures.domain_a.make_domain import PLANTED_BULLETIN_REF, PLANTED_QUOTE, PLANTED_SERIES_CODE
from fixtures.domain_a.unstructured_build import copy_built, unstructured_llm

from ctxlayer.common.llm import Reply, ToolCall
from ctxlayer.linking.build import build_links
from ctxlayer.reference_agent.loop import run_loop
from ctxlayer.serving.context import get_context
from ctxlayer.serving.tools import build_tools, open_context
from ctxlayer.unstructured.build import build_unstructured

_SECOND_REF = "FB-20250427-01"
_WHEAT_TAG = "Wheat (milling)"
_TOOL_NAMES = [
    "get_context", "resolve_entity", "query_metric", "validate_sql", "execute_semantic_sql", "search_docs",
    "graph_explore", "explain_move",
]


@pytest.fixture(scope="module")
def full_ctx(tmp_path_factory, domain_a, domain_a_built):
    settings = copy_built(tmp_path_factory.mktemp("full"), domain_a, domain_a_built)
    build_unstructured(settings, unstructured_llm(), FakeEmbedder())
    build_links(settings)

    ctx = open_context(settings, FakeEmbedder(), unstructured_llm())
    yield ctx
    ctx.con.close()
    if "con_sem" in vars(ctx):
        ctx.con_sem.close()


def _tool(ctx, name: str):
    return next(tool for tool in build_tools(ctx) if tool.name == name)


def test_agent_offers_exactly_the_full_tool_set(full_ctx):
    assert [tool.name for tool in build_tools(full_ctx)] == _TOOL_NAMES


def test_search_docs_with_an_entity_id_returns_only_that_entitys_documents(full_ctx):
    documents = [json.loads(line) for line in (full_ctx.docs_dir / "documents.jsonl").read_text().splitlines()]
    wheat_docs = {doc["doc_id"] for doc in documents if _WHEAT_TAG in doc["tags"]["crop_tag"]}
    assert len(wheat_docs) < len(documents)

    unfiltered = _tool(full_ctx, "search_docs").fn(query="harvest", k=20)
    filtered = _tool(full_ctx, "search_docs").fn(query="harvest", entity_ids=["crop:milling_wheat"], k=20)

    assert {hit["doc_id"] for hit in unfiltered["hits"]} - wheat_docs
    assert {hit["doc_id"] for hit in filtered["hits"]} == wheat_docs
    assert {c["doc_id"] for c in filtered["_citations"]} == wheat_docs


def test_explain_move_tool_returns_the_planted_event_as_top_driver(full_ctx):
    name = next(e.canonical_name for e in full_ctx.catalog if PLANTED_SERIES_CODE in e.aliases)

    result = _tool(full_ctx, "explain_move").fn(entity=name, start="2025-04", end="2025-05")

    driver = result["drivers"][0]
    assert driver["quote"] == PLANTED_QUOTE
    assert driver["docs"] == [PLANTED_BULLETIN_REF, _SECOND_REF]
    assert {"kind": "doc", "doc_id": PLANTED_BULLETIN_REF} in result["_citations"]


def test_get_context_adds_document_window_event_types_and_graph_neighbours(full_ctx):
    name = next(e.canonical_name for e in full_ctx.catalog if e.id == "crop:milling_wheat")

    result = get_context(full_ctx, f'What happened to "{name}"?')

    assert result["coverage"]["documents"] == ["2025-03-01", "2025-06-30"]
    assert result["event_types"] == [item["name"] for item in full_ctx.ontology["event_types"]]
    assert 0 < len(result["graph_neighbours"]) <= 5
    assert {edge["entity"] for edge in result["graph_neighbours"]} == {"crop:milling_wheat"}


def test_trace_keeps_the_search_docs_result_for_citation_checks(full_ctx):
    call = ToolCall("search_docs", {"query": "drought", "entity_ids": ["crop:milling_wheat"]})
    llm = FakeLLM({"answer": [Reply("", [call], 0.0, 1, 1), Reply("done", [], 0.0, 1, 1)]})

    answer = run_loop("q", "sys", build_tools(full_ctx), llm, 4, 6000)

    (entry,) = answer.tool_trace
    assert PLANTED_BULLETIN_REF in {hit["doc_id"] for hit in entry["result"]["hits"]}
    assert "_citations" not in entry["result"]


def test_trace_result_is_trimmed_like_the_transcript(full_ctx):
    call = ToolCall("search_docs", {"query": "drought", "k": 5})
    llm = FakeLLM({"answer": [Reply("", [call], 0.0, 1, 1), Reply("done", [], 0.0, 1, 1)]})

    answer = run_loop("q", "sys", build_tools(full_ctx), llm, 4, 800)

    kept = answer.tool_trace[0]["result"]["hits"]
    assert 0 < len(kept) < 5
    assert len(json.dumps(answer.tool_trace[0]["result"])) <= 800
