"""get_context, coverage and example questions over artifacts written from the synthetic domain."""
import dataclasses

import pytest

from ctxlayer.common.catalog import save_catalog
from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.files import write_yaml
from ctxlayer.serving.context import coverage, example_questions, get_context
from ctxlayer.serving.tools import open_context
from ctxlayer.unstructured.graph_store import ReviewFilter
from fakes import FakeEmbedder

_PLANTED_NAME = "Milling Wheat - Great Plains"
_FLAGGED_ID = "series:oat_pb"
_FLAGGED_NAME = "Oats - Prairie Basin"


@pytest.fixture(scope="module")
def ctx(domain_a_semantic, domain_a_catalog):
    settings, semantic = domain_a_semantic
    ontology, entities = domain_a_catalog

    # One entity is below the review threshold.
    entities = [dataclasses.replace(e, needs_review=e.id == _FLAGGED_ID) for e in entities]
    save_catalog(entities, artifacts_dir(settings) / "entities.yaml")
    write_yaml(artifacts_dir(settings) / "ontology.yaml", ontology)

    context = open_context(settings, embedder=FakeEmbedder())
    yield context
    context.con.close()


def _names(result: dict) -> list[str]:
    return [entity["canonical_name"] for entity in result["entities"]]


def test_entity_named_in_question_is_returned(ctx):
    result = get_context(ctx, f"What is the price of {_PLANTED_NAME}?")

    assert _names(result)[0] == _PLANTED_NAME
    assert result["entities"][0]["id"] == "series:wht_mill_gp"


def test_needs_review_entity_is_excluded_by_default(ctx):
    question = f"Show {_FLAGGED_NAME}"

    assert _FLAGGED_NAME not in _names(get_context(ctx, question))
    assert _names(get_context(ctx, question, ReviewFilter.INCLUDE))[0] == _FLAGGED_NAME


def test_semantic_items_rank_by_question_and_carry_source_and_confidence(ctx):
    result = get_context(ctx, "Date the quote was last revised")

    top = result["semantic_items"][0]
    assert top["text"] == "grain_quotes.revised_date: Date the quote was last revised."
    assert (top["source"], top["confidence"]) == ("inferred", 0.95)
    assert all(item["source"] and "confidence" in item for item in result["semantic_items"] + result["entities"])


def test_coverage_holds_time_range_and_now_of_every_view(ctx):
    result = get_context(ctx, "anything")["coverage"]

    assert result == coverage(ctx)
    quotes = next(v for v in result["views"] if v["view"] == "sem_grain_quotes")
    assert quotes["time_range"] == ["2024-01-01", "2025-12-29"]
    assert {v["now"] for v in result["views"]} == {ctx.semantic["now"]["value"]} == {"2025-06-30"}
    assert result["documents"] is None


def test_example_questions_name_measured_series_not_locations(ctx):
    questions = example_questions(ctx, n=3)
    series = {e.canonical_name for e in ctx.catalog if e.type == "Series"}

    assert [q.split(" of ")[0] for q in questions] == [
        "What is the latest value", "What is the change pct", "What is the volatility",
    ]
    assert all(q.split(" of ", 1)[1].rstrip("?") in series for q in questions)
