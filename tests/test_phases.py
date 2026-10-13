"""Phase functions on domain_a: same output as the wrappers, file inputs and outputs, event_types.yaml."""
import pytest
from fakes import FakeEmbedder, FakeLLM
from fixtures.domain_a.make_domain import PLANTED_BULLETIN_REF
from fixtures.domain_a.structured_build import structured_llm
from fixtures.domain_a.unstructured_build import copy_built, unstructured_llm

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.files import read_jsonl, read_yaml, write_yaml
from ctxlayer.common.phase_context import make_context
from ctxlayer.linking.build import build_links
from ctxlayer.structured.build import build_structured
from ctxlayer.structured.phases import (
    run_catalog, run_moves, run_ontology, run_profile, run_roles, run_semantic, run_views,
)
from ctxlayer.unstructured.build import build_unstructured
from ctxlayer.unstructured.phases import (
    run_dedupe, run_documents, run_event_types, run_extraction, run_graph, run_index, run_resolve,
)

# profile.json is left out: its sample values are drawn at random on every run.
_STRUCTURED_FILES = [
    "roles.yaml", "ontology.yaml", "entities.yaml", "semantic.yaml",
    "structured/views.sql", "structured/moves.jsonl", "REVIEW.md",
]
_UNSTRUCTURED_FILES = [
    "docs/documents.jsonl", "event_types.yaml", "docs/extractions.jsonl", "docs/failures.jsonl",
    "docs/resolved.jsonl", "docs/tag_edges.jsonl", "graph/entities_text.yaml", "docs/events.jsonl",
    "graph/nodes.jsonl", "graph/edges.jsonl", "REVIEW.md",
]
_SECOND_REF = "FB-20250427-01"
_STRUCTURED_PHASES = (run_profile, run_roles, run_ontology, run_catalog, run_semantic, run_views, run_moves)
_UNSTRUCTURED_PHASES = (
    run_documents, run_index, run_event_types, run_extraction, run_resolve, run_dedupe, run_graph,
)


def _no_model() -> FakeLLM:
    """Any call fails: a phase that only reads files must not ask a model."""
    return FakeLLM({})


@pytest.fixture(scope="module")
def wrapped(tmp_path_factory, domain_a, domain_a_built):
    """(settings, unstructured counts, links counts) of the wrappers on a copy of the structured build."""
    settings = copy_built(tmp_path_factory.mktemp("wrapped"), domain_a, domain_a_built)
    counts = build_unstructured(settings, unstructured_llm(), FakeEmbedder())
    return settings, counts, build_links(settings)


def _same_files(first, second, names) -> list[str]:
    return [
        name for name in names
        if (artifacts_dir(first) / name).read_bytes() != (artifacts_dir(second) / name).read_bytes()
    ]


def test_structured_phases_one_by_one_give_the_wrapper_counts_and_files(tmp_settings, tmp_path, domain_a):
    database = {"path": str(domain_a.db_path)}
    by_wrapper = tmp_settings({"database": database, "paths": {"artifacts": str(tmp_path / "wrapper")}})
    one_by_one = tmp_settings({"database": database, "paths": {"artifacts": str(tmp_path / "phases")}})
    expected = build_structured(by_wrapper, structured_llm(), FakeEmbedder())

    pc = make_context(one_by_one, structured_llm(), FakeEmbedder())
    counts = [run(pc) for run in _STRUCTURED_PHASES]

    assert counts[-1] == expected
    assert _same_files(by_wrapper, one_by_one, _STRUCTURED_FILES) == []
    assert "event_types" not in read_yaml(artifacts_dir(one_by_one) / "ontology.yaml")


def test_unstructured_phases_one_by_one_give_the_wrapper_counts_and_files(
    tmp_path_factory, domain_a, domain_a_built, wrapped,
):
    settings, expected, _ = wrapped
    other = copy_built(tmp_path_factory.mktemp("phases"), domain_a, domain_a_built)

    pc = make_context(other, unstructured_llm(), FakeEmbedder())
    counts = [run(pc) for run in _UNSTRUCTURED_PHASES]

    assert counts[-1] == expected
    assert _same_files(settings, other, _UNSTRUCTURED_FILES) == []


def test_run_graph_alone_rebuilds_the_graph_from_the_intermediates_without_a_model(
    tmp_path_factory, domain_a, wrapped,
):
    settings, expected, _ = wrapped
    other = copy_built(tmp_path_factory.mktemp("graph"), domain_a, settings)
    out = artifacts_dir(other)
    (out / "graph" / "nodes.jsonl").unlink()
    (out / "graph" / "edges.jsonl").unlink()
    llm = _no_model()

    counts = run_graph(make_context(other, llm, FakeEmbedder()))

    assert llm.calls == []
    assert counts == expected
    assert _same_files(settings, other, ["graph/nodes.jsonl", "graph/edges.jsonl", "REVIEW.md"]) == []


def test_resolved_events_carry_extraction_keys_and_canonical_events_list_their_members(wrapped):
    out = artifacts_dir(wrapped[0])
    rows = read_jsonl(out / "docs" / "resolved.jsonl")
    keys = [item["key"] for row in rows for item in row["events"]]
    planted = [row for row in rows if row["doc_id"] == PLANTED_BULLETIN_REF]

    assert all(set(row) == {"doc_id", "mentions", "events"} for row in rows)
    assert [item["key"] for item in planted[0]["events"]] == [f"{PLANTED_BULLETIN_REF}#0"]
    assert len(keys) == len(set(keys))

    events = read_jsonl(out / "docs" / "events.jsonl")
    assert sorted(key for event in events for key in event["keys"]) == sorted(keys)
    merged = next(event for event in events if f"{PLANTED_BULLETIN_REF}#0" in event["keys"])
    assert set(merged["keys"]) == {f"{PLANTED_BULLETIN_REF}#0", f"{_SECOND_REF}#0"}


def test_event_types_of_a_pre_plan_build_move_over_unchanged_without_a_model(tmp_path_factory, domain_a, wrapped):
    other = copy_built(tmp_path_factory.mktemp("legacy"), domain_a, wrapped[0])
    out = artifacts_dir(other)
    induced = read_yaml(out / "event_types.yaml")["event_types"]
    ontology = read_yaml(out / "ontology.yaml")
    write_yaml(out / "ontology.yaml", {**ontology, "event_types": induced})
    (out / "event_types.yaml").unlink()
    llm = _no_model()

    run_event_types(make_context(other, llm, FakeEmbedder()))

    assert llm.calls == []
    assert read_yaml(out / "event_types.yaml") == {"ontology_version": ontology["version"], "event_types": induced}


def test_event_types_are_induced_again_when_the_ontology_version_changed(tmp_path_factory, domain_a, wrapped):
    other = copy_built(tmp_path_factory.mktemp("newversion"), domain_a, wrapped[0])
    out = artifacts_dir(other)
    write_yaml(out / "event_types.yaml", {"ontology_version": "older", "event_types": [{"name": "stale"}]})
    llm = unstructured_llm()

    run_event_types(make_context(other, llm, FakeEmbedder()))

    names = [item["name"] for item in read_yaml(out / "event_types.yaml")["event_types"]]
    assert "stale" not in names and names[-1] == "other"
    assert sum(prompt.startswith("[task: event_types]") for prompt, _ in llm.calls) == 1
