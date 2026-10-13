"""Review decisions of the unstructured and linking phases: each is applied by its own phase and survives a rerun."""
import contextlib
import json
import shutil

import pytest
import yaml
from fakes import FakeEmbedder
from fixtures.domain_a.make_domain import PLANTED_BULLETIN_REF
from fixtures.domain_a.unstructured_build import copy_built, unstructured_llm

from ctxlayer.common.catalog import load_catalog
from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.decisions import Decisions
from ctxlayer.common.files import read_jsonl, read_yaml
from ctxlayer.common.phase_context import make_context
from ctxlayer.linking.bridge import explain_move
from ctxlayer.pipeline.phases import EXTRACT_PROMPT_VERSION, PHASES
from ctxlayer.pipeline.runner import Review, phase_status, run_build
from ctxlayer.structured.semantic_views import open_semantic
from ctxlayer.unstructured.dedupe import apply_same_event_decisions
from ctxlayer.unstructured.graph_store import GraphStore, ReviewFilter

_BY = {"by": "tester", "at": "2026-10-12"}
_FIRST_KEY = f"{PLANTED_BULLETIN_REF}#0"
_SECOND_KEY = "FB-20250427-01#0"
_MOVE = "move:WHT_MILL_GP:weekly:USD:per_tonne:2025-05-05"
_WHEAT = "crop:milling_wheat"


def _decision(kind: str, target: dict, verdict: str, fix: dict | None = None) -> dict:
    return {"kind": kind, "target": target, "verdict": verdict, "fix": fix or {}, **_BY}


@pytest.fixture(scope="module")
def built(tmp_path_factory, domain_a, domain_a_built):
    """Settings over one full run_build; read-only, every test rebuilds on a copy."""
    settings = copy_built(tmp_path_factory.mktemp("unstructured_decisions"), domain_a, domain_a_built)
    shutil.rmtree(artifacts_dir(settings))
    run_build(make_context(settings, unstructured_llm(), FakeEmbedder()), None, None, Review.OFF, force_start=False)
    return settings


@pytest.fixture
def rebuild(built, domain_a, tmp_path):
    """rebuild(entries, start, stop, drop) -> (pc, out dir): decisions written, the `drop` files removed from the
    copy, then the range forced from `start`."""

    def run(entries: list[dict], start: str, stop: str | None = None, drop: tuple[str, ...] = ()):
        settings = copy_built(tmp_path, domain_a, built)
        for name in drop:
            (artifacts_dir(settings) / name).unlink()
        review = settings.path("paths.review")
        review.mkdir(parents=True)
        (review / "decisions.yaml").write_text(yaml.safe_dump({"decisions": entries}))
        pc = make_context(settings, unstructured_llm(), FakeEmbedder())
        run_build(pc, start, stop, Review.OFF, force_start=True)
        return pc, artifacts_dir(settings)

    return run


def _tag_edges(out, text: str) -> list[dict]:
    return [edge for edge in read_jsonl(out / "docs" / "tag_edges.jsonl") if edge["text"] == text]


def _new_entities(out):
    return load_catalog(out / "graph" / "entities_text.yaml")


def _graph(out) -> GraphStore:
    return GraphStore.load([out / "graph" / "nodes.jsonl", out / "graph" / "edges.jsonl"])


def _affects(out) -> dict:
    return next(e for e in read_jsonl(out / "graph" / "edges.jsonl") if e["rel"] == "AFFECTS" and e["dst"] == _WHEAT)


def _judged(pc, text: str) -> list[str]:
    return [p for p, _ in pc.llm.calls if p.startswith("[task: resolve_judge") and f"Mention: {text}" in p]


def test_an_event_fix_shows_on_the_affects_edge_and_the_driver(rebuild):
    fix = {"affects": [{"entity": _WHEAT, "direction": "down"}]}

    pc, out = rebuild([_decision("event", {"key": _FIRST_KEY}, "fix", fix)], "dedupe")

    event = read_jsonl(out / "docs" / "events.jsonl")[0]
    edge = _affects(out)
    drivers = read_jsonl(out / "links" / "candidate_drivers.jsonl")
    assert {affect["direction"] for affect in event["affects"]} == {"down"}
    assert (event["source"], event["confidence"], event["needs_review"]) == ("human", 1.0, False)
    assert (edge["props"]["direction"], edge["source"], edge["needs_review"]) == ("down", "human", False)
    assert {item["direction"] for item in edge["evidence"]} == {"down"}
    assert [(row["move_id"], row["direction_match"]) for row in drivers] == [(_MOVE, "opposite")]
    assert pc.decisions.unmatched() == []


def test_an_event_decision_matches_any_member_key_and_fixes_type_date_and_status(rebuild):
    fix = {"type": "policy_change", "date": "2025-04-26", "status": "announced"}

    pc, out = rebuild([_decision("event", {"key": _SECOND_KEY}, "fix", fix)], "dedupe")

    events = read_jsonl(out / "docs" / "events.jsonl")
    assert [(e["type"], e["date"], e["status"]) for e in events] == [("policy_change", "2025-04-26", "announced")]
    # The graph still finds the member events by key although their type changed.
    reports = [e for e in read_jsonl(out / "graph" / "edges.jsonl") if e["rel"] == "REPORTS"]
    assert {e["evidence_quote"] != "" for e in reports} == {True}
    assert pc.decisions.unmatched() == []


def test_a_wrong_event_is_dropped_and_an_unknown_key_is_unmatched(rebuild):
    entries = [_decision("event", {"key": _FIRST_KEY}, "wrong"), _decision("event", {"key": "none#0"}, "correct")]

    pc, out = rebuild(entries, "dedupe")

    assert read_jsonl(out / "docs" / "events.jsonl") == []
    assert read_jsonl(out / "links" / "candidate_drivers.jsonl") == []
    assert [d.target["key"] for d in pc.decisions.unmatched()] == ["none#0"]


def test_a_name_match_fix_points_the_tag_at_the_entity_with_no_new_entity_and_no_judge_call(rebuild):
    entries = [_decision("name_match", {"text": "Sunflower"}, "fix", {"entity_id": "crop:oats"})]

    # Without the saved verdicts a judge call would be needed to settle the name, if no rule answered first.
    pc, out = rebuild(entries, "resolve", "resolve", drop=("resolve_judge.jsonl",))

    edge = _tag_edges(out, "Sunflower")[0]
    assert (edge["dst"], edge["source"], edge["needs_review"], edge["tag_column"]) == ("crop:oats", "human", False, "crop_tag")
    assert "crop:sunflower" not in {e.id for e in _new_entities(out)}
    assert _judged(pc, "Sunflower") == []
    assert pc.decisions.unmatched() == []


def test_a_wrong_name_match_leaves_the_tag_unmatched(rebuild):
    fix = {"not_entity_id": "location:great_plains"}

    pc, out = rebuild([_decision("name_match", {"text": "GP"}, "wrong", fix)], "resolve", "resolve")

    destinations = {edge["dst"] for edge in _tag_edges(out, "GP")}
    assert destinations == {"location:gp"}
    assert "location:gp" in {entity.id for entity in _new_entities(out)}
    assert pc.decisions.unmatched() == []


def test_a_new_entity_can_be_merged_into_a_catalog_entity(rebuild):
    entries = [_decision("new_entity", {"id": "crop:sunflower"}, "fix", {"merge_into": "crop:oats"})]

    pc, out = rebuild(entries, "resolve")

    edge = _tag_edges(out, "Sunflower")[0]
    assert (edge["dst"], edge["needs_review"], edge["source"]) == ("crop:oats", False, "human")
    assert "crop:sunflower" not in {entity.id for entity in _new_entities(out)}
    assert "crop:sunflower" not in {node["id"] for node in read_jsonl(out / "graph" / "nodes.jsonl")}
    assert pc.decisions.unmatched() == []


def test_approving_a_new_entity_unflags_it_and_its_tag_edges(built, rebuild):
    tagged = "location:nile_delta"
    doc = _tag_edges(artifacts_dir(built), "Nile Delta")[0]["src"]

    def kept(out):
        edges = _graph(out).neighbors(doc, rel="TAGGED_WITH", review=ReviewFilter.EXCLUDE)
        return [edge["dst"] for edge in edges if edge["src"] == doc]

    pc, out = rebuild([_decision("new_entity", {"id": tagged}, "correct")], "resolve")

    entity = next(e for e in _new_entities(out) if e.id == tagged)
    assert tagged not in kept(artifacts_dir(built))
    assert tagged in kept(out)
    assert (entity.source, entity.confidence, entity.needs_review) == ("human", 1.0, False)
    assert pc.decisions.unmatched() == []


def test_a_wrong_new_entity_is_dropped_with_its_tag_edges(rebuild):
    pc, out = rebuild([_decision("new_entity", {"id": "location:nile_delta"}, "wrong")], "resolve")

    assert _tag_edges(out, "Nile Delta") == []
    assert "location:nile_delta" not in {node["id"] for node in read_jsonl(out / "graph" / "nodes.jsonl")}
    assert pc.decisions.unmatched() == []


def test_a_wrong_same_event_keeps_the_two_bulletins_apart(rebuild, built):
    assert len(read_jsonl(artifacts_dir(built) / "docs" / "events.jsonl")) == 1

    pc, out = rebuild([_decision("same_event", {"keys": [_SECOND_KEY, _FIRST_KEY]}, "wrong")], "dedupe", "dedupe")

    events = read_jsonl(out / "docs" / "events.jsonl")
    assert sorted(event["keys"] for event in events) == [[_FIRST_KEY], [_SECOND_KEY]]
    assert pc.decisions.unmatched() == []


def test_a_correct_same_event_merges_two_groups(tmp_path):
    path = tmp_path / "decisions.yaml"
    path.write_text(yaml.safe_dump({"decisions": [_decision("same_event", {"keys": ["b#0", "a#0"]}, "correct")]}))
    decisions = Decisions.load(path)
    groups = [[{"key": "a#0"}], [{"key": "c#0"}], [{"key": "b#0"}]]

    regrouped = apply_same_event_decisions(groups, decisions)

    assert [[event["key"] for event in group] for group in regrouped] == [["a#0", "b#0"], ["c#0"]]
    assert decisions.unmatched() == []


def _explained(pc, out):
    semantic = read_yaml(out / "semantic.yaml")
    store = GraphStore.load([
        out / "graph" / "nodes.jsonl", out / "graph" / "edges.jsonl",
        out / "links" / "nodes.jsonl", out / "links" / "edges.jsonl",
    ])
    links = read_jsonl(out / "links" / "candidate_drivers.jsonl")
    moves = read_jsonl(out / "structured" / "moves.jsonl")
    with contextlib.closing(open_semantic(pc.settings)) as con_sem:
        return explain_move(con_sem, semantic, store, links, moves, "series:wht_mill_gp", "2025-04", "2025-05")


def test_a_wrong_driver_removes_it_and_explain_move_shows_none(rebuild, built):
    # Written with the second member's key: any member key of the event matches.
    entries = [_decision("driver", {"move_id": _MOVE, "key": _SECOND_KEY}, "wrong")]

    pc, out = rebuild(entries, "links")

    result = _explained(pc, out)
    edges = read_jsonl(out / "links" / "edges.jsonl")
    assert result["drivers"] == [] and [m["id"] for m in result["moves"]] == [_MOVE]
    assert not [e for e in edges if e["rel"] == "CANDIDATE_DRIVER_OF"]
    assert pc.decisions.unmatched() == []


def test_a_confirmed_driver_stays_and_is_human(rebuild):
    entries = [_decision("driver", {"move_id": _MOVE, "key": _FIRST_KEY}, "correct")]

    pc, out = rebuild(entries, "links")

    drivers = read_jsonl(out / "links" / "candidate_drivers.jsonl")
    edges = [e for e in read_jsonl(out / "links" / "edges.jsonl") if e["rel"] == "CANDIDATE_DRIVER_OF"]
    assert [(row["move_id"], row["source"]) for row in drivers] == [(_MOVE, "human")]
    assert [edge["source"] for edge in edges] == ["human"]
    assert len(_explained(pc, out)["drivers"]) == 1


def test_a_bumped_extraction_prompt_marks_extraction_stale(built, domain_a, tmp_path):
    settings = copy_built(tmp_path, domain_a, built)
    state_path = artifacts_dir(settings) / "build_state.json"
    state = json.loads(state_path.read_text())
    state["phases"]["extraction"]["parts"]["version"] = "1+extract-v0"
    state_path.write_text(json.dumps(state))

    rows = {row["phase"]: row for row in phase_status(make_context(settings, unstructured_llm(), FakeEmbedder()))}

    extraction = next(phase for phase in PHASES if phase.name == "extraction")
    assert extraction.version.endswith(EXTRACT_PROMPT_VERSION)
    assert (rows["extraction"]["state"], rows["extraction"]["reason"]) == ("stale", "phase logic changed")
    assert rows["event-types"]["state"] == "up to date"
