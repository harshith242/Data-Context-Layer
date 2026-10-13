"""`ctx review apply`, the REVIEW.md frame and the applied-decision record, on domain_a."""
import shutil

import pytest
import yaml
from fakes import FakeEmbedder
from fixtures.domain_a.make_domain import PLANTED_BULLETIN_REF
from fixtures.domain_a.unstructured_build import copy_built, unstructured_llm

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.files import read_json, read_jsonl, read_yaml
from ctxlayer.common.phase_context import make_context
from ctxlayer.pipeline.review import review_apply, review_header
from ctxlayer.pipeline.runner import Review, run_build

_BY = {"by": "tester", "at": "2026-10-12"}
_MOVE = "move:WHT_MILL_GP:weekly:USD:per_tonne:2025-05-05"
_DRIVER_KEY = f"{PLANTED_BULLETIN_REF}#0"
_DRIVER = {"kind": "driver", "target": {"move_id": _MOVE, "key": _DRIVER_KEY}, "verdict": "wrong", "fix": {}, **_BY}
_ROLE = {
    "kind": "column", "target": {"table": "grain_quotes", "column": "crop"}, "verdict": "fix",
    "fix": {"description": "Crop traded."}, **_BY,
}
_NEW_ENTITY = {"kind": "new_entity", "target": {"id": "crop:sunflower"}, "verdict": "correct", "fix": {}, **_BY}


@pytest.fixture(scope="module")
def built(tmp_path_factory, domain_a, domain_a_built):
    """Settings over one full run_build; read-only, every test works on a copy."""
    settings = copy_built(tmp_path_factory.mktemp("review"), domain_a, domain_a_built)
    shutil.rmtree(artifacts_dir(settings))
    run_build(make_context(settings, unstructured_llm(), FakeEmbedder()), None, None, Review.OFF, force_start=False)
    return settings


@pytest.fixture
def applying(built, domain_a, tmp_path):
    """applying(entries) -> (pc, llm): decisions written on a copy of the build, ready for review_apply."""

    def prepare(entries: list[dict]):
        settings = copy_built(tmp_path, domain_a, built)
        review = settings.path("paths.review")
        review.mkdir(parents=True)
        (review / "decisions.yaml").write_text(yaml.safe_dump({"decisions": entries}))
        llm = unstructured_llm()
        return make_context(settings, llm, FakeEmbedder()), llm

    return prepare


def _drivers(pc) -> list[dict]:
    return read_jsonl(pc.out_dir / "links" / "candidate_drivers.jsonl")


def test_a_driver_decision_runs_links_only_without_model_calls(applying):
    pc, llm = applying([_DRIVER])
    before = _drivers(pc)

    result = review_apply(pc)

    assert result == {"ran": ["links"], "needs_rerun": []}
    assert llm.calls == []
    assert [row["move_id"] for row in before] == [_MOVE] and _drivers(pc) == []


def test_a_new_entity_decision_runs_resolve_onwards_without_model_calls(applying):
    pc, llm = applying([_NEW_ENTITY])

    result = review_apply(pc)

    assert result["ran"] == ["resolve", "graph", "links"] and result["needs_rerun"] == []
    # Resolve still asks which entity type each tag column holds; the name-matching verdicts come from the log.
    assert all(prompt.startswith("[task: tag_types]") for prompt, _ in llm.calls)


def test_a_column_decision_is_listed_for_a_rerun_and_the_driver_is_applied(applying):
    pc, llm = applying([_ROLE, _DRIVER])

    result = review_apply(pc)

    assert result["ran"] == ["links"] and llm.calls == []
    assert result["needs_rerun"] == [
        {"kind": "column", "target": _ROLE["target"], "command": "ctx build --from roles"},
    ]
    assert _drivers(pc) == []


def test_review_apply_with_nothing_pending_runs_nothing(applying):
    pc, _ = applying([_DRIVER])
    review_apply(pc)

    assert review_apply(pc) == {"ran": [], "needs_rerun": []}


def test_review_md_starts_with_the_header_and_counts_applied_unmatched_and_invalid(applying):
    gone = {**_DRIVER, "target": {"move_id": "move:gone", "key": "x#0"}}
    broken = {"kind": "nonsense", "target": {}, "verdict": "wrong", **_BY}
    pc, _ = applying([_DRIVER, gone, broken])

    run_build(pc, "links", None, Review.OFF, force_start=True)
    text = (pc.out_dir / "REVIEW.md").read_text()
    state = read_json(pc.out_dir / "build_state.json")

    assert text.startswith(review_header(1, 1, 1))
    assert text.count("Decisions: ") == 1
    assert text.index("## Roles") < text.index("## Decisions that no longer match") < text.index("## Invalid decisions")
    assert "move:gone" in text.split("## Decisions that no longer match")[1].split("## Invalid decisions")[0]
    assert "unknown kind: nonsense" in text.split("## Invalid decisions")[1]
    assert len(state["phases"]["links"]["applied"]) == 1


def test_a_phase_skipped_as_up_to_date_keeps_its_decisions_applied(applying):
    pc, _ = applying([_DRIVER])
    run_build(pc, "links", None, Review.OFF, force_start=True)

    again = run_build(pc, None, None, Review.OFF, force_start=False)

    assert again["ran"] == []
    assert (pc.out_dir / "REVIEW.md").read_text().startswith(review_header(1, 0, 0))


def test_low_confidence_edges_are_one_line_per_target_sorted_by_count(built):
    text = (artifacts_dir(built) / "REVIEW.md").read_text()
    section = text.split("## Low-confidence edges")[1].split("\n## ")[0]

    assert "- crop:sunflower: new entity from tag column crop_tag; 1 edges, 1 articles" in section


@pytest.mark.parametrize("phase", ["catalog", "semantic"])
def test_an_entity_or_rule_decision_reapplies_the_saved_draft_without_model_calls(applying, built, phase):
    entity = read_yaml(artifacts_dir(built) / "entities.yaml")[0]["id"]
    entries = {
        "catalog": {"kind": "entity", "target": {"id": entity}, "verdict": "correct", "fix": {}, **_BY},
        "semantic": {"kind": "rule", "target": {"table": "grain_quotes", "field": "time.grains"}, "verdict": "fix",
                     "fix": {"value": ["weekly"]}, **_BY},
    }
    pc, llm = applying([entries[phase]])

    result = review_apply(pc)

    # Resolve, run after the catalog, still asks which type each tag column holds; nothing else is asked.
    assert phase in result["ran"] and all(prompt.startswith("[task: tag_types]") for prompt, _ in llm.calls)


def test_a_decision_waiting_for_its_phase_is_pending_not_unmatched(applying):
    pc, _ = applying([_DRIVER, _ROLE])

    run_build(pc, "links", None, Review.OFF, force_start=True)
    text = (pc.out_dir / "REVIEW.md").read_text()
    pending = text.split("## Pending")[1].split("## Decisions that no longer match")[0]
    unmatched = text.split("## Decisions that no longer match")[1].split("## Invalid decisions")[0]

    assert text.startswith(review_header(1, 0, 0, pending=1))
    assert "crop" in pending and unmatched.strip() == "none"
