"""Fixes from the Plan 4 final review (.sdd/reports/final-review.md) that need no build."""
import yaml

from ctxlayer.common.catalog import Entity, apply_entity_decisions
from ctxlayer.common.decisions import Decisions
from ctxlayer.common.files import write_yaml
from ctxlayer.common.ontology import _version, final_event_types, load_ontology

_BY = {"by": "T", "at": "2026-10-12"}


def _decisions(tmp_path, entries: list[dict]) -> Decisions:
    path = tmp_path / "decisions.yaml"
    path.write_text(yaml.safe_dump({"decisions": [{**_BY, **entry} for entry in entries]}))
    return Decisions.load(path)


def test_confirming_a_role_does_not_change_the_ontology_version():
    column = {"role": "measure", "confidence": 0.5, "needs_review": True, "source": "inferred"}
    roles = {"tables": {"t": {"columns": {"c": column}}}}
    confirmed = {"tables": {"t": {"columns": {"c": {**column, "confidence": 1.0, "needs_review": False, "source": "human"}}}}}
    changed = {"tables": {"t": {"columns": {"c": {**column, "role": "dimension"}}}}}

    assert _version(roles, {}, "") == _version(confirmed, {}, "") != _version(changed, {}, "")


def test_ontology_loads_the_final_event_types_but_extraction_the_prompt_ones(tmp_path):
    prompt = [{"name": "a"}, {"name": "b"}, {"name": "other"}]
    write_yaml(tmp_path / "ontology.yaml", {"types": []})
    write_yaml(tmp_path / "event_types.yaml", {
        "event_types": prompt, "final_types": final_event_types(prompt, {"a": "a2", "b": "other"}),
    })

    assert [item["name"] for item in load_ontology(tmp_path)["event_types"]] == ["a2", "other"]
    assert load_ontology(tmp_path, prompt=True)["event_types"] == prompt


def test_verdicts_the_spec_marks_as_none_and_empty_fixes_are_invalid(tmp_path):
    decisions = _decisions(tmp_path, [
        {"kind": "rule", "target": {"table": "t", "field": "f"}, "verdict": "correct"},
        {"kind": "column", "target": {"table": "t", "column": "c"}, "verdict": "wrong"},
        {"kind": "entity", "target": {"id": "x"}, "verdict": "fix", "fix": {}},
    ])

    assert [item["reason"] for item in decisions.invalid] == [
        "rule has no correct verdict", "column has no wrong verdict", "fix verdict needs a fix",
    ]


def test_a_removed_entity_is_dropped_from_the_references_of_the_others(tmp_path):
    gone = Entity("location:gone", "Location", "Gone")
    kept = Entity("location:kept", "Location", "Kept")
    item = Entity("crop:a", "Crop", "A", attributes={
        "parents": ["location:gone", "location:kept"], "locations": ["location:gone"], "part_of": "location:gone",
    })
    decisions = _decisions(tmp_path, [{"kind": "entity", "target": {"id": "location:gone"}, "verdict": "wrong"}])

    result = {entity.id: entity for entity in apply_entity_decisions([gone, kept, item], decisions)}

    assert set(result) == {"location:kept", "crop:a"}
    assert result["crop:a"].attributes == {"parents": ["location:kept"], "locations": [], "part_of": None}
