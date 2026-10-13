"""Tests for the decisions file: load, validate, merge, usage tracking."""
from pathlib import Path

import yaml

from ctxlayer.common.decisions import (
    Decisions,
    Kind,
    Verdict,
    merge_decisions,
)

VALID = [
    {"kind": "entity", "target": {"id": "crop:wheat"}, "verdict": "fix",
     "fix": {"canonical_name": "Wheat"}, "by": "Alex", "at": "2026-10-12", "note": "n"},
    {"kind": "driver", "target": {"move_id": "m1", "key": "d1"}, "verdict": "wrong",
     "by": "Alex", "at": "2026-10-12"},
    {"kind": "name_match", "target": {"text": "GP"}, "verdict": "fix",
     "fix": {"entity_id": "location:great_plains"}, "by": "Sam", "at": "2026-10-11"},
]
INVALID = [
    {"kind": "bogus", "target": {"id": "x"}, "verdict": "correct", "by": "A", "at": "2026-10-12"},
    {"kind": "entity", "target": {}, "verdict": "correct", "by": "A", "at": "2026-10-12"},
    {"kind": "entity", "target": {"id": "x"}, "verdict": "fix",
     "fix": {"nonsense": 1}, "by": "A", "at": "2026-10-12"},
    {"kind": "entity", "target": {"id": "y"}, "verdict": "correct", "at": "2026-10-12"},
]


def _write(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "decisions.yaml"
    path.write_text(yaml.safe_dump({"decisions": entries}))
    return path


def _entity(id_: str, at: str, verdict: str = "correct", fix: dict | None = None) -> dict:
    entry = {"kind": "entity", "target": {"id": id_}, "verdict": verdict,
             "by": "A", "at": at}
    if fix:
        entry["fix"] = fix
    return entry


def test_loads_valid_and_reports_invalid(tmp_path):
    decisions = Decisions.load(_write(tmp_path, VALID + INVALID))

    assert len(decisions.of(Kind.ENTITY)) == 1
    assert len(decisions.of(Kind.DRIVER)) == 1
    assert len(decisions.of(Kind.NAME_MATCH)) == 1
    assert len(decisions.invalid) == 4
    reasons = " | ".join(item["reason"] for item in decisions.invalid)
    assert "unknown kind" in reasons
    assert "missing target key: id" in reasons
    assert "fix key not allowed: nonsense" in reasons
    assert "missing by" in reasons
    assert decisions.invalid[0]["entry"] == INVALID[0]






def test_yaml_syntax_error_gives_one_invalid_entry(tmp_path):
    path = tmp_path / "decisions.yaml"
    path.write_text("decisions: [unclosed\n  - : :\n")

    decisions = Decisions.load(path)

    assert decisions.of(Kind.ENTITY) == []
    assert len(decisions.invalid) == 1
    assert "YAML" in decisions.invalid[0]["reason"]
    assert "\n" not in decisions.invalid[0]["reason"]


def test_bad_structure_is_invalid_not_a_crash(tmp_path):
    path = tmp_path / "decisions.yaml"
    path.write_text("decisions: 5\n")
    assert len(Decisions.load(path).invalid) == 1

    path.write_text("- just\n- a list\n")
    assert len(Decisions.load(path).invalid) == 1

    path.write_text("")
    assert Decisions.load(path).invalid == []

    path.write_text(yaml.safe_dump({"decisions": ["not a mapping"]}))
    assert len(Decisions.load(path).invalid) == 1


def test_unquoted_yaml_date_is_accepted(tmp_path):
    path = tmp_path / "decisions.yaml"
    path.write_text(
        "decisions:\n  - {kind: entity, target: {id: a}, verdict: correct, by: A, at: 2026-10-12}\n"
    )

    decision = Decisions.load(path).find(Kind.ENTITY, id="a")

    assert decision.at == "2026-10-12"


def test_later_at_wins(tmp_path):
    entries = [_entity("a", "2026-10-12", "wrong"), _entity("a", "2026-10-01", "correct")]

    decisions = Decisions.load(_write(tmp_path, entries))

    assert [d.verdict for d in decisions.of(Kind.ENTITY)] == [Verdict.WRONG]


def test_equal_at_later_in_file_wins(tmp_path):
    entries = [_entity("a", "2026-10-12", "correct"), _entity("a", "2026-10-12", "wrong")]

    decisions = Decisions.load(_write(tmp_path, entries))

    assert [d.verdict for d in decisions.of(Kind.ENTITY)] == [Verdict.WRONG]


def test_same_event_pair_order_is_irrelevant(tmp_path):
    entries = [
        {"kind": "same_event", "target": {"keys": ["d1#0", "d2#1"]}, "verdict": "correct",
         "by": "A", "at": "2026-10-01"},
        {"kind": "same_event", "target": {"keys": ["d2#1", "d1#0"]}, "verdict": "wrong",
         "by": "A", "at": "2026-10-02"},
    ]

    decisions = Decisions.load(_write(tmp_path, entries))

    assert [d.verdict for d in decisions.of(Kind.SAME_EVENT)] == [Verdict.WRONG]
    assert decisions.find(Kind.SAME_EVENT, keys=["d1#0", "d2#1"]).verdict is Verdict.WRONG






def test_unmatched_lists_only_unused(tmp_path):
    decisions = Decisions.load(_write(tmp_path, VALID))

    decisions.mark_used(decisions.find(Kind.ENTITY, id="crop:wheat"))

    unused = decisions.unmatched()
    assert {d.kind for d in unused} == {Kind.DRIVER, Kind.NAME_MATCH}


def test_superseded_decision_is_not_unmatched(tmp_path):
    entries = [_entity("a", "2026-10-01"), _entity("a", "2026-10-02", "wrong")]
    decisions = Decisions.load(_write(tmp_path, entries))

    decisions.mark_used(decisions.find(Kind.ENTITY, id="a"))

    assert decisions.unmatched() == []


def test_fingerprint_tracks_only_requested_kinds(tmp_path):
    base = Decisions.load(_write(tmp_path, VALID)).fingerprint({Kind.ENTITY})

    changed_entity = [dict(VALID[0], verdict="wrong")] + VALID[1:]
    changed_driver = VALID[:1] + [dict(VALID[1], verdict="correct")] + VALID[2:]

    assert Decisions.load(_write(tmp_path, changed_entity)).fingerprint({Kind.ENTITY}) != base
    assert Decisions.load(_write(tmp_path, changed_driver)).fingerprint({Kind.ENTITY}) == base
    assert Decisions.load(_write(tmp_path, VALID)).fingerprint({Kind.ENTITY}) == base
    assert Decisions.empty().fingerprint({Kind.ENTITY}) != base


def test_merge_newest_wins_others_kept():
    existing = [_entity("a", "2026-10-01", "correct"), _entity("b", "2026-10-01")]
    new = [_entity("a", "2026-10-12", "wrong"), _entity("c", "2026-10-12")]

    merged = merge_decisions(existing, new)

    by_id = {item["target"]["id"]: item for item in merged}
    assert set(by_id) == {"a", "b", "c"}
    assert len(merged) == 3
    assert by_id["a"]["verdict"] == "wrong"


def test_merge_keeps_existing_when_it_is_newer():
    existing = [_entity("a", "2026-10-12", "wrong")]
    new = [_entity("a", "2026-10-01", "correct")]

    merged = merge_decisions(existing, new)

    assert [item["verdict"] for item in merged] == ["wrong"]




def test_merge_keeps_unidentifiable_entries():
    junk = {"kind": "bogus"}

    merged = merge_decisions([junk], [_entity("a", "2026-10-12")])

    assert junk in merged
    assert len(merged) == 2
