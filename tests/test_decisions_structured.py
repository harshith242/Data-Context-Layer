"""Review decisions of the structured phases: each is applied by its own phase and survives a rerun of it."""
import shutil

import pytest
import yaml
from fakes import FakeEmbedder
from fixtures.domain_a.unstructured_build import copy_built, unstructured_llm

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.files import read_jsonl, read_yaml
from ctxlayer.common.phase_context import make_context
from ctxlayer.pipeline.runner import Review, phase_status, run_build

_BY = {"by": "tester", "at": "2026-10-12"}
_PLANTED_MOVE = "move:WHT_MILL_GP:weekly:USD:per_tonne:2025-05-05"
_OTHER_MOVE = "move:OAT_PB:monthly:EUR:per_bushel:2025-01-01"


def _decision(kind: str, target: dict, verdict: str, fix: dict | None = None) -> dict:
    return {"kind": kind, "target": target, "verdict": verdict, "fix": fix or {}, **_BY}


@pytest.fixture(scope="module")
def built(tmp_path_factory, domain_a, domain_a_built):
    """Settings over one full run_build; read-only, every test rebuilds on a copy."""
    settings = copy_built(tmp_path_factory.mktemp("structured_decisions"), domain_a, domain_a_built)
    shutil.rmtree(artifacts_dir(settings))
    run_build(make_context(settings, unstructured_llm(), FakeEmbedder()), None, None, Review.OFF, force_start=False)
    return settings


@pytest.fixture
def rebuild(built, domain_a, tmp_path):
    """rebuild(entries, start, stop) -> (pc, out dir): decisions written, then the range forced from `start`."""

    def run(entries: list[dict], start: str, stop: str | None = None):
        settings = copy_built(tmp_path, domain_a, built)
        review = settings.path("paths.review")
        review.mkdir(parents=True)
        (review / "decisions.yaml").write_text(yaml.safe_dump({"decisions": entries}))
        pc = make_context(settings, unstructured_llm(), FakeEmbedder())
        run_build(pc, start, stop, Review.OFF, force_start=True)
        return pc, artifacts_dir(settings)

    return run


def test_a_column_fix_is_in_roles_yaml_and_the_next_phases_are_stale(rebuild):
    target = {"table": "farm_bulletins", "column": "outlet"}
    entries = [_decision("column", target, "fix", {"role": "doc_tag", "description": "The outlet."})]

    pc, out = rebuild(entries, "roles", "roles")

    column = read_yaml(out / "roles.yaml")["tables"]["farm_bulletins"]["columns"]["outlet"]
    states = {row["phase"]: row for row in phase_status(pc)}
    assert (column["role"], column["description"]) == ("doc_tag", "The outlet.")
    assert (column["source"], column["confidence"], column["needs_review"]) == ("human", 1.0, False)
    assert pc.decisions.unmatched() == []
    assert (states["ontology"]["state"], states["ontology"]["reason"]) == ("stale", "roles.yaml changed")
    assert states["profile"]["state"] == "up to date"


def test_entity_type_and_relation_decisions_change_the_ontology(rebuild):
    entries = [
        _decision("entity_type", {"name": "CropGroup"}, "fix", {"name": "Grouping", "description": "A group."}),
        _decision("relation", {"from": "Crop", "name": "AT_LOCATION", "to": "Location"}, "wrong"),
        _decision("relation", {"from": "Series", "name": "AT_LOCATION", "to": "Location"}, "correct"),
        _decision("entity_type", {"name": "Nothing"}, "wrong"),
    ]

    pc, out = rebuild(entries, "ontology", "ontology")

    ontology = read_yaml(out / "ontology.yaml")
    grouping = next(item for item in ontology["types"] if item["name"] == "Grouping")
    relations = {(r["name"], r["from"], r["to"]): r for r in ontology["relations"]}
    assert (grouping["description"], grouping["source"]) == ("A group.", "human")
    assert "CropGroup" not in {item["name"] for item in ontology["types"]}
    assert ("HAS_PARENT", "Series", "Grouping") in relations
    assert ("AT_LOCATION", "Crop", "Location") not in relations
    assert relations[("AT_LOCATION", "Series", "Location")]["source"] == "human"
    assert [d.target["name"] for d in pc.decisions.unmatched()] == ["Nothing"]


def test_renaming_an_event_type_relabels_the_events_and_extracts_nothing_again(rebuild):
    entries = [_decision("event_type", {"name": "supply_disruption"}, "fix", {"name": "supply_shortfall"})]

    pc, out = rebuild(entries, "event-types")

    saved = read_yaml(out / "event_types.yaml")
    extract_calls = [prompt for prompt, _ in pc.llm.calls if prompt.startswith("[task: extract]")]
    assert extract_calls == []
    assert saved["relabels"] == {"supply_disruption": "supply_shortfall"}
    assert "supply_disruption" in {item["name"] for item in saved["event_types"]}
    assert {event["type"] for event in read_jsonl(out / "docs" / "events.jsonl")} == {"supply_shortfall"}
    assert pc.decisions.unmatched() == []


def test_a_wrong_event_type_merges_into_other_and_a_new_type_is_added(rebuild):
    entries = [
        _decision("event_type", {"name": "supply_disruption"}, "wrong"),
        _decision("event_type", {"name": "tariff_change"}, "fix", {"description": "A tariff changes."}),
    ]

    pc, out = rebuild(entries, "event-types", "event-types")

    saved = read_yaml(out / "event_types.yaml")
    names = [item["name"] for item in saved["event_types"]]
    assert saved["relabels"] == {"supply_disruption": "other"}
    assert names[-2:] == ["tariff_change", "other"]
    assert pc.decisions.unmatched() == []


def test_merging_an_entity_moves_its_aliases_and_sources_and_redirects_references(rebuild, built):
    before = {item["id"]: item for item in read_yaml(artifacts_dir(built) / "entities.yaml")}
    entries = [_decision("entity", {"id": "crop:oats"}, "fix", {"merge_into": "crop:maize"})]

    pc, out = rebuild(entries, "catalog", "catalog")

    after = {item["id"]: item for item in read_yaml(out / "entities.yaml")}
    maize = after["crop:maize"]
    assert "crop:oats" not in after
    assert {"Oats", "Oats (spring)", "Corn"} <= set(maize["aliases"])
    assert maize["attributes"]["sources"] == before["crop:maize"]["attributes"]["sources"] + before["crop:oats"]["attributes"]["sources"]
    assert pc.decisions.unmatched() == []


def test_references_to_a_merged_entity_point_at_the_target(rebuild):
    entries = [_decision("entity", {"id": "cropgroup:cereals"}, "fix", {"merge_into": "cropgroup:oilseeds"})]

    _, out = rebuild(entries, "catalog", "catalog")

    after = {item["id"]: item for item in read_yaml(out / "entities.yaml")}
    assert "cropgroup:cereals" not in after
    assert after["crop:maize"]["attributes"]["parents"] == ["cropgroup:oilseeds"]


def test_entity_fixes_wrong_and_unmatched(rebuild):
    entries = [
        _decision("entity", {"id": "crop:rapeseed"}, "fix", {"canonical_name": "Canola", "add_aliases": ["Colza"]}),
        _decision("entity", {"id": "location:great_plains"}, "fix", {"part_of": "location:europe"}),
        _decision("entity", {"id": "series:oat_pb"}, "fix", {"parents": ["crop:maize"]}),
        _decision("entity", {"id": "location:north_america"}, "wrong"),
        _decision("entity", {"id": "crop:nope"}, "correct"),
    ]

    pc, out = rebuild(entries, "catalog", "catalog")

    after = {item["id"]: item for item in read_yaml(out / "entities.yaml")}
    assert (after["crop:rapeseed"]["canonical_name"], after["crop:rapeseed"]["source"]) == ("Canola", "human")
    assert {"Rapeseed", "Colza"} <= set(after["crop:rapeseed"]["aliases"])
    assert after["location:great_plains"]["attributes"]["part_of"] == "location:europe"
    assert after["series:oat_pb"]["attributes"]["parents"] == ["crop:maize"]
    assert "location:north_america" not in after
    assert [d.target["id"] for d in pc.decisions.unmatched()] == ["crop:nope"]


def test_rule_decisions_set_the_named_semantic_fields(rebuild, built):
    semantic = read_yaml(artifacts_dir(built) / "semantic.yaml")
    other = next(m["logical"] for m in semantic["tables"]["grain_quotes"]["measures"] if m["logical"] != "price_mid")
    table = {"table": "grain_quotes"}
    entries = [
        _decision("rule", {**table, "field": "primary_measure"}, "fix", {"value": other}),
        _decision("rule", {**table, "field": "time.grains"}, "fix", {"value": ["month", "year"]}),
        _decision("rule", {**table, "field": "denominations"}, "fix", {"value": ["currency"]}),
        _decision("rule", {**table, "field": "scenario.values"}, "fix", {"value": {"observed": "actual"}}),
        _decision("rule", {**table, "field": "synonyms.price_mid"}, "fix", {"value": ["cost"]}),
        _decision("rule", {**table, "field": "conversions"}, "fix", {"value": [
            {"column": "price_mid", "from": "USD", "to": "EUR", "factor": 0.9},
        ]}),
        _decision("rule", {"table": "nope", "field": "primary_measure"}, "fix", {"value": other}),
    ]

    pc, out = rebuild(entries, "semantic", "semantic")

    semantic = read_yaml(out / "semantic.yaml")
    spec = semantic["tables"]["grain_quotes"]
    assert spec["primary_measure"] == other
    assert spec["time"]["grains"] == ["month", "year"] and spec["denominations"] == ["currency"]
    assert spec["scenario"]["values"] == {"observed": "actual"} and spec["scenario"]["source"] == "human"
    assert semantic["synonyms"]["price_mid"] == ["cost"]
    assert semantic["conversions"] == [
        {"column": "price_mid", "from": "USD", "to": "EUR", "factor": 0.9, "source": "human"},
    ]
    assert len(semantic["human_rules"]) == 6
    assert [d.target["table"] for d in pc.decisions.unmatched()] == ["nope"]


def test_an_invalid_rule_value_is_not_applied(rebuild, built):
    before = read_yaml(artifacts_dir(built) / "semantic.yaml")["tables"]["grain_quotes"]["denominations"]
    entries = [_decision("rule", {"table": "grain_quotes", "field": "denominations"}, "fix", {"value": ["ghost"]})]

    pc, out = rebuild(entries, "semantic", "semantic")

    assert read_yaml(out / "semantic.yaml")["tables"]["grain_quotes"]["denominations"] == before
    assert len(pc.decisions.unmatched()) == 1


def test_a_wrong_move_drops_the_move_and_its_driver_and_a_kept_one_is_human(rebuild, built):
    assert _PLANTED_MOVE in {m["id"] for m in read_jsonl(artifacts_dir(built) / "structured" / "moves.jsonl")}
    assert _PLANTED_MOVE in {r["move_id"] for r in read_jsonl(artifacts_dir(built) / "links" / "candidate_drivers.jsonl")}
    entries = [
        _decision("move", {"id": _PLANTED_MOVE}, "wrong"),
        _decision("move", {"id": _OTHER_MOVE}, "correct"),
        _decision("move", {"id": "move:gone"}, "wrong"),
    ]

    pc, out = rebuild(entries, "moves")

    moves = {m["id"]: m for m in read_jsonl(out / "structured" / "moves.jsonl")}
    drivers = read_jsonl(out / "links" / "candidate_drivers.jsonl")
    assert _PLANTED_MOVE not in moves and moves[_OTHER_MOVE]["source"] == "human"
    assert drivers == []
    assert [d.target["id"] for d in pc.decisions.unmatched()] == ["move:gone"]
