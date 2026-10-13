"""Tests for the entity catalog, seeded from the synthetic domain_a database with a scripted LLM."""
import json

import duckdb
import pytest

from ctxlayer.common.catalog import Entity, load_catalog, parse_aliases, save_catalog, seed_catalog
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.llm import LLMError
from ctxlayer.common.ontology import draft_ontology
from ctxlayer.common.profile import profile_database
from ctxlayer.common.context_files import ContextFiles
from fakes import FakeLLM
from fixtures.domain_a.catalog_build import ALIASES_REPLY, ONTOLOGY_REPLY, PARENTS_REPLY, build_catalog

_QUOTES = "grain_quotes"
_MALFORMED = ["not json", '{"aliases": {"crop:oats": [1]}}']


@pytest.fixture(scope="module")
def built(domain_a, domain_a_roles, project_settings):
    profile, roles = domain_a_roles
    ontology, entities = build_catalog(domain_a.db_path, profile, roles, project_settings)
    return {entity.id: entity for entity in entities}


@pytest.fixture(scope="module")
def source_con(domain_a):
    con = connect_readonly(domain_a.db_path)
    yield con
    con.close()


def _build(domain_a, domain_a_roles, project_settings, handlers: dict) -> dict[str, Entity]:
    profile, roles = domain_a_roles
    full = {
        "ontology": json.dumps(ONTOLOGY_REPLY),
        "aliases": json.dumps(ALIASES_REPLY),
        "location_parents": json.dumps(PARENTS_REPLY),
        **handlers,
    }
    llm = FakeLLM(full)
    ontology = draft_ontology(roles, profile, llm, ContextFiles())
    con = connect_readonly(domain_a.db_path)
    try:
        return {e.id: e for e in seed_catalog(con, roles, ontology, llm, project_settings)}
    finally:
        con.close()


def _seed_small(tmp_path, settings, select_sql: str, handlers: dict) -> dict[str, Entity]:
    """Catalog of a one-table database `t(item hierarchy_level, price measure)`."""
    path = tmp_path / "small.duckdb"
    con = duckdb.connect(str(path))
    con.execute(f"CREATE TABLE t AS {select_sql}")
    con.close()

    con = connect_readonly(path)
    try:
        profile = profile_database(con, settings)
        roles = {"tables": {"t": {"kind": "measures", "columns": {
            "item": {"role": "hierarchy_level"}, "price": {"role": "measure"},
        }}}}
        llm = FakeLLM({"ontology": "{}", **handlers})
        ontology = draft_ontology(roles, profile, llm, ContextFiles())
        return {e.id: e for e in seed_catalog(con, roles, ontology, llm, settings)}
    finally:
        con.close()


def _count(con, sql: str) -> int:
    return con.execute(sql).fetchone()[0]


def _of_type(built, type_name: str) -> list[Entity]:
    return [entity for entity in built.values() if entity.type == type_name]


# ------------------------------------------------------------ parse_aliases



def test_parse_aliases_takes_uppercase_code_tokens():
    assert parse_aliases("Diet Cola (12 OZ) MX") == ["12 OZ", "Diet Cola", "12", "OZ", "MX"]




# ------------------------------------------------------------------- seeding



def test_entity_counts_per_type_match_distinct_values(built, source_con):
    crops = _count(source_con, f"SELECT COUNT(DISTINCT crop) FROM {_QUOTES}")
    groups = _count(source_con, f"SELECT COUNT(DISTINCT crop_group) FROM {_QUOTES}")
    regions = _count(source_con, f"SELECT COUNT(DISTINCT market_region) FROM {_QUOTES}")

    assert (len(_of_type(built, "Crop")), len(_of_type(built, "CropGroup"))) == (crops, groups)
    # Seeded regions plus the two parent regions proposed by the LLM.
    assert len(_of_type(built, "Location")) == regions + 2
    assert len(built) == 6 + crops + groups + regions + 2


def test_series_label_is_the_name_and_the_code_is_an_alias(built, source_con):
    rows = source_con.execute(f"SELECT DISTINCT series_code, series_title FROM {_QUOTES}").fetchall()

    for code, title in rows:
        match = [e for e in _of_type(built, "Series") if e.canonical_name == title]
        assert len(match) == 1
        assert code in match[0].aliases
        assert match[0].source == "metadata"
        assert match[0].confidence == 1.0
        assert match[0].needs_review is False

    assert built["series:wht_mill_gp"].aliases == ["WHT_MILL_GP"]


def test_parenthesised_qualifier_in_a_value_becomes_an_alias(tmp_path, project_settings):
    entities = _seed_small(
        tmp_path, project_settings, "SELECT * FROM (VALUES ('Wheat (milling)', 1.0), ('Corn', 2.0)) v(item, price)",
        {"aliases": "{}"},
    )

    assert entities["item:wheat_milling"].aliases == ["milling", "Wheat"]
    assert entities["item:corn"].aliases == []


def test_parents_follow_the_fd_direction(built, source_con):
    expected = dict(source_con.execute(f"SELECT crop, ANY_VALUE(crop_group) FROM {_QUOTES} GROUP BY crop").fetchall())

    for crop, group in expected.items():
        entity = next(e for e in _of_type(built, "Crop") if e.canonical_name == crop)
        group_entity = next(e for e in _of_type(built, "CropGroup") if e.canonical_name == group)
        assert entity.attributes["parents"] == [group_entity.id]
        assert "parents" not in group_entity.attributes


def test_locations_and_denominations_are_those_seen_with_each_crop(built, source_con):
    rows = source_con.execute(
        f"SELECT crop, LIST(DISTINCT market_region ORDER BY 1), LIST(DISTINCT currency ORDER BY 1), "
        f"LIST(DISTINCT unit ORDER BY 1) FROM {_QUOTES} GROUP BY crop"
    ).fetchall()

    for crop, regions, currencies, units in rows:
        entity = next(e for e in _of_type(built, "Crop") if e.canonical_name == crop)
        assert [built[i].canonical_name for i in entity.attributes["locations"]] == regions
        assert entity.attributes["currency"] == currencies
        assert entity.attributes["unit"] == units


def test_seeded_entities_carry_their_source_table_column_and_raw_value(built, source_con):
    # Series are keyed by code (two codes sharing a title stay apart); other types by their value.
    expected = {
        "series": ("series_code", [r[0] for r in source_con.execute(f"SELECT DISTINCT series_code FROM {_QUOTES}").fetchall()]),
        "crop": ("crop", [r[0] for r in source_con.execute(f"SELECT DISTINCT crop FROM {_QUOTES}").fetchall()]),
        "cropgroup": ("crop_group", [r[0] for r in source_con.execute(f"SELECT DISTINCT crop_group FROM {_QUOTES}").fetchall()]),
        "location": ("market_region", [r[0] for r in source_con.execute(f"SELECT DISTINCT market_region FROM {_QUOTES}").fetchall()]),
    }

    seeded = [entity for entity in built.values() if entity.source == "metadata"]
    assert len(seeded) == 6 + 6 + 2 + 3
    for entity in seeded:
        column, values = expected[entity.id.split(":")[0]]
        found = [s for s in entity.attributes["sources"] if s["table"] == _QUOTES]
        assert len(found) == 1
        assert found[0]["column"] == column
        assert found[0]["value"] in values


def test_two_codes_sharing_a_label_stay_two_entities(tmp_path, project_settings):
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    con.execute("CREATE TABLE t (code VARCHAR, label VARCHAR)")
    con.execute("INSERT INTO t VALUES ('S1', 'Same Name'), ('S2', 'Same Name')")
    ontology = {"types": [{"name": "Item", "sources": [{"table": "t", "columns": ["code", "label"]}]}], "relations": []}
    roles = {"tables": {"t": {"columns": {}}}}

    entities = seed_catalog(con, roles, ontology, FakeLLM({"aliases": "{}", "location_parents": "{}"}), project_settings)

    assert sorted(e.id for e in entities) == ["item:s1", "item:s2"]
    assert {e.canonical_name for e in entities} == {"Same Name"}




# ------------------------------------------------------------------ aliases

def test_llm_aliases_cover_the_bulletin_tag_spellings(built):
    assert "Wheat (milling)" in built["crop:milling_wheat"].aliases
    assert "Corn" in built["crop:maize"].aliases
    assert built["location:rhine_valley"].aliases == ["Rhine Vly"]
    assert built["crop:milling_wheat"].attributes["inferred_aliases"] == ["Wheat (milling)"]
    assert "inferred_aliases" not in built["series:wht_mill_gp"].attributes


def test_alias_equal_to_the_name_is_not_added(domain_a, domain_a_roles, project_settings):
    reply = {"aliases": {"crop:oats": ["oats", "Oats", "Oats (spring)"]}}
    built = _build(domain_a, domain_a_roles, project_settings, {"aliases": json.dumps(reply)})

    assert built["crop:oats"].aliases == ["Oats (spring)"]


def test_aliases_are_requested_in_batches_of_fifty(tmp_path, project_settings):
    lines_per_call = []

    def reply(prompt, tools):
        lines_per_call.append(sum(1 for line in prompt.splitlines() if line.startswith("item:")))
        return "{}"

    entities = _seed_small(
        tmp_path, project_settings, "SELECT 'item ' || i AS item, i * 1.5 AS price FROM range(120) r(i)",
        {"aliases": reply},
    )

    assert len(entities) == 120
    assert lines_per_call == [50, 50, 20]


@pytest.mark.parametrize("bad", _MALFORMED)
def test_malformed_alias_reply_leaves_seed_entities_intact(domain_a, domain_a_roles, project_settings, built, bad):
    result = _build(domain_a, domain_a_roles, project_settings, {"aliases": bad})

    assert set(result) == set(built)
    assert result["crop:milling_wheat"].aliases == []
    assert result["series:wht_mill_gp"].aliases == ["WHT_MILL_GP"]
    assert result["crop:milling_wheat"].attributes["parents"] == ["cropgroup:cereals"]


def test_failing_llm_leaves_seed_entities_intact(domain_a, domain_a_roles, project_settings, built):
    def fail(prompt, tools):
        raise LLMError("llm down")

    result = _build(domain_a, domain_a_roles, project_settings, {"aliases": fail, "location_parents": fail})

    seeded = {i for i, e in built.items() if e.source == "metadata"}
    assert set(result) == seeded
    assert result["location:great_plains"].aliases == []


# ---------------------------------------------------------- location parents

def test_llm_region_parent_becomes_a_location_with_part_of(built):
    parent = built["location:north_america"]

    assert (parent.type, parent.canonical_name, parent.source, parent.confidence) == (
        "Location", "North America", "inferred", 0.9,
    )
    assert parent.needs_review is False
    assert built["location:great_plains"].attributes["part_of"] == "location:north_america"
    assert built["location:prairie_basin"].attributes["part_of"] == "location:north_america"
    assert built["location:rhine_valley"].attributes["part_of"] == "location:europe"


def test_parent_below_the_threshold_is_flagged_for_review(domain_a, domain_a_roles, project_settings):
    reply = {"parents": {"location:rhine_valley": {"name": "Europe", "confidence": 0.5}}}
    built = _build(domain_a, domain_a_roles, project_settings, {"location_parents": json.dumps(reply)})

    assert built["location:europe"].needs_review is True
    assert built["location:europe"].confidence == 0.5


def test_parent_naming_an_existing_location_adds_no_entity(domain_a, domain_a_roles, project_settings, built):
    reply = {"parents": {"location:great_plains": {"name": "prairie basin", "confidence": 0.9}}}
    result = _build(domain_a, domain_a_roles, project_settings, {"location_parents": json.dumps(reply)})

    assert result["location:great_plains"].attributes["part_of"] == "location:prairie_basin"
    assert len(_of_type(result, "Location")) == 3


@pytest.mark.parametrize("bad", ["not json", '{"parents": {"location:rhine_valley": {"name": 1}}}'])
def test_malformed_parent_reply_adds_no_locations(domain_a, domain_a_roles, project_settings, bad):
    result = _build(domain_a, domain_a_roles, project_settings, {"location_parents": bad})

    assert len(_of_type(result, "Location")) == 3
    assert "part_of" not in result["location:rhine_valley"].attributes


# -------------------------------------------------------------- persistence

def test_catalog_round_trips_through_yaml(built, tmp_path):
    entities = list(built.values())
    path = tmp_path / "artifacts" / "entities.yaml"

    save_catalog(entities, path)

    assert load_catalog(path) == entities


def test_second_source_in_a_table_adds_to_the_first_instead_of_replacing_it(monkeypatch):
    from types import SimpleNamespace

    from ctxlayer.common import catalog

    monkeypatch.setattr(catalog, "_distinct", lambda con, table, columns: [("k", "v")])
    monkeypatch.setattr(catalog, "_identify", lambda source, values: values)
    monkeypatch.setattr(catalog, "_own_id", lambda source, found: "e1")
    entities = {"e1": SimpleNamespace(attributes={})}
    source = SimpleNamespace(table="t", columns=["id"])

    catalog._attach(None, source, ["region"], lambda row: "region:a", "locations", entities)
    catalog._attach(None, source, ["country"], lambda row: "country:b", "locations", entities)

    assert entities["e1"].attributes["locations"] == ["country:b", "region:a"]
