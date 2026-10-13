"""Semantic layer: draft from roles/profile, generated views, SQL validation."""
import json

import duckdb
import pytest
from fakes import FakeLLM
from fixtures.domain_a.semantic_build import SYNONYMS_REPLY, build_semantic, scripted_llm

from ctxlayer.common.context_files import ContextFiles
from ctxlayer.common.db import connect_readonly
from ctxlayer.structured.semantic_draft import draft_semantic
from ctxlayer.structured.semantic_views import open_semantic, validate_sql

QUOTES = "grain_quotes"
BULLETINS = "farm_bulletins"


@pytest.fixture
def settings(tmp_settings, domain_a, tmp_path):
    return tmp_settings({
        "database": {"path": str(domain_a.db_path)},
        "paths": {"artifacts": str(tmp_path / "artifacts")},
    })


@pytest.fixture
def semantic(domain_a, domain_a_roles, settings):
    profile, roles = domain_a_roles
    return build_semantic(domain_a.db_path, profile, roles, settings)


@pytest.fixture
def source(domain_a):
    con = connect_readonly(domain_a.db_path)
    yield con
    con.close()


@pytest.fixture
def sem(settings, semantic):
    con = open_semantic(settings)
    yield con
    con.close()


def _draft(domain_a, domain_a_roles, llm, context=None):
    profile, roles = domain_a_roles
    con = connect_readonly(domain_a.db_path)
    try:
        return draft_semantic(roles, profile, con, llm, context or ContextFiles())
    finally:
        con.close()


def test_now_is_last_observed_date_not_outlook(semantic, source):
    observed = source.execute(
        f"SELECT CAST(max(quote_date) AS VARCHAR) FROM {QUOTES} WHERE basis = 'observed'").fetchone()[0]
    overall = source.execute(f"SELECT CAST(max(quote_date) AS VARCHAR) FROM {QUOTES}").fetchone()[0]

    assert semantic["now"] == {"value": observed, "source": "inferred"}
    assert observed < overall


def test_time_range_and_grains_cover_finest_frequency_and_coarser(semantic, source):
    low, high = source.execute(
        f"SELECT CAST(min(quote_date) AS VARCHAR), CAST(max(quote_date) AS VARCHAR) FROM {QUOTES}").fetchone()

    assert semantic["tables"][QUOTES]["time"] == {
        "business": "quote_date",
        "period_key": "period_ym",
        "range": [low, high],
        "grains": ["week", "month", "quarter", "year"],
    }


def test_measures_denominations_and_scenario_come_from_roles(semantic):
    table = semantic["tables"][QUOTES]

    assert table["kind"] == "measures"
    assert table["view"] == "sem_grain_quotes"
    assert table["entity"] == {"id": "series_code", "label": "series_title"}
    assert table["primary_measure"] == "price_mid"
    assert table["denominations"] == ["currency", "unit"]
    assert table["frequency"] == "cadence"
    assert table["scenario"] == {
        "column": "basis", "values": {"observed": "actual", "outlook": "projected"}, "default": "actual", "source": "inferred"}
    assert table["measures"] == [
        {"logical": "price_low", "band": "low", "additive": False, "aggregation": "avg"},
        {"logical": "price_mid", "band": "mid", "additive": False, "aggregation": "avg"},
        {"logical": "price_high", "band": "high", "additive": False, "aggregation": "avg"},
    ]


def test_text_table_is_mapped_without_measures(semantic):
    table = semantic["tables"][BULLETINS]

    assert table["kind"] == "text"
    assert table["view"] == "sem_farm_bulletins"
    assert "measures" not in table
    assert table["time"]["business"] == "released_at"


def test_view_row_counts_match_source(sem, source):
    for table, view in ((QUOTES, "sem_grain_quotes"), (BULLETINS, "sem_farm_bulletins")):
        expected = source.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert sem.execute(f"SELECT count(*) FROM {view}").fetchone()[0] == expected


def test_views_hide_audit_columns_and_use_logical_names(sem):
    quote_columns = [row[0] for row in sem.execute("DESCRIBE sem_grain_quotes").fetchall()]

    assert "loaded_at" not in quote_columns
    assert "loaded_by" not in quote_columns
    assert quote_columns[-1] == "scenario_label"
    assert {"price_mid", "quote_date", "currency"} <= set(quote_columns)


def test_scenario_label_maps_values_to_actual_and_projected(sem, source):
    expected = dict(source.execute(f"SELECT basis, count(*) FROM {QUOTES} GROUP BY basis").fetchall())
    got = dict(sem.execute("SELECT scenario_label, count(*) FROM sem_grain_quotes GROUP BY 1").fetchall())

    assert got == {"actual": expected["observed"], "projected": expected["outlook"]}


def test_open_semantic_cannot_read_files(sem, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")

    with pytest.raises(duckdb.Error):
        sem.execute(f"SELECT * FROM read_text('{secret}')")
    with pytest.raises(duckdb.Error):
        sem.execute("SET enable_external_access = true")


def test_open_semantic_cannot_write_source(sem, domain_a):
    with pytest.raises(duckdb.Error):
        sem.execute(f"DELETE FROM src.main.{QUOTES}")


def test_validate_sql_rejects_physical_table(semantic):
    result = validate_sql(f"SELECT * FROM {QUOTES}", semantic)

    assert result["ok"] is False
    assert result["tables"] == [QUOTES]
    assert any(QUOTES in error for error in result["errors"])


def test_validate_sql_rejects_delete(semantic):
    result = validate_sql("DELETE FROM sem_grain_quotes", semantic)

    assert result["ok"] is False
    assert result["errors"]


def test_validate_sql_ignores_cte_names_as_tables(semantic):
    result = validate_sql(
        "WITH q AS (SELECT currency, unit, price_mid FROM sem_grain_quotes WHERE scenario_label = 'actual') "
        "SELECT currency, unit, avg(price_mid) FROM q GROUP BY currency, unit", semantic)

    assert result["ok"] is True
    assert result["tables"] == ["sem_grain_quotes"]


def test_validate_sql_warns_on_cross_currency_avg(semantic):
    result = validate_sql("SELECT avg(price_mid) FROM sem_grain_quotes WHERE scenario_label = 'actual'", semantic)

    assert result["ok"] is True
    assert len(result["warnings"]) == 1
    assert "currency" in result["warnings"][0]
    assert "unit" in result["warnings"][0]


def test_validate_sql_accepts_filter_on_denominations(semantic):
    result = validate_sql(
        "SELECT avg(price_mid) FROM sem_grain_quotes "
        "WHERE currency = 'USD' AND unit = 'per_tonne' AND scenario_label = 'actual'", semantic)

    assert result["warnings"] == []


def test_validate_sql_warns_when_scenario_not_filtered(semantic):
    result = validate_sql(
        "SELECT currency, unit, avg(price_mid) FROM sem_grain_quotes GROUP BY currency, unit", semantic)

    assert result["ok"] is True
    assert len(result["warnings"]) == 1
    assert "scenario" in result["warnings"][0]


def test_validate_sql_group_by_all_counts_projected_columns(semantic):
    result = validate_sql(
        "SELECT currency, unit, avg(price_mid) FROM sem_grain_quotes "
        "WHERE scenario_label = 'actual' GROUP BY ALL", semantic)

    assert result["warnings"] == []


def test_conversions_empty_for_auto_and_llm_not_asked(domain_a, domain_a_roles):
    llm = scripted_llm()
    semantic = _draft(domain_a, domain_a_roles, llm)

    assert semantic["conversions"] == []
    assert not any(prompt.startswith("[task: conversions]") for prompt, _ in llm.calls)


def test_conversions_come_from_context_text(domain_a, domain_a_roles):
    reply = json.dumps([
        {"column": "unit", "from": "per_tonne", "to": "per_bushel", "factor": 0.0272},
        {"column": "unit", "from": "per_tonne", "to": "per_bushel", "factor": "n/a"},
        {"column": "no_such_column", "from": "a", "to": "b", "factor": 2},
    ])
    llm = FakeLLM({"grains": "{}", "synonyms": "{}", "conversions": reply})
    context = ContextFiles(database="One tonne of grain is 36.74 bushels.")

    semantic = _draft(domain_a, domain_a_roles, llm, context)

    assert semantic["conversions"] == [
        {"column": "unit", "from": "per_tonne", "to": "per_bushel", "factor": 0.0272, "source": "context.md"}]


def test_synonyms_keep_only_known_columns(semantic):
    assert semantic["synonyms"] == {
        "price_mid": ["mid price", "average price"],
        "series_title": ["series name"],
        "crop": ["commodity"],
    }
    assert "not_a_column" in SYNONYMS_REPLY


def test_malformed_llm_replies_allow_every_grain_and_no_synonyms(domain_a, domain_a_roles):
    reply = "not json at all"
    llm = FakeLLM({"grains": reply, "synonyms": reply, "conversions": reply})

    semantic = _draft(domain_a, domain_a_roles, llm)

    # Unknown native grain: every grain stays allowed so metrics still run.
    assert semantic["tables"][QUOTES]["time"]["grains"] == ["day", "week", "month", "quarter", "year"]
    assert semantic["synonyms"] == {}
    assert semantic["conversions"] == []


def test_validate_sql_rejects_qualified_table_hidden_behind_a_same_named_cte(semantic):
    result = validate_sql("WITH secret AS (SELECT * FROM src.main.secret) SELECT * FROM secret", semantic)

    assert result["ok"] is False
    assert any("src.main.secret" in error for error in result["errors"])
