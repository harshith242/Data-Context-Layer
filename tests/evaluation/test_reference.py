"""Reference re-execution, anchoring and the cross-check between baselines."""
import datetime as dt

import pytest
import yaml

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.db import connect_readonly
from ctxlayer.evaluation.questions import Contestant, Question, load_questions
from ctxlayer.evaluation.reference import (
    RefResult,
    build_references,
    execute_reference,
    reanchor_sql,
    resolve_anchor,
    results_agree,
)

_ANCHOR = dt.date(2024, 6, 30)
_SPARK = "spark"
_LIMIT = 100
_TOL = 0.005
_RAPESEED_DUCKDB = (
    "SELECT strftime(quote_date, '%Y-%m') AS month, ROUND(AVG(price_mid), 2) AS avg_mid FROM grain_quotes "
    "WHERE crop = 'Rapeseed' AND basis = 'observed' AND quote_date >= DATE '2025-01-01' GROUP BY 1 ORDER BY 1"
)
# Spark dialect: DATE_FORMAT with a Java pattern and ADD_MONTHS do not exist in DuckDB.
_RAPESEED_SPARK = (
    "SELECT DATE_FORMAT(quote_date, 'yyyy-MM') AS month, ROUND(AVG(price_mid), 2) AS avg_mid FROM grain_quotes "
    "WHERE crop = 'Rapeseed' AND basis = 'observed' AND quote_date >= ADD_MONTHS(DATE'2025-01-01', 0) "
    "GROUP BY 1 ORDER BY 1"
)
_OATS_COUNT = "SELECT COUNT(*) AS n FROM grain_quotes WHERE crop = 'Oats' AND quote_date <= CURRENT_DATE"
_TWO_BASELINES = {"eval": {"baselines": [{"name": "duck"}, {"name": "spark", "dialect": _SPARK}]}}


@pytest.fixture
def con(domain_a):
    connection = connect_readonly(domain_a.db_path)
    yield connection
    connection.close()


def _ref(rows, error=None) -> RefResult:
    return RefResult(sql="", columns=["x"], rows=rows, error=error, reanchored=False, transpiled=False)


def _questions(domain_a, settings) -> list[Question]:
    return load_questions(domain_a.questions_csv, "question", settings.get("eval.baselines"))


def _two_sided(qid: str, duck_sql: str | None, spark_sql: str | None) -> Question:
    return Question(
        qid=qid, text=qid,
        baselines={
            "duck": Contestant(sql=duck_sql, answer="", time_ms=None),
            "spark": Contestant(sql=spark_sql, answer="", time_ms=None),
        },
    )


# ---------------------------------------------------------------- reanchor_sql

@pytest.mark.parametrize("text, expected", [
    ("SELECT current_date", "SELECT DATE '2024-06-30'"),
    ("SELECT today()", "SELECT DATE '2024-06-30'"),
    ("SELECT NOW( )", "SELECT TIMESTAMP '2024-06-30 00:00:00'"),
])
def test_reanchor_replaces_current_date_and_time_functions(text, expected):
    assert reanchor_sql(text, _ANCHOR) == (expected, True)


def test_reanchor_replaces_every_occurrence():
    sql = "SELECT * FROM t WHERE a <= CURRENT_DATE AND b >= CURRENT_DATE - INTERVAL 7 DAY"

    new_sql, changed = reanchor_sql(sql, _ANCHOR)

    assert new_sql == (
        "SELECT * FROM t WHERE a <= DATE '2024-06-30' AND b >= DATE '2024-06-30' - INTERVAL 7 DAY"
    )
    assert changed


def test_reanchor_leaves_identifiers_that_only_contain_the_names():
    sql = "SELECT now, current_date_col, my_today() FROM t"

    assert reanchor_sql(sql, _ANCHOR) == (sql, False)


# ----------------------------------------------------------- execute_reference

def test_execute_reference_reanchors_current_date_question(con, domain_a, tmp_settings):
    sql = _questions(domain_a, tmp_settings())[0].baselines["reference"].sql

    result = execute_reference(con, sql, _ANCHOR, _LIMIT, None)

    # Oats is monthly from 2024-01-01: six quotes up to 2024-06-30.
    assert result.error is None
    assert result.rows == [[6]]
    assert result.columns == ["n"]
    assert result.reanchored
    assert not result.transpiled
    assert not result.uses_wall_clock


def test_execute_reference_transpiles_spark_sql_when_fallback_is_set(con):
    result = execute_reference(con, _RAPESEED_SPARK, None, _LIMIT, _SPARK)

    # Monthly average observed rapeseed mid price in 2025, computed independently (SQL rounds to 2 places).
    frame = con.execute(
        "SELECT quote_date, price_mid FROM grain_quotes"
        " WHERE crop = 'Rapeseed' AND basis = 'observed' AND quote_date >= DATE '2025-01-01'"
    ).df()
    frame["month"] = frame["quote_date"].dt.strftime("%Y-%m")
    expected = frame.groupby("month")["price_mid"].mean()
    assert result.error is None
    assert result.transpiled
    assert result.columns == ["month", "avg_mid"]
    assert [row[0] for row in result.rows] == list(expected.index)
    assert [row[1] for row in result.rows] == pytest.approx(list(expected.values), abs=0.006)


def test_execute_reference_rejects_non_select_without_touching_data(con):
    before = con.execute("SELECT COUNT(*) FROM grain_quotes").fetchone()

    result = execute_reference(con, "DELETE FROM grain_quotes", None, _LIMIT, None)

    assert "only SELECT" in result.error
    assert con.execute("SELECT COUNT(*) FROM grain_quotes").fetchone() == before


# --------------------------------------------------------------- results_agree

def test_results_agree_ignores_row_order_and_float_noise_within_tolerance():
    a = _ref([[1.0, "x"], [200.0, "y"]])
    b = _ref([[200.4, "z"], [1.0, "w"]])

    assert results_agree(a, b, _TOL) is True


def test_results_agree_is_false_when_a_value_differs_beyond_tolerance():
    assert results_agree(_ref([[100.0]]), _ref([[101.0]]), _TOL) is False


def test_results_agree_is_none_when_either_side_is_missing_or_errored():
    ok = _ref([[1.0]])

    assert results_agree(ok, _ref([], error="boom"), _TOL) is None
    assert results_agree(_ref([], error="boom"), ok, _TOL) is None
    assert results_agree(None, ok, _TOL) is None
    assert results_agree(ok, None, _TOL) is None


# -------------------------------------------------------------- resolve_anchor

def test_resolve_anchor_prefers_the_setting_over_semantic_yaml(tmp_settings):
    settings = tmp_settings({"eval": {"now_anchor": "2025-01-31"}})
    folder = artifacts_dir(settings)
    folder.mkdir(parents=True)
    (folder / "semantic.yaml").write_text(yaml.safe_dump({"now": {"value": "2025-06-30"}}))

    assert resolve_anchor(settings) == dt.date(2025, 1, 31)


@pytest.mark.parametrize("content", ['now: {value: "2025-06-30"}\n', "now:\n  value: 2025-06-30\n"])
def test_resolve_anchor_reads_semantic_yaml_now_value_quoted_or_as_yaml_date(tmp_settings, content):
    settings = tmp_settings()
    folder = artifacts_dir(settings)
    folder.mkdir(parents=True)
    (folder / "semantic.yaml").write_text(content)

    assert resolve_anchor(settings) == dt.date(2025, 6, 30)


@pytest.mark.parametrize("content", [None, "tables: {}\n"])
def test_resolve_anchor_is_none_without_a_usable_source(tmp_settings, content):
    settings = tmp_settings()
    if content is not None:
        folder = artifacts_dir(settings)
        folder.mkdir(parents=True)
        (folder / "semantic.yaml").write_text(content)

    assert resolve_anchor(settings) is None


# ------------------------------------------------------------ build_references


def test_build_references_runs_the_single_baseline_and_has_nothing_to_cross_check(con, domain_a, tmp_settings):
    settings = tmp_settings()
    refs = build_references(_questions(domain_a, settings), con, _ANCHOR, settings)

    # q001 uses CURRENT_DATE, re-anchored to 2024-06-30; q004 has no SQL.
    assert refs["q001"]["baselines"]["reference"]["rows"] == [[6]]
    assert refs["q001"]["baselines"]["reference"]["reanchored"] is True
    assert refs["q004"]["baselines"] == {"reference": None}
    assert refs["q001"]["agree"] is None


def test_build_references_without_baselines_has_no_entries(con, tmp_settings):
    settings = tmp_settings({"eval": {"baselines": []}})

    refs = build_references([Question(qid="q001", text="x", baselines={})], con, _ANCHOR, settings)

    assert refs == {"q001": {"baselines": {}, "agree": None}}


def test_build_references_cross_checks_baselines_and_transpiles_by_dialect(con, tmp_settings):
    settings = tmp_settings(_TWO_BASELINES)
    questions = [_two_sided("q001", _OATS_COUNT, _OATS_COUNT), _two_sided("q002", _RAPESEED_DUCKDB, _RAPESEED_SPARK)]

    refs = build_references(questions, con, _ANCHOR, settings)

    assert refs["q001"]["baselines"]["duck"]["rows"] == [[6]]
    assert refs["q001"]["agree"] is True

    # q002: the second baseline's Spark SQL only runs via transpile.
    assert refs["q002"]["baselines"]["duck"]["transpiled"] is False
    assert refs["q002"]["baselines"]["spark"]["transpiled"] is True
    assert refs["q002"]["agree"] is True


def test_build_references_reports_disagreement_between_baselines(con, tmp_settings):
    settings = tmp_settings(_TWO_BASELINES)

    refs = build_references([_two_sided("q001", "SELECT 1 AS n", "SELECT 2 AS n")], con, _ANCHOR, settings)

    assert refs["q001"]["agree"] is False


def test_build_references_continues_after_a_failing_query(con, domain_a, tmp_settings):
    settings = tmp_settings(_TWO_BASELINES)
    broken = _two_sided("q100", "SELECT nope FROM missing_table", "SELECT 1 AS one")

    refs = build_references([broken, _two_sided("q101", _OATS_COUNT, None)], con, _ANCHOR, settings)

    assert "missing_table" in refs["q100"]["baselines"]["duck"]["error"]
    assert refs["q100"]["baselines"]["spark"]["rows"] == [[1]]
    assert refs["q100"]["agree"] is None
    assert refs["q101"]["baselines"]["duck"]["rows"] == [[6]]
    assert refs["q101"]["baselines"]["spark"] is None


def test_build_references_keeps_more_rows_than_the_agent_limit(con, tmp_settings):
    settings = tmp_settings({"agent": {"sql_row_limit": 2}, "eval": {"reference_row_limit": 1000}})
    refs = build_references([_two_sided("q001", _RAPESEED_DUCKDB, None)], con, None, settings)

    assert len(refs["q001"]["baselines"]["duck"]["rows"]) > 2
