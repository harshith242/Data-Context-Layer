"""Metric compiler + library on domain_a: every number is recomputed with pandas from the semantic view."""
from datetime import date

import pandas as pd
import pytest
from fixtures.domain_a.make_domain import PLANTED_MOVE_DATE, PLANTED_SERIES_CODE

from ctxlayer.structured.compiler import MetricRequest, compile_metric, resolve_window, run_metric
from ctxlayer.structured.semantic_views import open_semantic

WEEKLY = PLANTED_SERIES_CODE  # WHT_MILL_GP, USD per_tonne
MONTHLY = "MZE_RV"  # EUR per_tonne
ALL_SERIES = ["WHT_MILL_GP", "BRL_MALT_GP", "RPS_RV", "MZE_RV", "SOY_PB", "OAT_PB"]
NOW = date(2025, 6, 30)
VALID = (date(2024, 1, 1), date(2025, 12, 29))
VALID_RANGE = ["2024-01-01", "2025-12-29"]
DENOMINATION = ["currency", "unit"]


@pytest.fixture(scope="module")
def sem(domain_a_semantic):
    settings, semantic = domain_a_semantic
    con = open_semantic(settings)
    frame = con.execute("SELECT * FROM sem_grain_quotes").df()
    frame["quote_date"] = pd.to_datetime(frame["quote_date"])
    yield con, semantic, frame
    con.close()


def run(sem, metric, entities, **kwargs):
    con, semantic, _ = sem
    return run_metric(con, MetricRequest(metric=metric, entities=entities, **kwargs), semantic, entities)


def records(result):
    return [dict(zip(result["columns"], row)) for row in result["rows"]]


def actual(sem, code, start=None, end=None):
    """Actual rows of one series in date order, optionally within [start, end]."""
    frame = sem[2]
    rows = frame[(frame.series_code == code) & (frame.scenario_label == "actual")].sort_values("quote_date")
    if start:
        rows = rows[rows.quote_date >= start]
    if end:
        rows = rows[rows.quote_date <= end]
    return rows


def month_means(rows):
    return rows.groupby(rows.quote_date.dt.to_period("M")).price_mid.mean()


def test_latest_value_returns_last_actual_row_per_entity(sem):
    result = run(sem, "latest_value", [WEEKLY, MONTHLY])

    expected = {code: actual(sem, code).iloc[-1] for code in (WEEKLY, MONTHLY)}
    got = {row["series_code"]: row for row in records(result)}
    for code, row in expected.items():
        assert got[code]["value"] == pytest.approx(row.price_mid)
        assert got[code]["as_of"] == row.quote_date.date().isoformat()
        assert got[code]["scenario"] == "actual"
    assert result["as_of"] == "2025-06-30"
    assert got[MONTHLY]["as_of"] == "2025-06-01"


def test_value_at_takes_latest_row_on_or_before_the_date(sem):
    # 2025-05-07 falls between weekly quotes: the 2025-05-05 row applies.
    result = run(sem, "value_at", [WEEKLY], end="2025-05-07")

    row = actual(sem, WEEKLY, end="2025-05-07").iloc[-1]
    assert row.quote_date == pd.Timestamp(PLANTED_MOVE_DATE)
    assert records(result)[0]["value"] == pytest.approx(row.price_mid)
    assert records(result)[0]["as_of"] == "2025-05-05"


def test_series_averages_values_per_month(sem):
    result = run(sem, "series", [WEEKLY], start="2025-01-01", end="2025-06-30", grain="month")

    rows = actual(sem, WEEKLY, "2025-01-01", "2025-06-30")
    expected = month_means(rows)
    assert [r["period"] for r in records(result)] == [p.start_time.date().isoformat() for p in expected.index]
    assert [r["value"] for r in records(result)] == pytest.approx(list(expected.values))
    assert sum(r["obs"] for r in records(result)) == len(rows)


def test_change_abs_and_pct_across_the_planted_move(sem):
    before = actual(sem, WEEKLY, end="2025-04-28").iloc[-1].price_mid
    after = actual(sem, WEEKLY, end="2025-05-05").iloc[-1].price_mid

    absolute = records(run(sem, "change_abs", [WEEKLY], start="2025-04-28", end="2025-05-05"))[0]
    percent = records(run(sem, "change_pct", [WEEKLY], start="2025-04-28", end="2025-05-05"))[0]

    assert absolute["change_abs"] == pytest.approx(after - before)
    assert percent["change_pct"] == pytest.approx((after - before) / before * 100)
    assert percent["change_pct"] > 10


def test_period_avg_flags_best_and_worst_quarter(sem):
    result = run(sem, "period_avg", [WEEKLY], start="2024-01-01", end="2025-06-30", grain="quarter")

    rows = actual(sem, WEEKLY, "2024-01-01", "2025-06-30")
    expected = rows.groupby(rows.quote_date.dt.to_period("Q")).price_mid.mean()
    got = records(result)
    assert [r["value"] for r in got] == pytest.approx(list(expected.values))
    flags = {r["period"]: r["flag"] for r in got}
    assert flags[expected.idxmax().start_time.date().isoformat()] == "best"
    assert flags[expected.idxmin().start_time.date().isoformat()] == "worst"
    assert sum(flag is not None for flag in flags.values()) == 2


def test_streak_finds_longest_and_current_monthly_runs(sem):
    result = run(sem, "streak", [MONTHLY], grain="month")

    diffs = month_means(actual(sem, MONTHLY)).diff().dropna()
    signs = [1 if d > 0 else -1 if d < 0 else 0 for d in diffs]
    longest = {1: 0, -1: 0}
    run_length = 0
    for index, sign in enumerate(signs):
        run_length = run_length + 1 if index and signs[index - 1] == sign else 1
        if sign:
            longest[sign] = max(longest[sign], run_length)

    got = records(result)[0]
    assert got["longest_increase"] == longest[1]
    assert got["longest_decrease"] == longest[-1]
    assert got["current_length"] == run_length
    assert got["current_direction"] == {1: "increase", -1: "decrease", 0: "flat"}[signs[-1]]


def test_volatility_is_stddev_of_monthly_pct_changes(sem):
    result = run(sem, "volatility", [MONTHLY], grain="month")

    changes = month_means(actual(sem, MONTHLY)).pct_change().dropna() * 100
    got = records(result)[0]
    assert got["volatility_pct"] == pytest.approx(changes.std())
    assert got["changes"] == len(changes)


def test_spread_across_dimension_is_computed_per_denomination_group(sem):
    result = run(sem, "spread", ALL_SERIES, dimension="crop")

    latest = pd.DataFrame([actual(sem, code).iloc[-1] for code in ALL_SERIES])
    got = {(r["currency"], r["unit"]): r for r in records(result)}
    for key, group in latest.groupby(DENOMINATION):
        assert got[key]["spread"] == pytest.approx(group.price_mid.max() - group.price_mid.min())
        assert got[key]["max_at"] == group.loc[group.price_mid.idxmax(), "crop"]
        assert got[key]["dimension_values"] == group.crop.nunique()
    assert len(got) == 4


def test_rank_groups_by_denomination_and_warns(sem):
    result = run(sem, "rank", ALL_SERIES, n=1)

    latest = pd.DataFrame([actual(sem, code).iloc[-1] for code in ALL_SERIES])
    expected = {key: group.loc[group.price_mid.idxmax(), "series_code"] for key, group in latest.groupby(DENOMINATION)}
    got = {(r["currency"], r["unit"]): r["series_code"] for r in records(result)}
    assert got == expected
    assert len(result["denominations"]) == 4
    assert "denomination" in result["warnings"][0]


def test_cross_currency_latest_value_is_not_blended(sem):
    mixed = run(sem, "latest_value", [WEEKLY, "RPS_RV"])
    single = run(sem, "latest_value", [WEEKLY, "BRL_MALT_GP"])

    assert {(r["series_code"], r["currency"]) for r in records(mixed)} == {(WEEKLY, "USD"), ("RPS_RV", "EUR")}
    assert len(mixed["warnings"]) == 1
    assert single["warnings"] == []


def test_projection_path_returns_only_outlook_rows_labelled_projected(sem):
    result = run(sem, "projection_path", [MONTHLY], n=3, grain="month")

    frame = sem[2]
    outlook = frame[(frame.series_code == MONTHLY) & (frame.basis == "outlook")].sort_values("quote_date")
    got = records(result)
    assert [r["value"] for r in got] == pytest.approx(list(outlook.price_mid.iloc[:3]))
    assert {r["scenario"] for r in got} == {"projected"}
    assert min(r["period"] for r in got) > "2025-06-30"


def test_compare_runs_inner_metric_for_each_entity(sem):
    window = {"start": "2025-01-01", "end": "2025-06-30"}
    result = run(sem, "compare", [WEEKLY, MONTHLY], inner_metric="change_pct", **window)

    got = {r["series_code"]: r["change_pct"] for r in records(result)}
    for code in (WEEKLY, MONTHLY):
        alone = records(run(sem, "change_pct", [code], **window))[0]["change_pct"]
        assert got[code] == pytest.approx(alone)


def test_actual_metrics_never_include_outlook_rows(sem):
    result = run(sem, "series", [WEEKLY], end="2025-12-29", grain="month")

    got = records(result)
    assert max(r["period"] for r in got) == "2025-06-01"
    assert sum(r["obs"] for r in got) == len(actual(sem, WEEKLY))
    assert result["scenario"] == "actual"


def test_last_6_months_is_anchored_on_data_now_not_wall_clock(sem):
    result = run(sem, "series", [WEEKLY], start="last_6_months", grain="month")

    assert result["definition"]["window"] == ["2024-12-30", "2025-06-30"]
    assert sum(r["obs"] for r in records(result)) == len(actual(sem, WEEKLY, "2024-12-30", "2025-06-30"))


@pytest.mark.parametrize("start, end, expected", [
    ("2025-03", None, ("2025-03-01", "2025-06-30")),
    ("ytd", None, ("2025-01-01", "2025-06-30")),
    ("last_2_weeks", None, ("2025-06-16", "2025-06-30")),
])
def test_resolve_window_formats(start, end, expected):
    got = resolve_window(start, end, NOW, VALID)

    assert tuple(day.isoformat() for day in got) == expected


@pytest.mark.parametrize("start, end", [("2023-01-01", None), (None, "2026-03-01")])
def test_window_outside_data_returns_valid_range(sem, start, end):
    result = run(sem, "series", [WEEKLY], start=start, end=end)

    assert result["error"]["type"] == "out_of_range"
    assert result["error"]["valid_range"] == VALID_RANGE


def test_malformed_date_and_inverted_window_are_errors(sem):
    assert run(sem, "series", [WEEKLY], start="yesterday")["error"]["type"] == "invalid_date"
    assert run(sem, "series", [WEEKLY], start="2025-05", end="2025-02")["error"]["type"] == "invalid_window"


def test_convert_to_applies_stated_factor_and_merges_denominations(sem):
    con, semantic, _ = sem
    conversion = {"column": "currency", "from": "USD", "to": "EUR", "factor": 0.9}
    request = MetricRequest(metric="latest_value", entities=[WEEKLY, "RPS_RV"], convert_to="EUR")

    result = run_metric(con, request, {**semantic, "conversions": [conversion]}, request.entities)

    got = {r["series_code"]: r for r in records(result)}
    assert got[WEEKLY]["value"] == pytest.approx(actual(sem, WEEKLY).iloc[-1].price_mid * 0.9)
    assert {r["currency"] for r in got.values()} == {"EUR"}
    assert result["conversions_applied"] == [conversion]
    assert result["warnings"] == []


def test_wrong_scenario_gives_no_data_with_available_range(sem):
    result = run(sem, "latest_value", [WEEKLY], scenario="projected")

    assert result["error"]["type"] == "no_data"
    assert result["error"]["available_range"] == ["2025-07-07", "2025-12-29"]


def test_entity_ids_are_escaped_as_literals(sem):
    hostile = "x'); DROP VIEW sem_grain_quotes; --"
    result = run(sem, "latest_value", [hostile])
    _, semantic, _ = sem
    sql, _ = compile_metric(MetricRequest(metric="latest_value", entities=[hostile]), semantic, [hostile])

    assert result["error"]["type"] == "no_data"
    assert "'x''); DROP VIEW sem_grain_quotes; --'" in sql
    assert run(sem, "latest_value", [WEEKLY])["rows"]
