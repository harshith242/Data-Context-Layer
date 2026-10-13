"""Metrics on a bare table: no frequency, denomination or scenario columns, INTEGER entity ids, ragged starts."""
import duckdb
import pytest

from ctxlayer.structured.compiler import MetricRequest, run_metric

# Entity 1 has monthly rows from January; entity 2 only starts in April.
_ROWS = [(1, f"2024-{m:02d}-01", 10.0 + m) for m in range(1, 7)] + [(2, f"2024-{m:02d}-01", 50.0 + m) for m in range(4, 7)]


@pytest.fixture
def bare():
    con = duckdb.connect()
    con.execute("CREATE TABLE t (code INTEGER, day DATE, amount DOUBLE)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?)", _ROWS)
    con.execute("CREATE VIEW sem_t AS SELECT code, day, amount FROM t")
    semantic = {
        "now": {"value": "2024-06-01"},
        "tables": {"t": {
            "view": "sem_t", "kind": "measures",
            "columns": [{"logical": "code", "role": "entity_id"}, {"logical": "day", "role": "time_business"}],
            "entity": {"id": "code"},
            "time": {"business": "day", "range": ["2024-01-01", "2024-06-01"]},
            "primary_measure": "amount", "measures": [{"logical": "amount"}],
        }},
    }
    yield con, semantic
    con.close()


def _run(bare, metric, ids, **kwargs):
    con, semantic = bare
    return run_metric(con, MetricRequest(metric=metric, entities=[str(i) for i in ids], **kwargs), semantic, ids)


def _records(result):
    return [dict(zip(result["columns"], row)) for row in result["rows"]]


def test_integer_ids_and_missing_optional_keys_still_answer(bare):
    latest = _run(bare, "latest_value", [1, 2])
    series = _run(bare, "series", [1], grain="month")

    assert {row["code"]: row["value"] for row in _records(latest)} == {1: 16.0, 2: 56.0}
    assert [row["value"] for row in _records(series)] == [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]


def test_change_pct_keeps_an_entity_that_starts_after_the_window_start(bare):
    result = _run(bare, "change_pct", [1, 2], start="2024-01-01", end="2024-06-01")
    rows = {row["code"]: row for row in _records(result)}

    # Entity 2 has no row on or before the start: its first row (April) is the start, and start_date says so.
    assert rows[2]["start_date"] == "2024-04-01"
    assert rows[2]["change_pct"] == pytest.approx((56.0 - 54.0) / 54.0 * 100)
    assert rows[1]["change_pct"] == pytest.approx((16.0 - 11.0) / 11.0 * 100)


def test_label_only_table_uses_the_label_column(bare):
    con, semantic = bare
    semantic["tables"]["t"]["entity"] = {"label": "code"}

    result = run_metric(con, MetricRequest(metric="latest_value", entities=["1"]), semantic, [1])

    assert _records(result)[0]["value"] == 16.0
