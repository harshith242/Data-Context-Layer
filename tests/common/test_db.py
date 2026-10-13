"""Tests for read-only DB access and the SELECT-only SQL guard."""
import datetime as dt
import math
from decimal import Decimal

import duckdb
import pytest

from ctxlayer.common.db import (
    SqlRejected,
    check_select,
    connect_readonly,
    list_tables,
    run_select,
    to_jsonable,
)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "t.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE sales (id INTEGER, day DATE, amount DECIMAL(10,2))")
    con.execute(
        "INSERT INTO sales VALUES (1, DATE '2024-01-05', 10.50),"
        " (2, DATE '2024-01-06', 20.25), (3, DATE '2024-01-07', 30.00)"
    )
    con.execute("CREATE VIEW big_sales AS SELECT * FROM sales WHERE amount > 15")
    con.execute("CREATE SCHEMA other")
    con.execute("CREATE TABLE other.hidden (x INTEGER)")
    con.close()
    return path


@pytest.fixture
def con(db_path):
    connection = connect_readonly(db_path)
    yield connection
    connection.close()




def test_connect_readonly_refuses_create_table(con):
    with pytest.raises(duckdb.Error):
        con.execute("CREATE TABLE x (a INTEGER)")


def test_run_select_blocks_reading_local_files(con, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("confidential")

    with pytest.raises(duckdb.Error):
        run_select(con, f"SELECT content FROM read_text('{secret}')", limit=1)


def test_list_tables_returns_tables_and_views_in_main_only(con):
    assert sorted(list_tables(con)) == ["big_sales", "sales"]




def test_check_select_accepts_with_select_and_strips_semicolon(con):
    sql = "WITH t AS (SELECT 1 AS a) SELECT a FROM t;"
    assert check_select(sql) == "WITH t AS (SELECT 1 AS a) SELECT a FROM t"


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO sales VALUES (9, DATE '2024-02-01', 1)",
        "SELECT 1; SELECT 2",
        "EXPLAIN SELECT 1",
    ],
)
def test_check_select_rejects_non_select(sql):
    with pytest.raises(SqlRejected):
        check_select(sql)




def test_run_select_truncates_at_limit(con):
    result = run_select(con, "SELECT id FROM sales ORDER BY id", limit=2)

    assert result.rows == [[1], [2]]
    assert result.row_count == 2
    assert result.truncated is True


def test_run_select_not_truncated_when_rows_equal_limit(con):
    result = run_select(con, "SELECT id FROM sales", limit=3)

    assert result.row_count == 3
    assert result.truncated is False






def test_to_jsonable_converts_values():
    assert to_jsonable(dt.date(2024, 1, 5)) == "2024-01-05"
    assert to_jsonable(dt.datetime(2024, 1, 5, 3, 4, 5)) == "2024-01-05T03:04:05"
    aware = dt.datetime(2024, 1, 5, 3, 4, 5, tzinfo=dt.timezone.utc)
    assert to_jsonable(aware) == "2024-01-05T03:04:05+00:00"
    assert to_jsonable(Decimal("1.25")) == 1.25
    assert to_jsonable(math.nan) is None
    assert to_jsonable(7) == 7
    assert to_jsonable("a") == "a"
    assert to_jsonable(None) is None


