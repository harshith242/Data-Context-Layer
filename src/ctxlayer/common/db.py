"""Read-only DuckDB access with a SELECT-only SQL guard."""
import datetime as dt
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import duckdb

_MAIN_SCHEMA = "main"


class SqlRejected(ValueError):
    """SQL is not a single SELECT statement."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool
    sql: str


def connect_readonly(path: Path) -> duckdb.DuckDBPyConnection:
    if not Path(path).is_file():
        raise FileNotFoundError(path)

    # No external access: SQL could otherwise read local files, e.g. FROM read_text('.env').
    return duckdb.connect(str(path), read_only=True, config={"enable_external_access": False})


def list_tables(con, table_type: str | None = None) -> list[str]:
    """Tables and views of schema main; only those of `table_type` ("BASE TABLE" or "VIEW") if given."""
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = ?"
        " AND (? IS NULL OR table_type = ?) ORDER BY table_name",
        [_MAIN_SCHEMA, table_type, table_type],
    ).fetchall()
    return [name for (name,) in rows]


def table_schema(con, table: str) -> list[tuple[str, str]]:
    """(column, type) in table order."""
    rows = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns"
        " WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
        [_MAIN_SCHEMA, table],
    ).fetchall()
    return [(name, data_type) for name, data_type in rows]


def check_select(sql: str) -> str:
    """Return `sql` without trailing ';' if it is exactly one SELECT."""
    cleaned = sql.strip().rstrip(";").strip()

    # Parse with DuckDB itself: EXPLAIN, INSERT, DROP etc. have other types.
    try:
        statements = duckdb.extract_statements(cleaned)
    except duckdb.Error as error:
        raise SqlRejected(f"cannot parse SQL: {error}") from error

    if len(statements) != 1:
        raise SqlRejected(f"expected exactly one statement, got {len(statements)}")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise SqlRejected(f"only SELECT is allowed, got {statements[0].type.name}")
    return cleaned


def run_select(con, sql: str, limit: int) -> QueryResult:
    checked = check_select(sql)
    cursor = con.execute(checked)
    columns = [column[0] for column in cursor.description]

    # Fetch one extra row to know whether the result was cut off.
    fetched = cursor.fetchmany(limit + 1)
    rows = [[to_jsonable(value) for value in row] for row in fetched[:limit]]
    return QueryResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=len(fetched) > limit,
        sql=checked,
    )


def to_jsonable(value):
    """date/datetime -> ISO string, Decimal -> float, NaN -> None."""
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def quote_ident(name: str) -> str:
    """Double-quoted SQL identifier; inner double quotes are doubled."""
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value) -> str:
    """SQL literal for any scalar (ints too, e.g. 42 -> '42'); None -> NULL."""
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"
