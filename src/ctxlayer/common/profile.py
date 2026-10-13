"""Profiles every table: column statistics, functional dependencies and repeat keys."""
from ctxlayer.common.config import Settings
from ctxlayer.common.db import list_tables, quote_ident, table_schema, to_jsonable

_MAX_TEXT = 80
_INTEGER_TYPES = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
}
_VARCHAR = "VARCHAR"
_MIN_FD_DISTINCT = 2
_MIN_REPEAT_SHARE = 0.01

# YYYYMM = year 1900..2999 and month 01..12, e.g. 202506.
_YYYYMM_MATCH = (
    "TRY_CAST({c} AS BIGINT) BETWEEN 190001 AND 299912 AND TRY_CAST({c} AS BIGINT) % 100 BETWEEN 1 AND 12"
)


def profile_database(con, settings: Settings) -> dict:
    """{"tables": {table: {rows, columns, fds, repeat_keys}}}; the caller writes profile.json."""
    tables = {}
    for table in list_tables(con):
        tables[table] = _profile_table(con, table, settings)
    return {"tables": tables}


def _cut(value):
    value = to_jsonable(value)
    if isinstance(value, str):
        return value[:_MAX_TEXT]
    return value


def _profile_table(con, table: str, settings: Settings) -> dict:
    schema = table_schema(con, table)
    rows = con.execute(f"SELECT count(*) FROM {quote_ident(table)}").fetchone()[0]

    columns = {}
    for name, sql_type in schema:
        columns[name] = _profile_column(con, table, name, sql_type, rows, settings)

    return {
        "rows": rows,
        "columns": columns,
        "fds": _find_fds(con, table, columns, settings),
        "repeat_keys": _find_repeat_keys(con, table, columns, rows),
    }


def _profile_column(con, table: str, name: str, sql_type: str, rows: int, settings: Settings) -> dict:
    col = quote_ident(name)
    is_text = sql_type == _VARCHAR
    is_integer = sql_type in _INTEGER_TYPES

    # One scan: counts, extremes, string lengths and the YYYYMM match count.
    selects = [f"count({col})", f"count(DISTINCT {col})", f"min({col})", f"max({col})"]
    selects += [f"avg(length({col}))", f"max(length({col}))"] if is_text else ["NULL", "NULL"]
    if is_text or is_integer:
        selects.append(f"count(*) FILTER (WHERE {_YYYYMM_MATCH.format(c=col)})")
    else:
        selects.append("0")
    non_null, distinct, low, high, avg_len, max_len, yyyymm = con.execute(
        f"SELECT {', '.join(selects)} FROM {quote_ident(table)}"
    ).fetchone()

    column = {
        "type": sql_type,
        "null_pct": round(100 * (rows - non_null) / rows, 2) if rows else 0.0,
        "distinct": distinct,
        "min": _cut(low),
        "max": _cut(high),
        "avg_len": round(avg_len, 1) if avg_len is not None else None,
        "max_len": max_len,
        "is_integer": is_integer,
        "looks_yyyymm": non_null > 0 and yyyymm == non_null,
    }

    # Top values only for low-cardinality columns: they are the vocabulary of the column.
    if distinct <= settings.get("profile.max_distinct_for_top"):
        column["top_values"] = _top_values(con, table, col, settings.get("profile.top_values"))
    column["samples"] = _samples(con, table, col, settings.get("profile.samples"))
    return column


def _top_values(con, table: str, col: str, limit: int) -> list[list]:
    found = con.execute(
        f"SELECT {col}, count(*) AS n FROM {quote_ident(table)} WHERE {col} IS NOT NULL "
        f"GROUP BY {col} ORDER BY n DESC, {col} LIMIT {limit}"
    ).fetchall()
    return [[_cut(value), count] for value, count in found]


def _samples(con, table: str, col: str, limit: int) -> list:
    found = con.execute(
        f"SELECT DISTINCT {col} FROM {quote_ident(table)} WHERE {col} IS NOT NULL LIMIT {limit}"
    ).fetchall()
    return [_cut(value) for (value,) in found]


def _find_fds(con, table: str, columns: dict, settings: Settings) -> list[list[str]]:
    """Pairs [a, b]: every a maps to one b and a has more values, so b is a parent level of a."""
    max_distinct = settings.get("profile.fd_max_distinct")
    candidates = [
        name for name, column in columns.items()
        if column["type"] == _VARCHAR and _MIN_FD_DISTINCT <= column["distinct"] <= max_distinct
    ]

    fds = []
    for child in candidates:
        for parent in candidates:
            if columns[child]["distinct"] <= columns[parent]["distinct"]:
                continue

            broken = con.execute(
                f"SELECT 1 FROM {quote_ident(table)} WHERE {quote_ident(child)} IS NOT NULL "
                f"AND {quote_ident(parent)} IS NOT NULL GROUP BY {quote_ident(child)} "
                f"HAVING count(DISTINCT {quote_ident(parent)}) > 1 LIMIT 1"
            ).fetchone()
            if broken is None:
                fds.append([child, parent])
    return fds


def _find_repeat_keys(con, table: str, columns: dict, rows: int) -> list[dict]:
    """Columns repeating across rows, with the other columns that vary within one key value."""
    keys = []
    for name, column in columns.items():
        distinct = column["distinct"]
        if not 0 < distinct < rows or distinct / rows < _MIN_REPEAT_SHARE:
            continue

        others = [other for other in columns if other != name]
        varying = []
        if others:
            row = con.execute(_varying_sql(table, name, others)).fetchone()
            varying = [other for other, varies in zip(others, row) if varies]
        keys.append({"column": name, "varying_columns": varying})
    return keys


def _varying_sql(table: str, key: str, others: list[str]) -> str:
    # Per key value: does column i take more than one value? Then: is that true for any key value?
    inner = ", ".join(f"count(DISTINCT {quote_ident(o)}) > 1 AS v{i}" for i, o in enumerate(others))
    outer = ", ".join(f"coalesce(bool_or(v{i}), false)" for i in range(len(others)))
    return (
        f"SELECT {outer} FROM (SELECT {inner} FROM {quote_ident(table)} "
        f"WHERE {quote_ident(key)} IS NOT NULL GROUP BY {quote_ident(key)})"
    )
