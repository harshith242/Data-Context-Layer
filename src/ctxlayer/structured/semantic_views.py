"""Semantic views over the source database: SQL generation, a locked-down connection, SQL validation."""
from pathlib import Path

import duckdb
import sqlglot
from sqlglot import exp

from ctxlayer.common.config import Settings, artifacts_dir
from ctxlayer.common.db import SqlRejected, check_select, quote_ident, quote_literal

_SOURCE_ALIAS = "src"
_SCENARIO_LABEL = "scenario_label"


def render_views_sql(semantic: dict) -> str:
    """One `CREATE TEMP VIEW` per table: logical column names, audit columns left out, scenario labelled."""
    statements = []
    for table, draft in semantic["tables"].items():
        selected = [f"{quote_ident(column['physical'])} AS {quote_ident(column['logical'])}" for column in draft["columns"]]

        scenario = draft.get("scenario")
        if scenario and scenario["values"]:
            selected.append(_scenario_case(draft, scenario))

        statements.append(
            f"CREATE OR REPLACE TEMP VIEW {draft['view']} AS\n"
            f"SELECT {', '.join(selected)}\n"
            f"FROM {_SOURCE_ALIAS}.main.{quote_ident(table)};"
        )
    return "\n\n".join(statements) + "\n"


def write_views(semantic: dict, settings: Settings) -> Path:
    """Write views.sql under the artifacts folder; open_semantic runs it."""
    path = _views_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_views_sql(semantic))
    return path


def open_semantic(settings: Settings) -> duckdb.DuckDBPyConnection:
    """In-memory connection with the source attached read-only and the views created, then locked."""
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH {quote_literal(str(settings.path('database.path')))} AS {_SOURCE_ALIAS} (READ_ONLY)")
    con.execute(_views_path(settings).read_text())

    # Model SQL must not read files (read_text('.env')); the locks come after ATTACH, which needs access.
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


def validate_sql(sql: str, semantic: dict) -> dict:
    """{ok, errors, warnings, tables}: only SELECTs over semantic views pass; warnings flag unsafe aggregation.

    Warnings look at the whole query, not at each sub-select.
    """
    try:
        checked = check_select(sql)
        tree = sqlglot.parse_one(checked, read="duckdb")
    except (SqlRejected, sqlglot.errors.SqlglotError) as error:
        return {"ok": False, "errors": [str(error)], "warnings": [], "tables": []}

    tables, errors = _tables(tree, {draft["view"] for draft in semantic["tables"].values()})
    warnings = [] if errors else _warnings(tree, tables, semantic)
    return {"ok": not errors, "errors": errors, "warnings": warnings, "tables": tables}


def _tables(tree: exp.Expression, views: set[str]) -> tuple[list[str], list[str]]:
    """Sorted table names read by the query, and an error per name that is not a semantic view."""
    ctes = {cte.alias for cte in tree.find_all(exp.CTE)}
    names, errors = set(), []
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            errors.append(f"table function not allowed: {table.sql(dialect='duckdb')}")
            continue
        if table.name in ctes and not (table.db or table.catalog):
            continue
        names.add(table.name)
        if table.name not in views or table.db or table.catalog:
            errors.append(f"not a semantic view: {table.sql(dialect='duckdb')}")
    return sorted(names), errors


def _warnings(tree: exp.Expression, tables: list[str], semantic: dict) -> list[str]:
    used = [draft for draft in semantic["tables"].values() if draft["view"] in tables]
    measures = {measure["logical"] for draft in used for measure in draft.get("measures", [])}
    if not measures:
        return []

    aggregated = {
        column.name for aggregate in tree.find_all(exp.AggFunc) for column in aggregate.find_all(exp.Column)
    } & measures
    referenced = {column.name for column in tree.find_all(exp.Column)} & measures
    warnings = []

    # Averaging prices over mixed currencies/units gives a number in no unit at all.
    denominations = {name for draft in used for name in draft.get("denominations", [])}
    fixed = _grouped_columns(tree) | _filtered_columns(tree)
    unfixed = sorted(denominations - fixed)
    if aggregated and unfixed:
        warnings.append(
            f"aggregate over {', '.join(sorted(aggregated))} mixes {', '.join(unfixed)} values;"
            " group by or filter them"
        )

    # Actual and projected rows in one number.
    scenario_columns = {_SCENARIO_LABEL} | {draft["scenario"]["column"] for draft in used if "scenario" in draft}
    scenario_used = any("scenario" in draft for draft in used)
    if referenced and scenario_used and not scenario_columns & (_filtered_columns(tree) | _grouped_columns(tree)):
        warnings.append("scenario not filtered or grouped: actual and projected rows are mixed; filter scenario_label")
    return warnings


def _filtered_columns(tree: exp.Expression) -> set[str]:
    return {column.name for where in tree.find_all(exp.Where) for column in where.find_all(exp.Column)}


def _grouped_columns(tree: exp.Expression) -> set[str]:
    names = set()
    for group in tree.find_all(exp.Group):
        names |= {column.name for column in group.find_all(exp.Column)}
        if not group.args.get("all"):
            continue

        # GROUP BY ALL groups by every projected column that is not an aggregate.
        for select in tree.find_all(exp.Select):
            names |= {
                column.name for projection in select.expressions for column in projection.find_all(exp.Column)
                if column.find_ancestor(exp.AggFunc) is None
            }
    return names


def _scenario_case(draft: dict, scenario: dict) -> str:
    physical = next(column["physical"] for column in draft["columns"] if column["logical"] == scenario["column"])
    branches = " ".join(f"WHEN {quote_literal(value)} THEN {quote_literal(label)}" for value, label in scenario["values"].items())
    return f"CASE {quote_ident(physical)} {branches} END AS {_SCENARIO_LABEL}"


def _views_path(settings: Settings) -> Path:
    return artifacts_dir(settings) / "structured" / "views.sql"


