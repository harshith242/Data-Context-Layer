"""Re-run the baselines' SQL on the local database and cross-check them against each other."""
import math
import re
from dataclasses import asdict, dataclass, replace
from datetime import date
from numbers import Number

import sqlglot
import yaml

from ctxlayer.common.config import Settings, artifacts_dir
from ctxlayer.common.db import run_select

_LOCAL_DIALECT = "duckdb"
_SEMANTIC_FILE = "semantic.yaml"
_ISO_DATE_LEN = len("2025-06-30")

# Empty call parentheses: "now()", "CURRENT_DATE()", "NOW( )".
_CALL = r"\s*\(\s*\)"
_DATE_NOW = rf"current_date(?:{_CALL})?|today{_CALL}|get_current_date{_CALL}"
_TIME_NOW = rf"current_timestamp(?:{_CALL})?|now{_CALL}|get_current_timestamp{_CALL}"
_NOW_PATTERN = re.compile(
    rf"(?<!\w)(?:(?P<date>{_DATE_NOW})|(?P<time>{_TIME_NOW}))(?!\w)", re.IGNORECASE
)


@dataclass
class RefResult:
    sql: str
    columns: list[str]
    rows: list[list]
    error: str | None
    reanchored: bool
    transpiled: bool
    uses_wall_clock: bool = False  # SQL uses "now" and no anchor was available


def reanchor_sql(sql: str, anchor: date | None) -> tuple[str, bool]:
    """Pin "now" to `anchor`: CURRENT_DATE -> DATE '2025-06-30'. Returns (sql, changed)."""
    if anchor is None:
        return sql, False

    def pin(match: re.Match) -> str:
        if match.group("date"):
            return f"DATE '{anchor.isoformat()}'"
        return f"TIMESTAMP '{anchor.isoformat()} 00:00:00'"

    new_sql, count = _NOW_PATTERN.subn(pin, sql)
    return new_sql, count > 0


def execute_reference(
    con, sql: str, anchor: date | None, limit: int, dialect_fallback: str | None
) -> RefResult:
    """Run `sql` read-only; on failure retry once transpiled from `dialect_fallback`."""
    anchored, reanchored = reanchor_sql(sql, anchor)

    # Without an anchor "now" is the wall clock, which may be long after the data ends.
    wall_clock = anchor is None and _NOW_PATTERN.search(sql) is not None
    result = replace(_attempt(con, anchored, limit), reanchored=reanchored, uses_wall_clock=wall_clock)
    if result.error is None or dialect_fallback is None:
        return result

    # e.g. Spark ADD_MONTHS(...) -> DuckDB date + INTERVAL; keep the first error if sqlglot fails.
    try:
        local_sql = sqlglot.transpile(anchored, read=dialect_fallback, write=_LOCAL_DIALECT)[0]
    except sqlglot.errors.SqlglotError:
        return result
    return replace(
        _attempt(con, local_sql, limit), reanchored=reanchored, transpiled=True, uses_wall_clock=wall_clock
    )


def results_agree(a: RefResult | None, b: RefResult | None, rel_tol: float) -> bool | None:
    """Same numbers (any order, within `rel_tol`), else same rows; None if one side is unusable."""
    if a is None or b is None or a.error or b.error:
        return None

    numbers_a, numbers_b = numeric_cells(a.rows), numeric_cells(b.rows)
    if not numbers_a and not numbers_b:
        return _row_texts(a.rows) == _row_texts(b.rows)
    if len(numbers_a) != len(numbers_b):
        return False

    pairs = zip(sorted(numbers_a), sorted(numbers_b))
    return all(math.isclose(x, y, rel_tol=rel_tol) for x, y in pairs)


def resolve_anchor(settings: Settings) -> date | None:
    """eval.now_anchor, else `now.value` of the built semantic.yaml, else None."""
    configured = settings.get("eval.now_anchor")
    if configured:
        return _to_date(configured)

    semantic_path = artifacts_dir(settings) / _SEMANTIC_FILE
    if not semantic_path.is_file():
        return None

    now = (yaml.safe_load(semantic_path.read_text()) or {}).get("now") or {}
    if not now.get("value"):
        return None
    return _to_date(now["value"])


def build_references(questions, con, anchor: date | None, settings: Settings) -> dict[str, dict]:
    """{qid: {baselines: {name: RefResult dict | None}, agree}}: every baseline's SQL re-run, JSON-ready.

    A baseline may name a `dialect` (e.g. spark): SQL that fails here is retried transpiled from it.
    """
    limit = settings.get("eval.reference_row_limit")
    rel_tol = settings.get("eval.rel_tolerance")
    dialects = {baseline["name"]: baseline.get("dialect") for baseline in settings.get("eval.baselines")}

    references = {}
    for question in questions:
        results = {
            name: _reference(con, contestant.sql, anchor, limit, dialects.get(name))
            for name, contestant in question.baselines.items()
        }
        references[question.qid] = {
            "baselines": {name: None if result is None else asdict(result) for name, result in results.items()},
            "agree": _baselines_agree(list(results.values()), rel_tol),
        }
    return references


def _baselines_agree(results: list[RefResult | None], rel_tol: float) -> bool | None:
    # Everyone against the first one with SQL; None when fewer than two sides are comparable.
    present = [result for result in results if result is not None]
    if len(present) < 2:
        return None

    verdicts = [results_agree(present[0], other, rel_tol) for other in present[1:]]
    if False in verdicts:
        return False
    if None in verdicts:
        return None
    return True


def _reference(con, sql: str | None, anchor, limit: int, dialect_fallback) -> RefResult | None:
    if sql is None:
        return None
    return execute_reference(con, sql, anchor, limit, dialect_fallback)


def _attempt(con, sql: str, limit: int) -> RefResult:
    # Reference SQL comes from a CSV: any failure is data for the report, never a crash.
    try:
        found = run_select(con, sql, limit)
    except Exception as error:
        return RefResult(sql, [], [], str(error), reanchored=False, transpiled=False)
    return RefResult(found.sql, found.columns, found.rows, None, reanchored=False, transpiled=False)


def numeric_cells(rows: list[list]) -> list[float]:
    """Numeric cells of `rows` in order; bool and text are skipped."""
    return [cell for row in rows for cell in row if _is_number(cell)]


def _is_number(cell) -> bool:
    return isinstance(cell, Number) and not isinstance(cell, bool)


def _row_texts(rows: list[list]) -> list[str]:
    return sorted(str(row) for row in rows)


def _to_date(value) -> date:
    # Accepts "2025-06-30", a YAML date or a datetime.
    return date.fromisoformat(str(value)[:_ISO_DATE_LEN])
