"""Metric compiler: validates a request against semantic.yaml and runs the metric SQL, never mixing denominations or scenarios."""
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date

from dateutil.relativedelta import relativedelta

from ctxlayer.common.db import quote_ident, quote_literal, to_jsonable
from ctxlayer.structured.metrics import METRICS, Metric, MetricContext, MetricError

_KIND_MEASURES = "measures"
_BAND_PRIMARY = "primary"
_BAND_ALL = "all"
_ACTUAL = "actual"
_PROJECTED = "projected"
_DEFAULT_GRAIN = "month"
_SCENARIO_LABEL = "scenario_label"
_DIMENSION_ROLES = {"dimension", "location", "hierarchy_level", "entity_id", "entity_label"}
_PROJECTIONS = {Metric.PROJECTION_PATH, Metric.PROJECTION_BAND}
_ALL_GRAINS = ["day", "week", "month", "quarter", "year"]

_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_YEAR = re.compile(r"^(\d{4})$")
_RELATIVE = re.compile(r"^last_(\d+)_(days|weeks|months|quarters|years)$")
_YTD = "ytd"
_UNIT_DELTA = {
    "days": lambda n: relativedelta(days=n),
    "weeks": lambda n: relativedelta(weeks=n),
    "months": lambda n: relativedelta(months=n),
    "quarters": lambda n: relativedelta(months=3 * n),
    "years": lambda n: relativedelta(years=n),
}

_DEFINITIONS = {
    Metric.LATEST_VALUE: "Latest value per entity on or before now.",
    Metric.VALUE_AT: "Latest value per entity on or before the end date (all bands when band=all).",
    Metric.SERIES: "Average value per period at the grain over the window.",
    Metric.CHANGE_ABS: "Latest value at end minus latest value at start.",
    Metric.CHANGE_PCT: "(Latest at end - latest at start) / latest at start x 100.",
    Metric.CUMULATIVE_CHANGE_PCT: "Percent change from the first observation in the window to the latest at end.",
    Metric.PERIOD_AVG: "Average value per period at the grain; highest period flagged best, lowest worst.",
    Metric.PERIODS_ABOVE_AVG: "Count of periods whose average exceeds the average of all observations in the window.",
    Metric.STREAK: "Longest and current runs of consecutive period-over-period increases / decreases at the grain.",
    Metric.VOLATILITY: "Sample standard deviation of period-over-period % change at the grain.",
    Metric.COEFFICIENT_OF_VARIATION: "Sample standard deviation / average of the values in the window x 100.",
    Metric.SPREAD: "Max - min of the latest value across the dimension's values at end.",
    Metric.RANK: "Top-n entities by latest value at end, ranked within each denomination group.",
    Metric.PROJECTION_PATH: "Projected value for the next n periods after now.",
    Metric.PROJECTION_BAND: "High - low band per projected period for the next n periods after now.",
    Metric.COMPARE: "The inner metric for every entity, side by side.",
}


@dataclass
class MetricRequest:
    metric: str
    entities: list[str]
    start: str | None = None
    end: str | None = None
    grain: str | None = None
    scenario: str | None = None
    band: str = _BAND_PRIMARY
    dimension: str | None = None
    n: int | None = None
    convert_to: str | None = None
    inner_metric: str | None = None


# --------------------------------------------------------------------- window

def _bounds(token: str, anchor: date) -> tuple[date, date]:
    """First and last day a token covers: '2025-03' -> (03-01, 03-31); 'last_6_months' -> (anchor - 6 months, anchor)."""
    if match := _DAY.match(token):
        try:
            day = date(*map(int, match.groups()))
        except ValueError:
            raise MetricError("invalid_date", f"not a calendar date: {token}") from None
        return day, day

    if match := _MONTH.match(token):
        year, month = map(int, match.groups())
        if not 1 <= month <= 12:
            raise MetricError("invalid_date", f"not a month: {token}")
        return date(year, month, 1), date(year, month, monthrange(year, month)[1])

    if match := _YEAR.match(token):
        year = int(token)
        return date(year, 1, 1), date(year, 12, 31)

    if token == _YTD:
        return date(anchor.year, 1, 1), anchor

    if match := _RELATIVE.match(token):
        count, unit = int(match.group(1)), match.group(2)
        return anchor - _UNIT_DELTA[unit](count), anchor

    raise MetricError(
        "invalid_date", f"unrecognised date '{token}'; use YYYY-MM-DD, YYYY-MM, YYYY, ytd or last_<n>_<unit>"
    )


def _check_overlap(token: str, low: date, high: date, valid: tuple[date, date]) -> None:
    """Wholly outside the data is an error; partly outside is clamped ("2025" when data stops in June)."""
    if high >= valid[0] and low <= valid[1]:
        return
    valid_range = [valid[0].isoformat(), valid[1].isoformat()]
    raise MetricError("out_of_range", f"'{token}' is outside the data", valid_range=valid_range)


def resolve_window(start: str | None, end: str | None, now: date, valid: tuple[date, date]) -> tuple[date, date]:
    """(start, end) as dates, relative tokens anchored on `now`, clamped to `valid`. Raises MetricError."""
    # End first: a relative start ("last_6_months") counts back from it.
    end_low, end_high = _bounds(end, now) if end else (now, now)
    _check_overlap(end or "now", end_low, end_high, valid)
    end_day = min(end_high, valid[1])
    if not start:
        return valid[0], end_day

    start_low, start_high = _bounds(start, end_day)
    _check_overlap(start, start_low, start_high, valid)
    start_day = max(start_low, valid[0])
    if start_day > end_day:
        raise MetricError("invalid_window", "start is after end", valid_range=[valid[0].isoformat(), valid[1].isoformat()])
    return start_day, end_day


# -------------------------------------------------------------------- compile

def _measures_tables(semantic: dict) -> dict[str, dict]:
    return {name: table for name, table in semantic["tables"].items() if table["kind"] == _KIND_MEASURES}


def _entity_column(spec: dict) -> str | None:
    """Entity id column, else its label column (label-only tables)."""
    entity = spec.get("entity", {})
    return entity.get("id") or entity.get("label")


def _check_complete(spec: dict, table: str) -> None:
    """Metrics need an entity, a business time and a primary measure; say which are missing."""
    missing = [
        key for key, present in (
            ("entity", _entity_column(spec)),
            ("time", spec.get("time", {}).get("business")),
            ("primary_measure", spec.get("primary_measure")),
        ) if not present
    ]
    if missing:
        raise MetricError("incomplete_table", f"table '{table}' lacks {', '.join(missing)}", missing=missing)


def _band_columns(spec: dict) -> dict[str, str]:
    return {measure["band"]: measure["logical"] for measure in spec.get("measures", []) if measure.get("band")}


def _scenario(req: MetricRequest, spec: dict, metric: Metric) -> str | None:
    """Label to filter on: the request's, else actual (projected for projection metrics)."""
    declared = spec.get("scenario")
    if not declared:
        return None

    labels = sorted(set(declared["values"].values()))
    chosen = req.scenario or (_PROJECTED if metric in _PROJECTIONS else declared["default"])
    if chosen not in labels:
        raise MetricError("invalid_scenario", f"unknown scenario '{chosen}'", valid_scenarios=labels)
    return chosen


def _value_column(req: MetricRequest, spec: dict, metric: Metric) -> str | None:
    """Measure to aggregate: primary, a band, or None for every band (value_at only)."""
    if req.band == _BAND_PRIMARY:
        return spec["primary_measure"]

    bands = _band_columns(spec)
    if req.band == _BAND_ALL and metric == Metric.VALUE_AT:
        return None
    if req.band not in bands:
        valid = [_BAND_PRIMARY, *bands, *([_BAND_ALL] if metric == Metric.VALUE_AT else [])]
        raise MetricError("invalid_band", f"band '{req.band}' not available", valid_bands=valid)
    return bands[req.band]


def _conversions(req: MetricRequest, semantic: dict, spec: dict) -> list[dict]:
    """Conversions into `convert_to` stated in context.md, for this table's denomination columns."""
    if not req.convert_to:
        return []

    found = [
        conversion for conversion in semantic.get("conversions", [])
        if conversion["to"] == req.convert_to and conversion["column"] in spec.get("denominations", [])
    ]
    if not found:
        raise MetricError(
            "no_conversion", f"no conversion to '{req.convert_to}' is stated in the context files"
        )
    return found


def _check_options(req: MetricRequest, spec: dict) -> str:
    """Validate grain, n and dimension; returns the grain to use."""
    grains = spec["time"].get("grains") or _ALL_GRAINS
    grain = req.grain or (_DEFAULT_GRAIN if _DEFAULT_GRAIN in grains else grains[0])
    if grain not in grains:
        raise MetricError("invalid_grain", f"grain '{grain}' not available", valid_grains=grains)

    if req.n is not None and req.n < 1:
        raise MetricError("invalid_n", "n must be at least 1")

    if req.dimension is None:
        return grain
    allowed = [c["logical"] for c in spec["columns"] if c["role"] in _DIMENSION_ROLES]
    if req.dimension not in allowed:
        raise MetricError("invalid_dimension", f"dimension '{req.dimension}' not available", dimensions=allowed)
    return grain


def _compile(req: MetricRequest, semantic: dict, table: str, entity_ids: list[str]) -> tuple[str, dict]:
    if req.metric not in METRICS:
        raise MetricError("unknown_metric", f"unknown metric '{req.metric}'", metrics=[m.value for m in Metric])
    if not entity_ids:
        raise MetricError("no_entities", "no entities given")

    metric = Metric(req.metric)
    spec = semantic["tables"][table]
    _check_complete(spec, table)
    grain = _check_options(req, spec)
    scenario = _scenario(req, spec, metric)
    value = _value_column(req, spec, metric)
    conversions = _conversions(req, semantic, spec)

    now = date.fromisoformat(semantic["now"]["value"])
    valid = tuple(date.fromisoformat(day) for day in spec["time"]["range"])
    start, end = (None, None) if metric in _PROJECTIONS else resolve_window(req.start, req.end, now, valid)

    ctx = MetricContext(
        view=spec["view"], entity=_entity_column(spec), denominations=spec.get("denominations", []),
        time=spec["time"]["business"], ids=entity_ids, value=value, bands=_band_columns(spec),
        now=now, grain=grain, scenario=scenario, start=start, end=end, n=req.n,
        dimension=req.dimension, inner_metric=req.inner_metric, conversions=conversions,
    )
    sql = METRICS[metric](ctx)

    as_of = now if metric in _PROJECTIONS or metric == Metric.LATEST_VALUE else end
    definition = {
        "metric": metric.value, "definition": _DEFINITIONS[metric], "table": table,
        "measure": value or "all bands", "scenario": scenario, "grain": grain,
        "window": [start.isoformat(), end.isoformat()] if start else None,
        "as_of": as_of.isoformat(), "table_time_range": spec["time"]["range"], "conversions": conversions,
    }
    return sql, definition


def compile_metric(req: MetricRequest, semantic: dict, entity_ids: list[str]) -> tuple[str, dict]:
    """(sql, definition) on the first measures table. Raises MetricError."""
    tables = _measures_tables(semantic)
    if not tables:
        raise MetricError("no_measures", "the semantic layer has no measures table")
    return _compile(req, semantic, next(iter(tables)), entity_ids)


# ------------------------------------------------------------------------ run

def _in_list(ids: list[str]) -> str:
    return ", ".join(quote_literal(i) for i in ids)


def _pick_table(con, semantic: dict, entity_ids: list[str]) -> str:
    """First measures table whose view has rows for the entity ids."""
    for name, spec in _measures_tables(semantic).items():
        entity = _entity_column(spec)
        if not entity:
            continue
        sql = f"SELECT 1 FROM {spec['view']} WHERE {quote_ident(entity)} IN ({_in_list(entity_ids)}) LIMIT 1"
        if con.execute(sql).fetchone():
            return name
    raise MetricError("no_data", "no rows for the requested entities", entities=entity_ids)


def _available_range(con, spec: dict, entity_ids: list[str], scenario: str | None) -> list:
    time = spec["time"]["business"]
    where = f"{quote_ident(_entity_column(spec))} IN ({_in_list(entity_ids)})"
    if scenario:
        where += f" AND {_SCENARIO_LABEL} = {quote_literal(scenario)}"
    low, high = con.execute(f'SELECT min("{time}"), max("{time}") FROM {spec["view"]} WHERE {where}').fetchone()
    return [to_jsonable(low), to_jsonable(high)]


def _denominations(spec: dict, columns: list[str], rows: list[list]) -> list[dict]:
    """Distinct denomination combinations in the result, e.g. [{'currency': 'CUR1', 'unit': 'unit1'}]."""
    names = [name for name in spec.get("denominations", []) if name in columns]
    positions = [columns.index(name) for name in names]
    seen = {tuple(row[i] for i in positions) for row in rows}
    return [dict(zip(names, combo)) for combo in sorted(seen, key=str)]


def run_metric(con_sem, req: MetricRequest, semantic: dict, entity_ids: list[str], limit: int | None = None) -> dict:
    """Compile and execute; the result dict (at most `limit` rows, flagged truncated), or {"error": {...}}."""
    try:
        if not entity_ids:
            raise MetricError("no_entities", "no entities given")
        table = _pick_table(con_sem, semantic, entity_ids)
        sql, definition = _compile(req, semantic, table, entity_ids)
    except MetricError as error:
        return {"error": error.error}

    cursor = con_sem.execute(sql)
    columns = [column[0] for column in cursor.description]
    fetched = cursor.fetchall() if limit is None else cursor.fetchmany(limit + 1)
    truncated = limit is not None and len(fetched) > limit
    rows = [[to_jsonable(cell) for cell in row] for row in fetched[:limit]]
    spec = semantic["tables"][table]
    if not rows:
        return {"error": {
            "type": "no_data", "message": "no rows for the entities in this window and scenario",
            "available_range": _available_range(con_sem, spec, entity_ids, definition["scenario"]),
        }}

    denominations = _denominations(spec, columns, rows)
    warnings = []
    if len(denominations) > 1:
        warnings.append(
            f"result spans {len(denominations)} denomination combinations; "
            "values are grouped, not comparable across groups"
        )
    return {
        "metric": req.metric, "definition": definition, "sql": sql, "columns": columns, "rows": rows,
        "as_of": definition["as_of"], "scenario": definition["scenario"], "denominations": denominations,
        "conversions_applied": definition["conversions"], "warnings": warnings, "truncated": truncated,
    }
