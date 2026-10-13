"""Metric library: one SQL builder per metric over a semantic measures view."""
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from ctxlayer.common.db import quote_ident, quote_literal

_SCENARIO_LABEL = "scenario_label"
_DEFAULT_N = 5
_BAND_LOW = "low"
_BAND_HIGH = "high"
_RANK = '"rank"'
_CHANGE_ABS = "end_value - start_value AS change_abs"
_CHANGE_PCT = "(end_value - start_value) / NULLIF(start_value, 0) * 100 AS change_pct"


class Metric(str, Enum):
    LATEST_VALUE = "latest_value"
    VALUE_AT = "value_at"
    SERIES = "series"
    CHANGE_ABS = "change_abs"
    CHANGE_PCT = "change_pct"
    CUMULATIVE_CHANGE_PCT = "cumulative_change_pct"
    PERIOD_AVG = "period_avg"
    PERIODS_ABOVE_AVG = "periods_above_avg"
    STREAK = "streak"
    VOLATILITY = "volatility"
    COEFFICIENT_OF_VARIATION = "coefficient_of_variation"
    SPREAD = "spread"
    RANK = "rank"
    PROJECTION_PATH = "projection_path"
    PROJECTION_BAND = "projection_band"
    COMPARE = "compare"


_CROSS_ENTITY = {Metric.RANK, Metric.SPREAD, Metric.COMPARE}


class MetricError(Exception):
    """A request the compiler refuses; `error` is the structured error body."""

    def __init__(self, kind: str, message: str, **details):
        super().__init__(message)
        self.error = {"type": kind, "message": message, **details}


@dataclass
class MetricContext:
    """Everything a builder needs, resolved and validated by the compiler."""

    view: str
    entity: str
    denominations: list[str]
    time: str
    ids: list[str]
    value: str | None  # None: every band (value_at)
    bands: dict[str, str]
    now: date
    grain: str
    scenario: str | None = None
    start: date | None = None
    end: date | None = None
    n: int | None = None
    dimension: str | None = None
    inner_metric: str | None = None
    conversions: list[dict] = field(default_factory=list)


# ----------------------------------------------------------------- SQL pieces

def _date(value: date) -> str:
    return f"DATE '{value.isoformat()}'"


def _period(ctx: MetricContext) -> str:
    """Start date of the grain period a row falls in, e.g. 2025-05-07 at month -> 2025-05-01."""
    return f"CAST(date_trunc({quote_literal(ctx.grain)}, _t) AS DATE)"


def _keys(ctx: MetricContext) -> str:
    """Entity + denomination columns: the grain every result row is unique on."""
    return ", ".join(quote_ident(name) for name in [ctx.entity, *ctx.denominations])


def _denomination_sql(ctx: MetricContext) -> tuple[list[str], str, list[str]]:
    """(select items, value multiplier, filters): denominations as-is, or converted via context.md factors."""
    items, factors, filters = [], [], []
    for name in ctx.denominations:
        column = quote_ident(name)
        own = [conversion for conversion in ctx.conversions if conversion["column"] == name]
        if not own:
            items.append(column)
            continue

        # USD -> EUR: relabel the column and scale the value; rows in any other denomination are dropped.
        relabel = " ".join(f"WHEN {quote_literal(c['from'])} THEN {quote_literal(c['to'])}" for c in own)
        scale = " ".join(f"WHEN {quote_literal(c['from'])} THEN {float(c['factor'])!r}" for c in own)
        items.append(f"CASE {column} {relabel} ELSE {column} END AS {column}")
        factors.append(f"CASE {column} {scale} ELSE 1 END")
        allowed = {c["from"] for c in own} | {own[0]["to"]}
        filters.append(f"{column} IN ({', '.join(quote_literal(v) for v in sorted(allowed))})")
    return items, " * ".join(factors) or "1", filters


def _base(ctx: MetricContext, where: list[str], values: dict[str, str] | None = None, columns: tuple = ()) -> str:
    """Rows of the view for the entities and scenario: keys, `_t`, one alias per value, extra plain columns."""
    values = values or {"_v": ctx.value}
    denominations, multiplier, filters = _denomination_sql(ctx)

    selected = [quote_ident(ctx.entity), *denominations, f"{quote_ident(ctx.time)} AS _t"]
    selected += [f"{quote_ident(column)} * {multiplier} AS {alias}" for alias, column in values.items()]
    selected += [quote_ident(column) for column in columns]

    predicates = [f"{quote_ident(ctx.entity)} IN ({', '.join(quote_literal(i) for i in ctx.ids)})", *filters, *where]
    if ctx.scenario:
        predicates.append(f"{_SCENARIO_LABEL} = {quote_literal(ctx.scenario)}")
    return f"SELECT {', '.join(selected)} FROM {quote_ident(ctx.view)} WHERE {' AND '.join(predicates)}"


def _on_or_before(ctx: MetricContext, day: date) -> str:
    return f"{quote_ident(ctx.time)} <= {_date(day)}"


def _window(ctx: MetricContext) -> list[str]:
    return [f"{quote_ident(ctx.time)} BETWEEN {_date(ctx.start)} AND {_date(ctx.end)}"]


def _pick(ctx: MetricContext, source: str, selected: str, order: str) -> str:
    """One row per entity and denomination: the latest (DESC) or earliest (ASC) of `source`."""
    return (
        f"SELECT {_keys(ctx)}, _t, {selected} FROM {source} "
        f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {_keys(ctx)} ORDER BY _t {order}) = 1"
    )


def _periods(ctx: MetricContext) -> str:
    """CTE `p`: average value per period at the grain, from CTE `b`."""
    return (
        f"p AS (SELECT {_keys(ctx)}, {_period(ctx)} AS period, avg(_v) AS pv, count(*) AS obs "
        f"FROM b GROUP BY {_keys(ctx)}, period)"
    )


# ------------------------------------------------------------------- builders

def _latest_value(ctx: MetricContext) -> str:
    return (
        f"WITH b AS ({_base(ctx, [_on_or_before(ctx, ctx.now)])}) "
        f"SELECT {_keys(ctx)}, _t AS as_of, _v AS value, {quote_literal(ctx.scenario)} AS scenario "
        f"FROM ({_pick(ctx, 'b', '_v', 'DESC')}) ORDER BY {_keys(ctx)}"
    )


def _value_at(ctx: MetricContext) -> str:
    # One column named `value`, or one per band (low, mid, high) when no single measure was chosen.
    values = {"value": ctx.value} if ctx.value else dict(ctx.bands)
    return (
        f"WITH b AS ({_base(ctx, [_on_or_before(ctx, ctx.end)], values)}) "
        f"SELECT {_keys(ctx)}, _t AS as_of, {', '.join(values)}, {quote_literal(ctx.scenario)} AS scenario "
        f"FROM ({_pick(ctx, 'b', ', '.join(values), 'DESC')}) ORDER BY {_keys(ctx)}"
    )


def _series(ctx: MetricContext) -> str:
    return (
        f"WITH b AS ({_base(ctx, _window(ctx))}), {_periods(ctx)} "
        f"SELECT {_keys(ctx)}, period, pv AS value, obs FROM p ORDER BY {_keys(ctx)}, period"
    )




def _change_sql(ctx: MetricContext, start_row: str, change: str) -> str:
    """`start_row` (one row per entity) vs the latest row on or before the window end."""
    return (
        f"WITH b AS ({_base(ctx, [_on_or_before(ctx, ctx.end)])}), "
        f"s AS (SELECT {_keys(ctx)}, _t AS start_date, _v AS start_value FROM ({start_row})), "
        f"e AS (SELECT {_keys(ctx)}, _t AS end_date, _v AS end_value FROM ({_pick(ctx, 'b', '_v', 'DESC')})) "
        f"SELECT {_keys(ctx)}, start_date, start_value, end_date, end_value, {change} "
        f"FROM e JOIN s USING ({_keys(ctx)}) ORDER BY {_keys(ctx)}"
    )


def _latest_at_start(ctx: MetricContext) -> str:
    """Latest row on or before start; an entity that begins later starts at its first row (start_date shows it)."""
    start = _date(ctx.start)
    return (
        f"SELECT {_keys(ctx)}, _t, _v FROM b "
        f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {_keys(ctx)} "
        f"ORDER BY (_t > {start}), CASE WHEN _t <= {start} THEN _t END DESC, _t) = 1"
    )


def _first_in_window(ctx: MetricContext) -> str:
    return _pick(ctx, f"(SELECT * FROM b WHERE _t >= {_date(ctx.start)})", "_v", "ASC")


def _change_abs(ctx: MetricContext) -> str:
    return _change_sql(ctx, _latest_at_start(ctx), _CHANGE_ABS)


def _change_pct(ctx: MetricContext) -> str:
    return _change_sql(ctx, _latest_at_start(ctx), _CHANGE_PCT)


def _cumulative_change_pct(ctx: MetricContext) -> str:
    return _change_sql(ctx, _first_in_window(ctx), _CHANGE_PCT)


def _period_avg(ctx: MetricContext) -> str:
    return (
        f"WITH b AS ({_base(ctx, _window(ctx))}), {_periods(ctx)} "
        f"SELECT {_keys(ctx)}, period, pv AS value, "
        f"CASE WHEN pv = max(pv) OVER w THEN 'best' WHEN pv = min(pv) OVER w THEN 'worst' END AS flag "
        f"FROM p WINDOW w AS (PARTITION BY {_keys(ctx)}) ORDER BY {_keys(ctx)}, period"
    )


def _periods_above_avg(ctx: MetricContext) -> str:
    return (
        f"WITH b AS ({_base(ctx, _window(ctx))}), {_periods(ctx)}, "
        f"w AS (SELECT {_keys(ctx)}, avg(_v) AS window_avg FROM b GROUP BY {_keys(ctx)}) "
        f"SELECT {_keys(ctx)}, window_avg, count(*) FILTER (WHERE pv > window_avg) AS periods_above, "
        f"count(*) AS periods FROM p JOIN w USING ({_keys(ctx)}) GROUP BY {_keys(ctx)}, window_avg "
        f"ORDER BY {_keys(ctx)}"
    )


def _streak(ctx: MetricContext) -> str:
    """Runs of equal sign of the period-over-period change, found as row-number gaps (gaps and islands)."""
    keys = _keys(ctx)
    return (
        f"WITH b AS ({_base(ctx, _window(ctx))}), {_periods(ctx)}, "
        f"d AS (SELECT {keys}, period, sign(pv - lag(pv) OVER (PARTITION BY {keys} ORDER BY period)) AS dir "
        f"FROM p QUALIFY dir IS NOT NULL), "
        f"g AS (SELECT {keys}, period, dir, ROW_NUMBER() OVER (PARTITION BY {keys} ORDER BY period) - "
        f"ROW_NUMBER() OVER (PARTITION BY {keys}, dir ORDER BY period) AS grp FROM d), "
        f"runs AS (SELECT {keys}, dir, count(*) AS len, max(period) AS last_period FROM g GROUP BY {keys}, dir, grp) "
        f"SELECT {keys}, "
        f"coalesce(max(len) FILTER (WHERE dir = 1), 0) AS longest_increase, "
        f"coalesce(max(len) FILTER (WHERE dir = -1), 0) AS longest_decrease, "
        f"arg_max(CASE dir WHEN 1 THEN 'increase' WHEN -1 THEN 'decrease' ELSE 'flat' END, last_period) "
        f"AS current_direction, arg_max(len, last_period) AS current_length "
        f"FROM runs GROUP BY {keys} ORDER BY {keys}"
    )


def _volatility(ctx: MetricContext) -> str:
    keys = _keys(ctx)
    return (
        f"WITH b AS ({_base(ctx, _window(ctx))}), {_periods(ctx)}, "
        f"c AS (SELECT {keys}, (pv / NULLIF(lag(pv) OVER (PARTITION BY {keys} ORDER BY period), 0) - 1) * 100 "
        f"AS pct FROM p) "
        f"SELECT {keys}, stddev_samp(pct) AS volatility_pct, count(pct) AS changes FROM c "
        f"GROUP BY {keys} ORDER BY {keys}"
    )


def _coefficient_of_variation(ctx: MetricContext) -> str:
    return (
        f"WITH b AS ({_base(ctx, _window(ctx))}) "
        f"SELECT {_keys(ctx)}, stddev_samp(_v) / NULLIF(avg(_v), 0) * 100 AS cv_pct, count(*) AS obs "
        f"FROM b GROUP BY {_keys(ctx)} ORDER BY {_keys(ctx)}"
    )


def _spread(ctx: MetricContext) -> str:
    if not ctx.dimension:
        raise MetricError("missing_dimension", "spread needs a dimension")

    # Spread is taken inside each denomination group, never across currencies / units.
    groups = [quote_ident(name) for name in ctx.denominations]
    group_select = "".join(f"{group}, " for group in groups)
    group_by = f"GROUP BY {', '.join(groups)} ORDER BY {', '.join(groups)}" if groups else ""
    dimension = quote_ident(ctx.dimension)
    return (
        f"WITH b AS ({_base(ctx, [_on_or_before(ctx, ctx.end)], columns=(ctx.dimension,))}), "
        f"l AS ({_pick(ctx, 'b', '_v, ' + dimension, 'DESC')}) "
        f"SELECT {group_select}count(DISTINCT {dimension}) AS dimension_values, "
        f"max(_v) - min(_v) AS spread, min(_v) AS min_value, arg_min({dimension}, _v) AS min_at, "
        f"max(_v) AS max_value, arg_max({dimension}, _v) AS max_at FROM l {group_by}"
    )


def _rank(ctx: MetricContext) -> str:
    # Entities are ranked inside their denomination group: a EUR value never outranks a USD value.
    groups = [quote_ident(name) for name in ctx.denominations]
    partition = f"PARTITION BY {', '.join(groups)}" if groups else ""
    return (
        f"WITH b AS ({_base(ctx, [_on_or_before(ctx, ctx.end)])}), l AS ({_pick(ctx, 'b', '_v', 'DESC')}) "
        f"SELECT {_keys(ctx)}, _t AS as_of, _v AS value, RANK() OVER ({partition} ORDER BY _v DESC) AS {_RANK} "
        f"FROM l QUALIFY {_RANK} <= {ctx.n or _DEFAULT_N} ORDER BY {''.join(g + ', ' for g in groups)}{_RANK}, "
        f"{quote_ident(ctx.entity)}"
    )


def _after_now(ctx: MetricContext) -> list[str]:
    return [f"{quote_ident(ctx.time)} > {_date(ctx.now)}"]


def _projection_path(ctx: MetricContext) -> str:
    return (
        f"WITH b AS ({_base(ctx, _after_now(ctx))}), {_periods(ctx)} "
        f"SELECT {_keys(ctx)}, period, pv AS value, {quote_literal(ctx.scenario)} AS scenario FROM p "
        f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {_keys(ctx)} ORDER BY period) <= {ctx.n or _DEFAULT_N} "
        f"ORDER BY {_keys(ctx)}, period"
    )


def _projection_band(ctx: MetricContext) -> str:
    if _BAND_LOW not in ctx.bands or _BAND_HIGH not in ctx.bands:
        raise MetricError("no_band", "the table has no low and high measures")

    values = {"_lo": ctx.bands[_BAND_LOW], "_hi": ctx.bands[_BAND_HIGH]}
    return (
        f"WITH b AS ({_base(ctx, _after_now(ctx), values)}), "
        f"p AS (SELECT {_keys(ctx)}, {_period(ctx)} AS period, avg(_lo) AS low, avg(_hi) AS high "
        f"FROM b GROUP BY {_keys(ctx)}, period) "
        f"SELECT {_keys(ctx)}, period, low, high, high - low AS band_width, {quote_literal(ctx.scenario)} AS scenario FROM p "
        f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {_keys(ctx)} ORDER BY period) <= {ctx.n or _DEFAULT_N} "
        f"ORDER BY {_keys(ctx)}, period"
    )




def _compare(ctx: MetricContext) -> str:
    """The inner metric over all entities: every builder already returns one row group per entity."""
    if len(ctx.ids) < 2:
        raise MetricError("needs_entities", "compare needs at least two entities")
    if ctx.inner_metric not in METRICS or ctx.inner_metric in _CROSS_ENTITY:
        inner = sorted(m.value for m in Metric if m not in _CROSS_ENTITY)
        raise MetricError("invalid_inner_metric", "compare needs a per-entity inner_metric", metrics=inner)
    return METRICS[ctx.inner_metric](ctx)


METRICS: dict[str, Callable[[MetricContext], str]] = {
    Metric.LATEST_VALUE: _latest_value,
    Metric.VALUE_AT: _value_at,
    Metric.SERIES: _series,
    Metric.CHANGE_ABS: _change_abs,
    Metric.CHANGE_PCT: _change_pct,
    Metric.CUMULATIVE_CHANGE_PCT: _cumulative_change_pct,
    Metric.PERIOD_AVG: _period_avg,
    Metric.PERIODS_ABOVE_AVG: _periods_above_avg,
    Metric.STREAK: _streak,
    Metric.VOLATILITY: _volatility,
    Metric.COEFFICIENT_OF_VARIATION: _coefficient_of_variation,
    Metric.SPREAD: _spread,
    Metric.RANK: _rank,
    Metric.PROJECTION_PATH: _projection_path,
    Metric.PROJECTION_BAND: _projection_band,
    Metric.COMPARE: _compare,
}
