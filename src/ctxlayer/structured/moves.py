"""Move detection: period-over-period changes of each entity's primary measure that are outliers vs its recent history."""
import pandas as pd

from ctxlayer.common.config import Settings

_KIND_MEASURES = "measures"
_SCENARIO_COLUMN = "scenario_label"
_ACTUAL = "actual"
_UP = "up"
_DOWN = "down"
_PERCENT = 100


def detect_moves(con_sem, semantic: dict, settings: Settings) -> list[dict]:
    """Moves of every measures table, sorted by (entity_id, date); `change_pct` is in percent (15.0 = +15%)."""
    moves = []
    for draft in semantic["tables"].values():
        if draft["kind"] != _KIND_MEASURES:
            continue
        moves.extend(_table_moves(con_sem, draft, settings))

    return sorted(moves, key=lambda move: (move["entity_id"], move["date"]))


def _table_moves(con_sem, draft: dict, settings: Settings) -> list[dict]:
    entity_spec = draft.get("entity", {})
    entity = entity_spec.get("id") or entity_spec.get("label")
    time = draft.get("time", {}).get("business")
    measure = draft.get("primary_measure")
    # A table whose roles fell back to heuristics may lack these; it simply has no moves.
    if not (entity and time and measure):
        return []

    frequency = draft.get("frequency")
    denominations = draft.get("denominations", [])

    # Moves are per series: entity at its native frequency, one denomination at a time (CUR1 vs CUR2 never mix).
    series_keys = [entity] + ([frequency] if frequency else []) + denominations
    selected = ", ".join(dict.fromkeys(series_keys + [time, measure]))
    where = f"WHERE {_SCENARIO_COLUMN} = '{_ACTUAL}'" if draft.get("scenario") else ""
    rows = con_sem.execute(
        f"SELECT {selected} FROM {draft['view']} {where} ORDER BY {entity}, {time}"
    ).df()

    moves = []
    for key, series in rows.groupby(series_keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(series_keys, key))
        moves.extend(_series_moves(series, labels, draft, entity, settings))
    return moves


def _series_moves(series, labels: dict, draft: dict, entity: str, settings: Settings) -> list[dict]:
    time = draft["time"]["business"]
    measure = draft["primary_measure"]
    frequency = draft.get("frequency")
    denominations = draft.get("denominations", [])

    window = settings.get("structured.moves.trailing_periods")
    min_history = settings.get("structured.moves.min_history")
    threshold = settings.get("structured.moves.z_threshold")
    min_change = settings.get("structured.moves.min_change_pct") / _PERCENT

    series = series.dropna(subset=[measure]).sort_values(time).reset_index(drop=True)
    change = series[measure].pct_change().replace([float("inf"), float("-inf")], float("nan"))

    # z of today's change vs the previous `window` changes (current excluded), e.g. +15% vs mean 0.1% / std 0.2%.
    history = change.shift(1).rolling(window, min_periods=min_history)
    std = history.std().where(lambda s: s > 0)
    z = (change - history.mean()) / std

    # Outlier AND material: z alone flags tiny changes on quiet series (e.g. +0.8% with z = 2.4).
    is_move = (z.abs() >= threshold) & (change.abs() >= min_change)

    # One id per series and date, e.g. "move:S1:freq1:CUR1:unit1:2025-05-05" (an entity may have several series).
    series_id = ":".join(str(value) for value in labels.values())
    moves = []
    for index in z.index[is_move]:
        date = series.loc[index, time].date().isoformat()
        moves.append({
            "id": f"move:{series_id}:{date}",
            "entity_id": labels[entity],
            "date": date,
            "prev_date": series.loc[index - 1, time].date().isoformat(),
            "value": float(series.loc[index, measure]),
            "prev_value": float(series.loc[index - 1, measure]),
            "change_pct": float(change[index] * _PERCENT),
            "z": float(z[index]),
            "direction": _UP if change[index] > 0 else _DOWN,
            "frequency": labels[frequency] if frequency else None,
            "denomination": {column: labels[column] for column in denominations},
        })
    return moves
