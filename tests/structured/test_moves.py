"""Move detection on domain_a: planted jump found, flat/outlook rows quiet, min_history respected."""
import pandas as pd
import pytest
from fixtures.domain_a.make_domain import PLANTED_MOVE_DATE, PLANTED_SERIES_CODE

from ctxlayer.common.config import load_settings
from ctxlayer.common.db import connect_readonly
from ctxlayer.structured.moves import detect_moves
from ctxlayer.structured.semantic_views import open_semantic

QUOTES = "grain_quotes"
_PERCENT = 100
_DOMINANT_Z = 20  # planted z is ~56; noise stays below 4


@pytest.fixture
def sem(domain_a_semantic):
    settings, _ = domain_a_semantic
    con = open_semantic(settings)
    yield con
    con.close()


def _moves(sem, domain_a_semantic, **min_history):
    settings, semantic = domain_a_semantic
    if min_history:
        settings = load_settings(root=settings.root, overrides={**settings.raw, "structured": {"moves": {
            **settings.get("structured.moves"), **min_history}}})
    return detect_moves(sem, semantic, settings)


def test_planted_move_is_detected_with_exact_change_pct(sem, domain_a_semantic, domain_a):
    source = connect_readonly(domain_a.db_path)
    raw = source.execute(
        f"SELECT quote_date, price_mid FROM {QUOTES} WHERE series_code = ? AND basis = 'observed' "
        "ORDER BY quote_date", [PLANTED_SERIES_CODE]).df()
    source.close()
    raw["quote_date"] = pd.to_datetime(raw["quote_date"])
    expected = raw.set_index("quote_date")["price_mid"].pct_change()[pd.Timestamp(PLANTED_MOVE_DATE)] * _PERCENT

    all_moves = _moves(sem, domain_a_semantic)
    move = next(m for m in all_moves if m["entity_id"] == PLANTED_SERIES_CODE and m["date"] == PLANTED_MOVE_DATE.isoformat())

    # Ordinary noise also crosses |z| 2 now and then; the planted jump must dominate all of it.
    assert move["z"] == max(m["z"] for m in all_moves)
    assert move["z"] > _DOMINANT_Z
    assert move["change_pct"] == pytest.approx(expected)
    assert move["direction"] == "up"
    assert abs(move["z"]) >= 2
    assert move["id"] == f"move:{PLANTED_SERIES_CODE}:weekly:USD:per_tonne:{PLANTED_MOVE_DATE.isoformat()}"
    assert move["frequency"] == "weekly"
    assert move["denomination"] == {"currency": "USD", "unit": "per_tonne"}
    assert move["value"] / move["prev_value"] - 1 == pytest.approx(move["change_pct"] / _PERCENT)


def test_min_change_filters_small_outliers(sem, domain_a_semantic):
    settings, _ = domain_a_semantic
    floor = settings.get("structured.moves.min_change_pct")
    moves = {m["id"]: m for m in _moves(sem, domain_a_semantic)}
    noisy = {m["id"]: m for m in _moves(sem, domain_a_semantic, min_change_pct=0.0)}

    # Default keeps the planted jump and drops every sub-floor outlier the unfiltered run finds.
    assert any(m["entity_id"] == PLANTED_SERIES_CODE and m["date"] == str(PLANTED_MOVE_DATE) for m in moves.values())
    assert set(moves) < set(noisy)
    assert all(abs(m["change_pct"]) >= floor for m in moves.values())
    assert all(abs(noisy[i]["change_pct"]) < floor for i in set(noisy) - set(moves))


def test_outlook_rows_never_produce_moves(sem, domain_a_semantic, domain_a):
    source = connect_readonly(domain_a.db_path)
    first_outlook = source.execute(f"SELECT min(quote_date) FROM {QUOTES} WHERE basis = 'outlook'").fetchone()[0]
    source.close()

    moves = _moves(sem, domain_a_semantic)

    assert moves and all(m["date"] < first_outlook.isoformat() for m in moves)


def test_high_min_history_yields_no_moves(sem, domain_a_semantic):
    assert _moves(sem, domain_a_semantic, min_history=1000, trailing_periods=1000) == []

