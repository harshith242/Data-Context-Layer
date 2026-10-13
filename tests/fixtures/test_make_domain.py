"""Tests for the synthetic domain fixture: structure, planted facts, questions file."""
import datetime as dt

import duckdb
import pandas as pd
import pytest

from fixtures.domain_a.make_domain import (
    PLANTED_BULLETIN_REF,
    PLANTED_MOVE_DATE,
    PLANTED_QUOTE,
    PLANTED_SERIES_CODE,
)

_MOVE = pd.Timestamp(PLANTED_MOVE_DATE)


def _planted_mid(quotes: pd.DataFrame) -> pd.Series:
    planted = quotes[quotes.series_code == PLANTED_SERIES_CODE]
    return planted.assign(quote_date=pd.to_datetime(planted.quote_date)).set_index("quote_date").price_mid


@pytest.fixture(scope="module")
def con(domain_a):
    connection = duckdb.connect(str(domain_a.db_path), read_only=True)
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def quotes(con) -> pd.DataFrame:
    return con.execute("SELECT * FROM grain_quotes ORDER BY series_code, quote_date").df()


@pytest.fixture(scope="module")
def bulletins(con) -> pd.DataFrame:
    frame = con.execute("SELECT * FROM farm_bulletins ORDER BY row_id").df()
    frame["released_at"] = frame.released_at.dt.tz_convert("UTC")
    return frame


@pytest.fixture(scope="module")
def questions(domain_a) -> pd.DataFrame:
    return pd.read_csv(domain_a.questions_csv)


def test_planted_move_is_a_fifteen_percent_weekly_jump_with_z_at_least_two(quotes):
    mid = _planted_mid(quotes)
    change = mid.pct_change()
    history = change[change.index < _MOVE].dropna()
    z_score = (change[_MOVE] - history.mean()) / history.std()

    assert change[_MOVE] == pytest.approx(0.15, abs=0.015)
    assert z_score >= 2


def test_bulletin_ref_repeats_once_per_tag_combination(bulletins):
    per_ref = bulletins.groupby("bulletin_ref").size()
    assert bulletins.bulletin_ref.nunique() == 20
    assert len(bulletins) > bulletins.bulletin_ref.nunique()

    for ref, group in bulletins.groupby("bulletin_ref"):
        combos = group.crop_tag.nunique() * group.region_tag.nunique()
        assert per_ref[ref] == combos
        assert group.headline.nunique() == 1 and group.digest.nunique() == 1


def test_tag_spellings_differ_and_some_tags_have_no_counterpart(bulletins, quotes):
    crops, regions = set(quotes.crop), set(quotes.market_region)
    assert not set(bulletins.crop_tag) & crops
    assert not set(bulletins.region_tag) & regions
    assert "Wheat (milling)" in set(bulletins.crop_tag)
    assert "Sunflower" in set(bulletins.crop_tag)
    assert "Nile Delta" in set(bulletins.region_tag)


def test_planted_drought_bulletin_carries_quote_for_wheat_in_great_plains(bulletins):
    planted = bulletins[bulletins.bulletin_ref == PLANTED_BULLETIN_REF]
    assert set(planted.released_at.dt.date) == {dt.date(2025, 4, 25)}
    assert set(planted.crop_tag) == {"Wheat (milling)"} and set(planted.region_tag) == {"GP"}
    assert PLANTED_QUOTE in planted.body_part_1.iloc[0]
    assert "drought" in planted.headline.iloc[0].lower()


def test_every_reference_query_runs_and_its_answer_quotes_the_result(con, questions):
    for _, row in questions.dropna(subset=["reference_sql"]).iterrows():
        result = con.execute(row["reference_sql"]).fetchall()
        assert result
        assert str(result[0][0]) in row["reference_answer"]
