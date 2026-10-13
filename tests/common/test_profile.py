"""Tests for the profiler, on the synthetic domain_a database."""
import pytest

from ctxlayer.common.config import load_settings
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.profile import profile_database

_QUOTES = "grain_quotes"
_BULLETINS = "farm_bulletins"
_MAX_TEXT = 80


@pytest.fixture(scope="module")
def con(domain_a):
    connection = connect_readonly(domain_a.db_path)
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def profile(con, domain_a):
    return profile_database(con, load_settings())


def _frame(con, table):
    return con.execute(f"SELECT * FROM {table}").df()




def test_column_counts_match_pandas(profile, con):
    frame = _frame(con, _BULLETINS)
    column = profile["tables"][_BULLETINS]["columns"]["body_part_2"]

    assert column["type"] == "VARCHAR"
    assert column["null_pct"] == round(100 * frame["body_part_2"].isna().mean(), 2)
    assert column["distinct"] == frame["body_part_2"].nunique()








def test_top_values_are_counts_for_low_cardinality_columns(profile, con):
    counts = _frame(con, _QUOTES)["crop_group"].value_counts()
    column = profile["tables"][_QUOTES]["columns"]["crop_group"]

    assert column["top_values"] == [[name, int(count)] for name, count in counts.items()]


def test_high_cardinality_column_has_no_top_values(profile, con):
    assert _frame(con, _QUOTES)["quote_id"].nunique() > 200
    assert "top_values" not in profile["tables"][_QUOTES]["columns"]["quote_id"]




def test_long_strings_are_cut_in_samples_and_extremes(profile):
    column = profile["tables"][_BULLETINS]["columns"]["body_part_1"]

    assert max(len(sample) for sample in column["samples"]) <= _MAX_TEXT
    assert len(column["min"]) <= _MAX_TEXT
    assert len(column["max"]) <= _MAX_TEXT


def test_integer_period_column_looks_like_yyyymm(profile):
    columns = profile["tables"][_QUOTES]["columns"]

    assert columns["period_ym"]["looks_yyyymm"] is True
    assert columns["period_ym"]["is_integer"] is True
    assert columns["quote_id"]["looks_yyyymm"] is False
    assert columns["price_mid"]["looks_yyyymm"] is False


def test_fd_finds_child_to_parent_and_not_reverse(profile):
    fds = profile["tables"][_QUOTES]["fds"]

    assert ["crop", "crop_group"] in fds
    assert ["crop_group", "crop"] not in fds


def test_fd_ignores_pairs_that_do_not_hold(profile):
    fds = profile["tables"][_QUOTES]["fds"]

    # Each region has crops from both groups.
    assert ["market_region", "crop_group"] not in fds


def test_bulletin_ref_is_a_repeat_key_with_varying_tags(profile, con):
    frame = _frame(con, _BULLETINS)
    keys = {key["column"]: key for key in profile["tables"][_BULLETINS]["repeat_keys"]}

    varying = set(keys["bulletin_ref"]["varying_columns"])
    assert {"crop_tag", "region_tag", "row_id"} <= varying
    assert varying.isdisjoint({"headline", "digest", "outlet", "released_at"})
    assert frame["bulletin_ref"].nunique() < len(frame)






