"""Execution check: do the answer's numbers match the reference result?"""
import pytest

from ctxlayer.evaluation.exec_check import exec_agreement, extract_numbers, key_values
from ctxlayer.evaluation.reference import RefResult

_TOL = 0.005
_MAX_VALUES = 10


def _ref(rows, error=None) -> RefResult:
    return RefResult(sql="", columns=["x"], rows=rows, error=error, reanchored=False, transpiled=False)


def _check(rows, answer, tool_rows=(), ref=None) -> dict:
    return exec_agreement(ref or _ref(rows), answer, list(tool_rows), _TOL, _MAX_VALUES)


# -------------------------------------------------------------- extract_numbers

@pytest.mark.parametrize("text, expected", [
    ("1,234.50", [(1234.5, 2, False)]),
    ("12.3%", [(12.3, 1, True)]),
    ("$1.2k", [(1200.0, -2, False)]),
    ("2.5 million", [(2500000.0, -5, False)]),
])
def test_extract_numbers_reads_value_and_decimals(text, expected):
    assert extract_numbers(text) == expected


def test_extract_numbers_does_not_treat_unit_words_as_magnitude_suffixes():
    assert extract_numbers("5 months and 3mid") == [(5.0, 0, False), (3.0, 0, False)]


# ------------------------------------------------------------------ key_values


def test_key_values_is_none_when_numeric_cells_exceed_max():
    rows = [[1, 2, 3], [4, 5, 6]]

    assert key_values(rows, 5) is None
    assert key_values(rows, 6) == [1, 2, 3, 4, 5, 6]


# --------------------------------------------------------------- exec_agreement


@pytest.mark.parametrize("reference, answer", [
    (0.1234, "It rose 12.3%."),
    (12.34, "The share is 0.1234."),
    (5.6789, "About 5.7 on average"),
    (1234.0, "Roughly $1.2k"),
])
def test_exec_agreement_passes_rounded_scaled_and_percent_forms(reference, answer):
    assert _check([[reference]], answer)["status"] == "pass"


def test_exec_agreement_fails_when_answer_number_is_off():
    assert _check([[5.6789]], "Average price 5.9")["status"] == "fail"


def test_exec_agreement_counts_matched_values_and_fails_on_a_missing_one():
    result = _check([[5.6789, 100.0]], "Average price 5.68")

    assert (result["status"], result["matched"], result["total"]) == ("fail", 1, 2)


def test_exec_agreement_reports_how_many_values_the_tool_rows_contained():
    result = _check([[5.6789, 100.0]], "5.68 and 100", tool_rows=[["a", 5.6789], ["b", 7]])

    assert (result["status"], result["matched"], result["tool_matched"]) == ("pass", 2, 1)


def test_exec_agreement_skips_when_reference_has_too_many_values():
    rows = [[float(i)] for i in range(_MAX_VALUES + 1)]

    result = _check(rows, "whatever")

    assert result["status"] == "skipped"
    assert "10" in result["reason"]


def test_exec_agreement_skips_when_reference_errored():
    assert _check([], "5", ref=_ref([], error="boom"))["status"] == "skipped"


def test_exec_agreement_skips_when_reference_has_no_numeric_values():
    assert _check([["a"], ["b"]], "a b")["status"] == "skipped"


# ----------------------------------------------- false matches (C1, I2)

@pytest.mark.parametrize("rows, answer", [
    ([[73]], "There is 1 series."),
    ([[202505]], "In 2025 prices rose."),
    ([[5]], "On 2025-05-05 it closed."),
    ([[45]], "Updated at 12:45:10."),
])
def test_exec_agreement_fails_when_answer_numbers_only_coincide_after_scaling(rows, answer):
    assert _check(rows, answer)["status"] == "fail"
