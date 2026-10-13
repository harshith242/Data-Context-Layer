"""Execution check: are the reference result's numbers present in the answer text?

Signs are ignored: direction ("fell 5.2%") is left to the correctness judge.
"""
import re
from decimal import Decimal
from enum import Enum
from typing import NamedTuple

from ctxlayer.evaluation.reference import RefResult, numeric_cells

# "1,234.50" -> 1,234 + .50; optional magnitude suffix "1.2k", "2.5 million", "3bn"; optional "%" / "percent".
_NUMBER = re.compile(
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?"
    r"(?:(?P<word>\s?(?:thousand|million|billion))(?![a-z])|(?P<abbr>bn|k|m|b)(?![a-z]))?"
    r"(?P<percent>\s?(?:%|percent(?![a-z])))?",
    re.IGNORECASE,
)

# Dates and times are not figures: "2025-05-05", "2025-05", "12:45:10".
_DATE_OR_TIME = re.compile(
    r"(?<!\d)\d{4}-\d{2}(?:-\d{2})?(?!\d)|(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)"
)
_MAGNITUDE_EXPONENT = {"k": 3, "thousand": 3, "m": 6, "million": 6, "b": 9, "bn": 9, "billion": 9}

_PERCENT = 100

# Tool-result cells are exact, so only the relative tolerance applies.
_EXACT_DECIMALS = 12

# Slack so a value on a rounding boundary (5.65 -> "5.7") still matches.
_BOUNDARY_SLACK = 1e-9


class ExecStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


class _Candidate(NamedTuple):
    value: float
    decimals: int
    is_percent: bool


def extract_numbers(text: str) -> list[tuple[float, int, bool]]:
    """(value, decimals, is_percent) per number in `text`; sign, dates and times ignored.

    "$1.2k" -> (1200.0, -2, False); "12.3%" -> (12.3, 1, True).
    """
    found = []
    for match in _NUMBER.finditer(_DATE_OR_TIME.sub(" ", text)):
        frac = match.group("frac") or ""
        suffix = (match.group("word") or match.group("abbr") or "").strip().lower()
        exponent = _MAGNITUDE_EXPONENT.get(suffix, 0)

        # Decimal keeps 1.2 * 10^3 exactly 1200.
        digits = Decimal(f"{match.group('int').replace(',', '')}.{frac or '0'}")
        found.append(_Candidate(float(digits.scaleb(exponent)), len(frac) - exponent, bool(match.group("percent"))))
    return found


def key_values(rows: list[list], max_values: int) -> list[float] | None:
    """Numeric cells of `rows` in order; None when there are more than `max_values`."""
    values = numeric_cells(rows)
    if len(values) > max_values:
        return None
    return values


def exec_agreement(
    ref: RefResult | None, answer_text: str, tool_rows: list[list],
    rel_tol: float, max_values: int,
) -> dict:
    """Pass when every reference number appears in the answer; `tool_matched` counts those in tool rows."""
    if ref is None:
        return _result(ExecStatus.SKIPPED, 0, 0, 0, "no reference result")
    if ref.error:
        return _result(ExecStatus.SKIPPED, 0, 0, 0, "reference query failed")

    values = key_values(ref.rows, max_values)
    if values is None:
        return _result(ExecStatus.SKIPPED, 0, 0, 0, f"more than {max_values} reference values")
    if not values:
        return _result(ExecStatus.SKIPPED, 0, 0, 0, "reference has no numeric values")

    answer_numbers = extract_numbers(answer_text)
    tool_numbers = [_Candidate(cell, _EXACT_DECIMALS, False) for cell in numeric_cells(tool_rows)]
    matched = sum(_is_found(value, answer_numbers, rel_tol) for value in values)
    tool_matched = sum(_is_found(value, tool_numbers, rel_tol) for value in values)

    if matched == len(values):
        return _result(ExecStatus.PASS, matched, len(values), tool_matched, "all reference values found")
    return _result(
        ExecStatus.FAIL, matched, len(values), tool_matched,
        f"{len(values) - matched} of {len(values)} reference values not in answer",
    )


def _result(status: ExecStatus, matched: int, total: int, tool_matched: int, reason: str) -> dict:
    return {
        "status": status.value, "matched": matched, "total": total,
        "tool_matched": tool_matched, "reason": reason,
    }


def _is_found(value: float, candidates: list[_Candidate], rel_tol: float) -> bool:
    # Tolerance: relative, or half a unit of the candidate's last digit (5.7 covers 5.65..5.75).
    for candidate in candidates:
        for scale in _scales(candidate):
            tolerance = max(rel_tol * abs(value), 0.5 * 10 ** -candidate.decimals * scale) + _BOUNDARY_SLACK
            if abs(abs(value) - candidate.value * scale) <= tolerance:
                return True
    return False


def _scales(candidate: _Candidate) -> tuple[float, ...]:
    # "12.3%" may answer a fraction reference (0.123); "0.123" may answer a percent reference (12.3).
    # Any other integer ("3 regions") only stands for itself.
    if candidate.is_percent:
        return (1, 1 / _PERCENT)
    if candidate.decimals > 0 and abs(candidate.value) <= 1:
        return (1, _PERCENT)
    return (1,)
