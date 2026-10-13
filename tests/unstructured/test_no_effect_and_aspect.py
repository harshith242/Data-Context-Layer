"""Extraction prompt v3: empty answers are allowed, effects need confidence, and every affect says what changes."""
import json

import pytest

from fakes import FakeLLM

from ctxlayer.unstructured.documents import Document
from ctxlayer.unstructured.extract import extract_document

_TYPES = ["Item"]
_EVENT_TYPES = [{"name": "supply_cut", "description": "Supply falls."}, {"name": "other", "description": "None fit."}]
_DOC = Document("D1", "2025-04-25", "Strike hits mills", "", "Output fell by a fifth. Buyers expect higher prices.")


def _extract(reply):
    llm = FakeLLM({"extract": reply})
    return extract_document(_DOC, llm, _EVENT_TYPES, _TYPES), llm


def _event(affects: list[dict]) -> str:
    return json.dumps({"events": [{"type": "supply_cut", "summary": "Strike", "s": [1], "affects": affects}]})


def _affect(**fields) -> dict:
    return {"entity": "Gadget", "aspect": "price", "direction": "up", "confidence": 0.9, "s": [3], **fields}


def test_an_empty_answer_is_accepted_without_a_retry():
    result, llm = _extract('{"events":[]}')

    assert result.events == [] and len(llm.calls) == 1


def test_effects_weaker_than_implied_are_dropped():
    affects = [_affect(entity="A", confidence=0.59), _affect(entity="B", confidence=0.6), _affect(entity="C", confidence=0.9)]

    result, _ = _extract(_event(affects))

    assert [a.entity for a in result.events[0].affects] == ["B", "C"]


def test_aspect_is_kept_and_a_missing_aspect_is_empty():
    no_aspect = {key: value for key, value in _affect(entity="B").items() if key != "aspect"}

    result, _ = _extract(_event([_affect(entity="A", aspect="output", direction="down", s=[2]), no_aspect]))
    kept = {a.entity: (a.aspect, a.direction, a.quote) for a in result.events[0].affects}

    assert kept["A"] == ("output", "down", "Output fell by a fifth.")
    assert kept["B"] == ("", "up", "Buyers expect higher prices.")


# ------------------------------------------------- status and event dates (prompt v4)

def _dated(**fields) -> str:
    return json.dumps({"events": [{"type": "supply_cut", "summary": "Strike", "s": [1], **fields}]})


@pytest.mark.parametrize("status", ["forecast"])
def test_status_is_kept(status):
    result, _ = _extract(_dated(status=status))

    assert (result.events[0].status, result.events[0].needs_review) == (status, False)


def test_missing_status_means_reported_and_an_unknown_one_is_flagged():
    missing, _ = _extract(_dated())
    odd, _ = _extract(_dated(status="rumoured"))

    assert (missing.events[0].status, missing.events[0].needs_review) == ("reported", False)
    assert (odd.events[0].status, odd.events[0].needs_review) == ("reported", True)


def test_a_stated_event_date_is_kept_and_marked_stated():
    result, _ = _extract(_dated(date="2025-03-02"))

    assert (result.events[0].date, result.events[0].date_source) == ("2025-03-02", "stated")


@pytest.mark.parametrize("date", [""])
def test_no_event_date_falls_back_to_the_document_date_and_says_so(date):
    result, _ = _extract(_dated(date=date))

    assert (result.events[0].date, result.events[0].date_source) == (_DOC.date, "publication")

