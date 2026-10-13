"""Event-type induction and extraction over the domain_a bulletins, with a scripted LLM."""
import json

import duckdb
import pytest
from fakes import FakeLLM

from ctxlayer.unstructured.documents import Document
from ctxlayer.unstructured.extract import extract_all, extract_document, induce_event_types
from fixtures.domain_a.extract_replies import EVENT_TYPES_REPLY, extract_reply, sentence_id
from fixtures.domain_a.make_domain import PLANTED_BULLETIN_REF, PLANTED_QUOTE

_TYPES = ["Document", "Event", "Organization", "Location", "Move", "Series", "Crop", "CropGroup"]
_SQL = """
SELECT DISTINCT bulletin_ref, CAST(released_at AS DATE), headline, digest,
       concat_ws(' ', body_part_1, body_part_2, body_part_3)
FROM farm_bulletins ORDER BY bulletin_ref
"""


@pytest.fixture(scope="module")
def docs(domain_a) -> list[Document]:
    con = duckdb.connect(str(domain_a.db_path), read_only=True)
    rows = con.execute(_SQL).fetchall()
    con.close()
    return [Document(str(ref), str(day), title, summary, body) for ref, day, title, summary, body in rows]


@pytest.fixture
def ontology() -> dict:
    return {
        "types": [{"name": name} for name in _TYPES],
        "event_types": EVENT_TYPES_REPLY["event_types"] + [{"name": "other", "description": "None fit."}],
    }


def _planted(docs: list[Document]) -> Document:
    return next(doc for doc in docs if doc.doc_id == PLANTED_BULLETIN_REF)


def _event_reply(doc: Document, **overrides):
    """Handler replying with one event whose evidence is the planted sentence (by its number)."""
    def reply(prompt, tools):
        event = {
            "type": "supply_disruption", "date": doc.date, "summary": "Drought.", "locations": [],
            "organizations": [], "affects": [], "s": sentence_id(prompt, PLANTED_QUOTE),
        }
        return json.dumps({"events": [{**event, **overrides}]})
    return reply


def _extract(doc: Document, ontology: dict, reply) -> object:
    llm = FakeLLM({"extract": reply})
    return extract_document(doc, llm, ontology["event_types"], _TYPES)


def test_planted_bulletin_event_gets_its_sentence_text_from_the_id(docs, ontology):
    result = _extract(_planted(docs), ontology, extract_reply)

    assert len(result.events) == 1
    event = result.events[0]
    assert event.type == "supply_disruption"
    assert event.evidence == PLANTED_QUOTE
    assert event.affects[0].direction == "up"
    assert event.affects[0].quote == PLANTED_QUOTE
    assert event.needs_review is False


def test_two_ids_join_their_sentences_and_out_of_range_ids_are_ignored(docs, ontology):
    doc = _planted(docs)
    def reply(prompt, tools):
        first = sentence_id(prompt, PLANTED_QUOTE)[0]
        event = {"type": "supply_disruption", "summary": "x", "s": [first, first + 1, 999]}
        return json.dumps({"events": [event]})

    evidence = _extract(doc, ontology, reply).events[0].evidence

    assert evidence.startswith(PLANTED_QUOTE + " Rainfall since February")


def test_event_without_a_valid_sentence_id_is_dropped(docs, ontology):
    result = _extract(_planted(docs), ontology, _event_reply(_planted(docs), s=[999]))

    assert result.events == []


def test_events_are_capped_and_summaries_trimmed(docs, ontology):
    long_summary = " ".join(f"w{i}" for i in range(40))

    result = _extract(_planted(docs), ontology, _event_reply(_planted(docs), summary=long_summary))
    many = _extract(_planted(docs), ontology, lambda prompt, tools: json.dumps({"events": [
        {"type": "supply_disruption", "summary": "x", "s": sentence_id(prompt, PLANTED_QUOTE)} for _ in range(9)
    ]}))

    assert len(result.events[0].summary.split()) == 20
    assert len(many.events) == 6


def test_unknown_event_type_becomes_other_and_flagged(docs, ontology):
    doc = _planted(docs)

    result = _extract(doc, ontology, _event_reply(doc, type="alien_invasion"))

    assert result.events[0].type == "other"
    assert result.events[0].needs_review is True


@pytest.mark.parametrize("bad_date", ["May 5"])
def test_malformed_event_date_uses_document_date_and_needs_review(docs, ontology, bad_date):
    doc = _planted(docs)

    result = _extract(doc, ontology, _event_reply(doc, date=bad_date))

    assert (result.events[0].date, result.events[0].needs_review) == (doc.date, True)


def test_affect_direction_downgraded_without_a_valid_sentence_id(docs, ontology):
    doc = _planted(docs)
    def reply(prompt, tools):
        affects = [
            {"entity": "Wheat (milling)", "direction": "up", "confidence": 0.9, "s": [999]},
            {"entity": "Maize", "direction": "down", "confidence": 0.7, "s": sentence_id(prompt, PLANTED_QUOTE)},
        ]
        event = {"type": "supply_disruption", "summary": "x", "affects": affects, "s": sentence_id(prompt, PLANTED_QUOTE)}
        return json.dumps({"events": [event]})

    kept = {item.entity: item for item in _extract(doc, ontology, reply).events[0].affects}

    assert (kept["Wheat (milling)"].direction, kept["Wheat (milling)"].quote) == ("unknown", "")
    assert (kept["Maize"].direction, kept["Maize"].quote) == ("down", PLANTED_QUOTE)


def test_bad_json_then_good_json_succeeds_with_two_calls(docs, ontology):
    doc = _planted(docs)
    llm = FakeLLM({"extract": ["not json at all", _event_reply(doc)]})

    result = extract_document(doc, llm, ontology["event_types"], _TYPES)

    assert len(result.events) == 1
    assert len(llm.calls) == 2
    assert "invalid" in llm.calls[1][0]


def test_extract_all_records_failure_and_keeps_other_documents(docs, ontology, tmp_path):
    def reply(prompt, tools):
        return "garbage" if PLANTED_QUOTE in prompt else extract_reply(prompt, tools)

    llm = FakeLLM({"extract": reply})

    results, failures = extract_all(docs, llm, ontology, tmp_path / "cache", workers=1)

    assert [item["doc_id"] for item in failures] == [PLANTED_BULLETIN_REF]
    assert failures[0]["error"]
    assert set(results) == {doc.doc_id for doc in docs} - {PLANTED_BULLETIN_REF}


def test_second_extract_all_run_makes_no_llm_calls(docs, ontology, tmp_path):
    cache = tmp_path / "cache"
    first_llm = FakeLLM({"extract": extract_reply})
    first, _ = extract_all(docs, first_llm, ontology, cache, workers=1)

    second_llm = FakeLLM({"extract": extract_reply})
    second, failures = extract_all(docs, second_llm, ontology, cache, workers=1)

    assert len(first_llm.calls) == len(docs)
    assert second_llm.calls == []
    assert failures == []
    assert second == first
    assert len(list(cache.glob("*.json"))) == len(docs)


def test_changed_document_text_misses_the_cache(docs, ontology, tmp_path):
    cache = tmp_path / "cache"
    extract_all(docs, FakeLLM({"extract": extract_reply}), ontology, cache, workers=1)
    edited = [Document(**{**vars(docs[0]), "body": docs[0].body + " Extra sentence."}), *docs[1:]]

    llm = FakeLLM({"extract": extract_reply})
    extract_all(edited, llm, ontology, cache, workers=1)

    assert len(llm.calls) == 1


def test_induce_event_types_appends_other_and_samples_deterministically(docs, ontology):
    llm = FakeLLM({"event_types": json.dumps(EVENT_TYPES_REPLY)})

    first = induce_event_types(docs, llm, ontology, sample=5)
    induce_event_types(docs, llm, ontology, sample=5)

    assert [item["name"] for item in first] == [
        *[item["name"] for item in EVENT_TYPES_REPLY["event_types"]], "other",
    ]
    assert llm.calls[0][0] == llm.calls[1][0]
    assert llm.calls[0][0].count("\n- ") == 5


def test_context_is_included_in_event_type_and_extraction_prompts(docs, ontology, tmp_path):
    marker = "COLLECTION-CONTEXT"
    llm = FakeLLM({"event_types": json.dumps(EVENT_TYPES_REPLY), "extract": extract_reply})

    induce_event_types(docs, llm, ontology, sample=1, context=marker)
    extract_all(docs[:1], llm, ontology, tmp_path, workers=1, context=marker)

    assert all(marker in prompt for prompt, _ in llm.calls)
