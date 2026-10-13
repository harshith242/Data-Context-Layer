"""Adversarial edge cases for the cost changes: catalog-name mentions, sentence-id evidence, compact replies."""
import json

import pytest
from fakes import FakeLLM

from ctxlayer.common.catalog import Entity
from ctxlayer.unstructured.documents import Document
from ctxlayer.unstructured.extract import extract_document
from ctxlayer.unstructured.mentions import find_mentions

_TYPES = ["Item", "Location"]
_EVENT_TYPES = [{"name": "supply_cut", "description": "Supply falls."}, {"name": "other", "description": "None fit."}]
_BODY = "Mills in North Vale shut for two weeks. Output fell by a fifth. Buyers expect higher prices."


def _doc(body: str = _BODY, title: str = "Strike hits mills", summary: str = "") -> Document:
    return Document("D1", "2025-04-25", title, summary, body)


def _ids(text: str, entities: list[Entity]) -> list[str]:
    return [m["entity_id"] for m in find_mentions([_doc(text, title="", summary="")], entities)["D1"]]


def _extract(reply, doc: Document | None = None):
    llm = FakeLLM({"extract": reply})
    return extract_document(doc or _doc(), llm, _EVENT_TYPES, _TYPES), llm


def _event(**overrides) -> dict:
    return {"type": "supply_cut", "date": "2025-04-25", "summary": "Mills shut.", "s": [2], **overrides}


# ------------------------------------------------------------ idea 1: mentions

def test_accented_text_matches_plain_alias_and_back():
    entities = [Entity("place:sao", "Location", "Sao Vale"), Entity("place:koln", "Location", "Köln Port")]

    assert _ids("Rain in São Vale and Koln Port.", entities) == ["place:koln", "place:sao"]


def test_common_english_words_are_not_mentions():
    entities = [Entity("place:may", "Location", "May"), Entity("item:will", "Item", "Will"), Entity("item:grain", "Item", "Grain")]

    assert _ids("Prices may rise and buyers will wait for grain.", entities) == ["item:grain"]


def test_nested_name_is_not_double_counted():
    entities = [Entity("place:ny", "Location", "New Vale"), Entity("place:v", "Location", "Vale")]

    assert _ids("Ships left New Vale today.", entities) == ["place:ny"]
    assert _ids("Ships left Vale today.", entities) == ["place:v"]


# --------------------------------------------------- idea 2: sentence-id evidence

@pytest.mark.parametrize("ids", [["[S2]"], 2])
def test_sentence_ids_in_loose_formats_still_resolve(ids):
    result, _ = _extract(json.dumps({"events": [_event(s=ids)]}))

    assert result.events[0].evidence == "Mills in North Vale shut for two weeks."


def test_duplicate_ids_paste_a_sentence_once():
    result, _ = _extract(json.dumps({"events": [_event(s=[2, 2, 3])]}))

    assert result.events[0].evidence == "Mills in North Vale shut for two weeks. Output fell by a fifth."


def test_old_verbatim_format_is_retried_not_silently_dropped():
    old = json.dumps({"events": [{"type": "supply_cut", "summary": "x", "evidence": "Mills in North Vale shut for two weeks."}]})
    new = json.dumps({"events": [_event()]})

    result, llm = _extract([old, new])

    assert len(llm.calls) == 2 and "sentence number" in llm.calls[1][0]
    assert result.events[0].evidence == "Mills in North Vale shut for two weeks."


def test_sentences_keep_decimals_and_cover_all_text():
    doc = _doc("Output fell 3.5 percent. U.S. buyers waited. Prices rose 1,200.5 points!")
    prompt_holder = {}

    def reply(prompt, tools):
        prompt_holder["p"] = prompt
        return json.dumps({"events": [_event(s=[2])]})

    _extract(reply, doc)
    numbered = [line.split("] ", 1)[1] for line in prompt_holder["p"].splitlines() if line.startswith("[S")]

    assert "Output fell 3.5 percent." in numbered
    assert " ".join(numbered).replace(" ", "") == ("Strike hits mills" + doc.body).replace(" ", "")


# ------------------------------------------------------ idea 3: compact replies

def test_fenced_and_pretty_printed_replies_still_parse():
    fenced = "```json\n" + json.dumps({"events": [_event()]}, indent=2) + "\n```"

    result, _ = _extract(fenced)

    assert len(result.events) == 1

