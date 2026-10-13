"""Tests for role detection, with a scripted LLM on the synthetic domain_a database."""
import copy
import json

import pytest

from ctxlayer.common.config import load_settings
from ctxlayer.common.context_files import ContextFiles, load_context_files
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.llm import LLMError
from ctxlayer.common.profile import profile_database
from ctxlayer.common.roles import detect_roles, heuristic_roles
from fakes import FakeLLM
from fixtures.domain_a.roles_reply import ROLES_REPLIES

_QUOTES = "grain_quotes"
_BULLETINS = "farm_bulletins"
_NO_CONTEXT = ContextFiles()


@pytest.fixture(scope="module")
def profile(domain_a):
    con = connect_readonly(domain_a.db_path)
    result = profile_database(con, load_settings())
    con.close()
    return result


def _llm(replies: dict):
    """FakeLLM answering each table's prompt with replies[table] (a dict, or raw text)."""

    def answer(prompt, tools):
        table = prompt.split("Table: ", 1)[1].split("\n", 1)[0]
        reply = replies[table]
        return reply if isinstance(reply, str) else json.dumps(reply)

    return FakeLLM({"roles": answer})


def _detect(profile, replies, tmp_settings, context=_NO_CONTEXT, llm=None):
    return detect_roles(profile, llm or _llm(replies), context, tmp_settings())


def _columns(roles, table):
    return roles["tables"][table]["columns"]


# ------------------------------------------------------------ heuristics

def test_heuristics_assign_roles_from_the_profile(profile):
    quotes = heuristic_roles(profile["tables"][_QUOTES])
    bulletins = heuristic_roles(profile["tables"][_BULLETINS])

    assert quotes["quote_id"]["role"] == "row_id"
    assert quotes["period_ym"]["role"] == "period_key"
    assert quotes["price_mid"]["role"] == "measure"
    assert quotes["quote_date"]["role"] == "time_business"
    assert quotes["crop_group"]["role"] == "dimension"
    assert bulletins["digest"]["role"] == "text_body"


def test_heuristics_find_doc_id_and_doc_tags_from_repeat_keys(profile):
    bulletins = heuristic_roles(profile["tables"][_BULLETINS])

    assert bulletins["bulletin_ref"]["role"] == "doc_id"
    assert bulletins["crop_tag"]["role"] == "doc_tag"
    assert bulletins["region_tag"]["role"] == "doc_tag"
    assert bulletins["outlet"]["role"] == "dimension"






# ------------------------------------------------------------ LLM replies

def test_valid_reply_gives_roles_logical_names_and_no_review(profile, tmp_settings):
    roles = _detect(profile, ROLES_REPLIES, tmp_settings)

    quotes = _columns(roles, _QUOTES)
    assert quotes["price_low"]["role"] == "measure_band"
    assert quotes["price_low"]["band"] == "low"
    assert quotes["price_low"]["additive"] is False
    assert quotes["revised_date"]["role"] == "time_other"
    assert quotes["revised_date"]["meaning"] == "last revision of the quote"
    assert quotes["crop"]["logical_name"] == "crop"
    assert set(quotes) == set(profile["tables"][_QUOTES]["columns"])
    assert not any(item["needs_review"] for item in quotes.values())
    assert {item["source"] for item in quotes.values()} == {"inferred"}


def test_table_kind_primary_measure_and_scenario_values_are_recorded(profile, tmp_settings):
    tables = _detect(profile, ROLES_REPLIES, tmp_settings)["tables"]

    assert tables[_QUOTES]["kind"] == "measures"
    assert tables[_QUOTES]["primary_measure"] == "price_mid"
    assert tables[_QUOTES]["scenario_values"] == {"observed": "actual", "outlook": "projected"}
    assert tables[_BULLETINS]["kind"] == "text"
    assert "primary_measure" not in tables[_BULLETINS]






def test_garbage_reply_falls_back_to_heuristics_marked_for_review(profile, tmp_settings):
    replies = {_QUOTES: "I cannot help with that.", _BULLETINS: "{not json"}

    roles = _detect(profile, replies, tmp_settings)

    quotes = _columns(roles, _QUOTES)
    assert quotes["quote_id"]["role"] == "row_id"
    assert quotes["price_mid"]["role"] == "measure"
    assert all(item["needs_review"] for table in roles["tables"].values() for item in table["columns"].values())
    assert all(item["confidence"] <= 0.5 for table in roles["tables"].values() for item in table["columns"].values())
    assert roles["tables"][_QUOTES]["kind"] == "measures"
    assert roles["tables"][_BULLETINS]["kind"] == "text"


def test_llm_failure_falls_back_without_stopping_the_build(profile, tmp_settings):
    def fail(prompt, tools):
        raise LLMError("llm down")

    roles = _detect(profile, None, tmp_settings, llm=FakeLLM({"roles": fail}))

    assert set(roles["tables"]) == {_QUOTES, _BULLETINS}
    assert _columns(roles, _QUOTES)["quote_id"]["needs_review"] is True


def test_invalid_role_name_falls_back_for_that_column_only(profile, tmp_settings):
    replies = copy.deepcopy(ROLES_REPLIES)
    replies[_QUOTES]["columns"]["price_mid"]["role"] = "price"

    quotes = _columns(_detect(profile, replies, tmp_settings), _QUOTES)

    assert quotes["price_mid"]["role"] == "measure"
    assert quotes["price_mid"]["needs_review"] is True
    assert quotes["price_mid"]["confidence"] <= 0.5
    assert quotes["price_low"]["role"] == "measure_band"
    assert quotes["price_low"]["needs_review"] is False


def test_missing_column_falls_back_to_heuristic(profile, tmp_settings):
    replies = copy.deepcopy(ROLES_REPLIES)
    del replies[_QUOTES]["columns"]["period_ym"]

    quotes = _columns(_detect(profile, replies, tmp_settings), _QUOTES)

    assert quotes["period_ym"]["role"] == "period_key"
    assert quotes["period_ym"]["needs_review"] is True


def test_duplicate_logical_name_falls_back_for_the_later_column(profile, tmp_settings):
    replies = copy.deepcopy(ROLES_REPLIES)
    replies[_QUOTES]["columns"]["price_high"]["logical_name"] = "price_low"

    quotes = _columns(_detect(profile, replies, tmp_settings), _QUOTES)

    assert quotes["price_low"]["logical_name"] == "price_low"
    assert quotes["price_high"]["needs_review"] is True
    names = [item["logical_name"] for item in quotes.values()]
    assert len(names) == len(set(names))




def test_confidence_below_threshold_needs_review(profile, tmp_settings):
    replies = copy.deepcopy(ROLES_REPLIES)
    replies[_QUOTES]["columns"]["crop"]["confidence"] = 0.79
    replies[_QUOTES]["columns"]["crop_group"]["confidence"] = 0.8

    quotes = _columns(_detect(profile, replies, tmp_settings), _QUOTES)

    assert quotes["crop"]["needs_review"] is True
    assert quotes["crop_group"]["needs_review"] is False


def test_llm_overruling_a_strong_heuristic_is_kept_but_flagged(profile, tmp_settings):
    replies = copy.deepcopy(ROLES_REPLIES)
    replies[_QUOTES]["columns"]["quote_id"]["role"] = "entity_id"

    quote_id = _columns(_detect(profile, replies, tmp_settings), _QUOTES)["quote_id"]

    assert quote_id["role"] == "entity_id"
    assert quote_id["confidence"] == 0.95
    assert quote_id["needs_review"] is True


def test_two_business_times_keep_the_more_confident_one(profile, tmp_settings):
    replies = copy.deepcopy(ROLES_REPLIES)
    replies[_QUOTES]["columns"]["revised_date"]["role"] = "time_business"
    replies[_QUOTES]["columns"]["revised_date"]["confidence"] = 0.85

    quotes = _columns(_detect(profile, replies, tmp_settings), _QUOTES)

    assert quotes["quote_date"]["role"] == "time_business"
    assert quotes["revised_date"]["role"] == "time_other"
    assert quotes["revised_date"]["needs_review"] is True


def test_primary_measure_not_a_measure_column_is_replaced(profile, tmp_settings):
    replies = copy.deepcopy(ROLES_REPLIES)
    replies[_QUOTES]["primary_measure"] = "crop"

    table = _detect(profile, replies, tmp_settings)["tables"][_QUOTES]

    assert table["primary_measure"] == "price_mid"


def test_unknown_scenario_label_is_dropped_and_scenario_flagged(profile, tmp_settings):
    replies = copy.deepcopy(ROLES_REPLIES)
    replies[_QUOTES]["scenario_values"] = {"observed": "actual", "outlook": "guess"}

    roles = _detect(profile, replies, tmp_settings)
    quotes = _columns(roles, _QUOTES)

    # The rest of the reply is kept; only the bad label goes, and the scenario column asks for review.
    assert roles["tables"][_QUOTES]["scenario_values"] == {"observed": "actual"}
    assert quotes["basis"]["role"] == "scenario" and quotes["basis"]["needs_review"]
    assert not quotes["series_code"].get("needs_review")


# ------------------------------------------------------------ context.md

def _context_settings(tmp_settings):
    settings = tmp_settings()
    folder = settings.path("paths.sources") / "domain_a"
    folder.mkdir(parents=True)
    (folder / "grain_quotes.context.md").write_text("ZEBRA-MARKER: basis holds observed and outlook rows.")
    return settings


def test_context_text_reaches_only_the_prompt_of_its_table(profile, tmp_settings):
    settings = _context_settings(tmp_settings)
    llm = _llm(ROLES_REPLIES)

    detect_roles(profile, llm, load_context_files(settings, "domain_a"), settings)

    quotes_prompt = next(p for p, _ in llm.calls if "Table: grain_quotes" in p)
    bulletins_prompt = next(p for p, _ in llm.calls if "Table: farm_bulletins" in p)
    assert "ZEBRA-MARKER" in quotes_prompt
    assert "ZEBRA-MARKER" not in bulletins_prompt


def test_column_credited_to_context_gets_context_source(profile, tmp_settings):
    settings = _context_settings(tmp_settings)
    replies = copy.deepcopy(ROLES_REPLIES)
    replies[_QUOTES]["columns"]["basis"]["from_context"] = True

    roles = detect_roles(profile, _llm(replies), load_context_files(settings, "domain_a"), settings)

    quotes = _columns(roles, _QUOTES)
    assert quotes["basis"]["source"] == "context.md"
    assert quotes["currency"]["source"] == "inferred"




# ------------------------------------------------------------ review + helper



