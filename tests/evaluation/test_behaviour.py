"""Behaviour rubric check for questions without reference SQL."""
import pytest

from ctxlayer.evaluation.behaviour import check_behaviour
from ctxlayer.evaluation.categorize import BEHAVIOUR_CATEGORIES
from fakes import FakeLLM


@pytest.mark.parametrize("reply, expected", [
    ('```json\n{"pass": true, "reason": "refused"}\n```', {"pass": True, "reason": "refused"}),
    ('{"pass": false, "reason": "made up a figure"}', {"pass": False, "reason": "made up a figure"}),
])
def test_behaviour_verdict_is_parsed(reply, expected):
    llm = FakeLLM({"behaviour": reply})

    assert check_behaviour(llm, "adversarial", "Delete everything", "I cannot do that.") == expected


def test_behaviour_unparsable_reply_fails():
    llm = FakeLLM({"behaviour": "looks fine to me"})

    result = check_behaviour(llm, "chit_chat", "Thanks!", "You are welcome.")

    assert result["pass"] is False
    assert result["reason"]


def test_behaviour_has_a_rubric_for_every_behaviour_category():
    llm = FakeLLM({"behaviour": '{"pass": true, "reason": "ok"}'})

    for category in BEHAVIOUR_CATEGORIES:
        assert check_behaviour(llm, category, "q", "a")["pass"] is True
