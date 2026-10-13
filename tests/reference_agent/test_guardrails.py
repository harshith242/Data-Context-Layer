"""precheck routing with a scripted LLM."""
import json

import pytest

from ctxlayer.common.llm import LLMError
from ctxlayer.reference_agent.guardrails import Action, precheck
from fakes import FakeLLM


def _llm(reply: str) -> FakeLLM:
    return FakeLLM({"guardrail": reply})


def test_llm_error_proceeds_instead_of_raising():
    def down(prompt, tools):
        raise LLMError("llm down")

    result = precheck("q", FakeLLM({"guardrail": down}))

    assert result["action"] == Action.PROCEED.value
    assert "llm down" in result["reason"]


def test_unknown_flags_are_dropped():
    reply = json.dumps({"action": "proceed", "flags": ["recommendation", "mystery"], "reason": ""})

    assert precheck("q", _llm(reply))["flags"] == ["recommendation"]


@pytest.mark.parametrize("reply", ["not json at all", '{"action": "explode"}'])
def test_unparseable_reply_proceeds_and_records_reason(reply):
    result = precheck("q", _llm(reply))

    assert result["action"] == Action.PROCEED.value
    assert result["flags"] == []
    assert result["reason"]
