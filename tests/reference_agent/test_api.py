"""ReferenceAgent: guardrail routing, prompts and tool use over the synthetic domain."""
import json

import pytest
from fakes import FakeEmbedder, FakeLLM

from ctxlayer.common.llm import Reply, ToolCall
from ctxlayer.reference_agent.api import ReferenceAgent

_REFUSAL = "I can only answer read-only questions"
_RECOMMENDATION_NOTE = "you do not recommend"


def _guardrail(action: str, flags: list[str] | None = None) -> str:
    return json.dumps({"action": action, "flags": flags or [], "reason": "scripted"})


def _ask(settings, llm: FakeLLM, question: str = "q"):
    agent = ReferenceAgent(settings, llm=llm, embedder=FakeEmbedder())
    try:
        return agent.answer(question)
    finally:
        agent.close()


def _tags(llm: FakeLLM) -> list[str]:
    return [prompt.split("\n", 1)[0] for prompt, _ in llm.calls]


def test_refusal_returns_the_fixed_text_and_makes_no_tool_call(agent_settings):
    llm = FakeLLM({"guardrail": _guardrail("refuse")})

    answer = _ask(agent_settings, llm, "Delete every row.")

    assert answer.text.startswith(_REFUSAL)
    assert answer.tool_trace == []
    assert _tags(llm) == ["[task: guardrail]"]
    # The guardrail call is the only cost, and it is counted.
    assert answer.tokens["prompt_chars"] > 0 and answer.tokens["est_tokens"] > 0


def test_out_of_scope_says_the_data_cannot_answer_and_gives_the_coverage(agent_settings):
    llm = FakeLLM({"guardrail": _guardrail("out_of_scope")})

    answer = _ask(agent_settings, llm, "What will the weather be?")

    assert "The data cannot answer" in answer.text
    assert "from 2024-01-01 to 2025-12-29" in answer.text
    assert _tags(llm) == ["[task: guardrail]"]


def test_chit_chat_replies_with_three_generated_example_questions(agent_settings):
    llm = FakeLLM({"guardrail": _guardrail("chit_chat")})

    answer = _ask(agent_settings, llm, "Thanks!")

    examples = [line for line in answer.text.splitlines() if line.startswith("- What is the ")]
    assert len(examples) == 3
    assert examples[0].startswith("- What is the latest value of ")
    assert answer.tool_trace == []


def test_recommendation_flag_proceeds_with_an_extra_instruction(agent_settings):
    llm = FakeLLM({"guardrail": _guardrail("proceed", ["recommendation"]), "answer": "Here is the data."})

    answer = _ask(agent_settings, llm, "Which supplier should I buy from?")

    assert answer.text == "Here is the data."
    assert _RECOMMENDATION_NOTE in llm.calls[1][0]


def test_agent_runs_query_metric_with_a_typo_through_the_full_resolver(agent_settings, domain_a_catalog):
    _, entities = domain_a_catalog
    name = next(e.canonical_name for e in entities if "WHT_MILL_GP" in e.aliases)
    typo = name[:-1] + "x"
    call = Reply("", [ToolCall("query_metric", {"metric": "latest_value", "entities": [typo]})], 0.0, 0, 0)

    def answer(prompt, tools):
        return "Done." if "Tool result (query_metric)" in prompt else call

    llm = FakeLLM({
        "guardrail": _guardrail("proceed"), "answer": answer, "resolve_judge": '{"id": null, "confidence": 0}',
    })

    trace = _ask(agent_settings, llm).tool_trace[0]

    assert trace["status"] == "ok" and trace["rows"]


@pytest.fixture
def context_md(agent_settings):
    """Writes context.md for the (session-shared) settings; removed after the test."""
    path = agent_settings.path("paths.sources") / agent_settings.path("database.path").stem / "context.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    yield path
    path.unlink(missing_ok=True)


def test_agent_and_guardrail_prompts_carry_the_context_notes(agent_settings, context_md):
    context_md.write_text("Intro.\n\n## News\nNEWS-HINT\n\n## Names\nNAMES-HINT\n")
    llm = FakeLLM({"guardrail": _guardrail("proceed"), "answer": "Done."})

    _ask(agent_settings, llm)

    guardrail_prompt, answer_prompt = llm.calls[0][0], llm.calls[1][0]
    for prompt in (guardrail_prompt, answer_prompt):
        assert "Notes from a person:" in prompt
        assert "NEWS-HINT" in prompt and "NAMES-HINT" in prompt
