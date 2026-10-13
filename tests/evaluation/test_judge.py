"""Correctness judge and pairwise judge."""
import json

import pytest

from ctxlayer.evaluation.judge import Judge
from fakes import FakeLLM

def _pairwise_reply(winner: str) -> str:
    return json.dumps(
        {"winner": winner, "traps_a": [], "traps_b": [], "answer_diff": "none",
         "confidence": "high", "why": "because"}
    )


def _judge(*winners: str) -> tuple[Judge, FakeLLM]:
    llm = FakeLLM({"judge_pairwise": [_pairwise_reply(w) for w in winners]})
    return Judge(llm), llm


def test_correctness_parses_verdict_and_reason():
    reply = '```json\n{"verdict": "partial", "reason": "misses 2024"}\n```'
    judge = Judge(FakeLLM({"judge_correctness": reply}))

    result = judge.correctness("How many?", "5 in total", [[5]], "5 in 2023")

    assert result == {"verdict": "partial", "reason": "misses 2024"}


@pytest.mark.parametrize("reply", ["I cannot say.", '{"verdict": "great", "reason": "x"}'])
def test_correctness_garbage_or_unknown_verdict_is_unusable(reply):
    judge = Judge(FakeLLM({"judge_correctness": reply}))

    assert judge.correctness("q", "ref", None, "a")["verdict"] == "unusable"


# Questions whose coin puts ours first / the opponent first (vs "rival").
_OURS_FIRST_Q = "question 1"
_THEIRS_FIRST_Q = "question 0"


def test_pairwise_alternates_order_between_passes():
    judge, llm = _judge("tie")

    judge.pairwise(_OURS_FIRST_Q, "OURS-TEXT", "rival", "THEIRS-TEXT", passes=4)

    orders = [p.index("OURS-TEXT") < p.index("THEIRS-TEXT") for p, _ in llm.calls]
    assert orders == [True, False, True, False]


def test_pairwise_prompt_hides_contestant_names():
    judge, llm = _judge("tie")

    judge.pairwise(_OURS_FIRST_Q, "ours text", "rival", "other text", passes=1)

    prompt = llm.calls[0][0]
    assert "Answer A" in prompt and "Answer B" in prompt
    assert "rival" not in prompt


def test_pairwise_maps_ab_back_to_contestants():
    # Passes 0,2 show ours as A; passes 1,3 show ours as B.
    judge, _ = _judge("A", "B", "A", "tie")

    result = judge.pairwise(_OURS_FIRST_Q, "o", "rival", "d", passes=4)

    assert [p["winner"] for p in result["passes"]] == ["ours", "ours", "ours", "tie"]
    assert result["votes"] == {"ours": 3, "rival": 0, "tie": 1, "both_wrong": 0, "unusable": 0}
    assert result["verdict"] == "ours"


def test_pairwise_two_two_is_no_majority():
    judge, _ = _judge("A", "A", "A", "A")  # ours, opponent, ours, opponent

    result = judge.pairwise(_OURS_FIRST_Q, "o", "rival", "d", passes=4)

    assert result["votes"]["ours"] == 2 and result["votes"]["rival"] == 2
    assert result["verdict"] == "no majority"


def test_pairwise_unparsable_pass_counts_as_unusable():
    llm = FakeLLM({"judge_pairwise": ["no idea", _pairwise_reply("A"), "{}", _pairwise_reply("tie")]})

    result = Judge(llm).pairwise(_OURS_FIRST_Q, "o", "rival", "d", passes=4)

    assert result["votes"] == {"ours": 0, "rival": 1, "tie": 1, "both_wrong": 0, "unusable": 2}
    assert result["verdict"] == "no majority"


def test_pairwise_keeps_trap_flags_by_contestant_whatever_the_order():
    first = {"winner": "tie", "traps_a": ["a1", "a2"], "traps_b": []}  # ours is A
    second = {"winner": "tie", "traps_a": ["a3"], "traps_b": ["b1", "b2"]}  # ours is B
    llm = FakeLLM({"judge_pairwise": [json.dumps(first), json.dumps(second)]})

    result = Judge(llm).pairwise(_OURS_FIRST_Q, "o", "rival", "d", passes=2)

    assert result["passes"][0]["traps_ours"] == ["a1", "a2"]
    assert result["passes"][0]["traps_opponent"] == []
    assert result["passes"][1]["traps_ours"] == ["b1", "b2"]
    assert result["passes"][1]["traps_opponent"] == ["a3"]
    assert "traps_a" not in result["passes"][0]
    assert result["traps"] == {"ours": 4, "opponent": 1}


def test_order_is_a_fixed_coin_per_question():
    first_orders = []
    for index in range(200):
        judge, llm = _judge("tie")
        judge.pairwise(f"question {index}", "OURS-TEXT", "rival", "THEIRS-TEXT", passes=1)
        prompt = llm.calls[0][0]
        first_orders.append(prompt.index("OURS-TEXT") < prompt.index("THEIRS-TEXT"))

    rerun, llm = _judge("tie")
    rerun.pairwise("question 0", "OURS-TEXT", "rival", "THEIRS-TEXT", passes=1)

    # Roughly half ours-first, and a rerun of the same question keeps its order.
    assert 70 <= sum(first_orders) <= 130
    assert (llm.calls[0][0].index("OURS-TEXT") < llm.calls[0][0].index("THEIRS-TEXT")) == first_orders[0]


def test_single_pass_maps_the_letter_back_whichever_side_went_first():
    ours_first, _ = _judge("A")
    theirs_first, _ = _judge("A")

    assert ours_first.pairwise(_OURS_FIRST_Q, "o", "rival", "d", passes=1)["verdict"] == "ours"
    assert theirs_first.pairwise(_THEIRS_FIRST_Q, "o", "rival", "d", passes=1)["verdict"] == "rival"
