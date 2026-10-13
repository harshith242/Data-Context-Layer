"""LLM judges: answer correctness and blind pairwise comparison."""
import hashlib
import json

from ctxlayer.common._names import NO_MAJORITY, OURS, UNUSABLE
from ctxlayer.common.llm import LLM, extract_json

_CORRECTNESS_VERDICTS = ("correct", "partial", "incorrect", "unusable")
_TIE = "tie"
_BOTH_WRONG = "both_wrong"
_SIDE_A = "A"
_SIDE_B = "B"
_WINNERS = (_SIDE_A, _SIDE_B, _TIE, _BOTH_WRONG)
_MAX_REFERENCE_ROWS = 20
_DEFAULT_PASSES = 4

_CORRECTNESS_PROMPT = """\
[task: judge_correctness]
Judge whether the answer is correct for the question, using the reference.

Question: {question}

Reference answer: {ref_answer}

Reference rows (JSON, at most {max_rows}):
{ref_rows}

Answer to judge: {answer}

Verdicts: correct (matches the reference), partial (some of it is right or incomplete),
incorrect (wrong), unusable (no real answer).
Reply with ONLY a JSON object:
{{"verdict": "correct|partial|incorrect|unusable", "reason": "<one sentence>"}}"""
_PAIRWISE_PROMPT = """\
[task: judge_pairwise]
Two answers to the same question. Decide which is better, or whether they tie
or both are wrong.

Question: {question}

Answer A:
{first}

Answer B:
{second}

Reply with ONLY a JSON object:
{{"winner": "A|B|tie|both_wrong", "traps_a": [<problems in A>], "traps_b": [<problems in B>],
"answer_diff": "timing|material|none", "confidence": "high|medium|low", "why": "<one sentence>"}}"""


class Judge:
    def __init__(self, llm: LLM):
        self._llm = llm

    def correctness(
        self, question: str, ref_answer: str, ref_rows: list[list] | None, answer: str
    ) -> dict:
        """{"verdict": correct|partial|incorrect|unusable, "reason": str}."""
        prompt = _CORRECTNESS_PROMPT.format(
            question=question,
            ref_answer=ref_answer,
            max_rows=_MAX_REFERENCE_ROWS,
            ref_rows=json.dumps((ref_rows or [])[:_MAX_REFERENCE_ROWS], default=str),
            answer=answer,
        )
        value = extract_json(self._llm.complete(prompt).text)

        if not isinstance(value, dict) or value.get("verdict") not in _CORRECTNESS_VERDICTS:
            return {"verdict": UNUSABLE, "reason": "judge reply not parsable"}
        return {"verdict": value["verdict"], "reason": str(value.get("reason", ""))}

    def pairwise(
        self,
        question: str,
        ours: str,
        opponent_label: str,
        opponent: str,
        passes: int = _DEFAULT_PASSES,
    ) -> dict:
        """{"verdict": "ours"|opponent_label|"no majority", "votes": {..}, "passes": [..]}."""
        results = []

        # Coin per question + opponent picks who goes first (ours first on about half the questions, same order
        # on every rerun); later passes alternate from there.
        start = hashlib.sha256(f"{question}|{opponent_label}".encode()).digest()[0] % 2
        for index in range(passes):
            ours_first = (start + index) % 2 == 0
            first, second = (ours, opponent) if ours_first else (opponent, ours)
            prompt = _PAIRWISE_PROMPT.format(question=question, first=first, second=second)
            reply = extract_json(self._llm.complete(prompt).text)
            results.append(_pass_result(reply, ours_first, opponent_label))

        votes = dict.fromkeys((OURS, opponent_label, _TIE, _BOTH_WRONG, UNUSABLE), 0)
        for result in results:
            votes[result["winner"]] += 1

        return {
            "verdict": _verdict(votes, opponent_label, passes),
            "votes": votes,
            "traps": {
                OURS: sum(len(result.get("traps_ours", [])) for result in results),
                "opponent": sum(len(result.get("traps_opponent", [])) for result in results),
            },
            "passes": results,
        }


def _pass_result(reply, ours_first: bool, opponent_label: str) -> dict:
    """One pass with the A/B winner mapped to a contestant: A when ours is first -> "ours"."""
    if not isinstance(reply, dict) or reply.get("winner") not in _WINNERS:
        return {"winner": UNUSABLE, "ours_first": ours_first}

    ours_letter = _SIDE_A if ours_first else _SIDE_B
    winner = reply["winner"]
    if winner in (_SIDE_A, _SIDE_B):
        winner = OURS if winner == ours_letter else opponent_label

    # Trap flags name A / B in the reply; keep them by contestant.
    traps_a, traps_b = _trap_list(reply, "traps_a"), _trap_list(reply, "traps_b")
    traps_ours, traps_opponent = (traps_a, traps_b) if ours_first else (traps_b, traps_a)
    kept = {key: value for key, value in reply.items() if key not in ("traps_a", "traps_b")}

    return {
        **kept, "winner": winner, "ours_first": ours_first,
        "traps_ours": traps_ours, "traps_opponent": traps_opponent,
    }


def _trap_list(reply: dict, key: str) -> list:
    value = reply.get(key)
    return value if isinstance(value, list) else []


def _verdict(votes: dict, opponent_label: str, passes: int) -> str:
    """The side with more than half of the passes, else "no majority"."""
    for side in (OURS, opponent_label):
        if votes[side] > passes / 2:
            return side
    return NO_MAJORITY
