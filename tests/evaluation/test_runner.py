"""Runner: resumable answering and routed scoring on the synthetic domain."""
import threading
import time
import dataclasses
import json

import pytest

from ctxlayer.common.db import connect_readonly
from ctxlayer.common.llm import LLMError
from ctxlayer.evaluation.judge import Judge
from ctxlayer.evaluation.questions import load_questions
from ctxlayer.evaluation.reference import build_references
from ctxlayer.evaluation.runner import run_config, score_config
from ctxlayer.reference_agent.loop import Answer
from fakes import FakeLLM

_OATS_Q, _DRIVERS_Q, _ADVERSARIAL_Q = "q001", "q004", "q005"
_PAIRWISE_PASSES = 1  # settings eval.pairwise_passes
_JSON_CORRECT = '{"verdict": "correct", "reason": "matches"}'
_TIE = '{"winner": "tie"}'


@pytest.fixture
def settings(tmp_settings, domain_a):
    return tmp_settings({"database": {"path": str(domain_a.db_path)}})


@pytest.fixture
def questions(domain_a, settings):
    return load_questions(domain_a.questions_csv, "question", settings.get("eval.baselines"))


@pytest.fixture
def references(domain_a, settings, questions):
    con = connect_readonly(domain_a.db_path)
    try:
        return build_references(questions, con, None, settings)
    finally:
        con.close()


@pytest.fixture
def oats_count(domain_a) -> int:
    con = connect_readonly(domain_a.db_path)
    try:
        return con.execute("SELECT COUNT(*) FROM grain_quotes WHERE crop = 'Oats'").fetchone()[0]
    finally:
        con.close()


class _StubAgent:
    """Stands in for ReferenceAgent: one LLM call per question, an LLM failure becomes an error answer."""

    def __init__(self, llm):
        self._llm = llm

    def answer(self, text: str) -> Answer:
        try:
            reply = self._llm.complete(f"[task: answer]\n{text}")
        except LLMError as exc:
            return Answer(text="", citations=[], tool_trace=[], latency_s=0.1, tokens={}, error=str(exc))
        return Answer(text=reply.text, citations=[], tool_trace=[], latency_s=0.1, tokens={})

    def close(self):
        pass


def _agent_factory(settings, llm, created: list):
    def build():
        created.append(1)
        return _StubAgent(llm)

    return build


def _answer(text: str, error: str | None = None) -> dict:
    return dataclasses.asdict(
        Answer(text=text, citations=[], tool_trace=[], latency_s=0.1, tokens={}, error=error)
    )


def _boom(prompt, tools):
    raise LLMError("llm timeout")


def _scores(settings, questions, references, answers, llm, categories):
    return score_config(questions, categories, references, answers, Judge(llm), llm, settings, "agent")


def _categories(**by_qid: str) -> dict:
    return {qid: {"category": category} for qid, category in by_qid.items()}


# ----------------------------------------------------------------- run_config


def test_run_config_resume_skips_finished_questions(settings, questions):
    llm = FakeLLM({"answer": "forty-two"})
    eval_dir = settings.path("paths.eval")
    run_config(questions, _agent_factory(settings, llm, []), "agent", eval_dir, limit=2)
    assert len(llm.calls) == 2

    answers = run_config(questions, _agent_factory(settings, llm, []), "agent", eval_dir, limit=3)

    assert len(llm.calls) == 3
    assert sorted(answers) == ["q001", "q002", "q003"]


def test_run_config_asks_again_when_a_saved_answer_file_is_truncated(settings, questions):
    llm = FakeLLM({"answer": "forty-two"})
    eval_dir = settings.path("paths.eval")
    answers_dir = eval_dir / "answers" / "agent"
    answers_dir.mkdir(parents=True)
    (answers_dir / "q001.json").write_text('{"text": "forty-t')

    answers = run_config(questions, _agent_factory(settings, llm, []), "agent", eval_dir, limit=1)

    assert answers["q001"]["text"] == "forty-two"
    assert json.loads((answers_dir / "q001.json").read_text())["text"] == "forty-two"


def test_run_config_stores_llm_failure_then_retries_it_on_rerun(settings, questions):
    llm = FakeLLM({"answer": [_boom, "recovered"]})
    eval_dir = settings.path("paths.eval")

    first = run_config(questions, _agent_factory(settings, llm, []), "agent", eval_dir, limit=2)

    assert first["q001"]["error"] == "llm timeout"
    assert first["q002"]["text"] == "recovered"

    second = run_config(questions, _agent_factory(settings, llm, []), "agent", eval_dir, limit=2)

    assert second["q001"]["text"] == "recovered"
    assert second["q001"]["error"] is None
    assert len(llm.calls) == 3


# ----------------------------------------------------------------- score_config

def test_question_without_sql_gets_skipped_exec_and_a_behaviour_result(settings, questions, references):
    llm = FakeLLM({"judge_pairwise": _TIE, "behaviour": '{"pass": true, "reason": "refused"}'})

    scores = _scores(
        settings, questions, references, {_ADVERSARIAL_Q: _answer("I cannot do that.")},
        llm, _categories(q005="adversarial"),
    )

    score = scores[_ADVERSARIAL_Q]
    assert score["exec"]["status"] == "skipped"
    assert score["behaviour"] == {"pass": True, "reason": "refused"}
    assert score["correctness"] is None


def test_lookup_answer_with_the_reference_count_passes_exec(settings, questions, references, oats_count):
    llm = FakeLLM({"judge_pairwise": _TIE, "judge_correctness": _JSON_CORRECT})

    scores = _scores(
        settings, questions, references, {_OATS_Q: _answer(f"There are {oats_count:,} Oats quotes.")},
        llm, _categories(q001="lookup"),
    )

    score = scores[_OATS_Q]
    assert score["exec"]["status"] == "pass"
    assert (score["exec"]["matched"], score["exec"]["total"]) == (1, 1)
    assert score["correctness"] == {"verdict": "correct", "reason": "matches"}
    assert score["behaviour"] is None


def test_exec_counts_reference_values_found_in_tool_rows(settings, questions, references, oats_count):
    llm = FakeLLM({"judge_pairwise": _TIE, "judge_correctness": _JSON_CORRECT})
    answer = _answer("The count was dropped from this answer.")
    answer["tool_trace"] = [
        {"tool": "query_metric", "args": {}, "status": "ok", "rows": [[oats_count]]},
        {"tool": "query_metric", "args": {}, "status": "repeat"},
    ]

    scores = _scores(settings, questions, references, {_OATS_Q: answer}, llm, _categories(q001="lookup"))

    exec_score = scores[_OATS_Q]["exec"]
    assert (exec_score["status"], exec_score["matched"], exec_score["tool_matched"]) == ("fail", 0, 1)


def test_drivers_question_without_reference_sql_gets_no_correctness_verdict(settings, questions, references):
    llm = FakeLLM({"judge_pairwise": _TIE})

    scores = _scores(
        settings, questions, references, {_DRIVERS_Q: _answer("No cause is stored.")},
        llm, _categories(q004="drivers"),
    )

    assert scores[_DRIVERS_Q]["exec"]["status"] == "skipped"
    assert scores[_DRIVERS_Q]["correctness"] is None
    assert not any("[task: judge_correctness]" in prompt for prompt, _ in llm.calls)


def test_error_answers_score_as_failures_and_pairwise_losses_without_calling_the_judge(settings, questions, references):
    llm = FakeLLM({})
    answers = {_OATS_Q: _answer("", error="llm timeout"), _ADVERSARIAL_Q: _answer("", error="llm timeout")}

    scores = _scores(
        settings, questions, references, answers, llm, _categories(q001="lookup", q005="adversarial"),
    )

    assert llm.calls == []
    assert scores[_OATS_Q]["error"] is True
    assert scores[_OATS_Q]["exec"]["status"] == "fail"
    assert scores[_OATS_Q]["correctness"] == {"verdict": "unusable", "reason": "error"}
    assert scores[_OATS_Q]["pairwise"] == {"reference": {"verdict": "reference", "reason": "error"}}
    assert scores[_ADVERSARIAL_Q]["behaviour"] == {"pass": False, "reason": "error"}


def test_pairwise_runs_against_the_baseline(settings, questions, references, oats_count):
    ours = f"There are {oats_count} Oats quotes."

    def prefer_ours(prompt, tools):
        winner = "A" if ours in prompt.split("Answer B:")[0] else "B"
        return json.dumps({"winner": winner})

    llm = FakeLLM({"judge_correctness": _JSON_CORRECT, "judge_pairwise": prefer_ours})
    args = (settings, questions, references, {_OATS_Q: _answer(ours)}, llm, _categories(q001="lookup"))

    scores = _scores(*args)

    assert scores[_OATS_Q]["pairwise"]["reference"]["verdict"] == "ours"
    assert scores[_OATS_Q]["pairwise"]["reference"]["votes"]["ours"] == _PAIRWISE_PASSES
    assert len(llm.calls) == 1 + _PAIRWISE_PASSES


def test_without_baselines_answers_are_scored_by_behaviour_only(tmp_settings, domain_a):
    settings = tmp_settings({"database": {"path": str(domain_a.db_path)}, "eval": {"baselines": []}})
    questions = load_questions(domain_a.questions_csv, "question", [])
    con = connect_readonly(domain_a.db_path)
    try:
        references = build_references(questions, con, None, settings)
    finally:
        con.close()
    llm = FakeLLM({"behaviour": '{"pass": true, "reason": "refuses"}'})
    answers = {_OATS_Q: _answer("There are some Oats quotes."), _ADVERSARIAL_Q: _answer("I will not do that.")}

    scores = _scores(settings, questions, references, answers, llm, _categories(q001="lookup", q005="adversarial"))

    assert scores[_OATS_Q]["exec"]["status"] == "skipped"
    assert scores[_OATS_Q]["correctness"] is None
    assert scores[_OATS_Q]["pairwise"] == {}
    assert scores[_ADVERSARIAL_Q]["behaviour"]["pass"] is True
    assert len(llm.calls) == 1


def test_sample_fills_quotas_prefers_in_window_skips_guardrail_categories_and_is_reused(tmp_path):
    from ctxlayer.common.files import read_yaml, write_yaml
    from ctxlayer.evaluation.questions import Contestant, Question
    from ctxlayer.evaluation.runner import sample_questions

    blank = Contestant(sql=None, answer="", time_ms=None)
    plan = ["drivers"] * 7 + ["lookup"] * 4 + ["adversarial"] * 2 + ["chit_chat"] * 1
    questions = [Question(f"q{i:03d}", f"text {i}", {"reference": blank}) for i in range(len(plan))]
    categories = {q.qid: {"category": cat, "has_sql": cat == "lookup"} for q, cat in zip(questions, plan)}
    # Drivers q004 and q005 are inside the document window: they must win over q000..q003.
    write_yaml(tmp_path / "windows.yaml", {q.qid: {"in_doc_window": q.qid in ("q004", "q005")} for q in questions})

    picked = [q.qid for q in sample_questions(questions, categories, tmp_path, 8)]
    by_category = [categories[qid]["category"] for qid in picked]

    # 5 drivers + 2 lookups by quota, then the 1 free slot goes to a spare driver/lookup, never a guardrail one.
    assert len(picked) == 8
    assert {"q004", "q005"} <= set(picked)
    assert by_category.count("drivers") + by_category.count("lookup") == 8
    assert "adversarial" not in by_category and "chit_chat" not in by_category

    # A hand-edited sample.yaml with the same n is used as is.
    saved = read_yaml(tmp_path / "sample.yaml")
    write_yaml(tmp_path / "sample.yaml", {**saved, "qids": ["q012", "q010"]})
    assert [q.qid for q in sample_questions(questions, categories, tmp_path, 8)] == ["q010", "q012"]


def test_parallel_run_gives_each_worker_its_own_agent_and_closes_them(tmp_path):
    from ctxlayer.evaluation.questions import Question
    from ctxlayer.evaluation.runner import run_config
    from ctxlayer.reference_agent.loop import Answer

    class _Agent:
        made: list = []

        def __init__(self):
            self.thread, self.closed = threading.get_ident(), False
            _Agent.made.append(self)

        def answer(self, text: str) -> Answer:
            assert threading.get_ident() == self.thread  # never shared across threads
            time.sleep(0.02)
            return Answer(f"answer to {text}", [], [], 0.0, {"est_tokens": 1})

        def close(self):
            self.closed = True

    questions = [Question(f"q{i:03d}", f"text {i}", {}) for i in range(8)]

    answers = run_config(questions, _Agent, "agent", tmp_path, workers=4)

    assert [answers[q.qid]["text"] for q in questions] == [f"answer to text {i}" for i in range(8)]
    assert 1 < len(_Agent.made) <= 4 and all(agent.closed for agent in _Agent.made)


def test_redo_asks_only_the_listed_questions_again(settings, questions):
    llm = FakeLLM({"answer": "forty-two"})
    eval_dir = settings.path("paths.eval")
    run_config(questions, _agent_factory(settings, llm, []), "agent", eval_dir, limit=3)

    run_config(questions, _agent_factory(settings, llm, []), "agent", eval_dir, limit=3, redo={"q002"})

    assert len(llm.calls) == 4  # three answers, then only q002 again


def test_scoring_with_only_keeps_the_other_saved_scores(settings, questions, references):
    answers = {question.qid: _answer("forty-two") for question in questions[:2]}
    categories = _categories(**{question.qid: "lookup" for question in questions[:2]})
    first = _scores(
        settings, questions, references, answers, FakeLLM({"judge_correctness": _JSON_CORRECT, "judge_pairwise": _TIE}), categories,
    )
    redo = questions[1].qid
    judge = FakeLLM({"judge_correctness": '{"verdict": "incorrect", "reason": "redone"}', "judge_pairwise": _TIE})

    again = score_config(questions, categories, references, answers, Judge(judge), judge, settings, "agent", {redo})

    assert again[questions[0].qid] == first[questions[0].qid]
    assert again[redo]["correctness"]["verdict"] == "incorrect"
    assert all(questions[1].text in prompt for prompt, _ in judge.calls)  # only the redone question is judged
