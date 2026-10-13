"""Run the reference agent over the questions and score each saved answer."""
import dataclasses
import json
import threading
from pathlib import Path
from typing import Callable

from ctxlayer.common._names import ANSWERS_DIR, SCORES_DIR, UNUSABLE
from ctxlayer.common.config import Settings, artifacts_dir
from ctxlayer.common.files import read_yaml, write_yaml
from ctxlayer.common.llm import LLM
from ctxlayer.common.parallel import parallel_map
from ctxlayer.common.runlog import progress
from ctxlayer.evaluation.behaviour import check_behaviour
from ctxlayer.evaluation.categorize import BEHAVIOUR_CATEGORIES
from ctxlayer.evaluation.citations import citation_validity, driver_checklist_score
from ctxlayer.evaluation.exec_check import exec_agreement
from ctxlayer.evaluation.judge import Judge
from ctxlayer.evaluation.questions import Question, question_windows, reference_baseline
from ctxlayer.evaluation.reference import RefResult, resolve_anchor
from ctxlayer.reference_agent.api import Answer, ReferenceAgent

_ERROR_REASON = "error"
_DRIVERS = "drivers"
RUN_LABEL = "agent"
_CHECKLIST_FILE = "driver_checklist.yaml"
_WINDOWS_FILE = "windows.yaml"
_SAMPLE_FILE = "sample.yaml"
# Questions per category in a sample run, strengths first (drivers across documents and measures).
# Guardrail-only categories (adversarial, out_of_scope, chit_chat) are left out: they do not test the context layer.
_SAMPLE_QUOTAS = (
    ("drivers", 5), ("trend_compute", 3), ("lookup", 2), ("comparison", 2), ("projection", 1),
    ("volatility", 1), ("data_trap", 1),
)
_SAMPLE_CATEGORIES = {category for category, _ in _SAMPLE_QUOTAS}
_DATE_CHARS = 10  # "2025-04-27" of "2025-04-27T08:00:00"


def sample_questions(questions: list[Question], categories: dict, eval_dir: Path, n: int) -> list[Question]:
    """`n` questions by category quota, in-window and with-SQL first; saved to eval/sample.yaml (ids only).

    An existing sample.yaml with the same `n` is reused as is, so a hand-edited list is respected.
    """
    path = eval_dir / _SAMPLE_FILE
    saved = read_yaml(path) if path.is_file() else {}
    texts = {q.qid: q.text for q in questions}
    # A saved sample made for other question texts (rows added, removed or reordered) is not reused.
    if saved.get("n") == n and all(texts.get(qid) == text for qid, text in saved.get("texts", {}).items()):
        keep = set(saved["qids"])
        return [question for question in questions if question.qid in keep]

    windows = read_yaml(eval_dir / _WINDOWS_FILE) if (eval_dir / _WINDOWS_FILE).is_file() else {}

    def in_window(qid: str) -> bool:
        return bool(windows.get(qid, {}).get("in_doc_window"))

    # Best first: inside the document window, has reference SQL (numeric check applies), then lowest id.
    ranked = sorted(questions, key=lambda q: (not in_window(q.qid), not categories[q.qid].get("has_sql"), q.qid))
    chosen: list[Question] = []
    for category, quota in _SAMPLE_QUOTAS:
        picks = [q for q in ranked if categories[q.qid]["category"] == category and q not in chosen]
        chosen += picks[:min(quota, n - len(chosen))]

    # Categories short of questions: fill the free slots in the same order, from the same categories only.
    spare = [q for q in ranked if q not in chosen and categories[q.qid]["category"] in _SAMPLE_CATEGORIES]
    chosen += spare[:n - len(chosen)]

    picked = [
        {"qid": q.qid, "category": categories[q.qid]["category"], "in_doc_window": in_window(q.qid),
         "has_sql": bool(categories[q.qid].get("has_sql"))}
        for q in sorted(chosen, key=lambda q: q.qid)
    ]
    write_yaml(path, {
        "n": n, "qids": [item["qid"] for item in picked], "texts": {item["qid"]: texts[item["qid"]] for item in picked},
        "picked": picked,
    })
    return sorted(chosen, key=lambda q: q.qid)


def prepare_windows(settings: Settings, questions: list[Question], llm: LLM) -> dict:
    """Save eval/windows.yaml ({qid: {start, end, in_doc_window}}); returns the counts."""
    now = resolve_anchor(settings)
    if now is None:
        raise ValueError("no data anchor; run ctx build-structured")

    docs_path = artifacts_dir(settings) / "docs" / "documents.jsonl"
    if not docs_path.is_file():
        raise FileNotFoundError(f"{docs_path} not found: run `ctx build-unstructured` first")
    days = [json.loads(line)["date"][:_DATE_CHARS] for line in docs_path.read_text().splitlines() if line.strip()]

    windows = question_windows(questions, llm, now, (min(days), max(days)), settings.get("llm.parallel_calls"))
    write_yaml(settings.path("paths.eval") / _WINDOWS_FILE, windows)
    return {"questions": len(windows), "in_doc_window": sum(w["in_doc_window"] for w in windows.values())}


def run_config(
    questions: list[Question],
    agent_factory: Callable[[], ReferenceAgent],
    label: str,
    out_dir: Path,
    limit: int | None = None,
    workers: int = 1,
    redo: set[str] | None = None,
) -> dict[str, dict]:
    """Answer the first `limit` questions into out_dir/answers/<label>/<qid>.json, `workers` questions at once.

    A saved answer without error is kept (resume); one with an error, or listed in `redo`, is asked again. Each worker thread
    builds its own agent (own database connections), lazily, so a fully finished run opens nothing.
    """
    answers_dir = out_dir / ANSWERS_DIR / label
    answers_dir.mkdir(parents=True, exist_ok=True)

    local = threading.local()
    agents: list[ReferenceAgent] = []
    agents_lock = threading.Lock()

    def answer_one(question: Question) -> tuple[str, dict]:
        path = answers_dir / f"{question.qid}.json"
        saved = None if redo and question.qid in redo else _finished_answer(path, question.text)
        if saved is not None:
            return question.qid, saved

        if not hasattr(local, "agent"):
            local.agent = agent_factory()
            with agents_lock:
                agents.append(local.agent)
        try:
            answer = dataclasses.asdict(local.agent.answer(question.text))
        except Exception as error:  # one bad question must not stop the run; it is scored as an error
            answer = dataclasses.asdict(Answer(
                "", [], [], 0.0, {"prompt_chars": 0, "reply_chars": 0, "est_tokens": 0},
                error=f"{type(error).__name__}: {error}",
            ))
        answer["question"] = question.text
        path.write_text(json.dumps(answer, indent=2, default=str))
        return question.qid, answer

    selected = questions[:limit]
    try:
        results = parallel_map(
            answer_one, selected, workers, lambda done, total: progress("eval_run.answers", done, total),
        )
    finally:
        for agent in agents:
            agent.close()

    return dict(results)


def score_config(
    questions: list[Question],
    categories: dict,
    references: dict,
    answers: dict,
    judge: Judge,
    llm: LLM,
    settings: Settings,
    label: str,
    only: set[str] | None = None,
) -> dict[str, dict]:
    """Score every answered question and save eval/scores/<label>.json.

    Per qid: {category, error, exec, correctness, behaviour, pairwise: {baseline name: verdict}};
    parts that do not apply are None. Answers without error add `citations`; drivers questions with an entry
    in eval/driver_checklist.yaml add `checklist`. With `only`, just those qids are judged again and every
    other saved score is kept.
    """
    scorer = _Scorer(judge, llm, settings)
    path = settings.path("paths.eval") / SCORES_DIR / f"{label}.json"
    saved = json.loads(path.read_text()) if only and path.is_file() else {}
    checklists = _read_checklists(settings.path("paths.eval") / _CHECKLIST_FILE)

    def score_one(question: Question) -> tuple[str, dict]:
        if only and question.qid not in only and question.qid in saved:
            return question.qid, saved[question.qid]

        answer = answers[question.qid]
        category = categories[question.qid]["category"]
        score = scorer.score(question, category, references[question.qid], answer)
        if answer.get("error") is None:
            score["citations"] = citation_validity(answer, llm)
        facts = checklists.get(question.qid)
        if category == _DRIVERS and facts:
            score["checklist"] = _checklist(answer, facts, llm)
        return question.qid, score

    # Questions are judged `llm.parallel_calls` at a time; each one's judge calls stay in order.
    answered = [question for question in questions if question.qid in answers]
    scores = dict(parallel_map(
        score_one, answered, settings.get("llm.parallel_calls"),
        lambda done, total: progress("eval_run.scores", done, total),
    ))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, indent=2, default=str))
    return scores


def _read_checklists(path: Path) -> dict:
    # {qid: [facts]}; the file is optional.
    if not path.is_file():
        return {}
    return read_yaml(path) or {}


def _checklist(answer: dict, facts: list[str], llm: LLM) -> dict:
    # A failed run covers nothing, without asking the LLM.
    if answer.get("error") is not None:
        return {"covered": 0, "total": len(facts)}
    return driver_checklist_score(answer["text"], facts, llm)


class _Scorer:
    def __init__(self, judge: Judge, llm: LLM, settings: Settings):
        self._judge = judge
        self._llm = llm
        self._rel_tol = settings.get("eval.rel_tolerance")
        self._max_values = settings.get("eval.max_key_values")
        self._passes = settings.get("eval.pairwise_passes")
        self._reference = reference_baseline(settings.get("eval.baselines"))

    def score(self, question: Question, category: str, reference: dict, answer: dict) -> dict:
        failed = answer.get("error") is not None
        text = answer["text"]
        score = {
            "category": category,
            "error": failed,
            "exec": None,
            "correctness": None,
            "behaviour": None,
            "pairwise": {},
        }

        # Behaviour categories have no reference result: rubric only, exec skipped.
        if category in BEHAVIOUR_CATEGORIES:
            score["exec"] = self._exec(None, answer)
            score["behaviour"] = self._behaviour(category, question, text, failed)
            return self._with_pairwise(score, question, text, failed)

        ref = self._reference_result(reference)
        score["exec"] = self._exec(ref, answer)

        # No reference baseline, or drivers without reference SQL (it only declined): not judged for now.
        if self._reference is None or (category == _DRIVERS and ref is None):
            return self._with_pairwise(score, question, text, failed)

        score["correctness"] = self._correctness(question, ref, text, failed)
        return self._with_pairwise(score, question, text, failed)

    def _reference_result(self, reference: dict) -> RefResult | None:
        found = None if self._reference is None else reference["baselines"].get(self._reference)
        return None if found is None else RefResult(**found)

    def _exec(self, ref: RefResult | None, answer: dict) -> dict:
        # The answer text decides; rows the tools returned only feed `tool_matched`.
        tool_rows = [row for entry in answer["tool_trace"] for row in entry.get("rows", [])]
        return exec_agreement(ref, answer["text"], tool_rows, self._rel_tol, self._max_values)

    def _behaviour(self, category: str, question: Question, text: str, failed: bool) -> dict:
        if failed:
            return {"pass": False, "reason": _ERROR_REASON}
        return check_behaviour(self._llm, category, question.text, text)

    def _correctness(self, question: Question, ref: RefResult | None, text: str, failed: bool) -> dict:
        if failed:
            return {"verdict": UNUSABLE, "reason": _ERROR_REASON}
        rows = ref.rows if ref is not None and not ref.error else None
        return self._judge.correctness(question.text, question.baselines[self._reference].answer, rows, text)

    def _with_pairwise(self, score: dict, question: Question, text: str, failed: bool) -> dict:
        for name, baseline in question.baselines.items():
            score["pairwise"][name] = self._versus(question, text, failed, name, baseline.answer)
        return score

    def _versus(self, question: Question, ours: str, failed: bool, opponent: str, opponent_answer: str) -> dict | None:
        # No stored answer from the baseline -> nothing to compare with.
        if not opponent_answer:
            return None

        # An errored answer loses without asking the judge.
        if failed:
            return {"verdict": opponent, "reason": _ERROR_REASON}
        return self._judge.pairwise(question.text, ours, opponent, opponent_answer, self._passes)


def _finished_answer(path: Path, question: str) -> dict | None:
    """The saved answer, or None when missing, unreadable (cut off by an interrupt), stored with an error or for another question."""
    if not path.is_file():
        return None

    try:
        answer = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if answer.get("error") is not None or answer.get("question") not in (None, question):
        return None
    return answer
