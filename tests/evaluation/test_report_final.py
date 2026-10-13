"""Final report sections, prepare_windows and the citation / checklist additions to scoring."""
import json

import pytest

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.files import read_yaml, write_yaml
from ctxlayer.evaluation.judge import Judge
from ctxlayer.evaluation.questions import Contestant, Question
from ctxlayer.evaluation.report import write_report
from ctxlayer.evaluation.runner import prepare_windows, score_config
from fakes import FakeLLM, per_citation

_QIDS = ["q001", "q002", "q003"]


def _question(qid: str) -> Question:
    contestant = Contestant(sql=None, answer="", time_ms=None)
    return Question(qid=qid, text=f"question {qid}", baselines={"reference": contestant})


def _score(exec_status: str, correct: bool | None = None, reference: str | None = None, citations=None) -> dict:
    score = {
        "category": "lookup",
        "error": False,
        "exec": {"status": exec_status, "reason": "x"},
        "correctness": None if correct is None else {"verdict": "correct" if correct else "incorrect"},
        "behaviour": None,
        "pairwise": {"reference": None if reference is None else {"verdict": reference}},
    }
    if citations is not None:
        score["citations"] = citations
    return score


def _answers(scores: dict) -> dict:
    return {qid: {"latency_s": 1.0, "tool_trace": [], "tokens": {"est_tokens": 10}} for qid in scores}


# exec 2/3, correct 2/3, reference 1/2, citations 3/5.
_SCORES = {
    "agent": {
        "q001": _score("pass", True, "ours", {"valid": 1, "total": 2}),
        "q002": _score("pass", True, "reference", {"valid": 2, "total": 2}),
        "q003": _score("fail", False, None, {"valid": 0, "total": 1}),
    },
}


def _report(tmp_path, scores=None, quality=None) -> str:
    scores = scores or _SCORES
    questions = [_question(qid) for qid in _QIDS]
    categories = {qid: {"category": "lookup"} for qid in _QIDS}
    answers = {label: _answers(by_qid) for label, by_qid in scores.items()}
    out_path = tmp_path / "eval" / "report.md"

    write_report(questions, categories, scores, answers, out_path, quality=quality or {}, database="domain_a")
    return out_path.read_text()


def _section(report: str, heading: str) -> str:
    return report.split(f"## {heading}", 1)[1].split("\n## ", 1)[0]


def _rows(section: str) -> list[list[str]]:
    lines = [line for line in section.splitlines() if line.startswith("|")][2:]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]


def test_window_table_counts_only_in_window_questions(tmp_path):
    windows = {
        "q001": {"start": "2025-01-01", "end": "2025-02-01", "in_doc_window": True},
        "q002": {"start": "2025-03-01", "end": "2025-04-01", "in_doc_window": True},
        "q003": {"start": "2020-01-01", "end": "2020-02-01", "in_doc_window": False},
    }
    write_yaml(tmp_path / "eval" / "windows.yaml", windows)

    section = _section(_report(tmp_path), "Questions inside the document window (n=2)")

    by_label = {row[0]: row[1:] for row in _rows(section)}
    assert by_label["agent"] == ["100.0% (2/2)", "100.0% (2/2)", "-", "50.0% (1/2)"]


def test_citation_table_sums_valid_over_total(tmp_path):
    rows = _rows(_section(_report(tmp_path), "Citation validity"))

    assert rows == [["agent", "3/5"]]


def test_missing_optional_files_skip_the_extra_sections(tmp_path):
    report = _report(tmp_path, {"agent": {qid: _score("pass", True) for qid in _QIDS}})

    for heading in ("Questions inside", "Citation validity", "Context-layer quality"):
        assert heading not in report
    assert "## Context-layer review files" in report
    assert "## Ablation" not in report


def test_report_has_ids_and_aggregates_but_no_answer_text(tmp_path):
    questions = [_question(qid) for qid in _QIDS]
    categories = {qid: {"category": "lookup"} for qid in _QIDS}
    answers = {label: _answers(by_qid) for label, by_qid in _SCORES.items()}
    for by_qid in answers.values():
        by_qid["q001"]["text"] = "SECRET-ANSWER-TEXT"

    out_path = tmp_path / "eval" / "report.md"
    write_report(questions, categories, _SCORES, answers, out_path, quality={}, database="domain_a")

    assert "SECRET-ANSWER-TEXT" not in out_path.read_text()


# ------------------------------------------------------------- prepare_windows

_WINDOW_REPLY = json.dumps([
    {"qid": "q001", "start": "2024-01-01", "end": "2024-02-01"},  # before the documents
    {"qid": "q002", "start": "2025-03-01", "end": "2025-03-31"},  # inside
    {"qid": "q003", "start": None, "end": None},  # default: 90 days before the anchor
])


@pytest.fixture
def window_settings(tmp_settings):
    return tmp_settings({"eval": {"now_anchor": "2025-06-30"}})


def test_prepare_windows_saves_windows_yaml_with_in_doc_window_flags(window_settings, tmp_path):
    docs = artifacts_dir(window_settings) / "docs" / "documents.jsonl"
    docs.parent.mkdir(parents=True)
    docs.write_text(
        json.dumps({"doc_id": "d1", "date": "2025-06-20"}) + "\n"
        + json.dumps({"doc_id": "d2", "date": "2025-01-10T08:00:00"}) + "\n"
    )
    questions = [_question(qid) for qid in _QIDS]

    counts = prepare_windows(window_settings, questions, FakeLLM({"question_window": _WINDOW_REPLY}))

    assert counts == {"questions": 3, "in_doc_window": 2}
    assert read_yaml(tmp_path / "eval" / "windows.yaml") == {
        "q001": {"start": "2024-01-01", "end": "2024-02-01", "in_doc_window": False},
        "q002": {"start": "2025-03-01", "end": "2025-03-31", "in_doc_window": True},
        "q003": {"start": "2025-04-01", "end": "2025-06-30", "in_doc_window": True},
    }


# ----------------------------------------------------------------- score_config

_TRACE = [{"result": {"doc_id": "d1", "text": "drought cut the oat harvest"}}]


def _answer(text: str) -> dict:
    return {"text": text, "error": None, "tool_trace": _TRACE, "citations": []}


def _score_label(settings, label: str, llm: FakeLLM) -> dict:
    # q001 chit_chat (rubric only), q002 drivers without reference SQL (exec only).
    questions = [_question("q001"), _question("q002")]
    categories = {"q001": {"category": "chit_chat"}, "q002": {"category": "drivers"}}
    references = {qid: {"baselines": {"reference": None}, "agree": None} for qid in ("q001", "q002")}
    answers = {"q001": _answer("Hello!"), "q002": _answer("Oats fell on the drought [doc:d1].")}

    return score_config(questions, categories, references, answers, Judge(llm), llm, settings, label)


def _llm() -> FakeLLM:
    return FakeLLM({
        "behaviour": '{"pass": true, "reason": "ok"}',
        "citation_check": per_citation(lambda block: True),
        "checklist": '{"covered": [0]}',
    })


def test_score_config_adds_citations(tmp_settings):
    scores = _score_label(tmp_settings(), "agent", _llm())

    assert {k: scores["q002"]["citations"][k] for k in ("valid", "total")} == {"valid": 1, "total": 1}
    assert scores["q001"]["citations"] == {"valid": 0, "total": 0, "items": []}


def test_score_config_checklist_only_for_drivers_questions_with_an_entry(tmp_settings, tmp_path):
    write_yaml(tmp_path / "eval" / "driver_checklist.yaml", {"q002": ["drought", "export ban"], "q001": ["x"]})

    scores = _score_label(tmp_settings(), "agent", _llm())

    assert scores["q002"]["checklist"] == {"covered": 1, "total": 2}
    assert "checklist" not in scores["q001"]


def test_position_check_counts_first_slot_and_win_rate_per_position(tmp_path):
    def pairwise(ours_first: bool, winner: str) -> dict:
        return {"verdict": winner, "votes": {}, "traps": {}, "passes": [{"ours_first": ours_first, "winner": winner}]}

    scores = {"agent": {
        "q1": {**_score("pass"), "pairwise": {"reference": pairwise(True, "ours")}},
        "q2": {**_score("pass"), "pairwise": {"reference": pairwise(True, "reference")}},
        "q3": {**_score("pass"), "pairwise": {"reference": pairwise(False, "ours")}},
    }}

    report = _report(tmp_path, scores)

    assert "- agent: ours shown first 2/3; win rate when first 50% (1/2), when second 100% (1/1)" in report
