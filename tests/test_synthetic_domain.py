"""End to end on the synthetic domain: prepare -> run the agent -> score -> report."""
import contextlib
import json
import shutil
import statistics
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest
import requests
import yaml

from ctxlayer.cli import main
from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.llm import Reply, ToolCall
from ctxlayer.evaluation.categorize import categorize
from ctxlayer.evaluation.judge import Judge
from ctxlayer.evaluation.questions import load_questions
from ctxlayer.evaluation.reference import build_references, resolve_anchor
from ctxlayer.evaluation.report import write_eval_review, write_report
from ctxlayer.evaluation.quality import quality_checks
from ctxlayer.evaluation.runner import RUN_LABEL, prepare_windows, run_config, score_config
from ctxlayer.linking.build import build_links
from ctxlayer.reference_agent.api import ReferenceAgent
from ctxlayer.unstructured.build import build_unstructured
from fakes import FakeEmbedder, FakeLLM, per_citation
from fixtures.domain_a.make_domain import PLANTED_BULLETIN_REF, PLANTED_QUOTE, PLANTED_SERIES_CODE
from fixtures.domain_a.unstructured_build import unstructured_llm

_ANCHOR = "2025-06-30"
_LOW_CONFIDENCE = 0.3
_CATEGORY_OF = {
    "q001": "lookup", "q002": "lookup", "q003": "trend_compute", "q004": "drivers",
    "q005": "adversarial", "q006": "out_of_scope", "q007": "chit_chat", "q008": "trend_compute",
}
_ABSENT_CATEGORIES = ("projection", "comparison", "volatility", "data_trap")
_PLANTED_NAME = "Milling Wheat - Great Plains"
_LATEST_PRICE_QID = "q002"
_ADVERSARIAL_QID = "q005"
_ANSWERS_FILE = f"answers/{RUN_LABEL}"


@pytest.fixture
def settings(tmp_settings, domain_a, domain_a_built):
    """Project settings over a copy of the structured build of domain_a, with the eval questions."""
    overrides = {
        "database": {"path": str(domain_a.db_path)},
        "eval": {"questions_csv": str(domain_a.questions_csv), "now_anchor": _ANCHOR},
    }
    settings = tmp_settings(overrides)
    shutil.copytree(artifacts_dir(domain_a_built), artifacts_dir(settings))
    return settings


@pytest.fixture
def oats_count(domain_a) -> int:
    con = duckdb.connect(str(domain_a.db_path), read_only=True)
    try:
        return con.execute(
            f"SELECT COUNT(*) FROM grain_quotes WHERE crop = 'Oats' AND quote_date <= DATE '{_ANCHOR}'"
        ).fetchone()[0]
    finally:
        con.close()


@pytest.fixture
def latest_price(domain_a) -> float:
    con = duckdb.connect(str(domain_a.db_path), read_only=True)
    try:
        return con.execute(
            "SELECT price_mid FROM grain_quotes "
            f"WHERE series_code = '{PLANTED_SERIES_CODE}' AND basis = 'observed' ORDER BY quote_date DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        con.close()


@pytest.fixture
def scripted_llm(domain_a, latest_price) -> FakeLLM:
    """The guardrail refuses by keyword; the latest-price question is answered through query_metric, others brushed off."""
    latest_question = pd.read_csv(domain_a.questions_csv)["question"][1]
    verdicts = [
        {"qid": qid, "category": category, "confidence": _LOW_CONFIDENCE if qid == "q004" else 0.9}
        for qid, category in _CATEGORY_OF.items()
    ]
    call = Reply("", [ToolCall("query_metric", {"metric": "latest_value", "entities": [_PLANTED_NAME]})], 0.0, 0, 0)

    def guardrail(prompt, tools):
        question = prompt.split("<question>\n")[1].split("\n</question>")[0].lower()
        action = "refuse" if "delete every row" in question else "proceed"
        return json.dumps({"action": action, "flags": [], "reason": "scripted"})

    def answer(prompt, tools):
        if f"User: {latest_question}" not in prompt:
            return "That is outside what I can say."
        if "Tool result (query_metric)" in prompt:
            return f"The latest observed mid price is {latest_price}."
        return call

    def correctness(prompt, tools):
        good = str(latest_price) in prompt.split("Answer to judge:")[1]
        return json.dumps({"verdict": "correct" if good else "incorrect", "reason": "scripted"})

    def pairwise(prompt, tools):
        # Prefer the side that states the price; no such side -> tie.
        first, second = prompt.split("Answer A:")[1].split("Answer B:")
        winner = "A" if str(latest_price) in first else "B" if str(latest_price) in second else "tie"
        return json.dumps({"winner": winner})

    return FakeLLM({
        "categorize": json.dumps(verdicts),
        "guardrail": guardrail,
        "answer": answer,
        "judge_correctness": correctness,
        "behaviour": '{"pass": true, "reason": "scripted"}',
        "judge_pairwise": pairwise,
        "resolve_judge": '{"id": null, "confidence": 0}',
        "ping": "OK",
    })


def _section(report: str, heading: str) -> list[str]:
    """Table rows under `heading`, up to the next heading."""
    lines = report.splitlines()
    rows = []
    for line in lines[lines.index(heading) + 1:]:
        if line.startswith("#"):
            break
        if line.startswith("| "):
            rows.append(line)
    return rows


def _csv_median_seconds(domain_a, column: str, rows: slice) -> float:
    return statistics.median(pd.read_csv(domain_a.questions_csv)[column][rows]) / 1000


def _assert_report_content(report: str, domain_a) -> None:
    # One row per category present, none for absent ones.
    assert _section(report, "### Questions")[1:] == [
        "| lookup | 2 |", "| trend_compute | 2 |", "| drivers | 1 |",
        "| adversarial | 1 |", "| out_of_scope | 1 |", "| chit_chat | 1 |",
    ]
    for category in _ABSENT_CATEGORIES:
        assert f"| {category} |" not in report

    # q002 states the reference price; q001 and the other numeric questions do not.
    assert "| lookup | 50% (1/2) |" in _section(report, "### Execution pass")
    assert "| trend_compute | 0% (0/2) |" in _section(report, "### Execution pass")
    assert "| drivers | - |" in _section(report, "### Execution pass")
    assert "| lookup | 50% (1/2) |" in _section(report, "### Correct")
    assert "| adversarial | 100% (1/1) |" in _section(report, "### Behaviour pass")

    # Drivers have no usable reference: not judged, so neither correct nor a failure.
    assert "| drivers | - |" in _section(report, "### Correct")

    # The baseline's latency comes from the CSV (ms -> s); ours is measured.
    reference_s = _csv_median_seconds(domain_a, "reference_time_ms", slice(0, 2))
    latency = _section(report, "### Median latency (s)")
    assert any(row.startswith(f"| lookup | {reference_s:.1f} | ") for row in latency)

    # q002 made one tool call, q001 none.
    assert "| lookup | 0.5 |" in _section(report, "### Median tool calls")

    # q002: ours wins every pass; q001: ties -> no majority.
    assert f"| lookup | {RUN_LABEL} | 2 | 1 | 0 | 1 | 50% (1/2) | 0 | 0 |" in _section(report, "## Pairwise vs reference")

    # Failures are ids and reasons with a link to the stored answer; q002 is not one.
    failures = _section(report, "## Failures")
    assert not any(row.startswith("| q002 |") for row in failures)
    assert not any(row.startswith("| q004 |") for row in failures)
    q001 = next(row for row in failures if row.startswith("| q001 |"))
    assert q001 == (
        f"| q001 | lookup | {RUN_LABEL} | exec: 1 of 1 reference values not in answer; correctness: incorrect "
        f"| {_ANSWERS_FILE}/q001.json |"
    )

    # No answer text in the report.
    assert "latest observed mid price" not in report
    assert "outside what I can say" not in report


def test_python_pipeline_prepare_run_and_report(settings, domain_a, scripted_llm, oats_count, latest_price):
    eval_dir = settings.path("paths.eval")
    questions = load_questions(domain_a.questions_csv, "question", settings.get("eval.baselines"))

    # prepare
    categories = categorize(questions, scripted_llm, eval_dir / "categories.yaml")
    con = connect_readonly(domain_a.db_path)
    references = build_references(questions, con, resolve_anchor(settings), settings)
    con.close()
    write_eval_review(categories, references, eval_dir / "REVIEW.md")

    # run the agent
    answers = run_config(
        questions, lambda: ReferenceAgent(settings, llm=scripted_llm, embedder=FakeEmbedder()), RUN_LABEL, eval_dir,
    )
    scores = score_config(
        questions, categories, references, answers, Judge(scripted_llm), scripted_llm, settings, RUN_LABEL,
    )

    # The reference of q001 is re-anchored; the latest-price answer states the reference price via query_metric.
    saved = json.loads((eval_dir / "scores" / f"{RUN_LABEL}.json").read_text())
    assert references["q001"]["baselines"]["reference"]["rows"] == [[oats_count]]
    assert references["q001"]["baselines"]["reference"]["reanchored"] is True
    assert references[_LATEST_PRICE_QID]["baselines"]["reference"]["rows"][0][0] == latest_price
    lookup = saved[_LATEST_PRICE_QID]
    assert lookup["exec"]["status"] == "pass"
    assert lookup["exec"]["tool_matched"] == lookup["exec"]["total"] == 1
    assert [entry["tool"] for entry in answers[_LATEST_PRICE_QID]["tool_trace"]] == ["query_metric"]
    assert answers[_LATEST_PRICE_QID]["citations"][0]["kind"] == "sql"
    assert "sem_grain_quotes" in answers[_LATEST_PRICE_QID]["citations"][0]["sql"]

    # Adversarial: refused by the guardrail, no tool ran, behaviour rubric passes.
    assert answers[_ADVERSARIAL_QID]["tool_trace"] == []
    assert scores[_ADVERSARIAL_QID]["exec"]["status"] == "skipped"
    assert scores[_ADVERSARIAL_QID]["behaviour"]["pass"] is True

    # report
    report_path = eval_dir / "report.md"
    write_report(
        questions, categories, {RUN_LABEL: scores}, {RUN_LABEL: answers}, report_path, quality={}, database="domain_a",
    )
    _assert_report_content(report_path.read_text(), domain_a)


def test_review_lists_low_confidence_transpiled_and_reanchored_questions(settings, domain_a, scripted_llm):
    eval_dir = settings.path("paths.eval")
    questions = load_questions(domain_a.questions_csv, "question", settings.get("eval.baselines"))
    categories = categorize(questions, scripted_llm, eval_dir / "categories.yaml")
    con = connect_readonly(domain_a.db_path)
    references = build_references(questions, con, resolve_anchor(settings), settings)
    con.close()

    write_eval_review(categories, references, eval_dir / "REVIEW.md")

    review = (eval_dir / "REVIEW.md").read_text()
    assert _bullets(review, "## Low-confidence categories") == [
        f"- q004 (drivers, confidence {_LOW_CONFIDENCE}): {questions[3].text}"
    ]
    assert _bullets(review, "## Baselines disagree") == ["- none"]
    assert _bullets(review, "## Transpiled for DuckDB") == ["- none"]
    assert "- q001 (reference)" in _bullets(review, "## Re-anchored to the data date")
    assert _bullets(review, "## Reference errors (details in references.json)") == ["- none"]


def test_review_lists_current_date_queries_that_ran_without_an_anchor(settings, domain_a, scripted_llm):
    eval_dir = settings.path("paths.eval")
    questions = load_questions(domain_a.questions_csv, "question", settings.get("eval.baselines"))
    categories = categorize(questions, scripted_llm, eval_dir / "categories.yaml")
    con = connect_readonly(domain_a.db_path)
    references = build_references(questions, con, None, settings)
    con.close()

    write_eval_review(categories, references, eval_dir / "REVIEW.md")

    review = (eval_dir / "REVIEW.md").read_text()
    assert _bullets(review, "## Uses current date, not re-anchored") == ["- q001 (reference)"]
    assert _bullets(review, "## Re-anchored to the data date") == ["- none"]


def _bullets(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    found = []
    for line in lines[lines.index(heading) + 1:]:
        if line.startswith("#"):
            break
        if line.startswith("- "):
            found.append(line)
    return found


# ------------------------------------------------------------------------ CLI

class _Response:
    status_code = 200
    text = ""

    def __init__(self, body: dict):
        self._body = body

    def json(self) -> dict:
        return self._body


class _Ollama:
    """Stands in for requests.post: relays the prompt to a FakeLLM in Ollama's reply shape."""

    def __init__(self, llm: FakeLLM):
        self._llm = llm

    def __call__(self, url, **kwargs):
        body = kwargs["json"]
        tools = [tool["function"] for tool in body.get("tools", [])]
        reply = self._llm.complete(body["messages"][0]["content"], tools or None)
        calls = [{"function": {"name": call.name, "arguments": call.args}} for call in reply.tool_calls]
        return _Response({"message": {"content": reply.text, "tool_calls": calls}})


@pytest.fixture
def cli_env(settings, scripted_llm, monkeypatch):
    """Project root for `main`: settings.yaml on disk, requests.post faked."""
    (settings.root / "config" / "settings.yaml").write_text(yaml.safe_dump(settings.raw))
    monkeypatch.setenv("CTX_ROOT", str(settings.root))
    monkeypatch.setattr(requests, "post", _Ollama(scripted_llm))
    return settings


def test_cli_eval_prepare_run_report_print_counts_and_write_files(cli_env, domain_a, capsys):
    eval_dir = cli_env.path("paths.eval")
    texts = pd.read_csv(domain_a.questions_csv)["question"].tolist()

    assert main(["eval", "prepare"]) == 0
    prepare_out = capsys.readouterr().out
    assert "questions: 8" in prepare_out
    assert "current-date-not-anchored=0" in prepare_out
    assert yaml.safe_load((eval_dir / "categories.yaml").read_text())["q001"]["category"] == "lookup"
    assert sorted(json.loads((eval_dir / "references.json").read_text())) == sorted(_CATEGORY_OF)
    assert (eval_dir / "REVIEW.md").is_file()

    assert main(["eval", "run"]) == 0
    run_out = capsys.readouterr().out
    scores = json.loads((eval_dir / "scores" / f"{RUN_LABEL}.json").read_text())
    assert scores[_LATEST_PRICE_QID]["exec"]["status"] == "pass"
    assert scores[_LATEST_PRICE_QID]["pairwise"]["reference"]["verdict"] == "ours"
    assert RUN_LABEL in run_out

    assert main(["eval", "report"]) == 0
    report_out = capsys.readouterr().out
    _assert_report_content((eval_dir / "report.md").read_text(), domain_a)
    assert str(eval_dir / "report.md") in report_out

    # Commands print counts and paths, never question or answer text.
    for printed in (prepare_out, run_out, report_out):
        assert not any(text in printed for text in texts)
        assert "latest observed mid price" not in printed


def test_cli_ask_prints_answer_and_sql_citation(cli_env, domain_a, latest_price, capsys):
    question = pd.read_csv(domain_a.questions_csv)["question"][1]

    assert main(["ask", question]) == 0

    out = capsys.readouterr().out
    assert f"The latest observed mid price is {latest_price}." in out
    assert "sem_grain_quotes" in out


def test_cli_ask_logs_the_operation_without_the_question(cli_env, monkeypatch):
    class Agent:
        def __init__(self, settings):
            pass

        def answer(self, question):
            return SimpleNamespace(error=None, text="ok", citations=[])

        def close(self):
            pass

    monkeypatch.setattr("ctxlayer.cli.ReferenceAgent", Agent)

    assert main(["ask", "question"]) == 0
    log = (cli_env.path("paths.runs") / "ctx.log").read_text()
    assert "operation=ask status=started" in log
    assert "operation=ask status=completed elapsed_s=" in log
    assert "question" not in log

    bundle = next(path for path in cli_env.path("paths.runs").iterdir() if path.is_dir())
    events = [json.loads(line) for line in (bundle / "events.jsonl").read_text().splitlines()]
    # Only stage start/end here: the fake agent makes no LLM calls, so no function payloads.
    assert {(item.get("operation"), item["status"]) for item in events} >= {("ask", "started"), ("ask", "completed")}
    assert not [item for item in events if item["kind"] == "function"]
    assert not list((bundle / "payloads").iterdir())


# ------------------------------------------------------- drivers end to end

_DRIVER_QID = "q004"


def test_planted_event_is_found_cited_and_checked_end_to_end(settings, domain_a):
    """build-unstructured -> build-links -> prepare -> run the agent -> score -> report."""
    embedder = FakeEmbedder()
    eval_dir = settings.path("paths.eval")
    driver_question = pd.read_csv(domain_a.questions_csv)["question"][3]
    call = Reply("", [ToolCall("explain_move", {"entity": _PLANTED_NAME, "start": "2025-04", "end": "2025-05"})], 0.0, 0, 0)
    reply_text = (
        f"Milling wheat rose in the week of 5 May 2025. A candidate driver is the drought that cut "
        f"Great Plains wheat supply [doc:{PLANTED_BULLETIN_REF}]."
    )

    def answer(prompt, tools):
        if "Tool result (explain_move)" in prompt:
            return reply_text
        return call

    # Supported only when the cited document's evidence holds the planted quote.
    citation_check = per_citation(lambda block: PLANTED_QUOTE in block.split("Evidence:", 1)[1])

    def windows(prompt, tools):
        return json.dumps([{"qid": _DRIVER_QID, "start": "2025-04-28", "end": "2025-05-11"}])

    llm = unstructured_llm(
        categorize=json.dumps([{"qid": _DRIVER_QID, "category": "drivers", "confidence": 0.9}]),
        question_window=windows,
        guardrail='{"action": "proceed", "flags": [], "reason": "scripted"}',
        answer=answer,
        citation_check=citation_check,
        judge_correctness='{"verdict": "correct", "reason": "scripted"}',
        judge_pairwise='{"winner": "tie"}',
        behaviour='{"pass": true, "reason": "scripted"}',
    )

    # Builds on the copied structured build: counts only, artifacts under tmp.
    build_unstructured(settings, llm, embedder)
    link_counts = build_links(settings)
    assert link_counts["moves_with_drivers"] >= 1

    # Prepare: categories, references and question windows.
    questions = [q for q in load_questions(domain_a.questions_csv, "question", settings.get("eval.baselines")) if q.qid == _DRIVER_QID]
    assert questions[0].text == driver_question
    categories = categorize(questions, llm, eval_dir / "categories.yaml")
    with contextlib.closing(connect_readonly(domain_a.db_path)) as con:
        references = build_references(questions, con, resolve_anchor(settings), settings)
    window_counts = prepare_windows(settings, questions, llm)
    assert window_counts == {"questions": 1, "in_doc_window": 1}

    # Run the agent and score it.
    label = RUN_LABEL
    answers = run_config(questions, lambda: ReferenceAgent(settings, llm=llm, embedder=embedder), label, eval_dir)
    scores = score_config(questions, categories, references, answers, Judge(llm), llm, settings, label)

    assert {"kind": "doc", "doc_id": PLANTED_BULLETIN_REF} in answers[_DRIVER_QID]["citations"]
    assert scores[_DRIVER_QID]["citations"]["valid"] == 1 and scores[_DRIVER_QID]["citations"]["total"] == 1

    # Report: in-window table, with the context-layer quality counts.
    report_path = eval_dir / "report.md"
    write_report(
        questions, categories, {label: scores}, {label: answers}, report_path, quality=quality_checks(settings),
        database="domain_a",
    )
    report = report_path.read_text()
    in_window = "## Questions inside the document window (n=1)"
    assert any(row.startswith(f"| {label} |") for row in _section(report, in_window))
    assert f"| {label} | 1/1 |" in _section(report, "## Citation validity")


def test_pipeline_without_baselines_answers_and_checks_behaviour_only(tmp_settings, domain_a, domain_a_built, scripted_llm):
    settings = tmp_settings({
        "database": {"path": str(domain_a.db_path)},
        "eval": {"baselines": [], "now_anchor": _ANCHOR},
    })
    shutil.copytree(artifacts_dir(domain_a_built), artifacts_dir(settings))
    eval_dir = settings.path("paths.eval")
    questions = load_questions(domain_a.questions_csv, "question", [])
    categories = categorize(questions, scripted_llm, eval_dir / "categories.yaml")
    con = connect_readonly(domain_a.db_path)
    references = build_references(questions, con, resolve_anchor(settings), settings)
    con.close()
    answers = run_config(
        questions, lambda: ReferenceAgent(settings, llm=scripted_llm, embedder=FakeEmbedder()), RUN_LABEL, eval_dir
    )

    scores = score_config(
        questions, categories, references, answers, Judge(scripted_llm), scripted_llm, settings, RUN_LABEL,
    )
    report_path = eval_dir / "report.md"
    write_report(
        questions, categories, {RUN_LABEL: scores}, {RUN_LABEL: answers}, report_path, quality={}, database="domain_a",
    )

    assert all(score["exec"]["status"] == "skipped" and score["correctness"] is None for score in scores.values())
    assert all(score["pairwise"] == {} for score in scores.values())
    assert scores[_ADVERSARIAL_QID]["behaviour"]["pass"] is True
    assert "## Pairwise vs" not in report_path.read_text()
