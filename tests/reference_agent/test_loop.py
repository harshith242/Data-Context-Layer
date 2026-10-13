"""Reference agent loop and public answer() on the synthetic domain, with a scripted LLM."""
import hashlib
import json
import re
import shutil

import duckdb
import pytest

from ctxlayer.common.config import load_settings, artifacts_dir
from ctxlayer.common.db import SqlRejected, connect_readonly, run_select
from ctxlayer.common.llm import LLMError, Reply, ToolCall
from ctxlayer.reference_agent.api import ReferenceAgent, answer
from ctxlayer.reference_agent.loop import run_loop
from ctxlayer.serving._errors import error_result
from ctxlayer.serving.tools import Tool
from fakes import FakeEmbedder, FakeLLM

_QUESTION = "How many quotes are there?"
_COUNT_SQL = "SELECT COUNT(*) AS n FROM grain_quotes"
_MAX_RESULT_CHARS = 6000
_PROCEED = '{"action": "proceed", "flags": [], "reason": "scripted"}'


def _calls(*calls: tuple[str, dict]) -> Reply:
    """A model turn that only asks for tools."""
    return Reply("", [ToolCall(name, args) for name, args in calls], 0.0, 100, 20)


def _final(text: str) -> Reply:
    return Reply(text, [], 0.0, 100, len(text))


def _count(domain_a) -> int:
    con = connect_readonly(domain_a.db_path)
    try:
        return con.execute(_COUNT_SQL).fetchone()[0]
    finally:
        con.close()


@pytest.fixture
def settings(tmp_settings, domain_a):
    return tmp_settings({"database": {"path": str(domain_a.db_path)}})


@pytest.fixture
def tools(settings):
    """One read-only SQL tool over the raw database, so the loop has a real tool to call."""
    con = connect_readonly(settings.path("database.path"))

    def select_rows(sql: str) -> dict:
        try:
            result = run_select(con, sql, 100)
        except SqlRejected as exc:
            return error_result("sql_rejected", str(exc))
        return {
            "columns": result.columns, "rows": result.rows, "row_count": result.row_count,
            "truncated": result.truncated, "_citations": [{"kind": "sql", "sql": result.sql}],
        }

    parameters = {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}
    yield [Tool("select_rows", "Run one SELECT.", parameters, select_rows)]
    con.close()


def _run(llm, tools, max_tool_calls=8, max_result_chars=_MAX_RESULT_CHARS):
    return run_loop(_QUESTION, "SYSTEM TEXT", tools, llm, max_tool_calls, max_result_chars)


def test_tool_result_is_fed_back_and_the_text_reply_is_the_answer(tools, domain_a):
    total = _count(domain_a)
    llm = FakeLLM({"answer": [_calls(("select_rows", {"sql": _COUNT_SQL})), _final(f"There are {total} quotes.")]})

    result = _run(llm, tools)

    assert result.text == f"There are {total} quotes."
    assert result.error is None
    assert result.citations == [{"kind": "sql", "sql": _COUNT_SQL}]
    assert [(entry["tool"], entry["status"]) for entry in result.tool_trace] == [("select_rows", "ok")]
    assert result.tool_trace[0]["rows"] == [[total]]
    assert result.tool_trace[0]["args"] == {"sql": _COUNT_SQL}

    first_prompt, first_tools = llm.calls[0]
    second_prompt, _ = llm.calls[1]
    assert first_prompt.splitlines()[0] == "[task: answer]"
    assert "SYSTEM TEXT" in first_prompt
    assert first_prompt.endswith(f"User: {_QUESTION}\nAssistant:")
    assert [tool["name"] for tool in first_tools] == ["select_rows"]
    assert f'Assistant (tool call): {json.dumps({"name": "select_rows", "args": {"sql": _COUNT_SQL}})}' in second_prompt
    assert f'Tool result (select_rows): {{"columns": ["n"], "rows": [[{total}]]' in second_prompt
    assert "_citations" not in second_prompt


def test_identical_repeat_is_not_executed_and_the_model_is_told_to_move_on():
    executed = []
    probe = Tool("probe", "d", {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
                 lambda x: executed.append(x) or {"seen": x})
    llm = FakeLLM({"answer": [_calls(("probe", {"x": 1})), _calls(("probe", {"x": 1})), _final("done")]})

    result = _run(llm, [probe])

    assert executed == [1]
    assert [entry["status"] for entry in result.tool_trace] == ["ok", "repeat"]
    assert "you already made this exact call" in llm.calls[2][0]
    assert result.text == "done"


def test_repeat_calls_still_count_against_the_budget():
    probe = Tool("probe", "d", {"type": "object", "properties": {}}, lambda: {"seen": True})
    llm = FakeLLM({"answer": lambda prompt, tools: _calls(("probe", {})) if tools else _final("gap noted")})

    result = _run(llm, [probe], max_tool_calls=3)

    assert [entry["status"] for entry in result.tool_trace] == ["ok", "repeat", "repeat"]
    assert result.text == "gap noted"


def test_budget_exhaustion_forces_a_final_answer_without_tools(tools):
    counter = iter(range(100))

    def endless(prompt, offered):
        if offered is None:
            return _final("Partial: the data gap is unknown.")
        return _calls(("select_rows", {"sql": f"SELECT {next(counter)} AS n"}))

    llm = FakeLLM({"answer": endless})

    result = _run(llm, tools, max_tool_calls=2)

    assert len(result.tool_trace) == 2
    assert len(llm.calls) == 3
    last_prompt, last_tools = llm.calls[-1]
    assert last_tools is None
    assert "tool budget exhausted" in last_prompt
    assert last_prompt.endswith("Assistant:")
    assert result.text == "Partial: the data gap is unknown."
    assert result.error is None


def test_calls_beyond_the_budget_within_one_turn_are_dropped(tools):
    batch = _calls(*[("select_rows", {"sql": f"SELECT {i} AS n"}) for i in range(5)])
    llm = FakeLLM({"answer": lambda prompt, offered: batch if offered else _final("stopped")})

    result = _run(llm, tools, max_tool_calls=2)

    assert [entry["args"]["sql"] for entry in result.tool_trace] == ["SELECT 0 AS n", "SELECT 1 AS n"]
    assert result.text == "stopped"


def test_unknown_tool_becomes_an_error_result_not_an_exception(tools):
    llm = FakeLLM({"answer": [_calls(("drop_everything", {})), _final("sorry")]})

    result = _run(llm, tools)

    assert [(entry["tool"], entry["status"]) for entry in result.tool_trace] == [("drop_everything", "error")]
    assert '"type": "unknown_tool"' in llm.calls[1][0]
    assert "select_rows" in llm.calls[1][0]
    assert result.text == "sorry"


@pytest.mark.parametrize(
    "args, fragment",
    [({}, "sql"), ({"sql": _COUNT_SQL, "limit": 3}, "limit")],
    ids=["missing_required", "unknown_argument"],
)
def test_bad_arguments_become_a_bad_args_error_result(tools, args, fragment):
    llm = FakeLLM({"answer": [_calls(("select_rows", args)), _final("sorry")]})

    result = _run(llm, tools)

    assert result.tool_trace[0]["status"] == "error"
    assert '"type": "bad_args"' in llm.calls[1][0]
    assert fragment in llm.calls[1][0]
    assert result.citations == []


def test_a_rejected_write_comes_back_as_an_error_dict(tools, domain_a):
    llm = FakeLLM({"answer": [_calls(("select_rows", {"sql": "DROP TABLE grain_quotes"})), _final("refused")]})

    result = _run(llm, tools)

    assert result.tool_trace[0]["status"] == "error"
    assert '"type": "sql_rejected"' in llm.calls[1][0]
    assert _count(domain_a) > 0


def test_a_tool_that_raises_does_not_crash_the_loop():
    broken = Tool("broken", "d", {"type": "object", "properties": {}}, lambda: 1 / 0)
    llm = FakeLLM({"answer": [_calls(("broken", {})), _final("recovered")]})

    result = _run(llm, [broken])

    assert result.tool_trace[0]["status"] == "error"
    assert "ZeroDivisionError" in llm.calls[1][0]
    assert result.text == "recovered"


def test_long_tool_results_are_trimmed_in_the_prompt(tools):
    llm = FakeLLM({"answer": [_calls(("select_rows", {"sql": "SELECT * FROM grain_quotes"})), _final("ok")]})

    _run(llm, tools, max_result_chars=300)

    line = next(row for row in llm.calls[1][0].splitlines() if row.startswith("Tool result (select_rows): "))
    assert line.startswith('Tool result (select_rows): {"columns": ["quote_id"')
    assert len(line) < 300 + len("Tool result (select_rows): ") + 60
    assert "truncated" in line


def test_trace_keeps_driver_quote_when_series_exceeds_limit():
    quote = "Port closure delayed exports."
    tool = Tool(
        "explain_move", "d", {"type": "object", "properties": {}},
        lambda: {"drivers": [{"quote": quote}], "series": [{"value": "x" * 200}]},
    )
    llm = FakeLLM({"answer": [_calls(("explain_move", {})), _final("ok")]})

    result = _run(llm, [tool], max_result_chars=100)

    assert result.tool_trace[0]["result"] == {"drivers": [{"quote": quote}]}


def test_llm_failure_becomes_an_error_answer_keeping_the_trace(tools):
    def fail_after_first_call(prompt, offered):
        if "Tool result" in prompt:
            raise LLMError("ollama returned 502")
        return _calls(("select_rows", {"sql": _COUNT_SQL}))

    result = _run(FakeLLM({"answer": fail_after_first_call}), tools)

    assert result.error == "ollama returned 502"
    assert result.text == ""
    assert len(result.tool_trace) == 1
    assert result.citations == [{"kind": "sql", "sql": _COUNT_SQL}]


def test_reference_agent_writes_a_trace_file(agent_settings):
    question = "Which question leaves a trace file?"  # unique: the runs folder is shared by the session
    llm = FakeLLM({"guardrail": _PROCEED, "answer": "42 quotes"})
    agent = ReferenceAgent(agent_settings, llm=llm, embedder=FakeEmbedder())
    try:
        result = agent.answer(question)
    finally:
        agent.close()

    assert result.text == "42 quotes"
    digest = hashlib.sha1(question.encode()).hexdigest()[:8]
    files = list(agent_settings.path("paths.runs").glob(f"*_{digest}.json"))
    assert len(files) == 1
    assert re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{8}\.json", files[0].name)
    trace = json.loads(files[0].read_text())
    assert trace["question"] == question
    assert trace["guardrail"]["action"] == "proceed"
    assert trace["text"] == "42 quotes"
    assert trace["error"] is None


def test_agent_close_and_public_answer_release_the_database_file(agent_settings, domain_a, tmp_path):
    db_copy = tmp_path / "copy.duckdb"
    shutil.copy(domain_a.db_path, db_copy)
    # The artifacts come from the domain_a database; only the file the agent opens is the copy.
    settings = load_settings(
        root=agent_settings.root,
        overrides={"database": {"path": str(db_copy)}, "paths": {"artifacts": str(artifacts_dir(agent_settings).parent)}},
    )
    shutil.copytree(artifacts_dir(agent_settings), artifacts_dir(settings), dirs_exist_ok=True)
    llm = FakeLLM({"guardrail": _PROCEED, "answer": "hi"})
    agent = ReferenceAgent(settings, llm=llm, embedder=FakeEmbedder())

    # A read-write open is refused while a read-only connection to the file is alive.
    with pytest.raises(duckdb.Error):
        duckdb.connect(str(db_copy))

    agent.close()
    duckdb.connect(str(db_copy)).close()

    answer(_QUESTION, settings=settings, llm=llm, embedder=FakeEmbedder())
    duckdb.connect(str(db_copy)).close()
