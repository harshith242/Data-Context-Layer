"""Tool-calling loop: prompt the model, run its tool calls, feed results back until it answers."""
import json
import time
from dataclasses import dataclass, field
from enum import Enum

from ctxlayer.common.llm import LLM, LLMError, Reply
from ctxlayer.serving._errors import error_result
from ctxlayer.serving.tools import Tool

_TASK_TAG = "[task: answer]"
_TRUNCATED = "... [truncated]"
_CHARS_PER_TOKEN = 4
_CITATIONS_KEY = "_citations"
_EVIDENCE_KEYS = ("hits", "drivers", "edges")
_REPEAT_CALL_NOTE = (
    "Note: you already made this exact call; it was not run again. "
    "Use the earlier result or try something different."
)
_BUDGET_EXHAUSTED_NOTE = "Note: tool budget exhausted; answer with what you know and state the gap."


@dataclass
class Answer:
    text: str
    citations: list[dict]
    tool_trace: list[dict]
    latency_s: float
    tokens: dict
    error: str | None = None


class _Status(Enum):
    OK = "ok"
    ERROR = "error"
    REPEAT = "repeat"  # identical to an earlier call, not run


@dataclass
class _State:
    """What the loop has seen so far; also feeds the final Answer."""

    transcript: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    tool_trace: list[dict] = field(default_factory=list)
    seen_calls: set[str] = field(default_factory=set)
    prompt_chars: int = 0
    reply_chars: int = 0


def run_loop(
    question: str,
    system_prompt: str,
    tools: list[Tool],
    llm: LLM,
    max_tool_calls: int,
    max_result_chars: int,
) -> Answer:
    start = time.perf_counter()
    state = _State()
    by_name = {tool.name: tool for tool in tools}
    schemas = [tool.schema() for tool in tools]
    head = f"{_TASK_TAG}\n{system_prompt}\n\nUser: {question}"

    try:
        text = _converse(head, schemas, by_name, llm, max_tool_calls, max_result_chars, state)
        error = None
    except LLMError as exc:
        text, error = "", str(exc)

    return Answer(
        text=text,
        citations=state.citations,
        tool_trace=state.tool_trace,
        latency_s=time.perf_counter() - start,
        tokens={
            "prompt_chars": state.prompt_chars,
            "reply_chars": state.reply_chars,
            "est_tokens": (state.prompt_chars + state.reply_chars) // _CHARS_PER_TOKEN,
        },
        error=error,
    )


def _converse(head, schemas, by_name, llm, max_tool_calls, max_result_chars, state: _State) -> str:
    """Final answer text; raises LLMError when the LLM fails."""
    while True:
        reply = _complete(llm, head, state, schemas)
        if not reply.tool_calls:
            return reply.text

        # Each call counts against the budget, repeats included, so the loop always ends.
        for call in reply.tool_calls:
            if len(state.tool_trace) >= max_tool_calls:
                break
            _run_call(call, by_name, max_result_chars, state)

        if len(state.tool_trace) >= max_tool_calls:
            break

    # Budget spent: one last turn without tools.
    state.transcript.append(_BUDGET_EXHAUSTED_NOTE)
    return _complete(llm, head, state, None).text


def _complete(llm: LLM, head: str, state: _State, schemas: list[dict] | None) -> Reply:
    prompt = "\n".join([head, *state.transcript, "Assistant:"])
    reply = llm.complete(prompt, schemas)
    state.prompt_chars += reply.prompt_chars
    state.reply_chars += reply.reply_chars
    return reply


def _run_call(call, by_name: dict[str, Tool], max_result_chars: int, state: _State) -> None:
    payload = {"name": call.name, "args": call.args}
    state.transcript.append(f"Assistant (tool call): {json.dumps(payload, default=str)}")

    # Same name + same args (any key order) as an earlier call: skip it, tell the model to move on.
    call_key = json.dumps(payload, sort_keys=True, default=str)
    if call_key in state.seen_calls:
        state.transcript.append(_REPEAT_CALL_NOTE)
        _trace(state, call, _Status.REPEAT)
        return
    state.seen_calls.add(call_key)

    result = _execute(by_name.get(call.name), call, list(by_name))
    for citation in result.pop(_CITATIONS_KEY, []):
        if citation not in state.citations:
            state.citations.append(citation)

    shown = json.dumps(result, default=str)
    if len(shown) > max_result_chars:
        shown = shown[:max_result_chars] + _TRUNCATED
    state.transcript.append(f"Tool result ({call.name}): {shown}")
    _trace(state, call, _Status.ERROR if "error" in result else _Status.OK, result, max_result_chars)


def _trace(state: _State, call, status: _Status, result: dict | None = None, max_result_chars: int = 0) -> None:
    entry = {"tool": call.name, "args": call.args, "status": status.value}

    # Result rows (already row-limited) let the evaluation see what the tool returned.
    if result is not None and "rows" in result:
        entry["rows"] = result["rows"]

    # Document / driver / edge results are kept, trimmed like the transcript, so citations can be checked.
    # Only the evidence keys (not series rows), so a long series never pushes the quotes out.
    if result is not None and any(key in result for key in _EVIDENCE_KEYS):
        evidence = {key: result[key] for key in _EVIDENCE_KEYS if key in result}
        entry["result"] = _trimmed(evidence, max_result_chars)
    state.tool_trace.append(entry)


def _trimmed(result: dict, max_chars: int) -> dict:
    """`result` cut to what the transcript shows: trailing hits / drivers / edges dropped until its JSON fits."""
    kept = dict(result)
    while len(json.dumps(kept, default=str)) > max_chars:
        lists = [key for key in _EVIDENCE_KEYS if isinstance(kept.get(key), list) and kept[key]]
        if not lists:
            break
        longest = max(lists, key=lambda key: len(kept[key]))
        kept[longest] = kept[longest][:-1]
    return kept


def _execute(tool: Tool | None, call, known: list[str]) -> dict:
    """Run one call; any problem comes back as an error result, never an exception."""
    if tool is None:
        return error_result("unknown_tool", f"No tool named {call.name!r}. Available: {known}.")

    properties = tool.parameters.get("properties", {})
    missing = [name for name in tool.parameters.get("required", []) if name not in call.args]
    unknown = [name for name in call.args if name not in properties]
    if missing or unknown:
        return error_result("bad_args", f"missing: {missing}; unknown: {unknown}; allowed: {list(properties)}.")

    # Tools return errors as results; this net is for a model sending a wrong argument type.
    try:
        return tool.fn(**call.args)
    except Exception as exc:
        return error_result("tool_failed", f"{type(exc).__name__}: {exc}")
