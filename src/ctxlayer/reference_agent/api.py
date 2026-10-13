"""Public entry point: answer a question with the context-layer tools and log the trace."""
import dataclasses
import hashlib
import json
import time
from datetime import datetime, timezone

from ctxlayer.common.config import Settings, load_settings
from ctxlayer.common.embeddings import Embedder
from ctxlayer.common.llm import LLM, OllamaLLM
from ctxlayer.reference_agent.guardrails import Action, Flag, precheck
from ctxlayer.reference_agent.loop import Answer, run_loop
from ctxlayer.reference_agent.prompts import (
    RECOMMENDATION_NOTE, REFUSAL_TEXT, chit_chat_text, data_summary, out_of_scope_text,
    structured_prompt,
)
from ctxlayer.serving.tools import build_tools, open_context

_TRACE_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
_QUESTION_HASH_CHARS = 8
_CHARS_PER_TOKEN = 4


class ReferenceAgent:
    def __init__(self, settings: Settings, llm: LLM | None = None, embedder: Embedder | None = None):
        self._settings = settings
        self._llm = llm or OllamaLLM.from_settings(settings)

        self._ctx = open_context(settings, embedder, self._llm)
        self._tools = build_tools(self._ctx)
        self._system_prompt = structured_prompt(self._ctx, self._tools)
        self._data_summary = data_summary(self._ctx)

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        check = precheck(question, self._llm, self._data_summary)
        if check["action"] != Action.PROCEED.value:
            return self._finish(question, self._canned_answer(check["action"], start), check)

        system_prompt = self._system_prompt
        if Flag.RECOMMENDATION.value in check["flags"]:
            system_prompt = f"{system_prompt}\n\n{RECOMMENDATION_NOTE}"
        return self._finish(question, self._run(question, system_prompt), check)

    def _run(self, question: str, system_prompt: str) -> Answer:
        return run_loop(
            question,
            system_prompt,
            self._tools,
            self._llm,
            self._settings.get("agent.max_tool_calls"),
            self._settings.get("agent.max_tool_result_chars"),
        )

    def close(self) -> None:
        """Release the database connections (the semantic one only if it was opened)."""
        if "con_sem" in vars(self._ctx):
            self._ctx.con_sem.close()
        self._ctx.con.close()

    def _finish(self, question: str, result: Answer, check: dict | None) -> Answer:
        """Add the guardrail call's cost, write the trace."""
        tokens = dict(result.tokens)
        if check:
            tokens["prompt_chars"] += check["prompt_chars"]
            tokens["reply_chars"] += check["reply_chars"]
            tokens["est_tokens"] = (tokens["prompt_chars"] + tokens["reply_chars"]) // _CHARS_PER_TOKEN
        result = dataclasses.replace(result, tokens=tokens)
        self._write_trace(question, result, check)
        return result

    def _canned_answer(self, action: str, start: float) -> Answer:
        """The fixed reply of a guardrail action; no tool runs."""
        if action == Action.REFUSE.value:
            text = REFUSAL_TEXT
        elif action == Action.OUT_OF_SCOPE.value:
            text = out_of_scope_text(self._ctx)
        else:
            text = chit_chat_text(self._ctx)
        tokens = {"prompt_chars": 0, "reply_chars": 0, "est_tokens": 0}
        return Answer(text, [], [], time.perf_counter() - start, tokens)

    def _write_trace(self, question: str, result: Answer, check: dict | None) -> None:
        # runs/<UTC time>_<hash of question>.json, e.g. 20261009T101500Z_1f2e3d4c.json
        stamp = datetime.now(timezone.utc).strftime(_TRACE_TIME_FORMAT)
        digest = hashlib.sha1(question.encode()).hexdigest()[:_QUESTION_HASH_CHARS]
        path = self._settings.path("paths.runs") / f"{stamp}_{digest}.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        trace = {"question": question, "guardrail": check, **dataclasses.asdict(result)}
        path.write_text(json.dumps(trace, indent=2, default=str))


def answer(question: str, **kwargs) -> Answer:
    """One-shot helper; `settings`, `llm`, `embedder` may be passed as keywords."""
    settings = kwargs.pop("settings", None) or load_settings()
    agent = ReferenceAgent(settings, **kwargs)
    try:
        return agent.answer(question)
    finally:
        agent.close()
