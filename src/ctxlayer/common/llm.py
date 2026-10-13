"""Ollama LLM client: one model, native tool calling over /api/chat."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Protocol

import requests

from ctxlayer.common.config import Settings
from ctxlayer.common.runlog import traced

_log = logging.getLogger("llm")

_HTTP_CLIENT_ERROR = 400
_HTTP_SERVER_ERROR = 500
_CHAT_PATH = "/api/chat"
DEFAULT_MODEL_NAME = "default"  # the model name of an LLM that has none configured (e.g. a test fake)
_EMPTY_SCHEMA = {"type": "object", "properties": {}}


class LLMError(RuntimeError):
    """Ollama unreachable, misconfigured or answering with an error."""


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall]
    latency_s: float
    prompt_chars: int
    reply_chars: int


class LLM(Protocol):
    def complete(self, prompt: str, tools: list[dict] | None = None) -> Reply: ...


class OllamaLLM:
    """Client for a local Ollama server (single user prompt in, thinking off)."""

    def __init__(self, base_url: str, model: str, timeout: float, retries: int, options: dict | None = None):
        self._url = base_url.rstrip("/") + _CHAT_PATH
        self._timeout = timeout
        self._retries = retries
        self._options = options
        self.model_name = model

    @classmethod
    def from_settings(cls, settings: Settings) -> "OllamaLLM":
        return cls(
            base_url=settings.get("llm.base_url"),
            model=settings.get("llm.model"),
            timeout=float(settings.get("llm.timeout_s")),
            retries=settings.get("llm.retries"),
            options=settings.get("llm.options"),
        )

    def complete(self, prompt: str, tools: list[dict] | None = None) -> Reply:
        start = time.perf_counter()
        message = self._post(prompt, tools).get("message") or {}
        text = message.get("content") or ""
        calls = _native_calls(message)

        # A tool-call turn has no text: count the call itself, e.g. {"name": "query_metric", ...}.
        reply_chars = len(text) + sum(len(json.dumps({"name": call.name, "args": call.args})) for call in calls)

        return Reply(
            text="" if calls else text,
            tool_calls=calls,
            latency_s=time.perf_counter() - start,
            prompt_chars=len(prompt),
            reply_chars=reply_chars,
        )

    @traced
    def _post(self, prompt: str, tools: list[dict] | None = None) -> dict:
        """POST to Ollama; network errors and 5xx are retried, 4xx raise at once."""
        body: dict = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
        }
        if self._options:
            body["options"] = self._options
        if tools:
            body["tools"] = _convert_tools(tools)

        failure = ""
        for _ in range(self._retries + 1):
            try:
                resp = requests.post(self._url, json=body, timeout=self._timeout)
            except requests.RequestException as exc:
                failure = f"request failed: {exc}"
                continue

            if resp.status_code >= _HTTP_SERVER_ERROR:
                failure = f"ollama returned {resp.status_code}"
                continue
            if resp.status_code >= _HTTP_CLIENT_ERROR:
                raise LLMError(f"ollama returned {resp.status_code}: {resp.text[:300]}")

            return _json_body(resp)

        raise LLMError(f"{failure} (after {self._retries + 1} attempts)")


def extract_json(text: str) -> dict | list | None:
    """First JSON object/array in `text`; tolerates ``` fences and surrounding prose."""
    decoder = json.JSONDecoder()

    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        return value

    return None


def _json_body(resp) -> dict:
    try:
        data = resp.json()
    except ValueError as exc:
        raise LLMError(f"ollama reply is not JSON: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Neutral {name, description, parameters} -> Ollama's function-tool shape."""
    converted = []

    for tool in tools:
        parameters = tool.get("parameters")
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": parameters if isinstance(parameters, dict) else _EMPTY_SCHEMA,
                },
            }
        )

    return converted


def _native_calls(message: dict) -> list[ToolCall]:
    """Tool calls from message.tool_calls; malformed entries are skipped."""
    calls = []

    for entry in message.get("tool_calls") or []:
        function = entry.get("function") if isinstance(entry, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            _log.warning("ignoring malformed tool-call entry: %r", entry)
            continue
        calls.append(ToolCall(function["name"], _native_args(function)))

    return calls


def _native_args(function: dict) -> dict:
    # Ollama sends arguments as an object; some models send a JSON string.
    args = function.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            # The tool's own error envelope will steer the model to a retry.
            _log.warning("unparseable tool-call arguments: %r", args)
            return {}

    return args if isinstance(args, dict) else {}
