"""Tests for the Ollama LLM client: request shape, tool calls, retries, JSON extraction."""
import pytest
import requests

from ctxlayer.common import llm as llm_module
from ctxlayer.common.llm import LLMError, OllamaLLM, ToolCall, extract_json

_BASE_URL = "http://ollama.test:11434"
_MODEL = "test-model:1b"

_TOOL = {
    "name": "query_sql",
    "description": "Run a query.",
    "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
}


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = str(self._payload)

    def json(self):
        return self._payload


class _Server:
    """Stands in for requests.post: replays queued responses, records requests."""

    def __init__(self, *responses):
        self._queue = list(responses)
        self.requests: list[dict] = []

    def post(self, url, json=None, timeout=None):
        self.requests.append({"url": url, "body": json, "timeout": timeout})
        item = self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def _server(monkeypatch, *responses) -> _Server:
    server = _Server(*responses)
    monkeypatch.setattr(llm_module.requests, "post", server.post)
    return server


def _ok(content="", tool_calls=None) -> _Response:
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return _Response(200, {"message": message})


def _client(retries=1) -> OllamaLLM:
    return OllamaLLM(base_url=_BASE_URL + "/", model=_MODEL, timeout=7.0, retries=retries)


def test_plain_reply_sends_one_user_message_with_thinking_off(monkeypatch):
    server = _server(monkeypatch, _ok("hi"))

    reply = _client().complete("hello")

    sent = server.requests[0]
    assert sent["url"] == _BASE_URL + "/api/chat"
    assert sent["timeout"] == 7.0
    assert sent["body"] == {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "think": False,
    }
    assert (reply.text, reply.tool_calls) == ("hi", [])
    assert (reply.prompt_chars, reply.reply_chars) == (5, 2)


def test_tools_are_sent_in_ollama_function_format(monkeypatch):
    server = _server(monkeypatch, _ok("x"))

    _client().complete("q", [_TOOL])

    assert server.requests[0]["body"]["tools"] == [{"type": "function", "function": _TOOL}]


@pytest.mark.parametrize("arguments", [{"sql": "SELECT 1"}, '{"sql": "SELECT 1"}'])
def test_tool_call_reply_maps_dict_and_string_arguments(monkeypatch, arguments):
    _server(monkeypatch, _ok("", [{"function": {"name": "query_sql", "arguments": arguments}}]))

    reply = _client().complete("q", [_TOOL])

    assert reply.tool_calls == [ToolCall("query_sql", {"sql": "SELECT 1"})]
    assert reply.text == ""


def test_unparseable_string_arguments_become_empty_args(monkeypatch):
    _server(monkeypatch, _ok("", [{"function": {"name": "query_sql", "arguments": "{not json"}}]))

    assert _client().complete("q", [_TOOL]).tool_calls == [ToolCall("query_sql", {})]


def test_5xx_is_retried_then_succeeds(monkeypatch):
    server = _server(monkeypatch, _Response(503), _ok("fine"))

    reply = _client().complete("q")

    assert reply.text == "fine"
    assert len(server.requests) == 2


def test_connection_errors_exhaust_retries(monkeypatch):
    server = _server(monkeypatch, requests.ConnectionError("down"))

    with pytest.raises(LLMError, match="after 3 attempts"):
        _client(retries=2).complete("q")

    assert len(server.requests) == 3


def test_4xx_raises_without_retry(monkeypatch):
    server = _server(monkeypatch, _Response(404, {"error": "model not found"}))

    with pytest.raises(LLMError, match="404"):
        _client().complete("q")

    assert len(server.requests) == 1


def test_from_settings_reads_the_llm_section(tmp_settings, monkeypatch):
    server = _server(monkeypatch, _ok("x"))
    settings = tmp_settings({"llm": {"model": "other:3b", "base_url": "http://h:1", "timeout_s": 9}})

    client = OllamaLLM.from_settings(settings)
    client.complete("q")

    assert client.model_name == "other:3b"
    assert server.requests[0]["url"] == "http://h:1/api/chat"
    assert server.requests[0]["timeout"] == 9.0


# ---------- extract_json ----------

@pytest.mark.parametrize("text, expected", [
    ('```json\n{"a": [1, 2]}\n```', {"a": [1, 2]}),
    ('Sure! Here it is: {"a": {"b": 2}} Hope that helps.', {"a": {"b": 2}}),
    ("no json here", None),
    ('{"a": [1, 2', None),
])
def test_extract_json_reads_the_first_json_value_or_none(text, expected):
    assert extract_json(text) == expected
