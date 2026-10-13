"""Test doubles for the LLM and embedder boundaries."""
import hashlib
import json
import re

import numpy as np

from ctxlayer.common.llm import Reply

_TAG = re.compile(r"\[task:\s*([^\]]+?)\s*\]")
_WORD = re.compile(r"[a-z0-9]+")


class FakeLLM:
    """Routes by the prompt's first-line tag `[task: X]`. A handler is a str, a Reply,
    a list (consumed in order, last one repeats) or a callable(prompt, tools) -> str|Reply."""

    def __init__(self, handlers: dict[str, object]):
        self._handlers = handlers
        self._cursor: dict[str, int] = {}
        self.calls: list[tuple[str, list | None]] = []

    def complete(self, prompt: str, tools: list[dict] | None = None) -> Reply:
        self.calls.append((prompt, tools))
        tag = self._tag(prompt)
        handler = self._handlers.get(tag)
        if handler is None:
            raise AssertionError(f"FakeLLM has no handler for task {tag!r}")

        item = self._next_item(tag, handler)
        if callable(item):
            item = item(prompt, tools)
        if isinstance(item, Reply):
            return item
        return Reply(text=item, tool_calls=[], latency_s=0.0, prompt_chars=len(prompt), reply_chars=len(item))

    @staticmethod
    def _tag(prompt: str) -> str:
        match = _TAG.fullmatch(prompt.split("\n", 1)[0].strip())
        if not match:
            raise AssertionError(f"FakeLLM prompt must start with a [task: X] line, got {prompt[:60]!r}")
        return match.group(1)

    def _next_item(self, tag: str, handler):
        if not isinstance(handler, list):
            return handler

        position = self._cursor.get(tag, 0)
        self._cursor[tag] = position + 1
        return handler[min(position, len(handler) - 1)]


class FakeEmbedder:
    """Hashed bag of lowercase words, L2-normalised: shared words raise the cosine."""

    def __init__(self, dim: int = 64, fingerprint: str = "fake-v1"):
        self.dim = dim
        self.cache_fingerprint = fingerprint

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)

        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                # md5, not hash(): stable across processes.
                column = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                matrix[row, column] += 1.0

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_passages(texts)


def per_citation(judge):
    """FakeLLM handler for a grouped [task: citation_check] prompt: judge(block) per "[Cn] ..." block."""
    def reply(prompt, tools):
        blocks = re.split(r"\n(?=\[C\d+\] )", prompt)
        return json.dumps({b.split("]", 1)[0][1:]: judge(b) for b in blocks if re.match(r"\[C\d+\] ", b)})
    return reply
