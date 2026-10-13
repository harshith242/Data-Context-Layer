import logging
from types import SimpleNamespace

import pytest

from ctxlayer import cli
from ctxlayer.pipeline.phases import STRUCTURED_ONLY
from ctxlayer.pipeline.runner import Review

_RESULT = {"ran": ["links"], "skipped": [], "paused_after": None, "llm": {"hits": 0, "misses": 0}}


@pytest.fixture
def calls(monkeypatch) -> list:
    """Every run_build call the CLI makes, as (positional, keyword) arguments."""
    made = []
    monkeypatch.setattr(cli, "load_settings", lambda overrides=None: SimpleNamespace())
    monkeypatch.setattr(cli, "_models", lambda settings: "context")
    monkeypatch.setattr(cli, "run_logger", lambda settings: logging.getLogger("test.cli"))
    monkeypatch.setattr(cli, "run_build", lambda *args, **kwargs: made.append((args, kwargs)) or _RESULT)
    return made


def test_build_passes_its_options_to_run_build(calls):
    assert cli.main(["build", "--human-review", "--from", "roles", "--to", "ontology"]) == 0
    assert calls == [(("context", "roles", "ontology", Review.STOP), {"force_start": True})]


@pytest.mark.parametrize("command, first, last, skip", [
    ("build-structured", "profile", "moves", frozenset()),
    ("build-unstructured", "documents", "graph", STRUCTURED_ONLY),
    ("build-links", "links", None, frozenset()),
])
def test_the_old_commands_are_ranges_of_build(calls, command, first, last, skip):
    assert cli.main([command]) == 0
    assert calls == [(("context", first, last, Review.OFF), {"force_start": True, "skip": skip})]
