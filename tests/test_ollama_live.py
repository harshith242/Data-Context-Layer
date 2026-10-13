"""Live check; run with `pytest -m live` (needs a running Ollama with the configured model)."""
import pytest

from ctxlayer.cli import main

pytestmark = pytest.mark.live


def test_check_llm_reaches_the_configured_ollama_model(capsys):
    assert main(["check-llm"]) == 0
    assert capsys.readouterr().out.startswith("ok ")
