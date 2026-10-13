"""Tests for the per-run operational log."""
from ctxlayer.common.runlog import operation, run_logger




def test_only_traced_calls_get_payloads_with_their_caller(tmp_settings):
    import json

    from ctxlayer.common.runlog import activate_run, finish_run, traced

    @traced
    def _post(prompt: str) -> dict:
        return {"response": prompt.upper()}

    # A caller defined as if it lived in a pipeline module (the test module itself is not one).
    scope = {"__name__": "ctxlayer.unstructured.extract", "_post": _post}
    exec("def extract_one(text):\n    return _post(text)\n", scope)
    extract_one = scope["extract_one"]

    settings = tmp_settings()
    activate_run()
    try:
        logger = run_logger(settings)
        with operation(logger, "build_unstructured"):
            assert extract_one("hi") == {"response": "HI"}
    finally:
        finish_run()

    bundle = next(path for path in settings.path("paths.runs").iterdir() if path.is_dir())
    events = [json.loads(line) for line in (bundle / "events.jsonl").read_text().splitlines()]
    calls = [item for item in events if item["kind"] == "function"]
    assert [item["status"] for item in events if item["kind"] == "operation"] == ["started", "completed"]
    assert len(calls) == 1 and calls[0]["function"].endswith("._post")
    assert calls[0]["caller"] == "extract.extract_one"
    payload = json.loads((bundle / calls[0]["payload_ref"]).read_text())
    assert payload["input"] == {"prompt": "hi"} and payload["output"] == {"response": "HI"}
