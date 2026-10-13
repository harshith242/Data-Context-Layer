"""Per-run logs for CLI execution: stage start/end events, and a full payload for each LLM call."""
import functools
import hashlib
import inspect
import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path

from ctxlayer.common.config import Settings

_LOGGER_NAME = "ctxlayer.run"
_LOG_FILE = "ctx.log"
_EVENTS_FILE = "events.jsonl"
_PAYLOADS_DIR = "payloads"
_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_PACKAGE_PREFIX = "ctxlayer."
# Frames skipped when naming who made a traced call: the client, this module and the thread pool helper.
_PLUMBING_MODULES = {"ctxlayer.common.llm", __name__, "ctxlayer.common.parallel"}


@dataclass
class _Call:
    call_id: int
    caller: str
    function: str
    input: dict
    started: float


@dataclass
class _Run:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    root: Path | None = None
    next_call_id: int = 1
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _State:
    run: _Run | None = None
    trace_enabled: bool = False


_state = _State()


def activate_run() -> None:
    """Enable payload tracing (of `traced` functions) for the current CLI invocation."""
    _state.run = _Run()
    _state.trace_enabled = True


def finish_run() -> None:
    """Stop tracing after the command completes, leaving the run bundle immutable."""
    _state.trace_enabled = False
    _state.run = None


def traced(fn):
    """Record each call's input, output and caller as a payload file while a run is active (used for LLM calls)."""
    signature = inspect.signature(fn)
    name = f"{fn.__module__}.{fn.__qualname__}"

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _tracing():
            return fn(*args, **kwargs)

        call = _start_call(name, signature.bind(*args, **kwargs).arguments, _caller(sys._getframe(1)))
        try:
            output = fn(*args, **kwargs)
        except Exception as error:
            _write_payload(call, {"exception_type": type(error).__name__, "message": str(error)}, "failed")
            raise
        _write_payload(call, output, "completed")
        return output

    return wrapper


def run_logger(settings: Settings) -> logging.Logger:
    """Logger writing summaries and linked payload bundles under `<paths.runs>`."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    runs_dir = settings.path("paths.runs")
    _configure_summary_log(logger, runs_dir / _LOG_FILE)
    _configure_bundle(runs_dir)
    return logger


@contextmanager
def operation(logger: logging.Logger, name: str, **fields):
    """Log an operation's metadata and elapsed time."""
    _event(logger, name, "started", **fields)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        _event(logger, name, "failed", elapsed_s=_elapsed(started), **fields)
        raise
    _event(logger, name, "completed", elapsed_s=_elapsed(started), **fields)


def event(logger: logging.Logger, name: str, **fields) -> None:
    """Write one aggregate progress event to the summary and metadata logs."""
    _event(logger, name, "progress", **fields)


def progress(name: str, completed: int, total: int) -> None:
    """Log completed and total work items for a long-running batch."""
    _event(logging.getLogger(_LOGGER_NAME), name, "progress", completed=completed, total=total)


def _configure_summary_log(logger: logging.Logger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _has_handler(logger, path):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    os.chmod(path, 0o600)


def _configure_bundle(runs_dir: Path) -> None:
    if not _state.trace_enabled or _state.run is None or _state.run.root is not None:
        return
    root = runs_dir / _state.run.run_id
    payloads = root / _PAYLOADS_DIR
    payloads.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    os.chmod(payloads, 0o700)
    _state.run.root = root


def _has_handler(logger: logging.Logger, path: Path) -> bool:
    target = str(path.resolve())
    return any(isinstance(handler, logging.FileHandler) and handler.baseFilename == target for handler in logger.handlers)


def _event(logger: logging.Logger, name: str, status: str, **fields) -> None:
    values = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("operation=%s status=%s%s", name, status, f" {values}" if values else "")
    _write_event({"kind": "operation", "operation": name, "status": status, "fields": fields})


def _tracing() -> bool:
    return _state.trace_enabled and _state.run is not None and _state.run.root is not None


def _caller(frame) -> str:
    """Nearest pipeline function up the stack, e.g. "extract._extract_or_error" (lambdas and plumbing skipped)."""
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        named = "<" not in frame.f_code.co_name
        if module.startswith(_PACKAGE_PREFIX) and module not in _PLUMBING_MODULES and named:
            return f"{module.rsplit('.', 1)[-1]}.{frame.f_code.co_name}"
        frame = frame.f_back
    return "unknown"


def _start_call(function: str, arguments: dict, caller: str) -> _Call:
    assert _state.run is not None
    with _state.run.lock:
        call_id = _state.run.next_call_id
        _state.run.next_call_id += 1
    return _Call(call_id, caller, function, arguments, time.perf_counter())


def _write_payload(call: _Call, output, status: str) -> None:
    if _state.run is None or _state.run.root is None:
        return
    payload = {
        "run_id": _state.run.run_id,
        "call_id": call.call_id,
        "caller": call.caller,
        "function": call.function,
        "status": status,
        "input": _jsonable(call.input),
        "output": _jsonable(output),
        "elapsed_s": _elapsed(call.started),
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, default=_fallback)
    relative = f"{_PAYLOADS_DIR}/{call.call_id:08d}.json"
    path = _state.run.root / relative
    path.write_text(encoded + "\n")
    os.chmod(path, 0o600)
    _write_event({
        "kind": "function",
        "run_id": _state.run.run_id,
        "call_id": call.call_id,
        "caller": call.caller,
        "function": call.function,
        "status": status,
        "elapsed_s": payload["elapsed_s"],
        "payload_ref": relative,
        "payload_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "input_bytes": len(json.dumps(payload["input"], ensure_ascii=False, default=_fallback).encode()),
        "output_bytes": len(json.dumps(payload["output"], ensure_ascii=False, default=_fallback).encode()),
    })


def _write_event(record: dict) -> None:
    if _state.run is None or _state.run.root is None:
        return
    path = _state.run.root / _EVENTS_FILE
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=_fallback) + "\n")
    os.chmod(path, 0o600)


def _jsonable(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=_fallback))
    except (TypeError, ValueError, RecursionError):
        return {"repr": repr(value)}


def _fallback(value):
    if is_dataclass(value):
        return value.__dict__
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict(orient="records")
        except TypeError:
            return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    return repr(value)


def _elapsed(started: float) -> str:
    return f"{time.perf_counter() - started:.3f}"
