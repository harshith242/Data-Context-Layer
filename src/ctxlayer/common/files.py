"""JSON and YAML read/write helpers for build artifacts."""
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import yaml


@contextmanager
def atomic_file(path: Path) -> Iterator[Path]:
    """Yield a temp path beside `path`; renamed over `path` when the block ends, removed when it raises.

    A build stopped mid-write (Ctrl+C) then leaves the old file or none, never a half-written one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        yield temp
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _write_text(path: Path, text: str) -> Path:
    # The caller serialized `text` first, so a failing dump never touches the target.
    with atomic_file(path) as temp:
        temp.write_text(text)
    return path


def write_json(path: Path, data) -> Path:
    """Sorted keys, indent 2, parent folders created; dates and other objects become strings."""
    return _write_text(path, json.dumps(data, indent=2, sort_keys=True, default=str))


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    """Read non-blank JSON Lines records from `path`."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> Path:
    """Write one JSON object per line, creating parent folders when needed."""
    return _write_text(path, "".join(json.dumps(record, ensure_ascii=False, default=str) + "\n" for record in records))


def write_yaml(path: Path, data) -> Path:
    """Keys keep insertion order, parent folders created."""
    return _write_text(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def read_yaml(path: Path):
    return yaml.safe_load(path.read_text())
