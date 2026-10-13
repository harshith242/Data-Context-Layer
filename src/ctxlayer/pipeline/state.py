"""Fingerprints of phase inputs and build_state.json, the record of what each phase last ran with (spec section 7)."""
import contextlib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ctxlayer.common.db import connect_readonly, list_tables, quote_ident
from ctxlayer.common.files import read_json, write_json
from ctxlayer.common.manifest import schema_fingerprint
from ctxlayer.common.phase_context import PhaseContext
from ctxlayer.pipeline.phases import DATABASE, Phase

STATE_FILE = "build_state.json"
PAUSED_AFTER = "paused_after"
HUMAN_REVIEW = "human_review"

_MISSING = "missing"
_SETTINGS = "settings"
_CONTEXT = "context.md"
_DECISIONS = "decisions"
_VERSION = "version"
_DATABASE_PART = "database"
# What a changed part is called in `ctx status`; a changed file is named after itself.
_REASONS = {
    _SETTINGS: "settings changed", _CONTEXT: "context.md changed", _DECISIONS: "decisions changed",
    _VERSION: "phase logic changed", _DATABASE_PART: "database changed",
}


def fingerprint(phase: Phase, pc: PhaseContext) -> str:
    """sha256 over the phase's input file hashes, settings, context text, decisions and version.

    DATABASE as an input adds the schema hash and each table's row count.
    """
    return _sha(json.dumps(parts(phase, pc), sort_keys=True))


def parts(phase: Phase, pc: PhaseContext) -> dict[str, str]:
    """Hash of each fact behind the fingerprint, by name, e.g. {"roles.yaml": "ab12...", "settings": "..."}."""
    found = {}
    for name in phase.inputs:
        found[_DATABASE_PART if name == DATABASE else name] = (
            _database_hash(pc) if name == DATABASE else _file_hash(pc.out_dir / name)
        )

    settings = {key: pc.settings.get(key) for key in phase.settings_keys}
    found[_SETTINGS] = _sha(json.dumps(settings, sort_keys=True, default=str))
    found[_CONTEXT] = _sha(json.dumps([pc.context.for_step(step) for step in phase.steps]))
    found[_DECISIONS] = pc.decisions.fingerprint(set(phase.kinds))
    found[_VERSION] = phase.version
    return found


def changed_reason(saved: dict[str, str], current: dict[str, str]) -> str:
    """"roles.yaml changed, context.md changed": the parts whose hash differs from the saved one."""
    names = [name for name in current if saved.get(name) != current[name]]
    return ", ".join(_REASONS.get(name, f"{name} changed") for name in names)


def load_state(out_dir: Path) -> dict:
    """build_state.json, or an empty state when it is missing or unreadable."""
    try:
        state = read_json(out_dir / STATE_FILE)
    except (OSError, ValueError):
        return {"phases": {}}
    if not isinstance(state, dict) or not isinstance(state.get("phases"), dict):
        return {"phases": {}}
    return state


def save_state(out_dir: Path, state: dict) -> None:
    write_json(out_dir / STATE_FILE, state)


def record(state: dict, phase: Phase, found: dict[str, str], applied: list[str] | None = None) -> None:
    """Note that `phase` ran (or was adopted) with the inputs behind `found`.

    `applied` is the identity of each decision the phase applied (an adopted phase's are unknown: none).
    """
    state["phases"][phase.name] = {
        "fingerprint": _sha(json.dumps(found, sort_keys=True)),
        "parts": found,
        "applied": applied or [],
        "at": datetime.now(timezone.utc).isoformat(),
    }


def applied_ids(state: dict) -> set[str]:
    """Identities of the decisions some phase applied when it last ran."""
    return {ident for saved in state["phases"].values() for ident in saved.get("applied", [])}


def _database_hash(pc: PhaseContext) -> str:
    with contextlib.closing(connect_readonly(pc.settings.path("database.path"))) as con:
        counts = {
            table: con.execute(f"SELECT count(*) FROM {quote_ident(table)}").fetchone()[0]
            for table in list_tables(con)
        }
        return _sha(json.dumps({"schema": schema_fingerprint(con)["sha256"], "rows": counts}, sort_keys=True))


def _file_hash(path: Path) -> str:
    if not path.is_file():
        return _MISSING
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
