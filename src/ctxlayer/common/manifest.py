"""Per-stage build manifest: schema, config, models and artifact hashes."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ctxlayer.common.config import Settings
from ctxlayer.common.db import list_tables, table_schema

_MANIFEST_NAME = "build_manifest.json"
_EXTRACT_CACHE_DIR = "extract_cache"


def schema_fingerprint(con) -> dict:
    """{"tables": {table: [[column, type], ...]}, "sha256": hash of tables}."""
    tables = {
        table: [[name, data_type] for name, data_type in table_schema(con, table)]
        for table in list_tables(con)
    }
    return {"tables": tables, "sha256": _sha256_json(tables)}


def write_manifest(
    artifacts_dir: Path,
    stage: str,
    *,
    settings: Settings,
    con,
    models: dict,
    extra: dict | None = None,
) -> Path:
    """Replace `stage` in artifacts_dir/build_manifest.json; other stages stay."""
    path = artifacts_dir / _MANIFEST_NAME
    manifest = json.loads(path.read_text()) if path.is_file() else {"stages": {}}

    manifest["stages"][stage] = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "db_schema": schema_fingerprint(con),
        "config_sha256": _sha256_json(settings.raw),
        "models": models,
        "artifacts": _hash_artifacts(artifacts_dir),
        "extra": extra or {},
    }

    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def _sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _hash_artifacts(artifacts_dir: Path) -> dict[str, str]:
    # {"sub/data.txt": sha256}; the manifest itself is skipped.
    hashes = {}
    for file in sorted(artifacts_dir.rglob("*")):
        if not file.is_file() or file.name == _MANIFEST_NAME:
            continue
        relative = file.relative_to(artifacts_dir)
        if _EXTRACT_CACHE_DIR in relative.parts:
            continue
        hashes[relative.as_posix()] = hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
    return hashes
