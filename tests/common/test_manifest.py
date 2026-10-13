"""Tests for the per-stage build manifest."""
import hashlib
import json

import duckdb
import pytest

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.manifest import schema_fingerprint, write_manifest


@pytest.fixture
def con(tmp_path):
    path = tmp_path / "m.duckdb"
    rw = duckdb.connect(str(path))
    rw.execute("CREATE TABLE a (x INTEGER, y VARCHAR)")
    rw.execute("CREATE TABLE b (z DATE)")
    rw.close()
    connection = connect_readonly(path)
    yield connection
    connection.close()




def test_schema_fingerprint_changes_when_schema_changes(tmp_path):
    def fingerprint(ddl):
        path = tmp_path / f"{abs(hash(ddl))}.duckdb"
        rw = duckdb.connect(str(path))
        rw.execute(ddl)
        rw.close()
        ro = connect_readonly(path)
        try:
            return schema_fingerprint(ro)["sha256"]
        finally:
            ro.close()

    assert fingerprint("CREATE TABLE t (a INTEGER)") != fingerprint(
        "CREATE TABLE t (a BIGINT)"
    )


def test_write_manifest_records_stage_fields_and_artifact_hashes(tmp_settings, con):
    settings = tmp_settings()
    out = artifacts_dir(settings)
    out.mkdir(parents=True)
    (out / "profile.json").write_text('{"k": 1}')
    (out / "sub").mkdir()
    (out / "sub" / "data.txt").write_text("hello")

    path = write_manifest(
        out,
        "profile",
        settings=settings,
        con=con,
        models={"llm": "m1", "embeddings": "e1"},
        extra={"rows": 5},
    )

    assert path == out / "build_manifest.json"
    stage = json.loads(path.read_text())["stages"]["profile"]
    assert stage["artifacts"] == {
        "profile.json": hashlib.sha256(b'{"k": 1}').hexdigest(),
        "sub/data.txt": hashlib.sha256(b"hello").hexdigest(),
    }
    assert stage["models"] == {"llm": "m1", "embeddings": "e1"}
    assert stage["extra"] == {"rows": 5}
    assert stage["db_schema"] == schema_fingerprint(con)
    assert stage["config_sha256"] == hashlib.sha256(
        json.dumps(settings.raw, sort_keys=True).encode()
    ).hexdigest()
    assert stage["built_at"]


def test_write_manifest_rewriting_a_stage_keeps_other_stages(tmp_settings, con):
    settings = tmp_settings()
    out = artifacts_dir(settings)
    out.mkdir(parents=True)
    (out / "f.txt").write_text("v1")

    write_manifest(out, "one", settings=settings, con=con, models={})
    write_manifest(out, "two", settings=settings, con=con, models={})
    (out / "f.txt").write_text("v2")
    path = write_manifest(out, "one", settings=settings, con=con, models={"llm": "x"})

    stages = json.loads(path.read_text())["stages"]
    assert set(stages) == {"one", "two"}
    assert stages["one"]["artifacts"]["f.txt"] == hashlib.sha256(b"v2").hexdigest()
    assert stages["one"]["models"] == {"llm": "x"}
    assert stages["two"]["artifacts"]["f.txt"] == hashlib.sha256(b"v1").hexdigest()
    assert stages["two"]["models"] == {}


def test_write_manifest_does_not_hash_itself(tmp_settings, con):
    settings = tmp_settings()
    out = artifacts_dir(settings)
    out.mkdir(parents=True)

    write_manifest(out, "one", settings=settings, con=con, models={})
    path = write_manifest(out, "two", settings=settings, con=con, models={})

    stages = json.loads(path.read_text())["stages"]
    assert "build_manifest.json" not in stages["two"]["artifacts"]
    assert stages["two"]["extra"] == {}


def test_write_manifest_skips_extraction_cache_files(tmp_settings, con):
    settings = tmp_settings()
    out = artifacts_dir(settings)
    cache = out / "docs" / "extract_cache"
    cache.mkdir(parents=True)
    (cache / "cached.json").write_text('{"events": []}')
    (out / "docs" / "documents.jsonl").write_text('{"doc_id": "d1"}\n')

    path = write_manifest(out, "unstructured", settings=settings, con=con, models={})
    hashes = json.loads(path.read_text())["stages"]["unstructured"]["artifacts"]

    assert hashes == {"docs/documents.jsonl": hashlib.sha256(b'{"doc_id": "d1"}\n').hexdigest()}
