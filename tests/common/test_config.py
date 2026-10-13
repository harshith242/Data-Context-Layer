"""Tests for settings loading: lookup, path resolution, overrides, root discovery."""
import os
from pathlib import Path

import pytest
import yaml

from ctxlayer.common.config import LegacyArtifacts, artifacts_dir, load_settings, require_artifacts

_SETTINGS = {
    "paths": {"artifacts": "artifacts", "eval": "eval"},
    "database": {"path": "data/example.duckdb"},
    "eval": {"rel_tolerance": 0.005, "pairwise_passes": 4},
    "env_files": [".env", "missing.env"],
}


def _write_root(root: Path, settings: dict = _SETTINGS) -> Path:
    (root / "config").mkdir(parents=True)
    (root / "config" / "settings.yaml").write_text(yaml.safe_dump(settings))
    return root








def test_overrides_deep_merge_keeps_sibling_keys(tmp_path):
    root = _write_root(tmp_path)

    settings = load_settings(root, overrides={"eval": {"rel_tolerance": 0.1}})

    assert settings.get("eval.rel_tolerance") == 0.1
    assert settings.get("eval.pairwise_passes") == 4
    assert settings.get("database.path") == "data/example.duckdb"


def test_artifacts_dir_is_named_after_the_database_file_stem(tmp_path):
    root = _write_root(tmp_path)

    settings = load_settings(root, overrides={"database": {"path": "elsewhere/domain_a.duckdb"}})

    assert artifacts_dir(settings) == tmp_path / "artifacts" / "domain_a"


def test_require_artifacts_returns_the_database_folder_whether_or_not_it_exists(tmp_path):
    settings = load_settings(_write_root(tmp_path))
    assert require_artifacts(settings) == tmp_path / "artifacts" / "example"

    (tmp_path / "artifacts" / "example").mkdir(parents=True)
    (tmp_path / "artifacts" / "auto").mkdir()
    assert require_artifacts(settings) == tmp_path / "artifacts" / "example"


def test_require_artifacts_names_the_mv_when_only_the_old_auto_folder_exists(tmp_path):
    settings = load_settings(_write_root(tmp_path))
    (tmp_path / "artifacts" / "auto").mkdir(parents=True)

    with pytest.raises(LegacyArtifacts, match="mv artifacts/auto artifacts/example"):
        require_artifacts(settings)


def test_env_file_loaded_without_overriding_existing_env(tmp_path, monkeypatch):
    root = _write_root(tmp_path)
    (root / ".env").write_text("CTX_TEST_NEW=from_file\nCTX_TEST_KEPT=from_file\n")
    monkeypatch.setenv("CTX_TEST_KEPT", "from_env")
    monkeypatch.delenv("CTX_TEST_NEW", raising=False)

    load_settings(root)

    assert os.environ["CTX_TEST_NEW"] == "from_file"
    assert os.environ["CTX_TEST_KEPT"] == "from_env"
    monkeypatch.delenv("CTX_TEST_NEW")


def test_root_comes_from_ctx_root_env(tmp_path, monkeypatch):
    root = _write_root(tmp_path / "proj")
    monkeypatch.setenv("CTX_ROOT", str(root))

    assert load_settings().root == root


def test_root_found_from_nearest_pyproject_parent(tmp_path, monkeypatch):
    root = _write_root(tmp_path / "proj")
    (root / "pyproject.toml").write_text("")
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.delenv("CTX_ROOT", raising=False)
    monkeypatch.chdir(nested)

    assert load_settings().root == root


