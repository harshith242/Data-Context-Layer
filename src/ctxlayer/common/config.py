"""Settings loader: YAML + overrides, project-root discovery, .env loading."""
import copy
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

_ROOT_ENV_VAR = "CTX_ROOT"
_ROOT_MARKER = "pyproject.toml"
_SETTINGS_FILE = Path("config") / "settings.yaml"
_MISSING = object()
_LEGACY_FOLDER = "auto"


class LegacyArtifacts(FileNotFoundError):
    """Only the pre-database-folder artifacts/auto exists; the user must rename it."""


@dataclass(frozen=True)
class Settings:
    root: Path
    raw: dict

    def get(self, dotted: str, default=None):
        """Look up "eval.rel_tolerance"; `default` when any part is missing."""
        node = self.raw
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, dotted: str) -> Path:
        """Resolve a path setting against the project root."""
        value = self.get(dotted, _MISSING)
        if value is _MISSING or value is None:
            raise KeyError(dotted)
        return self.root / value


def load_settings(root: Path | None = None, overrides: dict | None = None) -> Settings:
    root = Path(root) if root else _find_root()
    raw = yaml.safe_load((root / _SETTINGS_FILE).read_text())
    if overrides:
        raw = _merge(raw, overrides)

    # First existing env file wins; real environment variables are kept.
    for name in raw.get("env_files", []):
        env_file = root / name
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            break

    return Settings(root=root, raw=raw)


def artifacts_dir(settings: Settings) -> Path:
    """<paths.artifacts>/<database file stem>, e.g. artifacts/example."""
    return settings.path("paths.artifacts") / settings.path("database.path").stem


def require_artifacts(settings: Settings) -> Path:
    """artifacts_dir, or LegacyArtifacts when only the old artifacts/auto exists (never build into a new folder)."""
    folder = artifacts_dir(settings)
    legacy = folder.parent / _LEGACY_FOLDER
    if folder.exists() or not legacy.is_dir():
        return folder
    root = settings.get("paths.artifacts")
    raise LegacyArtifacts(f"found {root}/{_LEGACY_FOLDER}; run: mv {root}/{_LEGACY_FOLDER} {root}/{folder.name}")


def _find_root() -> Path:
    # Env var wins; else walk up from cwd to the first dir with pyproject.toml.
    from_env = os.environ.get(_ROOT_ENV_VAR)
    if from_env:
        return Path(from_env)

    cwd = Path.cwd().resolve()
    for folder in (cwd, *cwd.parents):
        if (folder / _ROOT_MARKER).is_file():
            return folder
    raise FileNotFoundError(f"no {_ROOT_MARKER} above {cwd}; set {_ROOT_ENV_VAR}")


def _merge(base: dict, overrides: dict) -> dict:
    # Nested dicts merge key by key; any other value is replaced.
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
            continue
        merged[key] = value
    return merged
