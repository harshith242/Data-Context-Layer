"""Shared fixtures."""
import shutil
from pathlib import Path

import pytest

from ctxlayer.common.config import Settings, load_settings
from fixtures.domain_a.make_domain import DomainA, make_domain

_REAL_SETTINGS = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


@pytest.fixture
def tmp_settings(tmp_path):
    """Factory: Settings rooted at a tmp dir, real project settings + overrides."""

    def make(overrides: dict | None = None) -> Settings:
        config_dir = tmp_path / "config"
        config_dir.mkdir(exist_ok=True)
        shutil.copy(_REAL_SETTINGS, config_dir / "settings.yaml")
        return load_settings(root=tmp_path, overrides=overrides)

    return make


@pytest.fixture(scope="session")
def domain_a(tmp_path_factory) -> DomainA:
    """Synthetic grain + bulletin database and questions CSV, built once per session."""
    return make_domain(tmp_path_factory.mktemp("domain_a"))


@pytest.fixture(scope="session")
def project_settings() -> Settings:
    """Real project settings (read-only use: thresholds and defaults)."""
    return load_settings(root=_REAL_SETTINGS.parent.parent)


@pytest.fixture(scope="session")
def domain_a_roles(domain_a, project_settings) -> tuple[dict, dict]:
    """(profile, roles) for domain_a from the scripted role replies."""
    from fixtures.domain_a.roles_reply import build_roles

    return build_roles(domain_a.db_path, project_settings)


@pytest.fixture(scope="session")
def domain_a_semantic(tmp_path_factory, domain_a, domain_a_roles) -> tuple[Settings, dict]:
    """(settings, semantic) for domain_a: settings point at the fixture DB and a tmp artifacts dir
    holding semantic.yaml + structured/views.sql, so `open_semantic(settings)` works."""
    from fixtures.domain_a.semantic_build import build_semantic

    root = tmp_path_factory.mktemp("domain_a_semantic")
    (root / "config").mkdir()
    shutil.copy(_REAL_SETTINGS, root / "config" / "settings.yaml")
    overrides = {"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(root / "artifacts")}}
    settings = load_settings(root=root, overrides=overrides)
    profile, roles = domain_a_roles
    return settings, build_semantic(domain_a.db_path, profile, roles, settings)


@pytest.fixture(scope="session")
def domain_a_catalog(domain_a, domain_a_roles, project_settings) -> tuple[dict, list]:
    """(ontology, entities) for domain_a from scripted replies."""
    from fixtures.domain_a.catalog_build import build_catalog

    profile, roles = domain_a_roles
    return build_catalog(domain_a.db_path, profile, roles, project_settings)


@pytest.fixture(scope="session")
def domain_a_built(tmp_path_factory, domain_a) -> Settings:
    """Settings whose artifacts dir holds a full structured build of domain_a (scripted LLM).

    Read-only for tests: copy it before writing new artifacts."""
    from fakes import FakeEmbedder
    from fixtures.domain_a.structured_build import structured_llm

    from ctxlayer.structured.build import build_structured

    root = tmp_path_factory.mktemp("domain_a_built")
    (root / "config").mkdir()
    shutil.copy(_REAL_SETTINGS, root / "config" / "settings.yaml")
    overrides = {"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(root / "artifacts")}}
    settings = load_settings(root=root, overrides=overrides)
    build_structured(settings, structured_llm(), FakeEmbedder())
    return settings
