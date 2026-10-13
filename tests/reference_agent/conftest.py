"""Fixtures shared by the reference agent tests."""
import pytest

from ctxlayer.common.catalog import save_catalog
from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.files import write_yaml


@pytest.fixture
def agent_settings(domain_a_semantic, domain_a_catalog):
    settings, _ = domain_a_semantic
    ontology, entities = domain_a_catalog
    save_catalog(entities, artifacts_dir(settings) / "entities.yaml")
    write_yaml(artifacts_dir(settings) / "ontology.yaml", ontology)
    return settings
