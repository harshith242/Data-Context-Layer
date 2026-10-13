"""build_structured on domain_a with scripted replies: every artifact, the planted move and the manifest."""
import json

import pytest
from fakes import FakeEmbedder
from fixtures.domain_a.make_domain import PLANTED_MOVE_DATE, PLANTED_SERIES_CODE
from fixtures.domain_a.roles_reply import _reply_for_table
from fixtures.domain_a.structured_build import structured_llm
from fixtures.domain_a.unstructured_build import unstructured_llm

from ctxlayer.common.catalog import load_catalog
from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.files import read_yaml, write_yaml
from ctxlayer.structured.build import build_structured
from ctxlayer.unstructured.build import build_unstructured

_ARTIFACTS = [
    "profile.json", "roles.yaml", "ontology.yaml", "entities.yaml", "semantic.yaml",
    "structured/views.sql", "structured/moves.jsonl", "REVIEW.md", "build_manifest.json",
]


def _llm(roles=_reply_for_table):
    return structured_llm(roles=roles)


@pytest.fixture
def built(tmp_settings, tmp_path, domain_a):
    settings = tmp_settings({"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(tmp_path / "out")}})
    counts = build_structured(settings, _llm(), FakeEmbedder())
    return counts, artifacts_dir(settings)


def test_build_writes_every_artifact_and_counts_the_catalog(built, domain_a_catalog):
    counts, out_dir = built
    _, expected_entities = domain_a_catalog

    assert [name for name in _ARTIFACTS if not (out_dir / name).is_file()] == []
    assert counts["entities"] == len(expected_entities) == len(load_catalog(out_dir / "entities.yaml"))
    assert counts["tables"] == 2
    assert "## Roles" in (out_dir / "REVIEW.md").read_text()


def test_build_records_the_planted_move_and_manifest_stage(built):
    counts, out_dir = built
    moves = [json.loads(line) for line in (out_dir / "structured/moves.jsonl").read_text().splitlines()]
    manifest = json.loads((out_dir / "build_manifest.json").read_text())

    assert (PLANTED_SERIES_CODE, PLANTED_MOVE_DATE.isoformat()) in {(m["entity_id"], m["date"]) for m in moves}
    assert counts["moves"] == len(moves)
    assert "structured" in manifest["stages"]


def test_build_completes_when_roles_reply_is_garbage(tmp_settings, tmp_path, domain_a):
    settings = tmp_settings({"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(tmp_path / "out")}})
    counts = build_structured(settings, _llm(roles="not json"), FakeEmbedder())
    review = (artifacts_dir(settings) / "REVIEW.md").read_text()

    # Heuristic fallback: the build still finishes and every column waits for a human.
    assert counts["tables"] == 2
    assert counts["needs_review"] > 0
    assert "## Roles" in review


def test_structured_build_leaves_ontology_without_event_types_and_keeps_event_types_file(tmp_settings, tmp_path, domain_a):
    settings = tmp_settings({"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(tmp_path / "out")}})
    build_structured(settings, _llm(), FakeEmbedder())
    out_dir = artifacts_dir(settings)
    saved = {"ontology_version": "kept", "event_types": [{"name": "generic_change", "description": "A generic change."}]}
    write_yaml(out_dir / "event_types.yaml", saved)

    build_structured(settings, _llm(), FakeEmbedder())

    assert "event_types" not in read_yaml(out_dir / "ontology.yaml")
    assert read_yaml(out_dir / "event_types.yaml") == saved


def test_same_version_structured_and_unstructured_rebuild_reuses_event_types_and_extractions(tmp_settings, tmp_path, domain_a):
    settings = tmp_settings({"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(tmp_path / "out")}})
    build_structured(settings, _llm(), FakeEmbedder())
    build_unstructured(settings, unstructured_llm(), FakeEmbedder())

    build_structured(settings, _llm(), FakeEmbedder())
    second_unstructured = unstructured_llm()
    build_unstructured(settings, second_unstructured, FakeEmbedder())

    tasks = [prompt.split("\n", 1)[0] for prompt, _ in second_unstructured.calls]
    assert tasks.count("[task: event_types]") == 0
    assert tasks.count("[task: extract]") == 0
