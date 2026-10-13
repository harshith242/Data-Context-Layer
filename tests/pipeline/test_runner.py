"""The pipeline runner on domain_a: staleness, pauses, ranges, status and adopt."""
import hashlib
import json
import logging
import shutil

import duckdb
import pytest
import yaml
from fakes import FakeEmbedder
from fixtures.domain_a.unstructured_build import copy_built, unstructured_llm

from ctxlayer import cli
from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.db import list_tables
from ctxlayer.common.files import read_json, read_jsonl, read_yaml, write_jsonl, write_yaml
from ctxlayer.common.phase_context import make_context
from ctxlayer.common.ontology import _without_samples
from ctxlayer.linking.build import build_links
from ctxlayer.linking.phases import run_links
from ctxlayer.pipeline.phases import PHASES, STRUCTURED_ONLY
from ctxlayer.pipeline.runner import Review, adopt, continue_build, phase_status, run_build
from ctxlayer.unstructured.build import build_unstructured

_NAMES = [phase.name for phase in PHASES]
_UP = "up to date"


def _pc(settings, llm=None):
    return make_context(settings, llm or unstructured_llm(), FakeEmbedder())


def _states(pc) -> dict[str, dict]:
    return {row["phase"]: row for row in phase_status(pc)}


def _full(root, domain_a, source):
    """A copy of `source`'s artifacts and the settings over it."""
    return copy_built(root, domain_a, source)


@pytest.fixture(scope="module")
def built(tmp_path_factory, domain_a, domain_a_built):
    """Settings whose artifacts come from one full run_build; read-only, copy before changing."""
    settings = copy_built(tmp_path_factory.mktemp("runner"), domain_a, domain_a_built)
    shutil.rmtree(artifacts_dir(settings))
    run_build(_pc(settings), None, None, Review.OFF, force_start=False)
    return settings


def test_first_build_runs_every_phase_and_a_second_runs_none_without_a_model(built):
    state = read_json(artifacts_dir(built) / "build_state.json")
    llm = unstructured_llm()

    second = run_build(_pc(built, llm), None, None, Review.OFF, force_start=False)

    assert sorted(state["phases"]) == sorted(_NAMES)
    assert second["ran"] == [] and second["skipped"] == _NAMES
    assert llm.calls == []
    assert (artifacts_dir(built) / "context_explorer.html").is_file()
    assert {row["state"] for row in phase_status(_pc(built))} == {_UP}


def test_human_review_pauses_after_roles_and_event_types_then_finishes(tmp_settings, tmp_path, domain_a):
    settings = tmp_settings({"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(tmp_path / "out")}})
    out = artifacts_dir(settings)
    pc = _pc(settings)

    first = run_build(pc, None, None, Review.STOP, force_start=False)
    state = read_json(out / "build_state.json")

    assert first["ran"] == ["profile", "roles"] and first["paused_after"] == "roles"
    assert (state["paused_after"], state["human_review"]) == ("roles", True)
    assert (out / "context_explorer.html").is_file()

    second = continue_build(pc)
    assert second["ran"] == ["ontology", "documents", "index", "event-types"]
    assert second["paused_after"] == "event-types"

    third = continue_build(pc)
    assert third["ran"] == _NAMES[6:] and third["paused_after"] is None
    assert "paused_after" not in read_json(out / "build_state.json")


def test_continue_with_nothing_paused_raises_and_the_cli_exits_1(built, monkeypatch, capsys):
    with pytest.raises(ValueError, match="no paused build; run ctx build"):
        continue_build(_pc(built))

    monkeypatch.setattr(cli, "load_settings", lambda overrides=None: built)
    monkeypatch.setattr(cli, "_models", lambda settings: _pc(built))
    monkeypatch.setattr(cli, "run_logger", lambda settings: logging.getLogger("test.cli"))

    assert cli.main(["build", "--continue"]) == 1
    assert "no paused build; run ctx build" in capsys.readouterr().err


def test_a_review_build_does_not_pause_at_a_phase_that_did_not_run(built):
    result = run_build(_pc(built), None, None, Review.STOP, force_start=False)

    assert result["paused_after"] is None


def test_continue_after_inputs_changed_during_the_pause_reruns_the_stale_phases(tmp_settings, tmp_path, domain_a):
    settings = tmp_settings({"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(tmp_path / "out")}})
    run_build(_pc(settings), None, None, Review.OFF, force_start=False)
    pc = _pc(settings)
    run_build(pc, "roles", "roles", Review.STOP, force_start=True)
    roles = read_yaml(artifacts_dir(settings) / "roles.yaml")
    roles["tables"]["grain_quotes"]["columns"]["crop"]["description"] = "edited during the pause"
    write_yaml(artifacts_dir(settings) / "roles.yaml", roles)

    result = continue_build(pc)

    assert result["ran"] == ["ontology", "documents", "event-types"]
    assert continue_build(pc)["ran"] == _NAMES[6:]


def test_continue_reruns_a_roles_decision_made_during_the_pause_then_pauses_at_event_types(tmp_settings, tmp_path, domain_a):
    settings = tmp_settings({"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(tmp_path / "out")}})
    first = run_build(_pc(settings), None, None, Review.STOP, force_start=False)
    entry = {
        "kind": "column", "target": {"table": "grain_quotes", "column": "crop"}, "verdict": "fix",
        "fix": {"description": "fixed during the pause"}, "by": "A", "at": "2026-10-12",
    }
    settings.path("paths.review").mkdir(parents=True)
    (settings.path("paths.review") / "decisions.yaml").write_text(yaml.safe_dump({"decisions": [entry]}))

    result = continue_build(_pc(settings))

    roles = read_yaml(artifacts_dir(settings) / "roles.yaml")
    assert first["paused_after"] == "roles"
    assert result["ran"] == ["roles", "ontology", "documents", "index", "event-types"]
    assert result["paused_after"] == "event-types"
    assert roles["tables"]["grain_quotes"]["columns"]["crop"]["description"] == "fixed during the pause"


def test_changing_the_names_section_makes_the_phases_that_read_it_stale_and_not_roles(built, tmp_path_factory, domain_a):
    settings = _full(tmp_path_factory.mktemp("names"), domain_a, built)
    folder = settings.path("paths.sources") / settings.path("database.path").stem
    folder.mkdir(parents=True)
    (folder / "context.md").write_text("## Names\nGP means Great Plains.\n")

    states = _states(_pc(settings))

    for name in ("catalog", "extraction", "resolve"):
        assert (states[name]["state"], states[name]["reason"]) == ("stale", "context.md changed")
    assert states["roles"]["state"] == _UP and states["profile"]["state"] == _UP
    assert states["dedupe"]["reason"] == "upstream stale"
    assert states["views"]["state"] == _UP


def test_editing_another_heading_leaves_the_ontology_and_event_types_up_to_date(built, tmp_path_factory, domain_a):
    settings = _full(tmp_path_factory.mktemp("news"), domain_a, built)
    folder = settings.path("paths.sources") / settings.path("database.path").stem
    folder.mkdir(parents=True)
    (folder / "context.md").write_text("## Currencies\nPrices are in USD.\n")

    states = _states(_pc(settings))

    assert states["ontology"]["state"] == _UP and states["roles"]["state"] == _UP
    assert states["semantic"]["reason"] == "context.md changed"


def test_a_decision_makes_only_the_phases_of_its_kind_stale(built, tmp_path_factory, domain_a):
    settings = _full(tmp_path_factory.mktemp("decision"), domain_a, built)
    review = settings.path("paths.review")
    review.mkdir(parents=True)
    entry = {"kind": "driver", "target": {"move_id": "m1", "key": "k1"}, "verdict": "wrong", "by": "A", "at": "2026-10-12"}
    (review / "decisions.yaml").write_text(yaml.safe_dump({"decisions": [entry]}))

    states = _states(_pc(settings))

    assert (states["links"]["state"], states["links"]["reason"]) == ("stale", "decisions changed")
    assert [name for name, row in states.items() if row["state"] != _UP] == ["links"]


def test_inserting_a_row_in_the_database_makes_profile_stale(tmp_settings, tmp_path, domain_a):
    database = tmp_path / domain_a.db_path.name
    shutil.copy(domain_a.db_path, database)
    settings = tmp_settings({"database": {"path": str(database)}, "paths": {"artifacts": str(tmp_path / "out")}})
    run_build(_pc(settings), None, "profile", Review.OFF, force_start=False)
    assert _states(_pc(settings))["profile"]["state"] == _UP

    con = duckdb.connect(str(database))
    table = list_tables(con)[0]
    con.execute(f'INSERT INTO "{table}" SELECT * FROM "{table}" LIMIT 1')
    con.close()

    row = _states(_pc(settings))["profile"]
    assert (row["state"], row["reason"]) == ("stale", "database changed")


def test_from_graph_reruns_graph_only_when_its_output_is_unchanged(built, tmp_path_factory, domain_a):
    settings = _full(tmp_path_factory.mktemp("from_graph"), domain_a, built)
    llm = unstructured_llm()

    result = run_build(_pc(settings, llm), "graph", None, Review.OFF, force_start=True)

    assert result["ran"] == ["graph"]  # links reads files graph rewrote unchanged
    assert llm.calls == []


def test_the_unstructured_shortcut_leaves_the_structured_only_phases_out(built, tmp_path_factory, domain_a):
    settings = _full(tmp_path_factory.mktemp("shortcut"), domain_a, built)

    result = run_build(
        _pc(settings), "documents", "graph", Review.OFF, force_start=True, skip=STRUCTURED_ONLY,
    )

    assert result["ran"] == ["documents"]  # its output is unchanged, so the phases after it are up to date
    assert STRUCTURED_ONLY <= set(result["skipped"])


@pytest.fixture
def old_build(tmp_path_factory, domain_a, domain_a_built):
    """A finished build made by the wrappers: no build_state.json."""
    settings = copy_built(tmp_path_factory.mktemp("old"), domain_a, domain_a_built)
    build_unstructured(settings, unstructured_llm(), FakeEmbedder())
    build_links(settings)
    return settings


def test_adopt_marks_a_build_without_state_as_up_to_date_and_the_next_build_runs_nothing(old_build):
    assert not (artifacts_dir(old_build) / "build_state.json").exists()
    llm = unstructured_llm()

    result = adopt(_pc(old_build, llm))
    build = run_build(_pc(old_build, llm), None, None, Review.OFF, force_start=False)

    assert result == {"adopted": _NAMES, "missing": {}}
    assert build["ran"] == [] and llm.calls == []


def test_adopt_moves_event_types_out_of_the_ontology_and_lists_missing_outputs(old_build):
    out = artifacts_dir(old_build)
    induced = read_yaml(out / "event_types.yaml")["event_types"]
    write_yaml(out / "ontology.yaml", {**read_yaml(out / "ontology.yaml"), "event_types": induced})
    (out / "event_types.yaml").unlink()
    (out / "graph" / "nodes.jsonl").unlink()
    llm = unstructured_llm()

    result = adopt(_pc(old_build, llm))

    assert read_yaml(out / "event_types.yaml")["event_types"] == induced and llm.calls == []
    assert result["missing"] == {"graph": ["graph/nodes.jsonl"]}
    assert "graph" not in result["adopted"] and "roles" in result["adopted"]


def test_adopt_restamps_a_pre_plan_ontology_version_so_event_types_are_not_induced_again(old_build):
    out = artifacts_dir(old_build)
    inputs = {
        "roles": read_yaml(out / "roles.yaml"), "profile": _without_samples(read_json(out / "profile.json")),
        "context": [None, {}],
    }
    legacy = hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()
    write_yaml(out / "ontology.yaml", {**read_yaml(out / "ontology.yaml"), "version": legacy})
    write_yaml(out / "event_types.yaml", {**read_yaml(out / "event_types.yaml"), "ontology_version": legacy})
    induced = read_yaml(out / "event_types.yaml")["event_types"]
    llm = unstructured_llm()

    adopt(_pc(old_build, llm))
    run_build(_pc(old_build, llm), "event-types", "event-types", Review.OFF, force_start=True)

    version = read_yaml(out / "ontology.yaml")["version"]
    assert version != legacy and read_yaml(out / "event_types.yaml")["ontology_version"] == version
    assert read_yaml(out / "event_types.yaml")["event_types"] == induced and llm.calls == []


def test_build_links_accepts_events_of_a_graph_built_before_event_keys(built, tmp_path_factory, domain_a):
    settings = _full(tmp_path_factory.mktemp("no_keys"), domain_a, built)
    path = artifacts_dir(settings) / "graph" / "nodes.jsonl"
    nodes = read_jsonl(path)
    for node in nodes:
        node["props"].pop("keys", None)
    write_jsonl(path, nodes)

    assert run_links(_pc(settings))["moves"] > 0


def test_build_refuses_a_folder_built_before_build_state_until_it_is_adopted(old_build):
    with pytest.raises(ValueError, match="run ctx adopt first"):
        run_build(_pc(old_build), None, None, Review.OFF, force_start=False)


def test_adopt_leaves_decisions_already_in_the_file_pending(old_build):
    entry = {"kind": "driver", "target": {"move_id": "m", "key": "k#0"}, "verdict": "wrong", "by": "A", "at": "2026-10-12"}
    old_build.path("paths.review").mkdir(parents=True)
    (old_build.path("paths.review") / "decisions.yaml").write_text(yaml.safe_dump({"decisions": [entry]}))

    adopt(_pc(old_build))

    row = _states(_pc(old_build))["links"]
    assert (row["state"], row["reason"]) == ("stale", "decisions changed")
