"""A whole phased build through the `ctx` commands on domain_a: pauses, decisions, review apply and a no-op rebuild."""
import hashlib
import logging

import yaml
from fakes import FakeEmbedder
from fixtures.domain_a.make_domain import PLANTED_BULLETIN_REF, PLANTED_QUOTE
from fixtures.domain_a.unstructured_build import unstructured_llm

from ctxlayer import cli
from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.files import read_json, read_jsonl, read_yaml
from ctxlayer.common.phase_context import make_context

_MOVE_ID = "move:WHT_MILL_GP:weekly:USD:per_tonne:2025-05-05"
_BY = {"by": "tester", "at": "2026-10-12"}
# Files a links-only change may rewrite besides links/: the review file, build record and pages.
_ALLOWED = {"REVIEW.md", "build_state.json", "context_explorer.html", "build_manifest.json"}


def _digests(out_dir) -> dict[str, str]:
    return {
        str(path.relative_to(out_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out_dir.rglob("*")) if path.is_file()
    }


def _write_decisions(settings, entries: list[dict]) -> None:
    review = settings.path("paths.review")
    review.mkdir(parents=True, exist_ok=True)
    (review / "decisions.yaml").write_text(yaml.safe_dump({"decisions": entries}))


def test_phased_build_with_human_review_and_a_links_only_fix(tmp_settings, tmp_path, domain_a, monkeypatch):
    settings = tmp_settings({"database": {"path": str(domain_a.db_path)}, "paths": {"artifacts": str(tmp_path / "out")}})
    out = artifacts_dir(settings)
    models = []

    def models_for(_settings):
        llm = unstructured_llm()
        models.append(llm)
        return make_context(settings, llm, FakeEmbedder())

    monkeypatch.setattr(cli, "load_settings", lambda overrides=None: settings)
    monkeypatch.setattr(cli, "_models", models_for)
    monkeypatch.setattr(cli, "run_logger", lambda _settings: logging.getLogger("test.cli"))

    # 1. Pauses after roles.
    assert cli.main(["build", "--human-review"]) == 0
    assert read_json(out / "build_state.json")["paused_after"] == "roles"

    # 2. A decision fixes a column description; 3. continuing reruns roles and pauses after event-types.
    column = {
        "kind": "column", "target": {"table": "grain_quotes", "column": "crop"}, "verdict": "fix",
        "fix": {"description": "Crop traded."}, **_BY,
    }
    _write_decisions(settings, [column])
    assert cli.main(["build", "--continue"]) == 0
    assert read_json(out / "build_state.json")["paused_after"] == "event-types"
    assert read_yaml(out / "roles.yaml")["tables"]["grain_quotes"]["columns"]["crop"]["description"] == "Crop traded."

    # 4. Continuing finishes the build.
    assert cli.main(["build", "--continue"]) == 0
    assert "paused_after" not in read_json(out / "build_state.json")

    # 5. The planted drought is the top driver of the planted move.
    drivers = [row for row in read_jsonl(out / "links/candidate_drivers.jsonl") if row["move_id"] == _MOVE_ID]
    assert (drivers[0]["evidence_doc"], drivers[0]["evidence_quote"]) == (PLANTED_BULLETIN_REF, PLANTED_QUOTE)

    # 6. A driver decision plus `review apply` changes links only.
    before = _digests(out)
    driver = {"kind": "driver", "target": {"move_id": _MOVE_ID, "key": f"{PLANTED_BULLETIN_REF}#0"}, "verdict": "wrong", **_BY}
    _write_decisions(settings, [column, driver])
    assert cli.main(["review", "apply"]) == 0
    after = _digests(out)
    changed = {name for name in before if before[name] != after.get(name)}

    assert read_jsonl(out / "links/candidate_drivers.jsonl") == []
    assert "links/candidate_drivers.jsonl" in changed
    assert all(name.startswith("links/") or name in _ALLOWED for name in changed)
    assert set(after) == set(before)

    # 7. A second plain build runs nothing and calls no model.
    assert cli.main(["build"]) == 0
    assert models[-1].calls == []
    assert read_json(out / "build_state.json")["phases"]["links"]["applied"]
