"""Context explorer page over a full domain_a build: counts match the artifacts, the planted story shows."""
import json
import re
import shutil
import subprocess

import pytest
import yaml
from fakes import FakeEmbedder
from fixtures.domain_a.make_domain import PLANTED_BULLETIN_REF, PLANTED_QUOTE, PLANTED_SERIES_CODE
from fixtures.domain_a.unstructured_build import copy_built, unstructured_llm

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.decisions import Decisions, Kind, merge_decisions
from ctxlayer.common.files import read_jsonl, read_yaml
from ctxlayer.linking.build import build_links
from ctxlayer.serving import explorer
from ctxlayer.serving.explorer import write_explorer
from ctxlayer.unstructured.build import build_unstructured


@pytest.fixture(scope="module")
def built(tmp_path_factory, domain_a, domain_a_built):
    settings = copy_built(tmp_path_factory.mktemp("explorer"), domain_a, domain_a_built)
    build_unstructured(settings, unstructured_llm(), FakeEmbedder())
    build_links(settings)
    return settings


_DATA = re.compile(r'<script type="application/json" id="data">(.*?)</script>', re.S)
_DOUBTFUL_TEXT, _UNMATCHED_TEXT = "wheat from the GP", "Zed"
_PAUSED_STATE = {"phases": {}, "paused_after": "roles", "human_review": True}


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory, domain_a, domain_a_built):
    """A build with a doubtful name match, an embedded decision on a column and a build paused after roles."""
    settings = copy_built(tmp_path_factory.mktemp("explorer_review"), domain_a, domain_a_built)
    build_unstructured(settings, unstructured_llm(), FakeEmbedder())
    build_links(settings)
    out = artifacts_dir(settings)

    doubtful = [
        {"entity": _DOUBTFUL_TEXT, "entity_id": "crop:milling_wheat", "match_score": 0.8},
        {"entity": _UNMATCHED_TEXT, "entity_id": None, "match_score": 0},
    ]
    resolved = out / "docs/resolved.jsonl"
    resolved.write_text(resolved.read_text() + json.dumps(
        {"doc_id": "FB-X", "mentions": [], "events": [{"affects": doubtful}]}) + "\n")
    (out / "build_state.json").write_text(json.dumps(_PAUSED_STATE))

    column = read_yaml(out / "roles.yaml")["tables"]
    table, columns = next((name, spec["columns"]) for name, spec in column.items() if spec["kind"] == "measures")
    review = settings.path("paths.review")
    review.mkdir(parents=True, exist_ok=True)
    (review / "decisions.yaml").write_text(yaml.safe_dump({"decisions": [
        {"kind": "column", "target": {"table": table, "column": next(iter(columns))}, "verdict": "correct",
         "by": "Alex", "at": "2026-10-12", "note": "checked by hand"},
    ]}))
    return settings


def _data(html: str) -> dict:
    return json.loads(_DATA.search(html).group(1))


def _reviewable(data: dict) -> list[dict]:
    """Every row of the embedded data that a reviewer can mark."""
    rows = [
        *(column for table in data["tables"] for column in table["columns"]), *data["rules"],
        *data["ontology"]["types"], *data["ontology"]["event_types"], *data["entities"], *data["name_matches"],
        *data["moves"], *(driver for move in data["moves"] for driver in move["drivers"]),
        *data["events"], *data["merged_events"],
    ]
    return [row for row in rows if "kind" in row]


def _script(html: str) -> str:
    return re.search(r"<script>(.*?)</script>", html, re.S).group(1)


@pytest.fixture(scope="module")
def page(built) -> tuple[str, dict]:
    html = write_explorer(built).read_text()
    return html, _data(html)


def test_counts_match_the_artifacts(built, page):
    _, data = page
    out = artifacts_dir(built)
    nodes = read_jsonl(out / "graph/nodes.jsonl")

    assert data["counts"]["entities"] == len(read_yaml(out / "entities.yaml")) + len(read_yaml(out / "graph/entities_text.yaml"))
    assert data["counts"]["events"] == sum(node["type"] == "Event" for node in nodes)
    assert data["counts"]["documents"] == sum(node["type"] == "Document" for node in nodes)
    assert data["counts"]["moves"] == len(read_jsonl(out / "structured/moves.jsonl"))
    assert data["counts"]["candidate_drivers"] == len(read_jsonl(out / "links/candidate_drivers.jsonl"))
    assert data["meta"]["now"] == read_yaml(out / "semantic.yaml")["now"]["value"]


def test_the_planted_move_shows_the_planted_drought_as_its_top_driver(page):
    _, data = page
    move = next(mv for mv in data["moves"] if mv["entity"] == PLANTED_SERIES_CODE and mv["drivers"])
    top = move["drivers"][0]

    assert (top["doc"], top["quote"], top["aspect"], top["direction"]) == (PLANTED_BULLETIN_REF, PLANTED_QUOTE, "price", "up")
    assert top["status"] == "reported" and top["lag_days"] == 10


def test_every_column_role_and_rule_is_listed(built, page):
    _, data = page
    roles = read_yaml(artifacts_dir(built) / "roles.yaml")

    shown = {(t["name"], c["column"]): c["role"] for t in data["tables"] for c in t["columns"]}
    expected = {(table, column): item["role"] for table, spec in roles["tables"].items() for column, item in spec["columns"].items()}
    assert shown == expected
    assert any(rule["rule"].endswith("never aggregate across") for rule in data["rules"])


def test_page_is_self_contained_and_script_safe(page):
    html, _ = page

    # Nothing loaded from outside, and embedded data cannot close the script block early.
    assert not re.search(r'(src|href)="https?://', html)
    assert html.count("</script>") == 2


def test_moves_in_the_news_period_come_first_and_the_rest_are_marked_outside(built, page):
    _, data = page
    out = artifacts_dir(built)
    days = [doc["date"] for doc in read_jsonl(out / "docs/documents.jsonl")]
    inside = [mv["date"] for mv in data["moves"] if mv["in_window"]]
    outside = [mv["date"] for mv in data["moves"] if not mv["in_window"]]

    assert inside and outside
    assert all(min(days) <= day <= max(days) for day in inside)
    assert not any(min(days) <= day <= max(days) for day in outside)
    assert [mv["in_window"] for mv in data["moves"]] == [True] * len(inside) + [False] * len(outside)
    assert all(not mv["drivers"] for mv in data["moves"] if not mv["in_window"])
    assert data["counts"]["moves_with_drivers"] == len({row["move_id"] for row in read_jsonl(out / "links/candidate_drivers.jsonl")})


def test_every_reviewable_row_has_a_kind_and_target_that_decisions_accepts(reviewed, tmp_path):
    data = _data(write_explorer(reviewed).read_text())
    rows = _reviewable(data)
    # A rule only has a fix and a name match no correct verdict (spec 8.2).
    verdicts = {"rule": ("fix", {"value": 1}), "name_match": ("wrong", {})}
    entries = [
        {"kind": row["kind"], "target": row["target"], "by": "T", "at": "2026-10-12",
         **dict(zip(("verdict", "fix"), verdicts.get(row["kind"], ("correct", {}))))}
        for row in rows
    ]
    path = tmp_path / "decisions.yaml"
    path.write_text(yaml.safe_dump({"decisions": entries}))

    loaded = Decisions.load(path)

    assert loaded.invalid == []
    expected = {"column", "entity_type", "event_type", "entity", "new_entity", "name_match", "event", "same_event",
                "move", "driver", "rule"}
    assert {row["kind"] for row in rows} == expected
    assert all(loaded.of(Kind(kind)) for kind in expected)


def test_name_matches_below_one_are_listed_least_sure_first(reviewed):
    data = _data(write_explorer(reviewed).read_text())

    assert [(row["text"], row["entity_id"], row["score"]) for row in data["name_matches"]] == [
        (_UNMATCHED_TEXT, None, 0), (_DOUBTFUL_TEXT, "crop:milling_wheat", 0.8)]
    assert data["name_matches"][1]["target"] == {"text": _DOUBTFUL_TEXT}


def test_an_embedded_decision_appears_on_its_row(reviewed):
    data = _data(write_explorer(reviewed).read_text())
    marked = [row for row in _reviewable(data) if row["decision"]]

    assert len(marked) == 1 and marked[0]["kind"] == "column"
    assert (marked[0]["decision"]["by"], marked[0]["decision"]["verdict"], marked[0]["decision"]["note"]) == (
        "Alex", "correct", "checked by hand")
    assert data["decisions"] == [marked[0]["decision"]]


def test_a_paused_build_shows_where_it_stopped_and_how_to_continue(reviewed):
    html = write_explorer(reviewed).read_text()

    assert _data(html)["state"] == {"paused_after": "roles", "stale": []}
    assert "ctx build --continue" in html


def test_stale_phases_passed_by_the_runner_are_listed(reviewed):
    phases = [
        {"phase": "roles", "state": "up to date", "reason": ""},
        {"phase": "catalog", "state": "stale", "reason": "context.md changed"},
        {"phase": "links", "state": "never built", "reason": ""},
    ]

    html = write_explorer(reviewed, {"paused_after": None, "phases": phases}).read_text()

    assert _data(html)["state"] == {"paused_after": None, "stale": [{"phase": "catalog", "reason": "context.md changed"}]}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_page_script_has_valid_syntax(page, tmp_path):
    html, _ = page
    (tmp_path / "page.js").write_text(_script(html))

    result = subprocess.run(["node", "--check", str(tmp_path / "page.js")], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_export_merges_embedded_and_new_decisions_newest_wins_as_valid_yaml(page, tmp_path):
    html, _ = page
    pure = re.search(r"// PURE-BEGIN[^\n]*\n(.*?)// PURE-END", html, re.S).group(1)

    def decision(kind, target, verdict, at, fix=None, note=""):
        return {"kind": kind, "target": target, "verdict": verdict, "fix": fix or {}, "by": "Alex", "at": at, "note": note}

    embedded = [
        decision("column", {"table": "t", "column": "c"}, "correct", "2026-10-10"),
        decision("same_event", {"keys": ["b#0", "a#0"]}, "correct", "2026-10-12"),
        decision("rule", {"table": "t", "field": "time.grains"}, "fix", "2026-10-09", {"value": ["weekly"]}),
    ]
    session = [
        decision("column", {"table": "t", "column": "c"}, "fix", "2026-10-11", {"role": "dimension"},
                 'say "hi": café # not a comment'),
        decision("same_event", {"keys": ["a#0", "b#0"]}, "wrong", "2026-10-11"),
        decision("name_match", {"text": "GP"}, "wrong", "2026-10-12", {"not_entity_id": "location:great_plains"}),
    ]
    script = pure + f"\nconsole.log(toYaml(mergeDecisions({json.dumps(embedded)}, {json.dumps(session)})));"
    (tmp_path / "export.js").write_text(script)

    out = subprocess.run(["node", str(tmp_path / "export.js")], capture_output=True, text=True, check=True).stdout

    assert yaml.safe_load(out)["decisions"] == merge_decisions(embedded, session)
    # One field per line: each decision is a "- kind" line then its other six fields.
    fields = [line.split(":")[0].strip(" -") for line in out.strip().splitlines()[1:]]
    assert fields == ["kind", "target", "verdict", "fix", "by", "at", "note"] * len(yaml.safe_load(out)["decisions"])


def test_connections_list_sentence_backed_links_first_then_tags_then_data(built, page):
    _, data = page

    def tier(item):
        if item["evidence"]:
            return 0
        return 1 if item["basis"].startswith("from a tag column") or item["basis"].startswith("from tag column") else 2

    lists = [[tier(item) for item in items] for items in data["neighbours"].values()]

    assert all(tiers == sorted(tiers) for tiers in lists)
    assert any(0 in tiers and 1 in tiers and 2 in tiers for tiers in lists)


def test_connection_evidence_shows_sentences_support_and_plain_language_reasons(built, page):
    _, data = page
    edges = read_jsonl(artifacts_dir(built) / "graph/edges.jsonl")
    wheat = data["neighbours"]["crop:milling_wheat"]
    by_rel = {rel: [item for item in wheat if item["rel"] == rel] for rel in {item["rel"] for item in wheat}}

    affects = {edge["support"] for edge in edges if edge["rel"] == "AFFECTS" and edge["dst"] == "crop:milling_wheat"}
    assert affects and affects == {item["support"] for item in by_rel["AFFECTS"]}
    assert max(affects) > 1
    assert all(1 <= len(item["evidence"]) <= 3 and all(e["quote"] and e["doc"] for e in item["evidence"]) for item in by_rel["AFFECTS"])
    assert {item["basis"] for item in by_rel["MENTIONS"] if not item["evidence"]} == {"name found in text"}
    assert {item["basis"] for item in by_rel["TAGGED_WITH"]} == {"from tag column crop_tag"}
    parts = {item["basis"] for item in wheat if item["rel"] in ("HAS_PARENT", "PART_OF", "AT_LOCATION")}
    assert parts <= {"from the data"}


def test_the_cap_keeps_the_sentence_backed_links(built, monkeypatch):
    out = artifacts_dir(built)
    touching = [e for e in read_jsonl(out / "graph/edges.jsonl") if "crop:milling_wheat" in (e["src"], e["dst"])]
    backed = [e for e in touching if any(item["quote"] for item in e["evidence"])]
    assert 0 < len(backed) < len(touching)

    monkeypatch.setattr(explorer, "_MAX_NEIGHBOURS", len(backed))
    kept = _data(write_explorer(built).read_text())["neighbours"]["crop:milling_wheat"]

    assert len(kept) == len(backed) and all(item["evidence"] for item in kept)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_drafts_the_page_already_embeds_or_that_are_older_are_dropped(page, tmp_path):
    html, _ = page
    pure = re.search(r"// PURE-BEGIN[^\n]*\n(.*?)// PURE-END", html, re.S).group(1)
    target = {"table": "t", "column": "c"}
    embedded = [{"kind": "column", "target": target, "verdict": "correct", "fix": {}, "at": "2026-10-12"}]
    drafts = [
        {**embedded[0]},
        {**embedded[0], "at": "2026-10-11", "verdict": "wrong"},
        {**embedded[0], "verdict": "fix", "fix": {"role": "dimension"}},
        {"kind": "column", "target": {"table": "t", "column": "d"}, "verdict": "correct", "fix": {}, "at": "2026-10-01"},
    ]
    (tmp_path / "drafts.js").write_text(
        pure + f"\nconsole.log(JSON.stringify(liveDrafts({json.dumps(embedded)}, {json.dumps(drafts)})));")

    out = subprocess.run(["node", str(tmp_path / "drafts.js")], capture_output=True, text=True, check=True).stdout

    assert json.loads(out) == drafts[2:]
