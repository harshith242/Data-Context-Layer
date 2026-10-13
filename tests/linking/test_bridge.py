"""Candidate drivers of domain_a moves, the links artifacts and explain_move."""
import contextlib
import json
import math
from datetime import date

import pytest
from fakes import FakeEmbedder
from fixtures.domain_a.make_domain import PLANTED_BULLETIN_REF, PLANTED_MOVE_DATE, PLANTED_QUOTE
from fixtures.domain_a.unstructured_build import copy_built, unstructured_llm

from ctxlayer.common.catalog import load_catalog
from ctxlayer.common.config import artifacts_dir, load_settings
from ctxlayer.common.files import read_yaml
from ctxlayer.linking.bridge import candidate_drivers, explain_move
from ctxlayer.linking.build import build_links
from ctxlayer.structured.semantic_views import open_semantic
from ctxlayer.unstructured.build import build_unstructured
from ctxlayer.unstructured.graph_store import GraphStore

_MOVE_ID = "move:WHT_MILL_GP:weekly:USD:per_tonne:2025-05-05"
_EVENT_DATE = date(2025, 4, 25)
_WINDOW = ("2025-03-01", "2025-06-30")
_SECOND_REF = "FB-20250427-01"
_AFFECT_CONFIDENCE = 0.8


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def linked(tmp_path_factory, domain_a, domain_a_built):
    """(settings, counts) after build_unstructured and build_links on a copy of the structured build."""
    settings = copy_built(tmp_path_factory.mktemp("links"), domain_a, domain_a_built)
    build_unstructured(settings, unstructured_llm(), FakeEmbedder())
    return settings, build_links(settings)


@pytest.fixture(scope="module")
def out_dir(linked):
    return artifacts_dir(linked[0])


@pytest.fixture(scope="module")
def store(out_dir) -> GraphStore:
    files = [out_dir / "graph" / "nodes.jsonl", out_dir / "graph" / "edges.jsonl"]
    return GraphStore.load([*files, out_dir / "links" / "nodes.jsonl", out_dir / "links" / "edges.jsonl"])


@pytest.fixture(scope="module")
def planted_move(out_dir) -> dict:
    move = next(m for m in _read_jsonl(out_dir / "structured" / "moves.jsonl") if m["id"] == _MOVE_ID)
    return {**move, "catalog_id": "series:wht_mill_gp"}


def _event(**changes) -> dict:
    """A drought-like event 10 days before the planted move, affecting the planted crop upwards."""
    affect = {
        "entity_id": "crop:milling_wheat", "direction": "up", "confidence": _AFFECT_CONFIDENCE,
        "quote": "q", "doc_id": PLANTED_BULLETIN_REF,
    }
    event = {
        "event_id": "event:test", "type": "supply_disruption", "date": "2025-04-25", "summary": "hand built",
        "entity_ids": ["crop:milling_wheat"], "location_ids": ["location:great_plains"], "org_names": [],
        "doc_ids": [PLANTED_BULLETIN_REF], "quotes": ["q"], "confidence": _AFFECT_CONFIDENCE,
        "affects": [affect], "needs_review": False,
    }
    return {**event, **changes}


def _drivers(project_settings, store, planted_move, events, out_dir) -> list[dict]:
    catalog = load_catalog(out_dir / "entities.yaml")
    return candidate_drivers([planted_move], store, events, catalog, project_settings, _WINDOW)


def test_planted_move_top_driver_is_the_planted_event_ten_days_before(out_dir):
    rows = [r for r in _read_jsonl(out_dir / "links" / "candidate_drivers.jsonl") if r["move_id"] == _MOVE_ID]

    top = rows[0]
    assert len(rows) == 1
    assert top["lag_days"] == (PLANTED_MOVE_DATE - _EVENT_DATE).days == 10
    assert (top["relation"], top["direction_match"], top["evidence_quote"]) == ("AFFECTS", "match", PLANTED_QUOTE)
    assert top["evidence_doc"] == PLANTED_BULLETIN_REF
    assert top["score"] == pytest.approx(_AFFECT_CONFIDENCE * math.exp(-10 / 14))


def test_links_files_hold_move_nodes_and_candidate_edges(out_dir, linked):
    nodes = {n["id"]: n for n in _read_jsonl(out_dir / "links" / "nodes.jsonl")}
    edges = [e for e in _read_jsonl(out_dir / "links" / "edges.jsonl") if e["rel"] == "CANDIDATE_DRIVER_OF"]

    assert nodes[_MOVE_ID]["type"] == "Move" and nodes[_MOVE_ID]["props"]["catalog_id"] == "series:wht_mill_gp"
    assert [(e["dst"], e["props"]["lag_days"], e["source"]) for e in edges] == [(_MOVE_ID, 10, "linking")]
    assert edges[0]["src"].startswith("event:")
    assert linked[1]["moves"] == 2 and linked[1]["drivers"] == 1 and linked[1]["moves_with_drivers"] == 1


def test_series_is_linked_to_its_crop_from_the_data(out_dir):
    parents = {(e["src"], e["dst"]) for e in _read_jsonl(out_dir / "links" / "edges.jsonl") if e["rel"] == "HAS_PARENT"}

    assert ("series:wht_mill_gp", "crop:milling_wheat") in parents
    assert ("series:wht_mill_gp", "cropgroup:cereals") not in parents  # already a catalog parent


def test_event_after_the_move_is_excluded(project_settings, store, planted_move, out_dir):
    events = [_event(event_id="event:after", date="2025-05-06"), _event(event_id="event:same_day", date="2025-05-05")]

    rows = _drivers(project_settings, store, planted_move, events, out_dir)

    assert [(r["event_id"], r["lag_days"]) for r in rows] == [("event:same_day", 0)]


def test_event_beyond_lookback_is_excluded(project_settings, store, planted_move, out_dir):
    events = [_event(event_id="event:edge", date="2025-04-05"), _event(event_id="event:old", date="2025-04-04")]

    rows = _drivers(project_settings, store, planted_move, events, out_dir)

    assert [(r["event_id"], r["lag_days"]) for r in rows] == [("event:edge", 30)]


def _with_weights(settings, weights: dict):
    return load_settings(root=settings.root, overrides={"linking": {"direction_weights": weights}})


def test_opposite_direction_ranks_below_match_when_direction_weights_are_set(project_settings, store, planted_move, out_dir):
    down = {**_event()["affects"][0], "direction": "down"}
    events = [
        _event(event_id="event:a_down", affects=[down]),
        _event(event_id="event:b_up"),
        _event(event_id="event:c_unknown", affects=[{**down, "direction": "unknown"}]),
    ]

    weighted = _with_weights(project_settings, {"match": 1.0, "unknown": 0.5, "opposite": 0.1})
    rows = _drivers(weighted, store, planted_move, events, out_dir)

    assert [(r["event_id"], r["direction_match"]) for r in rows] == [
        ("event:b_up", "match"), ("event:c_unknown", "unknown"), ("event:a_down", "opposite"),
    ]
    assert rows[0]["score"] / rows[2]["score"] == pytest.approx(10.0)


def test_parent_level_affects_counts(project_settings, store, planted_move, out_dir):
    group_affect = {**_event()["affects"][0], "entity_id": "cropgroup:cereals"}
    other_affect = {**group_affect, "entity_id": "crop:maize"}
    events = [_event(event_id="event:group", affects=[group_affect]), _event(event_id="event:other", affects=[other_affect], doc_ids=[])]

    rows = _drivers(project_settings, store, planted_move, events, out_dir)

    assert [(r["event_id"], r["relation"]) for r in rows] == [("event:group", "AFFECTS")]


def test_event_in_another_region_is_excluded_and_enclosing_region_kept(project_settings, store, planted_move, out_dir):
    events = [
        _event(event_id="event:rhine", location_ids=["location:rhine_valley"]),
        _event(event_id="event:prairie", location_ids=["location:prairie_basin"]),
        _event(event_id="event:continent", location_ids=["location:north_america"]),
        _event(event_id="event:nowhere", location_ids=[]),
    ]

    rows = _drivers(project_settings, store, planted_move, events, out_dir)

    assert sorted(r["event_id"] for r in rows) == ["event:continent", "event:nowhere"]


def test_tagged_document_links_an_event_without_affects(project_settings, store, planted_move, out_dir):
    rows = _drivers(project_settings, store, planted_move, [_event(affects=[])], out_dir)

    assert len(rows) == 1
    assert (rows[0]["relation"], rows[0]["direction_match"], rows[0]["evidence_doc"]) == (
        "TAGGED_WITH", "unknown", PLANTED_BULLETIN_REF,
    )
    unknown_weight = project_settings.get("linking.direction_weights")["unknown"]
    assert rows[0]["score"] == pytest.approx(_AFFECT_CONFIDENCE * unknown_weight * 0.5 * math.exp(-10 / 14))


def test_only_top_n_drivers_are_kept_per_move(project_settings, store, planted_move, out_dir):
    events = [_event(event_id=f"event:{i}", date=f"2025-04-{10 + i:02d}") for i in range(7)]

    rows = _drivers(project_settings, store, planted_move, events, out_dir)

    assert [r["event_id"] for r in rows] == [f"event:{i}" for i in (6, 5, 4, 3, 2)]


@pytest.fixture
def explain(linked, store, out_dir):
    settings = linked[0]
    semantic = read_yaml(out_dir / "semantic.yaml")
    links = _read_jsonl(out_dir / "links" / "candidate_drivers.jsonl")
    moves = _read_jsonl(out_dir / "structured" / "moves.jsonl")
    with contextlib.closing(open_semantic(settings)) as con_sem:
        yield lambda entity_id, start, end: explain_move(con_sem, semantic, store, links, moves, entity_id, start, end)


def test_explain_move_lists_the_planted_event_with_series_and_move(explain):
    result = explain("series:wht_mill_gp", "2025-04", "2025-05")

    assert [row[3:5] for row in result["series"]["rows"]] == [["2025-04-01", 300.2125], ["2025-05-01", 344.625]]
    assert [m["id"] for m in result["moves"]] == [_MOVE_ID]
    driver = result["drivers"][0]
    assert (driver["lag_days"], driver["quote"], driver["direction_match"]) == (10, PLANTED_QUOTE, "match")
    assert driver["docs"] == [PLANTED_BULLETIN_REF, _SECOND_REF]
    assert "Drought" in driver["event"] and "note" not in result
    assert result["coverage"] == list(_WINDOW)


def test_explain_move_outside_the_document_window_gives_coverage_note(explain):
    result = explain("series:wht_mill_gp", "2024-03", "2024-06")

    assert result["drivers"] == []
    assert result["note"] == "No documents cover 2024-03-01 to 2024-06-30; documents cover 2025-03-01 to 2025-06-30."
    assert result["series"]["rows"]


def test_explain_move_without_moves_says_so(explain):
    result = explain("series:mze_rv", "2025-04", "2025-05")

    assert (result["moves"], result["drivers"]) == ([], [])
    assert result["note"].startswith("No moves in this period")
    assert result["series"]["rows"]


def test_move_without_linked_events_says_no_candidate_drivers(linked, store, out_dir):
    semantic = read_yaml(out_dir / "semantic.yaml")
    moves = _read_jsonl(out_dir / "structured" / "moves.jsonl")
    with contextlib.closing(open_semantic(linked[0])) as con_sem:
        result = explain_move(con_sem, semantic, store, [], moves, "series:wht_mill_gp", "2025-04", "2025-05")

    assert [m["id"] for m in result["moves"]] == [_MOVE_ID]
    assert result["drivers"] == []
    assert result["note"].startswith("No candidate drivers found")


def test_drivers_come_before_a_capped_series(linked, store, out_dir):
    semantic = read_yaml(out_dir / "semantic.yaml")
    links = _read_jsonl(out_dir / "links" / "candidate_drivers.jsonl")
    moves = _read_jsonl(out_dir / "structured" / "moves.jsonl")
    with contextlib.closing(open_semantic(linked[0])) as con_sem:
        result = explain_move(con_sem, semantic, store, links, moves, "series:wht_mill_gp", None, None, limit=5)

    # Drivers lead the JSON so a transcript cut keeps them; the full-history series is capped and flagged.
    assert list(result)[0] == "drivers" and result["drivers"]
    assert len(result["series"]["rows"]) == 5 and result["series"]["truncated"]


def test_review_flagged_events_and_low_confidence_affects_are_excluded(project_settings, store, planted_move, out_dir):
    low_confidence = {**_event()["affects"][0], "confidence": 0.79}
    rows = _drivers(
        project_settings,
        store,
        planted_move,
        [
            _event(event_id="event:flagged", needs_review=True),
            _event(event_id="event:low", affects=[low_confidence]),
        ],
        out_dir,
    )

    assert rows == []


def test_review_flagged_hierarchy_and_location_edges_do_not_link_events(
    project_settings, store, planted_move, out_dir,
):
    def edge(src, rel, dst):
        store._graph.add_edge(
            src, dst, src=src, rel=rel, dst=dst, props={}, confidence=0.5, source="test",
            evidence_doc="", evidence_quote="", needs_review=True,
        )

    edge("series:wht_mill_gp", "HAS_PARENT", "crop:review-parent")
    edge("location:great_plains", "PART_OF", "location:review-region")
    events = [
        _event(
            event_id="event:review-parent",
            affects=[{**_event()["affects"][0], "entity_id": "crop:review-parent"}],
            doc_ids=[],
        ),
        _event(event_id="event:review-location", location_ids=["location:review-region"]),
    ]

    assert _drivers(project_settings, store, planted_move, events, out_dir) == []


def test_default_ranking_ignores_direction_and_drivers_carry_aspect(project_settings, store, planted_move, out_dir):
    up = _event()["affects"][0]
    events = [
        _event(event_id="event:a_down", affects=[{**up, "aspect": "output", "direction": "down"}]),
        _event(event_id="event:b_up", affects=[{**up, "aspect": "price", "direction": "up"}]),
    ]

    rows = _drivers(project_settings, store, planted_move, events, out_dir)
    by_id = {row["event_id"]: row for row in rows}

    # Same confidence and lag: same score whatever the direction; the agent judges it from aspect + direction.
    assert by_id["event:a_down"]["score"] == pytest.approx(by_id["event:b_up"]["score"])
    assert (by_id["event:a_down"]["aspect"], by_id["event:a_down"]["direction"]) == ("output", "down")
    assert (by_id["event:b_up"]["aspect"], by_id["event:b_up"]["direction"]) == ("price", "up")


def test_explain_move_matches_moves_whose_entity_id_is_not_a_str(linked, store, out_dir):
    class Number:  # stands in for an int id read back from JSON
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return self.value

    semantic = read_yaml(out_dir / "semantic.yaml")
    moves = [{**m, "entity_id": Number(m["entity_id"])} for m in _read_jsonl(out_dir / "structured" / "moves.jsonl")]
    with contextlib.closing(open_semantic(linked[0])) as con_sem:
        result = explain_move(con_sem, semantic, store, [], moves, "series:wht_mill_gp", "2025-04", "2025-05")

    assert [m["id"] for m in result["moves"]] == [_MOVE_ID]
