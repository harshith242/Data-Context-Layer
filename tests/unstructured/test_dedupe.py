"""Tag / mention resolution against the domain_a catalog, and cross-document event dedupe."""
import json

import duckdb
import pytest
from fakes import FakeEmbedder, FakeLLM

from ctxlayer.common.resolve import Resolver
from ctxlayer.unstructured.documents import Document
from ctxlayer.unstructured.dedupe import dedupe_events, resolve_extractions, resolve_tags
from ctxlayer.unstructured.extract import MIN_AFFECT_CONFIDENCE, Extraction, extract_all
from fixtures.domain_a.extract_replies import EVENT_TYPES_REPLY, extract_reply
from fixtures.domain_a.make_domain import PLANTED_BULLETIN_REF

_SECOND_REF = "FB-20250427-01"
_DAYS = 3
_SIMILARITY = 0.85
_TAG_TYPES = {"tag_types": {"crop_tag": "Crop", "region_tag": "Location"}}
_NO_MATCH_JUDGE = {"id": None, "confidence": 0.0}
_SQL = """
SELECT DISTINCT bulletin_ref, CAST(released_at AS DATE), headline, digest,
       concat_ws(' ', body_part_1, body_part_2, body_part_3)
FROM farm_bulletins ORDER BY bulletin_ref
"""


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM({"tag_types": json.dumps(_TAG_TYPES), "resolve_judge": json.dumps(_NO_MATCH_JUDGE), "resolve_judge_batch": "{}"})


@pytest.fixture
def resolver(domain_a_catalog, project_settings, llm) -> Resolver:
    _, entities = domain_a_catalog
    return Resolver(entities, FakeEmbedder(), llm, project_settings)


def _tag_doc(doc_id: str, crops: list[str], regions: list[str]) -> Document:
    return Document(doc_id, "2025-04-25", "t", "s", "b", {"crop_tag": crops, "region_tag": regions})


def _edges_to(edges: list[dict], doc_id: str) -> dict[str, dict]:
    return {edge["dst"]: edge for edge in edges if edge["evidence_doc"] == doc_id}


def test_parenthesised_tag_resolves_to_catalog_crop(domain_a_roles, domain_a_catalog, resolver, llm):
    _, roles = domain_a_roles
    ontology, _ = domain_a_catalog

    edges, new_entities = resolve_tags([_tag_doc("D1", ["Wheat (milling)"], ["GP"])], resolver, roles, ontology, llm)

    assert new_entities == []
    assert set(_edges_to(edges, "D1")) == {"crop:milling_wheat", "location:great_plains"}
    edge = _edges_to(edges, "D1")["crop:milling_wheat"]
    assert (edge["src"], edge["rel"], edge["source"], edge["confidence"]) == ("doc:D1", "TAGGED_WITH", "metadata", 1.0)
    assert edge["needs_review"] is False


def test_unknown_tag_becomes_flagged_entity_reused_across_docs(domain_a_roles, domain_a_catalog, resolver, llm):
    _, roles = domain_a_roles
    ontology, _ = domain_a_catalog
    docs = [_tag_doc("D1", ["Sunflower"], []), _tag_doc("D2", ["Sunflower"], [])]

    edges, new_entities = resolve_tags(docs, resolver, roles, ontology, llm)

    assert [(e.id, e.type, e.source, e.confidence, e.needs_review) for e in new_entities] == [
        ("crop:sunflower", "Crop", "inferred", 0.5, True)
    ]
    assert new_entities[0].attributes == {"tag_column": "crop_tag"}
    # The tag value is on the documents for certain (1.0); the edges stay flagged until the new entity is reviewed.
    assert sorted((edge["evidence_doc"], edge["dst"], edge["confidence"], edge["needs_review"]) for edge in edges) == [
        ("D1", "crop:sunflower", 1.0, True),
        ("D2", "crop:sunflower", 1.0, True),
    ]
    assert [prompt.split("\n", 1)[0] for prompt, _ in llm.calls].count("[task: resolve_judge_batch]") == 1


def test_region_tag_never_matches_a_crop(domain_a_roles, domain_a_catalog, resolver, llm):
    _, roles = domain_a_roles
    ontology, _ = domain_a_catalog

    edges, new_entities = resolve_tags([_tag_doc("D1", [], ["Wheat (milling)"])], resolver, roles, ontology, llm)

    assert [entity.id for entity in new_entities] == ["location:wheat_milling"]
    assert [edge["dst"] for edge in edges] == ["location:wheat_milling"]


@pytest.fixture(scope="module")
def drought_events(domain_a, domain_a_catalog, project_settings, tmp_path_factory) -> list[dict]:
    """Resolved events of every bulletin, flattened with their doc_id."""
    ontology, entities = domain_a_catalog
    ontology = {**ontology, "event_types": [*EVENT_TYPES_REPLY["event_types"], {"name": "other", "description": ""}]}
    con = duckdb.connect(str(domain_a.db_path), read_only=True)
    rows = con.execute(_SQL).fetchall()
    con.close()
    docs = [Document(str(ref), str(day), title, summary, body) for ref, day, title, summary, body in rows]

    cache_dir = tmp_path_factory.mktemp("extract_cache")
    extractions, failures = extract_all(docs, FakeLLM({"extract": extract_reply}), ontology, cache_dir, workers=1)
    assert failures == []

    judge = FakeLLM({"resolve_judge": json.dumps(_NO_MATCH_JUDGE), "resolve_judge_batch": "{}"})
    resolved = resolve_extractions(extractions, Resolver(entities, FakeEmbedder(), judge, project_settings))
    return [{**event, "doc_id": doc_id} for doc_id, item in resolved.items() for event in item["events"]]


def test_resolve_extractions_maps_names_to_catalog_ids(drought_events):
    event = next(e for e in drought_events if e["doc_id"] == PLANTED_BULLETIN_REF)

    assert event["entity_ids"] == ["crop:milling_wheat"]
    assert event["location_ids"] == ["location:great_plains"]
    assert event["affects"][0]["entity_id"] == "crop:milling_wheat"


def test_resolution_respects_affect_types_and_returns_no_llm_mentions(domain_a_catalog, resolver):
    ontology, _ = domain_a_catalog
    extraction = Extraction.model_validate({
        "events": [{
            "type": "other", "summary": "x", "evidence": "x", "locations": [], "organizations": [],
            "affects": [{"entity": "Great Plains", "direction": "unknown", "confidence": 0.5, "quote": ""}],
        }],
    })

    resolved = resolve_extractions({"D1": extraction}, resolver, ontology)["D1"]

    # A location is not an AFFECTS target; mentions now come from find_mentions, not the LLM.
    assert resolved["events"][0]["affects"][0]["entity_id"] is None
    assert resolved["mentions"] == []


def _dedupe(events: list[dict]) -> list[dict]:
    return dedupe_events(events, FakeEmbedder(), _DAYS, _SIMILARITY)


def _drought(drought_events: list[dict], **changes) -> list[dict]:
    """The two planted drought events (doc 1 and doc 2); `changes` apply to doc 2's event."""
    first = next(e for e in drought_events if e["doc_id"] == PLANTED_BULLETIN_REF)
    second = next(e for e in drought_events if e["doc_id"] == _SECOND_REF)
    return [first, {**second, **changes}]


def test_planted_drought_bulletins_merge_into_one_event(drought_events):
    events = _dedupe(_drought(drought_events))

    assert len(events) == 1
    event = events[0]
    assert event["doc_ids"] == [PLANTED_BULLETIN_REF, _SECOND_REF]
    assert (event["type"], event["date"]) == ("supply_disruption", "2025-04-25")
    assert event["entity_ids"] == ["crop:milling_wheat"]
    assert event["location_ids"] == ["location:great_plains"]
    assert len(event["quotes"]) == 2
    assert event["confidence"] == 0.8
    assert event["needs_review"] is False
    assert [(a["entity_id"], a["direction"], a["doc_id"]) for a in event["affects"]] == [
        ("crop:milling_wheat", "up", PLANTED_BULLETIN_REF),
        ("crop:milling_wheat", "up", _SECOND_REF),
    ]
    assert event["event_id"].startswith("event:") and len(event["event_id"]) == len("event:") + 10


def test_events_five_days_apart_do_not_merge(drought_events):
    assert len(_dedupe(_drought(drought_events, date="2025-04-30"))) == 2


def test_events_without_shared_entity_or_location_do_not_merge(drought_events):
    other = {"entity_ids": ["crop:maize"], "location_ids": ["location:rhine_valley"]}

    assert len(_dedupe(_drought(drought_events, **other))) == 2


def test_identical_unresolved_events_from_one_document_have_stable_distinct_ids(drought_events):
    source = next(e for e in drought_events if e["doc_id"] == PLANTED_BULLETIN_REF)
    unresolved = {**source, "entity_ids": [], "location_ids": [], "affects": []}

    first = _dedupe([unresolved, unresolved.copy()])
    second = _dedupe([unresolved, unresolved.copy()])

    assert len({event["event_id"] for event in first}) == 2
    assert [event["event_id"] for event in first] == [event["event_id"] for event in second]


def test_event_without_a_stated_effect_takes_the_implied_floor(domain_a_catalog, resolver):
    ontology, _ = domain_a_catalog
    extraction = Extraction.model_validate({
        "events": [{"type": "other", "summary": "x", "evidence": "x", "locations": ["Great Plains"],
                    "organizations": [], "affects": []}],
    })

    event = resolve_extractions({"D1": extraction}, resolver, ontology)["D1"]["events"][0]

    # No model-given number exists for it, so it gets the weakest claim extraction keeps, not an invented one.
    assert event["confidence"] == MIN_AFFECT_CONFIDENCE
    assert event["location_scores"] == {"location:great_plains": 1.0}
