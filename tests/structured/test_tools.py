"""Structured tools on domain_a: names resolve to catalog entities, bad names and bad SQL are refused."""
import pytest
from fakes import FakeEmbedder, FakeLLM
from fixtures.domain_a.make_domain import PLANTED_SERIES_CODE

from ctxlayer.common.catalog import Entity
from ctxlayer.common.resolve import Cascade, Resolver
from ctxlayer.structured.compiler import MetricRequest, run_metric
from ctxlayer.structured.semantic_views import open_semantic
from ctxlayer.structured.tools import execute_semantic_sql, query_metric

_METRIC = "latest_value"


@pytest.fixture
def sem(domain_a_semantic):
    settings, semantic = domain_a_semantic
    con = open_semantic(settings)
    yield con
    con.close()


@pytest.fixture
def resolver_parts(domain_a_semantic, domain_a_catalog):
    settings, _ = domain_a_semantic
    _, entities = domain_a_catalog
    llm = FakeLLM({"resolve_judge": '{"id": null, "confidence": 0}'})
    resolver = Resolver(entities, FakeEmbedder(), llm, settings, Cascade.FULL)
    return resolver, {entity.id: entity for entity in entities}, entities


def _planted(entities):
    return next(e for e in entities if PLANTED_SERIES_CODE in e.aliases)


def test_typo_entity_resolves_and_matches_direct_run_metric(sem, domain_a_semantic, resolver_parts):
    _, semantic = domain_a_semantic
    resolver, by_id, entities = resolver_parts
    name = _planted(entities).canonical_name
    typo = name[:-1] + "x" if name[-1] != "x" else name[:-1] + "y"

    result = query_metric(sem, semantic, resolver, by_id, _METRIC, [typo])
    direct = run_metric(
        sem, MetricRequest(metric=_METRIC, entities=[PLANTED_SERIES_CODE]), semantic, [PLANTED_SERIES_CODE],
    )

    assert result["rows"] == direct["rows"]
    assert result["rows"][0][result["columns"].index("series_code")] == PLANTED_SERIES_CODE
    assert result["_citations"] == [{"kind": "sql", "sql": direct["sql"]}]


def test_unknown_entity_returns_candidates_and_runs_no_sql(domain_a_semantic, resolver_parts):
    _, semantic = domain_a_semantic
    resolver, by_id, _ = resolver_parts

    class _NoSql:
        def execute(self, *args, **kwargs):
            raise AssertionError("SQL must not run for an unresolved entity")

    result = query_metric(_NoSql(), semantic, resolver, by_id, _METRIC, ["Zanzibar saffron"])

    assert result["error"]["type"] == "unknown_entity"
    assert result["error"]["query"] == "Zanzibar saffron"
    assert result["error"]["candidates"]


def test_execute_semantic_sql_blocks_a_physical_table(sem, domain_a_semantic):
    _, semantic = domain_a_semantic

    result = execute_semantic_sql(sem, semantic, "SELECT * FROM grain_quotes", 10)

    assert result["error"]["type"] == "invalid_sql"
    assert "grain_quotes" in result["error"]["message"]


def _resolver(settings, entities, judge_reply: str) -> Resolver:
    return Resolver(entities, FakeEmbedder(), FakeLLM({"resolve_judge": judge_reply}), settings, Cascade.FULL)


def test_hierarchy_name_expands_to_its_series_with_a_warning(sem, domain_a_semantic, resolver_parts, domain_a):
    _, semantic = domain_a_semantic
    resolver, by_id, entities = resolver_parts
    group = next(e for e in entities if e.type == "CropGroup")
    expected = {r[0] for r in sem.execute(
        "SELECT DISTINCT series_code FROM sem_grain_quotes WHERE crop_group = ?", [group.canonical_name]).fetchall()}

    result = query_metric(sem, semantic, resolver, by_id, _METRIC, [group.canonical_name])
    codes = {row[result["columns"].index("series_code")] for row in result["rows"]}

    assert codes == expected and len(codes) > 1
    assert f"{group.canonical_name!r} stands for {len(expected)} series" in result["warnings"]


def test_low_confidence_judge_match_is_not_queried(domain_a_semantic, domain_a_catalog):
    settings, semantic = domain_a_semantic
    _, entities = domain_a_catalog
    target = _planted(entities)
    resolver = _resolver(settings, entities, f'{{"id": "{target.id}", "confidence": 0.4}}')

    class _NoSql:
        def execute(self, *args, **kwargs):
            raise AssertionError("SQL must not run for an unconfirmed entity")

    # Shares words with the target so it reaches the embedding shortlist, but no exact / alias / fuzzy match.
    vague = f"{target.canonical_name.split()[0]} qqzz vvxx"
    result = query_metric(_NoSql(), semantic, resolver, {e.id: e for e in entities}, _METRIC, [vague])

    assert result["error"]["type"] == "unconfirmed_entity"
    assert result["error"]["candidates"] == [target.canonical_name]


def test_inferred_parent_without_rows_reports_no_data(sem, domain_a_semantic, resolver_parts):
    _, semantic = domain_a_semantic
    resolver, by_id, entities = resolver_parts
    parent = next(e for e in entities if e.source == "inferred" and e.type == "Location")

    result = query_metric(sem, semantic, resolver, by_id, _METRIC, [parent.canonical_name])

    assert result["error"]["type"] == "no_data"


def test_value_seen_in_a_text_table_first_still_finds_its_measures_rows(sem, domain_a_semantic, project_settings):
    _, semantic = domain_a_semantic
    region = sem.execute("SELECT market_region FROM sem_grain_quotes LIMIT 1").fetchone()[0]
    sources = [
        {"table": "farm_bulletins", "column": "region_tag", "value": region},
        {"table": "grain_quotes", "column": "market_region", "value": region},
    ]
    place = Entity("location:x", "Location", region, attributes={"sources": sources})
    resolver = Resolver([place], FakeEmbedder(), None, project_settings, Cascade.STRICT)
    expected = {r[0] for r in sem.execute(
        "SELECT DISTINCT series_code FROM sem_grain_quotes WHERE market_region = ?", [region]).fetchall()}

    result = query_metric(sem, semantic, resolver, {place.id: place}, _METRIC, [region])

    assert {row[result["columns"].index("series_code")] for row in result["rows"]} == expected
