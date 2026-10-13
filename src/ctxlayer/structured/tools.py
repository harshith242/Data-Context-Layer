"""Structured tools: metrics by entity name and validated SQL over the semantic views."""
from ctxlayer.common.catalog import Entity
from ctxlayer.common.db import quote_ident, run_select
from ctxlayer.common.resolve import ResolveResult, Resolver, Status
from ctxlayer.structured.compiler import MetricRequest, run_metric
from ctxlayer.structured.semantic_views import validate_sql

_UNKNOWN_ENTITY = "unknown_entity"
_AMBIGUOUS_ENTITY = "ambiguous_entity"
_UNCONFIRMED_ENTITY = "unconfirmed_entity"
_NO_DATA = "no_data"
_INVALID_SQL = "invalid_sql"
_KIND_MEASURES = "measures"


def query_metric(
    con_sem, semantic: dict, resolver: Resolver, catalog_by_id: dict[str, Entity],
    metric: str, entities: list[str], limit: int | None = None, **kw,
) -> dict:
    """Run `metric` for entity names (e.g. "Widget (large)"); unresolved or unconfirmed names return an error, no SQL runs."""
    entity_ids, warnings = [], []
    for name in entities:
        result = resolver.resolve(name)
        if result.status is not Status.MATCHED:
            return _entity_error(name, result)

        # A low-confidence guess is never queried: the agent retries with the exact name.
        match = result.matches[0]
        if match.needs_review:
            return _error(_UNCONFIRMED_ENTITY, name, [match.name], f"{name!r} may mean {match.name!r}; confirm")

        ids = _series_ids(con_sem, semantic, catalog_by_id[match.id])
        if not ids:
            return _error(_NO_DATA, name, [], f"{match.name!r} has no rows in the measures views")
        if len(ids) > 1:
            warnings.append(f"{name!r} stands for {len(ids)} series")
        entity_ids += ids

    request = MetricRequest(metric=metric, entities=entities, **kw)
    result = run_metric(con_sem, request, semantic, entity_ids, limit)
    if "error" in result:
        return result

    return {**result, "warnings": warnings + result["warnings"], "_citations": [{"kind": "sql", "sql": result["sql"]}]}


def execute_semantic_sql(con_sem, semantic: dict, sql: str, limit: int) -> dict:
    """Run a SELECT over the semantic views only; anything else returns an error."""
    check = validate_sql(sql, semantic)
    if not check["ok"]:
        return {"error": {"type": _INVALID_SQL, "message": "; ".join(check["errors"])}}

    result = run_select(con_sem, sql, limit)
    return {
        "columns": result.columns, "rows": result.rows, "row_count": result.row_count,
        "truncated": result.truncated, "sql": result.sql, "warnings": check["warnings"],
        "_citations": [{"kind": "sql", "sql": result.sql}],
    }


def _series_ids(con_sem, semantic: dict, entity: Entity) -> list:
    """Raw entity-id values an entity stands for, per measures table.

    The catalog records where the entity was seen ({"table", "column", "value"}); a hierarchy
    or location value expands to every series row carrying it. Inferred entities have no source.
    """
    ids = []
    for table, spec in semantic["tables"].items():
        id_column = spec.get("entity", {}).get("id")
        found = [s for s in entity.attributes.get("sources", []) if s["table"] == table]
        if spec["kind"] != _KIND_MEASURES or not id_column or not found:
            continue

        logical = {column["physical"]: column["logical"] for column in spec["columns"]}.get(found[0]["column"])
        if logical is None:
            continue
        sql = f"SELECT DISTINCT {quote_ident(id_column)} FROM {spec['view']} WHERE {quote_ident(logical)} = ?"
        ids += [row[0] for row in con_sem.execute(sql, [found[0]["value"]]).fetchall()]
    return ids


def _entity_error(name: str, result: ResolveResult) -> dict:
    # Ambiguous: the tied entities; no match: the closest names, so the caller can retry.
    if result.status is Status.AMBIGUOUS:
        return _error(_AMBIGUOUS_ENTITY, name, [match.name for match in result.matches], f"{name!r} is ambiguous")
    return _error(_UNKNOWN_ENTITY, name, result.suggestions, f"{name!r} is not a known entity")


def _error(kind: str, name: str, candidates: list[str], message: str) -> dict:
    return {"error": {"type": kind, "query": name, "candidates": candidates, "message": message}}
