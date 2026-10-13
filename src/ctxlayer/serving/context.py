"""get_context: question-relevant semantic items, metrics, entities, rules and coverage."""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import numpy as np

from ctxlayer.common.embeddings import cosine_top_k
from ctxlayer.common.resolve import Cascade, Status
from ctxlayer.structured.metrics import METRICS, Metric
from ctxlayer.unstructured.graph_store import ReviewFilter

if TYPE_CHECKING:  # tools.py imports this module for its get_context tool
    from ctxlayer.serving.tools import ServingContext

_MANIFEST_FILE = "build_manifest.json"
_UNSTRUCTURED_STAGE = "unstructured"
_FROM_CONTEXT = "context.md"
_DEFAULT_SOURCE = "inferred"
_ITEMS_KEY = "semantic_items"
_ENTITIES_KEY = "entities"

# Spans of the question that may name an entity: "quoted text" or runs of Capitalised words,
# e.g. "Product A - Region B".
_QUOTED = re.compile(r"\"([^\"]+)\"|'([^']+)'")
_CAPITALISED = re.compile(r"[A-Z][\w()]*(?:\s+(?:-\s+)?[A-Z][\w()]*)*")
_MAX_NEIGHBOURS = 5
_ENTITY_COLUMNS = ("id", "type", "canonical_name", "aliases", "source", "confidence")

# Metrics used for example questions, in order.
_EXAMPLE_METRICS = (Metric.LATEST_VALUE, Metric.CHANGE_PCT, Metric.VOLATILITY)


def get_context(ctx: ServingContext, question: str, review: ReviewFilter = ReviewFilter.EXCLUDE) -> dict:
    """Context for `question`: {semantic_items, metrics, entities, rules, coverage, provenance}."""
    query = ctx.embedder.embed_queries([question])[0]

    # Semantic items and entities, ranked by cosine to the question; spans the resolver knows come first.
    items = _ranked(query, _embedded(ctx, _ITEMS_KEY), ctx.settings.get("context.top_k_items"), review)
    ranked = _ranked(query, _embedded(ctx, _ENTITIES_KEY), ctx.settings.get("context.top_k_entities"), review)
    entities = _with_resolved(ctx, question, ranked, review)

    result = {
        "semantic_items": items,
        "metrics": sorted(METRICS),
        "entities": entities,
        "rules": _rules(ctx.semantic),
        "coverage": coverage(ctx),
        "provenance": {"sources": sorted({item["source"] for item in items + entities}), "review": review.value},
    }

    # Only after `ctx build-unstructured`: the induced event types and what the graph links to the resolved entities.
    if ctx.ontology.get("event_types"):
        result["event_types"] = [item["name"] for item in ctx.ontology["event_types"]]
    if ctx.has_graph:
        result["graph_neighbours"] = _neighbours(ctx, [item["id"] for item in entities if "method" in item])
    return result


def coverage(ctx: ServingContext) -> dict:
    """Time range of every view and the data's "now"; `documents` is the [first, last] document date, if any."""
    now = ctx.semantic["now"]["value"]
    views = []

    for table in ctx.semantic["tables"].values():
        time_range = table.get("time", {}).get("range")
        if not time_range:
            continue
        views.append({"view": table["view"], "time_range": list(time_range), "now": now})

    return {"views": views, "documents": ctx.cached("doc_window", lambda: _doc_window(ctx))}


def _doc_window(ctx: ServingContext) -> list[str] | None:
    path = ctx.artifacts_dir / _MANIFEST_FILE
    if not path.is_file():
        return None

    manifest = json.loads(path.read_text())
    coverage = manifest.get("stages", {}).get(_UNSTRUCTURED_STAGE, {}).get("extra", {}).get("coverage")
    return coverage if isinstance(coverage, list) and len(coverage) == 2 else None


def _neighbours(ctx: ServingContext, entity_ids: list[str]) -> list[dict]:
    """The most confident graph edges of the entities, e.g. {entity: "type:item", relation: "AFFECTS", ...}."""
    edges = []
    for entity_id in entity_ids:
        for edge in ctx.store.neighbors(entity_id, review=ReviewFilter.EXCLUDE):
            edges.append({
                "entity": entity_id, "relation": edge["rel"], "neighbour": edge["neighbor"],
                "confidence": edge["confidence"],
            })

    edges.sort(key=lambda edge: (-edge["confidence"], edge["entity"], edge["relation"], edge["neighbour"]))
    return edges[:_MAX_NEIGHBOURS]


def example_questions(ctx: ServingContext, n: int = 3) -> list[str]:
    """Template questions over metric names and measured entities, e.g. "What is the volatility of Widget A?"."""
    entities = [entity.canonical_name for entity in ctx.catalog if not entity.needs_review and _measured(ctx, entity)]
    if not entities:
        return []

    questions = []
    for position in range(n):
        metric = _EXAMPLE_METRICS[position % len(_EXAMPLE_METRICS)].value.replace("_", " ")
        questions.append(f"What is the {metric} of {entities[position % len(entities)]}?")
    return questions


def _measured(ctx: ServingContext, entity) -> bool:
    """True for entities seen in the entity column of a measures table (not a location or a level)."""
    for table, spec in ctx.semantic["tables"].items():
        entity_logicals = set(spec.get("entity", {}).values())
        physical = {column["physical"] for column in spec["columns"] if column["logical"] in entity_logicals}
        if any(s["table"] == table and s["column"] in physical for s in entity.attributes.get("sources", [])):
            return True
    return False


def _semantic_items(semantic: dict) -> list[dict]:
    """One item per column description, measure and synonym, each with its own text to rank by."""
    items = []
    columns = {}

    for table_name, table in semantic["tables"].items():
        for column in table["columns"]:
            columns[column["logical"]] = column
            text = f"{table_name}.{column['logical']}: {column['description']}"
            items.append(_item(column, text, kind="column", table=table_name))
        for measure in table.get("measures", []):
            column = columns.get(measure["logical"], measure)
            text = f"{table_name}.{measure['logical']} measure, {measure['band']} band, {measure['aggregation']}"
            items.append(_item(column, text, kind="measure", table=table_name))

    # A synonym inherits source and confidence from the column it names.
    for logical, synonyms in semantic.get("synonyms", {}).items():
        column = columns.get(logical)
        if column is None:
            continue
        for synonym in synonyms:
            items.append(_item(column, f"{logical}: {synonym}", kind="synonym", term=synonym))

    return items


def _item(origin: dict, text: str, **fields) -> dict:
    return {
        "text": text,
        **fields,
        "source": origin.get("source", _DEFAULT_SOURCE),
        "confidence": origin.get("confidence"),
        "needs_review": origin.get("needs_review", False),
    }


def _entity_items(ctx: ServingContext) -> list[dict]:
    items = []
    for entity in ctx.catalog:
        item = {name: getattr(entity, name) for name in _ENTITY_COLUMNS}
        items.append({**item, "text": entity.canonical_name, "needs_review": entity.needs_review})
    return items


def _embedded(ctx: ServingContext, key: str) -> tuple[list[dict], np.ndarray]:
    """(items, one embedding row per item), computed once per ServingContext."""
    def build():
        items = _semantic_items(ctx.semantic) if key == _ITEMS_KEY else _entity_items(ctx)
        return items, ctx.embedder.embed_passages([item["text"] for item in items])

    return ctx.cached(key, build)


def _ranked(query, embedded: tuple[list[dict], np.ndarray], k: int, review: ReviewFilter) -> list[dict]:
    """Top-k items by cosine of their "text" to `query`, with the score added and "text" kept as the label."""
    items, matrix = embedded
    rows = [row for row, item in enumerate(items) if review is ReviewFilter.INCLUDE or not item["needs_review"]]
    if not rows:
        return []

    return [{**items[rows[row]], "score": round(float(score), 4)} for row, score in cosine_top_k(query, matrix[rows], k)]


def _with_resolved(ctx: ServingContext, question: str, ranked: list[dict], review: ReviewFilter) -> list[dict]:
    """Entities the resolver finds for quoted / capitalised spans, then the ranked ones, without repeats."""
    # Full cascade only when an llm can judge the embedding step.
    # Exact + alias only: fuzzy matching and the LLM judge belong to resolve_entity / query_metric.
    resolver = ctx.resolver(Cascade.STRICT)
    by_id = {item["id"]: item for item in _embedded(ctx, _ENTITIES_KEY)[0]}

    found = []
    for span in _spans(question):
        result = resolver.resolve(span)
        if result.status is Status.NO_MATCH:
            continue
        for match in result.matches:
            item = by_id[match.id]
            if item["needs_review"] and review is ReviewFilter.EXCLUDE:
                continue
            found.append({**item, "score": match.score, "method": match.method.value})

    merged = {}
    for item in [*found, *ranked]:
        merged.setdefault(item["id"], item)
    return list(merged.values())[: ctx.settings.get("context.top_k_entities")]


def _spans(question: str) -> list[str]:
    """Quoted and capitalised spans; a lone first word is skipped (sentence case, e.g. "What")."""
    spans = [a or b for a, b in _QUOTED.findall(question)]
    first_word = question.split()[0] if question.split() else ""

    for span in _CAPITALISED.findall(question):
        if span != first_word:
            spans.append(span)
    return list(dict.fromkeys(spans))


def _rules(semantic: dict) -> list[dict]:
    """Denomination, scenario and conversion rules; always returned, never ranked."""
    rules = []

    for table_name, table in semantic["tables"].items():
        denominations = table.get("denominations")
        if denominations:
            text = (
                f"{table_name}: never aggregate across different {' / '.join(denominations)} values; "
                "group per denomination and state it."
            )
            sources = {c.get("source") for c in table["columns"] if c["logical"] in denominations}
            rules.append({"text": text, "source": _FROM_CONTEXT if _FROM_CONTEXT in sources else _DEFAULT_SOURCE})

        scenario = table.get("scenario")
        if scenario:
            projected = [label for label, value in scenario["values"].items() if value != scenario["default"]]
            text = (
                f"{table_name}: default scenario is {scenario['default']}; "
                f"{', '.join(projected)} rows are projections, label them and never mix them with actuals."
            )
            rules.append({"text": text, "source": scenario.get("source", _DEFAULT_SOURCE)})

    for conversion in semantic.get("conversions", []):
        text = f"conversion: {json.dumps(conversion, sort_keys=True)}"
        rules.append({"text": text, "source": conversion.get("source", "context.md")})

    return rules
