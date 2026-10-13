"""Agent tools: the Tool type, the serving context and the agent's tool set."""
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable

import duckdb

from ctxlayer.common.catalog import Entity, load_catalog
from ctxlayer.common.config import Settings, artifacts_dir
from ctxlayer.common.context_files import ContextFiles, Step, load_context_files
from ctxlayer.common.db import SqlRejected, connect_readonly
from ctxlayer.common.embeddings import Embedder, get_embedder
from ctxlayer.common.files import read_jsonl, read_yaml
from ctxlayer.common.llm import LLM
from ctxlayer.common.ontology import load_ontology
from ctxlayer.common.resolve import Cascade, Resolver, Status
from ctxlayer.serving._errors import error_result
from ctxlayer.serving.context import get_context
from ctxlayer.structured.metrics import Metric
from ctxlayer.linking.bridge import explain_move
from ctxlayer.structured.semantic_views import open_semantic, validate_sql
from ctxlayer.structured.tools import execute_semantic_sql, query_metric
from ctxlayer.unstructured.graph_store import GraphStore
from ctxlayer.unstructured.search import SearchIndex
from ctxlayer.unstructured.tools import graph_explore, search_docs_tool

_DOCS_DIR = "docs"
_GRAPH_DIR = "graph"
_LINKS_DIR = "links"
_ENTITIES_TEXT_FILE = "graph/entities_text.yaml"
_DRIVERS_FILE = "links/candidate_drivers.jsonl"
_MOVES_FILE = "structured/moves.jsonl"
_UNKNOWN_ENTITY = "unknown_entity"
_AMBIGUOUS_ENTITY = "ambiguous_entity"

_SEARCH_DOCS_DESCRIPTION = (
    "Search the documents (hybrid keyword + semantic). Optionally limit to documents about entity_ids "
    "(catalog ids, children included) and to dates start..end. Returns snippets with doc_id, date and the "
    "document coverage window; cite a document as [doc:<doc_id>]."
)
_GRAPH_EXPLORE_DESCRIPTION = (
    "Explore the knowledge graph around one catalog entity id: related entities, events and documents "
    "within depth 1-2, optionally for one relation and a date range."
)
_EXPLAIN_MOVE_DESCRIPTION = (
    "Explain the big moves of one entity's series: the series, its moves and the candidate drivers "
    "(events from documents that precede a move), with evidence quotes and documents. Candidate drivers "
    "are not proven causes. Says so when no document covers the period."
)

_GET_CONTEXT_DESCRIPTION = (
    "Context for a question: matching column descriptions, metrics, entities, rules and the "
    "time coverage of the data. Call it first."
)
_QUERY_METRIC_DESCRIPTION = (
    "Compute one named metric for entity names over a window. Returns rows and the compiled SQL. "
    "spread needs dimension; rank uses n; compare needs inner_metric; projection_path / projection_band "
    "read the projected scenario. An unknown or unconfirmed entity returns an error with candidate names."
)
_VALIDATE_SQL_DESCRIPTION = (
    "Check a SELECT over the semantic views (sem_*) without running it. Returns errors and warnings."
)
_EXECUTE_SQL_DESCRIPTION = (
    "Run one validated SELECT over the semantic views (sem_*) and return the rows."
)
_RESOLVE_ENTITY_DESCRIPTION = (
    "Map a name, abbreviation or misspelling to catalog entities: status matched, ambiguous or no_match."
)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., dict]

    def schema(self) -> dict:
        """Neutral schema handed to the LLM client."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class ServingContext:
    settings: Settings
    con: object
    embedder: Embedder
    artifacts_dir: Path
    llm: LLM | None = None
    _resolvers: dict = field(default_factory=dict, init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)  # derived values, see cached()

    def cached(self, key: str, build):
        """`build()` once per context, e.g. embeddings of every semantic item for get_context."""
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @cached_property
    def _context_files(self) -> ContextFiles:
        return load_context_files(self.settings, self.settings.path("database.path").stem)

    @property
    def notes(self) -> str:
        """Everything a person wrote in context.md, cut to context.max_chars; "" when there is none."""
        return self._context_files.for_step(Step.AGENT)

    @cached_property
    def semantic(self) -> dict:
        return read_yaml(self._artifact("semantic.yaml"))

    @cached_property
    def ontology(self) -> dict:
        self._artifact("ontology.yaml")  # the same "run the build" error when it is missing
        return load_ontology(self.artifacts_dir)

    @cached_property
    def catalog(self) -> list[Entity]:
        """entities.yaml plus the flagged entities found in documents, when `ctx build-unstructured` ran."""
        entities = load_catalog(self._artifact("entities.yaml"))
        text_entities = self.artifacts_dir / _ENTITIES_TEXT_FILE
        if text_entities.is_file():
            entities += load_catalog(text_entities)
        return entities

    @property
    def docs_dir(self) -> Path:
        return self.artifacts_dir / _DOCS_DIR

    @property
    def has_graph(self) -> bool:
        return (self.artifacts_dir / _GRAPH_DIR / "nodes.jsonl").is_file()

    @cached_property
    def store(self) -> GraphStore:
        """Knowledge graph over graph/ and links/ (a missing file is skipped)."""
        names = ("nodes.jsonl", "edges.jsonl")
        paths = [self.artifacts_dir / folder / name for folder in (_GRAPH_DIR, _LINKS_DIR) for name in names]
        return GraphStore.load(paths)

    @cached_property
    def links(self) -> list[dict]:
        path = self.artifacts_dir / _DRIVERS_FILE
        return read_jsonl(path) if path.is_file() else []

    @cached_property
    def moves(self) -> list[dict]:
        path = self.artifacts_dir / _MOVES_FILE
        return read_jsonl(path) if path.is_file() else []

    @cached_property
    def con_sem(self):
        """Semantic views over the source database, read-only."""
        return open_semantic(self.settings)

    def resolver(self, cascade: Cascade) -> Resolver:
        if cascade not in self._resolvers:
            self._resolvers[cascade] = Resolver(
                self.catalog, self.embedder, self.llm, self.settings, cascade,
                cache_path=self.artifacts_dir / "entity_embeddings.npy",
                context=self._context_files.for_step(Step.JUDGE),
            )
        return self._resolvers[cascade]

    def _artifact(self, name: str) -> Path:
        path = self.artifacts_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run `ctx build-structured` first")
        return path


def open_context(settings: Settings, embedder: Embedder | None = None, llm: LLM | None = None) -> ServingContext:
    """Open the database read-only; the embedder (which loads its model lazily) defaults to the configured one."""
    return ServingContext(
        settings=settings,
        con=connect_readonly(settings.path("database.path")),
        embedder=embedder or get_embedder(settings),
        artifacts_dir=artifacts_dir(settings),
        llm=llm,
    )


def build_tools(ctx: ServingContext) -> list[Tool]:
    return [
        _get_context_tool(ctx), _resolve_entity_tool(ctx), _query_metric_tool(ctx, Cascade.FULL),
        _validate_sql_tool(ctx), _execute_sql_tool(ctx), _search_docs_tool(ctx), _graph_explore_tool(ctx),
        _explain_move_tool(ctx),
    ]


def _get_context_tool(ctx: ServingContext) -> Tool:
    def get_context_tool(question: str) -> dict:
        return get_context(ctx, question)

    parameters = {
        "type": "object",
        "properties": {"question": {"type": "string", "description": "The user's question."}},
        "required": ["question"],
    }
    return Tool("get_context", _GET_CONTEXT_DESCRIPTION, parameters, get_context_tool)


def _query_metric_tool(ctx: ServingContext, cascade: Cascade) -> Tool:
    metrics = [metric.value for metric in Metric]
    by_id = {entity.id: entity for entity in ctx.catalog}

    def query_metric_tool(metric: str, entities: list[str], **kw) -> dict:
        limit = ctx.settings.get("agent.sql_row_limit")
        return query_metric(ctx.con_sem, ctx.semantic, ctx.resolver(cascade), by_id, metric, entities, limit, **kw)

    string = {"type": "string"}
    parameters = {
        "type": "object",
        "properties": {
            "metric": {"type": "string", "enum": metrics, "description": "Metric name."},
            "entities": {"type": "array", "items": string, "description": "Entity names."},
            "start": {**string, "description": "Window start: a date or a relative token such as last_6_months."},
            "end": {**string, "description": "Window end: a date."},
            "grain": {**string, "description": "Time grain, e.g. week or month."},
            "scenario": {**string, "description": "Scenario label; default is actuals."},
            "band": {**string, "description": "Value band, e.g. low, mid, high."},
            "dimension": {**string, "description": "Dimension to group or rank by."},
            "n": {"type": "integer", "description": "Count of periods or rows."},
            "inner_metric": {"type": "string", "enum": metrics, "description": "Metric run per entity by compare."},
            "convert_to": {**string, "description": "Denomination to convert to."},
        },
        "required": ["metric", "entities"],
    }
    return Tool("query_metric", _QUERY_METRIC_DESCRIPTION, parameters, query_metric_tool)


def _validate_sql_tool(ctx: ServingContext) -> Tool:
    def validate_sql_tool(sql: str) -> dict:
        return validate_sql(sql, ctx.semantic)

    parameters = {
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
        "required": ["sql"],
    }
    return Tool("validate_sql", _VALIDATE_SQL_DESCRIPTION, parameters, validate_sql_tool)


def _execute_sql_tool(ctx: ServingContext) -> Tool:
    limit = ctx.settings.get("agent.sql_row_limit")

    def execute_sql_tool(sql: str) -> dict:
        try:
            return execute_semantic_sql(ctx.con_sem, ctx.semantic, sql, limit)
        except (SqlRejected, duckdb.Error) as exc:
            return error_result("sql_error", str(exc))

    parameters = {
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "A single SELECT over sem_* views."}},
        "required": ["sql"],
    }
    return Tool("execute_semantic_sql", _EXECUTE_SQL_DESCRIPTION, parameters, execute_sql_tool)


def _resolve_entity_tool(ctx: ServingContext) -> Tool:
    def resolve_entity(text: str, type: str | None = None) -> dict:
        return asdict(ctx.resolver(Cascade.FULL).resolve(text, type))

    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The name as written by the user."},
            "type": {"type": "string", "description": "Restrict to one entity type."},
        },
        "required": ["text"],
    }
    return Tool("resolve_entity", _RESOLVE_ENTITY_DESCRIPTION, parameters, resolve_entity)


def _search_docs_tool(ctx: ServingContext) -> Tool:
    default_k = ctx.settings.get("agent.docs_top_k")
    max_k = ctx.settings.get("agent.docs_top_k_max")

    def search_docs(
        query: str, entity_ids: list[str] | None = None, start: str | None = None, end: str | None = None,
        k: int = default_k,
    ) -> dict:
        # The model picks k: keep it between 1 and top_k_max.
        result = search_docs_tool(
            ctx.docs_dir, ctx.embedder, ctx.store, query, entity_ids, start, end, max(1, min(k, max_k)),
            ctx.settings.get("unstructured.rrf_k"), ctx.cached("docs_index", lambda: SearchIndex(ctx.docs_dir)),
        )
        if "error" in result:
            return result

        doc_ids = dict.fromkeys(hit["doc_id"] for hit in result["hits"])
        result["_citations"] = [{"kind": "doc", "doc_id": doc_id} for doc_id in doc_ids]
        return result

    string = {"type": "string"}
    parameters = {
        "type": "object",
        "properties": {
            "query": {**string, "description": "What to look for, in plain words."},
            "entity_ids": {"type": "array", "items": string, "description": "Catalog entity ids to limit to."},
            "start": {**string, "description": "Earliest document date, e.g. 2025-03-01."},
            "end": {**string, "description": "Latest document date."},
            "k": {"type": "integer", "description": f"Number of snippets to return (default {default_k})."},
        },
        "required": ["query"],
    }
    return Tool("search_docs", _SEARCH_DOCS_DESCRIPTION, parameters, search_docs)


def _graph_explore_tool(ctx: ServingContext) -> Tool:
    def graph_explore_tool(
        entity_id: str, relation: str | None = None, depth: int = 1, start: str | None = None, end: str | None = None,
        min_confidence: float = 0.5,
    ) -> dict:
        result = graph_explore(ctx.store, entity_id, relation, depth, start, end, min_confidence)
        if "error" not in result:
            return result
        found = ctx.resolver(Cascade.FULL).resolve(entity_id)
        candidates = [match.name for match in found.matches] if found.status is Status.AMBIGUOUS else found.suggestions
        return _entity_error(found.status.value if found.status is Status.AMBIGUOUS else _UNKNOWN_ENTITY, entity_id, candidates)

    string = {"type": "string"}
    parameters = {
        "type": "object",
        "properties": {
            "entity_id": {**string, "description": "Catalog entity id, e.g. from get_context or resolve_entity."},
            "relation": {**string, "description": "Only this relation, e.g. AFFECTS."},
            "depth": {"type": "integer", "description": "Hops from the entity, 1 or 2 (default 1)."},
            "start": {**string, "description": "Earliest edge date."},
            "end": {**string, "description": "Latest edge date."},
            "min_confidence": {"type": "number", "description": "Minimum edge confidence (default 0.5)."},
        },
        "required": ["entity_id"],
    }
    return Tool("graph_explore", _GRAPH_EXPLORE_DESCRIPTION, parameters, graph_explore_tool)


def _explain_move_tool(ctx: ServingContext) -> Tool:
    def explain_move_tool(entity: str, start: str | None = None, end: str | None = None) -> dict:
        found = ctx.resolver(Cascade.FULL).resolve(entity)
        if found.status is Status.MATCHED:
            return explain_move(
                ctx.con_sem, ctx.semantic, ctx.store, ctx.links, ctx.moves, found.matches[0].id, start, end,
                ctx.settings.get("agent.sql_row_limit"),
            )

        # Ambiguous: the tied entities; no match: the closest names, so the caller can retry.
        if found.status is Status.AMBIGUOUS:
            return _entity_error(_AMBIGUOUS_ENTITY, entity, [match.name for match in found.matches])
        return _entity_error(_UNKNOWN_ENTITY, entity, found.suggestions)

    string = {"type": "string"}
    parameters = {
        "type": "object",
        "properties": {
            "entity": {**string, "description": "Entity name as written by the user."},
            "start": {**string, "description": "Window start: a date or month, e.g. 2025-04."},
            "end": {**string, "description": "Window end."},
        },
        "required": ["entity"],
    }
    return Tool("explain_move", _EXPLAIN_MOVE_DESCRIPTION, parameters, explain_move_tool)


def _entity_error(kind: str, name: str, candidates: list[str]) -> dict:
    message = f"{name!r} is not one known entity"
    return {"error": {"type": kind, "query": name, "candidates": candidates, "message": message}}
