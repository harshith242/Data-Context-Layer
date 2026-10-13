"""The phase registry: order, inputs and the facts a fingerprint is made of (spec section 4)."""
from dataclasses import dataclass
from typing import Callable

from ctxlayer.common.context_files import Step
from ctxlayer.common.decisions import Kind
from ctxlayer.common.phase_context import PhaseContext
from ctxlayer.linking.phases import run_links
from ctxlayer.structured.phases import (
    run_catalog, run_moves, run_ontology, run_profile, run_roles, run_semantic, run_views,
)
from ctxlayer.unstructured.extract import _PROMPT_VERSION as EXTRACT_PROMPT_VERSION
from ctxlayer.unstructured.phases import (
    run_dedupe, run_documents, run_event_types, run_extraction, run_graph, run_index, run_resolve,
)

# An input that is not a file: the database schema and each table's row count.
DATABASE = "@database"
_VERSION = "1"


@dataclass(frozen=True)
class Phase:
    name: str
    run: Callable[[PhaseContext], dict]
    inputs: tuple[str, ...]           # files under out_dir, or DATABASE
    outputs: tuple[str, ...]          # files under out_dir that `ctx adopt` looks for
    settings_keys: tuple[str, ...] = ()
    steps: tuple[Step, ...] = ()
    kinds: frozenset[Kind] = frozenset()
    version: str = _VERSION           # bump when the phase's logic or prompt changes
    review_point: bool = False


PHASES: tuple[Phase, ...] = (
    Phase("profile", run_profile, (DATABASE,), ("profile.json",), ("profile",)),
    Phase(
        "roles", run_roles, ("profile.json",), ("roles.yaml",), ("review.confidence_threshold",),
        (Step.ROLES,), frozenset({Kind.COLUMN}), review_point=True,
    ),
    Phase(
        "ontology", run_ontology, ("roles.yaml", "profile.json"), ("ontology.yaml",), (),
        (Step.ONTOLOGY,), frozenset({Kind.ENTITY_TYPE, Kind.RELATION}),
    ),
    Phase("documents", run_documents, ("roles.yaml", DATABASE), ("docs/documents.jsonl",)),
    Phase(
        "index", run_index, ("docs/documents.jsonl",), ("docs/chunks.parquet", "docs/fts.duckdb"),
        ("unstructured.chunk_tokens", "unstructured.chunk_overlap", "embeddings.model"),
    ),
    Phase(
        "event-types", run_event_types, ("ontology.yaml", "docs/documents.jsonl"), ("event_types.yaml",),
        ("unstructured.event_type_sample",), (Step.EVENTS,), frozenset({Kind.EVENT_TYPE}), review_point=True,
    ),
    Phase(
        "catalog", run_catalog, ("roles.yaml", "ontology.yaml", DATABASE), ("entities.yaml",),
        ("review.confidence_threshold",), (Step.CATALOG,), frozenset({Kind.ENTITY}),
    ),
    Phase(
        "semantic", run_semantic, ("roles.yaml", "profile.json", DATABASE), ("semantic.yaml",), (),
        (Step.SEMANTIC,), frozenset({Kind.RULE}),
    ),
    Phase("views", run_views, ("semantic.yaml",), ("structured/views.sql",)),
    # roles.yaml and entities.yaml: this phase also refreshes the structured part of REVIEW.md.
    Phase(
        "moves", run_moves,
        ("semantic.yaml", "structured/views.sql", "roles.yaml", "entities.yaml", DATABASE), ("structured/moves.jsonl",),
        ("structured.moves",), (), frozenset({Kind.MOVE}),
    ),
    Phase(
        "extraction", run_extraction, ("docs/documents.jsonl", "ontology.yaml", "event_types.yaml"),
        ("docs/extractions.jsonl", "docs/failures.jsonl"), ("llm.model",), (Step.EVENTS,),
        version=f"{_VERSION}+{EXTRACT_PROMPT_VERSION}",
    ),
    Phase(
        "resolve", run_resolve,
        ("docs/documents.jsonl", "roles.yaml", "ontology.yaml", "entities.yaml", "docs/extractions.jsonl"),
        ("docs/resolved.jsonl", "docs/tag_edges.jsonl", "graph/entities_text.yaml", "docs/judge_failures.json"),
        ("resolve", "review.confidence_threshold", "embeddings.model"), (Step.JUDGE, Step.TAG_TYPES),
        frozenset({Kind.NAME_MATCH, Kind.NEW_ENTITY}),
    ),
    Phase(
        "dedupe", run_dedupe, ("docs/resolved.jsonl", "event_types.yaml"), ("docs/events.jsonl",),
        ("unstructured.dedupe_days", "unstructured.dedupe_similarity", "embeddings.model"), (),
        frozenset({Kind.SAME_EVENT, Kind.EVENT}),
    ),
    Phase(
        "graph", run_graph,
        (
            "docs/documents.jsonl", "entities.yaml", "graph/entities_text.yaml", "docs/tag_edges.jsonl",
            "docs/resolved.jsonl", "docs/events.jsonl", "docs/failures.jsonl", "docs/judge_failures.json",
            "event_types.yaml", "docs/chunks.parquet",
        ),
        ("graph/nodes.jsonl", "graph/edges.jsonl"), ("review.confidence_threshold",),
    ),
    Phase(
        "links", run_links,
        (
            "semantic.yaml", "structured/views.sql", "entities.yaml", "structured/moves.jsonl",
            "graph/nodes.jsonl", "graph/edges.jsonl", DATABASE,
        ),
        ("links/candidate_drivers.jsonl", "links/nodes.jsonl", "links/edges.jsonl"),
        ("linking", "review.confidence_threshold"), (), frozenset({Kind.DRIVER}),
    ),
)

# Phases that only the structured side needs; the build-unstructured shortcut leaves them out.
STRUCTURED_ONLY = frozenset({"catalog", "semantic", "views", "moves"})
