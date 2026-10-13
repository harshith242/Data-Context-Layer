"""Unstructured phases: each reads its inputs from the artifacts folder and writes only its own outputs."""
import contextlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ctxlayer.common.catalog import Entity, load_catalog, save_catalog
from ctxlayer.common.context_files import Step
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.files import read_json, read_jsonl, read_yaml, write_json, write_jsonl, write_yaml
from ctxlayer.common.manifest import write_manifest
from ctxlayer.common.ontology import EVENT_TYPES_FILE, apply_event_type_decisions, final_event_types, load_ontology
from ctxlayer.common.phase_context import PhaseContext
from ctxlayer.common.resolve import Cascade, Resolver, name_rules
from ctxlayer.common.review import UNSTRUCTURED_MARKER
from ctxlayer.common.runlog import event
from ctxlayer.unstructured.chunk import chunk_documents
from ctxlayer.unstructured.dedupe import (
    apply_new_entity_decisions, dedupe_events, resolve_extractions, resolve_tags,
)
from ctxlayer.unstructured.documents import Document, build_documents, doc_window
from ctxlayer.unstructured.embed import index_chunks
from ctxlayer.unstructured.extract import _PROMPT_VERSION, Extraction, extract_all, induce_event_types
from ctxlayer.unstructured.graph_build import build_graph
from ctxlayer.unstructured.mentions import find_mentions

_STAGE = "unstructured"
_PHASE_EVENT = "build_unstructured.phase"
_REVIEW_FILE = "REVIEW.md"
_DOCS = "docs"
_DOCUMENTS_FILE = "docs/documents.jsonl"
_EXTRACTIONS_FILE = "docs/extractions.jsonl"
_FAILURES_FILE = "docs/failures.jsonl"
_RESOLVED_FILE = "docs/resolved.jsonl"
_TAG_EDGES_FILE = "docs/tag_edges.jsonl"
_JUDGE_FAILURES_FILE = "docs/judge_failures.json"
_EVENTS_FILE = "docs/events.jsonl"
_CHUNKS_FILE = "docs/chunks.parquet"
_TEXT_ENTITIES_FILE = "graph/entities_text.yaml"
_ONTOLOGY_FILE = "ontology.yaml"
_GRAPH_DIR = "graph"
_EXTRACT_CACHE_DIR = "extract_cache"
_OTHER_TYPE = "other"
_MIN_EVENT_TYPES = 8
_MAX_EVENT_TYPES = 12


def run_documents(pc: PhaseContext) -> dict:
    event(pc.logger(), _PHASE_EVENT, phase="documents")
    roles = read_yaml(pc.out_dir / "roles.yaml")
    with contextlib.closing(connect_readonly(pc.settings.path("database.path"))) as con:
        docs = build_documents(con, roles)

    write_jsonl(pc.out_dir / _DOCUMENTS_FILE, [asdict(doc) for doc in docs])
    return {"documents": len(docs)}


def run_index(pc: PhaseContext) -> dict:
    event(pc.logger(), _PHASE_EVENT, phase="index")
    chunks = chunk_documents(
        _load_docs(pc.out_dir), pc.settings.get("unstructured.chunk_tokens"), pc.settings.get("unstructured.chunk_overlap"),
    )
    return {"chunks": index_chunks(chunks, pc.embedder, pc.out_dir / _DOCS)["chunks"]}


def run_event_types(pc: PhaseContext) -> dict:
    """Write event_types.yaml; the induced types are reused while ontology.yaml's version is the one they came from.

    `event_types` is what extraction prompts list. When decisions changed anything the file also keeps `induced`
    (the model's types, to apply the decisions to again) and `relabels` ({induced name: final name}, applied to
    events in run_dedupe).
    """
    event(pc.logger(), _PHASE_EVENT, phase="event_types")
    ontology = read_yaml(pc.out_dir / _ONTOLOGY_FILE)
    induced = _saved_event_types(pc.out_dir, ontology)

    # Induced from a sample of document summaries, once per ontology version.
    if not induced:
        induced = induce_event_types(
            _load_docs(pc.out_dir), pc.llm, ontology, pc.settings.get("unstructured.event_type_sample"),
            pc.context.for_step(Step.EVENTS),
        )
    types, relabels = apply_event_type_decisions(induced, pc.decisions)
    saved = {"ontology_version": ontology["version"], "event_types": types}
    if types != induced or relabels:
        saved.update(induced=induced, relabels=relabels, final_types=final_event_types(types, relabels))
    write_yaml(pc.out_dir / EVENT_TYPES_FILE, saved)
    return {"event_types": len(types)}


def run_extraction(pc: PhaseContext) -> dict:
    event(pc.logger(), _PHASE_EVENT, phase="extract")
    docs = _load_docs(pc.out_dir)
    stats: Counter = Counter()
    extractions, failures = extract_all(
        docs, pc.extract_llm, load_ontology(pc.out_dir, prompt=True), pc.out_dir / _DOCS / _EXTRACT_CACHE_DIR,
        pc.settings.get("unstructured.extract_workers"), pc.context.for_step(Step.EVENTS), stats,
    )

    write_jsonl(pc.out_dir / _FAILURES_FILE, failures)
    write_jsonl(pc.out_dir / _EXTRACTIONS_FILE, [
        {"doc_id": doc_id, "events": [item.model_dump() for item in extraction.events]}
        for doc_id, extraction in sorted(extractions.items())
    ])
    return {"extraction_failures": len(failures), "cache_hits": stats["hits"], "cache_misses": stats["misses"]}


def run_resolve(pc: PhaseContext) -> dict:
    event(pc.logger(), _PHASE_EVENT, phase="resolve")
    docs = _load_docs(pc.out_dir)
    roles = read_yaml(pc.out_dir / "roles.yaml")
    ontology = read_yaml(pc.out_dir / _ONTOLOGY_FILE)
    catalog = load_catalog(pc.out_dir / "entities.yaml")
    extractions = {
        row["doc_id"]: Extraction.model_validate({"events": row["events"]})
        for row in read_jsonl(pc.out_dir / _EXTRACTIONS_FILE)
    }

    overrides, blocked = name_rules(pc.decisions)
    resolver = Resolver(
        catalog, pc.embedder, pc.llm, pc.settings, Cascade.FULL,
        cache_path=pc.out_dir / "entity_embeddings.npy", context=pc.context.for_step(Step.JUDGE),
        overrides=overrides, blocked=blocked,
    )
    tag_edges, new_entities = resolve_tags(docs, resolver, roles, ontology, pc.llm, pc.context)
    resolved = resolve_extractions(extractions, resolver, ontology)
    resolver.mark_rules_used(pc.decisions)
    new_entities, tag_edges = apply_new_entity_decisions(
        new_entities, tag_edges, pc.decisions, known=frozenset(entity.id for entity in catalog),
    )

    # Mentions: catalog names found in each document's text (no LLM), for every document.
    for doc_id, mentions in find_mentions(docs, catalog).items():
        resolved.setdefault(doc_id, {"mentions": [], "events": []})["mentions"] = mentions

    write_jsonl(pc.out_dir / _RESOLVED_FILE, [_resolved_row(doc_id, item) for doc_id, item in resolved.items()])
    write_jsonl(pc.out_dir / _TAG_EDGES_FILE, tag_edges)
    save_catalog(new_entities, pc.out_dir / _TEXT_ENTITIES_FILE)
    write_json(pc.out_dir / _JUDGE_FAILURES_FILE, sorted(set(resolver.judge_failures)))
    return {"new_entities": len(new_entities), "judge_failures": len(set(resolver.judge_failures))}


def run_dedupe(pc: PhaseContext) -> dict:
    event(pc.logger(), _PHASE_EVENT, phase="dedupe")
    # Renamed or merged event types change the events' type here, so nothing is extracted again.
    relabels = _relabels(pc.out_dir)
    resolved_events = [
        {**item, "doc_id": row["doc_id"], "type": relabels.get(item["type"], item["type"])}
        for row in read_jsonl(pc.out_dir / _RESOLVED_FILE) for item in row["events"]
    ]
    events = dedupe_events(
        resolved_events, pc.embedder, pc.settings.get("unstructured.dedupe_days"),
        pc.settings.get("unstructured.dedupe_similarity"), pc.decisions,
    )

    write_jsonl(pc.out_dir / _EVENTS_FILE, events)
    return {"events": len(events)}


def run_graph(pc: PhaseContext) -> dict:
    """Build graph/*, the unstructured part of REVIEW.md and the manifest; returns the counts of the whole build."""
    event(pc.logger(), _PHASE_EVENT, phase="graph")
    docs = _load_docs(pc.out_dir)
    catalog = load_catalog(pc.out_dir / "entities.yaml")
    new_entities = load_catalog(pc.out_dir / _TEXT_ENTITIES_FILE)
    resolved = {
        row["doc_id"]: {"mentions": row["mentions"], "events": row["events"]}
        for row in read_jsonl(pc.out_dir / _RESOLVED_FILE)
    }
    events = read_jsonl(pc.out_dir / _EVENTS_FILE)

    graph_counts = build_graph(
        catalog, new_entities, docs, read_jsonl(pc.out_dir / _TAG_EDGES_FILE), resolved, events,
        pc.out_dir / _GRAPH_DIR, pc.settings.get("review.confidence_threshold"),
    )
    flagged_edges = _flagged_edges(pc.out_dir / _GRAPH_DIR / "edges.jsonl")
    failures = read_jsonl(pc.out_dir / _FAILURES_FILE)
    induced = load_ontology(pc.out_dir, prompt=True)["event_types"]
    type_count = len([item for item in induced if item["name"] != _OTHER_TYPE])
    warning = None
    if not _MIN_EVENT_TYPES <= type_count <= _MAX_EVENT_TYPES:
        warning = f"Expected {_MIN_EVENT_TYPES} to {_MAX_EVENT_TYPES} event types; got {type_count}."
    _append_review(
        pc.out_dir / _REVIEW_FILE, failures, new_entities, _edge_lines(flagged_edges, new_entities, events), warning,
        read_json(pc.out_dir / _JUDGE_FAILURES_FILE),
    )

    counts = {
        "documents": len(docs), "chunks": len(pd.read_parquet(pc.out_dir / _CHUNKS_FILE, columns=["doc_id"])),
        "events": len(events), "extraction_failures": len(failures), "new_entities": len(new_entities),
        "nodes": graph_counts["nodes"], "edges": graph_counts["edges"], "needs_review": len(flagged_edges),
    }
    event(pc.logger(), _PHASE_EVENT, phase="manifest")
    with contextlib.closing(connect_readonly(pc.settings.path("database.path"))) as con:
        write_manifest(
            pc.out_dir, _STAGE, settings=pc.settings, con=con,
            models={"llm": type(pc.llm).__name__, "model": pc.settings.get("llm.model"),
                    "embeddings": pc.settings.get("embeddings.model")},
            extra={**counts, "coverage": list(doc_window(docs)) if docs else None, "extract_prompt": _PROMPT_VERSION},
        )
    return counts


def _load_docs(out_dir: Path) -> list[Document]:
    return [Document(**row) for row in read_jsonl(out_dir / _DOCUMENTS_FILE)]


def _relabels(out_dir: Path) -> dict[str, str]:
    path = out_dir / EVENT_TYPES_FILE
    return read_yaml(path).get("relabels", {}) if path.is_file() else {}


def _saved_event_types(out_dir: Path, ontology: dict) -> list[dict]:
    """Event types induced for this ontology version, else []; a pre-plan build's types are moved to event_types.yaml."""
    path = out_dir / EVENT_TYPES_FILE
    if not path.is_file() and ontology.get("event_types"):
        # Unchanged, so extraction prompts (and the extraction cache keyed on them) stay the same.
        write_yaml(path, {"ontology_version": ontology["version"], "event_types": ontology["event_types"]})

    if not path.is_file():
        return []
    saved = read_yaml(path)
    if saved.get("ontology_version") != ontology["version"]:
        return []
    induced = saved.get("induced", saved["event_types"])
    # Only the fallback means induction failed; it is kept for this build's prompts but asked again next build.
    return [] if [item["name"] for item in induced] == [_OTHER_TYPE] else induced


def _resolved_row(doc_id: str, item: dict) -> dict:
    """{doc_id, mentions, events}; each event gets its extraction key "<doc_id>#<n>", n counting from 0."""
    events = [{"key": f"{doc_id}#{position}", **item_event} for position, item_event in enumerate(item["events"])]
    return {"doc_id": doc_id, "mentions": item["mentions"], "events": events}


def _flagged_edges(path: Path) -> list[dict]:
    edges = read_jsonl(path)
    return sorted((edge for edge in edges if edge["needs_review"]), key=lambda edge: edge["confidence"])


def _edge_lines(flagged: list[dict], new_entities: list[Entity], events: list[dict]) -> list[str]:
    """One line per target entity of the flagged edges, most edges first.

    e.g. "crop:oats: new entity from tag column crop_tag; 40 edges, 40 articles". The reason is the lowest-confidence
    edge's: a new entity from a tag column, a flagged event, else the low name-match score.
    """
    columns = {entity.id: entity.attributes.get("tag_column") for entity in new_entities}
    flagged_events = {item["event_id"] for item in events if item["needs_review"]}
    groups: dict[str, list[dict]] = {}
    for edge in flagged:
        groups.setdefault(edge["dst"], []).append(edge)

    lines = []
    for target, edges in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        first = edges[0]
        if target in columns:
            reason = f"new entity from tag column {columns[target]}"
        elif first["src"] in flagged_events:
            reason = "flagged event"
        else:
            reason = f"low name-match score {first['confidence']:.2f}"
        articles = {item["doc"] for edge in edges for item in edge.get("evidence", []) if item["doc"]}
        lines.append(f"{target}: {reason}; {len(edges)} edges, {len(articles)} articles")
    return lines


def _append_review(
    path: Path, failures: list[dict], new_entities: list[Entity], flagged: list[str], event_type_warning: str | None,
    judge_failures: list[str],
) -> None:
    """Replace the unstructured sections of REVIEW.md; `flagged` are the low-confidence edge lines."""
    sections = [
        ("Extraction failures", [f"{item['doc_id']}: {item['error']}" for item in failures]),
        ("Event type induction", [event_type_warning] if event_type_warning else []),
        ("New entities from documents", [f"{entity.id} ({entity.type})" for entity in new_entities]),
        ("Low-confidence edges", flagged),
        ("Names the resolution judge could not answer (treated as no match)", sorted(set(judge_failures))),
    ]
    lines = [UNSTRUCTURED_MARKER, ""]
    for title, items in sections:
        lines += [f"## {title}", ""]
        lines += [f"- {item}" for item in items] if items else ["none"]
        lines.append("")

    existing = path.read_text() if path.is_file() else ""
    kept = existing.split(UNSTRUCTURED_MARKER)[0].rstrip("\n")
    path.write_text((kept + "\n\n" if kept else "") + "\n".join(lines))
