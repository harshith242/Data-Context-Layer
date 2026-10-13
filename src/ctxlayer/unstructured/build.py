"""Unstructured build: runs the unstructured phases in order, from documents to graph."""
from enum import Enum
from pathlib import Path

from ctxlayer.common.config import Settings, artifacts_dir
from ctxlayer.common.embeddings import Embedder
from ctxlayer.common.llm import LLM
from ctxlayer.common.phase_context import make_context
from ctxlayer.unstructured.phases import (
    run_dedupe, run_documents, run_event_types, run_extraction, run_graph, run_index, run_resolve,
)

# Read by Start.AFTER_EXTRACT; inputs must be older than the saved extractions, or they were rebuilt since.
_EXTRACTIONS_FILE = "docs/extractions.jsonl"
_SAVED_INPUTS = ("roles.yaml", "ontology.yaml", "entities.yaml", "docs/documents.jsonl")
_SAVED_OUTPUTS = (_EXTRACTIONS_FILE, "docs/failures.jsonl", "docs/chunks.parquet", "docs/fts.duckdb")

_THROUGH_EXTRACTION = (run_documents, run_index, run_event_types, run_extraction)
_FROM_RESOLVE = (run_resolve, run_dedupe, run_graph)


class Start(Enum):
    """Where the unstructured build begins."""

    BEGINNING = "beginning"
    # Debug shortcut: reuse the saved documents, index, event types and extractions; begin at resolution.
    AFTER_EXTRACT = "after_extract"


def build_unstructured(
    settings: Settings, llm: LLM, embedder: Embedder, extract_llm: LLM | None = None, start: Start = Start.BEGINNING,
) -> dict:
    """Build the document index and knowledge graph from the structured artifacts; returns counts only."""
    pc = make_context(settings, llm, embedder, extract_llm)
    phases = _FROM_RESOLVE
    if start is Start.BEGINNING:
        phases = _THROUGH_EXTRACTION + _FROM_RESOLVE
    else:
        _check_saved_extraction(artifacts_dir(settings))

    counts: dict = {}
    for run in phases:
        counts = run(pc)

    # run_graph, the last phase, reports the counts of the whole build.
    return counts


def _check_saved_extraction(out_dir: Path) -> None:
    """Raise unless an earlier build left a finished extraction that its inputs still match, for Start.AFTER_EXTRACT."""
    missing = [name for name in (*_SAVED_INPUTS, *_SAVED_OUTPUTS) if not (out_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"--after-extract needs a finished extraction; missing under {out_dir}: {', '.join(missing)}")

    saved_at = (out_dir / _EXTRACTIONS_FILE).stat().st_mtime
    newer = [name for name in _SAVED_INPUTS if (out_dir / name).stat().st_mtime > saved_at]
    if newer:
        raise ValueError(
            f"--after-extract: {', '.join(newer)} changed after {_EXTRACTIONS_FILE} was saved; run a full build-all"
        )
