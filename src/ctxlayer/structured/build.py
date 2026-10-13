"""Structured build: runs the structured phases in order, from profile to moves."""
from ctxlayer.common.config import Settings
from ctxlayer.common.embeddings import Embedder
from ctxlayer.common.llm import LLM
from ctxlayer.common.phase_context import make_context
from ctxlayer.structured.phases import (
    run_catalog, run_moves, run_ontology, run_profile, run_roles, run_semantic, run_views,
)

_PHASES = (run_profile, run_roles, run_ontology, run_catalog, run_semantic, run_views, run_moves)


def build_structured(settings: Settings, llm: LLM, embedder: Embedder) -> dict:
    """Build every structured artifact; returns counts only.

    `embedder` is accepted for the serving resolver; its embedding cache fills on first query
    because `Resolver` has no warm-up call.
    """
    pc = make_context(settings, llm, embedder)
    counts: dict = {}
    for run in _PHASES:
        counts = run(pc)

    # run_moves, the last phase, reports the counts of the whole build.
    return counts
