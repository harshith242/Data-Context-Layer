"""Unstructured tools: graph exploration around an entity and document search limited by entity and dates."""
from pathlib import Path

from ctxlayer.common.embeddings import Embedder
from ctxlayer.unstructured.graph_store import GraphStore, ReviewFilter
from ctxlayer.unstructured.search import SearchIndex, search_docs

_MIN_DEPTH = 1
_MAX_DEPTH = 2
_MAX_EDGES = 100
_MAX_EVIDENCE = 3  # evidence items per edge returned to the agent
_UNKNOWN_ENTITY = "unknown_entity"


def graph_explore(
    store: GraphStore, entity_id: str, relation: str | None = None, depth: int = 1,
    start: str | None = None, end: str | None = None, min_confidence: float = 0.5,
    review: ReviewFilter = ReviewFilter.EXCLUDE,
) -> dict:
    """Nodes and edges within `depth` (1-2) hops of `entity_id`, flagged-for-review edges hidden by default.

    At most 100 edges are returned (`truncated` says when more exist), lowest confidence cut first.
    """
    depth = max(_MIN_DEPTH, min(depth, _MAX_DEPTH))
    found = store.subgraph(
        entity_id, depth, rel=relation, min_confidence=min_confidence, start=start, end=end, review=review,
    )
    if not found["nodes"]:
        return {"error": {"type": _UNKNOWN_ENTITY, "message": f"{entity_id!r} is not in the graph"}}

    edges = sorted(found["edges"], key=lambda edge: (-edge["confidence"], edge["src"], edge["rel"], edge["dst"]))
    # The strongest evidence items only, to keep the agent's context small; `support` still counts every source.
    kept = [{**edge, "evidence": edge.get("evidence", [])[:_MAX_EVIDENCE]} for edge in edges[:_MAX_EDGES]]
    used = {entity_id} | {end_id for edge in kept for end_id in (edge["src"], edge["dst"])}
    return {
        "entity": entity_id,
        "depth": depth,
        "nodes": [node for node in found["nodes"] if node["id"] in used],
        "edges": kept,
        "truncated": len(edges) > len(kept),
    }


def search_docs_tool(
    index_dir: Path, embedder: Embedder, store: GraphStore, query: str, entity_ids: list[str] | None = None,
    start: str | None = None, end: str | None = None, k: int = 5, rrf_k: int = 60,
    index: SearchIndex | None = None,
) -> dict:
    """`search_docs` limited to documents tagged with / mentioning the entities (and their children)."""
    if not entity_ids:
        return search_docs(index_dir, embedder, query, k, None, start, end, rrf_k, index)
    unknown = [entity_id for entity_id in entity_ids if not store.subgraph(entity_id, depth=0)["nodes"]]
    if unknown:
        return {
            "error": {
                "type": _UNKNOWN_ENTITY,
                "ids": unknown,
                "message": "Use entity ids returned by graph exploration or context.",
            }
        }
    doc_ids = store.docs_for_entities(entity_ids)
    if not doc_ids:
        return {"hits": [], "note": "No documents are tagged with or mention these entities."}
    return search_docs(index_dir, embedder, query, k, doc_ids, start, end, rrf_k, index)
