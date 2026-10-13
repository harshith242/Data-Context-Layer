"""Hybrid document search: vector + BM25 rankings fused by reciprocal rank."""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ctxlayer.common.embeddings import Embedder, cosine_top_k
from ctxlayer.unstructured._names import CHUNKS_FILE, FTS_FILE, FTS_TABLE

_CANDIDATES = 50  # per ranking, before fusion
_SNIPPET_CHARS = 300


class SearchIndex:
    """Chunks, embeddings and FTS connection reused for a serving context."""

    def __init__(self, index_dir: Path):
        self.chunks = pd.read_parquet(index_dir / CHUNKS_FILE)
        self.matrix = np.array(self.chunks["embedding"].tolist(), dtype=np.float32) if not self.chunks.empty else None
        self.positions = {chunk_id: position for position, chunk_id in enumerate(self.chunks["chunk_id"])}
        self.con = duckdb.connect(str(index_dir / FTS_FILE), read_only=True)
        self.con.execute("LOAD fts")


def search_docs(index_dir: Path, embedder: Embedder, query: str, k: int = 5,
                doc_ids: set[str] | None = None, start: str | None = None, end: str | None = None,
                rrf_k: int = 60, index: SearchIndex | None = None) -> dict:
    """{hits: [{doc_id, chunk_id, date, title, snippet, score}], coverage: [first, last], note?, _citations}.

    Each ranking's top-50 is fused by score = sum 1/(rrf_k + rank); a document keeps its best chunk.
    A range outside the documents' dates returns no hits and a note naming the coverage.
    """
    chunks = index.chunks if index else pd.read_parquet(index_dir / CHUNKS_FILE)
    if chunks.empty:
        return {"hits": [], "coverage": [None, None], "_citations": []}

    first, last = chunks["date"].min(), chunks["date"].max()
    result = {"hits": [], "coverage": [first, last], "_citations": []}
    if (start and start > last) or (end and end < first):
        result["note"] = f"No documents in the requested range; documents cover {first} to {last}."
        return result

    allowed = pd.Series(True, index=chunks.index)
    if doc_ids is not None:
        allowed &= chunks["doc_id"].isin(doc_ids)
    if start:
        allowed &= chunks["date"] >= start
    if end:
        allowed &= chunks["date"] <= end
    candidates = chunks[allowed].reset_index(drop=True)

    scores: dict[str, float] = {}
    for ranking in (
        _vector_ranking(candidates, embedder, query, index),
        _bm25_ranking(index_dir, candidates, query, index),
    ):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    by_id = candidates.set_index("chunk_id")
    seen_docs = set()
    for chunk_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        row = by_id.loc[chunk_id]
        if row["doc_id"] in seen_docs:
            continue

        seen_docs.add(row["doc_id"])
        result["hits"].append(
            {
                "doc_id": row["doc_id"],
                "chunk_id": chunk_id,
                "date": row["date"],
                "title": row["title"],
                "snippet": row["text"][:_SNIPPET_CHARS],
                "score": score,
            }
        )
        result["_citations"].append({"kind": "doc", "doc_id": row["doc_id"], "chunk_id": chunk_id})
        if len(result["hits"]) == k:
            break
    return result


def _vector_ranking(
    candidates: pd.DataFrame, embedder: Embedder, query: str, index: SearchIndex | None = None,
) -> list[str]:
    if candidates.empty:
        return []
    matrix = (
        index.matrix[[index.positions[chunk_id] for chunk_id in candidates["chunk_id"]]]
        if index else np.array(candidates["embedding"].tolist(), dtype=np.float32)
    )
    query_vector = embedder.embed_queries([query])[0]
    return [candidates["chunk_id"][position] for position, _ in cosine_top_k(query_vector, matrix, _CANDIDATES)]


def _bm25_ranking(index_dir: Path, candidates: pd.DataFrame, query: str, index: SearchIndex | None = None) -> list[str]:
    """Chunk ids by BM25 score, best first; chunks sharing no term with the query are absent."""
    if index:
        rows = index.con.execute(
            f"SELECT chunk_id FROM (SELECT chunk_id, fts_main_{FTS_TABLE}.match_bm25(chunk_id, ?) AS score "
            f"FROM {FTS_TABLE}) WHERE score IS NOT NULL ORDER BY score DESC, chunk_id",
            [query],
        ).fetchall()
    else:
        con = duckdb.connect(str(index_dir / FTS_FILE), read_only=True)
        try:
            con.execute("LOAD fts")
            rows = con.execute(
                f"SELECT chunk_id FROM (SELECT chunk_id, fts_main_{FTS_TABLE}.match_bm25(chunk_id, ?) AS score "
                f"FROM {FTS_TABLE}) WHERE score IS NOT NULL ORDER BY score DESC, chunk_id",
                [query],
            ).fetchall()
        finally:
            con.close()

    allowed = set(candidates["chunk_id"])
    return [chunk_id for (chunk_id,) in rows if chunk_id in allowed][:_CANDIDATES]
