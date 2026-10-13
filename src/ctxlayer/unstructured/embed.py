"""Document index: chunk embeddings in parquet and a DuckDB full-text (BM25) index."""
from pathlib import Path

import duckdb
import pandas as pd

from ctxlayer.common.embeddings import Embedder
from ctxlayer.common.files import atomic_file
from ctxlayer.unstructured._names import CHUNKS_FILE, FTS_FILE, FTS_TABLE, HEAD_KIND
from ctxlayer.unstructured.chunk import Chunk

_COLUMNS = ["chunk_id", "doc_id", "date", "kind", "text"]


def index_chunks(chunks: list[Chunk], embedder: Embedder, out_dir: Path) -> dict:
    """Write chunks.parquet (+ title, embedding) and fts.duckdb into `out_dir`; returns counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [[chunk.chunk_id, chunk.doc_id, chunk.date, chunk.kind, chunk.text] for chunk in chunks],
        columns=_COLUMNS,
    )
    # A head chunk reads "title\nsummary": its first line is the document title.
    head_titles = {chunk.doc_id: chunk.text.split("\n", 1)[0] for chunk in chunks if chunk.kind == HEAD_KIND}
    frame["title"] = frame["doc_id"].map(head_titles)

    old_path = out_dir / CHUNKS_FILE
    old_embeddings = {}
    fingerprint = embedder.cache_fingerprint
    old = _read_old(old_path)
    if old is not None:
        old_embeddings = {
            (row.chunk_id, row.text): row.embedding
            for row in old.itertuples(index=False)
            if getattr(row, "embedding_fingerprint", None) == fingerprint
        }
    missing = [index for index, row in frame.iterrows() if (row.chunk_id, row.text) not in old_embeddings]
    new_embeddings = iter(embedder.embed_passages(frame.loc[missing, "text"].tolist())) if missing else iter(())
    frame["embedding"] = [
        old_embeddings[(row.chunk_id, row.text)] if (row.chunk_id, row.text) in old_embeddings else next(new_embeddings)
        for row in frame.itertuples(index=False)
    ]
    frame["embedding_fingerprint"] = fingerprint

    with atomic_file(out_dir / CHUNKS_FILE) as temp:
        frame.to_parquet(temp)
    _write_fts(frame[_COLUMNS], out_dir / FTS_FILE)
    return {"chunks": len(frame), "documents": int(frame["doc_id"].nunique())}


def _read_old(path: Path) -> pd.DataFrame | None:
    """The saved chunks, or None when missing or unreadable (e.g. cut short by Ctrl+C): nothing to reuse."""
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return None


def _write_fts(frame: pd.DataFrame, path: Path) -> None:
    # Rebuilt from scratch: the index is a function of the chunks.
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.execute("INSTALL fts")
        con.execute("LOAD fts")
        con.register("chunk_frame", frame)
        con.execute(f"CREATE TABLE {FTS_TABLE} AS SELECT * FROM chunk_frame")
        con.execute(f"PRAGMA create_fts_index('{FTS_TABLE}', 'chunk_id', 'text')")
    finally:
        con.close()
