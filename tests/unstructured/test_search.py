"""Hybrid search over the chunk index: BM25, vector, fusion and filters."""
import pytest
from fakes import FakeEmbedder
from fixtures.domain_a.make_domain import PLANTED_BULLETIN_REF, PLANTED_QUOTE

from ctxlayer.common.config import artifacts_dir
from ctxlayer.common.db import connect_readonly
from ctxlayer.common.files import read_yaml
from ctxlayer.unstructured.chunk import Chunk, chunk_documents
from ctxlayer.unstructured.documents import build_documents, doc_window
from ctxlayer.unstructured.embed import index_chunks
from ctxlayer.unstructured.search import search_docs

_SECOND_PLANTED_REF = "FB-20250427-01"


@pytest.fixture(scope="module")
def bulletin_index(tmp_path_factory, domain_a, domain_a_built):
    roles = read_yaml(artifacts_dir(domain_a_built) / "roles.yaml")
    con = connect_readonly(domain_a.db_path)
    docs = build_documents(con, roles)
    con.close()

    out_dir = tmp_path_factory.mktemp("doc_index")
    counts = index_chunks(chunk_documents(docs, tokens=500, overlap=60), FakeEmbedder(), out_dir)
    return out_dir, docs, counts


def _tiny_index(tmp_path, texts: list[str]):
    chunks = [Chunk(f"d{n}#1", f"d{n}", "2025-01-01", text, "body") for n, text in enumerate(texts)]
    index_chunks(chunks, FakeEmbedder(), tmp_path)
    return tmp_path


def test_planted_quote_finds_its_bulletin_first(bulletin_index):
    index_dir, _, _ = bulletin_index

    result = search_docs(index_dir, FakeEmbedder(), PLANTED_QUOTE, k=3)

    top = result["hits"][0]
    assert top["doc_id"] == PLANTED_BULLETIN_REF
    assert top["title"] == "Drought cuts Great Plains wheat supply"
    assert top["date"] == "2025-04-25"
    assert result["_citations"][0] == {"kind": "doc", "doc_id": PLANTED_BULLETIN_REF, "chunk_id": top["chunk_id"]}
    assert len({hit["doc_id"] for hit in result["hits"]}) == len(result["hits"])


def test_keyword_match_without_vector_signal_is_found_by_bm25(tmp_path):
    # "planting" stems to the same term as "planted" in BM25; the hashed embedder sees two different words.
    index_dir = _tiny_index(tmp_path, ["alpha beta gamma", "delta epsilon zeta", "eta theta planted"])

    hits = search_docs(index_dir, FakeEmbedder(), "planting", k=3)["hits"]

    assert hits[0]["doc_id"] == "d2"


def test_stopword_query_is_found_by_vector_only(tmp_path):
    # BM25 drops "were" as a stopword, so only the embedding can rank this chunk first.
    index_dir = _tiny_index(tmp_path, ["wheat prices rose", "were were were rain", "barley stocks fell"])

    hits = search_docs(index_dir, FakeEmbedder(), "were", k=3)["hits"]

    assert hits[0]["doc_id"] == "d1"


def test_date_filter_keeps_only_chunks_in_range(bulletin_index):
    index_dir, _, _ = bulletin_index

    result = search_docs(index_dir, FakeEmbedder(), "drought wheat", k=10, start="2025-04-25", end="2025-04-27")

    assert {hit["doc_id"] for hit in result["hits"]} == {PLANTED_BULLETIN_REF, _SECOND_PLANTED_REF}
    assert all("2025-04-25" <= hit["date"] <= "2025-04-27" for hit in result["hits"])


def test_range_outside_the_window_returns_note_naming_coverage(bulletin_index):
    index_dir, docs, _ = bulletin_index
    first, last = doc_window(docs)

    result = search_docs(index_dir, FakeEmbedder(), "wheat price", start="2024-01-01", end="2024-06-30")

    assert result["hits"] == []
    assert result["coverage"] == [first, last]
    assert first in result["note"] and last in result["note"]


def test_doc_ids_filter_restricts_hits(bulletin_index):
    index_dir, _, _ = bulletin_index

    result = search_docs(index_dir, FakeEmbedder(), "drought", k=10, doc_ids={_SECOND_PLANTED_REF})

    assert {hit["doc_id"] for hit in result["hits"]} == {_SECOND_PLANTED_REF}


def test_rebuild_reuses_unchanged_chunk_embeddings(tmp_path):
    class CountingEmbedder(FakeEmbedder):
        def __init__(self, fingerprint="v1"):
            super().__init__(fingerprint=fingerprint)
            self.calls = []

        def embed_passages(self, texts):
            self.calls.append(list(texts))
            return super().embed_passages(texts)

    chunks = [
        Chunk("d1#1", "d1", "2025-01-01", "first text", "body"),
        Chunk("d2#1", "d2", "2025-01-01", "second text", "body"),
    ]
    embedder = CountingEmbedder()
    index_chunks(chunks, embedder, tmp_path)

    index_chunks([chunks[0], Chunk("d2#1", "d2", "2025-01-01", "changed text", "body")], embedder, tmp_path)

    assert embedder.calls == [["first text", "second text"], ["changed text"]]


def test_rebuild_reembeds_chunks_when_embedder_fingerprint_changes(tmp_path):
    class CountingEmbedder(FakeEmbedder):
        def __init__(self, fingerprint):
            super().__init__(fingerprint=fingerprint)
            self.calls = []

        def embed_passages(self, texts):
            self.calls.append(list(texts))
            return super().embed_passages(texts)

    chunks = [Chunk("d1#1", "d1", "2025-01-01", "first text", "body")]
    first = CountingEmbedder("v1")
    second = CountingEmbedder("v2")

    index_chunks(chunks, first, tmp_path)
    index_chunks(chunks, second, tmp_path)

    assert first.calls == [["first text"]]
    assert second.calls == [["first text"]]

