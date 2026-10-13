"""Tests for cosine_top_k, FakeEmbedder and (slow) the real FastEmbedder."""
import numpy as np
import pytest

from ctxlayer.common.embeddings import FastEmbedder, cosine_top_k

_PASSAGES = [
    "the cat sat on the warm mat",
    "stock markets fell sharply on monday",
    "a recipe for roasted tomato soup",
]










def test_cosine_top_k_orders_by_score_and_returns_scores():
    matrix = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    query = np.array([0.0, 1.0], dtype=np.float32)

    result = cosine_top_k(query, matrix, k=2)

    assert [index for index, _ in result] == [1, 2]
    assert [score for _, score in result] == pytest.approx([1.0, 0.8])










@pytest.mark.slow
def test_fast_embedder_retrieves_matching_passage():
    embedder = FastEmbedder("Qwen/Qwen3-Embedding-0.6B-Q", query_prefix="", batch_size=4)
    queries = ["how to cook tomato soup", "why did equity prices drop", "where did the cat sleep"]

    passages = embedder.embed_passages(_PASSAGES)
    found = embedder.embed_queries(queries)

    assert passages.shape == (3, embedder.dim)
    assert np.allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-4)
    assert [cosine_top_k(q, passages, k=1)[0][0] for q in found] == [2, 1, 0]
