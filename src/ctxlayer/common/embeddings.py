"""Local text embeddings (fastembed) and cosine top-k search."""
from __future__ import annotations

import hashlib
import json
from typing import Protocol

import urllib.error
import urllib.request
import numpy as np

from ctxlayer.common.config import Settings


class Embedder(Protocol):
    dim: int

    @property
    def cache_fingerprint(self) -> str: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    """fastembed model, loaded on first use; `query_prefix` is prepended to queries."""

    def __init__(self, model: str, query_prefix: str, batch_size: int):
        self._model_name = model
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._model = None

    @property
    def dim(self) -> int:
        return self._load().embedding_size

    @property
    def cache_fingerprint(self) -> str:
        config = {
            "model": self._model_name,
            "query_prefix": self._query_prefix,
            "batch_size": self._batch_size,
            "provider": "fastembed",
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return self._empty()
        vectors = self._load().passage_embed(texts, batch_size=self._batch_size)
        return _normalise(np.stack(list(vectors)))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return self._empty()
        prefixed = [self._query_prefix + text for text in texts]
        vectors = self._load().query_embed(prefixed, batch_size=self._batch_size)
        return _normalise(np.stack(list(vectors)))

    def _empty(self) -> np.ndarray:
        return np.zeros((0, self.dim), dtype=np.float32)

    def _load(self):
        # Deferred: importing fastembed and loading the weights takes seconds.
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self._model_name)
        return self._model



class OllamaEmbedder:
    """Ollama-backed text embeddings using its local HTTP API."""

    def __init__(
        self,
        model: str,
        query_prefix: str,
        batch_size: int,
        base_url: str,
    ):
        if batch_size < 1:
            raise ValueError("Embedding batch_size must be positive")

        self._model_name = model
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._base_url = base_url.rstrip("/")
        self._dim = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._embed(["dimension probe"])
        return self._dim

    @property
    def cache_fingerprint(self) -> str:
        config = {
            "provider": "ollama",
            "model": self._model_name,
            "query_prefix": self._query_prefix,
            "batch_size": self._batch_size,
            "base_url": self._base_url,
        }
        return hashlib.sha256(
            json.dumps(
                config, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return self._empty()
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return self._empty()

        prefixed = [self._query_prefix + text for text in texts]
        return self._embed(prefixed)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]

            body = json.dumps({
                "model": self._model_name,
                "input": batch,
                "keep_alive": "5m",
            }).encode("utf-8")

            request = urllib.request.Request(
                f"{self._base_url}/api/embed",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(
                    request, timeout=300
                ) as response:
                    result = json.loads(
                        response.read().decode("utf-8")
                    )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(
                    "utf-8", errors="replace"
                )
                raise RuntimeError(
                    f"Ollama embedding failed (HTTP {exc.code}): "
                    f"{detail}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"Cannot reach Ollama at {self._base_url}. "
                    "Check that Ollama is running."
                ) from exc

            batch_vectors = result.get("embeddings")
            if not isinstance(batch_vectors, list):
                raise RuntimeError(
                    "Ollama response did not contain 'embeddings'"
                )

            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(batch_vectors)} vectors "
                    f"for {len(batch)} texts"
                )

            vectors.extend(batch_vectors)

        matrix = np.asarray(vectors, dtype=np.float32)

        if matrix.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D embedding matrix, got {matrix.shape}"
            )

        if self._dim is None:
            self._dim = int(matrix.shape[1])
        elif matrix.shape[1] != self._dim:
            raise RuntimeError(
                f"Embedding dimension changed from {self._dim} "
                f"to {matrix.shape[1]}"
            )

        return _normalise(matrix)

    def _empty(self) -> np.ndarray:
        return np.zeros((0, self.dim), dtype=np.float32)



def get_embedder(settings: Settings) -> Embedder:
    provider = settings.get("embeddings.provider")

    model = settings.get("embeddings.model")
    query_prefix = settings.get("embeddings.query_prefix")
    batch_size = settings.get("embeddings.batch_size")

    if provider == "fastembed":
        return FastEmbedder(
            model=model,
            query_prefix=query_prefix,
            batch_size=batch_size,
        )

    if provider == "ollama":
        return OllamaEmbedder(
            model=model,
            query_prefix=query_prefix,
            batch_size=batch_size,
            base_url=settings.get("embeddings.base_url"),
        )

    raise ValueError(
        f"Unsupported embeddings provider: {provider!r}. "
        "Use 'fastembed' or 'ollama'."
    )



def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-k (row index, cosine) of L2-normalised `matrix` rows vs `query`, best first."""
    scores = matrix @ query
    # Stable sort keeps the lower row index first on ties.
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(index), float(scores[index])) for index in order]


def _normalise(matrix: np.ndarray) -> np.ndarray:
    # Unit rows make the dot product a cosine; zero rows stay zero.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32)
