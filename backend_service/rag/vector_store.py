"""In-memory cosine-similarity vector store for Phase 2.6 RAG.

Tiny by design — no external dep beyond numpy (already in the chat
runtime). Stores per-chunk embeddings + a parallel list of citation
metadata. Persists as a JSON blob the existing DocumentIndex storage
can hold alongside its TF-IDF state.

Embeddings are assumed to be L2-normalised at insert time (the
`llama-embedding --embd-normalize 2` flag the EmbeddingClient sets
guarantees this). With normalised vectors, cosine similarity =
dot product = a single matmul — fast enough for thousands of chunks
without an ANN index.
"""

from __future__ import annotations

import math
from typing import Any


class VectorStore:
    """Append + search over normalised dense vectors.

    The store keeps embeddings in a 2-D list of floats rather than a
    numpy array on disk; numpy comes back into play only at query
    time so the JSON serialisation stays portable across Python
    versions / numpy upgrades.
    """

    def __init__(self) -> None:
        self._vectors: list[list[float]] = []
        self._dim: int | None = None

    @property
    def size(self) -> int:
        return len(self._vectors)

    @property
    def dim(self) -> int | None:
        return self._dim

    def add(self, vector: list[float]) -> None:
        if not vector:
            raise ValueError("VectorStore.add received an empty vector")
        if self._dim is None:
            self._dim = len(vector)
        elif len(vector) != self._dim:
            raise ValueError(
                f"VectorStore vector length mismatch: got {len(vector)}, store dim is {self._dim}"
            )
        self._vectors.append(list(vector))

    def add_batch(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            self.add(vector)

    def reset(self) -> None:
        self._vectors = []
        self._dim = None

    def remove_indices(self, indices: set[int]) -> None:
        """Drop vectors at the given positions. Renumbers the rest.

        Used when DocumentIndex.remove_document needs to drop a
        document's chunks — both the chunk list and the vector list
        must stay in lockstep.
        """
        if not indices:
            return
        self._vectors = [v for i, v in enumerate(self._vectors) if i not in indices]
        if not self._vectors:
            self._dim = None

    def search(self, query: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        """Return (index, similarity) pairs for the top-k matches.

        Both the stored vectors and the query are assumed normalised
        (L2 = 1). When that holds, dot product equals cosine
        similarity. The function still falls back to the explicit
        normalisation form if the assumption is violated, so it
        works even on hand-built test fixtures.
        """
        if not self._vectors or not query:
            return []
        if self._dim is not None and len(query) != self._dim:
            raise ValueError(
                f"VectorStore.search query dim {len(query)} does not match store dim {self._dim}"
            )

        query_norm = math.sqrt(sum(q * q for q in query))
        if query_norm == 0:
            return []

        scores: list[tuple[int, float]] = []
        for idx, vec in enumerate(self._vectors):
            dot = sum(q * v for q, v in zip(query, vec))
            vec_norm = math.sqrt(sum(v * v for v in vec))
            if vec_norm == 0:
                continue
            similarity = dot / (query_norm * vec_norm)
            scores.append((idx, similarity))
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {"vectors": self._vectors, "dim": self._dim}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VectorStore":
        store = cls()
        vectors = payload.get("vectors") if isinstance(payload, dict) else None
        if isinstance(vectors, list):
            for vector in vectors:
                if isinstance(vector, list) and vector and all(isinstance(v, (int, float)) for v in vector):
                    store.add([float(v) for v in vector])
        return store
