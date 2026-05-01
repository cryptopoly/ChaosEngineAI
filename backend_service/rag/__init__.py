"""Cross-platform RAG primitives — Phase 2.6.

Two collaborators replace (or augment) the existing TF-IDF + BM25
retrieval that lives in `helpers/documents.py`:

  * `embedding_client` — subprocess wrapper around the llama.cpp
    `llama-embedding` CLI. Returns dense vectors for arbitrary text.
    Cross-platform because llama.cpp ships binaries for macOS, Linux,
    and Windows; same wire format on every host.

  * `vector_store` — numpy cosine-similarity index. No new dep
    (numpy is already part of the chat runtime). Persistable as a
    JSON blob alongside session documents.

The integration in `helpers/documents.DocumentIndex` is opt-in: when
the embedding client reports availability (model + binary present),
search ranks chunks by cosine similarity over embeddings, falls
back to the existing TF-IDF + BM25 hybrid when the embedding path
errors out at runtime. Either way the public `search()` shape stays
identical so call sites (state.py `_retrieve_session_context`)
don't change.
"""

from backend_service.rag.embedding_client import (
    EmbeddingClient,
    EmbeddingClientUnavailable,
    resolve_embedding_client,
)
from backend_service.rag.vector_store import VectorStore

__all__ = [
    "EmbeddingClient",
    "EmbeddingClientUnavailable",
    "VectorStore",
    "resolve_embedding_client",
]
