"""Document processing: text extraction, chunking, vector search, and retrieval.

Provides a ``DocumentIndex`` class that supports:
- Sliding-window chunking with sentence-boundary detection
- TF-IDF vector embeddings (zero-dependency, always available)
- Hybrid search: cosine vector similarity + BM25 keyword scoring
- Citation tracking per chunk
- Optional external embedding backend (llama.cpp ``/embedding``, sentence-transformers)
- Persistent index serialisation to JSON
"""

from __future__ import annotations

from importlib import metadata
import json
import math
import os
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


CHUNK_SIZE_CHARS = 1600  # ~400 tokens
CHUNK_OVERLAP_CHARS = 200  # ~50 tokens overlap
_MIN_SAFE_PYPDF_VERSION = (6, 10, 2)


# =========================================================================
# File utilities
# =========================================================================

def _sanitize_filename(name: str) -> str:
    """Strip path traversal and dangerous characters from a filename."""
    name = os.path.basename(name).strip()
    name = re.sub(r"[^\w\-. ]", "_", name)
    return name[:200] or "file"


def _parse_version_tuple(raw_version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in raw_version.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _require_safe_pypdf() -> None:
    try:
        installed = metadata.version("pypdf")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("PDF support requires pypdf>=6.10.2 to be installed.") from exc
    if _parse_version_tuple(installed) < _MIN_SAFE_PYPDF_VERSION:
        raise RuntimeError(
            "PDF support is disabled until pypdf>=6.10.2 is installed to address known parser DoS issues."
        )


def _extract_text_from_file(path: Path) -> str:
    """Extract plain text from a supported document file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            _require_safe_pypdf()
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            parts: list[str] = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n\n".join(parts)
        except Exception as exc:
            raise RuntimeError(f"Could not read PDF: {exc}") from exc
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise RuntimeError(f"Could not read file: {exc}") from exc


# =========================================================================
# Chunking with sentence-boundary detection
# =========================================================================

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _chunk_text(
    text: str,
    *,
    size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Sliding-window chunker with sentence boundary snapping."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))

        # Snap to nearest sentence boundary (within the last 20% of the chunk)
        if end < len(text):
            search_start = max(start, end - size // 5)
            boundaries = [m.start() for m in _SENTENCE_BOUNDARY.finditer(text, search_start, end)]
            if boundaries:
                end = boundaries[-1]

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

    return chunks


# =========================================================================
# TF-IDF Vectoriser (zero-dependency)
# =========================================================================

_TOKENIZE_RE = re.compile(r"\b\w{2,}\b")
_STOPWORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would "
    "shall should may might can could am it its this that these those i me my we "
    "us our you your he him his she her they them their what which who whom how "
    "where when why if or and but not no nor so at by for from in into of on to "
    "with as up out about after all also between than too very just because".split()
)


def _tokenize(text: str) -> list[str]:
    return [w for w in _TOKENIZE_RE.findall(text.lower()) if w not in _STOPWORDS]


class TFIDFVectoriser:
    """Simple TF-IDF vectoriser that builds an inverted index for fast retrieval."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}  # term → index
        self._idf: dict[str, float] = {}
        self._doc_vectors: list[dict[str, float]] = []  # sparse vectors
        self._n_docs: int = 0

    def fit(self, documents: list[str]) -> None:
        """Build vocabulary and IDF from a corpus of documents."""
        self._n_docs = len(documents)
        doc_freq: Counter[str] = Counter()
        all_terms: set[str] = set()

        tokenized = [_tokenize(doc) for doc in documents]

        for tokens in tokenized:
            unique = set(tokens)
            for term in unique:
                doc_freq[term] += 1
            all_terms.update(unique)

        self._vocab = {term: idx for idx, term in enumerate(sorted(all_terms))}
        self._idf = {
            term: math.log((self._n_docs + 1) / (df + 1)) + 1
            for term, df in doc_freq.items()
        }

        self._doc_vectors = []
        for tokens in tokenized:
            tf = Counter(tokens)
            vec: dict[str, float] = {}
            for term, count in tf.items():
                if term in self._idf:
                    vec[term] = (count / max(len(tokens), 1)) * self._idf[term]
            self._doc_vectors.append(vec)

    def query(self, text: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return (doc_index, similarity_score) pairs for the best matches."""
        tokens = _tokenize(text)
        tf = Counter(tokens)
        q_vec: dict[str, float] = {}
        for term, count in tf.items():
            if term in self._idf:
                q_vec[term] = (count / max(len(tokens), 1)) * self._idf[term]

        if not q_vec:
            return []

        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0:
            return []

        scores: list[tuple[int, float]] = []
        for idx, d_vec in enumerate(self._doc_vectors):
            dot = sum(q_vec.get(t, 0) * d_vec.get(t, 0) for t in q_vec)
            d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
            if d_norm > 0:
                scores.append((idx, dot / (q_norm * d_norm)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab": self._vocab,
            "idf": self._idf,
            "doc_vectors": self._doc_vectors,
            "n_docs": self._n_docs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TFIDFVectoriser":
        v = cls()
        v._vocab = data.get("vocab", {})
        v._idf = data.get("idf", {})
        v._doc_vectors = data.get("doc_vectors", [])
        v._n_docs = data.get("n_docs", 0)
        return v


# =========================================================================
# BM25 keyword scorer
# =========================================================================

class BM25Scorer:
    """BM25 keyword scorer for hybrid retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_dl: float = 0
        self._n_docs: int = 0

    def fit(self, documents: list[str]) -> None:
        self._doc_tokens = [_tokenize(doc) for doc in documents]
        self._n_docs = len(documents)
        total_len = 0
        self._doc_freq = Counter()
        for tokens in self._doc_tokens:
            total_len += len(tokens)
            for term in set(tokens):
                self._doc_freq[term] += 1
        self._avg_dl = total_len / max(self._n_docs, 1)

    def query(self, text: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_tokens = _tokenize(text)
        if not q_tokens:
            return []

        scores: list[tuple[int, float]] = []
        for idx, doc_tokens in enumerate(self._doc_tokens):
            score = 0.0
            dl = len(doc_tokens)
            tf_map = Counter(doc_tokens)
            for qt in q_tokens:
                if qt not in self._doc_freq:
                    continue
                tf = tf_map.get(qt, 0)
                df = self._doc_freq[qt]
                idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1)))
                score += idf * tf_norm
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# =========================================================================
# Citation tracking
# =========================================================================

class Citation:
    """Tracks source location for a retrieved chunk."""

    __slots__ = ("doc_id", "doc_name", "chunk_index", "page", "text_preview")

    def __init__(
        self,
        doc_id: str,
        doc_name: str,
        chunk_index: int,
        page: int | None = None,
        text_preview: str = "",
    ):
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.chunk_index = chunk_index
        self.page = page
        self.text_preview = text_preview

    def to_dict(self) -> dict[str, Any]:
        return {
            "docId": self.doc_id,
            "docName": self.doc_name,
            "chunkIndex": self.chunk_index,
            "page": self.page,
            "preview": self.text_preview[:200],
        }


# =========================================================================
# Document Index (main interface)
# =========================================================================

class DocumentIndex:
    """Hybrid vector + keyword document index with citation tracking.

    Designed to be embedded in a session or shared globally. Persists to
    a JSON file so indexes survive across app restarts.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._chunks: list[str] = []
        self._citations: list[Citation] = []
        self._vectoriser = TFIDFVectoriser()
        self._bm25 = BM25Scorer()
        self._fitted = False
        self._persist_path = persist_path

        if persist_path and persist_path.exists():
            self._load(persist_path)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def add_document(
        self,
        text: str,
        doc_id: str | None = None,
        doc_name: str = "document",
    ) -> int:
        """Add a document to the index. Returns number of chunks created."""
        if not text.strip():
            return 0

        doc_id = doc_id or f"doc-{uuid.uuid4().hex[:12]}"
        chunks = _chunk_text(text)

        for i, chunk in enumerate(chunks):
            self._chunks.append(chunk)
            self._citations.append(Citation(
                doc_id=doc_id,
                doc_name=doc_name,
                chunk_index=i,
                text_preview=chunk[:200],
            ))

        # Rebuild index with all chunks
        self._vectoriser.fit(self._chunks)
        self._bm25.fit(self._chunks)
        self._fitted = True

        if self._persist_path:
            self._save()

        return len(chunks)

    def remove_document(self, doc_id: str) -> int:
        """Remove all chunks for a document. Returns number removed."""
        indices_to_remove = {
            i for i, c in enumerate(self._citations) if c.doc_id == doc_id
        }
        if not indices_to_remove:
            return 0

        self._chunks = [c for i, c in enumerate(self._chunks) if i not in indices_to_remove]
        self._citations = [c for i, c in enumerate(self._citations) if i not in indices_to_remove]

        if self._chunks:
            self._vectoriser.fit(self._chunks)
            self._bm25.fit(self._chunks)
            self._fitted = True
        else:
            self._vectoriser = TFIDFVectoriser()
            self._bm25 = BM25Scorer()
            self._fitted = False

        if self._persist_path:
            self._save()

        return len(indices_to_remove)

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector similarity and BM25 keyword matching.

        Returns list of ``{"text": str, "citation": dict, "score": float}`` dicts.
        """
        if not self._fitted or not self._chunks:
            return []

        # Get scores from both methods
        vec_results = self._vectoriser.query(query, top_k=top_k * 2)
        bm25_results = self._bm25.query(query, top_k=top_k * 2)

        # Normalise scores to [0, 1]
        vec_scores: dict[int, float] = {}
        if vec_results:
            max_vec = max(s for _, s in vec_results) or 1
            vec_scores = {idx: s / max_vec for idx, s in vec_results}

        bm25_scores: dict[int, float] = {}
        if bm25_results:
            max_bm25 = max(s for _, s in bm25_results) or 1
            bm25_scores = {idx: s / max_bm25 for idx, s in bm25_results}

        # Merge with weighted combination
        all_indices = set(vec_scores.keys()) | set(bm25_scores.keys())
        combined: list[tuple[int, float]] = []
        for idx in all_indices:
            score = (
                vector_weight * vec_scores.get(idx, 0)
                + bm25_weight * bm25_scores.get(idx, 0)
            )
            combined.append((idx, score))

        combined.sort(key=lambda x: x[1], reverse=True)

        results: list[dict[str, Any]] = []
        for idx, score in combined[:top_k]:
            results.append({
                "text": self._chunks[idx],
                "citation": self._citations[idx].to_dict(),
                "score": round(score, 4),
            })

        return results

    def _save(self) -> None:
        if self._persist_path is None:
            return
        data = {
            "chunks": self._chunks,
            "citations": [c.to_dict() for c in self._citations],
        }
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._chunks = data.get("chunks", [])
            self._citations = [
                Citation(
                    doc_id=c.get("docId", ""),
                    doc_name=c.get("docName", ""),
                    chunk_index=c.get("chunkIndex", 0),
                    page=c.get("page"),
                    text_preview=c.get("preview", ""),
                )
                for c in data.get("citations", [])
            ]
            if self._chunks:
                self._vectoriser.fit(self._chunks)
                self._bm25.fit(self._chunks)
                self._fitted = True
        except (json.JSONDecodeError, OSError):
            pass


# =========================================================================
# Backward compatibility — keep old functions available for state.py
# =========================================================================

def _retrieve_relevant_chunks(prompt: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Legacy keyword-only retrieval. Used by state.py's _retrieve_session_context."""
    if not chunks:
        return []

    # Build a temporary index for this session's chunks
    index = DocumentIndex()
    for chunk in chunks:
        text = chunk.get("text", "")
        source = chunk.get("source", "doc")
        if text:
            index._chunks.append(text)
            index._citations.append(Citation(
                doc_id=source,
                doc_name=source,
                chunk_index=len(index._chunks) - 1,
            ))

    if index._chunks:
        index._vectoriser.fit(index._chunks)
        index._bm25.fit(index._chunks)
        index._fitted = True

    results = index.search(prompt, top_k=top_k)
    return [{"text": r["text"], "source": r["citation"]["docName"]} for r in results]
