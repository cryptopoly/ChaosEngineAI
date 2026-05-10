"""Document text extraction + chunking + tokenisation primitives.

Pure helpers shared by the ``DocumentIndex`` class and the session /
workspace upload paths. No dependency on the TF-IDF / BM25 vectorisers
that build on top of these — they live in ``helpers/documents``.

Extracted from ``backend_service/helpers/documents.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.documents`` so existing
imports keep working.
"""

from __future__ import annotations

import os
import re
from importlib import metadata
from pathlib import Path


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
# Tokenisation
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
