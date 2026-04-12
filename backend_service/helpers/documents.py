"""Document processing: text extraction, chunking, and retrieval."""
from __future__ import annotations

import os
import re
from pathlib import Path


CHUNK_SIZE_CHARS = 2000  # Approximately 500 tokens
CHUNK_OVERLAP_CHARS = 400


def _sanitize_filename(name: str) -> str:
    """Strip path traversal and dangerous characters from a filename."""
    name = os.path.basename(name).strip()
    name = re.sub(r"[^\w\-. ]", "_", name)
    return name[:200] or "file"


def _extract_text_from_file(path: Path) -> str:
    """Extract plain text from a supported document file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n\n".join(parts)
        except Exception as exc:
            raise RuntimeError(f"Could not read PDF: {exc}") from exc
    # Plain text files (including code)
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise RuntimeError(f"Could not read file: {exc}") from exc


def _chunk_text(text: str, *, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Sliding-window chunker. Returns list of overlapping text chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _retrieve_relevant_chunks(prompt: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Score chunks by keyword overlap with the prompt and return top K."""
    if not chunks:
        return []
    prompt_terms = set(re.findall(r"\w+", prompt.lower()))
    if not prompt_terms:
        return chunks[:top_k]

    scored = []
    for chunk in chunks:
        text = chunk.get("text", "")
        chunk_terms = set(re.findall(r"\w+", text.lower()))
        if not chunk_terms:
            continue
        overlap = len(prompt_terms & chunk_terms)
        if overlap > 0:
            scored.append((overlap, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]
