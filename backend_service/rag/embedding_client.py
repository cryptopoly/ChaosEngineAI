"""Subprocess wrapper around `llama-embedding` for cross-platform RAG.

Phase 2.6: takes a string, returns a normalised dense vector. Detects
the binary via env var override or PATH. Detects the model via env var
or a per-data-dir convention (`<dataDir>/embeddings/*.gguf`). When
either is missing, every method raises `EmbeddingClientUnavailable`
and the caller falls back to the existing TF-IDF + BM25 path —
behaviour preserves a graceful degradation rather than refusing
generations when no embedding model is shipped.

The CLI is invoked with `--embd-output-format json` so we don't have
to parse the human-readable text dump. JSON output looks like:

    {"object": "list", "data": [{"index": 0, "embedding": [...]}], ...}

Embeddings are L2-normalised (`--embd-normalize 2`) so cosine
similarity is the same as dot product downstream.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


CHAOSENGINE_LLAMA_EMBEDDING_BIN = "CHAOSENGINE_LLAMA_EMBEDDING"
CHAOSENGINE_EMBEDDING_MODEL = "CHAOSENGINE_EMBEDDING_MODEL"

# Default subprocess deadline. Embedding a single chunk on CPU should
# return within a couple of seconds; the ceiling exists to prevent a
# wedged binary from hanging the chat send path.
DEFAULT_TIMEOUT_S = 30.0


class EmbeddingClientUnavailable(RuntimeError):
    """Raised when the binary or model is missing.

    Callers treat this as "use the keyword fallback" — it must not
    surface as a chat error.
    """


@dataclass(frozen=True)
class EmbeddingClient:
    """Concrete client. Constructed via `resolve_embedding_client`."""

    binary: str
    model_path: str
    timeout: float = DEFAULT_TIMEOUT_S

    def is_available(self) -> bool:
        return Path(self.binary).is_file() and Path(self.model_path).is_file()

    def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a normalised float vector."""
        vectors = self.embed_batch([text])
        return vectors[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings via repeated CLI calls.

        The llama-embedding CLI accepts a single `--prompt` per
        invocation (`--prompt-file` for batch is also supported but the
        format is awkward to thread through). For chunk counts the
        chat path actually sees (typically <50 per session), the
        per-call overhead is acceptable. Switch to `--prompt-file`
        if profiling shows this is hot.
        """
        if not texts:
            return []
        if not self.is_available():
            raise EmbeddingClientUnavailable(
                f"Embedding binary or model missing (binary={self.binary}, model={self.model_path})"
            )
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(self._embed_one(text))
        return vectors

    def _embed_one(self, text: str) -> list[float]:
        # `llama-embedding` only accepts text via stdin or file; passing
        # via `--prompt` works for short strings but trips on shell
        # quoting + newlines. Use stdin.
        cmd = [
            self.binary,
            "-m", self.model_path,
            "--embd-output-format", "json",
            "--embd-normalize", "2",
            "-f", "/dev/stdin",
            "--no-warmup",
            "--log-disable",
        ]
        try:
            result = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise EmbeddingClientUnavailable(
                f"llama-embedding timed out after {self.timeout:.0f}s"
            ) from exc
        except FileNotFoundError as exc:
            raise EmbeddingClientUnavailable(
                f"llama-embedding binary not found: {self.binary}"
            ) from exc

        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip()[-500:]
            raise EmbeddingClientUnavailable(
                f"llama-embedding failed (rc={result.returncode}): {stderr_tail}"
            )

        return parse_embedding_output(result.stdout)


def parse_embedding_output(stdout: str) -> list[float]:
    """Pure helper for tests — extracts the first vector from the JSON.

    The JSON envelope has shape ``{"data": [{"embedding": [...]}, ...]}``
    when ``--embd-output-format json`` is used. We always submit a
    single prompt so we always want the first entry's vector.
    """
    if not stdout.strip():
        raise EmbeddingClientUnavailable("llama-embedding returned empty stdout")
    # Some llama.cpp builds prefix the JSON with metadata lines on
    # stderr-merged stdout; find the first '{' and parse from there.
    start = stdout.find("{")
    if start < 0:
        raise EmbeddingClientUnavailable("llama-embedding output had no JSON object")
    try:
        payload = json.loads(stdout[start:])
    except json.JSONDecodeError as exc:
        raise EmbeddingClientUnavailable(
            f"llama-embedding output unparseable: {exc}"
        ) from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        raise EmbeddingClientUnavailable("llama-embedding output had no 'data' list")
    first = data[0]
    if not isinstance(first, dict):
        raise EmbeddingClientUnavailable("llama-embedding output 'data[0]' was not an object")
    embedding = first.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise EmbeddingClientUnavailable("llama-embedding output had no 'embedding' vector")
    if not all(isinstance(v, (int, float)) for v in embedding):
        raise EmbeddingClientUnavailable("llama-embedding output embedding had non-numeric values")
    return [float(v) for v in embedding]


def _resolve_binary() -> str | None:
    override = os.environ.get(CHAOSENGINE_LLAMA_EMBEDDING_BIN)
    if override and Path(override).is_file():
        return override
    found = shutil.which("llama-embedding")
    return found


def _resolve_model(data_dir: Path | None) -> str | None:
    override = os.environ.get(CHAOSENGINE_EMBEDDING_MODEL)
    if override and Path(override).is_file():
        return override
    if data_dir is not None:
        candidate_dir = data_dir / "embeddings"
        if candidate_dir.is_dir():
            ggufs = sorted(candidate_dir.glob("*.gguf"))
            if ggufs:
                return str(ggufs[0])
    return None


def resolve_embedding_client(
    data_dir: Path | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> EmbeddingClient | None:
    """Best-effort discovery — returns an EmbeddingClient or None.

    None means "no embedding path is available right now"; callers
    should fall back to the keyword/TF-IDF retrieval. Callers that
    cache the result MUST tolerate the result flipping to non-None
    after the user drops a model into `<dataDir>/embeddings/`.
    """
    binary = _resolve_binary()
    if binary is None:
        return None
    model = _resolve_model(data_dir)
    if model is None:
        return None
    return EmbeddingClient(binary=binary, model_path=model, timeout=timeout)


def warm_test(client: EmbeddingClient) -> tuple[bool, str | None]:
    """Best-effort embedding round-trip — used in diagnostics.

    Returns (ok, error_message). Never raises; callers can render the
    result on a Setup tab without try/except.
    """
    started = time.perf_counter()
    try:
        vec = client.embed("ping")
    except EmbeddingClientUnavailable as exc:
        return False, str(exc)
    if not vec:
        return False, "embedding returned empty vector"
    elapsed = time.perf_counter() - started
    return True, f"OK ({len(vec)}-dim, {elapsed:.2f}s)"
