"""One-click embedding-model installer + RAG readiness status.

ChaosEngineAI's RAG path (``state.documents.retrieve_session_context``)
uses semantic cosine similarity when an ``llama-embedding`` binary and an
embedding GGUF are both discoverable, and transparently falls back to
TF-IDF + BM25 lexical retrieval otherwise. Out of the box no embedding
GGUF is shipped, so retrieval silently runs in lexical mode.

This module closes that gap:

* ``GET  /api/rag/status`` — reports whether semantic retrieval is wired
  (binary present + model present) so the UI can show a "vector" vs
  "lexical" badge and offer the install.
* ``POST /api/setup/install-embedding-model`` — downloads the recommended
  embedding GGUF into ``<dataDir>/embeddings/`` so ``_resolve_model``
  picks it up on the next retrieval (no restart, no env var needed).
* ``GET  /api/setup/install-embedding-model/status`` — poll progress.

Single-job background pattern, mirroring ``setup/longlive.py``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter()


# Recommended embedding model. Q8_0 is the sweet spot for retrieval
# quality vs size (146 MB) — embeddings are quant-sensitive enough that
# the tiny K-quants degrade recall, but f16 (274 MB) is overkill for the
# cosine-similarity blend we use. Nomic Embed v1.5 is Apache-2.0, 768-dim,
# and the de-facto default embedding model for local RAG stacks.
RECOMMENDED_EMBEDDING_REPO = "nomic-ai/nomic-embed-text-v1.5-GGUF"
RECOMMENDED_EMBEDDING_FILE = "nomic-embed-text-v1.5.Q8_0.gguf"
RECOMMENDED_EMBEDDING_LABEL = "Nomic Embed Text v1.5"
RECOMMENDED_EMBEDDING_SIZE_LABEL = "146 MB"


def embeddings_dir() -> Path:
    """Directory ``_resolve_model`` globs for ``*.gguf``.

    Lives next to the documents dir under the app data root. Imported
    locally to avoid a circular import at module load (``app`` registers
    this router during startup).
    """
    from backend_service.app import DOCUMENTS_DIR  # noqa: PLC0415

    return DOCUMENTS_DIR.parent / "embeddings"


def rag_status() -> dict[str, Any]:
    """Pure status snapshot — cheap enough to poll. Never raises.

    ``mode`` is ``vector`` only when both the binary and a model resolve;
    otherwise retrieval still works via the lexical fallback, reported as
    ``lexical``. ``binaryAvailable`` is surfaced separately so the UI can
    explain that installing the model alone won't enable semantic search
    on a build without the ``llama-embedding`` binary.
    """
    from backend_service.rag.embedding_client import _resolve_binary, _resolve_model  # noqa: PLC0415

    binary = _resolve_binary()
    model = _resolve_model(embeddings_dir().parent)
    binary_ok = binary is not None
    model_ok = model is not None
    return {
        "mode": "vector" if (binary_ok and model_ok) else "lexical",
        "binaryAvailable": binary_ok,
        "binaryPath": binary,
        "modelAvailable": model_ok,
        "modelPath": model,
        "installed": model_ok,
        "recommended": {
            "repo": RECOMMENDED_EMBEDDING_REPO,
            "file": RECOMMENDED_EMBEDDING_FILE,
            "label": RECOMMENDED_EMBEDDING_LABEL,
            "sizeLabel": RECOMMENDED_EMBEDDING_SIZE_LABEL,
        },
    }


@router.get("/api/rag/status")
def get_rag_status() -> dict[str, Any]:
    return rag_status()


@dataclass
class _EmbeddingJobState:
    """In-memory status for the embedding-model download.

    Same single-job semantics as the LongLive installer: a second POST
    while running returns the running job's state; state sticks around
    after completion so a late poll sees the final outcome.
    """

    id: str = ""
    phase: str = "idle"  # idle | downloading | verifying | done | error
    message: str = ""
    percent: float = 0.0
    target_path: str | None = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase,
            "message": self.message,
            "percent": round(self.percent, 1),
            "targetPath": self.target_path,
            "error": self.error,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "done": self.done,
        }


_EMBEDDING_JOB = _EmbeddingJobState()
_EMBEDDING_LOCK = threading.Lock()


def _download_embedding_model(dest_dir: Path) -> Path:
    """Download the recommended GGUF into ``dest_dir``. Returns its path.

    Uses ``hf_hub_download`` with ``local_dir`` so the file lands directly
    where ``_resolve_model`` globs, with no symlink into the HF cache and
    no double storage. Resumes a partial download automatically.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved = hf_hub_download(
        repo_id=RECOMMENDED_EMBEDDING_REPO,
        filename=RECOMMENDED_EMBEDDING_FILE,
        local_dir=str(dest_dir),
    )
    return Path(resolved)


def _embedding_job_worker() -> None:
    job = _EMBEDDING_JOB
    try:
        job.phase = "downloading"
        job.message = f"Downloading {RECOMMENDED_EMBEDDING_LABEL} ({RECOMMENDED_EMBEDDING_SIZE_LABEL})"
        dest = embeddings_dir()
        path = _download_embedding_model(dest)

        job.phase = "verifying"
        job.message = "Verifying download"
        if not path.is_file() or path.stat().st_size < 1_000_000:
            raise RuntimeError(f"downloaded file missing or truncated: {path}")
    except Exception as exc:  # noqa: BLE001 — daemon thread has no parent to catch
        job.phase = "error"
        job.error = str(exc)
        job.message = f"Embedding model install failed: {exc}"
    else:
        job.phase = "done"
        job.percent = 100.0
        job.target_path = str(path)
        job.message = "Semantic search enabled."
    finally:
        job.finished_at = time.time()
        job.done = True


@router.post("/api/setup/install-embedding-model")
def start_install_embedding_model() -> dict[str, Any]:
    """Kick off the embedding-model download in the background.

    Returns immediately. Poll ``/api/setup/install-embedding-model/status``.
    A second call while running returns the running job's state.
    """
    with _EMBEDDING_LOCK:
        if _EMBEDDING_JOB.phase in {"downloading", "verifying"}:
            return _EMBEDDING_JOB.to_dict()

        _EMBEDDING_JOB.id = f"embedding-{int(time.time() * 1000)}"
        _EMBEDDING_JOB.phase = "downloading"
        _EMBEDDING_JOB.message = "Starting download"
        _EMBEDDING_JOB.percent = 0.0
        _EMBEDDING_JOB.target_path = None
        _EMBEDDING_JOB.error = None
        _EMBEDDING_JOB.started_at = time.time()
        _EMBEDDING_JOB.finished_at = 0.0
        _EMBEDDING_JOB.done = False

        thread = threading.Thread(
            target=_embedding_job_worker,
            name="chaosengine-embedding-install",
            daemon=True,
        )
        thread.start()

    return _EMBEDDING_JOB.to_dict()


@router.get("/api/setup/install-embedding-model/status")
def install_embedding_model_status() -> dict[str, Any]:
    return _EMBEDDING_JOB.to_dict()
