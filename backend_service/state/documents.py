"""Per-session and per-workspace document storage for ``ChaosEngineState``.

Eight helpers lifted out of ``backend_service/state/__init__.py``. They
all touch the same pieces of state — ``state._lock``, the chat-session
list, the workspace registry, the disk roots from ``app`` — so they
take the state instance as the first argument and stay free functions.

The class methods on ``ChaosEngineState`` become thin wrappers; tests
that patch the class methods keep working through the wrapper, and
external callers (``routes/sessions.py``, ``routes/workspaces.py``)
keep importing them via ``state.upload_document(...)`` etc.

Extracted as part of the v0.8.0 Phase 1a-4 refactor.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from backend_service.helpers.documents import (
    _chunk_text,
    _extract_text_from_file,
    _sanitize_filename,
)


if TYPE_CHECKING:
    from backend_service.state import ChaosEngineState


def session_docs_dir(state: ChaosEngineState, session_id: str) -> Path:
    """Return the on-disk directory for a session's uploaded documents.

    The session id is sanitised to a filesystem-safe slug so user input
    can't escape the documents tree even if the upstream session id is
    weird (we accept arbitrary strings on the create path).
    """
    import re

    from backend_service.app import DOCUMENTS_DIR

    safe_id = re.sub(r"[^\w\-]", "_", session_id)
    return DOCUMENTS_DIR / safe_id


def workspace_docs_dir(state: ChaosEngineState, workspace_id: str) -> Path:
    """Return the on-disk directory for a workspace's documents."""
    from backend_service.app import WORKSPACES_DIR

    safe_id = "".join(ch for ch in workspace_id if ch.isalnum() or ch in "-_")
    return WORKSPACES_DIR / safe_id


def list_session_documents(state: ChaosEngineState, session_id: str) -> list[dict[str, Any]]:
    with state._lock:
        session = state._ensure_session(session_id)
        return list(session.get("documents", []))


def upload_session_document(
    state: ChaosEngineState,
    session_id: str,
    original_name: str,
    raw_bytes: bytes,
) -> dict[str, Any]:
    from backend_service.app import (
        DOC_ALLOWED_EXTENSIONS,
        MAX_DOC_SIZE_BYTES,
        MAX_SESSION_DOCS_BYTES,
    )

    if len(raw_bytes) > MAX_DOC_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_DOC_SIZE_BYTES // (1024 * 1024)}MB limit.",
        )
    sanitized = _sanitize_filename(original_name)
    ext = Path(sanitized).suffix.lower()
    if ext not in DOC_ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type not supported: {ext}")

    with state._lock:
        session = state._ensure_session(session_id)
        existing = session.get("documents") or []
        current_total = sum(d.get("sizeBytes", 0) for d in existing)
        if current_total + len(raw_bytes) > MAX_SESSION_DOCS_BYTES:
            raise HTTPException(status_code=413, detail="Session document quota exceeded (200MB).")

        doc_id = f"doc-{uuid.uuid4().hex[:12]}"
        session_dir = session_docs_dir(state, session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        try:
            session_dir.chmod(0o700)
        except OSError:
            pass

        doc_path = session_dir / f"{doc_id}{ext}"
        doc_path.write_bytes(raw_bytes)
        try:
            doc_path.chmod(0o600)
        except OSError:
            pass

    try:
        text = _extract_text_from_file(doc_path)
    except RuntimeError as exc:
        doc_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = _chunk_text(text)
    chunks_path = session_dir / f"{doc_id}.chunks.json"
    chunks_path.write_text(
        json.dumps([{"index": i, "text": c} for i, c in enumerate(chunks)], indent=2),
        encoding="utf-8",
    )

    with state._lock:
        session = state._ensure_session(session_id)
        doc_meta = {
            "id": doc_id,
            "filename": doc_path.name,
            "originalName": sanitized,
            "sizeBytes": len(raw_bytes),
            "chunkCount": len(chunks),
            "uploadedAt": state._time_label(),
        }
        session.setdefault("documents", []).append(doc_meta)
        session["updatedAt"] = state._time_label()
        state.add_log(
            "chat",
            "info",
            f"Document uploaded to session {session_id}: {sanitized} ({len(chunks)} chunks)",
        )
        state._persist_sessions()
        return doc_meta


def delete_session_document(
    state: ChaosEngineState, session_id: str, doc_id: str
) -> dict[str, Any]:
    with state._lock:
        session = state._ensure_session(session_id)
        docs = session.get("documents") or []
        target = next((d for d in docs if d.get("id") == doc_id), None)
        if not target:
            raise HTTPException(status_code=404, detail="Document not found.")
        session["documents"] = [d for d in docs if d.get("id") != doc_id]
        session["updatedAt"] = state._time_label()
        session_dir = session_docs_dir(state, session_id)
        for f in session_dir.glob(f"{doc_id}*"):
            try:
                f.unlink()
            except OSError:
                pass
        state.add_log("chat", "info", f"Document removed: {target.get('originalName')}")
        state._persist_sessions()
        return {"deleted": doc_id}


def upload_workspace_document(
    state: ChaosEngineState,
    workspace_id: str,
    filename: str,
    data: bytes,
) -> dict[str, Any]:
    """Phase 3.7: ingest a document into a workspace.

    Mirrors `upload_session_document` but writes under
    `<dataDir>/workspaces/<id>/`. The chunked text JSON sits next
    to the original file so the RAG retriever can read both
    session and workspace docs through the same DocumentIndex
    helpers without bespoke logic.
    """
    from backend_service.app import (
        DOC_ALLOWED_EXTENSIONS,
        MAX_DOC_SIZE_BYTES,
        WORKSPACES_DIR,
        WORKSPACES_PATH,
    )
    from backend_service.helpers.workspaces import WorkspaceRegistry

    if len(data) > MAX_DOC_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_DOC_SIZE_BYTES // (1024 * 1024)}MB limit.",
        )
    sanitized = _sanitize_filename(filename)
    ext = Path(sanitized).suffix.lower()
    if ext not in DOC_ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type not supported: {ext}")

    registry = WorkspaceRegistry(WORKSPACES_PATH, WORKSPACES_DIR)
    workspace = registry.get(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    doc_id = f"doc-{uuid.uuid4().hex[:12]}"
    workspace_dir = workspace_docs_dir(state, workspace_id)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    doc_path = workspace_dir / f"{doc_id}{ext}"
    doc_path.write_bytes(data)
    try:
        doc_path.chmod(0o600)
    except OSError:
        pass

    try:
        text = _extract_text_from_file(doc_path)
    except RuntimeError as exc:
        doc_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = _chunk_text(text)
    chunks_path = workspace_dir / f"{doc_id}.chunks.json"
    chunks_path.write_text(
        json.dumps([{"index": i, "text": c} for i, c in enumerate(chunks)], indent=2),
        encoding="utf-8",
    )

    doc_meta = {
        "id": doc_id,
        "filename": doc_path.name,
        "originalName": sanitized,
        "sizeBytes": len(data),
        "chunkCount": len(chunks),
        "uploadedAt": state._time_label(),
    }

    # Persist on the workspace registry too so the doc list comes
    # back on subsequent /api/workspaces calls without reading the
    # filesystem again.
    existing_docs = list(workspace.get("documents") or [])
    existing_docs.append(doc_meta)
    registry.update(workspace_id, title=workspace["title"])
    # The update() call doesn't currently support documents — read
    # the entry back, mutate, save by writing the full payload.
    # Workaround: write directly via the registry's internal map.
    registry._workspaces[workspace_id]["documents"] = existing_docs
    registry._workspaces[workspace_id]["updatedAt"] = state._time_label()
    registry.save()
    state.add_log(
        "chat",
        "info",
        f"Document uploaded to workspace {workspace_id}: {sanitized} ({len(chunks)} chunks)",
    )
    return doc_meta


def delete_workspace_document(
    state: ChaosEngineState, workspace_id: str, doc_id: str
) -> dict[str, Any]:
    """Phase 3.7: remove a document from a workspace's stack."""
    from backend_service.app import WORKSPACES_DIR, WORKSPACES_PATH
    from backend_service.helpers.workspaces import WorkspaceRegistry

    registry = WorkspaceRegistry(WORKSPACES_PATH, WORKSPACES_DIR)
    workspace = registry.get(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    docs = list(workspace.get("documents") or [])
    target = next((d for d in docs if d.get("id") == doc_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Document not found.")
    remaining = [d for d in docs if d.get("id") != doc_id]
    registry._workspaces[workspace_id]["documents"] = remaining
    registry._workspaces[workspace_id]["updatedAt"] = state._time_label()
    registry.save()

    workspace_dir = workspace_docs_dir(state, workspace_id)
    for f in workspace_dir.glob(f"{doc_id}*"):
        try:
            f.unlink()
        except OSError:
            pass
    state.add_log("chat", "info", f"Workspace document removed: {target.get('originalName')}")
    return {"deleted": doc_id}


def retrieve_session_context(
    state: ChaosEngineState,
    session_id: str,
    prompt: str,
    top_k: int = 5,
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve relevant document chunks for a prompt.

    Returns ``(context_text, citations)`` where citations is a list of
    dicts with docId, docName, chunkIndex, page, preview keys.

    Phase 2.6: when an llama-embedding binary + embedding GGUF are
    both discoverable via env vars or ``<dataDir>/embeddings/``,
    retrieval uses semantic cosine similarity blended with BM25
    (70/30) instead of TF-IDF + BM25. The embedding client is
    resolved per-call so newly-installed models pick up without a
    restart, and the legacy lexical path remains the fallback when
    anything goes wrong.
    """
    from backend_service.helpers.documents import DocumentIndex
    from backend_service.rag import resolve_embedding_client

    # Phase 3.7: collect document directories from both the session
    # and (when assigned) the session's workspace, so the RAG
    # retriever sees the merged corpus. Workspace docs survive
    # session deletion + are visible across every session in the
    # workspace.
    chunk_dirs: list[Path] = []
    session_dir = session_docs_dir(state, session_id)
    if session_dir.exists():
        chunk_dirs.append(session_dir)

    with state._lock:
        session = next(
            (s for s in state.chat_sessions if s.get("id") == session_id),
            None,
        )
    workspace_id = session.get("workspaceId") if session else None
    if workspace_id:
        workspace_dir = workspace_docs_dir(state, workspace_id)
        if workspace_dir.exists():
            chunk_dirs.append(workspace_dir)

    if not chunk_dirs:
        return "", []

    # Embedding client discovery: env vars override path; if no
    # CHAOSENGINE_EMBEDDING_MODEL is set we look under
    # `<documents-parent>/embeddings/*.gguf`. Returns None when
    # nothing is wired, in which case retrieval transparently
    # falls back to TF-IDF + BM25.
    from backend_service.app import DOCUMENTS_DIR

    embedding_client = resolve_embedding_client(DOCUMENTS_DIR.parent)

    # Build a temporary index from all collected directories.
    index = DocumentIndex()
    for chunk_dir in chunk_dirs:
        for chunk_file in chunk_dir.glob("*.chunks.json"):
            try:
                doc_chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
                doc_name = chunk_file.stem.replace(".chunks", "")
                full_text = "\n\n".join(c.get("text", "") for c in doc_chunks)
                if full_text.strip():
                    index.add_document(
                        full_text,
                        doc_id=doc_name,
                        doc_name=doc_name,
                        embedding_client=embedding_client,
                    )
            except (OSError, json.JSONDecodeError):
                continue

    results = index.search(prompt, top_k=top_k, embedding_client=embedding_client)
    if not results:
        return "", []

    context = "\n\n---\n\n".join(r["text"] for r in results)
    citations = [r["citation"] for r in results]
    return context, citations
