from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile, File

from backend_service.models import (
    AddVariantRequest,
    CreateSessionRequest,
    ForkSessionRequest,
    UpdateSessionRequest,
    GenerateRequest,
)
from backend_service.tools import registry as tool_registry

router = APIRouter()


@router.post("/api/chat/sessions")
def create_session(request: Request, body: CreateSessionRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    session = state.create_session(title=body.title)
    return {"session": session}


@router.post("/api/chat/sessions/{session_id}/variants")
def add_message_variant(request: Request, session_id: str, body: AddVariantRequest) -> dict[str, Any]:
    """Phase 2.5: generate a sibling variant of an assistant message
    using a different model. Returns the updated session payload so
    the frontend can swap its local copy in one round-trip."""
    state = request.app.state.chaosengine
    try:
        session = state.add_message_variant(
            session_id=session_id,
            message_index=body.messageIndex,
            model_ref=body.modelRef,
            model_name=body.modelName,
            canonical_repo=body.canonicalRepo,
            source=body.source,
            path=body.path,
            backend=body.backend,
            max_tokens=body.maxTokens,
            temperature=body.temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session": session}


@router.post("/api/chat/sessions/{session_id}/fork")
def fork_session(request: Request, session_id: str, body: ForkSessionRequest) -> dict[str, Any]:
    """Phase 2.4: fork an existing thread at a chosen message.

    Returns the freshly-created session payload (same shape as
    create_session) plus the parent linkage on its
    `parentSessionId` / `forkedAtMessageIndex` fields. Frontend
    swaps the active chat to the new fork and lets the user
    continue divergently.
    """
    state = request.app.state.chaosengine
    try:
        session = state.fork_session(
            source_session_id=session_id,
            fork_at_message_index=body.forkAtMessageIndex,
            title=body.title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session": session}


@router.patch("/api/chat/sessions/{session_id}")
def update_session(request: Request, session_id: str, body: UpdateSessionRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    session = state.update_session(session_id, body)
    return {"session": session}


@router.delete("/api/chat/sessions/{session_id}")
def delete_session(request: Request, session_id: str) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return state.delete_session(session_id)


@router.post("/api/chat/generate")
def generate(request: Request, body: GenerateRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return state.generate(body)


@router.post("/api/chat/generate/stream")
def generate_stream(request: Request, body: GenerateRequest):
    state = request.app.state.chaosengine
    return state.generate_stream(body)


@router.post("/api/chat/generate/{session_id}/cancel")
def cancel_generate(request: Request, session_id: str) -> dict[str, Any]:
    """Mark an in-flight chat generation for cancellation.

    The streaming loop checks this flag between events and stops gracefully,
    persisting whatever output has accumulated. Returning is fast — the
    actual stream termination happens on the client's open SSE connection.
    """
    state = request.app.state.chaosengine
    return state.request_cancel_chat(session_id)


@router.get("/api/chat/sessions/{session_id}/documents")
def list_session_documents(request: Request, session_id: str) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return {"documents": state.list_documents(session_id)}


@router.post("/api/chat/sessions/{session_id}/documents")
async def upload_session_document(request: Request, session_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    state = request.app.state.chaosengine
    raw = await file.read()
    return {"document": state.upload_document(session_id, file.filename or "document", raw)}


@router.delete("/api/chat/sessions/{session_id}/documents/{doc_id}")
def delete_session_document(request: Request, session_id: str, doc_id: str) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return state.delete_document(session_id, doc_id)


@router.get("/api/tools")
def list_tools() -> dict[str, Any]:
    """List all available agent tools with their schemas.

    Phase 2.10: each entry now carries a `provenance` field — either
    ``"builtin"`` for the in-tree tools (web search, calculator,
    file reader, code executor) or ``"mcp:<server-id>"`` for tools
    sourced from a configured MCP server. The frontend renders a
    badge per source so users can tell which tools came from where.
    """
    tools = tool_registry.list_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "schema": t.openai_schema(),
                "provenance": getattr(t, "provenance", "builtin"),
            }
            for t in tools
        ],
    }
