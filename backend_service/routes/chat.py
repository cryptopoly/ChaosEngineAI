from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, UploadFile, File

from backend_service.models import (
    CreateSessionRequest,
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
    """List all available agent tools with their schemas."""
    tools = tool_registry.list_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "schema": t.openai_schema(),
            }
            for t in tools
        ],
    }
