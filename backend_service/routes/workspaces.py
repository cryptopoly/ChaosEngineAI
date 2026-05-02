"""Phase 3.7: workspace knowledge stack routes.

CRUD over workspace metadata + per-workspace document listing.
Document upload / delete reuse the existing `state.upload_document`
path with a different target dir; ChatSession assignment is a
PATCH on the session.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from pydantic import BaseModel, Field

from backend_service.helpers.workspaces import WorkspaceRegistry

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])

_registry: WorkspaceRegistry | None = None


def _get_registry(_request: Request) -> WorkspaceRegistry:
    global _registry
    if _registry is not None:
        return _registry
    from backend_service.app import WORKSPACES_PATH, WORKSPACES_DIR
    _registry = WorkspaceRegistry(WORKSPACES_PATH, WORKSPACES_DIR)
    return _registry


class WorkspaceRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class WorkspaceUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)


@router.get("")
def list_workspaces(request: Request) -> dict[str, Any]:
    registry = _get_registry(request)
    return {"workspaces": registry.list_all()}


@router.post("")
def create_workspace(request: Request, body: WorkspaceRequest) -> dict[str, Any]:
    registry = _get_registry(request)
    return {"workspace": registry.create(body.title, body.description)}


@router.patch("/{workspace_id}")
def update_workspace(
    request: Request,
    workspace_id: str,
    body: WorkspaceUpdateRequest,
) -> dict[str, Any]:
    registry = _get_registry(request)
    updated = registry.update(workspace_id, title=body.title, description=body.description)
    if updated is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"workspace": updated}


@router.delete("/{workspace_id}")
def delete_workspace(request: Request, workspace_id: str) -> dict[str, Any]:
    registry = _get_registry(request)
    if not registry.delete(workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"deleted": True, "id": workspace_id}


@router.post("/{workspace_id}/documents")
async def upload_workspace_document(
    request: Request,
    workspace_id: str,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    registry = _get_registry(request)
    workspace = registry.get(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    state = request.app.state.chaosengine
    raw = await file.read()
    return {
        "document": state.upload_workspace_document(
            workspace_id=workspace_id,
            filename=file.filename or "document",
            data=raw,
        )
    }


@router.delete("/{workspace_id}/documents/{doc_id}")
def delete_workspace_document(
    request: Request,
    workspace_id: str,
    doc_id: str,
) -> dict[str, Any]:
    registry = _get_registry(request)
    if registry.get(workspace_id) is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    state = request.app.state.chaosengine
    return state.delete_workspace_document(workspace_id, doc_id)
