"""Prompt template library routes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend_service.helpers.prompts import PromptLibrary

router = APIRouter(prefix="/api", tags=["prompts"])

# Lazily initialized per-app instance
_library: PromptLibrary | None = None


def _get_library(request: Request) -> PromptLibrary:
    """Return (or create) the singleton PromptLibrary for this app."""
    global _library
    if _library is not None:
        return _library

    state = request.app.state.chaosengine
    data_dir_str = state.settings.get("dataDirectory", "")
    if data_dir_str:
        data_dir = Path(data_dir_str).expanduser()
    else:
        # Fallback: use the app's configured data location
        from backend_service.app import DATA_LOCATION
        data_dir = DATA_LOCATION.data_dir

    _library = PromptLibrary(data_dir)
    return _library


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PromptTemplateRequest(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1, max_length=200)
    systemPrompt: str = Field(default="", max_length=16000)
    tags: list[str] = Field(default_factory=list)
    category: str = Field(default="General", max_length=80)
    fewShotExamples: list[dict[str, Any]] = Field(default_factory=list)
    # Phase 2.7: optional variable declarations + preset samplers + preset model
    variables: list[dict[str, Any]] = Field(default_factory=list)
    presetSamplers: dict[str, Any] | None = None
    presetModelRef: str | None = Field(default=None, max_length=200)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/prompts")
async def list_prompts(
    request: Request,
    query: str | None = None,
    category: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """List all prompt templates, with optional search / filter."""
    lib = _get_library(request)
    tags = [tag] if tag else None
    if query or category or tags:
        templates = lib.search(query=query, category=category, tags=tags)
    else:
        templates = lib.list_all()
    return {"templates": templates, "count": len(templates)}


@router.post("/prompts")
async def create_or_update_prompt(
    body: PromptTemplateRequest,
    request: Request,
) -> dict[str, Any]:
    """Create a new template or update an existing one (if id is provided)."""
    lib = _get_library(request)
    data = body.model_dump()

    if body.id:
        existing = lib.get(body.id)
        if existing:
            updated = lib.update(body.id, data)
            return {"template": updated, "created": False}

    template = lib.create(data)
    return {"template": template, "created": True}


@router.delete("/prompts/{template_id}")
async def delete_prompt(template_id: str, request: Request) -> dict[str, Any]:
    """Delete a prompt template by ID."""
    lib = _get_library(request)
    if not lib.delete(template_id):
        raise HTTPException(status_code=404, detail="Template not found")
    return {"deleted": True, "id": template_id}


# ---------------------------------------------------------------------------
# FU-022: LLM-based prompt enhancer
# ---------------------------------------------------------------------------


class PromptEnhanceRequest(BaseModel):
    """Body for ``POST /api/prompt/enhance``. ``repo`` selects the
    family-specific system prompt; ``modelId`` overrides the default
    enhancer model (Apple Silicon dev machines all default to
    ``mlx-community/Qwen2.5-0.5B-Instruct-4bit``)."""

    prompt: str = Field(min_length=1, max_length=4000)
    repo: str = Field(min_length=1, max_length=200)
    modelId: str | None = None
    maxTokens: int = Field(default=256, ge=32, le=1024)


class PromptEnhanceResponse(BaseModel):
    enhanced: str
    note: str | None
    modelUsed: str | None
    family: str


@router.post("/prompt/enhance")
async def enhance_prompt(payload: PromptEnhanceRequest) -> PromptEnhanceResponse:
    """Rewrite a short prompt into the structured format the requested
    image / video model expects. Apple Silicon path uses ``mlx_lm`` —
    other platforms get a graceful no-op + runtimeNote in the response.

    Synchronous because the model is small (~700 MB / 0.5B params,
    sub-second after a warm cache); first call pays the load cost.
    """
    from backend_service.helpers.prompt_enhancer import (
        enhance_prompt as _enhance,
        _DEFAULT_ENHANCER_MODEL,
    )

    model_id = payload.modelId or _DEFAULT_ENHANCER_MODEL
    result = _enhance(
        payload.prompt,
        repo=payload.repo,
        enabled=True,
        model_id=model_id,
        max_tokens=payload.maxTokens,
    )
    return PromptEnhanceResponse(
        enhanced=result.enhanced,
        note=result.note,
        modelUsed=result.modelUsed,
        family=result.family,
    )
