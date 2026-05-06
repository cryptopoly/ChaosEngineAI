from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from backend_service.models import (
    OpenAIChatCompletionRequest,
    OpenAIEmbeddingsRequest,
)

router = APIRouter()


@router.get("/v1/models")
def list_openai_models(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return state.openai_models()


@router.post("/v1/chat/completions")
def openai_chat_completion(request: Request, body: OpenAIChatCompletionRequest):
    state = request.app.state.chaosengine
    return state.openai_chat_completion(body)


@router.post("/v1/embeddings")
def openai_embeddings(request: Request, body: OpenAIEmbeddingsRequest) -> dict[str, Any]:
    """Phase 2.13: OpenAI-compatible embeddings via the bundled GGUF.

    Lets external scripts / IDE plugins / Jupyter hit local models
    without re-implementing inference. Falls back to a 503 when no
    embedding binary or model is configured — the caller should
    decide whether to keyword-search or surface the gap.
    """
    state = request.app.state.chaosengine
    return state.openai_embeddings(body)
