from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from backend_service.models import OpenAIChatCompletionRequest

router = APIRouter()


@router.get("/v1/models")
def list_openai_models(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return state.openai_models()


@router.post("/v1/chat/completions")
def openai_chat_completion(request: Request, body: OpenAIChatCompletionRequest):
    state = request.app.state.chaosengine
    return state.openai_chat_completion(body)
