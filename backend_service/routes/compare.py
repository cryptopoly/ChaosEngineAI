"""Multi-model comparison endpoint.

Sends the same prompt to two models simultaneously and returns
interleaved SSE streams with model tags so the frontend can render
a side-by-side comparison view.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1)
    modelRefA: str = Field(min_length=1)
    modelRefB: str = Field(min_length=1)
    systemPrompt: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    maxTokens: int = Field(default=2048, ge=1, le=32768)
    # Launch settings (shared for both models)
    cacheStrategy: str = "native"
    cacheBits: int = Field(default=0, ge=0, le=8)
    fp16Layers: int = Field(default=0, ge=0, le=16)
    fusedAttention: bool = False
    fitModelInMemory: bool = True
    contextTokens: int = Field(default=8192, ge=256, le=2097152)
    speculativeDecoding: bool = False


router = APIRouter()


@router.post("/api/chat/compare")
def compare_models(request: Request, body: CompareRequest) -> StreamingResponse:
    """Generate responses from two models side-by-side.

    Returns an SSE stream with events tagged by model (``"a"`` or ``"b"``):
    - ``{"model": "a", "token": "..."}`` — text token from model A
    - ``{"model": "b", "token": "..."}`` — text token from model B
    - ``{"model": "a", "done": true, "text": "...", "tokS": ...}`` — model A finished
    - ``{"model": "b", "done": true, "text": "...", "tokS": ...}`` — model B finished
    - ``{"allDone": true}`` — both models finished
    """
    state = request.app.state.chaosengine

    def _sse_event(data: dict[str, Any]) -> str:
        return f"data: {json.dumps(data)}\n\n"

    def _load_model(model_tag: str, model_ref: str):
        """Load a model with the shared launch settings from the compare request.

        The *model_ref* may be either a HuggingFace repo id **or** a local
        file-system path (the compare UI sends ``item.path || item.name``).
        """
        from backend_service.models import LoadModelRequest
        from pathlib import Path

        is_path = "/" in model_ref and Path(model_ref).exists()
        req = LoadModelRequest(
            modelRef=model_ref,
            path=model_ref if is_path else None,
            backend="auto",
            cacheStrategy=body.cacheStrategy,
            cacheBits=body.cacheBits,
            fp16Layers=body.fp16Layers,
            fusedAttention=body.fusedAttention,
            fitModelInMemory=body.fitModelInMemory,
            contextTokens=body.contextTokens,
            speculativeDecoding=body.speculativeDecoding,
        )
        state.load_model(req)

    def _sse_stream():
        # --- Model A ---
        yield _sse_event({"model": "a", "loading": True, "message": "Loading model A..."})
        try:
            _load_model("a", body.modelRefA)
        except Exception as exc:
            yield _sse_event({"model": "a", "error": str(exc)})
            yield _sse_event({"allDone": True})
            return

        full_text_a = ""
        gen_start_a = time.perf_counter()
        try:
            for chunk in state.runtime.stream_generate(
                prompt=body.prompt,
                history=[],
                system_prompt=body.systemPrompt,
                max_tokens=body.maxTokens,
                temperature=body.temperature,
            ):
                if chunk.text:
                    full_text_a += chunk.text
                    yield _sse_event({"model": "a", "token": chunk.text})
                if chunk.done:
                    elapsed_a = round(time.perf_counter() - gen_start_a, 2)
                    tok_s = chunk.tok_s or (chunk.completion_tokens / max(elapsed_a, 0.01) if chunk.completion_tokens else 0)
                    yield _sse_event({"model": "a", "done": True, "text": full_text_a, "tokS": round(tok_s, 1), "promptTokens": chunk.prompt_tokens or 0, "completionTokens": chunk.completion_tokens or 0, "responseSeconds": elapsed_a})
        except Exception as exc:
            yield _sse_event({"model": "a", "error": str(exc)})

        # --- Model B ---
        yield _sse_event({"model": "b", "loading": True, "message": "Loading model B..."})
        try:
            _load_model("b", body.modelRefB)
        except Exception as exc:
            yield _sse_event({"model": "b", "error": str(exc)})
            yield _sse_event({"allDone": True})
            return

        full_text_b = ""
        gen_start_b = time.perf_counter()
        try:
            for chunk in state.runtime.stream_generate(
                prompt=body.prompt,
                history=[],
                system_prompt=body.systemPrompt,
                max_tokens=body.maxTokens,
                temperature=body.temperature,
            ):
                if chunk.text:
                    full_text_b += chunk.text
                    yield _sse_event({"model": "b", "token": chunk.text})
                if chunk.done:
                    elapsed_b = round(time.perf_counter() - gen_start_b, 2)
                    tok_s = chunk.tok_s or (chunk.completion_tokens / max(elapsed_b, 0.01) if chunk.completion_tokens else 0)
                    yield _sse_event({"model": "b", "done": True, "text": full_text_b, "tokS": round(tok_s, 1), "promptTokens": chunk.prompt_tokens or 0, "completionTokens": chunk.completion_tokens or 0, "responseSeconds": elapsed_b})
        except Exception as exc:
            yield _sse_event({"model": "b", "error": str(exc)})

        yield _sse_event({"allDone": True})

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
