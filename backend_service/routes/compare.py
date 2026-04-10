"""Multi-model comparison endpoint.

Sends the same prompt to two models simultaneously and returns
interleaved SSE streams with model tags so the frontend can render
a side-by-side comparison view.
"""

from __future__ import annotations

import json
import time
import threading
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

    def _sse_stream():
        results: dict[str, dict[str, Any]] = {"a": {}, "b": {}}
        errors: dict[str, str] = {}
        done_events = threading.Event()
        lock = threading.Lock()
        finished_count = [0]

        def _generate_for_model(model_tag: str, model_ref: str):
            full_text = ""
            try:
                # Use the runtime's stream_generate
                for chunk in state.runtime.stream_generate(
                    prompt=body.prompt,
                    history=[],
                    system_prompt=body.systemPrompt,
                    max_tokens=body.maxTokens,
                    temperature=body.temperature,
                ):
                    if chunk.text:
                        full_text += chunk.text
                    if chunk.done:
                        with lock:
                            results[model_tag] = {
                                "text": full_text,
                                "tokS": chunk.tok_s or 0,
                                "promptTokens": chunk.prompt_tokens or 0,
                                "completionTokens": chunk.completion_tokens or 0,
                            }
            except Exception as exc:
                with lock:
                    errors[model_tag] = str(exc)
                    results[model_tag] = {"text": full_text, "error": str(exc)}
            finally:
                with lock:
                    finished_count[0] += 1
                    if finished_count[0] >= 2:
                        done_events.set()

        # Load model A first, generate, then load model B
        # (Sequential to avoid memory issues — warm pool handles caching)
        try:
            # Generate with model A (use currently loaded or load it)
            from backend_service.models import LoadModelRequest
            state.load_model(LoadModelRequest(modelRef=body.modelRefA))
        except Exception as exc:
            yield f"data: {json.dumps({'model': 'a', 'error': str(exc)})}\n\n"
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
                    yield f"data: {json.dumps({'model': 'a', 'token': chunk.text})}\n\n"
                if chunk.done:
                    elapsed_a = round(time.perf_counter() - gen_start_a, 2)
                    tok_s = chunk.tok_s or (chunk.completion_tokens / max(elapsed_a, 0.01) if chunk.completion_tokens else 0)
                    yield f"data: {json.dumps({'model': 'a', 'done': True, 'text': full_text_a, 'tokS': round(tok_s, 1), 'promptTokens': chunk.prompt_tokens or 0, 'completionTokens': chunk.completion_tokens or 0, 'responseSeconds': elapsed_a})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'model': 'a', 'error': str(exc)})}\n\n"

        # Now load model B
        try:
            state.load_model(LoadModelRequest(modelRef=body.modelRefB))
        except Exception as exc:
            yield f"data: {json.dumps({'model': 'b', 'error': str(exc)})}\n\n"
            yield f"data: {json.dumps({'allDone': True})}\n\n"
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
                    yield f"data: {json.dumps({'model': 'b', 'token': chunk.text})}\n\n"
                if chunk.done:
                    elapsed_b = round(time.perf_counter() - gen_start_b, 2)
                    tok_s = chunk.tok_s or (chunk.completion_tokens / max(elapsed_b, 0.01) if chunk.completion_tokens else 0)
                    yield f"data: {json.dumps({'model': 'b', 'done': True, 'text': full_text_b, 'tokS': round(tok_s, 1), 'promptTokens': chunk.prompt_tokens or 0, 'completionTokens': chunk.completion_tokens or 0, 'responseSeconds': elapsed_b})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'model': 'b', 'error': str(exc)})}\n\n"

        yield f"data: {json.dumps({'allDone': True})}\n\n"

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
