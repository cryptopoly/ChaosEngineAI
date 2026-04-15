"""Multi-model comparison endpoint.

Sends the same prompt to two models sequentially and returns SSE events
tagged by model so the frontend can render a side-by-side comparison view.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class CompareLaunchSettings(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    maxTokens: int = Field(default=2048, ge=1, le=32768)
    cacheStrategy: str = "native"
    cacheBits: int = Field(default=0, ge=0, le=8)
    fp16Layers: int = Field(default=0, ge=0, le=16)
    fusedAttention: bool = False
    fitModelInMemory: bool = True
    contextTokens: int = Field(default=8192, ge=256, le=2097152)
    speculativeDecoding: bool = False
    treeBudget: int = Field(default=0, ge=0, le=64)


class CompareModelRequest(BaseModel):
    modelRef: str = Field(min_length=1)
    modelName: str | None = None
    canonicalRepo: str | None = None
    source: str = "catalog"
    backend: str = "auto"
    path: str | None = None
    launch: CompareLaunchSettings = Field(default_factory=CompareLaunchSettings)


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1)
    modelA: CompareModelRequest
    modelB: CompareModelRequest
    systemPrompt: str | None = None


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

    def _requested_runtime_payload(launch: CompareLaunchSettings) -> dict[str, Any]:
        return state._requested_runtime_metrics_fields(
            cache_strategy=launch.cacheStrategy,
            cache_bits=launch.cacheBits,
            fp16_layers=launch.fp16Layers,
            fit_model_in_memory=launch.fitModelInMemory,
            speculative_decoding=launch.speculativeDecoding,
            tree_budget=launch.treeBudget,
        )

    def _compare_loaded_model_metrics() -> dict[str, Any]:
        metrics = state._loaded_model_metrics_fields().copy()
        metrics.pop("model", None)
        return metrics

    def _applied_runtime_payload(requested_runtime: dict[str, Any]) -> dict[str, Any]:
        loaded = state.runtime.loaded_model
        if loaded is None:
            return requested_runtime
        cache_label = state._cache_label(
            cache_strategy=str(loaded.cacheStrategy),
            bits=int(loaded.cacheBits),
            fp16_layers=int(loaded.fp16Layers),
        )
        parts = [cache_label]
        if loaded.contextTokens:
            parts.append(
                f"{round(loaded.contextTokens / 1024)}K ctx"
                if loaded.contextTokens >= 1024
                else f"{loaded.contextTokens} ctx"
            )
        if loaded.speculativeDecoding:
            spec_label = f"DDTree {loaded.treeBudget}" if loaded.treeBudget > 0 else "DFlash"
            if loaded.dflashDraftModel:
                spec_label += f" ({loaded.dflashDraftModel.split('/')[-1]})"
            parts.append(spec_label)
        return {
            **_compare_loaded_model_metrics(),
            **requested_runtime,
            "appliedSummary": " · ".join(parts),
            "runtimeNote": loaded.runtimeNote,
        }

    def _done_runtime_payload(
        *,
        final_chunk: Any,
        elapsed_seconds: float,
        requested_runtime: dict[str, Any],
    ) -> dict[str, Any]:
        completion_tokens = final_chunk.completion_tokens if final_chunk else 0
        prompt_tokens = final_chunk.prompt_tokens if final_chunk else 0
        tok_s = final_chunk.tok_s or (
            completion_tokens / max(elapsed_seconds, 0.01) if completion_tokens else 0
        )
        payload = {
            **_compare_loaded_model_metrics(),
            **state._result_runtime_metrics_fields(final_chunk),
            **requested_runtime,
            "finishReason": final_chunk.finish_reason if final_chunk else "stop",
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": prompt_tokens + completion_tokens,
            "tokS": round(tok_s, 1),
            "responseSeconds": elapsed_seconds,
            "runtimeNote": (
                final_chunk.runtime_note
                if final_chunk and getattr(final_chunk, "runtime_note", None) is not None
                else state.runtime.loaded_model.runtimeNote if state.runtime.loaded_model else None
            ),
        }
        if final_chunk and getattr(final_chunk, "dflash_acceptance_rate", None) is not None:
            payload["dflashAcceptanceRate"] = final_chunk.dflash_acceptance_rate
        return payload

    def _load_model(model: CompareModelRequest):
        """Load a model with its own launch settings from the compare request."""
        from backend_service.models import LoadModelRequest

        launch = model.launch
        req = LoadModelRequest(
            modelRef=model.modelRef,
            modelName=model.modelName,
            canonicalRepo=model.canonicalRepo,
            source=model.source,
            path=model.path,
            backend=model.backend,
            cacheStrategy=launch.cacheStrategy,
            cacheBits=launch.cacheBits,
            fp16Layers=launch.fp16Layers,
            fusedAttention=launch.fusedAttention,
            fitModelInMemory=launch.fitModelInMemory,
            contextTokens=launch.contextTokens,
            speculativeDecoding=launch.speculativeDecoding,
            treeBudget=launch.treeBudget,
        )
        state.load_model(req, keep_warm_previous=False)

    def _sse_stream():
        cleared_warm_models = state.runtime.clear_warm_pool()
        if cleared_warm_models:
            state.add_log(
                "runtime",
                "info",
                f"Compare cleared {cleared_warm_models} warm model(s) before exclusive loading.",
            )

        # --- Model A ---
        yield _sse_event({"model": "a", "loading": True, "message": f"Loading {body.modelA.modelName or body.modelA.modelRef}..."})
        try:
            _load_model(body.modelA)
            requested_runtime_a = _requested_runtime_payload(body.modelA.launch)
            yield _sse_event({"model": "a", "loaded": True, **_applied_runtime_payload(requested_runtime_a)})
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
                max_tokens=body.modelA.launch.maxTokens,
                temperature=body.modelA.launch.temperature,
            ):
                if chunk.reasoning:
                    yield _sse_event({"model": "a", "reasoning": chunk.reasoning})
                if chunk.reasoning_done:
                    yield _sse_event({"model": "a", "reasoningDone": True})
                if chunk.text:
                    full_text_a += chunk.text
                    yield _sse_event({"model": "a", "token": chunk.text})
                if chunk.done:
                    elapsed_a = round(time.perf_counter() - gen_start_a, 2)
                    yield _sse_event({
                        "model": "a",
                        "done": True,
                        "text": full_text_a,
                        **_done_runtime_payload(
                            final_chunk=chunk,
                            elapsed_seconds=elapsed_a,
                            requested_runtime=requested_runtime_a,
                        ),
                    })
        except Exception as exc:
            yield _sse_event({"model": "a", "error": str(exc)})

        # --- Model B ---
        yield _sse_event({"model": "b", "loading": True, "message": f"Loading {body.modelB.modelName or body.modelB.modelRef}..."})
        try:
            _load_model(body.modelB)
            requested_runtime_b = _requested_runtime_payload(body.modelB.launch)
            yield _sse_event({"model": "b", "loaded": True, **_applied_runtime_payload(requested_runtime_b)})
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
                max_tokens=body.modelB.launch.maxTokens,
                temperature=body.modelB.launch.temperature,
            ):
                if chunk.reasoning:
                    yield _sse_event({"model": "b", "reasoning": chunk.reasoning})
                if chunk.reasoning_done:
                    yield _sse_event({"model": "b", "reasoningDone": True})
                if chunk.text:
                    full_text_b += chunk.text
                    yield _sse_event({"model": "b", "token": chunk.text})
                if chunk.done:
                    elapsed_b = round(time.perf_counter() - gen_start_b, 2)
                    yield _sse_event({
                        "model": "b",
                        "done": True,
                        "text": full_text_b,
                        **_done_runtime_payload(
                            final_chunk=chunk,
                            elapsed_seconds=elapsed_b,
                            requested_runtime=requested_runtime_b,
                        ),
                    })
        except Exception as exc:
            yield _sse_event({"model": "b", "error": str(exc)})

        yield _sse_event({"allDone": True})

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
