"""Multi-model comparison endpoint.

Sends the same prompt to two to four models sequentially and returns SSE
events tagged by slot so the frontend can render a side-by-side comparison
view without keeping prior models resident in warm memory.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend_service.i18n import localized_detail


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
    displayLabel: str | None = None
    displayDetail: str | None = None
    format: str | None = None
    quantization: str | None = None
    sizeGb: float | None = None
    contextWindow: str | None = None
    canonicalRepo: str | None = None
    source: str = "catalog"
    backend: str = "auto"
    path: str | None = None
    launch: CompareLaunchSettings = Field(default_factory=CompareLaunchSettings)


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1)
    models: list[CompareModelRequest] | None = None
    # Backwards-compatible shape for older clients/tests. New clients should
    # send ``models`` so the queue can contain 2-4 slots.
    modelA: CompareModelRequest | None = None
    modelB: CompareModelRequest | None = None
    systemPrompt: str | None = None


router = APIRouter()
COMPARE_SLOT_IDS = ("a", "b", "c", "d")


def resolve_compare_models(body: Any, request: Request | None = None) -> list[CompareModelRequest]:
    models = list(body.models or [])
    if not models and body.modelA is not None and body.modelB is not None:
        models = [body.modelA, body.modelB]
    if len(models) < 2 or len(models) > 4:
        message = "Compare requires between 2 and 4 models."
        if request is not None:
            raise HTTPException(
                status_code=422,
                detail=localized_detail(request, message),
            )
        raise HTTPException(status_code=422, detail=message)
    return models


@router.post("/api/chat/compare")
def compare_models(request: Request, body: CompareRequest) -> StreamingResponse:
    """Generate responses from two models side-by-side.

    Returns an SSE stream with events tagged by model slot (``"a"``-``"d"``):
    - ``{"model": "a", "token": "..."}`` — text token from model A
    - ``{"model": "a", "done": true, "text": "...", "tokS": ...}`` — model A finished
    - ``{"allDone": true}`` — all queued models finished
    """
    state = request.app.state.chaosengine
    compare_models = resolve_compare_models(body, request)

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

    def _unload_active_model():
        try:
            state.unload_model()
        except Exception as exc:
            state.add_log(
                "runtime",
                "warning",
                f"Compare could not unload active model after slot: {type(exc).__name__}: {exc}",
            )

    def _run_slot(slot_id: str, model: CompareModelRequest):
        requested_runtime = _requested_runtime_payload(model.launch)
        model_label = model.modelName or model.modelRef
        yield _sse_event({
            "model": slot_id,
            "loading": True,
            "message": f"Loading {model_label}...",
        })

        load_start = time.perf_counter()
        try:
            _load_model(model)
            load_seconds = round(time.perf_counter() - load_start, 2)
            yield _sse_event({
                "model": slot_id,
                "loaded": True,
                "loadSeconds": load_seconds,
                **_applied_runtime_payload(requested_runtime),
            })
        except Exception as exc:
            yield _sse_event({"model": slot_id, "error": str(exc)})
            return

        full_text = ""
        final_chunk = None
        gen_start = time.perf_counter()
        try:
            for chunk in state.runtime.stream_generate(
                prompt=body.prompt,
                history=[],
                system_prompt=body.systemPrompt,
                max_tokens=model.launch.maxTokens,
                temperature=model.launch.temperature,
            ):
                if chunk.reasoning:
                    yield _sse_event({"model": slot_id, "reasoning": chunk.reasoning})
                if chunk.reasoning_done:
                    yield _sse_event({"model": slot_id, "reasoningDone": True})
                if chunk.text:
                    full_text += chunk.text
                    yield _sse_event({"model": slot_id, "token": chunk.text})
                if chunk.done:
                    final_chunk = chunk
                    elapsed = round(time.perf_counter() - gen_start, 2)
                    yield _sse_event({
                        "model": slot_id,
                        "done": True,
                        "text": full_text,
                        "loadSeconds": load_seconds,
                        "totalSeconds": round(load_seconds + elapsed, 2),
                        **_done_runtime_payload(
                            final_chunk=chunk,
                            elapsed_seconds=elapsed,
                            requested_runtime=requested_runtime,
                        ),
                    })
        except Exception as exc:
            yield _sse_event({"model": slot_id, "error": str(exc)})
        finally:
            if final_chunk is None:
                # Even failed/aborted slots should not leave a heavyweight
                # model resident before the next comparison slot starts.
                pass
            _unload_active_model()
            state.runtime.clear_warm_pool()

    def _sse_stream():
        cleared_warm_models = state.runtime.clear_warm_pool()
        if cleared_warm_models:
            state.add_log(
                "runtime",
                "info",
                f"Compare cleared {cleared_warm_models} warm model(s) before exclusive loading.",
            )

        for index, model in enumerate(compare_models):
            slot_id = COMPARE_SLOT_IDS[index]
            yield from _run_slot(slot_id, model)

        yield _sse_event({"allDone": True})

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
