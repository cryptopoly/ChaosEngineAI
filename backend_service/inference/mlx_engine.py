"""Apple Silicon MLX inference engine.

Wraps the ``backend_service.mlx_worker`` subprocess via the JSON-RPC
bridge. The subprocess split is forced because mlx_lm imports a stack
that segfaults if it cohabits with PyTorch in the same address space —
the worker runs in its own Python interpreter spawned with the
embedded ``capabilities.pythonExecutable``.

Surface mirrors ``BaseInferenceEngine``:

- ``load_model`` / ``unload_model`` / ``update_profile``
- ``generate`` (one-shot) / ``stream_generate`` (token-stream)
- ``eval_perplexity`` / ``eval_task_accuracy`` (perf benchmarks)
- ``process_pid`` (so the runtime status panel can show the PID)

DFLASH speculative decoding is resolved at load time via
``resolve_dflash_target_ref`` + the optional ``dflash`` package. When
the requested target has no registered draft model (or dflash isn't
installed), speculative decoding silently falls back to the standard
single-model path and a runtime note explains why.

Extracted from ``inference.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from backend_service.inference._constants import (
    DEFAULT_MLX_TIMEOUT_SECONDS,
    MLX_LOAD_TIMEOUT_SECONDS,
)
from backend_service.inference._utils import _now_label
from backend_service.inference.base import (
    BackendCapabilities,
    BaseInferenceEngine,
    GenerationResult,
    LoadedModelInfo,
    StreamChunk,
)
from backend_service.inference.jsonrpc import JsonRpcProcess
from backend_service.model_resolution import resolve_dflash_target_ref


class MLXWorkerEngine(BaseInferenceEngine):
    engine_name = "mlx"
    engine_label = "MLX"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self.worker = JsonRpcProcess(
            [self.capabilities.pythonExecutable, "-m", "backend_service.mlx_worker", "serve"],
            timeout=DEFAULT_MLX_TIMEOUT_SECONDS,
        )
        self.loaded_model: LoadedModelInfo | None = None

    def _base_runtime_note(self) -> str:
        return (
            f"Using {Path(self.capabilities.pythonExecutable).name} with MLX {self.capabilities.mlxVersion or 'unknown'} "
            f"and mlx-lm {self.capabilities.mlxLmVersion or 'unknown'}."
        )

    def _compose_runtime_note(
        self,
        *,
        worker_note: str | None,
        dflash_target_ref: str | None,
        requested_speculative: bool,
        actual_speculative: bool,
        actual_draft_model: str | None,
        actual_tree_budget: int,
    ) -> str:
        runtime_note = self._base_runtime_note()
        if worker_note:
            runtime_note = f"{runtime_note} {worker_note}"
        elif actual_speculative and actual_draft_model:
            if actual_tree_budget > 0:
                runtime_note = (
                    f"{runtime_note} DDTree speculative decoding active "
                    f"(budget={actual_tree_budget}, draft: {actual_draft_model})."
                )
            else:
                runtime_note = f"{runtime_note} DFLASH speculative decoding active (draft: {actual_draft_model})."
        elif requested_speculative:
            resolved_ref = dflash_target_ref or (self.loaded_model.ref if self.loaded_model else None) or "unknown target"
            runtime_note = (
                f"{runtime_note} DFLASH unavailable for '{resolved_ref}': no compatible draft model is registered."
            )
        return runtime_note

    def load_model(
        self,
        *,
        model_ref: str,
        model_name: str,
        canonical_repo: str | None,
        source: str,
        backend: str,
        path: str | None,
        runtime_target: str | None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        fit_model_in_memory: bool,
        context_tokens: int,
        speculative_decoding: bool = False,
        tree_budget: int = 0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LoadedModelInfo:
        if not self.capabilities.mlxUsable:
            raise RuntimeError(self.capabilities.mlxMessage or "MLX is not available.")

        # Resolve DFLASH draft model when speculative decoding is requested
        draft_model: str | None = None
        dflash_target_ref = resolve_dflash_target_ref(
            canonical_repo=canonical_repo,
            path=path,
            model_ref=model_ref,
        )
        if speculative_decoding:
            try:
                from dflash import get_draft_model, is_mlx_available
                if is_mlx_available():
                    draft_model = get_draft_model(dflash_target_ref or model_ref)
            except ImportError:
                pass

        target = runtime_target or path or model_ref
        result = self.worker.request_with_progress(
            {
                "op": "load_model",
                "target": target,
                "cacheStrategy": cache_strategy,
                "cacheBits": cache_bits,
                "fp16Layers": fp16_layers,
                "fusedAttention": fused_attention,
                "contextTokens": context_tokens,
                "speculativeDecoding": speculative_decoding and draft_model is not None,
                "dflashDraftModel": draft_model,
                "treeBudget": tree_budget if speculative_decoding and draft_model else 0,
            },
            on_progress=progress_callback,
            timeout=MLX_LOAD_TIMEOUT_SECONDS,
        )
        actual_cache_strategy = str(result.get("cacheStrategy") or cache_strategy)
        actual_cache_bits = int(result.get("cacheBits") if result.get("cacheBits") is not None else cache_bits)
        actual_fp16_layers = int(result.get("fp16Layers") if result.get("fp16Layers") is not None else fp16_layers)
        actual_fused_attention = bool(
            result.get("fusedAttention") if result.get("fusedAttention") is not None else fused_attention
        )
        actual_speculative = bool(result.get("speculativeDecoding"))
        actual_tree_budget = int(result.get("treeBudget") or 0)
        actual_draft_model = (
            str(result.get("dflashDraftModel"))
            if result.get("dflashDraftModel")
            else (draft_model if actual_speculative else None)
        )
        runtime_note = self._compose_runtime_note(
            worker_note=str(result.get("note") or "").strip() or None,
            dflash_target_ref=dflash_target_ref,
            requested_speculative=speculative_decoding,
            actual_speculative=actual_speculative,
            actual_draft_model=actual_draft_model,
            actual_tree_budget=actual_tree_budget,
        )

        self.loaded_model = LoadedModelInfo(
            ref=model_ref,
            name=model_name,
            backend=backend,
            source=source,
            engine=self.engine_name,
            cacheStrategy=actual_cache_strategy,
            cacheBits=actual_cache_bits,
            fp16Layers=actual_fp16_layers,
            fusedAttention=actual_fused_attention,
            fitModelInMemory=fit_model_in_memory,
            contextTokens=context_tokens,
            loadedAt=_now_label(),
            canonicalRepo=canonical_repo,
            path=path,
            runtimeTarget=target,
            runtimeNote=runtime_note,
            speculativeDecoding=actual_speculative,
            dflashDraftModel=actual_draft_model,
            treeBudget=actual_tree_budget,
        )
        return self.loaded_model

    def update_profile(
        self,
        *,
        canonical_repo: str | None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
    ) -> LoadedModelInfo:
        if self.loaded_model is None:
            raise RuntimeError("No MLX model is loaded.")
        if not self.worker.is_alive():
            self.loaded_model = None
            raise RuntimeError(
                "The MLX worker process exited and the model is no longer loaded. "
                "Please reload the model from My Models."
            )

        result = self.worker.request_with_progress(
            {
                "op": "update_profile",
                "cacheStrategy": cache_strategy,
                "cacheBits": cache_bits,
                "fp16Layers": fp16_layers,
                "fusedAttention": fused_attention,
            },
            on_progress=None,
            timeout=DEFAULT_MLX_TIMEOUT_SECONDS,
        )

        self.loaded_model.cacheStrategy = str(result.get("cacheStrategy") or cache_strategy)
        self.loaded_model.cacheBits = int(result.get("cacheBits") if result.get("cacheBits") is not None else cache_bits)
        self.loaded_model.fp16Layers = int(result.get("fp16Layers") if result.get("fp16Layers") is not None else fp16_layers)
        self.loaded_model.fusedAttention = bool(
            result.get("fusedAttention") if result.get("fusedAttention") is not None else fused_attention
        )
        if canonical_repo is not None:
            self.loaded_model.canonicalRepo = canonical_repo
        dflash_target_ref = resolve_dflash_target_ref(
            canonical_repo=self.loaded_model.canonicalRepo,
            path=self.loaded_model.path,
            model_ref=self.loaded_model.ref,
        )
        self.loaded_model.runtimeNote = self._compose_runtime_note(
            worker_note=str(result.get("note") or "").strip() or None,
            dflash_target_ref=dflash_target_ref,
            requested_speculative=self.loaded_model.speculativeDecoding,
            actual_speculative=self.loaded_model.speculativeDecoding,
            actual_draft_model=self.loaded_model.dflashDraftModel,
            actual_tree_budget=self.loaded_model.treeBudget,
        )
        return self.loaded_model

    def unload_model(self) -> None:
        self.loaded_model = None
        # Skip the in-worker cleanup RPC (gc.collect / mx.metal.clear_cache)
        # and kill the process immediately.  The OS reclaims all process memory
        # on exit, so the manual cleanup is redundant and was the main source
        # of multi-second blocking during unload.
        self.worker.close(force=True)

    def process_pid(self) -> int | None:
        process = self.worker.process
        if process is None or process.poll() is not None:
            return None
        return int(process.pid)

    def generate(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        samplers: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        # Detect worker process restart: if the process died, the model is gone.
        if not self.worker.is_alive():
            self.loaded_model = None
            raise RuntimeError(
                "The MLX worker process was restarted and the model is no longer loaded. "
                "Please reload the model from My Models."
            )

        started_at = time.perf_counter()
        payload: dict[str, Any] = {
            "op": "generate",
            "prompt": prompt,
            "history": history,
            "systemPrompt": system_prompt,
            "maxTokens": max_tokens,
            "temperature": temperature,
        }
        if images:
            payload["images"] = images
        if tools:
            payload["tools"] = tools
        # Phase 2.2: forward whatever sampler subset mlx-lm supports.
        # Worker side reads these out of the payload and ignores keys it
        # doesn't recognise, so this is forward-compatible.
        if samplers:
            payload["samplers"] = samplers
        if reasoning_effort:
            payload["reasoningEffort"] = reasoning_effort
        if json_schema:
            payload["jsonSchema"] = json_schema
        result = self.worker.request(payload)
        elapsed = max(time.perf_counter() - started_at, 1e-6)
        return GenerationResult(
            text=str(result.get("text") or ""),
            finishReason=str(result.get("finishReason") or "stop"),
            promptTokens=int(result.get("promptTokens") or 0),
            completionTokens=int(result.get("completionTokens") or 0),
            totalTokens=int(result.get("totalTokens") or 0),
            tokS=float(result.get("tokS") or 0.0),
            responseSeconds=round(float(result.get("responseSeconds") or elapsed), 2),
            runtimeNote=str(result.get("runtimeNote") or self.loaded_model.runtimeNote),
            dflashAcceptanceRate=result.get("dflashAcceptanceRate"),
            cache_strategy=str(result.get("cacheStrategy")) if result.get("cacheStrategy") is not None else None,
            cache_bits=int(result.get("cacheBits")) if result.get("cacheBits") is not None else None,
            fp16_layers=int(result.get("fp16Layers")) if result.get("fp16Layers") is not None else None,
            speculative_decoding=(
                bool(result.get("speculativeDecoding"))
                if result.get("speculativeDecoding") is not None
                else None
            ),
            tree_budget=int(result.get("treeBudget")) if result.get("treeBudget") is not None else None,
        )

    def stream_generate(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking_mode: str | None = None,
        samplers: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        if not self.worker.is_alive():
            self.loaded_model = None
            raise RuntimeError(
                "The MLX worker process exited and the model is no longer loaded. "
                "Please reload the model from My Models."
            )

        payload: dict[str, Any] = {
            "op": "stream_generate",
            "prompt": prompt,
            "history": history,
            "systemPrompt": system_prompt,
            "maxTokens": max_tokens,
            "temperature": temperature,
        }
        if thinking_mode:
            payload["thinkingMode"] = thinking_mode
        if images:
            payload["images"] = images
        if tools:
            payload["tools"] = tools
        # Phase 2.2: forward sampler / reasoning / schema overrides. The
        # MLX worker reads these from the payload and applies what it
        # supports (top_p, top_k, min_p, repeat_penalty, seed via
        # mlx-lm); reasoning_effort + json_schema are accepted for
        # forward-compat with future mlx-lm releases.
        if samplers:
            payload["samplers"] = samplers
        if reasoning_effort:
            payload["reasoningEffort"] = reasoning_effort
        if json_schema:
            payload["jsonSchema"] = json_schema
        try:
            request_iter = self.worker.stream_request(payload)
        except RuntimeError as exc:
            if "No MLX model is loaded" in str(exc):
                self.loaded_model = None
                raise RuntimeError(
                    "The MLX worker lost the loaded model. "
                    "Please reload the model from My Models."
                ) from exc
            raise
        try:
            for response in request_iter:
                chunk = response.get("chunk")
                if chunk:
                    if chunk.get("reasoning"):
                        yield StreamChunk(reasoning=chunk["reasoning"])
                    if chunk.get("reasoningDone"):
                        yield StreamChunk(reasoning_done=True)
                    if chunk.get("text"):
                        token_logprobs = chunk.get("tokenLogprobs")
                        yield StreamChunk(
                            text=chunk["text"],
                            token_logprobs=token_logprobs if token_logprobs else None,
                        )
                    elif chunk.get("tokenLogprobs"):
                        # Phase 3.3 follow-up: forward logprobs even when
                        # the chunk has no text (e.g. emitted alongside
                        # reasoning) so the frontend overlay still gets
                        # a complete trace.
                        yield StreamChunk(token_logprobs=chunk["tokenLogprobs"])
                if response.get("done"):
                    result = response.get("result") or {}
                    yield StreamChunk(
                        done=True,
                        finish_reason=str(result.get("finishReason") or "stop"),
                        prompt_tokens=int(result.get("promptTokens") or 0),
                        completion_tokens=int(result.get("completionTokens") or 0),
                        total_tokens=int(result.get("totalTokens") or 0),
                        tok_s=float(result.get("tokS") or 0.0),
                        runtime_note=str(result.get("runtimeNote") or self.loaded_model.runtimeNote),
                        dflash_acceptance_rate=result.get("dflashAcceptanceRate"),
                        cache_strategy=str(result.get("cacheStrategy")) if result.get("cacheStrategy") is not None else None,
                        cache_bits=int(result.get("cacheBits")) if result.get("cacheBits") is not None else None,
                        fp16_layers=int(result.get("fp16Layers")) if result.get("fp16Layers") is not None else None,
                        speculative_decoding=(
                            bool(result.get("speculativeDecoding"))
                            if result.get("speculativeDecoding") is not None
                            else None
                        ),
                        tree_budget=int(result.get("treeBudget")) if result.get("treeBudget") is not None else None,
                        # Phase 3.1: forward accepted-span data when DDTree
                        # populated it. Llama path leaves these as None.
                        accepted_spans=result.get("acceptedSpans"),
                        accepted_token_text=result.get("acceptedTokenText"),
                    )
        except RuntimeError as exc:
            if "No MLX model is loaded" in str(exc):
                self.loaded_model = None
                raise RuntimeError(
                    "The MLX worker lost the loaded model. "
                    "Please reload the model from My Models."
                ) from exc
            raise

    def eval_perplexity(
        self,
        *,
        dataset: str = "wikitext-2",
        num_samples: int = 64,
        seq_length: int = 512,
        batch_size: int = 4,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        return self.worker.request_with_progress(
            {
                "op": "eval_perplexity",
                "dataset": dataset,
                "numSamples": num_samples,
                "seqLength": seq_length,
                "batchSize": batch_size,
            },
            on_progress=progress_callback,
            timeout=600,
        )

    def eval_task_accuracy(
        self,
        *,
        task_name: str = "mmlu",
        limit: int = 100,
        num_shots: int = 5,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        return self.worker.request_with_progress(
            {
                "op": "eval_task_accuracy",
                "taskName": task_name,
                "limit": limit,
                "numShots": num_shots,
            },
            on_progress=progress_callback,
            timeout=900,
        )
