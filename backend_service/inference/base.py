"""Inference base classes — data types + the abstract engine interface.

The six classes here are the shared vocabulary every inference engine
in ``backend_service.inference`` speaks. They're lifted out of the old
monolithic ``inference.py`` so engines can be split into separate
modules (mlx_engine, llama_cpp_engine, remote_engine, …) without
circling through the package's ``__init__``.

Contents:

- ``RepeatedLineGuard`` — abort runaway streams that emit the same long
  line repeatedly. Wired into both the llama-server stdout pump and
  the MLX worker's stream output to catch model-melt cases.
- ``BackendCapabilities`` — what each runtime knows how to do
  (mlx vs gguf vs vllm, llama-server paths, version strings). Built by
  ``RuntimeController.refresh_capabilities()``.
- ``LoadedModelInfo`` — everything about the currently-loaded model
  (ref, backend, cache profile, fit-in-memory flag, dflash draft model,
  …). Engines return one of these from ``load_model()``.
- ``GenerationResult`` — single-shot completion payload (text +
  metrics). Returned by ``BaseInferenceEngine.generate()``.
- ``StreamChunk`` — token-stream event (delta text or terminal
  metrics). Yielded by ``BaseInferenceEngine.stream_generate()``.
- ``BaseInferenceEngine`` — the abstract interface every concrete
  engine subclasses. Default implementations of ``stream_generate``,
  ``update_profile``, ``eval_perplexity``, ``eval_task_accuracy`` raise
  if a subclass doesn't override.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any


class RepeatedLineGuard:
    """Abort obviously runaway streams that emit the same long line repeatedly."""

    def __init__(self, *, min_line_length: int = 48, max_repeats: int = 6) -> None:
        self.min_line_length = min_line_length
        self.max_repeats = max_repeats
        self._buffer = ""
        self._last_line: str | None = None
        self._repeat_count = 0

    def feed(self, text: str) -> None:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._check_line(line)

    def flush(self) -> None:
        if self._buffer:
            self._check_line(self._buffer)
            self._buffer = ""

    def _check_line(self, line: str) -> None:
        normalized = " ".join(line.strip().lower().split())
        if len(normalized) < self.min_line_length:
            self._last_line = None
            self._repeat_count = 0
            return

        if normalized == self._last_line:
            self._repeat_count += 1
        else:
            self._last_line = normalized
            self._repeat_count = 1

        if self._repeat_count >= self.max_repeats:
            raise RuntimeError(
                "Stopped runaway generation after repeated identical output from the model."
            )


@dataclass
class BackendCapabilities:
    pythonExecutable: str
    mlxAvailable: bool
    mlxLmAvailable: bool
    mlxUsable: bool
    mlxVersion: str | None = None
    mlxLmVersion: str | None = None
    mlxMessage: str | None = None
    ggufAvailable: bool = False
    llamaCliPath: str | None = None
    llamaServerPath: str | None = None
    llamaServerTurboPath: str | None = None
    converterAvailable: bool = False
    vllmAvailable: bool = False
    vllmVersion: str | None = None
    mtplxAvailable: bool = False
    mtplxPythonPath: str | None = None
    # FU-047: GGUF MTP speculative decoding via llama.cpp PR #22673. Set
    # when the resolved llama-server binary advertises --spec-type in its
    # help text. The UI keys an MTP affordance for GGUF models off this
    # alongside mtplxAvailable for MLX models.
    ggufMtpAvailable: bool = False
    probing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pythonExecutable": self.pythonExecutable,
            "mlxAvailable": self.mlxAvailable,
            "mlxLmAvailable": self.mlxLmAvailable,
            "mlxUsable": self.mlxUsable,
            "mlxVersion": self.mlxVersion,
            "mlxLmVersion": self.mlxLmVersion,
            "mlxMessage": self.mlxMessage,
            "ggufAvailable": self.ggufAvailable,
            "llamaCliPath": self.llamaCliPath,
            "llamaServerPath": self.llamaServerPath,
            "llamaServerTurboPath": self.llamaServerTurboPath,
            "converterAvailable": self.converterAvailable,
            "vllmAvailable": self.vllmAvailable,
            "vllmVersion": self.vllmVersion,
            "mtplxAvailable": self.mtplxAvailable,
            "mtplxPythonPath": self.mtplxPythonPath,
            "ggufMtpAvailable": self.ggufMtpAvailable,
            "probing": self.probing,
        }


@dataclass
class LoadedModelInfo:
    ref: str
    name: str
    backend: str
    source: str
    engine: str
    cacheStrategy: str
    cacheBits: int
    fp16Layers: int
    fusedAttention: bool
    fitModelInMemory: bool
    contextTokens: int
    loadedAt: str
    canonicalRepo: str | None = None
    path: str | None = None
    runtimeTarget: str | None = None
    runtimeNote: str | None = None
    speculativeDecoding: bool = False
    dflashDraftModel: str | None = None
    treeBudget: int = 0
    # Hotfix (2026-05-01 v2): the runtime currently has no mmproj path
    # wired for either backend — `_resolve_gguf_path` strips mmproj
    # files, and the MLX worker has never carried images. Until those
    # paths land (Phase 2.6+ work), `visionEnabled` stays False on every
    # load and the capability resolver demotes the typed `supportsVision`
    # flag accordingly. The catalog `tags` keep "vision" so the UI can
    # still surface "this model supports vision once mmproj loads".
    visionEnabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        # Phase 2.11: include resolved capabilities so the frontend can
        # gate composer affordances (vision, tools, reasoning, etc.)
        # without a separate fetch. Resolved lazily — adding a field on
        # the dataclass would force a migration in every load path.
        # The active engine is passed so capability flags get demoted
        # for runtime gaps (e.g. MLX worker doesn't carry images).
        from backend_service.catalog.capabilities import resolve_capabilities

        capabilities = resolve_capabilities(
            self.ref,
            self.canonicalRepo,
            engine=self.engine,
            vision_enabled=self.visionEnabled,
        ).to_dict()
        return {
            "ref": self.ref,
            "name": self.name,
            "canonicalRepo": self.canonicalRepo,
            "backend": self.backend,
            "source": self.source,
            "engine": self.engine,
            "cacheStrategy": self.cacheStrategy,
            "cacheBits": self.cacheBits,
            "fp16Layers": self.fp16Layers,
            "fusedAttention": self.fusedAttention,
            "fitModelInMemory": self.fitModelInMemory,
            "contextTokens": self.contextTokens,
            "loadedAt": self.loadedAt,
            "path": self.path,
            "runtimeTarget": self.runtimeTarget,
            "runtimeNote": self.runtimeNote,
            "speculativeDecoding": self.speculativeDecoding,
            "dflashDraftModel": self.dflashDraftModel,
            "treeBudget": self.treeBudget,
            "visionEnabled": self.visionEnabled,
            "capabilities": capabilities,
        }


@dataclass
class GenerationResult:
    text: str
    finishReason: str
    promptTokens: int
    completionTokens: int
    totalTokens: int
    tokS: float
    responseSeconds: float
    runtimeNote: str | None = None
    dflashAcceptanceRate: float | None = None
    cache_strategy: str | None = None
    cache_bits: int | None = None
    fp16_layers: int | None = None
    speculative_decoding: bool | None = None
    tree_budget: int | None = None

    def to_metrics(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "finishReason": self.finishReason,
            "promptTokens": self.promptTokens,
            "completionTokens": self.completionTokens,
            "totalTokens": self.totalTokens,
            "tokS": self.tokS,
            "responseSeconds": self.responseSeconds,
            "runtimeNote": self.runtimeNote,
        }
        if self.dflashAcceptanceRate is not None:
            d["dflashAcceptanceRate"] = self.dflashAcceptanceRate
        return d


@dataclass
class StreamChunk:
    text: str | None = None
    reasoning: str | None = None
    reasoning_done: bool = False
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tok_s: float = 0.0
    runtime_note: str | None = None
    dflash_acceptance_rate: float | None = None
    cache_strategy: str | None = None
    cache_bits: int | None = None
    fp16_layers: int | None = None
    speculative_decoding: bool | None = None
    tree_budget: int | None = None
    done: bool = False
    # Phase 3.3: per-token logprobs. When set, contains the chosen
    # token's logprob plus the top-k alternatives. Only populated
    # when the request had `logprobs: N` set.
    token_logprobs: list[dict[str, Any]] | None = None
    # Phase 3.1: DDTree accepted-span overlay data. `accepted_spans`
    # is a run-length-encoded list of {start, length, accepted} over
    # the per-token rendered text in `accepted_token_text`. Only
    # populated when DFLASH speculative decoding ran.
    accepted_spans: list[dict[str, Any]] | None = None
    accepted_token_text: str | None = None


class BaseInferenceEngine:
    engine_name = "base"
    engine_label = "Base runtime"

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
        raise NotImplementedError

    def update_profile(
        self,
        *,
        canonical_repo: str | None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
    ) -> LoadedModelInfo:
        raise RuntimeError(f"{self.engine_name} does not support in-place profile updates.")

    def unload_model(self) -> None:
        raise NotImplementedError

    def process_pid(self) -> int | None:
        return None

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
        raise NotImplementedError

    def eval_perplexity(
        self,
        *,
        dataset: str = "wikitext-2",
        num_samples: int = 64,
        seq_length: int = 512,
        batch_size: int = 4,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"Perplexity evaluation is not supported by the {self.engine_name} backend.")

    def eval_task_accuracy(
        self,
        *,
        task_name: str = "mmlu",
        limit: int = 100,
        num_shots: int = 5,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"Task accuracy evaluation is not supported by the {self.engine_name} backend.")

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
        result = self.generate(
            prompt=prompt,
            history=history,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            images=images,
            tools=tools,
            samplers=samplers,
            reasoning_effort=reasoning_effort,
            json_schema=json_schema,
        )
        yield StreamChunk(text=result.text)
        yield StreamChunk(
            done=True,
            finish_reason=result.finishReason,
            prompt_tokens=result.promptTokens,
            completion_tokens=result.completionTokens,
            total_tokens=result.totalTokens,
            tok_s=result.tokS,
            runtime_note=result.runtimeNote,
        )
