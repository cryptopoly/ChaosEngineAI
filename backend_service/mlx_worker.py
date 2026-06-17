from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from backend_service.reasoning_split import (
    RAW_REASONING_HEADING_RE,
    ThinkingTokenFilter,
    ThinkingStreamResult,
    reasoning_delimiters_for,
    strip_harmony_boilerplate,
    strip_thinking_tokens as _strip_thinking_tokens,
)
from backend_service.mlx_worker_prompt import (
    TranscriptLoopFilter,
    _build_prompt_text,
    _fallback_chat_prompt,
    _merge_runtime_notes,
    _plain_chat_fallback_active,
    _should_retry_cache_failure,
    _trim_transcript_continuation,
    _TRANSCRIPT_ROLE_LINE_RE,
)
from backend_service.mlx_worker_request import (
    _apply_mlx_seed,
    _build_mlx_sampler,
    _extract_top_logprobs,
    _format_tools_for_prompt,
    _normalize_message_content,
    _sampler_seed,
    _sanitize_messages,
)
from backend_service.mlx_worker_multimodal import (
    decode_images_to_paths,
    format_multimodal_prompt,
    generate_multimodal,
    stream_generate_multimodal,
    vlm_generate_kwargs,
)
from backend_service.mlx_worker_cache import (
    make_mlx_cache,
    runtime_fields,
)
from backend_service.mlx_worker_eval import (
    eval_perplexity,
    eval_task_accuracy,
)
from backend_service.mlx_worker_loader import resolve_local_snapshot

# Phase 1f-10..1f-12: lifecycle + speculative + plain generation paths
# now live in their own modules. Re-import as namespaces so the thin
# class wrappers can dispatch into them.
from backend_service import mlx_worker_lifecycle as _lifecycle
from backend_service import mlx_worker_speculative as _speculative
from backend_service import mlx_worker_generate as _generate
from backend_service import mlx_worker_prompt_cache as _prompt_cache

# Phase 1f-4: model + runtime introspection helpers now live in
# ``backend_service.mlx_worker_diagnostics``. Re-export so existing imports
# + test patches against ``mlx_worker._reject_unsupported_quant`` /
# ``probe`` / ``gguf_metadata`` keep working.
from backend_service.mlx_worker_diagnostics import (  # noqa: E402,F401
    _UNSUPPORTED_QUANT_ALGOS,
    _reject_unsupported_quant,
    gguf_metadata,
    probe,
)


# Phase 2.0.5-F: RunawayGuard now lives in `backend_service.runaway_guard`
# so the llama.cpp stream loop in `state.py` can use the same detector. Re-
# export the symbol here so existing callers / tests keep working without
# import-path churn.
from backend_service.runaway_guard import RunawayGuard  # noqa: E402,F401


# Phase 1f-3: JSON IPC channel + stdio redirect now live in
# ``backend_service.mlx_worker_io``. Re-export so existing
# ``from backend_service.mlx_worker import _emit`` test patches keep
# intercepting the worker's calls (the worker reads ``_emit`` through its
# own re-exported name, not the originating module).
from backend_service.mlx_worker_io import (  # noqa: E402,F401
    _emit,
    _install_stdio_redirect,
    emit_progress,
)


class WorkerState:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        # Multimodal (vision-language) state. ``processor`` is the HF
        # AutoProcessor returned by mlx_vlm.load (image preprocessor +
        # tokenizer). ``is_multimodal`` flips the generate path to
        # ``_generate_multimodal`` / ``_stream_generate_multimodal``
        # which decode the chat ``images`` field into temp files and
        # call ``mlx_vlm.generate`` / ``stream_generate``. Stays
        # ``None`` / ``False`` for plain text-only mlx-lm models.
        self.processor = None
        self.is_multimodal = False
        self.config: dict[str, Any] | None = None
        self.cache_strategy = "native"
        self.cache_bits = 0
        self.fp16_layers = 0
        self.fused_attention = False
        self.context_tokens = 8192
        self.speculative_decoding = False
        self._dflash_generator = None  # Loaded DFlash draft model when active
        self._dflash_target = None     # Target model prepared by dflash_mlx.runtime
        self.tree_budget = 0
        self._ddtree_draft = None     # DFlashDraftModel for DDTree
        self._ddtree_target = None    # target model loaded via dflash_mlx for DDTree
        # FU-002: TriAttention MLX kv_budget. Number of KV positions kept
        # per layer; older positions get scored + evicted by the
        # apply_triattention_mlx compressor. ~2048 is the upstream default
        # and matches the spike result on Qwen2.5-0.5B (2.6x speedup,
        # identical output).
        self.kv_budget = 2048
        # Bug 2 / Gemma 4 channel-token leak: track the currently loaded
        # model ref so the reasoning split layer can pick model-specific
        # delimiters via ``reasoning_delimiters_for``. Default
        # (``<think>...</think>``) still applies when ``None``.
        self._loaded_model_ref: str | None = None
        # Tier 4: persistent single-slot prompt cache for native-strategy chat
        # so follow-up turns prefill only the new suffix. Managed by
        # backend_service.mlx_worker_prompt_cache; invalidated on any model
        # load / unload / profile change.
        self._persist_cache: Any | None = None
        self._persist_tokens: list[int] = []
        self._persist_cache_model_ref: str | None = None

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        op = request.get("op")
        if op == "load_model":
            return self.load_model(request)
        if op == "update_profile":
            return self.update_profile(request)
        if op == "unload_model":
            return self.unload_model()
        if op == "generate":
            return self.generate(request)
        if op == "stream_generate":
            self.stream_generate(request)
            return None
        if op == "eval_perplexity":
            return self.eval_perplexity(request)
        if op == "eval_task_accuracy":
            return self.eval_task_accuracy(request)
        raise ValueError(f"Unsupported worker operation: {op}")

    def load_model(self, request: dict[str, Any]) -> dict[str, Any]:
        _prompt_cache.invalidate(self)
        return _lifecycle.load_model(self, request)

    def unload_model(self) -> dict[str, Any]:
        _prompt_cache.invalidate(self)
        return _lifecycle.unload_model(self)

    def update_profile(self, request: dict[str, Any]) -> dict[str, Any]:
        _prompt_cache.invalidate(self)
        return _lifecycle.update_profile(self, request)

    def _apply_cache_profile(
        self,
        *,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
    ) -> str | None:
        return _lifecycle.apply_cache_profile(
            self,
            cache_strategy=cache_strategy,
            cache_bits=cache_bits,
            fp16_layers=fp16_layers,
            fused_attention=fused_attention,
        )

    def _apply_triattention_mlx_compressor(self) -> str | None:
        return _lifecycle.apply_triattention_mlx_compressor(self)

    def _runtime_fields(
        self,
        *,
        prompt_cache: Any | None,
        speculative_decoding: bool = False,
        tree_budget: int = 0,
    ) -> dict[str, Any]:
        return runtime_fields(
            cache_strategy=self.cache_strategy,
            cache_bits=self.cache_bits,
            fp16_layers=self.fp16_layers,
            prompt_cache=prompt_cache,
            speculative_decoding=speculative_decoding,
            tree_budget=tree_budget,
        )

    def _make_cache(self) -> tuple[Any | None, str | None]:
        return make_mlx_cache(
            model=self.model,
            cache_strategy=self.cache_strategy,
            cache_bits=self.cache_bits,
            fp16_layers=self.fp16_layers,
            fused_attention=self.fused_attention,
        )

    def _generate_dflash(self, request: dict[str, Any]) -> dict[str, Any]:
        return _speculative.generate_dflash(self, request)

    def _generate_ddtree(self, request: dict[str, Any]) -> dict[str, Any]:
        return _speculative.generate_ddtree(self, request)

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        return _generate.generate(self, request)

    def _generate_standard(self, request: dict[str, Any]) -> dict[str, Any]:
        return _generate.generate_standard(self, request)

    @staticmethod
    @staticmethod
    def _decode_images_to_paths(
        images_b64: list[str], temp_dir: str
    ) -> list[str]:
        return decode_images_to_paths(images_b64, temp_dir)

    def _format_multimodal_prompt(
        self,
        request: dict[str, Any],
        num_images: int,
    ) -> str:
        return format_multimodal_prompt(
            self.processor, self.config, self.tokenizer, request, num_images
        )

    def _vlm_generate_kwargs(self, request: dict[str, Any]) -> dict[str, Any]:
        return vlm_generate_kwargs(request)

    def _generate_multimodal(self, request: dict[str, Any]) -> dict[str, Any]:
        return generate_multimodal(
            model=self.model,
            processor=self.processor,
            tokenizer=self.tokenizer,
            config=self.config,
            loaded_model_ref=self._loaded_model_ref,
            request=request,
        )

    def _stream_generate_multimodal(self, request: dict[str, Any]) -> None:
        stream_generate_multimodal(
            model=self.model,
            processor=self.processor,
            tokenizer=self.tokenizer,
            config=self.config,
            loaded_model_ref=self._loaded_model_ref,
            request=request,
        )


    def stream_generate(self, request: dict[str, Any]) -> None:
        return _generate.stream_generate(self, request)

    def eval_perplexity(self, request: dict[str, Any]) -> dict[str, Any]:
        return eval_perplexity(
            model=self.model, tokenizer=self.tokenizer, request=request
        )

    def eval_task_accuracy(self, request: dict[str, Any]) -> dict[str, Any]:
        return eval_task_accuracy(
            model=self.model, tokenizer=self.tokenizer, request=request
        )


def serve() -> int:
    state = WorkerState()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            result = state.handle(request)
            if result is not None:
                _emit({"ok": True, "result": result})
        except Exception as exc:
            _emit(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=4),
                }
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    # Install the stdout split before any subcommand runs — probe() and
    # gguf_metadata() call _emit too, and both import mlx/gguf machinery
    # that can print to stdout on their own.
    _install_stdio_redirect()

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m backend_service.mlx_worker [probe|gguf-metadata|serve]", file=sys.stderr)
        return 1

    command = argv[0]
    if command == "probe":
        return probe()
    if command == "gguf-metadata":
        if len(argv) < 2:
            print("gguf-metadata requires a path argument", file=sys.stderr)
            return 1
        return gguf_metadata(argv[1])
    if command == "serve":
        return serve()

    print(f"unknown command: {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
