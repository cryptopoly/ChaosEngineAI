"""Worker lifecycle (load / unload / update_profile + cache profile) for the MLX worker.

Five helpers lifted out of ``WorkerState`` covering the full
load / cache-profile lifecycle:

* ``load_model`` — resolve target, snapshot-download via
  ``resolve_local_snapshot``, run mlx_lm.load (or mlx_vlm.load for
  multimodal repos) under a heartbeat thread, optionally bootstrap
  DFLASH + DDTree speculative decoding, and apply the cache profile.
* ``unload_model`` — drop the model + tokenizer + processor + DFlash /
  DDTree state and free MLX Metal cache.
* ``update_profile`` — apply a cache profile change in-place against
  an already-loaded model.
* ``apply_cache_profile`` — set cache_strategy/bits/fp16_layers/fused
  on the worker; instantiate the prompt cache or fall back to native
  on failure. TriAttention path branches off via the next helper.
* ``apply_triattention_mlx_compressor`` — apply the FU-002
  ``apply_triattention_mlx`` compressor to the loaded model in-place.

All five take ``state: WorkerState`` as the first argument so the
class methods become 1-3 line wrappers.

Extracted from ``backend_service/mlx_worker.py`` as part of the
v0.8.0 Phase 1f-10 refactor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend_service.mlx_worker_cache import make_mlx_cache
from backend_service.mlx_worker_diagnostics import _reject_unsupported_quant
from backend_service.mlx_worker_io import emit_progress
from backend_service.mlx_worker_loader import resolve_local_snapshot
from backend_service.mlx_worker_prompt import _merge_runtime_notes


if TYPE_CHECKING:
    from backend_service.mlx_worker import WorkerState


def load_model(state: WorkerState, request: dict[str, Any]) -> dict[str, Any]:
    from mlx_lm import load

    target = str(request["target"])
    requested_cache_strategy = str(request.get("cacheStrategy", "native"))
    requested_cache_bits = int(request.get("cacheBits", 0))
    requested_fp16_layers = int(request.get("fp16Layers", 0))
    requested_fused_attention = bool(request.get("fusedAttention", False))
    # FU-002: kv_budget for the TriAttention MLX compressor. Ignored
    # when cache_strategy != "triattention". Falls back to 2048 (the
    # upstream default validated by scripts/spike_triattention_mlx.py).
    state.kv_budget = max(64, int(request.get("kvBudget", 2048)))
    state.context_tokens = int(request.get("contextTokens", 8192))
    state.speculative_decoding = bool(request.get("speculativeDecoding", False))
    dflash_draft_model = request.get("dflashDraftModel")
    state._dflash_generator = None
    state._dflash_target = None
    state._ddtree_draft = None
    state._ddtree_target = None
    state.tree_budget = 0

    emit_progress("resolving", 5.0, f"Resolving model target: {target}")

    # Pre-resolve the snapshot so we can stream download progress. The
    # helper hands a local path back, downloading from HF when needed
    # and translating gated/404/auth errors into user-readable
    # RuntimeError messages.
    local_path = resolve_local_snapshot(target)

    # Start a heartbeat that ticks the UI every 2s while mlx_lm.load
    # blocks. mlx_lm doesn't expose a progress callback, so large models
    # (20B+) would otherwise sit at a frozen 60% for 1-2 minutes.
    import threading
    load_done = threading.Event()
    load_start = time.monotonic()
    emit_progress("loading", 60.0, "Loading weights into MLX")

    def _heartbeat() -> None:
        tick = 0
        while not load_done.wait(2.0):
            tick += 1
            elapsed = int(time.monotonic() - load_start)
            # Creep the percent very slowly from 60 → 90 so the bar feels
            # alive without overstating progress we can't measure.
            pct = min(90.0, 60.0 + tick * 1.2)
            emit_progress(
                "loading",
                pct,
                f"Loading weights into MLX... ({elapsed}s)",
            )

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()

    # Multimodal branch: vision-capable repos (Gemma 4, Qwen2.5-VL,
    # LLaVA family) load via mlx_vlm.load → ``(model, processor)``.
    # The processor wraps the HF tokenizer so downstream code that
    # reads ``state.tokenizer`` keeps working. When the multimodal
    # extra isn't installed, fall back to mlx_lm.load with a
    # runtimeNote so the user gets a clear "install mlx-vlm" hint.
    from backend_service.helpers.chat_template import is_multimodal_family
    multimodal_note: str | None = None
    use_multimodal = is_multimodal_family(target)
    try:
        # Reject quantisation formats that MLX cannot dequantize.
        _reject_unsupported_quant(local_path)
        if use_multimodal:
            try:
                from mlx_vlm import load as mlx_vlm_load  # type: ignore[import-untyped]
            except ImportError as exc:
                multimodal_note = (
                    f"Vision model {target!r} requires mlx-vlm but the "
                    f"package isn't installed ({exc}). Falling back to "
                    "mlx_lm text-only load — image inputs will be ignored."
                )
                use_multimodal = False

        if use_multimodal:
            state.model, state.processor = mlx_vlm_load(local_path)
            state.tokenizer = getattr(state.processor, "tokenizer", None)
            # mlx_vlm.load doesn't return a config dict — read it from
            # the snapshot directly so prompt-formatter + chat-template
            # paths can still introspect (e.g. ``num_attention_heads``
            # for cache estimation).
            config_path = Path(local_path) / "config.json"
            if config_path.exists():
                try:
                    state.config = json.loads(config_path.read_text())
                except Exception:
                    state.config = {}
            else:
                state.config = {}
            state.is_multimodal = True
        else:
            state.model, state.tokenizer, state.config = load(local_path, return_config=True)
            state.processor = None
            state.is_multimodal = False
        state._loaded_model_ref = target
    finally:
        load_done.set()
        heartbeat_thread.join(timeout=0.5)
    emit_progress("ready", 95.0, "Finalising")

    # Initialise DFLASH speculative decoding if requested
    dflash_note = None
    state.tree_budget = int(request.get("treeBudget") or 0)
    if state.speculative_decoding and dflash_draft_model:
        try:
            from dflash_mlx.runtime import configure_full_attention_split, load_draft_bundle
            emit_progress("dflash", 96.0, f"Loading DFLASH draft model: {dflash_draft_model}")
            # Reuse the already loaded MLX target model. Loading a second
            # target bundle can duplicate the full model footprint and
            # trigger SIGKILL on large models during DFLASH startup.
            state._dflash_target = state.model
            configure_full_attention_split(state._dflash_target, enabled=True)
            state._dflash_generator, _ = load_draft_bundle(dflash_draft_model, lazy=True)
            dflash_note = f"DFLASH speculative decoding active (draft: {dflash_draft_model})."
        except ImportError as exc:
            dflash_note = f"dflash-mlx could not be imported ({exc}). Falling back to standard generation."
            state.speculative_decoding = False
        except Exception as exc:
            dflash_note = f"DFLASH initialisation failed: {exc}. Falling back to standard generation."
            state.speculative_decoding = False

        # Load DDTree components when tree budget is set
        if state.speculative_decoding and state.tree_budget > 0:
            try:
                emit_progress("ddtree", 97.0, "Preparing DDTree runtime")
                state._ddtree_target = state._dflash_target
                state._ddtree_draft = state._dflash_generator
                dflash_note = f"DDTree speculative decoding active (budget={state.tree_budget}, draft: {dflash_draft_model})."
            except Exception as exc:
                dflash_note = f"DDTree init failed ({exc}). Using linear DFLASH."
                state.tree_budget = 0
                state._ddtree_draft = None
                state._ddtree_target = None

    profile_note = state._apply_cache_profile(
        cache_strategy=requested_cache_strategy,
        cache_bits=requested_cache_bits,
        fp16_layers=requested_fp16_layers,
        fused_attention=requested_fused_attention,
    )

    return {
        "resolvedTarget": target,
        "layerCount": len(getattr(state.model, "layers", [])),
        "config": {
            "numHiddenLayers": (state.config or {}).get("num_hidden_layers"),
            "numAttentionHeads": (state.config or {}).get("num_attention_heads"),
            "hiddenSize": (state.config or {}).get("hidden_size"),
        },
        "cacheStrategy": state.cache_strategy,
        "cacheBits": state.cache_bits,
        "fp16Layers": state.fp16_layers,
        "fusedAttention": state.fused_attention,
        "speculativeDecoding": bool(state.speculative_decoding and state._dflash_generator is not None),
        "dflashDraftModel": (
            str(dflash_draft_model)
            if state.speculative_decoding and state._dflash_generator is not None and dflash_draft_model
            else None
        ),
        "treeBudget": state.tree_budget if state.speculative_decoding and state._dflash_generator is not None else 0,
        "note": _merge_runtime_notes(profile_note, dflash_note),
    }


def unload_model(state: WorkerState) -> dict[str, Any]:
    state.model = None
    state.tokenizer = None
    state.processor = None
    state.is_multimodal = False
    state._loaded_model_ref = None
    state._dflash_generator = None
    state._dflash_target = None
    state._ddtree_draft = None
    state._ddtree_target = None
    state.speculative_decoding = False
    state.tree_budget = 0
    state.config = None
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass
    return {"unloaded": True}


def update_profile(state: WorkerState, request: dict[str, Any]) -> dict[str, Any]:
    if state.model is None or state.tokenizer is None:
        raise RuntimeError("No MLX model is loaded.")
    note = state._apply_cache_profile(
        cache_strategy=str(request.get("cacheStrategy", state.cache_strategy)),
        cache_bits=int(request.get("cacheBits", state.cache_bits)),
        fp16_layers=int(request.get("fp16Layers", state.fp16_layers)),
        fused_attention=bool(request.get("fusedAttention", state.fused_attention)),
    )
    return {
        "cacheStrategy": state.cache_strategy,
        "cacheBits": state.cache_bits,
        "fp16Layers": state.fp16_layers,
        "fusedAttention": state.fused_attention,
        "note": note,
    }


def apply_cache_profile(
    state: WorkerState,
    *,
    cache_strategy: str,
    cache_bits: int,
    fp16_layers: int,
    fused_attention: bool,
) -> str | None:
    state.cache_strategy = cache_strategy
    state.cache_bits = cache_bits
    state.fp16_layers = fp16_layers
    state.fused_attention = fused_attention

    if state.cache_strategy == "native":
        state.cache_bits = 0
        state.fp16_layers = 0
        return None

    # FU-002: TriAttention MLX path. Doesn't make a prompt_cache
    # object — instead applies the compressor in-place to the loaded
    # model so subsequent ``mlx_lm.generate`` calls run against the
    # wrapped attention. Falls back to native on any failure (model
    # missing, triattention unavailable, apply raises).
    if state.cache_strategy == "triattention":
        return state._apply_triattention_mlx_compressor()

    preview_cache, note = state._make_cache()
    if preview_cache is not None:
        preview_cache = None
        import gc
        gc.collect()

    if note:
        state.cache_strategy = "native"
        state.cache_bits = 0
        state.fp16_layers = 0

    return note


def apply_triattention_mlx_compressor(state: WorkerState) -> str | None:
    """Apply ``apply_triattention_mlx`` to the loaded model in-place.

    Returns a runtimeNote describing what happened. On any failure
    the worker falls back to the native cache so generation keeps
    working without TriAttention.
    """
    if state.model is None:
        state.cache_strategy = "native"
        state.cache_bits = 0
        state.fp16_layers = 0
        return "TriAttention requested but no model is loaded; using native cache."
    try:
        from cache_compression import registry
    except Exception as exc:
        state.cache_strategy = "native"
        return f"TriAttention failed to import strategy registry ({exc}); using native cache."
    strategy = registry.get("triattention")
    if strategy is None or not strategy.is_available():
        state.cache_strategy = "native"
        return (
            "TriAttention is not available in this runtime "
            "(install ``triattention`` + ``mlx_lm``); using native cache."
        )
    try:
        apply_compressor = getattr(strategy, "apply_mlx_compressor", None)
        if apply_compressor is None:
            raise AttributeError("strategy.apply_mlx_compressor missing")
        apply_compressor(state.model, kv_budget=state.kv_budget)
    except Exception as exc:
        state.cache_strategy = "native"
        return (
            f"TriAttention apply_mlx_compressor raised "
            f"({type(exc).__name__}: {exc}); using native cache."
        )
    return f"TriAttention MLX compressor applied (kv_budget={state.kv_budget})."
