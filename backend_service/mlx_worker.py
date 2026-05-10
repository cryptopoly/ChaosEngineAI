from __future__ import annotations

import json
import os
import re
import sys
import tempfile
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
    vlm_generate_kwargs,
)
from backend_service.mlx_worker_cache import (
    make_mlx_cache,
    runtime_fields,
)

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
        from mlx_lm import load

        target = str(request["target"])
        requested_cache_strategy = str(request.get("cacheStrategy", "native"))
        requested_cache_bits = int(request.get("cacheBits", 0))
        requested_fp16_layers = int(request.get("fp16Layers", 0))
        requested_fused_attention = bool(request.get("fusedAttention", False))
        # FU-002: kv_budget for the TriAttention MLX compressor. Ignored
        # when cache_strategy != "triattention". Falls back to 2048 (the
        # upstream default validated by scripts/spike_triattention_mlx.py).
        self.kv_budget = max(64, int(request.get("kvBudget", 2048)))
        self.context_tokens = int(request.get("contextTokens", 8192))
        self.speculative_decoding = bool(request.get("speculativeDecoding", False))
        dflash_draft_model = request.get("dflashDraftModel")
        self._dflash_generator = None
        self._dflash_target = None
        self._ddtree_draft = None
        self._ddtree_target = None
        self.tree_budget = 0

        emit_progress("resolving", 5.0, f"Resolving model target: {target}")

        # Pre-resolve the snapshot so we can stream download progress. Skip if
        # `target` is already a local path, and fall back to letting mlx_lm.load
        # handle non-HF targets on any failure.
        local_path = target
        is_local = False
        try:
            candidate = Path(target).expanduser()
            if target.startswith("/") or target.startswith("~") or candidate.exists():
                is_local = True
                local_path = str(candidate)
        except Exception:
            is_local = False

        if not is_local:
            try:
                from huggingface_hub import snapshot_download  # type: ignore
                from huggingface_hub.utils import (  # type: ignore
                    GatedRepoError,
                    RepositoryNotFoundError,
                    HfHubHTTPError,
                )
                from tqdm import tqdm  # type: ignore
            except ImportError:
                # huggingface_hub / tqdm not installed — let mlx_lm.load
                # handle resolution itself. Matches pre-progress behaviour.
                local_path = target
            else:
                class ProgressTqdm(tqdm):  # type: ignore[misc]
                    def update(self, n: int = 1):  # type: ignore[override]
                        result = super().update(n)
                        try:
                            total = float(self.total or 0)
                            done = float(self.n or 0)
                            if total > 0:
                                frac = max(0.0, min(1.0, done / total))
                                pct = 20.0 + frac * 40.0  # 20% -> 60%
                                done_mb = int(done // (1024 * 1024))
                                total_mb = int(total // (1024 * 1024))
                                emit_progress(
                                    "downloading",
                                    pct,
                                    f"{done_mb} / {total_mb} MB",
                                )
                            else:
                                emit_progress("downloading", 20.0, "Fetching weights")
                        except Exception:
                            pass
                        return result

                emit_progress("downloading", 20.0, "Fetching weights from Hugging Face")
                try:
                    # Use max_workers=1 to avoid multiprocessing semaphore
                    # leaks on macOS that crash the worker subprocess.
                    local_path = snapshot_download(
                        repo_id=target,
                        tqdm_class=ProgressTqdm,
                        max_workers=1,
                    )
                except GatedRepoError as exc:
                    raise RuntimeError(
                        f"This model is gated on Hugging Face. Accept the licence "
                        f"at https://huggingface.co/{target} and set HF_TOKEN in "
                        f"Settings, then retry."
                    ) from exc
                except RepositoryNotFoundError as exc:
                    raise RuntimeError(
                        f"Hugging Face repository not found: {target}"
                    ) from exc
                except HfHubHTTPError as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status in (401, 403):
                        raise RuntimeError(
                            f"Hugging Face refused access to {target} (HTTP {status}). "
                            f"Set HF_TOKEN in Settings and make sure you have accepted "
                            f"the licence at https://huggingface.co/{target}."
                        ) from exc
                    raise RuntimeError(
                        f"Hugging Face download failed for {target}: {exc}"
                    ) from exc
                except OSError as exc:
                    # Network / filesystem failures — bubble up the detail.
                    raise RuntimeError(
                        f"Could not download {target} from Hugging Face: {exc}"
                    ) from exc

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
        # reads ``self.tokenizer`` keeps working. When the multimodal
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
                self.model, self.processor = mlx_vlm_load(local_path)
                self.tokenizer = getattr(self.processor, "tokenizer", None)
                # mlx_vlm.load doesn't return a config dict — read it from
                # the snapshot directly so prompt-formatter + chat-template
                # paths can still introspect (e.g. ``num_attention_heads``
                # for cache estimation).
                config_path = Path(local_path) / "config.json"
                if config_path.exists():
                    try:
                        self.config = json.loads(config_path.read_text())
                    except Exception:
                        self.config = {}
                else:
                    self.config = {}
                self.is_multimodal = True
            else:
                self.model, self.tokenizer, self.config = load(local_path, return_config=True)
                self.processor = None
                self.is_multimodal = False
            self._loaded_model_ref = target
        finally:
            load_done.set()
            heartbeat_thread.join(timeout=0.5)
        emit_progress("ready", 95.0, "Finalising")

        # Initialise DFLASH speculative decoding if requested
        dflash_note = None
        self.tree_budget = int(request.get("treeBudget") or 0)
        if self.speculative_decoding and dflash_draft_model:
            try:
                from dflash_mlx.runtime import configure_full_attention_split, load_draft_bundle
                emit_progress("dflash", 96.0, f"Loading DFLASH draft model: {dflash_draft_model}")
                # Reuse the already loaded MLX target model. Loading a second
                # target bundle can duplicate the full model footprint and
                # trigger SIGKILL on large models during DFLASH startup.
                self._dflash_target = self.model
                configure_full_attention_split(self._dflash_target, enabled=True)
                self._dflash_generator, _ = load_draft_bundle(dflash_draft_model, lazy=True)
                dflash_note = f"DFLASH speculative decoding active (draft: {dflash_draft_model})."
            except ImportError as exc:
                dflash_note = f"dflash-mlx could not be imported ({exc}). Falling back to standard generation."
                self.speculative_decoding = False
            except Exception as exc:
                dflash_note = f"DFLASH initialisation failed: {exc}. Falling back to standard generation."
                self.speculative_decoding = False

            # Load DDTree components when tree budget is set
            if self.speculative_decoding and self.tree_budget > 0:
                try:
                    emit_progress("ddtree", 97.0, "Preparing DDTree runtime")
                    self._ddtree_target = self._dflash_target
                    self._ddtree_draft = self._dflash_generator
                    dflash_note = f"DDTree speculative decoding active (budget={self.tree_budget}, draft: {dflash_draft_model})."
                except Exception as exc:
                    dflash_note = f"DDTree init failed ({exc}). Using linear DFLASH."
                    self.tree_budget = 0
                    self._ddtree_draft = None
                    self._ddtree_target = None

        profile_note = self._apply_cache_profile(
            cache_strategy=requested_cache_strategy,
            cache_bits=requested_cache_bits,
            fp16_layers=requested_fp16_layers,
            fused_attention=requested_fused_attention,
        )

        return {
            "resolvedTarget": target,
            "layerCount": len(getattr(self.model, "layers", [])),
            "config": {
                "numHiddenLayers": (self.config or {}).get("num_hidden_layers"),
                "numAttentionHeads": (self.config or {}).get("num_attention_heads"),
                "hiddenSize": (self.config or {}).get("hidden_size"),
            },
            "cacheStrategy": self.cache_strategy,
            "cacheBits": self.cache_bits,
            "fp16Layers": self.fp16_layers,
            "fusedAttention": self.fused_attention,
            "speculativeDecoding": bool(self.speculative_decoding and self._dflash_generator is not None),
            "dflashDraftModel": (
                str(dflash_draft_model)
                if self.speculative_decoding and self._dflash_generator is not None and dflash_draft_model
                else None
            ),
            "treeBudget": self.tree_budget if self.speculative_decoding and self._dflash_generator is not None else 0,
            "note": _merge_runtime_notes(profile_note, dflash_note),
        }

    def unload_model(self) -> dict[str, Any]:
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_multimodal = False
        self._loaded_model_ref = None
        self._dflash_generator = None
        self._dflash_target = None
        self._ddtree_draft = None
        self._ddtree_target = None
        self.speculative_decoding = False
        self.tree_budget = 0
        self.config = None
        import gc
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass
        return {"unloaded": True}

    def update_profile(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")
        note = self._apply_cache_profile(
            cache_strategy=str(request.get("cacheStrategy", self.cache_strategy)),
            cache_bits=int(request.get("cacheBits", self.cache_bits)),
            fp16_layers=int(request.get("fp16Layers", self.fp16_layers)),
            fused_attention=bool(request.get("fusedAttention", self.fused_attention)),
        )
        return {
            "cacheStrategy": self.cache_strategy,
            "cacheBits": self.cache_bits,
            "fp16Layers": self.fp16_layers,
            "fusedAttention": self.fused_attention,
            "note": note,
        }

    def _apply_cache_profile(
        self,
        *,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
    ) -> str | None:
        self.cache_strategy = cache_strategy
        self.cache_bits = cache_bits
        self.fp16_layers = fp16_layers
        self.fused_attention = fused_attention

        if self.cache_strategy == "native":
            self.cache_bits = 0
            self.fp16_layers = 0
            return None

        # FU-002: TriAttention MLX path. Doesn't make a prompt_cache
        # object — instead applies the compressor in-place to the loaded
        # model so subsequent ``mlx_lm.generate`` calls run against the
        # wrapped attention. Falls back to native on any failure (model
        # missing, triattention unavailable, apply raises).
        if self.cache_strategy == "triattention":
            return self._apply_triattention_mlx_compressor()

        preview_cache, note = self._make_cache()
        if preview_cache is not None:
            preview_cache = None
            import gc
            gc.collect()

        if note:
            self.cache_strategy = "native"
            self.cache_bits = 0
            self.fp16_layers = 0

        return note

    def _apply_triattention_mlx_compressor(self) -> str | None:
        """Apply ``apply_triattention_mlx`` to the loaded model in-place.

        Returns a runtimeNote describing what happened. On any failure
        the worker falls back to the native cache so generation keeps
        working without TriAttention.
        """
        if self.model is None:
            self.cache_strategy = "native"
            self.cache_bits = 0
            self.fp16_layers = 0
            return "TriAttention requested but no model is loaded; using native cache."
        try:
            from cache_compression import registry
        except Exception as exc:
            self.cache_strategy = "native"
            return f"TriAttention failed to import strategy registry ({exc}); using native cache."
        strategy = registry.get("triattention")
        if strategy is None or not strategy.is_available():
            self.cache_strategy = "native"
            return (
                "TriAttention is not available in this runtime "
                "(install ``triattention`` + ``mlx_lm``); using native cache."
            )
        try:
            apply_compressor = getattr(strategy, "apply_mlx_compressor", None)
            if apply_compressor is None:
                raise AttributeError("strategy.apply_mlx_compressor missing")
            apply_compressor(self.model, kv_budget=self.kv_budget)
        except Exception as exc:
            self.cache_strategy = "native"
            return (
                f"TriAttention apply_mlx_compressor raised "
                f"({type(exc).__name__}: {exc}); using native cache."
            )
        return f"TriAttention MLX compressor applied (kv_budget={self.kv_budget})."

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
        """Generate using DFLASH speculative decoding."""
        from dflash_mlx.runtime import stream_dflash_generate

        # Build prompt text
        system_prompt = request.get("systemPrompt")
        tools_block = _format_tools_for_prompt(request.get("tools"))
        if tools_block:
            system_prompt = (tools_block + "\n\n" + (system_prompt or "")).strip()

        prompt_text, prompt_note = _build_prompt_text(
            self.tokenizer,
            history=list(request.get("history") or []),
            prompt=str(request.get("prompt") or ""),
            system_prompt=system_prompt,
        )

        prompt_tokens = self.tokenizer.encode(prompt_text)
        eos_token_ids = list(getattr(self.tokenizer, "eos_token_ids", None) or [])
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and int(eos_token_id) not in eos_token_ids:
            eos_token_ids.append(int(eos_token_id))

        # ``stream_dflash_generate`` (upstream v0.1.4) yields per-token events
        # followed by a final ``{"event": "summary", ...}`` payload whose shape
        # matches what the old ``generate_dflash_once`` helper returned.
        summary: dict[str, Any] = {}
        # Phase 3.1: per-token accepted-from-draft tracking. Tokens that
        # share `cycles_completed` with the previous token are commits
        # from the same DDTree cycle — the first is verifier-decoded,
        # the rest are draft-accepted. Build a parallel list of
        # (token_text, accepted: bool) so the UI can tint accepted runs.
        per_token_accepted: list[bool] = []
        per_token_text: list[str] = []
        prev_cycle: int = -1
        prev_gen_count: int = 0
        for event in stream_dflash_generate(
            target_model=self._dflash_target or self.model,
            tokenizer=self.tokenizer,
            draft_model=self._dflash_generator,
            prompt=prompt_text,
            max_new_tokens=int(request.get("maxTokens") or 256),
            use_chat_template=False,
            stop_token_ids=eos_token_ids,
            prompt_tokens_override=prompt_tokens,
        ):
            if event.get("event") == "summary":
                summary = dict(event)
                continue
            if event.get("event") != "token":
                continue
            cycle = int(event.get("cycles_completed") or 0)
            gen_count = int(event.get("generated_tokens") or 0)
            token_id = event.get("token_id")
            if token_id is None:
                continue
            # First token of a new cycle (cycle increments) is
            # verifier-decoded; subsequent tokens within the same
            # cycle are draft-accepted. Cycle 0 (the initial seed
            # token) is also verifier-decoded.
            if gen_count <= prev_gen_count:
                # Defensive — skip duplicates / out-of-order events.
                continue
            accepted = cycle == prev_cycle and prev_cycle > 0
            per_token_accepted.append(accepted)
            try:
                per_token_text.append(self.tokenizer.decode([int(token_id)]))
            except Exception:
                per_token_text.append("")
            prev_cycle = cycle
            prev_gen_count = gen_count

        gen_tokens = [int(token_id) for token_id in summary.get("generated_token_ids", [])]
        text = self.tokenizer.decode(gen_tokens).strip() if gen_tokens else ""
        # Respect thinkingMode: only strip raw reasoning patterns when thinking
        # is enabled. XML <think> tags are always processed regardless.
        thinking_mode = request.get("thinkingMode") or "off"
        if text:
            _open_tag, _close_tag = reasoning_delimiters_for(self._loaded_model_ref)
            think_filter = ThinkingTokenFilter(
                detect_raw_reasoning=(thinking_mode != "off"),
                open_tag=_open_tag,
                close_tag=_close_tag,
            )
            result = think_filter.feed(text)
            flushed = think_filter.flush()
            text = strip_harmony_boilerplate(f"{result.text}{flushed.text}".strip())
        if not text:
            text = "Generation completed without decoded text."

        output_tokens = int(summary.get("generation_tokens") or len(gen_tokens))
        prompt_token_count = int(summary.get("prompt_token_count") or len(prompt_tokens))
        elapsed = max(float(summary.get("elapsed_us") or 0.0) / 1e6, 1e-6)
        phase_timings = dict(summary.get("phase_timings_us") or {})
        prefill_elapsed = max(0.0, float(phase_timings.get("prefill") or 0.0) / 1e6)
        generation_elapsed = max(elapsed - prefill_elapsed, 1e-6)
        tok_s = round(output_tokens / generation_elapsed, 1) if output_tokens else 0.0
        cycles_completed = int(summary.get("cycles_completed") or 0)
        accepted_from_draft = int(summary.get("accepted_from_draft") or 0)
        acceptance_rate = (
            accepted_from_draft / cycles_completed
            if cycles_completed > 0
            else None
        )

        runtime_note = _merge_runtime_notes(
            prompt_note,
            (
                f"DFLASH speculative decoding. Acceptance rate: {acceptance_rate:.1f} avg tokens."
                if acceptance_rate is not None
                else "DFLASH speculative decoding."
            ),
        )

        # Phase 3.1: build run-length-encoded accepted spans from the
        # per-token accepted bools. Each span has start (char offset
        # into the rendered text), length (chars), and accepted (bool).
        accepted_spans: list[dict[str, Any]] = []
        if per_token_accepted and per_token_text:
            offset = 0
            run_start = 0
            run_kind = per_token_accepted[0]
            for idx, accepted in enumerate(per_token_accepted):
                tok_text = per_token_text[idx] if idx < len(per_token_text) else ""
                if accepted != run_kind:
                    accepted_spans.append({
                        "start": run_start,
                        "length": offset - run_start,
                        "accepted": run_kind,
                    })
                    run_start = offset
                    run_kind = accepted
                offset += len(tok_text)
            accepted_spans.append({
                "start": run_start,
                "length": offset - run_start,
                "accepted": run_kind,
            })

        return {
            "text": text,
            "finishReason": "stop",
            "promptTokens": prompt_token_count,
            "completionTokens": output_tokens,
            "totalTokens": prompt_token_count + output_tokens,
            "tokS": tok_s,
            "promptTokS": 0.0,
            "peakMemoryGb": round(float(summary.get("peak_memory_gb") or 0.0), 3),
            "runtimeNote": runtime_note,
            "dflashAcceptanceRate": round(float(acceptance_rate), 2) if acceptance_rate is not None else None,
            "acceptedSpans": accepted_spans,
            "acceptedTokenText": "".join(per_token_text) if per_token_text else None,
            **self._runtime_fields(prompt_cache=None, speculative_decoding=True, tree_budget=0),
        }

    def _generate_ddtree(self, request: dict[str, Any]) -> dict[str, Any]:
        """Generate using DDTree tree-based speculative decoding."""
        from backend_service.ddtree import generate_ddtree_mlx

        system_prompt = request.get("systemPrompt")
        tools_block = _format_tools_for_prompt(request.get("tools"))
        if tools_block:
            system_prompt = (tools_block + "\n\n" + (system_prompt or "")).strip()

        prompt_text, prompt_note = _build_prompt_text(
            self.tokenizer,
            history=list(request.get("history") or []),
            prompt=str(request.get("prompt") or ""),
            system_prompt=system_prompt,
        )

        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt_text)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        stop_ids = [eos] if eos is not None else []

        result = generate_ddtree_mlx(
            target_model=self._ddtree_target,
            tokenizer=self.tokenizer,
            draft_model=self._ddtree_draft,
            prompt_tokens=prompt_tokens,
            max_new_tokens=int(request.get("maxTokens") or 256),
            tree_budget=self.tree_budget,
            stop_token_ids=stop_ids,
        )

        # Decode output tokens
        gen_tokens = result["generated_tokens"]
        text = self.tokenizer.decode(gen_tokens).strip()
        # Respect thinkingMode: only strip raw reasoning patterns when thinking
        # is enabled. XML <think> tags are always processed regardless.
        thinking_mode = request.get("thinkingMode") or "off"
        if text:
            _open_tag, _close_tag = reasoning_delimiters_for(self._loaded_model_ref)
            think_filter = ThinkingTokenFilter(
                detect_raw_reasoning=(thinking_mode != "off"),
                open_tag=_open_tag,
                close_tag=_close_tag,
            )
            filter_result = think_filter.feed(text)
            flushed = think_filter.flush()
            text = strip_harmony_boilerplate(f"{filter_result.text}{flushed.text}".strip())
        if not text:
            text = "Generation completed without decoded text."

        output_tokens = result["output_tokens"]
        elapsed = result["elapsed_seconds"]
        tok_s = round(output_tokens / max(elapsed, 1e-6), 1)
        acceptance_rate = result["avg_acceptance_length"]

        runtime_note = _merge_runtime_notes(
            prompt_note,
            f"DDTree speculative decoding (budget={result['tree_budget']}). Acceptance rate: {acceptance_rate:.1f} avg tokens."
            if acceptance_rate else f"DDTree speculative decoding (budget={result['tree_budget']}).",
        )

        return {
            "text": text,
            "finishReason": "stop",
            "promptTokens": len(prompt_tokens),
            "completionTokens": output_tokens,
            "totalTokens": len(prompt_tokens) + output_tokens,
            "tokS": tok_s,
            "promptTokS": 0.0,
            "peakMemoryGb": 0.0,
            "runtimeNote": runtime_note,
            "dflashAcceptanceRate": round(float(acceptance_rate), 2) if acceptance_rate else None,
            # Phase 3.1 follow-up: DDTree path now ships accepted-span
            # data alongside the linear DFLASH path so the frontend
            # AcceptedTokenOverlay tints draft-accepted ranges for
            # both speculative-decode strategies.
            "acceptedSpans": result.get("accepted_spans") or [],
            "acceptedTokenText": result.get("accepted_token_text"),
            **self._runtime_fields(
                prompt_cache=None,
                speculative_decoding=True,
                tree_budget=result["tree_budget"],
            ),
        }

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")

        # Multimodal short-circuit: vision-capable models loaded via
        # mlx_vlm always route through the multimodal generate path,
        # whether or not the request carries an ``images`` field
        # (mlx_vlm.generate accepts ``image=None`` for text-only turns).
        # DFlash speculative decoding doesn't apply on the VLM branch
        # because the draft-model registry doesn't ship multimodal drafts.
        if self.is_multimodal:
            return self._generate_multimodal(request)

        # Apply caller-supplied seed before any sampler runs — speculative
        # paths sample inside their own helpers, so seed must be set
        # up-front and not just in ``_generate_standard``.
        _apply_mlx_seed(request)

        # Use DDTree if tree budget is set and components are loaded
        if self.speculative_decoding and self.tree_budget > 0 and self._ddtree_draft is not None:
            try:
                return self._generate_ddtree(request)
            except Exception as exc:
                runtime_fallback_note = f"DDTree generation failed ({exc}). Falling back to linear DFLASH."
                # Fall through to linear DFLASH below

        # Use DFLASH if active
        if self.speculative_decoding and self._dflash_generator is not None:
            try:
                return self._generate_dflash(request)
            except Exception as exc:
                # Fall back to standard generation on DFLASH failure
                runtime_fallback_note = f"DFLASH generation failed ({exc}). Fell back to standard generation."
                result = self._generate_standard(request)
                result["runtimeNote"] = _merge_runtime_notes(result.get("runtimeNote"), runtime_fallback_note)
                return result

        return self._generate_standard(request)

    def _generate_standard(self, request: dict[str, Any]) -> dict[str, Any]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        # Inject tool schemas into system prompt for open-source models
        system_prompt = request.get("systemPrompt")
        tools_block = _format_tools_for_prompt(request.get("tools"))
        if tools_block:
            system_prompt = (tools_block + "\n\n" + (system_prompt or "")).strip()

        prompt_text, prompt_note = _build_prompt_text(
            self.tokenizer,
            history=list(request.get("history") or []),
            prompt=str(request.get("prompt") or ""),
            system_prompt=system_prompt,
        )
        sampler = _build_mlx_sampler(request)
        prompt_cache, runtime_note = self._make_cache()
        runtime_note = _merge_runtime_notes(runtime_note, prompt_note)
        runtime_fields = self._runtime_fields(prompt_cache=prompt_cache)
        transcript_fallback = _plain_chat_fallback_active(prompt_note)

        runaway_guard = RunawayGuard()
        runaway_stopped = False
        try:
            text_parts: list[str] = []
            last_response = None
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt_text,
                    max_tokens=int(request.get("maxTokens") or 256),
                    sampler=sampler,
                    prompt_cache=prompt_cache,
            ):
                if response.text:
                    text_parts.append(response.text)
                    try:
                        runaway_guard.feed(response.text)
                    except RuntimeError:
                        runaway_stopped = True
                        break
                last_response = response
        except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
            _should_retry = (
                prompt_cache is not None
                and _should_retry_cache_failure(exc)
            )
            if _should_retry:
                # Cache strategy produced incompatible shapes or mask errors.
                # Retry with the model's default (native) cache.
                runtime_note = (
                    _merge_runtime_notes(
                        prompt_note,
                        f"Cache strategy failed ({exc}). Fell back to native f16 cache.",
                    )
                )
                runtime_fields = self._runtime_fields(prompt_cache=None)
                runaway_guard = RunawayGuard()
                runaway_stopped = False
                text_parts = []
                last_response = None
                for response in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt_text,
                    max_tokens=int(request.get("maxTokens") or 256),
                    sampler=sampler,
                    prompt_cache=None,
                ):
                    if response.text:
                        text_parts.append(response.text)
                    last_response = response
            else:
                raise

        if last_response is None:
            raise RuntimeError("MLX generation did not return a response.")

        if runaway_stopped:
            runtime_note = _merge_runtime_notes(
                runtime_note,
                "Stopped runaway generation: model was repeating itself.",
            )

        raw_text = "".join(text_parts).strip()
        # Respect thinkingMode: only strip raw reasoning when thinking is on.
        thinking_mode = request.get("thinkingMode") or "off"
        _open_tag, _close_tag = reasoning_delimiters_for(self._loaded_model_ref)
        think_filter = ThinkingTokenFilter(
            detect_raw_reasoning=(thinking_mode != "off"),
            open_tag=_open_tag,
            close_tag=_close_tag,
        )
        filter_result = think_filter.feed(raw_text)
        flushed = think_filter.flush()
        text = strip_harmony_boilerplate(f"{filter_result.text}{flushed.text}".strip())
        if transcript_fallback:
            text, transcript_trimmed = _trim_transcript_continuation(text)
            if transcript_trimmed:
                runtime_note = _merge_runtime_notes(
                    runtime_note,
                    "Suppressed a plain-chat transcript continuation to stop a runaway loop.",
                )
        if not text:
            text = "Generation completed without decoded text."

        return {
            "text": text,
            "finishReason": last_response.finish_reason or "stop",
            "promptTokens": int(last_response.prompt_tokens),
            "completionTokens": int(last_response.generation_tokens),
            "totalTokens": int(last_response.prompt_tokens + last_response.generation_tokens),
            "tokS": round(float(last_response.generation_tps), 1),
            "promptTokS": round(float(last_response.prompt_tps), 1),
            "peakMemoryGb": round(float(last_response.peak_memory), 3),
            "runtimeNote": runtime_note,
            **runtime_fields,
        }

    # ------------------------------------------------------------------
    # Multimodal (vision-language) generation via mlx-vlm
    # ------------------------------------------------------------------

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
        """Synchronous mlx-vlm generation. Decodes any attached images,
        runs ``mlx_vlm.generate``, applies the thinking-token filter,
        and returns the same response shape as ``_generate_standard``.
        """
        try:
            from mlx_vlm import generate as vlm_generate  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                f"mlx-vlm is not installed but a multimodal model is loaded: {exc}. "
                "Install via ``pip install mlx-vlm``."
            ) from exc

        images_b64 = list(request.get("images") or [])
        _apply_mlx_seed(request)
        kwargs = self._vlm_generate_kwargs(request)

        with tempfile.TemporaryDirectory(prefix="chaosengine-mm-") as tmpdir:
            image_paths = self._decode_images_to_paths(images_b64, tmpdir)
            prompt_text = self._format_multimodal_prompt(request, num_images=len(image_paths))
            if image_paths:
                result = vlm_generate(
                    self.model, self.processor, prompt_text,
                    image=image_paths, **kwargs,
                )
            else:
                result = vlm_generate(
                    self.model, self.processor, prompt_text, **kwargs,
                )

        raw_text = getattr(result, "text", None) or str(result)
        thinking_mode = request.get("thinkingMode") or "off"
        _open_tag, _close_tag = reasoning_delimiters_for(self._loaded_model_ref)
        think_filter = ThinkingTokenFilter(
            detect_raw_reasoning=(thinking_mode != "off"),
            open_tag=_open_tag,
            close_tag=_close_tag,
        )
        filter_result = think_filter.feed(raw_text)
        flushed = think_filter.flush()
        text = strip_harmony_boilerplate(f"{filter_result.text}{flushed.text}".strip())
        if not text:
            text = "Generation completed without decoded text."

        runtime_note = (
            f"Multimodal generation via mlx-vlm "
            f"({len(image_paths)} image{'s' if len(image_paths) != 1 else ''})."
        )

        return {
            "text": text,
            "finishReason": getattr(result, "finish_reason", None) or "stop",
            "promptTokens": int(getattr(result, "prompt_tokens", 0) or 0),
            "completionTokens": int(getattr(result, "generation_tokens", 0) or 0),
            "totalTokens": int(
                (getattr(result, "prompt_tokens", 0) or 0)
                + (getattr(result, "generation_tokens", 0) or 0)
            ),
            "tokS": round(float(getattr(result, "generation_tps", 0.0) or 0.0), 1),
            "promptTokS": round(float(getattr(result, "prompt_tps", 0.0) or 0.0), 1),
            "peakMemoryGb": round(float(getattr(result, "peak_memory", 0.0) or 0.0), 3),
            "runtimeNote": runtime_note,
            "cacheStrategy": "native",
            "cacheBits": 0,
            "fp16Layers": 0,
            "fusedAttention": False,
            "speculativeDecoding": False,
        }

    def _stream_generate_multimodal(self, request: dict[str, Any]) -> None:
        """Streaming mlx-vlm generation. Emits chunks via the standard
        ``_emit`` protocol used by the text-only path so the caller
        sees the same shape regardless of which engine produced the run.
        """
        try:
            from mlx_vlm import stream_generate as vlm_stream  # type: ignore[import-untyped]
        except ImportError as exc:
            _emit({"error": (
                f"mlx-vlm is not installed but a multimodal model is loaded: {exc}. "
                "Install via ``pip install mlx-vlm``."
            )})
            return

        images_b64 = list(request.get("images") or [])
        _apply_mlx_seed(request)
        kwargs = self._vlm_generate_kwargs(request)
        thinking_mode = request.get("thinkingMode") or "off"
        _open_tag, _close_tag = reasoning_delimiters_for(self._loaded_model_ref)
        think_filter = ThinkingTokenFilter(
            detect_raw_reasoning=(thinking_mode != "off"),
            open_tag=_open_tag,
            close_tag=_close_tag,
        )

        text_parts: list[str] = []
        completion_tokens = 0
        last_chunk: Any = None

        with tempfile.TemporaryDirectory(prefix="chaosengine-mm-") as tmpdir:
            image_paths = self._decode_images_to_paths(images_b64, tmpdir)
            prompt_text = self._format_multimodal_prompt(request, num_images=len(image_paths))
            if image_paths:
                stream = vlm_stream(
                    self.model, self.processor, prompt_text,
                    image=image_paths, **kwargs,
                )
            else:
                stream = vlm_stream(
                    self.model, self.processor, prompt_text, **kwargs,
                )

            for chunk in stream:
                last_chunk = chunk
                chunk_text = chunk if isinstance(chunk, str) else (
                    getattr(chunk, "text", None) or ""
                )
                if not chunk_text:
                    continue
                text_parts.append(chunk_text)
                completion_tokens += 1
                filtered = think_filter.feed(chunk_text)
                if filtered.text:
                    _emit({"ok": True, "chunk": {"text": filtered.text}})

        flushed = think_filter.flush()
        if flushed.text:
            _emit({"ok": True, "chunk": {"text": flushed.text}})

        runtime_note = (
            f"Multimodal stream via mlx-vlm "
            f"({len(image_paths)} image{'s' if len(image_paths) != 1 else ''})."
        )
        _emit({
            "ok": True,
            "done": True,
            "result": {
                "finishReason": getattr(last_chunk, "finish_reason", None) or "stop",
                "promptTokens": int(getattr(last_chunk, "prompt_tokens", 0) or 0),
                "completionTokens": int(
                    getattr(last_chunk, "generation_tokens", 0) or completion_tokens
                ),
                "totalTokens": int(
                    (getattr(last_chunk, "prompt_tokens", 0) or 0)
                    + (getattr(last_chunk, "generation_tokens", 0) or completion_tokens)
                ),
                "tokS": round(float(getattr(last_chunk, "generation_tps", 0.0) or 0.0), 1),
                "promptTokS": round(float(getattr(last_chunk, "prompt_tps", 0.0) or 0.0), 1),
                "peakMemoryGb": round(float(getattr(last_chunk, "peak_memory", 0.0) or 0.0), 3),
                "runtimeNote": runtime_note,
                "cacheStrategy": "native",
                "cacheBits": 0,
                "fp16Layers": 0,
                "fusedAttention": False,
                "speculativeDecoding": False,
            },
        })


    def stream_generate(self, request: dict[str, Any]) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")

        # Multimodal short-circuit (see ``generate`` for context). The
        # streaming variant emits chunks via ``_emit`` so the caller
        # protocol matches the text-only path exactly.
        if self.is_multimodal:
            self._stream_generate_multimodal(request)
            return

        # Apply caller-supplied seed before any sampler runs — speculative
        # paths (DDTree / DFLASH) sample inside their own helpers, so the
        # seed must be set up-front, not just before the standard mlx-lm
        # path below.
        _apply_mlx_seed(request)

        speculative_stream_fallback_note = None
        # DFLASH/DDTree don't support token-level streaming natively, so
        # emit the full result as a single chunk in the streaming protocol.
        # Prefer DDTree (tree-based) when tree_budget > 0, else linear DFlash.
        if self.speculative_decoding and self.tree_budget > 0 and self._ddtree_draft is not None:
            try:
                result = self._generate_ddtree(request)
                if result.get("text"):
                    _emit({"ok": True, "chunk": {"text": result["text"]}})
                _emit({
                    "ok": True,
                    "done": True,
                    "result": {
                        "finishReason": result.get("finishReason", "stop"),
                        "promptTokens": result.get("promptTokens", 0),
                        "completionTokens": result.get("completionTokens", 0),
                        "totalTokens": result.get("totalTokens", 0),
                        "tokS": result.get("tokS", 0.0),
                        "promptTokS": result.get("promptTokS", 0.0),
                        "peakMemoryGb": result.get("peakMemoryGb", 0.0),
                        "runtimeNote": result.get("runtimeNote"),
                        "dflashAcceptanceRate": result.get("dflashAcceptanceRate"),
                        "cacheStrategy": result.get("cacheStrategy"),
                        "cacheBits": result.get("cacheBits"),
                        "fp16Layers": result.get("fp16Layers"),
                        "speculativeDecoding": result.get("speculativeDecoding"),
                        "treeBudget": result.get("treeBudget"),
                    },
                })
                return
            except Exception as exc:
                speculative_stream_fallback_note = (
                    f"DDTree stream path failed ({exc}). "
                    "Falling back to linear DFLASH."
                )
                # Fall through to linear DFLASH below

        if self.speculative_decoding and self._dflash_generator is not None:
            try:
                result = self._generate_dflash(request)
                if result.get("text"):
                    _emit({"ok": True, "chunk": {"text": result["text"]}})
                _emit({
                    "ok": True,
                    "done": True,
                    "result": {
                        "finishReason": result.get("finishReason", "stop"),
                        "promptTokens": result.get("promptTokens", 0),
                        "completionTokens": result.get("completionTokens", 0),
                        "totalTokens": result.get("totalTokens", 0),
                        "tokS": result.get("tokS", 0.0),
                        "promptTokS": result.get("promptTokS", 0.0),
                        "peakMemoryGb": result.get("peakMemoryGb", 0.0),
                        "runtimeNote": result.get("runtimeNote"),
                        "dflashAcceptanceRate": result.get("dflashAcceptanceRate"),
                        "cacheStrategy": result.get("cacheStrategy"),
                        "cacheBits": result.get("cacheBits"),
                        "fp16Layers": result.get("fp16Layers"),
                        "speculativeDecoding": result.get("speculativeDecoding"),
                        "treeBudget": result.get("treeBudget"),
                    },
                })
                return
            except Exception as exc:
                speculative_stream_fallback_note = (
                    f"Speculative decoding stream path failed ({exc}). "
                    "Fell back to standard generation."
                )

        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler

        # Inject tool schemas into system prompt for open-source models
        system_prompt = request.get("systemPrompt")
        tools_block = _format_tools_for_prompt(request.get("tools"))
        if tools_block:
            system_prompt = (tools_block + "\n\n" + (system_prompt or "")).strip()

        prompt_text, prompt_note = _build_prompt_text(
            self.tokenizer,
            history=list(request.get("history") or []),
            prompt=str(request.get("prompt") or ""),
            system_prompt=system_prompt,
        )
        sampler = _build_mlx_sampler(request)
        prompt_cache, runtime_note = self._make_cache()
        runtime_note = _merge_runtime_notes(runtime_note, prompt_note)
        runtime_note = _merge_runtime_notes(runtime_note, speculative_stream_fallback_note)
        runtime_fields = self._runtime_fields(prompt_cache=prompt_cache)
        transcript_fallback = _plain_chat_fallback_active(prompt_note)

        thinking_mode = request.get("thinkingMode") or "off"
        _open_tag, _close_tag = reasoning_delimiters_for(self._loaded_model_ref)
        think_filter = ThinkingTokenFilter(
            detect_raw_reasoning=(thinking_mode != "off"),
            open_tag=_open_tag,
            close_tag=_close_tag,
        )
        transcript_filter = TranscriptLoopFilter() if transcript_fallback else None
        transcript_trimmed = False
        runaway_guard = RunawayGuard()
        runaway_stopped = False
        # Phase 3.3 follow-up: when the request opted into logprobs,
        # extract top-k per token via the helper and forward inline
        # with each text chunk.
        logprobs_top_k = int(request.get("logprobs") or 0)

        try:
            last_response = None
            for response in mlx_stream_generate(
                self.model,
                self.tokenizer,
                prompt_text,
                max_tokens=int(request.get("maxTokens") or 256),
                sampler=sampler,
                prompt_cache=prompt_cache,
            ):
                if response.text:
                    # Check for runaway loops before emitting
                    try:
                        runaway_guard.feed(response.text)
                    except RuntimeError:
                        runaway_stopped = True
                        last_response = response
                        break
                    filtered = think_filter.feed(response.text)
                    if filtered.reasoning:
                        _emit({"ok": True, "chunk": {"reasoning": filtered.reasoning}})
                    if filtered.reasoning_done:
                        _emit({"ok": True, "chunk": {"reasoningDone": True}})
                    visible_text = filtered.text
                    if visible_text and transcript_filter is not None:
                        visible_text = transcript_filter.feed(visible_text)
                        if transcript_filter.stopped:
                            transcript_trimmed = True
                    if visible_text:
                        chunk_payload: dict[str, Any] = {"text": visible_text}
                        if logprobs_top_k > 0:
                            entries = _extract_top_logprobs(response, self.tokenizer, logprobs_top_k)
                            if entries:
                                chunk_payload["tokenLogprobs"] = entries
                        _emit({"ok": True, "chunk": chunk_payload})
                    if transcript_filter is not None and transcript_filter.stopped:
                        last_response = response
                        break
                last_response = response
            # Flush any remaining buffered text
            flushed = think_filter.flush()
            if flushed.reasoning:
                _emit({"ok": True, "chunk": {"reasoning": flushed.reasoning}})
            if flushed.reasoning_done:
                _emit({"ok": True, "chunk": {"reasoningDone": True}})
            visible_text = flushed.text
            if visible_text and transcript_filter is not None:
                visible_text = transcript_filter.feed(visible_text) + transcript_filter.flush()
                transcript_trimmed = transcript_trimmed or transcript_filter.stopped
            if visible_text:
                _emit({"ok": True, "chunk": {"text": visible_text}})
        except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
            _should_retry = (
                prompt_cache is not None
                and _should_retry_cache_failure(exc)
            )
            if _should_retry:
                runtime_note = (
                    _merge_runtime_notes(
                        prompt_note,
                        f"Cache strategy failed ({exc}). Fell back to native f16 cache.",
                    )
                )
                runtime_fields = self._runtime_fields(prompt_cache=None)
                _open_tag, _close_tag = reasoning_delimiters_for(self._loaded_model_ref)
                think_filter = ThinkingTokenFilter(
                    detect_raw_reasoning=(thinking_mode != "off"),
                    open_tag=_open_tag,
                    close_tag=_close_tag,
                )
                transcript_filter = TranscriptLoopFilter() if transcript_fallback else None
                transcript_trimmed = False
                runaway_guard = RunawayGuard()
                runaway_stopped = False
                last_response = None
                for response in mlx_stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt_text,
                    max_tokens=int(request.get("maxTokens") or 256),
                    sampler=sampler,
                    prompt_cache=None,
                ):
                    if response.text:
                        try:
                            runaway_guard.feed(response.text)
                        except RuntimeError:
                            runaway_stopped = True
                            last_response = response
                            break
                        filtered = think_filter.feed(response.text)
                        if filtered.reasoning:
                            _emit({"ok": True, "chunk": {"reasoning": filtered.reasoning}})
                        if filtered.reasoning_done:
                            _emit({"ok": True, "chunk": {"reasoningDone": True}})
                        visible_text = filtered.text
                        if visible_text and transcript_filter is not None:
                            visible_text = transcript_filter.feed(visible_text)
                            if transcript_filter.stopped:
                                transcript_trimmed = True
                        if visible_text:
                            _emit({"ok": True, "chunk": {"text": visible_text}})
                        if transcript_filter is not None and transcript_filter.stopped:
                            last_response = response
                            break
                    last_response = response
                flushed = think_filter.flush()
                if flushed.reasoning:
                    _emit({"ok": True, "chunk": {"reasoning": flushed.reasoning}})
                if flushed.reasoning_done:
                    _emit({"ok": True, "chunk": {"reasoningDone": True}})
                visible_text = flushed.text
                if visible_text and transcript_filter is not None:
                    visible_text = transcript_filter.feed(visible_text) + transcript_filter.flush()
                    transcript_trimmed = transcript_trimmed or transcript_filter.stopped
                if visible_text:
                    _emit({"ok": True, "chunk": {"text": visible_text}})
            else:
                raise

        if last_response is None:
            raise RuntimeError("MLX generation did not return a response.")

        if transcript_trimmed:
            runtime_note = _merge_runtime_notes(
                runtime_note,
                "Suppressed a plain-chat transcript continuation to stop a runaway loop.",
            )
        if runaway_stopped:
            runtime_note = _merge_runtime_notes(
                runtime_note,
                "Stopped runaway generation: model was repeating itself.",
            )

        _emit({
            "ok": True,
            "done": True,
            "result": {
                "finishReason": last_response.finish_reason or "stop",
                "promptTokens": int(last_response.prompt_tokens),
                "completionTokens": int(last_response.generation_tokens),
                "totalTokens": int(last_response.prompt_tokens + last_response.generation_tokens),
                "tokS": round(float(last_response.generation_tps), 1),
                "promptTokS": round(float(last_response.prompt_tps), 1),
                "peakMemoryGb": round(float(last_response.peak_memory), 3),
                "runtimeNote": runtime_note,
                **runtime_fields,
            },
        })


    def eval_perplexity(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")

        import math
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np

        dataset = request.get("dataset", "wikitext-2")
        num_samples = int(request.get("numSamples", 64))
        seq_length = int(request.get("seqLength", 512))
        batch_size = int(request.get("batchSize", 4))

        dataset_map = {
            "wikitext-2": "wikitext/wikitext-2-raw-v1",
        }
        data_path = dataset_map.get(dataset, dataset)

        emit_progress("loading_data", 10.0, "Loading evaluation dataset...")
        from mlx_lm.perplexity import load_data
        np.random.seed(123)
        data = load_data(self.tokenizer, data_path, num_samples, seq_length)

        emit_progress("evaluating", 20.0, f"Evaluating perplexity on {len(data)} samples...")
        start = time.monotonic()

        all_losses: list[mx.array] = []
        num_batches = (len(data) + batch_size - 1) // batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            logits = self.model(batch[:, :-1]).astype(mx.float32)
            losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
            mx.eval(losses)
            all_losses.append(losses.flatten())

            pct = 20.0 + (i / len(data)) * 70.0
            emit_progress("evaluating", pct, f"Batch {i // batch_size + 1}/{num_batches}")

        all_losses_cat = mx.concatenate(all_losses)
        mean_loss = all_losses_cat.mean().item()
        ppl = math.exp(mean_loss)
        std_dev = mx.sqrt(mx.var(all_losses_cat, ddof=1)).item()
        se_ppl = ppl * (std_dev / math.sqrt(all_losses_cat.size))

        elapsed = time.monotonic() - start
        tokens_eval = data.shape[0] * (data.shape[1] - 1)

        emit_progress("done", 100.0, f"Perplexity: {ppl:.2f}")
        return {
            "perplexity": round(ppl, 3),
            "standardError": round(se_ppl, 3),
            "evalSeconds": round(elapsed, 2),
            "evalTokensPerSecond": round(tokens_eval / elapsed, 1),
            "numSamples": len(data),
            "seqLength": seq_length,
            "dataset": dataset,
        }

    def eval_task_accuracy(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")

        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler
        from backend_service.task_datasets import load_task_data, score_answer

        task_name = request.get("taskName", "mmlu")
        limit = int(request.get("limit", 100))
        num_shots = int(request.get("numShots", 5))

        emit_progress("loading_tasks", 10.0, f"Loading {task_name} task data...")
        tasks = load_task_data(task_name, limit, num_shots)

        sampler = make_sampler(temp=0.0)  # greedy for accuracy
        correct = 0
        total = len(tasks)
        start = time.monotonic()

        for idx, task in enumerate(tasks):
            text_parts: list[str] = []
            for resp in mlx_stream_generate(
                self.model,
                self.tokenizer,
                task["prompt"],
                max_tokens=task.get("max_tokens", 3),
                sampler=sampler,
            ):
                if resp.text:
                    text_parts.append(resp.text)

            answer = "".join(text_parts).strip()
            if score_answer(task_name, answer, task["correct_answer"], task.get("choices")):
                correct += 1

            pct = 10.0 + ((idx + 1) / total) * 85.0
            emit_progress(
                "evaluating", pct,
                f"Question {idx + 1}/{total} — {correct}/{idx + 1} correct",
            )

        elapsed = time.monotonic() - start
        accuracy = round(correct / total, 4) if total > 0 else 0.0
        emit_progress("done", 100.0, f"Accuracy: {accuracy:.1%} ({correct}/{total})")
        return {
            "taskName": task_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "numShots": num_shots,
            "evalSeconds": round(elapsed, 2),
        }


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
