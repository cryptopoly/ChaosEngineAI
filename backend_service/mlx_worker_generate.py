"""Plain text generation paths for the MLX worker.

Three helpers lifted out of ``WorkerState``:

* ``generate`` — dispatch entrypoint. Routes to multimodal /
  speculative / standard paths based on loaded model + request flags.
* ``generate_standard`` — synchronous mlx_lm.stream_generate against
  the loaded model with cache profile + sampler + RunawayGuard.
* ``stream_generate`` — chunk-by-chunk streaming variant, emits SSE-
  shaped chunks via ``_emit``. Multimodal short-circuit + speculative
  fallback to standard on cache failure.

All three take ``state: WorkerState`` as the first argument.

Extracted from ``backend_service/mlx_worker.py`` as part of the
v0.8.0 Phase 1f-12 refactor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend_service.mlx_worker_io import _emit
from backend_service.mlx_worker_multimodal import (
    generate_multimodal,
    stream_generate_multimodal,
)
from backend_service.mlx_worker_prompt import (
    _build_prompt_text,
    _merge_runtime_notes,
    _plain_chat_fallback_active,
    _should_retry_cache_failure,
    _trim_transcript_continuation,
)
from backend_service.mlx_worker_request import (
    _apply_mlx_seed,
    _build_mlx_logits_processors,
    _build_mlx_sampler,
    _extract_top_logprobs,
    _format_tools_for_prompt,
    _sampler_seed,
)
from backend_service.reasoning_split import (
    RAW_REASONING_HEADING_RE,
    ThinkingTokenFilter,
    reasoning_delimiters_for,
    strip_harmony_boilerplate,
)
from backend_service.runaway_guard import RunawayGuard
from backend_service import mlx_worker_prompt_cache as _prompt_cache


if TYPE_CHECKING:
    from backend_service.mlx_worker import WorkerState


def generate(state: WorkerState, request: dict[str, Any]) -> dict[str, Any]:
    if state.model is None or state.tokenizer is None:
        raise RuntimeError("No MLX model is loaded.")

    # Multimodal short-circuit: vision-capable models loaded via
    # mlx_vlm always route through the multimodal generate path,
    # whether or not the request carries an ``images`` field
    # (mlx_vlm.generate accepts ``image=None`` for text-only turns).
    # DFlash speculative decoding doesn't apply on the VLM branch
    # because the draft-model registry doesn't ship multimodal drafts.
    if state.is_multimodal:
        return state._generate_multimodal(request)

    # Apply caller-supplied seed before any sampler runs — speculative
    # paths sample inside their own helpers, so seed must be set
    # up-front and not just in ``_generate_standard``.
    _apply_mlx_seed(request)

    # Use DDTree if tree budget is set and components are loaded
    if state.speculative_decoding and state.tree_budget > 0 and state._ddtree_draft is not None:
        try:
            return state._generate_ddtree(request)
        except Exception as exc:
            runtime_fallback_note = f"DDTree generation failed ({exc}). Falling back to linear DFLASH."
            # Fall through to linear DFLASH below

    # Use DFLASH if active
    if state.speculative_decoding and state._dflash_generator is not None:
        try:
            return state._generate_dflash(request)
        except Exception as exc:
            # Fall back to standard generation on DFLASH failure
            runtime_fallback_note = f"DFLASH generation failed ({exc}). Fell back to standard generation."
            result = state._generate_standard(request)
            result["runtimeNote"] = _merge_runtime_notes(result.get("runtimeNote"), runtime_fallback_note)
            return result

    return state._generate_standard(request)


def generate_standard(state: WorkerState, request: dict[str, Any]) -> dict[str, Any]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    # Inject tool schemas into system prompt for open-source models
    system_prompt = request.get("systemPrompt")
    tools_block = _format_tools_for_prompt(request.get("tools"))
    if tools_block:
        system_prompt = (tools_block + "\n\n" + (system_prompt or "")).strip()

    prompt_text, prompt_note = _build_prompt_text(
        state.tokenizer,
        history=list(request.get("history") or []),
        prompt=str(request.get("prompt") or ""),
        system_prompt=system_prompt,
    )
    sampler = _build_mlx_sampler(request)
    acq = _prompt_cache.acquire(state, prompt_text)
    prompt_cache = acq.cache
    prompt_feed = acq.prompt_feed
    managed = acq.managed
    runtime_note = _merge_runtime_notes(acq.note, prompt_note)
    runtime_fields = state._runtime_fields(prompt_cache=acq.fields_cache)
    transcript_fallback = _plain_chat_fallback_active(prompt_note)

    runaway_guard = RunawayGuard()
    runaway_stopped = False
    generated_ids: list[int] = []
    try:
        text_parts: list[str] = []
        last_response = None
        for response in stream_generate(
            state.model,
            state.tokenizer,
            prompt_feed,
                max_tokens=int(request.get("maxTokens") or 256),
                sampler=sampler,
                logits_processors=_build_mlx_logits_processors(request),
                prompt_cache=prompt_cache,
        ):
            _tok = getattr(response, "token", None)
            if isinstance(_tok, int):
                generated_ids.append(_tok)
            if response.text:
                text_parts.append(response.text)
                try:
                    runaway_guard.feed(response.text)
                except RuntimeError:
                    runaway_stopped = True
                    break
            last_response = response
        if managed:
            _prompt_cache.commit(
                state,
                cache=prompt_cache,
                commit_tokens=acq.commit_tokens,
                generated_ids=generated_ids,
                model_ref=state._loaded_model_ref,
            )
    except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
        was_managed = managed
        if managed:
            _prompt_cache.invalidate(state)
            managed = False
        _should_retry = was_managed or (
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
            runtime_fields = state._runtime_fields(prompt_cache=None)
            runaway_guard = RunawayGuard()
            runaway_stopped = False
            text_parts = []
            last_response = None
            for response in stream_generate(
                state.model,
                state.tokenizer,
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
    _open_tag, _close_tag = reasoning_delimiters_for(state._loaded_model_ref)
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


def stream_generate(state: WorkerState, request: dict[str, Any]) -> None:
    if state.model is None or state.tokenizer is None:
        raise RuntimeError("No MLX model is loaded.")

    # Multimodal short-circuit (see ``generate`` for context). The
    # streaming variant emits chunks via ``_emit`` so the caller
    # protocol matches the text-only path exactly.
    if state.is_multimodal:
        state._stream_generate_multimodal(request)
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
    if state.speculative_decoding and state.tree_budget > 0 and state._ddtree_draft is not None:
        try:
            result = state._generate_ddtree(request)
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

    if state.speculative_decoding and state._dflash_generator is not None:
        try:
            result = state._generate_dflash(request)
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
        state.tokenizer,
        history=list(request.get("history") or []),
        prompt=str(request.get("prompt") or ""),
        system_prompt=system_prompt,
    )
    sampler = _build_mlx_sampler(request)
    acq = _prompt_cache.acquire(state, prompt_text)
    prompt_cache = acq.cache
    prompt_feed = acq.prompt_feed
    managed = acq.managed
    runtime_note = _merge_runtime_notes(acq.note, prompt_note)
    runtime_note = _merge_runtime_notes(runtime_note, speculative_stream_fallback_note)
    runtime_fields = state._runtime_fields(prompt_cache=acq.fields_cache)
    transcript_fallback = _plain_chat_fallback_active(prompt_note)

    thinking_mode = request.get("thinkingMode") or "off"
    _open_tag, _close_tag = reasoning_delimiters_for(state._loaded_model_ref)
    think_filter = ThinkingTokenFilter(
        detect_raw_reasoning=(thinking_mode != "off"),
        open_tag=_open_tag,
        close_tag=_close_tag,
    )
    transcript_filter = TranscriptLoopFilter() if transcript_fallback else None
    transcript_trimmed = False
    runaway_guard = RunawayGuard()
    runaway_stopped = False
    generated_ids: list[int] = []
    # Phase 3.3 follow-up: when the request opted into logprobs,
    # extract top-k per token via the helper and forward inline
    # with each text chunk.
    logprobs_top_k = int(request.get("logprobs") or 0)

    try:
        last_response = None
        for response in mlx_stream_generate(
            state.model,
            state.tokenizer,
            prompt_feed,
            max_tokens=int(request.get("maxTokens") or 256),
            sampler=sampler,
            logits_processors=_build_mlx_logits_processors(request),
            prompt_cache=prompt_cache,
        ):
            _tok = getattr(response, "token", None)
            if isinstance(_tok, int):
                generated_ids.append(_tok)
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
                        entries = _extract_top_logprobs(response, state.tokenizer, logprobs_top_k)
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
        if managed:
            _prompt_cache.commit(
                state,
                cache=prompt_cache,
                commit_tokens=acq.commit_tokens,
                generated_ids=generated_ids,
                model_ref=state._loaded_model_ref,
            )
    except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
        was_managed = managed
        if managed:
            _prompt_cache.invalidate(state)
            managed = False
        _should_retry = was_managed or (
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
            runtime_fields = state._runtime_fields(prompt_cache=None)
            _open_tag, _close_tag = reasoning_delimiters_for(state._loaded_model_ref)
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
                state.model,
                state.tokenizer,
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
