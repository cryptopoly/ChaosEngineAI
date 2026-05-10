"""Speculative decoding generation paths for the MLX worker.

Two helpers lifted out of ``WorkerState``:

* ``generate_dflash`` — DFLASH speculative decoding (linear). Streams
  per-token events from ``stream_dflash_generate`` (upstream
  dflash-mlx 0.1.4+), tracks per-token accepted-from-draft for the
  Phase 3.1 frontend overlay, applies the thinking-token filter,
  computes accepted spans, returns the response payload.
* ``generate_ddtree`` — DDTree tree-based speculative decoding. Calls
  ``generate_ddtree_mlx``, decodes output, applies the thinking-token
  filter, returns the response payload with acceptance rate.

Both take ``state: WorkerState`` as the first argument.

Extracted from ``backend_service/mlx_worker.py`` as part of the
v0.8.0 Phase 1f-11 refactor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend_service.mlx_worker_prompt import (
    _build_prompt_text,
    _merge_runtime_notes,
)
from backend_service.mlx_worker_request import _format_tools_for_prompt
from backend_service.reasoning_split import (
    ThinkingTokenFilter,
    reasoning_delimiters_for,
    strip_harmony_boilerplate,
)


if TYPE_CHECKING:
    from backend_service.mlx_worker import WorkerState


def generate_dflash(state: WorkerState, request: dict[str, Any]) -> dict[str, Any]:
    """Generate using DFLASH speculative decoding."""
    from dflash_mlx.runtime import stream_dflash_generate

    # Build prompt text
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

    prompt_tokens = state.tokenizer.encode(prompt_text)
    eos_token_ids = list(getattr(state.tokenizer, "eos_token_ids", None) or [])
    eos_token_id = getattr(state.tokenizer, "eos_token_id", None)
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
        target_model=state._dflash_target or state.model,
        tokenizer=state.tokenizer,
        draft_model=state._dflash_generator,
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
            per_token_text.append(state.tokenizer.decode([int(token_id)]))
        except Exception:
            per_token_text.append("")
        prev_cycle = cycle
        prev_gen_count = gen_count

    gen_tokens = [int(token_id) for token_id in summary.get("generated_token_ids", [])]
    text = state.tokenizer.decode(gen_tokens).strip() if gen_tokens else ""
    # Respect thinkingMode: only strip raw reasoning patterns when thinking
    # is enabled. XML <think> tags are always processed regardless.
    thinking_mode = request.get("thinkingMode") or "off"
    if text:
        _open_tag, _close_tag = reasoning_delimiters_for(state._loaded_model_ref)
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
        **state._runtime_fields(prompt_cache=None, speculative_decoding=True, tree_budget=0),
    }


def generate_ddtree(state: WorkerState, request: dict[str, Any]) -> dict[str, Any]:
    """Generate using DDTree tree-based speculative decoding."""
    from backend_service.ddtree import generate_ddtree_mlx

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

    # Tokenize prompt
    prompt_tokens = state.tokenizer.encode(prompt_text)
    eos = getattr(state.tokenizer, "eos_token_id", None)
    stop_ids = [eos] if eos is not None else []

    result = generate_ddtree_mlx(
        target_model=state._ddtree_target,
        tokenizer=state.tokenizer,
        draft_model=state._ddtree_draft,
        prompt_tokens=prompt_tokens,
        max_new_tokens=int(request.get("maxTokens") or 256),
        tree_budget=state.tree_budget,
        stop_token_ids=stop_ids,
    )

    # Decode output tokens
    gen_tokens = result["generated_tokens"]
    text = state.tokenizer.decode(gen_tokens).strip()
    # Respect thinkingMode: only strip raw reasoning patterns when thinking
    # is enabled. XML <think> tags are always processed regardless.
    thinking_mode = request.get("thinkingMode") or "off"
    if text:
        _open_tag, _close_tag = reasoning_delimiters_for(state._loaded_model_ref)
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
        **state._runtime_fields(
            prompt_cache=None,
            speculative_decoding=True,
            tree_budget=result["tree_budget"],
        ),
    }
