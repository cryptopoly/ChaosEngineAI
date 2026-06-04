"""Chat generation entrypoints for ``ChaosEngineState``.

Two helpers lifted out of ``state/__init__.py``:

* ``generate`` — synchronous chat completion. Resolves the effective
  runtime profile (request → session → launch_preferences cascade),
  reloads the model when the profile changed, runs RAG retrieval,
  invokes either the agent loop (when ``enableTools``) or the plain
  ``runtime.generate``, and returns the session + assistant message
  + runtime status payload routes hand back to the UI.
* ``generate_stream`` — SSE streaming version. Same profile cascade
  + RAG injection, then yields ``phase`` / ``token`` /
  ``reasoning`` / ``toolCallStart`` / ``toolCallResult`` / ``done``
  events. Ships with five guards (memory pre-flight, output-length
  runaway, repetition / loop, tok/s floor, in-stream panic + thermal)
  that abort or surface warnings without wedging the UI.

Both take the ``ChaosEngineState`` instance as the first argument so
the class methods stay 1-line wrappers.

Extracted as part of the v0.8.0 Phase 1a-11 refactor.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from backend_service.models import GenerateRequest, LoadModelRequest
from backend_service.state._helpers import (
    _build_history_with_reasoning,
    _build_sampler_overrides,
    _compose_chat_system_prompt,
    _history_token_budget,
)


if TYPE_CHECKING:
    from backend_service.state import ChaosEngineState


def generate(state: ChaosEngineState, request: GenerateRequest) -> dict[str, Any]:
    with state._lock:
        session = state._ensure_session(request.sessionId, request.title)
        launch_preferences = state._launch_preferences()
        effective_model_ref = request.modelRef or session.get("modelRef")
        effective_model_name = request.modelName or session.get("model")
        effective_canonical_repo = request.canonicalRepo or session.get("canonicalRepo")
        effective_source = request.source or session.get("modelSource") or "catalog"
        effective_path = request.path if request.path is not None else session.get("modelPath")
        effective_backend = request.backend or session.get("modelBackend") or "auto"
        effective_thinking_mode = request.thinkingMode or session.get("thinkingMode") or "off"
        desired_cache_strategy = (
            request.cacheStrategy if request.cacheStrategy is not None
            else session.get("cacheStrategy") or launch_preferences["cacheStrategy"]
        )
        desired_cache_bits = (
            request.cacheBits if request.cacheBits is not None
            else session.get("cacheBits") if session.get("cacheBits") is not None
            else launch_preferences["cacheBits"]
        )
        desired_fp16_layers = (
            request.fp16Layers if request.fp16Layers is not None
            else session.get("fp16Layers") if session.get("fp16Layers") is not None
            else launch_preferences["fp16Layers"]
        )
        desired_fused_attention = (
            request.fusedAttention if request.fusedAttention is not None
            else session.get("fusedAttention") if session.get("fusedAttention") is not None
            else launch_preferences["fusedAttention"]
        )
        desired_fit_model = (
            request.fitModelInMemory if request.fitModelInMemory is not None
            else session.get("fitModelInMemory") if session.get("fitModelInMemory") is not None
            else launch_preferences["fitModelInMemory"]
        )
        desired_context_tokens = (
            request.contextTokens if request.contextTokens is not None
            else session.get("contextTokens") if session.get("contextTokens") is not None
            else launch_preferences["contextTokens"]
        )
        desired_speculative_decoding = (
            request.speculativeDecoding if request.speculativeDecoding is not None
            else session.get("speculativeDecoding") if session.get("speculativeDecoding") is not None
            else launch_preferences["speculativeDecoding"]
        )
        desired_tree_budget = (
            request.treeBudget if request.treeBudget is not None
            else session.get("treeBudget") if session.get("treeBudget") is not None
            else launch_preferences["treeBudget"]
        )
        requested_runtime = state._requested_runtime_metrics_fields(
            cache_strategy=str(desired_cache_strategy),
            cache_bits=int(desired_cache_bits),
            fp16_layers=int(desired_fp16_layers),
            fit_model_in_memory=bool(desired_fit_model),
            speculative_decoding=bool(desired_speculative_decoding),
            tree_budget=int(desired_tree_budget),
        )
        effective_cache_strategy = "native" if desired_speculative_decoding else desired_cache_strategy
        effective_cache_bits = 0 if desired_speculative_decoding else desired_cache_bits
        effective_fp16_layers = 0 if desired_speculative_decoding else desired_fp16_layers

        should_reload_model = state._should_reload_for_profile(
            model_ref=effective_model_ref,
            cache_bits=effective_cache_bits,
            fp16_layers=effective_fp16_layers,
            fused_attention=desired_fused_attention,
            cache_strategy=effective_cache_strategy,
            fit_model_in_memory=desired_fit_model,
            context_tokens=desired_context_tokens,
            speculative_decoding=desired_speculative_decoding,
            tree_budget=desired_tree_budget,
        )

        if effective_model_ref and should_reload_model:
            state.load_model(
                LoadModelRequest(
                    modelRef=effective_model_ref,
                    modelName=effective_model_name,
                    canonicalRepo=effective_canonical_repo,
                    source=effective_source,
                    backend=effective_backend,
                    path=effective_path,
                    cacheStrategy=desired_cache_strategy,
                    cacheBits=desired_cache_bits,
                    fp16Layers=desired_fp16_layers,
                    fusedAttention=desired_fused_attention,
                    fitModelInMemory=desired_fit_model,
                    contextTokens=desired_context_tokens,
                    speculativeDecoding=desired_speculative_decoding,
                    treeBudget=desired_tree_budget,
                )
            )

        if state.runtime.loaded_model is None:
            raise HTTPException(status_code=409, detail="Load a model before sending prompts.")

        if effective_canonical_repo and state.runtime.loaded_model.canonicalRepo != effective_canonical_repo:
            state.runtime.loaded_model.canonicalRepo = effective_canonical_repo

        history = _build_history_with_reasoning(
            session["messages"],
            # Don't replay prior <think> reasoning — upstream chat templates
            # (Qwen3 / DeepSeek-R1) strip it, and re-feeding it bloats the
            # prompt every turn. token_budget windows the oldest turns out so
            # a long chat can't silently overflow the context.
            preserve_reasoning=False,
            token_budget=_history_token_budget(
                context_tokens=desired_context_tokens,
                max_tokens=request.maxTokens,
                system_prompt=request.systemPrompt,
                prompt=request.prompt,
            ),
        )
        session["messages"].append({"role": "user", "text": request.prompt, "metrics": None})
        session["updatedAt"] = state._time_label()
        # Phase 2.12: if `oneTurnOverride` is set, skip persisting the
        # active runtime's model identity onto the session so the
        # session default (the previously-loaded model) sticks for
        # the next plain message. Other session metadata (cache
        # strategy, context, thinking mode) still updates so the
        # picked model's runtime profile is reflected on this turn.
        if not getattr(request, "oneTurnOverride", False):
            session["model"] = state.runtime.loaded_model.name
            session["modelRef"] = state.runtime.loaded_model.ref
            session["canonicalRepo"] = state.runtime.loaded_model.canonicalRepo
            session["modelSource"] = state.runtime.loaded_model.source
            session["modelPath"] = state.runtime.loaded_model.path
            session["modelBackend"] = state.runtime.loaded_model.backend
        session["thinkingMode"] = effective_thinking_mode
        session["cacheLabel"] = state._cache_label(
            cache_strategy=str(state.runtime.loaded_model.cacheStrategy),
            bits=int(state.runtime.loaded_model.cacheBits),
            fp16_layers=int(state.runtime.loaded_model.fp16Layers),
        )
        session["cacheStrategy"] = state.runtime.loaded_model.cacheStrategy
        session["cacheBits"] = state.runtime.loaded_model.cacheBits
        session["fp16Layers"] = state.runtime.loaded_model.fp16Layers
        session["fusedAttention"] = state.runtime.loaded_model.fusedAttention
        session["fitModelInMemory"] = state.runtime.loaded_model.fitModelInMemory
        session["contextTokens"] = state.runtime.loaded_model.contextTokens
        session["speculativeDecoding"] = state.runtime.loaded_model.speculativeDecoding
        session["dflashDraftModel"] = state.runtime.loaded_model.dflashDraftModel
        session["treeBudget"] = state.runtime.loaded_model.treeBudget
        if session["title"] == "New chat":
            requested_title = str(request.title or "").strip()
            session["title"] = (
                requested_title
                if requested_title and requested_title != "New chat"
                else state._auto_session_title(request.prompt, exclude_session_id=session["id"])
            )
        model_tag = state.runtime.loaded_model.name if state.runtime.loaded_model else "unknown"
        msg_count = len(history) + 1
        state.add_log("chat", "info", f"[{model_tag}] Running chat completion on conversation with {msg_count} messages.")
        state.add_log("chat", "info", f"[{model_tag}] Generating response...")
        state.active_requests += 1
        effective_system_prompt = _compose_chat_system_prompt(request.systemPrompt, effective_thinking_mode)
        doc_context, rag_citations = state._retrieve_session_context(session["id"], request.prompt)
        if doc_context:
            rag_preamble = (
                "You have access to the following document context retrieved from the user's uploaded files. "
                "Use it to answer their questions when relevant.\n\n--- DOCUMENT CONTEXT ---\n"
                + doc_context
                + "\n--- END CONTEXT ---"
            )
            effective_system_prompt = (rag_preamble + "\n\n" + effective_system_prompt).strip()
            state.add_log("chat", "info", f"[{model_tag}] Injected {len(doc_context)} chars of document context.")

    gen_start = time.perf_counter()
    try:
        if request.enableTools:
            from backend_service.agent import run_agent_loop
            agent_result = run_agent_loop(
                generate_fn=state.runtime.generate,
                prompt=request.prompt,
                history=history,
                system_prompt=effective_system_prompt,
                max_tokens=request.maxTokens,
                temperature=request.temperature,
                images=request.images,
                available_tools=request.availableTools,
            )
            # Synthesize a GenerationResult-like object for metrics
            class _AgentResultProxy:
                text = agent_result.text
                finishReason = "stop"
                promptTokens = agent_result.total_prompt_tokens
                completionTokens = agent_result.total_completion_tokens
                totalTokens = agent_result.total_prompt_tokens + agent_result.total_completion_tokens
                tokS = 0.0
                runtimeNote = f"Agent loop: {agent_result.iterations} iterations, {len(agent_result.tool_calls)} tool calls"
                responseSeconds = 0.0
                tool_calls = None
            result = _AgentResultProxy()
            tool_call_payloads = [
                {
                    "id": tc.tool_call_id,
                    "name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                    "elapsed": tc.elapsed_seconds,
                    # Phase 2.8: forward structured output hint +
                    # data through to the frontend `ToolCallInfo`.
                    # When `render_as` is None the frontend falls
                    # back to the legacy collapsible-JSON view.
                    "renderAs": tc.render_as,
                    "data": tc.data,
                }
                for tc in agent_result.tool_calls
            ]
        else:
            result = state.runtime.generate(
                prompt=request.prompt,
                history=history,
                system_prompt=effective_system_prompt,
                max_tokens=request.maxTokens,
                temperature=request.temperature,
                images=request.images,
                samplers=_build_sampler_overrides(request),
                reasoning_effort=request.reasoningEffort,
                json_schema=request.jsonSchema,
            )
            tool_call_payloads = []
    except RuntimeError as exc:
        with state._lock:
            if (session["messages"]
                    and session["messages"][-1].get("role") == "user"
                    and session["messages"][-1].get("text") == request.prompt):
                session["messages"].pop()
                session["updatedAt"] = state._time_label()
                state._persist_sessions()
            state.active_requests = max(0, state.active_requests - 1)
            state.add_log("chat", "error", f"[{model_tag}] Generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    gen_elapsed = round(time.perf_counter() - gen_start, 2)
    with state._lock:
        state.active_requests = max(0, state.active_requests - 1)
        state.requests_served += 1
        metrics = state._assistant_metrics_payload(result, requested_runtime=requested_runtime)
        if tool_call_payloads:
            metrics["toolCalls"] = tool_call_payloads
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "text": result.text,
            "metrics": metrics,
        }
        if tool_call_payloads:
            assistant_message["toolCalls"] = tool_call_payloads
        if rag_citations:
            assistant_message["citations"] = rag_citations
        session["messages"].append(assistant_message)
        session["updatedAt"] = state._time_label()
        state._promote_session(session)
        state.add_log(
            "chat", "info",
            f"[{model_tag}] Finished response -- {result.completionTokens} tokens in {gen_elapsed}s "
            f"({result.tokS} tok/s, {result.promptTokens} prompt tokens).",
        )
        state.add_activity("Chat completion", session["title"])
        state._persist_sessions()

        return {
            "session": session,
            "assistant": assistant_message,
            "runtime": state.runtime.status(active_requests=state.active_requests, requests_served=state.requests_served),
        }


def generate_stream(state: ChaosEngineState, request: GenerateRequest):
    """SSE streaming version of generate(). Returns a StreamingResponse."""
    with state._lock:
        session = state._ensure_session(request.sessionId, request.title)
        launch_preferences = state._launch_preferences()
        effective_model_ref = request.modelRef or session.get("modelRef")
        effective_model_name = request.modelName or session.get("model")
        effective_canonical_repo = request.canonicalRepo or session.get("canonicalRepo")
        effective_source = request.source or session.get("modelSource") or "catalog"
        effective_path = request.path if request.path is not None else session.get("modelPath")
        effective_backend = request.backend or session.get("modelBackend") or "auto"
        effective_thinking_mode = request.thinkingMode or session.get("thinkingMode") or "off"
        desired_cache_strategy = (
            request.cacheStrategy if request.cacheStrategy is not None
            else session.get("cacheStrategy") or launch_preferences["cacheStrategy"]
        )
        desired_cache_bits = (
            request.cacheBits if request.cacheBits is not None
            else session.get("cacheBits") if session.get("cacheBits") is not None
            else launch_preferences["cacheBits"]
        )
        desired_fp16_layers = (
            request.fp16Layers if request.fp16Layers is not None
            else session.get("fp16Layers") if session.get("fp16Layers") is not None
            else launch_preferences["fp16Layers"]
        )
        desired_fused_attention = (
            request.fusedAttention if request.fusedAttention is not None
            else session.get("fusedAttention") if session.get("fusedAttention") is not None
            else launch_preferences["fusedAttention"]
        )
        desired_fit_model = (
            request.fitModelInMemory if request.fitModelInMemory is not None
            else session.get("fitModelInMemory") if session.get("fitModelInMemory") is not None
            else launch_preferences["fitModelInMemory"]
        )
        desired_context_tokens = (
            request.contextTokens if request.contextTokens is not None
            else session.get("contextTokens") if session.get("contextTokens") is not None
            else launch_preferences["contextTokens"]
        )
        desired_speculative_decoding = (
            request.speculativeDecoding if request.speculativeDecoding is not None
            else session.get("speculativeDecoding") if session.get("speculativeDecoding") is not None
            else launch_preferences["speculativeDecoding"]
        )
        desired_tree_budget = (
            request.treeBudget if request.treeBudget is not None
            else session.get("treeBudget") if session.get("treeBudget") is not None
            else launch_preferences["treeBudget"]
        )
        requested_runtime = state._requested_runtime_metrics_fields(
            cache_strategy=str(desired_cache_strategy),
            cache_bits=int(desired_cache_bits),
            fp16_layers=int(desired_fp16_layers),
            fit_model_in_memory=bool(desired_fit_model),
            speculative_decoding=bool(desired_speculative_decoding),
            tree_budget=int(desired_tree_budget),
        )
        effective_cache_strategy = "native" if desired_speculative_decoding else desired_cache_strategy
        effective_cache_bits = 0 if desired_speculative_decoding else desired_cache_bits
        effective_fp16_layers = 0 if desired_speculative_decoding else desired_fp16_layers

        should_reload = state._should_reload_for_profile(
            model_ref=effective_model_ref, cache_bits=effective_cache_bits,
            fp16_layers=effective_fp16_layers, fused_attention=desired_fused_attention,
            cache_strategy=effective_cache_strategy, fit_model_in_memory=desired_fit_model,
            context_tokens=desired_context_tokens,
            speculative_decoding=desired_speculative_decoding,
            tree_budget=desired_tree_budget,
        )
        if effective_model_ref and should_reload:
            state.load_model(LoadModelRequest(
                modelRef=effective_model_ref, modelName=effective_model_name,
                canonicalRepo=effective_canonical_repo,
                source=effective_source, backend=effective_backend, path=effective_path,
                cacheStrategy=desired_cache_strategy, cacheBits=desired_cache_bits,
                fp16Layers=desired_fp16_layers,
                fusedAttention=desired_fused_attention,
                fitModelInMemory=desired_fit_model, contextTokens=desired_context_tokens,
                speculativeDecoding=desired_speculative_decoding,
                treeBudget=desired_tree_budget,
            ))

        if state.runtime.loaded_model is None:
            raise HTTPException(status_code=409, detail="Load a model before sending prompts.")

        if effective_canonical_repo and state.runtime.loaded_model.canonicalRepo != effective_canonical_repo:
            state.runtime.loaded_model.canonicalRepo = effective_canonical_repo

        history = _build_history_with_reasoning(
            session["messages"],
            # Don't replay prior <think> reasoning — upstream chat templates
            # (Qwen3 / DeepSeek-R1) strip it, and re-feeding it bloats the
            # prompt every turn. token_budget windows the oldest turns out so
            # a long chat can't silently overflow the context.
            preserve_reasoning=False,
            token_budget=_history_token_budget(
                context_tokens=desired_context_tokens,
                max_tokens=request.maxTokens,
                system_prompt=request.systemPrompt,
                prompt=request.prompt,
            ),
        )
        session["messages"].append({"role": "user", "text": request.prompt, "metrics": None})
        session["updatedAt"] = state._time_label()
        # Phase 2.12: if `oneTurnOverride` is set, skip persisting the
        # active runtime's model identity onto the session so the
        # session default (the previously-loaded model) sticks for
        # the next plain message. Other session metadata (cache
        # strategy, context, thinking mode) still updates so the
        # picked model's runtime profile is reflected on this turn.
        if not getattr(request, "oneTurnOverride", False):
            session["model"] = state.runtime.loaded_model.name
            session["modelRef"] = state.runtime.loaded_model.ref
            session["canonicalRepo"] = state.runtime.loaded_model.canonicalRepo
            session["modelSource"] = state.runtime.loaded_model.source
            session["modelPath"] = state.runtime.loaded_model.path
            session["modelBackend"] = state.runtime.loaded_model.backend
        session["thinkingMode"] = effective_thinking_mode
        session["cacheLabel"] = state._cache_label(
            cache_strategy=str(state.runtime.loaded_model.cacheStrategy),
            bits=int(state.runtime.loaded_model.cacheBits),
            fp16_layers=int(state.runtime.loaded_model.fp16Layers),
        )
        session["cacheStrategy"] = state.runtime.loaded_model.cacheStrategy
        session["cacheBits"] = state.runtime.loaded_model.cacheBits
        session["fp16Layers"] = state.runtime.loaded_model.fp16Layers
        session["fusedAttention"] = state.runtime.loaded_model.fusedAttention
        session["fitModelInMemory"] = state.runtime.loaded_model.fitModelInMemory
        session["contextTokens"] = state.runtime.loaded_model.contextTokens
        session["speculativeDecoding"] = state.runtime.loaded_model.speculativeDecoding
        session["dflashDraftModel"] = state.runtime.loaded_model.dflashDraftModel
        session["treeBudget"] = state.runtime.loaded_model.treeBudget
        if session["title"] == "New chat":
            requested_title = str(request.title or "").strip()
            session["title"] = (
                requested_title
                if requested_title and requested_title != "New chat"
                else state._auto_session_title(request.prompt, exclude_session_id=session["id"])
            )
        model_tag = state.runtime.loaded_model.name
        state.add_log("chat", "info", f"[{model_tag}] Streaming response...")
        state.active_requests += 1
        # Hotfix (2026-05-01 v2): vision input has no working path
        # on either runtime today. The MLX worker subprocess never
        # wired images, and `_resolve_gguf_path` strips mmproj
        # projector files so llama-server never gets `--mmproj`.
        # Until mmproj wiring lands (Phase 2.6+ work), the
        # `visionEnabled` flag on LoadedModelInfo stays False on
        # every load and we strip + warn loudly here. The capability
        # resolver also demotes vision via this same flag so the
        # composer hides the attach button — this branch is the
        # belt-and-braces for legacy clients that bypass the gate.
        if request.images and not state.runtime.loaded_model.visionEnabled:
            engine_label = state.runtime.loaded_model.engine or "current"
            state.add_log(
                "chat", "warning",
                f"[{model_tag}] Stripped {len(request.images)} attached "
                f"image(s): the {engine_label} runtime has no mmproj "
                "vision projector wired up, so images would be silently "
                "dropped and the model would hallucinate. Vision support "
                "lands with the mmproj loader.",
            )
            request.images = None
        effective_system_prompt = _compose_chat_system_prompt(request.systemPrompt, effective_thinking_mode)
        doc_context, stream_rag_citations = state._retrieve_session_context(session["id"], request.prompt)
        if doc_context:
            rag_preamble = (
                "You have access to the following document context retrieved from the user's uploaded files. "
                "Use it to answer their questions when relevant.\n\n--- DOCUMENT CONTEXT ---\n"
                + doc_context
                + "\n--- END CONTEXT ---"
            )
            effective_system_prompt = (rag_preamble + "\n\n" + effective_system_prompt).strip()
            state.add_log("chat", "info", f"[{model_tag}] Injected {len(doc_context)} chars of document context.")

    chaosengine = state
    enable_tools = request.enableTools
    available_tools = request.availableTools
    gen_start = time.perf_counter()
    # Reset any stale cancellation flag from a prior turn so this fresh
    # generation isn't aborted before it starts.
    chaosengine.clear_chat_cancel(session["id"])
    session_id_for_cancel = session["id"]

    def _sse_stream():
        full_text = ""
        full_reasoning = ""
        final_chunk = None
        agent_tool_calls: list[dict[str, Any]] = []
        cancelled = False
        # Phase 2.0: track prompt-eval → generating phase transition so the
        # client can render an explicit "Processing prompt..." indicator
        # instead of a blank flashing cursor while the model is still
        # ingesting the prompt. The OpenAI-compat streaming endpoint
        # exposes nothing until the first decoded token, so phase here is
        # binary (prompt_eval | generating) plus a TTFT measurement on
        # transition.
        phase_first_output_seen = False
        ttft_seconds: float | None = None

        # Phase 2.0.5-B: pre-flight memory gate. Refuse the generation
        # before it starts when the host is already memory-starved, so
        # the user gets an actionable error instead of a silent OOM /
        # swap-thrash that wedges the laptop. The gate is conservative
        # — it does not predict working-set size, just bails when the
        # available-memory floor or pressure ceiling is breached.
        try:
            from backend_service.helpers.memory_gate import (
                gate_chat_generation,
                snapshot_memory_signals,
            )

            available_gb, pressure_percent = snapshot_memory_signals()
            refusal = gate_chat_generation(available_gb, pressure_percent)
            if refusal is not None:
                chaosengine.add_log(
                    "chat", "warning",
                    f"[{model_tag}] Memory gate refused generation: "
                    f"{refusal['code']} (avail={available_gb:.1f} GB, "
                    f"pressure={pressure_percent:.0f}%).",
                )
                with chaosengine._lock:
                    # Roll back the optimistic user message we appended
                    # earlier so the refusal looks like the request never
                    # happened, matching the existing RuntimeError path.
                    if (session["messages"]
                            and session["messages"][-1].get("role") == "user"
                            and session["messages"][-1].get("text") == request.prompt):
                        session["messages"].pop()
                        session["updatedAt"] = chaosengine._time_label()
                        chaosengine._persist_sessions()
                    chaosengine.active_requests = max(0, chaosengine.active_requests - 1)
                yield f"data: {json.dumps({'error': refusal['message']})}\n\n"
                return
        except Exception as exc:
            # Gate failure must not block legitimate generations. Log and
            # continue — better to risk a possible OOM than to refuse
            # everything when psutil glitches.
            chaosengine.add_log(
                "chat", "warning",
                f"[{model_tag}] Memory gate skipped due to error: {exc}",
            )

        yield f"data: {json.dumps({'phase': 'prompt_eval'})}\n\n"

        # Phase 2.0.5-D: output-length runaway guard. Abort the generation
        # if accumulated visible text exceeds the user's max_tokens budget
        # by 1.5×, which catches decoder loops that ignore the EOS token
        # (a known failure mode on certain quantised models). Char count
        # is a fast proxy — average ~4 chars per token across English +
        # markdown code, so the threshold is `max_tokens * 6` chars.
        runaway_char_budget = max(2000, int(request.maxTokens) * 6)
        runaway_triggered = False
        runaway_loop_reason: str | None = None

        # Phase 2.0.5-F: per-stream repetition / reasoning-loop guard for
        # the llama.cpp path. The MLX worker has run this guard inside the
        # subprocess for a while; the llama-server REST stream had no
        # equivalent and a runaway model could decode tokens indefinitely
        # against a paused UI. Same RunawayGuard module both paths use.
        from backend_service.runaway_guard import RunawayGuard as _RunawayGuard

        llama_path_guard = _RunawayGuard()

        # Phase 2.0.5-C: tok/s floor monitor. After the model has
        # produced output for a 30-second window, check the rolling
        # decode rate. Falling below 0.3 tok/s for that long usually
        # means thermal throttle, GPU stall, or a corrupted model
        # state — none of which recovers on its own. Abort with a
        # diagnostic so the user can switch model / cool down /
        # restart the worker.
        TOKS_FLOOR_WINDOW_S = 30.0
        TOKS_FLOOR_MIN = 0.3
        window_started_at: float | None = None
        window_tokens = 0
        stall_triggered = False

        # Phase 2.0.5-G: in-stream panic monitor. While a generation
        # is in flight, sample memory every PANIC_SAMPLE_INTERVAL_S
        # and emit a `panic` SSE event when free RAM crosses the
        # critical floor or pressure goes critical. The front-end
        # renders a non-blocking banner offering Cancel / Unload
        # warm / Continue. Generation is NOT auto-cancelled here —
        # that's the user's call. The stricter pre-flight gate
        # (Phase 2.0.5-B) blocks tight starts, this catches mid-
        # flight degradation as KV cache or other activity grows.
        PANIC_SAMPLE_INTERVAL_S = 5.0
        PANIC_AVAILABLE_FLOOR_GB = 0.5
        PANIC_PRESSURE_CEILING = 96.0
        last_panic_sample_at: float | None = None
        panic_emitted = False
        # Phase 2.0.5-I: thermal pressure watch. `pmset -g therm` on
        # macOS reports warning levels when CPU/GPU is throttling.
        # We surface the first transition to "critical" via a SSE
        # event so the user sees why decode just slowed. Linux /
        # Windows: read returns None and this watch is a no-op.
        thermal_warning_emitted = False

        def _maybe_emit_generating_phase() -> str:
            nonlocal phase_first_output_seen, ttft_seconds
            if phase_first_output_seen:
                return ""
            phase_first_output_seen = True
            ttft_seconds = round(time.perf_counter() - gen_start, 3)
            return f"data: {json.dumps({'phase': 'generating', 'ttftSeconds': ttft_seconds})}\n\n"

        # Token coalescing: batch visible token frames so a fast decoder
        # doesn't pay a json.dumps + SSE frame per token. Flush on size, a
        # short time window, any non-token event, or stream end. Disabled
        # when per-token logprobs are requested (they must stay 1:1 aligned).
        _COALESCE_CHARS = 24
        _COALESCE_SECS = 0.05
        _coalesce_tokens = not (request.logprobs and int(request.logprobs) > 0)
        _tok: dict[str, Any] = {"buf": [], "chars": 0, "started": 0.0}

        def _flush_tokens() -> str:
            if not _tok["buf"]:
                return ""
            merged = "".join(_tok["buf"])
            _tok["buf"] = []
            _tok["chars"] = 0
            _tok["started"] = 0.0
            return f"data: {json.dumps({'token': merged})}\n\n"

        try:
            if enable_tools:
                from backend_service.agent import run_agent_loop_streaming
                for event in run_agent_loop_streaming(
                    generate_fn=chaosengine.runtime.generate,
                    stream_generate_fn=chaosengine.runtime.stream_generate,
                    prompt=request.prompt, history=history,
                    system_prompt=effective_system_prompt,
                    max_tokens=request.maxTokens, temperature=request.temperature,
                    images=request.images,
                    available_tools=available_tools,
                ):
                    if chaosengine.is_chat_cancel_requested(session_id_for_cancel):
                        cancelled = True
                        break
                    if "token" in event:
                        phase_event = _maybe_emit_generating_phase()
                        if phase_event:
                            yield phase_event
                        full_text += event["token"]
                        yield f"data: {json.dumps({'token': event['token']})}\n\n"
                        if len(full_text) > runaway_char_budget:
                            runaway_triggered = True
                            cancelled = True
                            break
                    elif "tool_call_start" in event:
                        phase_event = _maybe_emit_generating_phase()
                        if phase_event:
                            yield phase_event
                        yield f"data: {json.dumps({'toolCallStart': event['tool_call_start']})}\n\n"
                    elif "tool_call_result" in event:
                        agent_tool_calls.append(event["tool_call_result"])
                        yield f"data: {json.dumps({'toolCallResult': event['tool_call_result']})}\n\n"
                    elif event.get("done"):
                        # Agent loop finished — handled below
                        pass
            else:
                for chunk in chaosengine.runtime.stream_generate(
                    prompt=request.prompt, history=history,
                    system_prompt=effective_system_prompt,
                    max_tokens=request.maxTokens, temperature=request.temperature,
                    images=request.images,
                    thinking_mode=effective_thinking_mode,
                    samplers=_build_sampler_overrides(request),
                    reasoning_effort=request.reasoningEffort,
                    json_schema=request.jsonSchema,
                ):
                    if chaosengine.is_chat_cancel_requested(session_id_for_cancel):
                        cancelled = True
                        break
                    if chunk.reasoning:
                        phase_event = _maybe_emit_generating_phase()
                        if phase_event:
                            yield phase_event
                        _f = _flush_tokens()
                        if _f:
                            yield _f
                        full_reasoning += chunk.reasoning
                        yield f"data: {json.dumps({'reasoning': chunk.reasoning})}\n\n"
                    if chunk.reasoning_done:
                        _f = _flush_tokens()
                        if _f:
                            yield _f
                        yield f"data: {json.dumps({'reasoningDone': True})}\n\n"
                    if chunk.text:
                        phase_event = _maybe_emit_generating_phase()
                        if phase_event:
                            yield phase_event
                        full_text += chunk.text
                        if _coalesce_tokens:
                            if not _tok["buf"]:
                                _tok["started"] = time.perf_counter()
                            _tok["buf"].append(chunk.text)
                            _tok["chars"] += len(chunk.text)
                            if (
                                _tok["chars"] >= _COALESCE_CHARS
                                or time.perf_counter() - _tok["started"] >= _COALESCE_SECS
                            ):
                                _f = _flush_tokens()
                                if _f:
                                    yield _f
                        else:
                            yield f"data: {json.dumps({'token': chunk.text})}\n\n"
                        # Phase 3.3: forward per-token logprobs when
                        # the inference layer captured them.
                        if chunk.token_logprobs:
                            yield f"data: {json.dumps({'tokenLogprobs': chunk.token_logprobs})}\n\n"
                        if len(full_text) > runaway_char_budget:
                            runaway_triggered = True
                            cancelled = True
                            break
                        # Phase 2.0.5-F: feed loop / repetition guard.
                        try:
                            llama_path_guard.feed(chunk.text)
                        except RuntimeError as guard_exc:
                            runaway_triggered = True
                            runaway_loop_reason = str(guard_exc)
                            cancelled = True
                            break
                        # Phase 2.0.5-C: tok/s floor sampling. Each
                        # chunk roughly maps to one token from the
                        # SSE stream; chunk count is a workable proxy.
                        now = time.perf_counter()
                        if window_started_at is None:
                            window_started_at = now
                            window_tokens = 0
                        window_tokens += 1
                        if now - window_started_at >= TOKS_FLOOR_WINDOW_S:
                            rate = window_tokens / max(1e-6, now - window_started_at)
                            if rate < TOKS_FLOOR_MIN:
                                stall_triggered = True
                                cancelled = True
                                runaway_loop_reason = (
                                    f"Decode stalled at {rate:.2f} tok/s "
                                    f"for {TOKS_FLOOR_WINDOW_S:.0f}s — "
                                    "likely thermal throttle, GPU stall, "
                                    "or worker deadlock. Aborting."
                                )
                                break
                            window_started_at = now
                            window_tokens = 0
                        # Phase 2.0.5-G + I: panic + thermal monitors.
                        # Sampled at PANIC_SAMPLE_INTERVAL_S together to
                        # keep subprocess / psutil cost bounded. Each
                        # emits at most once per turn.
                        if (
                            (not panic_emitted or not thermal_warning_emitted)
                            and (
                                last_panic_sample_at is None
                                or now - last_panic_sample_at >= PANIC_SAMPLE_INTERVAL_S
                            )
                        ):
                            last_panic_sample_at = now
                            if not panic_emitted:
                                try:
                                    from backend_service.helpers.memory_gate import (
                                        snapshot_memory_signals as _panic_snapshot,
                                    )
                                    p_avail, p_pressure = _panic_snapshot()
                                    if (
                                        p_avail < PANIC_AVAILABLE_FLOOR_GB
                                        or p_pressure > PANIC_PRESSURE_CEILING
                                    ):
                                        panic_emitted = True
                                        chaosengine.add_log(
                                            "chat", "warning",
                                            f"[{model_tag}] Panic: avail="
                                            f"{p_avail:.1f} GB, "
                                            f"pressure={p_pressure:.0f}%.",
                                        )
                                        _f = _flush_tokens()
                                        if _f:
                                            yield _f
                                        yield (
                                            "data: "
                                            + json.dumps({
                                                "panic": True,
                                                "availableGb": p_avail,
                                                "pressurePercent": p_pressure,
                                                "message": (
                                                    "System memory critical mid-"
                                                    "generation. Consider cancelling "
                                                    "this turn or unloading warm "
                                                    "models before retrying."
                                                ),
                                            })
                                            + "\n\n"
                                        )
                                except Exception as panic_exc:
                                    chaosengine.add_log(
                                        "chat", "warning",
                                        f"[{model_tag}] Panic sample skipped: {panic_exc}",
                                    )
                            if not thermal_warning_emitted:
                                try:
                                    from backend_service.helpers.thermal import (
                                        read_thermal_state,
                                    )
                                    thermal_state = read_thermal_state()
                                    if thermal_state == "critical":
                                        thermal_warning_emitted = True
                                        chaosengine.add_log(
                                            "chat", "warning",
                                            f"[{model_tag}] Thermal warning: critical.",
                                        )
                                        _f = _flush_tokens()
                                        if _f:
                                            yield _f
                                        yield (
                                            "data: "
                                            + json.dumps({
                                                "thermalWarning": True,
                                                "state": thermal_state,
                                                "message": (
                                                    "System is thermally throttling. "
                                                    "Decode speed will drop until the "
                                                    "machine cools. Consider pausing "
                                                    "and retrying after a cooldown."
                                                ),
                                            })
                                            + "\n\n"
                                        )
                                except Exception as thermal_exc:
                                    chaosengine.add_log(
                                        "chat", "warning",
                                        f"[{model_tag}] Thermal sample skipped: {thermal_exc}",
                                    )
                    if chunk.done:
                        final_chunk = chunk
        except RuntimeError as exc:
            with chaosengine._lock:
                if (session["messages"]
                        and session["messages"][-1].get("role") == "user"
                        and session["messages"][-1].get("text") == request.prompt):
                    session["messages"].pop()
                    session["updatedAt"] = chaosengine._time_label()
                    chaosengine._persist_sessions()
                chaosengine.active_requests = max(0, chaosengine.active_requests - 1)
                chaosengine.add_log("chat", "error", f"[{model_tag}] Streaming failed: {exc}")
            chaosengine.clear_chat_cancel(session_id_for_cancel)
            _f = _flush_tokens()
            if _f:
                yield _f
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return
        finally:
            chaosengine.clear_chat_cancel(session_id_for_cancel)

        # Flush any tokens still buffered by the coalescer before the
        # terminal done / cancelled events (covers normal end + all breaks).
        _f = _flush_tokens()
        if _f:
            yield _f

        if cancelled:
            yield f"data: {json.dumps({'cancelled': True})}\n\n"
            if runaway_loop_reason is not None:
                chaosengine.add_log(
                    "chat", "warning",
                    f"[{model_tag}] {runaway_loop_reason} "
                    f"(after {len(full_text)} chars).",
                )
            elif runaway_triggered:
                chaosengine.add_log(
                    "chat", "warning",
                    f"[{model_tag}] Output runaway guard tripped at "
                    f"{len(full_text)} chars (budget {runaway_char_budget}); "
                    "stream aborted to prevent decoder loop.",
                )
            else:
                chaosengine.add_log("chat", "info", f"[{model_tag}] Generation cancelled by user.")

        gen_elapsed = round(time.perf_counter() - gen_start, 2)
        with chaosengine._lock:
            chaosengine.active_requests = max(0, chaosengine.active_requests - 1)
            chaosengine.requests_served += 1

            tok_s = final_chunk.tok_s if final_chunk else 0
            prompt_tokens = final_chunk.prompt_tokens if final_chunk else 0
            completion_tokens = final_chunk.completion_tokens if final_chunk else 0
            if (not tok_s or tok_s == 0) and completion_tokens > 0 and gen_elapsed > 0:
                tok_s = round(completion_tokens / gen_elapsed, 1)

            metrics = chaosengine._stream_assistant_metrics_payload(
                final_chunk=final_chunk,
                tok_s=tok_s,
                response_seconds=gen_elapsed,
                requested_runtime=requested_runtime,
                ttft_seconds=ttft_seconds,
            )
            if agent_tool_calls:
                metrics["toolCalls"] = agent_tool_calls

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "text": full_text,
                "metrics": metrics,
            }
            if full_reasoning:
                assistant_message["reasoning"] = full_reasoning
            if agent_tool_calls:
                assistant_message["toolCalls"] = agent_tool_calls
            if stream_rag_citations:
                assistant_message["citations"] = stream_rag_citations
            session["messages"].append(assistant_message)
            session["updatedAt"] = chaosengine._time_label()
            chaosengine._promote_session(session)
            chaosengine.add_log(
                "chat", "info",
                f"[{model_tag}] Finished streaming -- {completion_tokens} tokens in {gen_elapsed}s ({tok_s} tok/s).",
            )
            chaosengine._persist_sessions()

            done_payload = {
                "done": True,
                "session": session,
                "assistant": assistant_message,
                "runtime": chaosengine.runtime.status(
                    active_requests=chaosengine.active_requests,
                    requests_served=chaosengine.requests_served,
                ),
            }
            if cancelled:
                done_payload["cancelled"] = True
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
