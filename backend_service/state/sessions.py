"""Chat session lifecycle for ``ChaosEngineState``.

Thirteen helpers lifted out of ``state/__init__.py`` covering the
full session-and-message lifecycle:

* ``default_session_model`` — populate a new session's runtime profile
  from the loaded model, falling back to the first non-broken library
  entry, then to the catalog default chat variant.
* ``promote_session`` — move a session to the top of the recency list.
* ``persist_sessions`` — best-effort write of the session list to disk.
* ``unique_session_title`` + ``auto_session_title`` +
  ``normalize_auto_generated_session_titles`` — title disambiguation
  helpers (``Foo`` / ``Foo (2)`` / ``Foo (3)``…).
* ``ensure_session`` — fetch existing session or create a new one
  with the default runtime profile.
* ``create_session`` — public CRUD entrypoint for the API.
* ``add_message_variant`` (Phase 2.5) — generate a sibling variant
  of an assistant message against a different (already loaded) model.
* ``delve_message`` (Phase 3.6) — re-process an assistant message
  with a critique system prompt, attached as a variant.
* ``fork_session`` (Phase 2.4) — branch a thread at a specific
  message; the fork carries the parent's runtime profile + linkage.
* ``update_session`` — apply a settings/title patch.
* ``delete_session`` — remove from the list + persist.

All take the ``ChaosEngineState`` instance as the first argument so
the class methods stay 1-3 line wrappers.

Extracted as part of the v0.8.0 Phase 1a-9 refactor.
"""

from __future__ import annotations

import copy
import time
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from backend_service.helpers.persistence import _default_chat_variant
from backend_service.models import UpdateSessionRequest
from backend_service.state._helpers import (
    _build_history_with_reasoning,
    _compose_chat_system_prompt,
    _legacy_title_from_prompt,
    _title_from_prompt,
    _title_variant_pattern,
)


if TYPE_CHECKING:
    from backend_service.state import ChaosEngineState


def default_session_model(state: ChaosEngineState) -> dict[str, Any]:
    model_info = state.runtime.loaded_model
    launch_preferences = state._launch_preferences()
    if model_info is not None:
        return {
            "model": model_info.name,
            "modelRef": model_info.ref,
            "canonicalRepo": model_info.canonicalRepo,
            "modelSource": model_info.source,
            "modelPath": model_info.path,
            "modelBackend": model_info.backend,
            "cacheLabel": state._cache_label(
                cache_strategy=str(model_info.cacheStrategy),
                bits=int(model_info.cacheBits),
                fp16_layers=int(model_info.fp16Layers),
            ),
            "cacheStrategy": model_info.cacheStrategy,
            "cacheBits": model_info.cacheBits,
            "fp16Layers": model_info.fp16Layers,
            "fusedAttention": model_info.fusedAttention,
            "fitModelInMemory": model_info.fitModelInMemory,
            "contextTokens": model_info.contextTokens,
            "speculativeDecoding": model_info.speculativeDecoding,
            "dflashDraftModel": model_info.dflashDraftModel,
            "treeBudget": model_info.treeBudget,
        }

    # No model is currently loaded. Prefer a model the user actually has
    # downloaded over a catalog default — surfacing a catalog-only entry
    # (e.g. nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF) just produces a
    # confusing "Failed to load … isn't downloaded on this machine"
    # error when the user clicks Load.
    for entry in state._library():
        entry_type = entry.get("modelType")
        if entry_type and entry_type != "text":
            continue
        if entry.get("broken"):
            continue
        return {
            "model": entry["name"],
            "modelRef": entry["name"],
            "canonicalRepo": entry.get("canonicalRepo") or entry.get("repo"),
            "modelSource": "library",
            "modelPath": entry["path"],
            "modelBackend": entry.get("backend", "auto"),
            "cacheLabel": state._cache_label(
                cache_strategy=str(launch_preferences["cacheStrategy"]),
                bits=int(launch_preferences["cacheBits"]),
                fp16_layers=int(launch_preferences["fp16Layers"]),
            ),
            "cacheStrategy": launch_preferences["cacheStrategy"],
            "cacheBits": launch_preferences["cacheBits"],
            "fp16Layers": launch_preferences["fp16Layers"],
            "fusedAttention": launch_preferences["fusedAttention"],
            "fitModelInMemory": launch_preferences["fitModelInMemory"],
            "contextTokens": launch_preferences["contextTokens"],
            "speculativeDecoding": launch_preferences.get("speculativeDecoding", False),
            "dflashDraftModel": None,
            "treeBudget": launch_preferences.get("treeBudget", 0),
        }

    default_variant = _default_chat_variant()
    return {
        "model": default_variant["name"],
        "modelRef": default_variant["id"],
        "canonicalRepo": str(default_variant.get("repo") or "").strip() or None,
        "modelSource": "catalog",
        "modelPath": None,
        "modelBackend": default_variant.get("backend", "auto"),
        "cacheLabel": state._cache_label(
            cache_strategy=str(launch_preferences["cacheStrategy"]),
            bits=int(launch_preferences["cacheBits"]),
            fp16_layers=int(launch_preferences["fp16Layers"]),
        ),
        "cacheStrategy": launch_preferences["cacheStrategy"],
        "cacheBits": launch_preferences["cacheBits"],
        "fp16Layers": launch_preferences["fp16Layers"],
        "fusedAttention": launch_preferences["fusedAttention"],
        "fitModelInMemory": launch_preferences["fitModelInMemory"],
        "contextTokens": launch_preferences["contextTokens"],
        "speculativeDecoding": launch_preferences.get("speculativeDecoding", False),
        "dflashDraftModel": None,
        "treeBudget": launch_preferences.get("treeBudget", 0),
    }


def promote_session(state: ChaosEngineState, session: dict[str, Any]) -> None:
    state.chat_sessions = [
        session,
        *[item for item in state.chat_sessions if item["id"] != session["id"]],
    ]


def persist_sessions(state: ChaosEngineState) -> None:
    from backend_service.app import _save_chat_sessions

    try:
        _save_chat_sessions(state.chat_sessions, state._chat_sessions_path)
    except OSError:
        pass  # Non-critical -- don't crash if disk is full


def unique_session_title(
    state: ChaosEngineState,
    base_title: str,
    *,
    exclude_session_id: str | None = None,
) -> str:
    base = base_title.strip() or "New chat"
    if base == "New chat":
        return base

    pattern = _title_variant_pattern(base)
    highest_suffix = 0
    for session in state.chat_sessions:
        if exclude_session_id and session.get("id") == exclude_session_id:
            continue
        title = str(session.get("title") or "").strip()
        match = pattern.match(title)
        if not match:
            continue
        suffix = match.group(1)
        highest_suffix = max(highest_suffix, int(suffix) if suffix else 1)

    if highest_suffix == 0:
        return base
    return f"{base} ({highest_suffix + 1})"


def auto_session_title(
    state: ChaosEngineState,
    prompt: str | None,
    *,
    exclude_session_id: str | None = None,
) -> str:
    return unique_session_title(
        state,
        _title_from_prompt(prompt),
        exclude_session_id=exclude_session_id,
    )


def normalize_auto_generated_session_titles(state: ChaosEngineState) -> bool:
    seen_counts: dict[str, int] = {}
    changed = False

    for session in state.chat_sessions:
        messages = session.get("messages") if isinstance(session.get("messages"), list) else []
        first_user_message = next(
            (
                message.get("text")
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            None,
        )
        base_title = _title_from_prompt(first_user_message)
        legacy_base_title = _legacy_title_from_prompt(first_user_message)
        if base_title == "New chat":
            continue

        current_title = str(session.get("title") or "").strip()
        matches_current_title = _title_variant_pattern(base_title).match(current_title)
        matches_legacy_title = (
            legacy_base_title != base_title
            and _title_variant_pattern(legacy_base_title).match(current_title)
        )
        if not matches_current_title and not matches_legacy_title:
            continue

        seen_counts[base_title] = seen_counts.get(base_title, 0) + 1
        next_index = seen_counts[base_title]
        normalized_title = base_title if next_index == 1 else f"{base_title} ({next_index})"
        if current_title != normalized_title:
            session["title"] = normalized_title
            changed = True

    return changed


def ensure_session(
    state: ChaosEngineState,
    session_id: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    if session_id:
        for session in state.chat_sessions:
            if session["id"] == session_id:
                return session

    model_defaults = default_session_model(state)
    session = {
        "id": session_id or f"session-{uuid.uuid4().hex[:8]}",
        "title": title or "New chat",
        "updatedAt": state._time_label(),
        "pinned": False,
        "thinkingMode": "off",
        **model_defaults,
        "messages": [],
    }
    state.chat_sessions.insert(0, session)
    state.add_activity("Chat session created", session["title"])
    persist_sessions(state)
    return session


def create_session(state: ChaosEngineState, title: str | None = None) -> dict[str, Any]:
    with state._lock:
        return ensure_session(state, title=title)


def add_message_variant(
    state: ChaosEngineState,
    session_id: str,
    message_index: int,
    model_ref: str,
    model_name: str,
    canonical_repo: str | None,
    source: str,
    path: str | None,
    backend: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Phase 2.5: generate a sibling variant of an assistant message.

    Truncates the session's message list to the user message that
    produced the target assistant turn (i.e. messages[0..index-1]
    plus the user prompt at index-1), then runs a non-streaming
    generation against the override model. The result is attached
    to ``messages[message_index].variants`` so the frontend can
    render it side-by-side with the original answer.

    The override model must already be loaded as the current
    runtime — callers should preload via the existing My Models
    flow before invoking compare. Raising on misalignment keeps
    the contract simple: variant generation never reloads the
    runtime under the user.

    Returns the updated session dict so the frontend can replace
    its local copy in one round-trip.
    """
    with state._lock:
        session = next(
            (s for s in state.chat_sessions if s.get("id") == session_id),
            None,
        )
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        messages = session.get("messages") or []
        if message_index < 0 or message_index >= len(messages):
            raise ValueError(
                f"message_index {message_index} out of range "
                f"(session has {len(messages)} messages)"
            )
        target = messages[message_index]
        if target.get("role") != "assistant":
            raise ValueError(
                f"Variants can only be added to assistant messages "
                f"(message {message_index} role: {target.get('role')})"
            )
        if message_index == 0:
            raise ValueError("Cannot add a variant to the first message — no prompt available")
        user_msg = messages[message_index - 1]
        if user_msg.get("role") != "user":
            raise ValueError(
                f"Variant prompt must come from a user message at index "
                f"{message_index - 1}, got role {user_msg.get('role')}"
            )
        history = _build_history_with_reasoning(
            messages[: message_index - 1],
            preserve_reasoning=False,
        )
        user_prompt = str(user_msg.get("text") or "")

        if state.runtime.loaded_model is None:
            raise ValueError("Load the override model before requesting a variant")
        loaded = state.runtime.loaded_model
        # Sanity check the runtime is the requested model. We don't
        # auto-reload because the user explicitly wants to compare
        # against an already-warm choice.
        if loaded.ref != model_ref and loaded.runtimeTarget != model_ref:
            raise ValueError(
                f"Loaded runtime is {loaded.ref}, but variant requested {model_ref}. "
                "Load the desired model first via My Models, then retry."
            )

        started_at = time.perf_counter()
        try:
            result = state.runtime.generate(
                prompt=user_prompt,
                history=history,
                system_prompt=_compose_chat_system_prompt(None),
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except RuntimeError as exc:
            raise ValueError(f"Variant generation failed: {exc}") from exc
        elapsed = round(time.perf_counter() - started_at, 2)

        metrics = state._stream_assistant_metrics_payload(
            final_chunk=type("Chunk", (), {
                "finish_reason": result.finishReason,
                "prompt_tokens": result.promptTokens,
                "completion_tokens": result.completionTokens,
                "tok_s": result.tokS,
                "runtime_note": result.runtimeNote,
                "dflash_acceptance_rate": getattr(result, "dflashAcceptanceRate", None),
            })(),
            tok_s=result.tokS,
            response_seconds=elapsed,
        )
        metrics["model"] = model_name
        metrics["modelRef"] = model_ref
        metrics["canonicalRepo"] = canonical_repo
        metrics["modelSource"] = source
        metrics["modelPath"] = path
        metrics["backend"] = backend

        variant = {
            "modelRef": model_ref,
            "modelName": model_name,
            "text": result.text,
            "metrics": metrics,
            "generatedAt": state._time_label(),
        }
        target.setdefault("variants", []).append(variant)
        session["updatedAt"] = state._time_label()
        persist_sessions(state)
        return session


def delve_message(
    state: ChaosEngineState,
    session_id: str,
    message_index: int,
    max_tokens: int = 1024,
    temperature: float = 0.5,
) -> dict[str, Any]:
    """Phase 3.6: re-process an assistant message with a critique system
    prompt and attach the result as a variant.

    The Delve pass asks the currently-loaded model to read the prior
    answer with a critic's eye and surface anything wrong / missing
    / misleading, then propose a corrected response. Attached as a
    ``modelName: "Delve critique"`` variant so the frontend's
    existing variant rendering surfaces it under the original turn.

    Like add_message_variant, requires the model to already be
    loaded (no auto-reload).
    """
    with state._lock:
        session = next(
            (s for s in state.chat_sessions if s.get("id") == session_id),
            None,
        )
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        messages = session.get("messages") or []
        if message_index < 0 or message_index >= len(messages):
            raise ValueError(
                f"message_index {message_index} out of range "
                f"(session has {len(messages)} messages)"
            )
        target = messages[message_index]
        if target.get("role") != "assistant":
            raise ValueError(
                f"Delve only works on assistant messages "
                f"(message {message_index} role: {target.get('role')})"
            )
        if message_index == 0:
            raise ValueError("Cannot delve on the first message — no prompt available")
        user_msg = messages[message_index - 1]
        user_prompt = str(user_msg.get("text") or "")
        original_answer = str(target.get("text") or "")

        if state.runtime.loaded_model is None:
            raise ValueError("Load a model before requesting a Delve pass")
        loaded = state.runtime.loaded_model

        # Build the critique-mode system prompt. We deliberately ask
        # for both critique + improved answer in one pass so the
        # variant card renders something the user can drop straight
        # back into the thread if they like the result.
        critique_system = (
            "You are a careful reviewer. Read the prior assistant answer with a "
            "critic's eye. First, list any factual errors, missing context, or "
            "misleading claims under a 'Critique:' heading. Then, under a 'Revised "
            "answer:' heading, write a corrected response that fixes the issues "
            "you identified. Be concise."
        )

        history = _build_history_with_reasoning(
            messages[: message_index - 1],
            preserve_reasoning=False,
        )
        # Append the user prompt + original answer as context, then
        # ask the model to delve into it.
        history.append({"role": "user", "text": user_prompt})
        history.append({"role": "assistant", "text": original_answer})
        delve_prompt = (
            "Apply the Critique / Revised answer treatment to the assistant's "
            "previous response."
        )

        started_at = time.perf_counter()
        try:
            result = state.runtime.generate(
                prompt=delve_prompt,
                history=history,
                system_prompt=critique_system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except RuntimeError as exc:
            raise ValueError(f"Delve generation failed: {exc}") from exc
        elapsed = round(time.perf_counter() - started_at, 2)

        metrics = state._stream_assistant_metrics_payload(
            final_chunk=type("Chunk", (), {
                "finish_reason": result.finishReason,
                "prompt_tokens": result.promptTokens,
                "completion_tokens": result.completionTokens,
                "tok_s": result.tokS,
                "runtime_note": result.runtimeNote,
                "dflash_acceptance_rate": getattr(result, "dflashAcceptanceRate", None),
            })(),
            tok_s=result.tokS,
            response_seconds=elapsed,
        )
        metrics["model"] = "Delve critique"
        metrics["modelRef"] = loaded.ref

        variant = {
            "modelRef": loaded.ref,
            "modelName": "Delve critique",
            "text": result.text,
            "metrics": metrics,
            "generatedAt": state._time_label(),
        }
        target.setdefault("variants", []).append(variant)
        session["updatedAt"] = state._time_label()
        persist_sessions(state)
        return session


def fork_session(
    state: ChaosEngineState,
    source_session_id: str,
    fork_at_message_index: int,
    title: str | None = None,
) -> dict[str, Any]:
    """Phase 2.4: branch a thread at a specific message.

    Creates a new session containing a deep copy of the source's
    messages up to (and including) `fork_at_message_index`, plus
    the source's runtime profile (model, cache, thinking mode) so
    the fork resumes exactly where the user diverged. The new
    session carries `parentSessionId` and `forkedAtMessageIndex`
    metadata so the sidebar can render a relationship hint and
    future features (compare-vs-parent, merge) have the linkage.

    Raises ``ValueError`` when the source session doesn't exist
    or the fork index is out of range.
    """
    with state._lock:
        source = next(
            (s for s in state.chat_sessions if s.get("id") == source_session_id),
            None,
        )
        if source is None:
            raise ValueError(f"Source session not found: {source_session_id}")
        messages = source.get("messages") or []
        if fork_at_message_index < 0 or fork_at_message_index >= len(messages):
            raise ValueError(
                f"fork_at_message_index {fork_at_message_index} out of range "
                f"(session has {len(messages)} messages)"
            )

        fork_title = title or f"{source.get('title', 'Chat')} (fork)"
        new_id = f"session-{uuid.uuid4().hex[:8]}"
        new_session: dict[str, Any] = {
            "id": new_id,
            "title": fork_title,
            "updatedAt": state._time_label(),
            "pinned": False,
            # Carry the runtime profile so the fork resumes on the
            # same model + cache config as the parent.
            "model": source.get("model"),
            "modelRef": source.get("modelRef"),
            "canonicalRepo": source.get("canonicalRepo"),
            "modelSource": source.get("modelSource"),
            "modelPath": source.get("modelPath"),
            "modelBackend": source.get("modelBackend"),
            "thinkingMode": source.get("thinkingMode") or "off",
            "cacheLabel": source.get("cacheLabel"),
            "cacheStrategy": source.get("cacheStrategy"),
            "cacheBits": source.get("cacheBits"),
            "fp16Layers": source.get("fp16Layers"),
            "fusedAttention": source.get("fusedAttention"),
            "fitModelInMemory": source.get("fitModelInMemory"),
            "contextTokens": source.get("contextTokens"),
            "speculativeDecoding": source.get("speculativeDecoding"),
            "dflashDraftModel": source.get("dflashDraftModel"),
            "treeBudget": source.get("treeBudget"),
            # Branching linkage so the UI can render the
            # parent-child relationship and so future features
            # (diff, merge) have the tie.
            "parentSessionId": source_session_id,
            "forkedAtMessageIndex": fork_at_message_index,
            "messages": copy.deepcopy(messages[: fork_at_message_index + 1]),
        }
        state.chat_sessions.insert(0, new_session)
        state.add_activity(
            "Chat session forked",
            f"{source.get('title', 'Chat')} → {fork_title}",
        )
        persist_sessions(state)
        return new_session


def update_session(
    state: ChaosEngineState, session_id: str, request: UpdateSessionRequest
) -> dict[str, Any]:
    with state._lock:
        session = ensure_session(state, session_id=session_id)
        fields_set = getattr(request, "model_fields_set", set())
        if request.title is not None and request.title.strip():
            session["title"] = request.title.strip()
        if request.model is not None:
            session["model"] = request.model
        if "modelRef" in fields_set:
            session["modelRef"] = request.modelRef
        if "canonicalRepo" in fields_set:
            session["canonicalRepo"] = request.canonicalRepo
        if "modelSource" in fields_set:
            session["modelSource"] = request.modelSource
        if "modelPath" in fields_set:
            session["modelPath"] = request.modelPath
        if "modelBackend" in fields_set:
            session["modelBackend"] = request.modelBackend
        if "thinkingMode" in fields_set:
            session["thinkingMode"] = request.thinkingMode
        if "pinned" in fields_set:
            session["pinned"] = request.pinned
        if "cacheStrategy" in fields_set:
            session["cacheStrategy"] = request.cacheStrategy
        if "cacheBits" in fields_set:
            session["cacheBits"] = request.cacheBits
        if "fp16Layers" in fields_set:
            session["fp16Layers"] = request.fp16Layers
        if "fusedAttention" in fields_set:
            session["fusedAttention"] = request.fusedAttention
        if "fitModelInMemory" in fields_set:
            session["fitModelInMemory"] = request.fitModelInMemory
        if "contextTokens" in fields_set:
            session["contextTokens"] = request.contextTokens
        if "speculativeDecoding" in fields_set:
            session["speculativeDecoding"] = request.speculativeDecoding
        if "treeBudget" in fields_set:
            session["treeBudget"] = request.treeBudget
        if "dflashDraftModel" in fields_set:
            session["dflashDraftModel"] = request.dflashDraftModel
        if "workspaceId" in fields_set:
            # Phase 3.7: empty string clears the assignment.
            session["workspaceId"] = request.workspaceId or None
        if request.messages is not None:
            session["messages"] = request.messages
        session["updatedAt"] = state._time_label()
        promote_session(state, session)
        state.add_activity("Thread updated", session["title"])
        persist_sessions(state)
        return session


def delete_session(state: ChaosEngineState, session_id: str) -> dict[str, Any]:
    with state._lock:
        target = next((s for s in state.chat_sessions if s.get("id") == session_id), None)
        if not target:
            raise HTTPException(status_code=404, detail="Session not found.")
        state.chat_sessions = [s for s in state.chat_sessions if s.get("id") != session_id]
        state.add_log("chat", "info", f"Session deleted: {target.get('title', session_id)}")
        persist_sessions(state)
        return {"deleted": session_id}
