"""Correct chat request settings the loaded model or engine cannot use.

A setting the engine rejects fails the whole reply (llama-server answers 400 to
a repeat penalty of 0), and one the model cannot use quietly yields nothing or
noise (an image sent to a text-only model, XTC with a zero threshold). Every
correction is returned as a note the reply carries, so a changed or ignored
setting never goes unexplained.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

# Engines that read each sampler. mlx-lm builds its sampler from top_p, top_k,
# min_p, repeat_penalty, XTC and the seed; the rest is llama-server only.
_LLAMA_ONLY_SAMPLERS: tuple[tuple[str, str], ...] = (
    ("mirostat", "Mirostat"),
    ("dry_multiplier", "DRY"),
    ("logprobs", "token probabilities"),
)

# llama.cpp's own default; XTC with a threshold of 0 removes every likely token.
_XTC_DEFAULT_THRESHOLD = 0.1
_XTC_MIN_THRESHOLD = 0.01


@dataclass
class ChatGuard:
    images: list[str] | None
    samplers: dict[str, Any] | None
    jsonSchema: dict[str, Any] | None
    notes: list[str] = field(default_factory=list)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def guard_chat_request(
    *,
    engine: str,
    vision_enabled: bool,
    images: list[str] | None,
    samplers: dict[str, Any] | None,
    json_schema: dict[str, Any] | None,
) -> ChatGuard:
    notes: list[str] = []
    kept_samplers = dict(samplers) if samplers else None

    if images and engine in {"llama.cpp", "mlx"} and not vision_enabled:
        count = len(images)
        notes.append(
            f"This model cannot read images, so {count} attached image{'s were' if count != 1 else ' was'} not sent."
        )
        images = None

    if kept_samplers is not None:
        penalty = kept_samplers.get("repeat_penalty")
        if penalty is not None and (not _is_finite_number(penalty) or penalty <= 0):
            kept_samplers.pop("repeat_penalty", None)
            notes.append("Repeat penalty must be above 0, so it was not applied.")

        top_p = kept_samplers.get("top_p")
        if top_p is not None and (not _is_finite_number(top_p) or top_p <= 0):
            kept_samplers.pop("top_p", None)

        probability = kept_samplers.get("xtc_probability")
        if probability is not None:
            if not _is_finite_number(probability) or probability <= 0:
                kept_samplers.pop("xtc_probability", None)
                kept_samplers.pop("xtc_threshold", None)
            else:
                threshold = kept_samplers.get("xtc_threshold")
                if threshold is None:
                    kept_samplers["xtc_threshold"] = _XTC_DEFAULT_THRESHOLD
                elif not _is_finite_number(threshold) or threshold < _XTC_MIN_THRESHOLD:
                    kept_samplers.pop("xtc_probability", None)
                    kept_samplers.pop("xtc_threshold", None)
                    notes.append(
                        "XTC with a threshold near 0 removes every likely token and produces noise, so it was not applied."
                    )

        if engine != "llama.cpp":
            unused = [label for key, label in _LLAMA_ONLY_SAMPLERS if kept_samplers.get(key) not in (None, 0, False)]
            if unused:
                notes.append(f"Not supported by this engine, so it was ignored: {', '.join(unused)}.")

        if not kept_samplers:
            kept_samplers = None

    if json_schema and engine in {"mlx", "mtplx"}:
        notes.append("This engine does not enforce a JSON schema, so the reply may not match it.")

    return ChatGuard(images=images, samplers=kept_samplers, jsonSchema=json_schema, notes=notes)


def empty_reply_note(
    *,
    completion_tokens: int,
    finish_reason: str | None,
    had_reasoning: bool,
    json_schema: dict[str, Any] | None,
) -> str | None:
    """Explain a reply that produced tokens but no visible text."""
    if completion_tokens <= 0:
        return None
    ran_out = (finish_reason or "") == "length"
    if json_schema and ran_out:
        return (
            "The reply is empty: with a JSON schema this model only produced whitespace or hidden thinking. "
            "Remove the schema, turn Thinking off, or try another model."
        )
    if had_reasoning and ran_out:
        return (
            "The reply is empty: the model used its whole token budget on thinking. "
            "Raise Max tokens or turn Thinking off."
        )
    if ran_out:
        return "The reply is empty: the token budget ran out before any text. Raise Max tokens."
    return "The model returned no visible text for this prompt."


SCHEMA_FALLBACK_NOTE = (
    "This model produced no text under the JSON grammar (reasoning models can spend the whole reply "
    "on thinking), so the reply was asked for as plain JSON instead. Check it before you rely on it."
)


def schema_fallback_prompt(system_prompt: str | None, schema: dict[str, Any]) -> str:
    """The system prompt for a retry that asks for JSON in words instead of through a grammar."""
    instruction = (
        "Reply with only one JSON value that matches this JSON schema. No explanation, no markdown fence.\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    return f"{system_prompt}\n\n{instruction}" if system_prompt else instruction


def strip_json_fence(text: str) -> str:
    """Remove a single ```json fence around the whole reply, which models add even when told not to."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```") and stripped.count("```") == 2:
        inner = stripped[3:-3]
        newline = inner.find("\n")
        first = inner[:newline].strip().lower() if newline != -1 else ""
        if newline != -1 and first in ("", "json"):
            return inner[newline + 1:].strip()
    return text


def merge_notes(existing: str | None, notes: list[str]) -> str | None:
    parts = [part for part in [existing, *notes] if part]
    return " · ".join(parts) if parts else None


class RequestNotUsable(RuntimeError):
    """The request cannot run as sent (conversation larger than the loaded context).

    A RuntimeError so every existing handler still catches it; the HTTP routes
    answer 400 for it instead of 500, because the person can fix it.
    """


def llama_error(detail: str, fallback: str) -> RuntimeError:
    """The exception for a llama-server error body: 400-class for a context overflow."""
    message = friendly_llama_error(detail) if detail else fallback
    try:
        kind = str((json.loads(detail).get("error") or {}).get("type") or "")
    except (ValueError, AttributeError, TypeError):
        kind = ""
    if kind == "exceed_context_size_error":
        return RequestNotUsable(message)
    return RuntimeError(message)


def friendly_llama_error(detail: str) -> str:
    """Turn a llama-server error body into a message a person can act on."""
    try:
        error = json.loads(detail).get("error") or {}
    except (ValueError, AttributeError):
        return detail
    if not isinstance(error, dict):
        return detail
    kind = str(error.get("type") or "")
    message = str(error.get("message") or "")
    if kind == "exceed_context_size_error":
        prompt_tokens = error.get("n_prompt_tokens")
        context = error.get("n_ctx")
        if prompt_tokens and context:
            return (
                f"The conversation is about {prompt_tokens} tokens, but the model is loaded with a {context}-token "
                "context. Raise Context in the launch settings, shorten the message or start a new chat."
            )
    if "image input is not supported" in message:
        return "This model cannot read images. Remove the attachment or load a vision model."
    return message or detail
