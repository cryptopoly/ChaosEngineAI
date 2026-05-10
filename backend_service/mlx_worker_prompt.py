"""Prompt assembly + transcript-loop guard for the MLX worker.

Two pieces:

* ``TranscriptLoopFilter`` + ``_trim_transcript_continuation`` —
  suppress models that keep continuing a fake ``USER:`` / ``ASSISTANT:``
  transcript indefinitely, used only when the tokenizer had no chat
  template and we fell back to a raw text prompt.
* ``_build_prompt_text`` — render the chat template against the
  tokeniser, with the Gemma-family ``fold_system_into_first_user``
  auto-fix and a clean fallback to ``_fallback_chat_prompt`` when
  ``apply_chat_template`` is unavailable.

Plus a couple of small predicates the worker reuses.

Extracted from ``backend_service/mlx_worker.py`` as part of the v0.8.0
refactor. Re-exported from ``mlx_worker`` so existing
``from backend_service.mlx_worker import _build_prompt_text`` etc.
imports keep working.
"""

from __future__ import annotations

import re
from typing import Any

from backend_service.mlx_worker_request import (
    _normalize_message_content,
    _sanitize_messages,
)


_TRANSCRIPT_ROLE_LINE_RE = re.compile(r"^\s*(SYSTEM|USER|ASSISTANT):\s*(.*)$", re.IGNORECASE)


class TranscriptLoopFilter:
    """Suppress plain transcript continuations like USER:/ASSISTANT: loops.

    This is only used when we had to fall back to a raw text chat prompt because
    the tokenizer had no usable chat template. In that mode, some models keep
    continuing the transcript forever instead of answering once.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._stopped = False
        self._at_start = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    def feed(self, text: str) -> str:
        if self._stopped:
            return ""
        self._buffer += text
        output_parts: list[str] = []
        while "\n" in self._buffer and not self._stopped:
            line, self._buffer = self._buffer.split("\n", 1)
            processed = self._process_line(line)
            if processed:
                output_parts.append(processed + "\n")
        return "".join(output_parts)

    def flush(self) -> str:
        if self._stopped:
            self._buffer = ""
            return ""
        if not self._buffer:
            return ""
        remaining = self._process_line(self._buffer)
        self._buffer = ""
        return remaining

    def _process_line(self, line: str) -> str:
        if self._stopped:
            return ""
        match = _TRANSCRIPT_ROLE_LINE_RE.match(line)
        if match:
            role = match.group(1).upper()
            content = match.group(2)
            if role == "ASSISTANT" and self._at_start:
                self._at_start = False
                return content
            self._stopped = True
            return ""
        if line.strip():
            self._at_start = False
        return line


def _plain_chat_fallback_active(runtime_note: str | None) -> bool:
    return bool(runtime_note and "plain chat fallback prompt" in runtime_note.lower())


def _trim_transcript_continuation(text: str) -> tuple[str, bool]:
    filter_ = TranscriptLoopFilter()
    emitted = filter_.feed(text)
    emitted += filter_.flush()
    return emitted.strip(), filter_.stopped


def _fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        lines.append(f"{message['role'].upper()}: {message['content']}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def _merge_runtime_notes(*notes: str | None) -> str | None:
    merged = " ".join(note.strip() for note in notes if note and note.strip())
    return merged or None


def _should_retry_cache_failure(exc: BaseException) -> bool:
    detail = str(exc).lower()
    return (
        "broadcast" in detail
        or "shape" in detail
        or "create_attention_mask" in detail
        or "swapaxes" in detail
    )


def _build_prompt_text(
    tokenizer: Any,
    history: list[dict[str, Any]],
    prompt: str,
    system_prompt: str | None,
    model_ref: str | None = None,
) -> tuple[str, str | None]:
    # Phase 3.8: detect chat-template quirks at render time and apply
    # the matching auto-fix. Today: Gemma family rejects the system role
    # entirely, so we fold the system prompt into the first user message
    # before handing off to apply_chat_template. The report's
    # `to_runtime_note()` surfaces the fix to the UI's substrate badge.
    from backend_service.helpers.chat_template import (
        fold_system_into_first_user,
        inspect_chat_template,
        is_gemma_family,
    )

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for message in history:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        messages.append({"role": role, "content": _normalize_message_content(message.get("text", ""))})
    messages.append({"role": "user", "content": prompt})
    messages = _sanitize_messages(messages)

    template_note: str | None = None
    if is_gemma_family(model_ref):
        messages = fold_system_into_first_user(messages)
        report = inspect_chat_template(getattr(tokenizer, "chat_template", None), model_ref)
        template_note = report.to_runtime_note()

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            rendered = apply_template(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(rendered, str):
                return rendered, template_note
        except TypeError:
            try:
                rendered = apply_template(messages, add_generation_prompt=True)
                if isinstance(rendered, str):
                    return rendered, template_note
                if isinstance(rendered, list):
                    return tokenizer.decode(rendered), template_note
            except Exception as exc:  # pragma: no cover - exercised via fallback path below
                reason = str(exc).strip() or exc.__class__.__name__
                return (
                    _fallback_chat_prompt(messages),
                    f"Tokenizer chat template was unavailable, so MLX used a plain chat fallback prompt. ({reason})",
                )
        except Exception as exc:
            reason = str(exc).strip() or exc.__class__.__name__
            return (
                _fallback_chat_prompt(messages),
                f"Tokenizer chat template was unavailable, so MLX used a plain chat fallback prompt. ({reason})",
            )

    return (
        _fallback_chat_prompt(messages),
        "Tokenizer chat template was unavailable, so MLX used a plain chat fallback prompt.",
    )
