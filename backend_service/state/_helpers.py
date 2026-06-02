"""Module-level helpers for ``ChaosEngineState``.

Pure functions + small constants that don't depend on ``self``. Kept as a
sibling module so the parent ``__init__`` can stay focused on the class
itself; the package ``__init__`` re-exports each helper so the historical
``backend_service.state._<name>`` import paths keep working.

Extracted from ``state/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import HTTPException


_CATALOG_REF_ALIASES = {
    # The non-it MLX Gemma variant lacks a chat template and is a poor default
    # for the chat picker. Resolve old saved refs to the instruct checkpoint.
    "mlx-community/gemma-4-26b-a4b-5bit": "mlx-community/gemma-4-26b-a4b-it-5bit",
}


def _compose_chat_system_prompt(system_prompt: str | None, thinking_mode: str | None = None) -> str:
    return (system_prompt or "").strip()


def _build_sampler_overrides(request: Any) -> dict[str, Any]:
    """Phase 2.2: collect the request's sampler overrides into a flat dict
    keyed using the llama-server `/v1/chat/completions` field names.

    The dict contains only fields the user actually set — `None` defaults
    are skipped so the backend's defaults stay in force when the UI sends
    no override. Both engines treat unknown keys as no-ops, so the output
    is forward-compatible across llama-server / mlx-lm versions.
    """
    overrides: dict[str, Any] = {}

    def _put(dst: str, value: Any) -> None:
        if value is not None:
            overrides[dst] = value

    _put("top_p", getattr(request, "topP", None))
    _put("top_k", getattr(request, "topK", None))
    _put("min_p", getattr(request, "minP", None))
    _put("repeat_penalty", getattr(request, "repeatPenalty", None))
    _put("seed", getattr(request, "seed", None))
    mirostat_mode = getattr(request, "mirostatMode", None)
    if mirostat_mode is not None:
        overrides["mirostat"] = mirostat_mode
    _put("mirostat_tau", getattr(request, "mirostatTau", None))
    _put("mirostat_eta", getattr(request, "mirostatEta", None))
    # Phase 3.3: when the user enables logprobs on a request the
    # frontend sends a top-k count; map it onto llama-server's
    # `logprobs` + `top_logprobs` parameters so the response delta
    # carries the per-token info.
    logprobs = getattr(request, "logprobs", None)
    if logprobs is not None and logprobs > 0:
        overrides["logprobs"] = True
        overrides["top_logprobs"] = int(logprobs)
    return overrides


def _estimate_tokens(text: str) -> int:
    """Cheap, deliberately CONSERVATIVE token estimate (no tokenizer here).

    Assumes ~3 chars/token vs the ~4 typical for English so the history
    window UNDER-fills the context rather than risking an overflow the MLX
    path can't recover from. Code and CJK are denser than English, so
    erring small protects them too. Off by a constant factor — fine for a
    safety budget, not for billing.
    """
    return (len(text) // 3) + 1


def _history_token_budget(
    *,
    context_tokens: int,
    max_tokens: int,
    system_prompt: str | None,
    prompt: str | None,
) -> int:
    """Token budget left for *prior* history after reserving room for the
    system prompt, the current user prompt, the generation, and chat-template
    overhead. Floors at 512 so a single recent turn is always kept.
    """
    reserved = (
        _estimate_tokens(system_prompt or "")
        + _estimate_tokens(prompt or "")
        + int(max_tokens or 0)
        + 512  # chat-template + role-tag + tool-schema overhead headroom
    )
    return max(512, int(context_tokens or 0) - reserved)


def _build_history_with_reasoning(
    messages: list[dict[str, Any]],
    *,
    preserve_reasoning: bool,
    token_budget: int | None = None,
) -> list[dict[str, Any]]:
    """Project a session's stored messages into the history list passed to the
    inference layer.

    When `preserve_reasoning` is true and an assistant message has a
    `reasoning` field captured by ThinkingTokenFilter on a previous turn,
    the reasoning is re-emitted inside `<think>...</think>` tags ahead of
    the visible answer. (Upstream chat templates for Qwen3 / DeepSeek-R1
    actually strip prior reasoning, so the live chat path now passes
    `preserve_reasoning=False`; the option is kept for callers that want it.)
    Falsy / missing reasoning is skipped, so this is safe to call
    unconditionally.

    When `token_budget` is set, a sliding window keeps every system message
    plus the NEWEST conversation turns that fit the budget (estimated, no
    tokenizer), dropping the oldest. This bounds prompt growth across a long
    chat — preventing silent truncation on llama.cpp and out-of-context
    errors on MLX. ``None`` disables windowing (unchanged behaviour).
    """
    history: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        text = str(message.get("text") or "")
        if (
            preserve_reasoning
            and role == "assistant"
            and message.get("reasoning")
        ):
            reasoning_str = str(message["reasoning"]).strip()
            if reasoning_str:
                text = f"<think>\n{reasoning_str}\n</think>\n\n{text}"
        history.append({"role": role, "text": text})

    if token_budget is None or token_budget <= 0:
        return history

    # System messages are always kept; window the conversation tail.
    system_msgs = [m for m in history if m["role"] == "system"]
    convo = [m for m in history if m["role"] != "system"]
    used = sum(_estimate_tokens(m["text"]) for m in system_msgs)
    kept_tail: list[dict[str, Any]] = []
    for message in reversed(convo):
        cost = _estimate_tokens(message["text"])
        # Always keep the most recent turn even if it alone blows the budget;
        # dropping the latest context is worse than a small overflow the
        # engine can still truncate.
        if kept_tail and used + cost > token_budget:
            break
        used += cost
        kept_tail.append(message)
    kept_tail.reverse()
    return system_msgs + kept_tail


_TITLE_LEADING_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(?:please\s+)+",
        r"^(?:can|could|would|will)\s+you\s+",
        r"^(?:can|could|would|will)\s+we\s+",
        r"^i\s+(?:need|want|would\s+like)\s+(?:you\s+to\s+)?",
        r"^help\s+me\s+",
        r"^make\s+it\s+so\s+that\s+",
        r"^tell\s+me\s+(?:about\s+)?(?:the\s+)?",
        r"^show\s+me\s+(?:how\s+to\s+)?",
        r"^give\s+me\s+",
    )
]


def _legacy_title_from_prompt(prompt: str | None) -> str:
    words = str(prompt or "").strip().split()
    return " ".join(words[:4]) or "New chat"


def _clean_prompt_for_title(prompt: str | None) -> str:
    text = str(prompt or "")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"^[\s#>*\-\d.)]+", "", text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    candidate = first_sentence.strip(" \t\n\r\"'`*_~:;,.!?()[]{}")
    for _ in range(4):
        previous = candidate
        for pattern in _TITLE_LEADING_PATTERNS:
            candidate = pattern.sub("", candidate).strip()
        candidate = re.sub(r"\s+please$", "", candidate, flags=re.IGNORECASE).strip()
        if candidate == previous:
            break
    return candidate.strip(" \t\n\r\"'`*_~:;,.!?()[]{}")


def _title_from_prompt(prompt: str | None) -> str:
    candidate = _clean_prompt_for_title(prompt)
    if not candidate:
        return "New chat"

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+'’._/#-]*", candidate)
    if not words:
        return "New chat"

    title = " ".join(words[:6]).strip()
    if len(title) > 64:
        title = title[:64].rsplit(" ", 1)[0] or title[:64]
    title = title.strip(" \t\n\r\"'`*_~:;,.!?()[]{}")
    if not title:
        return "New chat"
    if title.islower():
        title = title[:1].upper() + title[1:]
    return title


def _title_variant_pattern(base_title: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(base_title)}(?: \((\d+)\))?$")


def _read_text_tail(path: str | None, *, limit: int = 4096) -> str:
    if not path:
        return ""
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(content) <= limit:
        return content
    return content[-limit:]


def _spawn_snapshot_download(
    repo: str,
    env: dict[str, str],
    log_handle: Any,
    allow_patterns: list[str] | None = None,
) -> subprocess.Popen[str]:
    from backend_service.app import HF_SNAPSHOT_DOWNLOAD_HELPER

    args = [sys.executable, "-c", HF_SNAPSHOT_DOWNLOAD_HELPER, repo]
    # The helper treats an empty string as "no allowlist"; a JSON-encoded
    # list restricts the download to matching files. This is how we keep
    # diffusers video repos from ballooning to hundreds of GB when the
    # repo ships legacy standalone checkpoints alongside the pipeline
    # layout.
    args.append(json.dumps(allow_patterns) if allow_patterns else "")
    return subprocess.Popen(
        args,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def _normalize_remote_provider_api_base(raw_api_base: str) -> str:
    api_base = raw_api_base.strip().rstrip("/")
    parsed = urlparse(api_base)
    hostname = (parsed.hostname or "").lower()
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Remote provider API base must be a valid absolute URL.")
    if parsed.scheme == "https":
        return api_base
    if parsed.scheme == "http" and hostname in {"127.0.0.1", "localhost"}:
        return api_base
    raise HTTPException(
        status_code=400,
        detail="Remote providers must use HTTPS unless they point to localhost.",
    )
