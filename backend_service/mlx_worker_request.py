"""Request-shaping helpers for the MLX worker.

Pure-ish utilities the worker calls per-request: message normalisation,
top-k logprob extraction from an mlx-lm GenerationResponse, sampler
construction, deterministic seeding, and tool-schema injection into the
system prompt for open-source chat models that lack a native function-call
API.

Extracted from ``backend_service/mlx_worker.py`` as part of the v0.8.0
refactor. The names are re-exported from ``mlx_worker`` so existing
imports (e.g. ``from backend_service.mlx_worker import _sanitize_messages``
in ``vllm_engine``) keep resolving.
"""

from __future__ import annotations

from typing import Any


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return " ".join(parts)
    return str(content or "")


def _sanitize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Ensure strict role alternation (user/assistant) after an optional system message.

    - Removes empty assistant messages.
    - Merges consecutive same-role messages with a newline separator.
    """
    sanitized: list[dict[str, str]] = []
    for msg in messages:
        content = msg.get("content", "").strip()
        role = msg.get("role", "")
        # Drop empty assistant messages (from failed/mock responses)
        if role == "assistant" and not content:
            continue
        # Merge consecutive same-role messages
        if sanitized and sanitized[-1]["role"] == role and role != "system":
            sanitized[-1]["content"] += "\n" + content
        else:
            sanitized.append({"role": role, "content": content})
    return sanitized


def _extract_top_logprobs(
    response: Any,
    tokenizer: Any,
    top_k: int,
) -> list[dict[str, Any]] | None:
    """Phase 3.3 follow-up: extract top-k logprob entries from an
    mlx-lm GenerationResponse for the just-emitted token.

    Returns a list with a single entry shaped like the OpenAI
    `logprobs.content[]` payload — token + logprob + alternatives —
    so the frontend overlay treats MLX and llama-server output
    identically. Returns None on any failure (missing logprobs,
    unsupported tensor shape, etc.) — logprobs are diagnostic, not
    correctness-critical.
    """
    if top_k <= 0:
        return None
    logprobs = getattr(response, "logprobs", None)
    chosen_token_id = getattr(response, "token", None)
    if logprobs is None or chosen_token_id is None:
        return None
    try:
        import numpy as np  # noqa: WPS433 — keep import lazy

        arr = np.array(logprobs, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return None
        # argpartition gets top-k unsorted; sort just the slice.
        k = min(int(top_k), int(arr.size))
        if k >= int(arr.size):
            top_idx = np.argsort(-arr)
        else:
            partial = np.argpartition(-arr, k - 1)[:k]
            top_idx = partial[np.argsort(-arr[partial])]
        alternatives: list[dict[str, Any]] = []
        for token_id in top_idx[:k].tolist():
            try:
                token_text = tokenizer.decode([int(token_id)])
            except Exception:
                token_text = ""
            alternatives.append({
                "token": token_text,
                "logprob": float(arr[token_id]),
            })
        try:
            chosen_text = tokenizer.decode([int(chosen_token_id)])
        except Exception:
            chosen_text = ""
        chosen_logprob: float | None
        try:
            chosen_logprob = float(arr[int(chosen_token_id)])
        except Exception:
            chosen_logprob = None
        return [{
            "token": chosen_text,
            "logprob": chosen_logprob,
            "alternatives": alternatives,
        }]
    except Exception:
        return None


def _build_mlx_sampler(request: dict[str, Any]) -> Any:
    """Phase 2.2: build an mlx-lm sampler with whichever Phase 2.2 sampler
    overrides the installed `make_sampler` actually supports.

    `mlx_lm.sample_utils.make_sampler` has gained kwargs across versions
    (top_p, top_k, min_p, ...). Call sites used to pass `temp` only — we
    now collect the request's `samplers` block and forward whatever
    survives a signature filter, so newer mlx-lm builds get the full
    sampler chain while older builds fall back gracefully.
    """
    import inspect

    from mlx_lm.sample_utils import make_sampler

    kwargs: dict[str, Any] = {"temp": float(request.get("temperature") or 0.0)}
    samplers = request.get("samplers") or {}
    if isinstance(samplers, dict):
        for src in ("top_p", "top_k", "min_p"):
            value = samplers.get(src)
            if value is not None:
                kwargs[src] = value

    try:
        sig = inspect.signature(make_sampler)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        filtered = {"temp": kwargs["temp"]}
    return make_sampler(**filtered)


def _sampler_seed(request: dict[str, Any]) -> int | None:
    samplers = request.get("samplers") or {}
    if not isinstance(samplers, dict):
        return None
    value = samplers.get("seed")
    if value is None:
        return None
    try:
        seed = int(value)
    except (TypeError, ValueError):
        return None
    return seed if seed >= 0 else None


def _apply_mlx_seed(request: dict[str, Any]) -> None:
    seed = _sampler_seed(request)
    if seed is None:
        return
    try:
        import mlx.core as mx
        seed_fn = getattr(getattr(mx, "random", None), "seed", None)
        if callable(seed_fn):
            seed_fn(seed)
    except Exception:
        pass


def _format_tools_for_prompt(tools: list[dict[str, Any]] | None) -> str | None:
    """Format tool schemas into a system prompt block for open-source models.

    Since MLX models don't have a native function-calling API, we inject
    tool descriptions into the system prompt so the model knows what tools
    are available and how to call them.
    """
    if not tools:
        return None

    lines = [
        "You have access to the following tools. To use a tool, respond with a JSON block wrapped in <tool_call> tags.",
        "Example: <tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"2+2\"}}</tool_call>",
        "",
        "Available tools:",
    ]
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "")
            req = " (required)" if pname in required else ""
            param_parts.append(f"    - {pname}: {ptype}{req} — {pdesc}")

        lines.append(f"\n- {name}: {desc}")
        if param_parts:
            lines.append("  Parameters:")
            lines.extend(param_parts)

    lines.append("")
    lines.append("If you don't need a tool, just respond normally without <tool_call> tags.")
    return "\n".join(lines)
