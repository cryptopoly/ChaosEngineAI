"""Multimodal (mlx-vlm) prompt + arg + generation helpers for the worker.

Five pieces lifted out of ``WorkerState``:

* ``decode_images_to_paths`` — materialise base64 image blobs into a temp
  directory; returns the file paths mlx-vlm's ``image=`` kwarg expects.
* ``format_multimodal_prompt`` — render the chat history through
  ``mlx_vlm.prompt_utils.apply_chat_template`` (or fall back to the plain
  text builder) accounting for image placeholder count.
* ``vlm_generate_kwargs`` — translate the worker's request shape into the
  ``temperature`` / ``top_p`` / ``max_tokens`` kwargs mlx-vlm's
  ``generate`` / ``stream_generate`` accept.
* ``generate_multimodal`` — synchronous mlx-vlm path: decode any images,
  run ``mlx_vlm.generate``, apply the thinking-token filter, return the
  text-only response shape.
* ``stream_generate_multimodal`` — streaming mlx-vlm path: emits chunks
  via the same ``_emit`` JSON protocol the text-only path uses so the
  caller sees the same shape regardless of which engine produced the run.

Extracted from ``backend_service/mlx_worker.py`` as part of the v0.8.0
refactor. ``WorkerState`` now thin-wraps each function.
"""

from __future__ import annotations

import base64
import binascii
import tempfile
from pathlib import Path
from typing import Any

from backend_service.mlx_worker_io import _emit
from backend_service.mlx_worker_prompt import _fallback_chat_prompt
from backend_service.mlx_worker_request import (
    _apply_mlx_seed,
    _normalize_message_content,
    _sanitize_messages,
)
from backend_service.reasoning_split import (
    ThinkingTokenFilter,
    reasoning_delimiters_for,
    strip_harmony_boilerplate,
)


def decode_images_to_paths(images_b64: list[str], temp_dir: str) -> list[str]:
    """Decode base64-encoded images into ``temp_dir`` and return paths.

    The chat payload sends each image as a raw base64 string (no
    data-URL prefix — that's stripped client-side in
    ``ChatComposer.tsx``). mlx-vlm's ``image=`` kwarg accepts a list
    of file paths, so we materialise each blob to a temp file with
    a deterministic suffix.
    """
    paths: list[str] = []
    for index, blob in enumerate(images_b64 or []):
        if not blob:
            continue
        try:
            raw = base64.b64decode(blob, validate=False)
        except (binascii.Error, ValueError):
            # Skip malformed entries rather than aborting the whole
            # generation — the model will still answer using text.
            continue
        path = Path(temp_dir) / f"img_{index:03d}.png"
        path.write_bytes(raw)
        paths.append(str(path))
    return paths


def format_multimodal_prompt(
    processor: Any,
    config: Any,
    tokenizer: Any,
    request: dict[str, Any],
    num_images: int,
) -> str:
    """Render the chat history into a single prompt string the
    VLM tokenizer expects, accounting for ``num_images`` image
    placeholders. Falls back to the plain-text prompt builder when
    the processor doesn't expose ``apply_chat_template`` or the
    helper raises (some VLMs ship templates that reject our
    history shape).
    """
    history = list(request.get("history") or [])
    prompt = str(request.get("prompt") or "")
    system_prompt = request.get("systemPrompt")
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    for message in history:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        messages.append(
            {"role": role, "content": _normalize_message_content(message.get("text", ""))}
        )
    messages.append({"role": "user", "content": prompt})
    messages = _sanitize_messages(messages)

    try:
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore[import-untyped]
    except ImportError:
        return _fallback_chat_prompt(messages)

    try:
        rendered = apply_chat_template(
            processor,
            config or {},
            messages,
            add_generation_prompt=True,
            num_images=num_images,
        )
    except Exception:
        return _fallback_chat_prompt(messages)

    if isinstance(rendered, str):
        return rendered
    if isinstance(rendered, list):
        decoder = getattr(tokenizer, "decode", None) if tokenizer is not None else None
        if callable(decoder):
            try:
                return decoder(rendered)
            except Exception:
                pass
    return _fallback_chat_prompt(messages)


def vlm_generate_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    """Sampling kwargs accepted by ``mlx_vlm.generate`` /
    ``stream_generate``. The VLM API takes ``temperature`` and
    ``top_p`` directly (no separate sampler factory like mlx-lm),
    so we forward only the knobs that map cleanly. Missing fields
    fall back to the underlying mlx-vlm defaults.
    """
    kwargs: dict[str, Any] = {
        "max_tokens": int(request.get("maxTokens") or 256),
    }
    temperature = request.get("temperature")
    if temperature is not None:
        try:
            kwargs["temperature"] = float(temperature)
        except (TypeError, ValueError):
            pass
    top_p = request.get("topP")
    samplers = request.get("samplers") or {}
    if top_p is None and isinstance(samplers, dict):
        top_p = samplers.get("top_p")
    if top_p is not None:
        try:
            kwargs["top_p"] = float(top_p)
        except (TypeError, ValueError):
            pass
    return kwargs


def generate_multimodal(
    *,
    model: Any,
    processor: Any,
    tokenizer: Any,
    config: Any,
    loaded_model_ref: str | None,
    request: dict[str, Any],
) -> dict[str, Any]:
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
    kwargs = vlm_generate_kwargs(request)

    with tempfile.TemporaryDirectory(prefix="chaosengine-mm-") as tmpdir:
        image_paths = decode_images_to_paths(images_b64, tmpdir)
        prompt_text = format_multimodal_prompt(
            processor, config, tokenizer, request, num_images=len(image_paths)
        )
        if image_paths:
            result = vlm_generate(
                model, processor, prompt_text, image=image_paths, **kwargs,
            )
        else:
            result = vlm_generate(
                model, processor, prompt_text, **kwargs,
            )

    raw_text = getattr(result, "text", None) or str(result)
    thinking_mode = request.get("thinkingMode") or "off"
    open_tag, close_tag = reasoning_delimiters_for(loaded_model_ref)
    think_filter = ThinkingTokenFilter(
        detect_raw_reasoning=(thinking_mode != "off"),
        open_tag=open_tag,
        close_tag=close_tag,
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


def stream_generate_multimodal(
    *,
    model: Any,
    processor: Any,
    tokenizer: Any,
    config: Any,
    loaded_model_ref: str | None,
    request: dict[str, Any],
) -> None:
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
    kwargs = vlm_generate_kwargs(request)
    thinking_mode = request.get("thinkingMode") or "off"
    open_tag, close_tag = reasoning_delimiters_for(loaded_model_ref)
    think_filter = ThinkingTokenFilter(
        detect_raw_reasoning=(thinking_mode != "off"),
        open_tag=open_tag,
        close_tag=close_tag,
    )

    text_parts: list[str] = []
    completion_tokens = 0
    last_chunk: Any = None

    with tempfile.TemporaryDirectory(prefix="chaosengine-mm-") as tmpdir:
        image_paths = decode_images_to_paths(images_b64, tmpdir)
        prompt_text = format_multimodal_prompt(
            processor, config, tokenizer, request, num_images=len(image_paths)
        )
        if image_paths:
            stream = vlm_stream(
                model, processor, prompt_text, image=image_paths, **kwargs,
            )
        else:
            stream = vlm_stream(
                model, processor, prompt_text, **kwargs,
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
