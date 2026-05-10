"""Multimodal (mlx-vlm) prompt + arg helpers for the MLX worker.

Three pieces lifted out of ``WorkerState``:

* ``decode_images_to_paths`` — materialise base64 image blobs into a temp
  directory; returns the file paths mlx-vlm's ``image=`` kwarg expects.
* ``format_multimodal_prompt`` — render the chat history through
  ``mlx_vlm.prompt_utils.apply_chat_template`` (or fall back to the plain
  text builder) accounting for image placeholder count.
* ``vlm_generate_kwargs`` — translate the worker's request shape into the
  ``temperature`` / ``top_p`` / ``max_tokens`` kwargs mlx-vlm's
  ``generate`` / ``stream_generate`` accept.

Extracted from ``backend_service/mlx_worker.py`` as part of the v0.8.0
refactor. ``WorkerState._decode_images_to_paths`` etc. now thin-wrap
these so the test surface stays identical.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any

from backend_service.mlx_worker_prompt import _fallback_chat_prompt
from backend_service.mlx_worker_request import (
    _normalize_message_content,
    _sanitize_messages,
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
