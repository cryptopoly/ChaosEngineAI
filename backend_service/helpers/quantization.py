"""Quantization label parsers — pure helpers, no filesystem deps.

Used by the discovery scanner + the runtime layer to interpret
``config.json`` quantization metadata, infer bit-width labels from file
names, and explicitly reject NVFP4 / NVINT4 model_opt builds that the
MLX runtime can't load.

Extracted from ``backend_service/helpers/discovery.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.discovery`` so existing
``from backend_service.helpers.discovery import _quantization_label_from_text``
imports keep working.
"""

from __future__ import annotations

import re
from typing import Any


_UNSUPPORTED_MLX_QUANT_ALGOS = {"NVFP4", "NVINT4"}


def _quantization_label_from_text(text: str) -> str | None:
    lowered = text.lower()
    match = re.search(r"\b(q\d(?:_[a-z0-9]+)*)\b", lowered)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(\d+)[-_ ]?bit\b", lowered)
    if match:
        return f"{int(match.group(1))}-bit"
    if "bf16" in lowered or "bfloat16" in lowered:
        return "BF16"
    if "fp16" in lowered or "float16" in lowered:
        return "FP16"
    if "fp8" in lowered or "float8" in lowered:
        return "FP8"
    if "fp32" in lowered or "float32" in lowered:
        return "FP32"
    return None


def _quantization_algo_label(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    payload = config.get("quantization_config")
    if not isinstance(payload, dict):
        return None
    algo = payload.get("quant_algo")
    if isinstance(algo, str) and algo.strip():
        return algo.strip().upper()
    return None


def _unsupported_mlx_quantization_reason(config: dict[str, Any] | None) -> str | None:
    algo = _quantization_algo_label(config)
    if not algo or algo not in _UNSUPPORTED_MLX_QUANT_ALGOS:
        return None
    method = ""
    if isinstance(config, dict):
        payload = config.get("quantization_config")
        if isinstance(payload, dict):
            raw_method = payload.get("quant_method")
            if isinstance(raw_method, str) and raw_method.strip():
                method = raw_method.strip()
    method_label = f" (via {method})" if method else ""
    return (
        f"This model uses {algo} quantisation{method_label}, which is not supported by the MLX runtime. "
        f"It needs a CUDA/NVIDIA runtime such as vLLM with modelopt support, or a different build such as GGUF or MLX."
    )


def _mlx_quantization_bits(config: dict[str, Any] | None) -> int | None:
    if not isinstance(config, dict):
        return None
    if _unsupported_mlx_quantization_reason(config):
        return None
    for key in ("quantization", "quantization_config"):
        payload = config.get(key)
        if isinstance(payload, dict):
            bits = payload.get("bits")
            if isinstance(bits, (int, float)) and bits > 0:
                try:
                    return int(bits)
                except (TypeError, ValueError):
                    return None
    return None


def _dtype_quantization_label(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    candidates: list[Any] = [config.get("torch_dtype"), config.get("dtype")]
    for nested_key in ("text_config", "llm_config"):
        nested = config.get(nested_key)
        if isinstance(nested, dict):
            candidates.extend([nested.get("torch_dtype"), nested.get("dtype")])
    for value in candidates:
        if not value:
            continue
        label = _quantization_label_from_text(str(value))
        if label:
            return label
    return None
