"""Formatting utilities for labels, byte conversions, and context detection."""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any


def _bytes_to_gb(value: int | float) -> float:
    size_gb = float(value) / (1024 ** 3)
    if 0 < size_gb < 0.01:
        return 0.01
    return round(size_gb, 2)


def _context_label(value: int | None) -> str:
    if value is None:
        return "Unknown"
    if value >= 1_000_000:
        return f"{round(value / 1_000_000, 1)}M"
    if value >= 1_000:
        return f"{round(value / 1_000)}K"
    return str(value)


_CONTEXT_LABEL_RE = re.compile(r"^\s*([\d.]+)\s*([KMkm]?)\s*$")


def _parse_context_label(label: str | None) -> int | None:
    """Parse a human-readable context window label (e.g. '128K', '1M', '262K')
    into an integer token count. Returns None on failure."""
    if label is None:
        return None
    match = _CONTEXT_LABEL_RE.match(str(label))
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    unit = (match.group(2) or "").upper()
    if unit == "K":
        # Labels like "256K" / "262K" / "128K" / "32K" -> power-of-two style values
        # Use 1024 multiplier so "128K" -> 131072 which matches model configs.
        return int(round(value * 1024))
    if unit == "M":
        return int(round(value * 1_000_000))
    return int(round(value))


def _benchmark_label(model_name: str, *, cache_strategy: str, bits: int, fp16_layers: int, context_tokens: int) -> str:
    from cache_compression import registry as _strategy_registry
    strat = _strategy_registry.get(cache_strategy) or _strategy_registry.default()
    cache_label = strat.label(bits, fp16_layers)
    return f"{model_name} / {cache_label} / {_context_label(context_tokens)} ctx"


# GGUF value type codes (see https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
_GGUF_UINT8 = 0
_GGUF_INT8 = 1
_GGUF_UINT16 = 2
_GGUF_INT16 = 3
_GGUF_UINT32 = 4
_GGUF_INT32 = 5
_GGUF_FLOAT32 = 6
_GGUF_BOOL = 7
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_UINT64 = 10
_GGUF_INT64 = 11
_GGUF_FLOAT64 = 12

_GGUF_SCALAR_SIZES = {
    _GGUF_UINT8: 1,
    _GGUF_INT8: 1,
    _GGUF_UINT16: 2,
    _GGUF_INT16: 2,
    _GGUF_UINT32: 4,
    _GGUF_INT32: 4,
    _GGUF_FLOAT32: 4,
    _GGUF_BOOL: 1,
    _GGUF_UINT64: 8,
    _GGUF_INT64: 8,
    _GGUF_FLOAT64: 8,
}


def _read_gguf_context_length(path: Path) -> int | None:
    """Cheaply read the GGUF header and return the architecture's context_length.

    We parse just enough of the metadata KV section to find any key ending in
    ".context_length". We never read tensor data. Returns None on any failure.
    """
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                return None
            header = fh.read(4 + 8 + 8)  # version u32, tensor_count u64, kv_count u64
            if len(header) < 20:
                return None
            version, _tensor_count, kv_count = struct.unpack("<IQQ", header)
            if version < 2 or kv_count > 10_000:
                return None

            def _read_exact(n: int) -> bytes:
                buf = fh.read(n)
                if len(buf) != n:
                    raise EOFError
                return buf

            def _read_string() -> str:
                (length,) = struct.unpack("<Q", _read_exact(8))
                if length > 1 << 20:
                    raise ValueError("gguf string too long")
                return _read_exact(length).decode("utf-8", errors="replace")

            def _skip_value(vtype: int) -> None:
                if vtype in _GGUF_SCALAR_SIZES:
                    _read_exact(_GGUF_SCALAR_SIZES[vtype])
                    return
                if vtype == _GGUF_STRING:
                    _read_string()
                    return
                if vtype == _GGUF_ARRAY:
                    (inner_type,) = struct.unpack("<I", _read_exact(4))
                    (count,) = struct.unpack("<Q", _read_exact(8))
                    if count > 1 << 24:
                        raise ValueError("gguf array too long")
                    if inner_type in _GGUF_SCALAR_SIZES:
                        _read_exact(_GGUF_SCALAR_SIZES[inner_type] * count)
                        return
                    for _ in range(count):
                        _skip_value(inner_type)
                    return
                raise ValueError(f"unknown gguf type {vtype}")

            def _read_value(vtype: int) -> Any:
                if vtype == _GGUF_UINT32:
                    return struct.unpack("<I", _read_exact(4))[0]
                if vtype == _GGUF_INT32:
                    return struct.unpack("<i", _read_exact(4))[0]
                if vtype == _GGUF_UINT64:
                    return struct.unpack("<Q", _read_exact(8))[0]
                if vtype == _GGUF_INT64:
                    return struct.unpack("<q", _read_exact(8))[0]
                if vtype == _GGUF_UINT16:
                    return struct.unpack("<H", _read_exact(2))[0]
                if vtype == _GGUF_INT16:
                    return struct.unpack("<h", _read_exact(2))[0]
                _skip_value(vtype)
                return None

            best: int | None = None
            for _ in range(kv_count):
                key = _read_string()
                (vtype,) = struct.unpack("<I", _read_exact(4))
                if key.endswith(".context_length") or key == "context_length":
                    value = _read_value(vtype)
                    if isinstance(value, int) and value > 0:
                        if best is None or value > best:
                            best = value
                else:
                    _skip_value(vtype)
            return best
    except (OSError, ValueError, EOFError, struct.error):
        return None


def _read_config_max_context(config_path: Path) -> int | None:
    """Read an HF/MLX config.json and extract the model's max context length."""
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None

    candidates = (
        "max_position_embeddings",
        "max_sequence_length",
        "max_seq_len",
        "n_positions",
        "seq_length",
        "model_max_length",
    )
    base: int | None = None
    for key in candidates:
        value = config.get(key)
        if isinstance(value, (int, float)) and value > 0:
            base = int(value)
            break

    if base is None:
        # Some models nest under text_config / llm_config
        for nested_key in ("text_config", "llm_config"):
            nested = config.get(nested_key)
            if isinstance(nested, dict):
                for key in candidates:
                    value = nested.get(key)
                    if isinstance(value, (int, float)) and value > 0:
                        base = int(value)
                        break
            if base is not None:
                break

    if base is None:
        return None

    rope_scaling = config.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        factor = rope_scaling.get("factor")
        if isinstance(factor, (int, float)) and factor > 1:
            try:
                base = int(base * float(factor))
            except (TypeError, ValueError):
                pass
    return base


def _detect_model_max_context(path: Path, fmt: str) -> int | None:
    """Return the detected max context length for a discovered model, or None.

    Never raises -- returns None on any parse failure.
    """
    try:
        fmt_upper = (fmt or "").upper()
        if fmt_upper == "GGUF" or path.suffix.lower() == ".gguf":
            if path.is_file():
                return _read_gguf_context_length(path)
            main_file = _main_gguf_file(path)
            if main_file is not None:
                return _read_gguf_context_length(main_file)
            return None
        # Directory-based (MLX / Transformers / HF cache)
        search_dir = path if path.is_dir() else path.parent
        config_path = search_dir / "config.json"
        if config_path.exists():
            return _read_config_max_context(config_path)
        # HF cache layout: models--org--name/snapshots/<rev>/config.json
        snapshots = search_dir / "snapshots"
        if snapshots.is_dir():
            for snap in snapshots.iterdir():
                candidate = snap / "config.json"
                if candidate.exists():
                    result = _read_config_max_context(candidate)
                    if result is not None:
                        return result
    except Exception:
        return None
    return None


def _main_gguf_file(path: Path) -> Path | None:
    try:
        candidates = [
            candidate for candidate in path.rglob("*.gguf")
            if "mmproj" not in candidate.name.lower()
        ]
    except OSError:
        return None
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda candidate: candidate.stat().st_size)
    except OSError:
        return candidates[0]
