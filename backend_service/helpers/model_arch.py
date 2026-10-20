"""Real attention layout of a local model, for sizing its KV cache.

The chat memory estimate used to guess layers and heads from the parameter
count alone, which was 3-15x off for current models: grouped-query heads,
sliding-window layers, hybrid linear-attention layers and compressed (MLA)
caches all shrink the cache, and a mixture-of-experts model's total
parameter count says nothing about its attention. This module reads the
numbers that decide the cache from ``config.json`` (Hugging Face / MLX
folders) or the GGUF header (llama.cpp files).

Only what grows with context is counted: full-attention layers keep every
token, sliding-window layers keep at most the window, and recurrent /
linear-attention / Mamba layers keep a fixed-size state that is ignored.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KvArchitecture:
    num_layers: int
    full_layers: int
    sliding_layers: int
    sliding_window: int
    kv_heads: int
    key_dim: int
    value_dim: int
    # Compressed-attention (MLA) models cache one latent vector per token and
    # layer instead of per-head keys and values; this is its width.
    latent_dim: int = 0
    source: str = "config"

    def bytes_per_token_per_layer(self, bytes_per_element: int = 2) -> int:
        if self.latent_dim:
            return self.latent_dim * bytes_per_element
        return self.kv_heads * (self.key_dim + self.value_dim) * bytes_per_element

    def cache_bytes(self, context_tokens: int, bytes_per_element: int = 2) -> int:
        per_layer = self.bytes_per_token_per_layer(bytes_per_element)
        sliding_tokens = min(context_tokens, self.sliding_window) if self.sliding_window else context_tokens
        return per_layer * (self.full_layers * context_tokens + self.sliding_layers * sliding_tokens)


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


# ---------------------------------------------------------------------------
# config.json
# ---------------------------------------------------------------------------

_FULL_TYPES = {"full_attention", "attention", "global_attention", "full"}
_SLIDING_TYPES = {"sliding_attention", "local_attention", "sliding_window_attention", "chunked_attention"}


def _layer_counts_from_config(cfg: dict[str, Any], num_layers: int) -> tuple[int, int] | None:
    """(full, sliding) attention layer counts, or None if the config is silent."""
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and len(layer_types) == num_layers:
        full = sum(1 for t in layer_types if t in _FULL_TYPES)
        sliding = sum(1 for t in layer_types if t in _SLIDING_TYPES)
        return full, sliding
    pattern = cfg.get("hybrid_override_pattern")  # Nemotron-H: "*" marks attention
    if isinstance(pattern, str) and len(pattern) == num_layers:
        return pattern.count("*"), 0
    interval = _int(cfg.get("full_attention_interval"))  # Qwen3.5 / 3.6 / 3.8, Qwen3-Next
    if interval and interval > 0:
        return num_layers // interval, 0
    swa_pattern = _int(cfg.get("sliding_window_pattern"))  # Gemma 3: every Nth layer is global
    if swa_pattern and swa_pattern > 1 and _int(cfg.get("sliding_window")):
        full = num_layers // swa_pattern
        return full, num_layers - full
    return None


def architecture_from_config(config: dict[str, Any]) -> KvArchitecture | None:
    cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    model_type = str(cfg.get("model_type") or config.get("model_type") or "")
    num_layers = _int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers"))
    if not num_layers or num_layers <= 0:
        return None
    if model_type in {"mamba", "mamba2", "falcon_mamba", "rwkv"}:
        return KvArchitecture(num_layers, 0, 0, 0, 0, 0, 0, source="config")

    heads = _int(cfg.get("num_attention_heads") or cfg.get("n_head"))
    hidden = _int(cfg.get("hidden_size") or cfg.get("n_embd"))
    kv_heads = _int(cfg.get("num_key_value_heads")) or heads
    head_dim = _int(cfg.get("head_dim")) or (hidden // heads if hidden and heads else None)
    kv_lora_rank = _int(cfg.get("kv_lora_rank"))
    rope_dim = _int(cfg.get("qk_rope_head_dim"))

    counts = _layer_counts_from_config(cfg, num_layers)
    full, sliding = counts if counts else (num_layers, 0)
    window = (_int(cfg.get("sliding_window")) or 0) if sliding else 0
    if sliding and not window:
        full, sliding = full + sliding, 0

    if kv_lora_rank:
        return KvArchitecture(
            num_layers, full, sliding, window, 1, 0, 0,
            latent_dim=kv_lora_rank + (rope_dim or 0), source="config",
        )
    if not kv_heads or not head_dim:
        return None
    value_dim = _int(cfg.get("v_head_dim")) or head_dim
    return KvArchitecture(num_layers, full, sliding, window, kv_heads, head_dim, value_dim, source="config")


# ---------------------------------------------------------------------------
# GGUF header
# ---------------------------------------------------------------------------

_GGUF_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_GGUF_STRING, _GGUF_ARRAY = 8, 9
# llama.cpp hard-codes the sliding-window layout of these architectures
# instead of writing it to the header: every Nth layer is global.
_GGUF_SWA_EVERY_NTH_GLOBAL = {"gemma2": 2, "gemma3": 6, "cohere2": 4}
_MAX_KV = 4096


def _read_gguf_metadata(path: Path) -> dict[str, Any] | None:
    """Every key/value pair of a GGUF header. String arrays are skipped (the
    tokenizer vocabulary), numeric arrays are kept (per-layer head counts)."""
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != b"GGUF":
                return None
            (version,) = struct.unpack("<I", handle.read(4))
            if version < 2:
                return None
            _tensors, kv_count = struct.unpack("<QQ", handle.read(16))

            def read_string() -> str:
                (length,) = struct.unpack("<Q", handle.read(8))
                return handle.read(length).decode("utf-8", "replace")

            def read_value(value_type: int) -> Any:
                if value_type in _GGUF_SCALARS:
                    fmt = _GGUF_SCALARS[value_type]
                    return struct.unpack(fmt, handle.read(struct.calcsize(fmt)))[0]
                if value_type == _GGUF_STRING:
                    return read_string()
                if value_type == _GGUF_ARRAY:
                    (item_type,) = struct.unpack("<I", handle.read(4))
                    (count,) = struct.unpack("<Q", handle.read(8))
                    if item_type == _GGUF_STRING:
                        for _ in range(count):
                            (length,) = struct.unpack("<Q", handle.read(8))
                            handle.seek(length, 1)
                        return None
                    if item_type in _GGUF_SCALARS:
                        fmt = _GGUF_SCALARS[item_type]
                        size = struct.calcsize(fmt)
                        if count > 65536:
                            handle.seek(size * count, 1)
                            return None
                        data = handle.read(size * count)
                        return [v[0] for v in struct.iter_unpack(fmt, data)]
                    raise ValueError(f"unsupported GGUF array type {item_type}")
                raise ValueError(f"unsupported GGUF value type {value_type}")

            metadata: dict[str, Any] = {}
            for _ in range(min(kv_count, _MAX_KV)):
                key = read_string()
                (value_type,) = struct.unpack("<I", handle.read(4))
                metadata[key] = read_value(value_type)
            return metadata
    except (OSError, struct.error, ValueError, UnicodeDecodeError):
        return None


def architecture_from_gguf_metadata(meta: dict[str, Any]) -> KvArchitecture | None:
    arch = meta.get("general.architecture")
    if not isinstance(arch, str):
        return None

    def key(name: str) -> Any:
        return meta.get(f"{arch}.{name}")

    num_layers = _int(key("block_count"))
    if not num_layers:
        return None
    heads_raw = key("attention.head_count")
    kv_raw = key("attention.head_count_kv")
    heads = max(heads_raw) if isinstance(heads_raw, list) and heads_raw else _int(heads_raw)
    hidden = _int(key("embedding_length"))

    # Per-layer KV head counts mark recurrent layers with 0 (hybrid models).
    if isinstance(kv_raw, list) and len(kv_raw) == num_layers:
        attention_layers = [n for n in kv_raw if n]
        kv_heads = max(attention_layers) if attention_layers else 0
        attn_count = len(attention_layers)
    else:
        kv_heads = _int(kv_raw) or heads or 0
        attn_count = num_layers
    interval = _int(key("full_attention_interval"))
    if interval and interval > 0 and attn_count == num_layers:
        attn_count = num_layers // interval
    if not kv_heads or not attn_count:
        return KvArchitecture(num_layers, 0, 0, 0, 0, 0, 0, source="gguf")

    key_dim = _int(key("attention.key_length")) or (hidden // heads if hidden and heads else 0)
    value_dim = _int(key("attention.value_length")) or key_dim

    window = _int(key("attention.sliding_window")) or 0
    swa_pattern = key("attention.sliding_window_pattern")
    full, sliding = attn_count, 0
    if window and isinstance(swa_pattern, list) and len(swa_pattern) == num_layers:
        # llama.cpp writes one flag per layer: true = sliding.
        sliding = sum(1 for flag in swa_pattern if flag)
        full = attn_count - sliding
    else:
        every = _int(swa_pattern) or _GGUF_SWA_EVERY_NTH_GLOBAL.get(arch)
        if window and every and every > 1:
            full = attn_count // every
            sliding = attn_count - full

    kv_lora_rank = _int(key("attention.kv_lora_rank"))
    if kv_lora_rank:
        rope_dim = _int(key("rope.dimension_count")) or 0
        return KvArchitecture(num_layers, full, sliding, window, 1, 0, 0, latent_dim=kv_lora_rank + rope_dim, source="gguf")
    if not key_dim:
        return None
    return KvArchitecture(num_layers, full, sliding, window, kv_heads, key_dim, value_dim, source="gguf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _first_gguf(directory: Path) -> Path | None:
    candidates = sorted(
        (p for p in directory.rglob("*.gguf") if "mmproj" not in p.name.lower()),
        key=lambda p: ("-00001-of-" not in p.name and "-of-" in p.name, p.name),
    )
    return candidates[0] if candidates else None


@lru_cache(maxsize=256)
def _read_cached(path_str: str, _mtime: float) -> KvArchitecture | None:
    path = Path(path_str)
    if path.is_file():
        if path.suffix.lower() != ".gguf":
            return None
        meta = _read_gguf_metadata(path)
        return architecture_from_gguf_metadata(meta) if meta else None
    config_path = path / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = None
        if isinstance(config, dict):
            found = architecture_from_config(config)
            if found is not None:
                return found
    gguf = _first_gguf(path)
    if gguf is not None:
        meta = _read_gguf_metadata(gguf)
        return architecture_from_gguf_metadata(meta) if meta else None
    return None


def read_kv_architecture(path: str | Path) -> KvArchitecture | None:
    """Attention layout of the model at ``path`` (a folder or a ``.gguf``)."""
    # Not resolved: Hugging Face snapshot files are symlinks to extension-less
    # blobs, and the ``.gguf`` suffix is how a GGUF file is recognised.
    target = Path(path).expanduser().absolute()
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return None
    return _read_cached(str(target), mtime)


# ---------------------------------------------------------------------------
# Hugging Face Hub (models that are not downloaded yet)
# ---------------------------------------------------------------------------

_HUB_CACHE: dict[str, tuple[float, KvArchitecture | None]] = {}
_HUB_TTL_SECONDS = 6 * 3600
_HUB_FAILURE_TTL_SECONDS = 15 * 60
_HUB_TIMEOUT_SECONDS = 5.0


def _fetch_hub_config(repo: str) -> dict[str, Any] | None:
    """``config.json`` of a Hub repo, read over HTTP.

    Deliberately not ``hf_hub_download``: that would create a cache entry
    holding only ``config.json``, which the library scan and the "installed"
    checks would then see as a partly downloaded model.
    """
    import os
    import urllib.error
    import urllib.parse
    import urllib.request

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{urllib.parse.quote(repo, safe='/')}/resolve/main/config.json"
    headers = {"User-Agent": "ChaosEngineAI"}
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:
        token = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        import ssl

        import certifi  # what huggingface_hub's own client trusts; python.org builds ship no CA bundle

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = None
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=_HUB_TIMEOUT_SECONDS, context=context) as resp:
            data = json.loads(resp.read(2_000_000).decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _hub_base_models(repo: str) -> list[str]:
    """``base_model`` from a repo's model card (GGUF repos ship no config.json)."""
    try:
        from huggingface_hub import HfApi

        card = HfApi().model_info(repo, timeout=_HUB_TIMEOUT_SECONDS).card_data
    except Exception:
        return []
    base = card.get("base_model") if card else None
    if isinstance(base, str):
        return [base]
    return [b for b in base if isinstance(b, str)] if isinstance(base, list) else []


def hub_kv_architecture(repo: str) -> KvArchitecture | None:
    """Attention layout of a Hub repo that is not on disk, from its config.

    A GGUF repo has no ``config.json``; its model card's ``base_model`` is
    tried instead. Results (including failures, for a shorter time) are kept
    in memory so the launch preview does not hit the network on every change.
    """
    import time

    now = time.time()
    cached = _HUB_CACHE.get(repo)
    if cached is not None:
        stored_at, value = cached
        if now - stored_at < (_HUB_TTL_SECONDS if value is not None else _HUB_FAILURE_TTL_SECONDS):
            return value

    found: KvArchitecture | None = None
    import itertools

    for candidate in itertools.chain([repo], _hub_base_models_if_needed(repo)):
        config = _fetch_hub_config(candidate)
        if config is not None:
            found = architecture_from_config(config)
            if found is not None:
                found = KvArchitecture(**{**found.__dict__, "source": "hub"})
                break
    _HUB_CACHE[repo] = (now, found)
    return found


def _hub_base_models_if_needed(repo: str):
    """Lazily yields base models, so the card is only fetched when the repo has no config."""
    yield from _hub_base_models(repo)
