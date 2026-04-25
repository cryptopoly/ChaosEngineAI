"""Built-in native cache strategy — always available, no compression."""

from __future__ import annotations

from typing import Any

from cache_compression import CacheStrategy


class NativeStrategy(CacheStrategy):

    @property
    def strategy_id(self) -> str:
        return "native"

    @property
    def name(self) -> str:
        return "Native f16"

    def is_available(self) -> bool:
        return True

    def availability_badge(self) -> str:
        return "Ready"

    def availability_tone(self) -> str:
        return "ready"

    def make_mlx_cache(self, num_layers, bits, fp16_layers, fused, model) -> Any | None:
        return None  # mlx-lm uses its own default KVCache

    def llama_cpp_cache_flags(self, bits: int) -> list[str]:
        return ["--cache-type-k", "f16", "--cache-type-v", "f16"]

    def estimate_cache_bytes(self, num_layers, num_heads, hidden_size, context_tokens, bits, fp16_layers, num_kv_heads=None):
        kv_heads = num_kv_heads if num_kv_heads and num_kv_heads > 0 else num_heads
        kv_elements = 2 * num_layers * kv_heads * (hidden_size // max(num_heads, 1)) * context_tokens
        baseline = kv_elements * 2
        return baseline, baseline

    def label(self, bits: int, fp16_layers: int) -> str:
        return "Native f16"
