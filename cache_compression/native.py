"""Built-in native cache strategy: always available, f16 by default.

On llama.cpp the native strategy can also store the KV cache in the
standard quantized types every llama-server build has (FU-178):
``cacheBits`` 8 runs ``q8_0`` and 4 runs ``q4_0``, about half and a
quarter of the f16 cache. Any other value, and every other engine, keeps
f16. FP8 (``f8_e4m3``) is not offered: llama-server aborts on Metal when
it allocates that cache (see ``llama_cpp_engine._refused_cache_type``).
"""

from __future__ import annotations

from typing import Any

from cache_compression import CacheStrategy

# cacheBits -> llama.cpp cache type of the native strategy.
NATIVE_LLAMA_CACHE_TYPES: dict[int, str] = {8: "q8_0", 4: "q4_0"}
# Bits per cached element, read off the KV buffer sizes llama.cpp prints
# (a 32-element block carries one f16 scale: q8_0 is 8.5, q4_0 is 4.5).
_LLAMA_CACHE_BITS: dict[str, float] = {"q8_0": 8.5, "q4_0": 4.5}


def native_llama_cache_type(bits: int) -> str:
    """The llama.cpp cache type native runs for ``bits`` (f16 unless 8 or 4)."""
    return NATIVE_LLAMA_CACHE_TYPES.get(int(bits or 0), "f16")


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
        cache_type = native_llama_cache_type(bits)
        return ["--cache-type-k", cache_type, "--cache-type-v", cache_type]

    def estimate_cache_bytes(self, num_layers, num_heads, hidden_size, context_tokens, bits, fp16_layers, num_kv_heads=None):
        kv_heads = num_kv_heads if num_kv_heads and num_kv_heads > 0 else num_heads
        kv_elements = 2 * num_layers * kv_heads * (hidden_size // max(num_heads, 1)) * context_tokens
        baseline = kv_elements * 2
        cache_type = native_llama_cache_type(bits)
        if cache_type == "f16":
            return baseline, baseline
        return baseline, int(baseline * _LLAMA_CACHE_BITS[cache_type] / 16)

    def label(self, bits: int, fp16_layers: int) -> str:
        cache_type = native_llama_cache_type(bits)
        return "Native f16" if cache_type == "f16" else f"Native {cache_type}"
