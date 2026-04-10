"""Optional adapter for TriAttention (WeianMao/triattention).

Install: ``pip install triattention``
"""

from __future__ import annotations

from typing import Any

from backend_service.cache_strategies import CacheStrategy


_triattention = None
try:
    import triattention as _triattention  # type: ignore[import-untyped]
except ImportError:
    pass


class TriAttentionStrategy(CacheStrategy):

    @property
    def strategy_id(self) -> str:
        return "triattention"

    @property
    def name(self) -> str:
        return "TriAttention"

    def is_available(self) -> bool:
        return _triattention is not None

    def supported_bit_range(self) -> tuple[int, int] | None:
        return (1, 4)

    def default_bits(self) -> int | None:
        return 3

    def supports_fp16_layers(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Engine integration — fill in once the triattention API is stable
    # ------------------------------------------------------------------

    def make_mlx_cache(self, num_layers, bits, fp16_layers, fused, model) -> Any | None:
        """Create a TriAttention cache list for mlx-lm.

        Expected to return a ``list[KVCache]`` of length *num_layers* where
        the first/last *fp16_layers* use standard FP16 caches and the middle
        layers use TriAttention compressed caches.
        """
        raise NotImplementedError(
            "TriAttention MLX cache adapter not yet implemented. "
            "See https://github.com/WeianMao/triattention for the upstream API."
        )

    def llama_cpp_cache_flags(self, bits: int) -> list[str]:
        """Return llama-server cache-type flags for TriAttention.

        Fill in once llama.cpp ships TriAttention cache types.
        """
        raise NotImplementedError(
            "TriAttention llama.cpp flags not yet implemented."
        )

    def estimate_cache_bytes(self, num_layers, num_heads, hidden_size, context_tokens, bits, fp16_layers):
        kv_elements = 2 * num_layers * num_heads * (hidden_size // max(num_heads, 1)) * context_tokens
        baseline = kv_elements * 2
        compressed_layers = max(0, num_layers - 2 * fp16_layers)
        fp16_layer_count = num_layers - compressed_layers
        elements_per_layer = kv_elements // max(num_layers, 1)
        optimised = (fp16_layer_count * elements_per_layer * 2) + (compressed_layers * elements_per_layer * bits / 8)
        return baseline, int(optimised)

    def label(self, bits: int, fp16_layers: int) -> str:
        return f"TriAttn {bits}-bit {fp16_layers}+{fp16_layers}"
