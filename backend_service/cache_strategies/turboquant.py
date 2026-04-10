"""Optional adapter for TurboQuant (arozanov/turboquant-mlx).

TurboQuant provides PolarQuant KV cache compression with fused Metal
kernels for MLX on Apple Silicon, and cache-type flags for llama.cpp.

Install: ``pip install turboquant-mlx``
"""

from __future__ import annotations

from typing import Any

from backend_service.cache_strategies import CacheStrategy


_available = False
_make_adaptive_cache = None
_apply_patch = None
try:
    from turboquant_mlx import make_adaptive_cache as _make_adaptive_cache  # type: ignore[import-untyped]
    from turboquant_mlx import apply_patch as _apply_patch  # type: ignore[import-untyped]
    _available = True
except ImportError:
    pass


class TurboQuantStrategy(CacheStrategy):

    @property
    def strategy_id(self) -> str:
        return "turboquant"

    @property
    def name(self) -> str:
        return "TurboQuant"

    def is_available(self) -> bool:
        return _available

    def supported_bit_range(self) -> tuple[int, int] | None:
        return (1, 4)

    def default_bits(self) -> int | None:
        return 3

    def supports_fp16_layers(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Engine integration
    # ------------------------------------------------------------------

    def make_mlx_cache(self, num_layers, bits, fp16_layers, fused, model) -> Any | None:
        """Create adaptive TurboQuant cache for mlx-lm."""
        if _make_adaptive_cache is None or _apply_patch is None:
            raise NotImplementedError("turboquant-mlx is not installed.")
        _apply_patch()
        return _make_adaptive_cache(
            num_layers,
            bits=bits,
            fp16_layers=fp16_layers,
            fused=fused,
            model=model,
        )

    def llama_cpp_cache_flags(self, bits: int) -> list[str]:
        """Cache-type flags for the TurboQuant llama.cpp fork."""
        clamped = max(2, min(4, bits))
        return ["--cache-type-k", f"turbo{clamped}", "--cache-type-v", f"turbo{clamped}"]

    def estimate_cache_bytes(self, num_layers, num_heads, hidden_size, context_tokens, bits, fp16_layers):
        kv_elements = 2 * num_layers * num_heads * (hidden_size // max(num_heads, 1)) * context_tokens
        baseline = kv_elements * 2
        compressed_layers = max(0, num_layers - 2 * fp16_layers)
        fp16_layer_count = num_layers - compressed_layers
        elements_per_layer = kv_elements // max(num_layers, 1)
        optimised = (fp16_layer_count * elements_per_layer * 2) + (compressed_layers * elements_per_layer * bits / 8)
        return baseline, int(optimised)

    def label(self, bits: int, fp16_layers: int) -> str:
        return f"TurboQ {bits}-bit {fp16_layers}+{fp16_layers}"
