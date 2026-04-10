"""Optional adapter for TriAttention (WeianMao/triattention).

TriAttention integrates via vLLM monkeypatching — it does NOT provide
standalone KV cache objects.  Both ``triattention`` and ``vllm`` must be
installed for this strategy to report as available.

Install: ``pip install chaosengine-ai[triattention]``
"""

from __future__ import annotations

from typing import Any

from backend_service.cache_strategies import CacheStrategy


_triattention = None
_vllm = None
try:
    import triattention as _triattention  # type: ignore[import-untyped]
except ImportError:
    pass
try:
    import vllm as _vllm  # type: ignore[import-untyped]
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
        return _triattention is not None and _vllm is not None

    def supported_bit_range(self) -> tuple[int, int] | None:
        return (1, 4)

    def default_bits(self) -> int | None:
        return 3

    def supports_fp16_layers(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # vLLM integration
    # ------------------------------------------------------------------

    def apply_vllm_patches(self) -> None:
        """Install TriAttention monkeypatches into vLLM.

        Must be called BEFORE creating a ``vllm.LLM`` instance.
        """
        if _triattention is None:
            raise RuntimeError("triattention is not installed.")
        try:
            from triattention.vllm.runtime.integration_monkeypatch import (
                install_vllm_integration_monkeypatches,
            )
            install_vllm_integration_monkeypatches(
                patch_scheduler=True, patch_worker=True,
            )
        except ImportError as exc:
            raise NotImplementedError(
                "TriAttention vLLM integration module not found. "
                "Ensure triattention is installed with vLLM support."
            ) from exc

    # ------------------------------------------------------------------
    # Engine integration
    # ------------------------------------------------------------------

    def make_mlx_cache(self, num_layers, bits, fp16_layers, fused, model) -> Any | None:
        raise NotImplementedError(
            "TriAttention does not provide standalone KV cache objects. "
            "Use the vLLM backend with TriAttention enabled instead."
        )

    def llama_cpp_cache_flags(self, bits: int) -> list[str]:
        raise NotImplementedError(
            "TriAttention does not support llama.cpp. "
            "Use the vLLM backend instead."
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
