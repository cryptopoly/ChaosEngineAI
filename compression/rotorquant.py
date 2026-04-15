"""Optional adapter for RotorQuant (scrya-com/rotorquant).

RotorQuant provides IsoQuant (4D quaternion rotation) and PlanarQuant
(2D Givens rotation) KV cache compression.  PyTorch/CUDA only — no MLX.

The llama.cpp integration uses cache-type flags like ``iso3`` / ``planar3``
via the RotorQuant llama.cpp fork.

Install: ``pip install chaosengine-ai[rotorquant]``
(installs the ``turboquant`` PyPI package)
"""

from __future__ import annotations

import importlib
from typing import Any

from compression import CacheStrategy


def _load_turboquant_module() -> Any | None:
    try:
        return importlib.import_module("turboquant")
    except ImportError:
        return None


def _has_rotorquant_marker(module: Any | None) -> bool:
    if module is None:
        return False
    return any(
        hasattr(module, name)
        for name in ("IsoQuantMSE", "PlanarQuantMSE", "TurboQuantMSE", "TurboQuantIP", "TurboQuantCache")
    )


class RotorQuantStrategy(CacheStrategy):

    @property
    def strategy_id(self) -> str:
        return "rotorquant"

    @property
    def name(self) -> str:
        return "RotorQuant"

    def is_available(self) -> bool:
        # The Python package is only used as an installation marker here.
        # Actual execution still routes through the RotorQuant llama.cpp fork.
        return _has_rotorquant_marker(_load_turboquant_module())

    def availability_badge(self) -> str:
        return "Ready" if self.is_available() else "Install"

    def availability_tone(self) -> str:
        return "ready" if self.is_available() else "install"

    def availability_reason(self) -> str | None:
        if self.is_available():
            return None
        return "Install turboquant into ChaosEngineAI's backend runtime, then restart the app."

    def supported_bit_range(self) -> tuple[int, int] | None:
        return (3, 4)

    def default_bits(self) -> int | None:
        return 3

    def supports_fp16_layers(self) -> bool:
        return True

    def required_llama_binary(self) -> str:
        return "turbo"

    # ------------------------------------------------------------------
    # Engine integration
    # ------------------------------------------------------------------

    def make_mlx_cache(self, num_layers, bits, fp16_layers, fused, model) -> Any | None:
        """RotorQuant is PyTorch/CUDA only — no MLX support."""
        raise NotImplementedError(
            "RotorQuant requires PyTorch/CUDA and does not support MLX. "
            "Use the llama.cpp (GGUF) backend with RotorQuant cache types, "
            "or the vLLM backend."
        )

    def llama_cpp_cache_flags(self, bits: int) -> list[str]:
        """Return cache-type flags for the TurboQuant llama.cpp fork.

        The fork (github.com/TheTom/llama-cpp-turboquant, branch
        ``feature/turboquant-kv-cache``) supports ``turbo2``, ``turbo3``,
        ``turbo4`` as cache-type values.

        RotorQuant maps to the same turbo cache types — both are
        rotation-based KV cache quantization.
        """
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
        return f"Rotor {bits}-bit {fp16_layers}+{fp16_layers}"
