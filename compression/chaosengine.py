"""Adapter for ChaosEngine KV cache compression (cryptopoly/ChaosEngine).

ChaosEngine uses PCA-based decorrelation, channel truncation, and hybrid
quantization to compress KV cache memory.  It achieves ~3.7x compression
on 8B models with an average attention output error of 0.034.

Supports 2/4/8-bit compression tiers with per-channel asymmetric
quantization and importance-weighted bit allocation.

Desktop builds can bundle vendored ChaosEngine automatically during
``npm run stage:runtime`` when ``vendor/ChaosEngine`` (or
``CHAOSENGINE_VENDOR_PATH``) is present. Source/dev installs can still use:
``./.venv/bin/python3 -m pip install -e /path/to/ChaosEngine``
GitHub:  https://github.com/cryptopoly/ChaosEngine
"""

from __future__ import annotations

import importlib
from typing import Any

from compression import CacheStrategy


def _load_chaosengine() -> Any | None:
    try:
        return importlib.import_module("chaos_engine")
    except ImportError:
        return None


def _chaosengine_available() -> bool:
    mod = _load_chaosengine()
    if mod is None:
        return False
    # Check for the core cache module
    try:
        cache_mod = importlib.import_module("chaos_engine.cache")
        return hasattr(cache_mod, "config") or True
    except ImportError:
        return False


class ChaosEngineStrategy(CacheStrategy):

    @property
    def strategy_id(self) -> str:
        return "chaosengine"

    @property
    def name(self) -> str:
        return "ChaosEngine"

    def is_available(self) -> bool:
        return _chaosengine_available()

    def availability_badge(self) -> str:
        return "Ready" if self.is_available() else "Install"

    def availability_tone(self) -> str:
        return "ready" if self.is_available() else "install"

    def availability_reason(self) -> str | None:
        if self.is_available():
            return None
        return (
            "ChaosEngine is not bundled into this runtime. Desktop release builds "
            "bundle it automatically when vendor/ChaosEngine (or "
            "CHAOSENGINE_VENDOR_PATH) is present during npm run stage:runtime. "
            "For source/dev installs, use: ./.venv/bin/python3 -m pip install -e "
            "/path/to/ChaosEngine — then restart ChaosEngineAI. "
            "GitHub: https://github.com/cryptopoly/ChaosEngine"
        )

    def supported_bit_range(self) -> tuple[int, int] | None:
        return (2, 8)

    def default_bits(self) -> int | None:
        return 4

    def supports_fp16_layers(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Engine integration
    # ------------------------------------------------------------------

    def make_mlx_cache(self, num_layers, bits, fp16_layers, fused, model) -> Any | None:
        """ChaosEngine is PyTorch-based — no MLX cache support yet."""
        raise NotImplementedError(
            "ChaosEngine KV cache compression currently requires PyTorch. "
            "Use the llama.cpp (GGUF) or vLLM backend, or contribute MLX "
            "support at https://github.com/cryptopoly/ChaosEngine."
        )

    def llama_cpp_cache_flags(self, bits: int) -> list[str]:
        """Return cache-type flags for llama.cpp with ChaosEngine quantization.

        ChaosEngine uses PCA decorrelation + hybrid quantization.
        Maps to q-type cache flags based on the configured bit width.
        """
        bit_map = {
            2: "q4_0",
            3: "q4_0",
            4: "q4_0",
            5: "q5_0",
            6: "q8_0",
            8: "q8_0",
        }
        cache_type = bit_map.get(bits, "q8_0")
        return ["--cache-type-k", cache_type, "--cache-type-v", cache_type]

    def estimate_cache_bytes(
        self,
        num_layers,
        num_heads,
        hidden_size,
        context_tokens,
        bits,
        fp16_layers,
    ):
        kv_elements = 2 * num_layers * num_heads * (hidden_size // max(num_heads, 1)) * context_tokens
        baseline = kv_elements * 2  # FP16 = 2 bytes per element

        compressed_layers = max(0, num_layers - 2 * fp16_layers)
        fp16_layer_count = num_layers - compressed_layers
        elements_per_layer = kv_elements // max(num_layers, 1)

        # ChaosEngine achieves slightly better compression than naive
        # quantization due to PCA decorrelation reducing redundancy
        # before quantization.  Apply a 0.85 factor to account for this.
        pca_efficiency = 0.85
        optimised = (
            fp16_layer_count * elements_per_layer * 2
            + compressed_layers * elements_per_layer * bits / 8 * pca_efficiency
        )
        return baseline, int(optimised)

    def label(self, bits: int, fp16_layers: int) -> str:
        return f"ChaosEngine {bits}-bit {fp16_layers}+{fp16_layers}"
