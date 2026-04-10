"""Pluggable cache/compression strategy system for ChaosEngineAI.

Each strategy adapts a third-party KV-cache compression library (or the
built-in native path) behind a common interface so that both the MLX and
llama.cpp engines can use it without knowing the details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import platform
from typing import Any


class CacheStrategy(ABC):
    """Base class every cache/compression backend must implement."""

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Machine key, e.g. ``"native"``, ``"triattention"``."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` when the backing library is importable."""

    def availability_badge(self) -> str:
        return "Ready" if self.is_available() else "Install"

    def availability_tone(self) -> str:
        return "ready" if self.is_available() else "install"

    def availability_reason(self) -> str | None:
        return None

    def on_macos(self) -> bool:
        return platform.system() == "Darwin"

    def supported_bit_range(self) -> tuple[int, int] | None:
        """Min/max quantisation bits, or ``None`` if the concept does not apply."""
        return None

    def default_bits(self) -> int | None:
        return None

    def supports_fp16_layers(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Engine integration
    # ------------------------------------------------------------------

    def make_mlx_cache(
        self,
        num_layers: int,
        bits: int,
        fp16_layers: int,
        fused: bool,
        model: Any,
    ) -> Any | None:
        """Return an mlx-lm compatible *prompt_cache* list, or ``None`` to use
        the model's default cache."""
        return None

    def llama_cpp_cache_flags(self, bits: int) -> list[str]:
        """Return CLI flags for llama-server's ``--cache-type-k``/``-v``."""
        return ["--cache-type-k", "f16", "--cache-type-v", "f16"]

    # ------------------------------------------------------------------
    # Preview / estimation helpers
    # ------------------------------------------------------------------

    def estimate_cache_bytes(
        self,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        context_tokens: int,
        bits: int,
        fp16_layers: int,
    ) -> tuple[int, int]:
        """Return ``(baseline_bytes, optimised_bytes)``.

        *baseline_bytes* is the uncompressed FP16 cache size.
        *optimised_bytes* is the estimated size under this strategy.
        The default implementation returns equal values (no compression).
        """
        kv_elements = 2 * num_layers * num_heads * (hidden_size // max(num_heads, 1)) * context_tokens
        baseline = kv_elements * 2  # FP16 = 2 bytes
        return baseline, baseline

    def apply_vllm_patches(self) -> None:
        """Hook for strategies that monkeypatch vLLM (e.g. TriAttention).

        Called by ``VLLMEngine.load_model()`` before creating the LLM instance.
        Default is a no-op.
        """

    def label(self, bits: int, fp16_layers: int) -> str:
        """Short UI label, e.g. ``"Native f16"`` or ``"TriAttn 3-bit 4+4"``."""
        return self.name


# ======================================================================
# Registry
# ======================================================================

class CacheStrategyRegistry:
    """Discovers and exposes all known cache strategies."""

    def __init__(self) -> None:
        self._strategies: dict[str, CacheStrategy] = {}

    def register(self, strategy: CacheStrategy) -> None:
        self._strategies[strategy.strategy_id] = strategy

    def get(self, strategy_id: str) -> CacheStrategy | None:
        return self._strategies.get(strategy_id)

    def default(self) -> CacheStrategy:
        return self._strategies["native"]

    def available(self) -> list[dict[str, Any]]:
        """Return a JSON-friendly list for the frontend."""
        out: list[dict[str, Any]] = []
        for s in self._strategies.values():
            out.append({
                "id": s.strategy_id,
                "name": s.name,
                "available": s.is_available(),
                "bitRange": list(s.supported_bit_range()) if s.supported_bit_range() else None,
                "defaultBits": s.default_bits(),
                "supportsFp16Layers": s.supports_fp16_layers(),
                "availabilityBadge": s.availability_badge(),
                "availabilityTone": s.availability_tone(),
                "availabilityReason": s.availability_reason(),
            })
        return out

    def discover(self) -> list[CacheStrategy]:
        """Import all known adapter modules and return available strategies."""
        from backend_service.cache_strategies.native import NativeStrategy
        from backend_service.cache_strategies.rotorquant import RotorQuantStrategy
        from backend_service.cache_strategies.triattention import TriAttentionStrategy
        from backend_service.cache_strategies.turboquant import TurboQuantStrategy

        for cls in (NativeStrategy, RotorQuantStrategy, TriAttentionStrategy, TurboQuantStrategy):
            instance = cls()
            self.register(instance)
        return list(self._strategies.values())


# Module-level singleton — import and use ``registry`` directly.
registry = CacheStrategyRegistry()
registry.discover()
