"""Pluggable cache/compression strategy system for ChaosEngineAI.

Each strategy adapts a third-party KV-cache compression library (or the
built-in native path) behind a common interface so that both the MLX and
llama.cpp engines can use it without knowing the details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
import platform
from threading import RLock
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
        num_kv_heads: int | None = None,
    ) -> tuple[int, int]:
        """Return ``(baseline_bytes, optimised_bytes)``.

        *baseline_bytes* is the uncompressed FP16 cache size.
        *optimised_bytes* is the estimated size under this strategy.
        The default implementation returns equal values (no compression).

        ``num_kv_heads`` distinguishes Grouped Query Attention models
        where the KV cache uses fewer heads than Q projection. Defaults
        to ``num_heads`` (multi-head attention) for backward compat.
        """
        kv_heads = num_kv_heads if num_kv_heads and num_kv_heads > 0 else num_heads
        kv_elements = 2 * num_layers * kv_heads * (hidden_size // max(num_heads, 1)) * context_tokens
        baseline = kv_elements * 2  # FP16 = 2 bytes
        return baseline, baseline

    def required_llama_binary(self) -> str:
        """Return which llama-server binary variant this strategy needs.

        ``"standard"`` — upstream / Homebrew llama-server.
        ``"turbo"``    — johndpope/llama-cpp-turboquant fork (supports
                         iso/planar/turbo cache types in addition to all
                         standard types).

        The runtime uses this to pick the correct binary path.
        """
        return "standard"

    def applies_to(self) -> frozenset[str]:
        """Inference domains this strategy applies to.

        ``"text"``  — text LLM inference (MLX, llama.cpp, vLLM).
        ``"image"`` — diffusion image pipelines.
        ``"video"`` — diffusion video pipelines.

        Default is text, matching every existing KV-cache strategy. Diffusion
        strategies (e.g. TeaCache) override this so the UI surfaces them in
        the correct Studio.
        """
        return frozenset({"text"})

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
        self._discovered = False
        self._lock = RLock()

    def register(self, strategy: CacheStrategy) -> None:
        self._strategies[strategy.strategy_id] = strategy

    def get(self, strategy_id: str) -> CacheStrategy | None:
        self._ensure_discovered()
        return self._strategies.get(strategy_id)

    def default(self) -> CacheStrategy:
        self._ensure_discovered()
        return self._strategies["native"]

    def strategies(self) -> list[CacheStrategy]:
        self._ensure_discovered()
        return list(self._strategies.values())

    def available(self) -> list[dict[str, Any]]:
        """Return a JSON-friendly list for the frontend."""
        self._ensure_discovered()
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
                "requiredLlamaBinary": s.required_llama_binary(),
                "appliesTo": sorted(s.applies_to()),
            })
        return out

    def _ensure_discovered(self) -> None:
        if self._discovered:
            return
        with self._lock:
            if not self._discovered:
                self.discover()

    def discover(self) -> list[CacheStrategy]:
        """Import all known adapter modules and return available strategies."""
        with self._lock:
            self._strategies = {}

            strategy_specs = [
            {
                "id": "native",
                "name": "Native f16",
                "module": "cache_compression.native",
                "class_name": "NativeStrategy",
                "bit_range": None,
                "default_bits": None,
                "supports_fp16_layers": False,
                "required_llama_binary": "standard",
            },
            {
                "id": "rotorquant",
                "name": "RotorQuant",
                "module": "cache_compression.rotorquant",
                "class_name": "RotorQuantStrategy",
                "bit_range": (3, 4),
                "default_bits": 3,
                "supports_fp16_layers": True,
                "required_llama_binary": "turbo",
            },
            {
                "id": "triattention",
                "name": "TriAttention",
                "module": "cache_compression.triattention",
                "class_name": "TriAttentionStrategy",
                "bit_range": (1, 4),
                "default_bits": 3,
                "supports_fp16_layers": True,
                "required_llama_binary": "standard",
            },
            {
                "id": "turboquant",
                "name": "TurboQuant",
                "module": "cache_compression.turboquant",
                "class_name": "TurboQuantStrategy",
                "bit_range": (1, 4),
                "default_bits": 3,
                "supports_fp16_layers": True,
                "required_llama_binary": "turbo",
            },
            {
                "id": "chaosengine",
                "name": "ChaosEngine",
                "module": "cache_compression.chaosengine",
                "class_name": "ChaosEngineStrategy",
                "bit_range": (2, 8),
                "default_bits": 4,
                "supports_fp16_layers": True,
                "required_llama_binary": "standard",
            },
            {
                # Diffusion-pipeline cache — applies to image/video DiTs,
                # not text LLMs. The `bit_range`/`default_bits` fields are
                # N/A (TeaCache's knob is a rel-L1 threshold, not bits).
                # `required_llama_binary="standard"` is the neutral fallback;
                # TeaCache's llama.cpp hook raises NotImplementedError so
                # this metadata is only shape-filling for the _BrokenStrategy
                # path.
                "id": "teacache",
                "name": "TeaCache",
                "module": "cache_compression.teacache",
                "class_name": "TeaCacheStrategy",
                "bit_range": None,
                "default_bits": None,
                "supports_fp16_layers": False,
                "required_llama_binary": "standard",
            },
            ]

            for spec in strategy_specs:
                try:
                    module = importlib.import_module(spec["module"])
                    cls = getattr(module, spec["class_name"])
                    instance = cls()
                except Exception as exc:
                    if spec["id"] == "native":
                        raise
                    instance = _BrokenStrategy(
                        strategy_id=str(spec["id"]),
                        name=str(spec["name"]),
                        bit_range=spec["bit_range"],
                        default_bits=spec["default_bits"],
                        supports_fp16_layers=bool(spec["supports_fp16_layers"]),
                        required_llama_binary=str(spec.get("required_llama_binary", "standard")),
                        reason=(
                            f"{spec['name']} could not be loaded in this runtime. "
                            f"ChaosEngineAI kept the card visible so the UI does not silently collapse to Native f16 only. "
                            f"Import error: {exc}"
                        ),
                    )
                self.register(instance)
            self._discovered = True
            return list(self._strategies.values())


class _BrokenStrategy(CacheStrategy):
    def __init__(
        self,
        *,
        strategy_id: str,
        name: str,
        bit_range: tuple[int, int] | None,
        default_bits: int | None,
        supports_fp16_layers: bool,
        reason: str,
        required_llama_binary: str = "standard",
    ) -> None:
        self._strategy_id = strategy_id
        self._name = name
        self._bit_range = bit_range
        self._default_bits = default_bits
        self._supports_fp16_layers = supports_fp16_layers
        self._reason = reason
        self._required_llama_binary = required_llama_binary

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return False

    def availability_badge(self) -> str:
        return "Unavailable"

    def availability_tone(self) -> str:
        return "warning"

    def availability_reason(self) -> str | None:
        return self._reason

    def supported_bit_range(self) -> tuple[int, int] | None:
        return self._bit_range

    def default_bits(self) -> int | None:
        return self._default_bits

    def supports_fp16_layers(self) -> bool:
        return self._supports_fp16_layers

    def required_llama_binary(self) -> str:
        return self._required_llama_binary


# Module-level singleton — import and use ``registry`` directly.
registry = CacheStrategyRegistry()


def apply_diffusion_cache_strategy(
    pipeline: Any,
    *,
    strategy_id: str | None,
    num_inference_steps: int,
    rel_l1_thresh: float | None,
    domain: str,
) -> str | None:
    """Apply a diffusion cache strategy (e.g. TeaCache) to a diffusers pipeline.

    Called by image_runtime / video_runtime after pipeline load, before the
    first denoise. Shared so both entry points agree on error handling.

    ``strategy_id`` — registry id (e.g. ``"teacache"``). ``None`` or empty
        skips silently.
    ``num_inference_steps`` — final step count (after any FLUX / Turbo clamp).
    ``rel_l1_thresh`` — optional threshold override; ``None`` uses strategy
        default.
    ``domain`` — ``"image"`` or ``"video"``; the strategy's ``applies_to()``
        must include this.

    Returns a short note describing what happened (for the runtime note on
    generated assets), or ``None`` when no strategy was applied. Never raises
    for expected failure modes (unsupported pipeline, strategy not available)
    — those produce a note and the caller falls back to the stock pipeline.
    Programming errors (wrong types) still surface.
    """
    if not strategy_id:
        return None
    strategy = registry.get(strategy_id)
    if strategy is None:
        return f"Cache strategy '{strategy_id}' not found; using stock pipeline."
    if domain not in strategy.applies_to():
        return (
            f"{strategy.name} does not apply to {domain} pipelines "
            f"(domains: {sorted(strategy.applies_to())}); using stock pipeline."
        )
    hook = getattr(strategy, "apply_diffusers_hook", None)
    if hook is None:
        return (
            f"{strategy.name} has no diffusers hook; using stock pipeline."
        )
    try:
        hook(
            pipeline,
            num_inference_steps=num_inference_steps,
            rel_l1_thresh=rel_l1_thresh,
        )
    except NotImplementedError as exc:
        # Expected path when a pipeline's patch hasn't landed yet. Return
        # the message so the UI can show "TeaCache not applied: <reason>".
        return f"{strategy.name} not applied: {exc}"
    return f"{strategy.name} applied (thresh={rel_l1_thresh or 'default'})."
