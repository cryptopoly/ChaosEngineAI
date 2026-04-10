"""Optional adapter for TurboQuant (arozanov/turboquant-mlx).

TurboQuant provides PolarQuant KV cache compression with fused Metal
kernels for MLX on Apple Silicon, and cache-type flags for llama.cpp.

Install: ``./.venv/bin/python3 -m pip install turboquant-mlx``
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

from backend_service.cache_strategies import CacheStrategy


_REQUIRED_HOOKS = ("make_adaptive_cache", "apply_patch")


def _turboquant_mlx_source_blobs() -> list[str]:
    spec = importlib.util.find_spec("turboquant_mlx")
    if spec is None or spec.origin is None:
        return []

    origin = Path(spec.origin)
    package_dir = origin.parent
    candidates: list[Path] = []
    if origin.exists():
        candidates.append(origin)
    if package_dir.exists():
        candidates.extend(sorted(path for path in package_dir.rglob("*.py") if path != origin))

    sources: list[str] = []
    for path in candidates:
        try:
            sources.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return sources


def _has_required_turboquant_mlx_hooks() -> bool:
    sources = _turboquant_mlx_source_blobs()
    if not sources:
        return False
    return all(any(hook in source for source in sources) for hook in _REQUIRED_HOOKS)


def _load_turboquant_mlx_hooks() -> tuple[Any | None, Any | None]:
    if not _has_required_turboquant_mlx_hooks():
        return None, None
    try:
        module = importlib.import_module("turboquant_mlx")
    except ImportError:
        return None, None
    return getattr(module, "make_adaptive_cache", None), getattr(module, "apply_patch", None)


class TurboQuantStrategy(CacheStrategy):

    @property
    def strategy_id(self) -> str:
        return "turboquant"

    @property
    def name(self) -> str:
        return "TurboQuant"

    def is_available(self) -> bool:
        # Keep availability probing side-effect free. Some MLX packages touch
        # Metal during import, so we only report ready when the expected hooks
        # are present in the installed source tree.
        return _has_required_turboquant_mlx_hooks()

    def availability_badge(self) -> str:
        return "Ready" if self.is_available() else "Experimental"

    def availability_tone(self) -> str:
        return "ready" if self.is_available() else "warning"

    def availability_reason(self) -> str | None:
        if self.is_available():
            return None
        return (
            "The current PyPI turboquant-mlx package does not expose the MLX adapter hooks "
            "ChaosEngineAI expects yet, so this option stays disabled in the current build."
        )

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
        make_adaptive_cache, apply_patch = _load_turboquant_mlx_hooks()
        if make_adaptive_cache is None or apply_patch is None:
            raise NotImplementedError(
                "turboquant-mlx is not installed, or the installed package does not "
                "expose ChaosEngineAI's required MLX adapter hooks yet."
            )
        apply_patch()
        return make_adaptive_cache(
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
