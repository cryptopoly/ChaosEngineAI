"""Optional adapter for TriAttention (WeianMao/triattention).

TriAttention ships two runtime backends:

* **vLLM** (Linux/CUDA) — KV compression via monkeypatches into vLLM.
* **MLX**  (Apple Silicon, experimental, 2026-04-09) — compressor wrapper
  applied to an ``mlx_lm`` model via ``apply_triattention_mlx``.

Either backend is enough for the strategy to report as available; on
macOS we prefer the MLX path.  Install from git (not yet published on
PyPI)::

    pip install "triattention @ git+https://github.com/WeianMao/triattention.git"

Then add either ``mlx_lm`` (macOS) or ``vllm`` (Linux/CUDA).
"""

from __future__ import annotations

import importlib.util
from typing import Any

from cache_compression import CacheStrategy


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _has_mlx_entrypoint() -> bool:
    # Keep availability checks side-effect free. Importing mlx_lm can touch
    # MLX/Metal at module load and can abort in headless or sandboxed contexts.
    return _module_available("triattention") and _module_available("mlx_lm")


class TriAttentionStrategy(CacheStrategy):

    @property
    def strategy_id(self) -> str:
        return "triattention"

    @property
    def name(self) -> str:
        return "TriAttention"

    def has_mlx_backend(self) -> bool:
        return _has_mlx_entrypoint()

    def has_vllm_backend(self) -> bool:
        return _module_available("triattention") and _module_available("vllm")

    def is_available(self) -> bool:
        return self.has_mlx_backend() or self.has_vllm_backend()

    def availability_badge(self) -> str:
        if self.has_mlx_backend():
            return "Experimental"
        if self.has_vllm_backend():
            return "Ready"
        return "Install"

    def availability_tone(self) -> str:
        if self.has_mlx_backend():
            return "warning"
        if self.has_vllm_backend():
            return "ready"
        return "install"

    def availability_reason(self) -> str | None:
        if self.has_mlx_backend():
            return (
                "MLX backend (experimental, since 2026-04-09). Requires a calibration "
                "stats file; see triattention/docs/mlx.md."
            )
        if self.has_vllm_backend():
            return None
        if self.on_macos():
            return (
                "Install TriAttention + mlx_lm on macOS: pip install "
                "'triattention @ git+https://github.com/WeianMao/triattention.git' "
                "mlx-lm — then restart the app."
            )
        return (
            "Install TriAttention + vLLM on Linux/CUDA: pip install "
            "'triattention @ git+https://github.com/WeianMao/triattention.git' vllm."
        )

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
        if not _module_available("triattention"):
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
    # MLX integration (experimental)
    # ------------------------------------------------------------------

    def apply_mlx_compressor(
        self,
        model: Any,
        *,
        kv_budget: int = 2048,
        stats_path: str | None = None,
    ) -> Any:
        """Wrap an ``mlx_lm`` model with TriAttention MLX compression.

        Returns the compressor object; the caller uses it for generation in
        place of the bare model.  Requires triattention + mlx_lm installed.
        """
        if not self.has_mlx_backend():
            raise NotImplementedError(
                "TriAttention MLX backend not available. Install triattention "
                "and mlx_lm first."
            )
        from triattention.mlx import apply_triattention_mlx  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {"kv_budget": kv_budget}
        if stats_path is not None:
            kwargs["stats_path"] = stats_path
        return apply_triattention_mlx(model, **kwargs)

    # ------------------------------------------------------------------
    # Engine integration
    # ------------------------------------------------------------------

    def make_mlx_cache(self, num_layers, bits, fp16_layers, fused, model) -> Any | None:
        raise NotImplementedError(
            "TriAttention does not provide standalone KV cache objects. "
            "Use the vLLM backend on Linux, or apply_mlx_compressor() on macOS."
        )

    def llama_cpp_cache_flags(self, bits: int) -> list[str]:
        raise NotImplementedError(
            "TriAttention does not support llama.cpp. "
            "Use the vLLM backend instead."
        )

    def estimate_cache_bytes(self, num_layers, num_heads, hidden_size, context_tokens, bits, fp16_layers, num_kv_heads=None):
        kv_heads = num_kv_heads if num_kv_heads and num_kv_heads > 0 else num_heads
        kv_elements = 2 * num_layers * kv_heads * (hidden_size // max(num_heads, 1)) * context_tokens
        baseline = kv_elements * 2
        compressed_layers = max(0, num_layers - 2 * fp16_layers)
        fp16_layer_count = num_layers - compressed_layers
        elements_per_layer = kv_elements // max(num_layers, 1)
        optimised = (fp16_layer_count * elements_per_layer * 2) + (compressed_layers * elements_per_layer * bits / 8)
        return baseline, int(optimised)

    def label(self, bits: int, fp16_layers: int) -> str:
        return f"TriAttn {bits}-bit {fp16_layers}+{fp16_layers}"
