"""Optional adapter for TriAttention (WeianMao/triattention).

TriAttention ships two runtime backends:

* **vLLM** (Linux/CUDA) — KV compression via monkeypatches into vLLM.
* **MLX**  (Apple Silicon, experimental, 2026-04-09) — compressor wrapper
  applied to an ``mlx_lm`` model via ``apply_triattention_mlx``.

Either backend is enough for the strategy to report as available; on
macOS we prefer the MLX path.  Install from git (not yet published on
PyPI)::

    pip install "triattention @ git+https://github.com/WeianMao/triattention.git"

Then add either ``mlx_lm`` (macOS) or ``vllm`` (Linux/CUDA). The vLLM
patches only work on vLLM 0.18 and older (see ``vllm_unsupported_reason``),
and they need a calibration stats file per model (``find_stats_file``).
"""

from __future__ import annotations

import importlib.util
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cache_compression import CacheStrategy

# vLLM 0.19 changed two model-runner internals the TriAttention plugin
# (commit a4bc3c8) depends on: each attention layer's ``kv_cache`` became a
# bare tensor (was a one-element list), so its KV group resolver found no
# tensors and every compaction ended ``no_compactable_groups``; and
# ``GPUModelRunner.positions`` became a GPU-only tensor, which its copied
# ``_prepare_inputs`` reads as a CPU/GPU buffer (``.positions.np``), so the
# engine crashes on the first step after a compaction. Measured on the RTX
# 4090 (FU-171): 0.19.1 compacts nothing (and crashes once the first is
# forced), 0.33.0 dies on the first request, 0.18.1 compacts.
FIRST_UNSUPPORTED_VLLM = (0, 19)

# Prefill chunk size the compaction was measured with. The plugin compacts
# before the step's forward pass, so a step that prefills the whole prompt
# compacts KV that has not been written yet (the 7.8K-token needle prompt in
# one chunk produced garbage from the first token; 2048-token chunks did not).
TRIATTENTION_MAX_BATCHED_TOKENS = 2048
DEFAULT_KV_BUDGET = 2048


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _installed_vllm_version() -> str | None:
    """The installed vLLM version from package metadata, without importing it."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - stdlib since 3.8
        return None
    try:
        return version("vllm")
    except PackageNotFoundError:
        return None


def vllm_unsupported_reason(vllm_version: str | None) -> str | None:
    """Why TriAttention cannot run on this vLLM, or None when it can (or is unknown)."""
    if not vllm_version:
        return None
    match = re.match(r"(\d+)\.(\d+)", vllm_version)
    if match is None:
        return None
    if (int(match.group(1)), int(match.group(2))) < FIRST_UNSUPPORTED_VLLM:
        return None
    return (
        f"TriAttention's vLLM patches work on vLLM 0.18 and older; vLLM "
        f"{vllm_version} is installed. From 0.19 vLLM changed the KV cache and "
        "input buffers the patches rewrite, so nothing is compressed and the "
        "engine then crashes. Use TriAttention from a separate vLLM 0.18 "
        "environment."
    )


def stats_dir() -> Path:
    """Where per-model TriAttention calibration stats files are looked up."""
    override = os.environ.get("CHAOSENGINE_TRIATTENTION_STATS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".chaosengine" / "triattention" / "stats"


def _stats_names(model_id: str) -> list[str]:
    base = Path(model_id.replace("\\", "/").rstrip("/")).name
    if not base or base in {".", ".."}:
        return []
    names = [f"{base.lower()}.pt"]
    if base != base.lower():
        names.append(f"{base}.pt")
    return names


def find_stats_file(model_ids: Iterable[str | None]) -> Path | None:
    """The stats file for the first model id that has one: ``<stats_dir>/<repo name>.pt``.

    TriAttention scores keys with per-head frequency statistics made by its
    ``scripts/calibrate.py``; without them the plugin stops the engine at the
    first compression. ``Qwen/Qwen3-0.6B`` and a local ``.../Qwen3-0.6B``
    folder both look for ``qwen3-0.6b.pt``.
    """
    root = stats_dir()
    try:
        root_resolved = root.resolve()
    except OSError:
        return None
    for model_id in model_ids:
        if not model_id:
            continue
        for name in _stats_names(model_id):
            candidate = (root / name).resolve()
            if candidate.parent != root_resolved:
                continue
            if candidate.is_file():
                return candidate
    return None


def _kv_budget() -> int:
    raw = os.environ.get("CHAOSENGINE_TRIATTENTION_KV_BUDGET", "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_KV_BUDGET
    return value if value > 0 else DEFAULT_KV_BUDGET


def _install_skip_stamp_fix() -> bool:
    """Keep a skipped compression from re-arming TriAttention's dedup guard.

    The worker skips a compression signal that arrives one step after the
    last one (``batch_queue_dedup``). ``mark_compression_skipped`` also stamps
    the skip's step as ``last_compression_step``, so once the scheduler
    signals every step (its length estimate stays above the threshold after a
    worker-side ``under_budget`` skip) each signal is a step after the
    previous skip and the request is never compacted again. Measured on vLLM
    0.18.1: 1 compaction, then 2304 consecutive dedup skips over 3000 tokens;
    with this fix, 20 compactions. The guard is meant to follow applied
    compressions only, so a skip keeps the step of the last applied one.
    Idempotent; True when installed.
    """
    try:
        from triattention.vllm.runtime.state import RequestStateStore  # type: ignore[import-untyped]
    except ImportError:
        return False
    current = getattr(RequestStateStore, "mark_compression_skipped", None)
    if current is None:
        return False
    if getattr(current, "_chaosengine_skip_fix", False):
        return True

    def mark_compression_skipped(self: Any, req_id: str, reason: str, step: int) -> None:
        state = self.get(req_id) if hasattr(self, "get") else None
        previous = getattr(state, "last_compression_step", None)
        current(self, req_id=req_id, reason=reason, step=step)
        if state is not None and previous is not None:
            state.last_compression_step = previous

    mark_compression_skipped._chaosengine_skip_fix = True  # type: ignore[attr-defined]
    RequestStateStore.mark_compression_skipped = mark_compression_skipped
    return True


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

    def _vllm_installed(self) -> bool:
        return _module_available("triattention") and _module_available("vllm")

    def vllm_blocked_reason(self) -> str | None:
        """Why the installed triattention + vLLM pair cannot compress, if it cannot."""
        if not self._vllm_installed():
            return None
        return vllm_unsupported_reason(_installed_vllm_version())

    def has_vllm_backend(self) -> bool:
        return self._vllm_installed() and self.vllm_blocked_reason() is None

    def is_available(self) -> bool:
        return self.has_mlx_backend() or self.has_vllm_backend()

    def availability_badge(self) -> str:
        if self.has_mlx_backend():
            return "Experimental"
        if self.has_vllm_backend():
            return "Experimental"
        if self.vllm_blocked_reason():
            return "Unavailable"
        return "Install"

    def availability_tone(self) -> str:
        if self.has_mlx_backend() or self.has_vllm_backend():
            return "warning"
        if self.vllm_blocked_reason():
            return "warning"
        return "install"

    def availability_reason(self) -> str | None:
        if self.has_mlx_backend():
            return (
                "MLX backend (experimental, since 2026-04-09). Requires a calibration "
                "stats file; see triattention/docs/mlx.md."
            )
        if self.has_vllm_backend():
            return (
                "vLLM backend (experimental). Needs a calibration stats file per "
                f"model, <repo name>.pt in {stats_dir()}, made with TriAttention's "
                "scripts/calibrate.py. Runs vLLM without CUDA graphs, so decoding "
                "is slower than the native cache."
            )
        blocked = self.vllm_blocked_reason()
        if blocked:
            return blocked
        if self.on_macos():
            return (
                "Install TriAttention + mlx_lm on macOS: pip install "
                "'triattention @ git+https://github.com/WeianMao/triattention.git' "
                "mlx-lm — then restart the app."
            )
        return (
            "Install TriAttention + vLLM 0.18 on Linux/CUDA: pip install "
            "'triattention @ git+https://github.com/WeianMao/triattention.git' "
            "'vllm<0.19'."
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

        Must be called BEFORE creating a ``vllm.LLM`` instance, and the engine
        core must run in this process (``vllm_runtime`` sets that) for the
        dedup fix to reach the scheduler and worker.
        """
        if not _module_available("triattention"):
            raise RuntimeError("triattention is not installed.")
        reason = vllm_unsupported_reason(_installed_vllm_version())
        if reason:
            raise RuntimeError(reason)
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
        _install_skip_stamp_fix()

    def vllm_runtime(
        self, model_ids: Iterable[str | None],
    ) -> tuple[dict[str, str], dict[str, Any], str]:
        """Env vars, ``vllm.LLM`` kwargs and a note for a TriAttention load.

        Raises ``RuntimeError`` when no stats file exists for the model (the
        plugin would stop the engine at the first compression).

        * The prompt is kept whole (``PROTECT_PREFILL``, not counted in the
          budget); only generated tokens past the budget are compacted. With
          the prompt in the budget, a 7.8K-token needle prompt at budget 2048
          or 4096 lost the needle (``PURPLE-4821`` for ``PURPLE-OTTER-4821``).
        * ``enforce_eager`` and no prefix caching are what the plugin requires.
        * The engine core runs in this process so the dedup fix applies.
        """
        ids = list(model_ids)
        stats = find_stats_file(ids)
        if stats is None:
            names = next((_stats_names(i) for i in ids if i and _stats_names(i)), ["<repo name>.pt"])
            raise RuntimeError(
                "TriAttention needs a calibration stats file for this model; none "
                f"found ({names[0]} in {stats_dir()}). Make one with TriAttention's "
                "scripts/calibrate.py."
            )
        budget = _kv_budget()
        env = {
            "TRIATTN_RUNTIME_KV_BUDGET": str(budget),
            "TRIATTN_RUNTIME_SPARSE_STATS_PATH": str(stats),
            "TRIATTN_RUNTIME_PROTECT_PREFILL": "true",
            "TRIATTN_RUNTIME_INCLUDE_PREFILL_IN_BUDGET": "false",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        }
        llm_kwargs: dict[str, Any] = {
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "max_num_batched_tokens": TRIATTENTION_MAX_BATCHED_TOKENS,
        }
        note = (
            f"TriAttention: generated KV kept to {budget} tokens past the prompt "
            f"(stats {stats.name}); CUDA graphs off."
        )
        return env, llm_kwargs, note

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
