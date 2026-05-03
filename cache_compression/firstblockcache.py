"""First Block Cache (FBCache) — diffusers 0.36+ generic DiT cache hook.

FU-015. Replaces the per-model vendored TeaCache forwards with a single
model-agnostic hook that diffusers ships in ``diffusers.hooks``. Closes
FU-007 (Wan TeaCache) — the Wan signature mismatch that motivated the
deferral disappears here because FBCache attaches to ``pipeline.transformer``
without needing a custom forward.

The hook compares each step's first-block residual against the previous
step's. When the L1-relative delta is below the threshold, all subsequent
blocks reuse cached residuals, skipping a full forward through the rest
of the DiT. Threshold 0.12 is the diffusers-blog recommendation for
FLUX.1-dev (≈1.8× speedup, no visible quality loss).

Applies to image + video DiTs (FLUX, SD3.5, Wan2.1/2.2, HunyuanVideo,
LTX-Video, CogVideoX, Mochi). Does NOT apply to UNet pipelines
(SD1.5/SDXL); ``applies_to`` would still report ``{"image","video"}`` so
the strategy is *visible* to those Studios, but the runtime hook will
raise ``NotImplementedError`` for non-DiT pipelines and the engine
swallows that into a "not applied" runtimeNote.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from . import CacheStrategy


# Default threshold matching diffusers blog post on FBCache for FLUX:
# 0.12 yields ~1.8× speedup with imperceptible quality drift on a wide
# prompt set. Lower (0.08) is safer for video DiTs where temporal
# consistency is more sensitive; higher (0.20) is more aggressive.
_DEFAULT_THRESHOLD = 0.12


class FirstBlockCacheStrategy(CacheStrategy):
    """Generic block-cache strategy backed by ``diffusers.hooks.apply_first_block_cache``."""

    @property
    def strategy_id(self) -> str:
        return "fbcache"

    @property
    def name(self) -> str:
        return "First Block Cache"

    def is_available(self) -> bool:
        if importlib.util.find_spec("diffusers") is None:
            return False
        try:
            from diffusers.hooks import apply_first_block_cache  # noqa: F401
            from diffusers.hooks import FirstBlockCacheConfig  # noqa: F401
        except Exception:
            return False
        return True

    def availability_badge(self) -> str:
        if self.is_available():
            return "Ready"
        return "Upgrade"

    def availability_reason(self) -> str | None:
        if self.is_available():
            return None
        return (
            "First Block Cache needs diffusers >= 0.36. "
            "Run the GPU runtime installer to upgrade diffusers."
        )

    def applies_to(self) -> frozenset[str]:
        return frozenset({"image", "video"})

    def recommended_thresholds(self) -> dict[str, float]:
        """UI hints for the threshold slider per domain."""
        return {"image": 0.12, "video": 0.08}

    def apply_diffusers_hook(
        self,
        pipeline: Any,
        *,
        num_inference_steps: int,
        rel_l1_thresh: float | None,
    ) -> None:
        """Attach FBCache to ``pipeline.transformer``.

        Raises ``NotImplementedError`` for pipelines without a ``transformer``
        attribute (UNet-based SD1.5/SDXL) — caller swallows this into a
        runtimeNote so the user sees "not applied" instead of a crash.
        """
        try:
            from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
        except ImportError as exc:
            raise NotImplementedError(
                f"diffusers FBCache hook unavailable: {exc}"
            ) from exc

        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise NotImplementedError(
                "First Block Cache requires a DiT pipeline (with .transformer); "
                "this pipeline appears to be UNet-based. Use TeaCache or stay on stock."
            )

        threshold = (
            rel_l1_thresh
            if rel_l1_thresh is not None and rel_l1_thresh > 0
            else _DEFAULT_THRESHOLD
        )
        # ``num_inference_steps`` is accepted for API parity with TeaCache
        # but FBCache derives its own warmup internally — diffusers' hook
        # only takes a threshold + optional num_blocks_to_skip.
        del num_inference_steps  # noqa: F841 — intentionally unused

        try:
            config = FirstBlockCacheConfig(threshold=float(threshold))
        except TypeError:
            # Older 0.36 betas exposed positional-only construction. Fall
            # back to the no-arg form and set threshold post-construction
            # if available.
            config = FirstBlockCacheConfig()
            if hasattr(config, "threshold"):
                try:
                    config.threshold = float(threshold)
                except Exception:
                    pass

        apply_first_block_cache(transformer, config)
