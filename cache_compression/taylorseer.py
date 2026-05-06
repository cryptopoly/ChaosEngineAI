"""TaylorSeer Cache — diffusers 0.38+ core cache hook.

Post-FU-026. Approximates intermediate transformer activations across denoise
steps via a Taylor series expansion, reusing them at fixed intervals to skip
full forwards. Strong wall-time wins on FLUX (~1.6× at cache_interval=5,
max_order=1, disable_cache_before_step=10).

Unlike FBCache (threshold-based), TaylorSeer is interval-based. Reuses the
shared ``apply_diffusion_cache_strategy`` dispatcher's ``rel_l1_thresh``
field as the *cache_interval* knob (rounded to nearest int, clamped >= 2).
When ``rel_l1_thresh`` is ``None`` or below 2, falls back to the
diffusers-blog default of 5.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from . import CacheStrategy


_DEFAULT_CACHE_INTERVAL = 5
_DEFAULT_MAX_ORDER = 1


def _import_config():
    try:
        from diffusers import TaylorSeerCacheConfig
        return TaylorSeerCacheConfig
    except ImportError:
        from diffusers.hooks import TaylorSeerCacheConfig
        return TaylorSeerCacheConfig


class TaylorSeerCacheStrategy(CacheStrategy):
    """Taylor-series interval cache backed by diffusers 0.38 ``TaylorSeerCacheConfig``."""

    @property
    def strategy_id(self) -> str:
        return "taylorseer"

    @property
    def name(self) -> str:
        return "TaylorSeer Cache"

    def is_available(self) -> bool:
        if importlib.util.find_spec("diffusers") is None:
            return False
        try:
            _import_config()
        except Exception:
            return False
        return True

    def availability_badge(self) -> str:
        return "Ready" if self.is_available() else "Upgrade"

    def availability_reason(self) -> str | None:
        if self.is_available():
            return None
        return (
            "TaylorSeer Cache needs diffusers >= 0.38. "
            "Run the GPU runtime installer to upgrade diffusers."
        )

    def applies_to(self) -> frozenset[str]:
        return frozenset({"image", "video"})

    def recommended_thresholds(self) -> dict[str, float]:
        return {"image": 5.0, "video": 4.0}

    def apply_diffusers_hook(
        self,
        pipeline: Any,
        *,
        num_inference_steps: int,
        rel_l1_thresh: float | None,
    ) -> None:
        try:
            TaylorSeerCacheConfig = _import_config()
        except ImportError as exc:
            raise NotImplementedError(
                f"diffusers TaylorSeer hook unavailable: {exc}"
            ) from exc

        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise NotImplementedError(
                "TaylorSeer Cache requires a DiT pipeline (with .transformer); "
                "this pipeline appears to be UNet-based. Use TeaCache or stay on stock."
            )
        if not hasattr(transformer, "enable_cache"):
            raise NotImplementedError(
                "transformer.enable_cache is not available on this pipeline. "
                "Diffusers >= 0.38 is required for the TaylorSeer registry path."
            )

        if rel_l1_thresh is not None and rel_l1_thresh >= 2:
            cache_interval = int(round(rel_l1_thresh))
        else:
            cache_interval = _DEFAULT_CACHE_INTERVAL

        steps = max(1, int(num_inference_steps))
        warmup = max(0, min(steps // 2, max(2, steps // 4))) if steps >= 4 else 0

        try:
            config = TaylorSeerCacheConfig(
                cache_interval=cache_interval,
                max_order=_DEFAULT_MAX_ORDER,
                disable_cache_before_step=warmup,
            )
        except TypeError:
            config = TaylorSeerCacheConfig()

        transformer.enable_cache(config)
