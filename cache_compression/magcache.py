"""MagCache — diffusers 0.38+ core cache hook (FLUX-only without calibration).

Post-FU-026. Skips transformer blocks based on residual-magnitude decay over
the diffusion process. Requires per-model "magnitude ratios" — diffusers
ships pre-calibrated ratios for FLUX (``FLUX_MAG_RATIOS`` in
``diffusers.hooks.mag_cache``); other model families need a calibration
pass before MagCache can run.

This adapter:
- Detects FLUX pipelines via class name and uses the shipped ratios.
- Raises ``NotImplementedError`` with a helpful message for other DiTs,
  pointing to the ``MagCacheConfig(calibrate=True, ...)`` flow.

Calibration UX is a planned follow-up; for now MagCache is FLUX-only in the
registry path. ``applies_to()`` stays ``{"image", "video"}`` so the strategy
is visible in both Studios — non-FLUX video DiTs surface the calibration
message via ``runtimeNote`` rather than crashing.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from . import CacheStrategy


def _import_config():
    try:
        from diffusers import MagCacheConfig
        return MagCacheConfig
    except ImportError:
        from diffusers.hooks import MagCacheConfig
        return MagCacheConfig


def _import_flux_ratios():
    from diffusers.hooks.mag_cache import FLUX_MAG_RATIOS
    return FLUX_MAG_RATIOS


class MagCacheStrategy(CacheStrategy):
    """Magnitude-based cache backed by diffusers 0.38 ``MagCacheConfig``."""

    @property
    def strategy_id(self) -> str:
        return "magcache"

    @property
    def name(self) -> str:
        return "MagCache"

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
            "MagCache needs diffusers >= 0.38. "
            "Run the GPU runtime installer to upgrade diffusers."
        )

    def applies_to(self) -> frozenset[str]:
        return frozenset({"image", "video"})

    def recommended_thresholds(self) -> dict[str, float]:
        # MagCache's main knob is the calibration ratio array, not a
        # single threshold. The slider value is ignored by this adapter
        # and the dispatcher passes through whatever the UI sends.
        return {"image": 0.0, "video": 0.0}

    @staticmethod
    def _is_flux_pipeline(pipeline: Any) -> bool:
        cls_name = pipeline.__class__.__name__.lower()
        return "flux" in cls_name

    def apply_diffusers_hook(
        self,
        pipeline: Any,
        *,
        num_inference_steps: int,
        rel_l1_thresh: float | None,
    ) -> None:
        try:
            MagCacheConfig = _import_config()
        except ImportError as exc:
            raise NotImplementedError(
                f"diffusers MagCache hook unavailable: {exc}"
            ) from exc

        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise NotImplementedError(
                "MagCache requires a DiT pipeline (with .transformer); "
                "this pipeline appears to be UNet-based."
            )
        if not hasattr(transformer, "enable_cache"):
            raise NotImplementedError(
                "transformer.enable_cache is not available on this pipeline. "
                "Diffusers >= 0.38 is required for the MagCache registry path."
            )

        del rel_l1_thresh  # MagCache has no single-threshold knob.

        if not self._is_flux_pipeline(pipeline):
            raise NotImplementedError(
                "MagCache requires per-model calibration. Pre-calibrated ratios "
                "ship only for FLUX (FLUX_MAG_RATIOS). For other DiTs, run a "
                "calibration pass first via "
                "MagCacheConfig(calibrate=True, num_inference_steps=...) and "
                "pass the printed ratios via mag_ratios=[...]. Until "
                "calibration UX lands, use FBCache or TaylorSeer."
            )

        try:
            flux_ratios = _import_flux_ratios()
        except ImportError as exc:
            raise NotImplementedError(
                f"FLUX_MAG_RATIOS missing from diffusers.hooks.mag_cache: {exc}"
            ) from exc

        try:
            config = MagCacheConfig(
                mag_ratios=list(flux_ratios),
                num_inference_steps=int(num_inference_steps),
            )
        except TypeError:
            config = MagCacheConfig(mag_ratios=list(flux_ratios))

        transformer.enable_cache(config)
