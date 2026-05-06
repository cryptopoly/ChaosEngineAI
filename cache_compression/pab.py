"""Pyramid Attention Broadcast — diffusers 0.38+ core cache hook.

Post-FU-026. Skips spatial-attention computations on a fixed timestep
schedule, exploiting the small differences in attention outputs between
successive denoise steps. Most effective on video DiTs where timestep
schedules are long (CogVideoX, HunyuanVideo, Wan).

Reuses the shared ``apply_diffusion_cache_strategy`` dispatcher's
``rel_l1_thresh`` field as the *spatial_attention_block_skip_range* knob
(rounded to int, clamped >= 2). Default 2 = skip every other step's
spatial attention.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from . import CacheStrategy


_DEFAULT_SKIP_RANGE = 2
# Diffusers blog default for CogVideoX. Smaller intervals slow inference;
# larger intervals harm quality. Validated for video DiTs.
_DEFAULT_TIMESTEP_RANGE = (100, 800)


def _import_config():
    try:
        from diffusers import PyramidAttentionBroadcastConfig
        return PyramidAttentionBroadcastConfig
    except ImportError:
        from diffusers.hooks import PyramidAttentionBroadcastConfig
        return PyramidAttentionBroadcastConfig


class PyramidAttentionBroadcastStrategy(CacheStrategy):
    """Spatial-attention skip schedule backed by diffusers 0.38 PAB hook."""

    @property
    def strategy_id(self) -> str:
        return "pab"

    @property
    def name(self) -> str:
        return "Pyramid Attention Broadcast"

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
            "Pyramid Attention Broadcast needs diffusers >= 0.38. "
            "Run the GPU runtime installer to upgrade diffusers."
        )

    def applies_to(self) -> frozenset[str]:
        return frozenset({"image", "video"})

    def recommended_thresholds(self) -> dict[str, float]:
        # Slider repurposed as skip_range. Image DiTs run shorter
        # schedules where larger skips bite harder; video DiTs tolerate
        # bigger intervals.
        return {"image": 2.0, "video": 3.0}

    def apply_diffusers_hook(
        self,
        pipeline: Any,
        *,
        num_inference_steps: int,
        rel_l1_thresh: float | None,
    ) -> None:
        try:
            PyramidAttentionBroadcastConfig = _import_config()
        except ImportError as exc:
            raise NotImplementedError(
                f"diffusers PAB hook unavailable: {exc}"
            ) from exc

        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise NotImplementedError(
                "Pyramid Attention Broadcast requires a DiT pipeline "
                "(with .transformer); this pipeline appears to be UNet-based."
            )
        if not hasattr(transformer, "enable_cache"):
            raise NotImplementedError(
                "transformer.enable_cache is not available on this pipeline. "
                "Diffusers >= 0.38 is required for the PAB registry path."
            )

        if rel_l1_thresh is not None and rel_l1_thresh >= 2:
            skip_range = int(round(rel_l1_thresh))
        else:
            skip_range = _DEFAULT_SKIP_RANGE

        del num_inference_steps  # PAB derives its own schedule from timesteps.

        try:
            config = PyramidAttentionBroadcastConfig(
                spatial_attention_block_skip_range=skip_range,
                spatial_attention_timestep_skip_range=_DEFAULT_TIMESTEP_RANGE,
                current_timestep_callback=lambda: getattr(pipeline, "current_timestep", 0),
            )
        except TypeError:
            config = PyramidAttentionBroadcastConfig()

        transformer.enable_cache(config)
