"""FasterCache — diffusers 0.38+ core cache hook.

Post-FU-026. Caches and reuses attention features similar to PAB, plus
optionally skips the unconditional CFG branch when residuals between
successive timesteps are highly correlated. Best on video DiTs running
classifier-free guidance.

Reuses the shared ``apply_diffusion_cache_strategy`` dispatcher's
``rel_l1_thresh`` field as the *spatial_attention_block_skip_range* knob
(rounded to int, clamped >= 2). Default 2.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from . import CacheStrategy


_DEFAULT_SKIP_RANGE = 2
_DEFAULT_TIMESTEP_RANGE = (-1, 681)
_DEFAULT_UNCOND_SKIP_RANGE = 5
_DEFAULT_UNCOND_TIMESTEP_RANGE = (-1, 781)
_DEFAULT_ATTENTION_WEIGHT = 0.3


def _import_config():
    try:
        from diffusers import FasterCacheConfig
        return FasterCacheConfig
    except ImportError:
        from diffusers.hooks import FasterCacheConfig
        return FasterCacheConfig


class FasterCacheStrategy(CacheStrategy):
    """Attention + uncond-branch cache backed by diffusers 0.38 FasterCache hook."""

    @property
    def strategy_id(self) -> str:
        return "fastercache"

    @property
    def name(self) -> str:
        return "FasterCache"

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
            "FasterCache needs diffusers >= 0.38. "
            "Run the GPU runtime installer to upgrade diffusers."
        )

    def applies_to(self) -> frozenset[str]:
        return frozenset({"image", "video"})

    def recommended_thresholds(self) -> dict[str, float]:
        return {"image": 2.0, "video": 2.0}

    def apply_diffusers_hook(
        self,
        pipeline: Any,
        *,
        num_inference_steps: int,
        rel_l1_thresh: float | None,
    ) -> None:
        try:
            FasterCacheConfig = _import_config()
        except ImportError as exc:
            raise NotImplementedError(
                f"diffusers FasterCache hook unavailable: {exc}"
            ) from exc

        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise NotImplementedError(
                "FasterCache requires a DiT pipeline (with .transformer); "
                "this pipeline appears to be UNet-based."
            )
        if not hasattr(transformer, "enable_cache"):
            raise NotImplementedError(
                "transformer.enable_cache is not available on this pipeline. "
                "Diffusers >= 0.38 is required for the FasterCache registry path."
            )

        if rel_l1_thresh is not None and rel_l1_thresh >= 2:
            skip_range = int(round(rel_l1_thresh))
        else:
            skip_range = _DEFAULT_SKIP_RANGE

        del num_inference_steps  # FasterCache derives schedule from timesteps.

        try:
            config = FasterCacheConfig(
                spatial_attention_block_skip_range=skip_range,
                spatial_attention_timestep_skip_range=_DEFAULT_TIMESTEP_RANGE,
                current_timestep_callback=lambda: getattr(pipeline, "current_timestep", 0),
                attention_weight_callback=lambda _: _DEFAULT_ATTENTION_WEIGHT,
                unconditional_batch_skip_range=_DEFAULT_UNCOND_SKIP_RANGE,
                unconditional_batch_timestep_skip_range=_DEFAULT_UNCOND_TIMESTEP_RANGE,
                tensor_format="BFCHW",
            )
        except TypeError:
            config = FasterCacheConfig()

        transformer.enable_cache(config)
