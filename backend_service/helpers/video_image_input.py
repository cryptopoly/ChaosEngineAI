"""Which video models take a start / end image (image-to-video).

One place for the rule that Studio, the video route, the storyboard and the
agent bridge all apply: ltx-2-mlx packs take an optional start and end frame,
and the Wan 2.2 I2V A14B rows (diffusers ``WanImageToVideoPipeline``) need a
start frame and take an optional end frame.
"""

from __future__ import annotations

from typing import Any

# Diffusers repos whose pipeline is image-to-video (``image=`` / ``last_image=``).
# Must match the ``img2video`` rows of ``video_runtime.repos.PIPELINE_REGISTRY``.
WAN_I2V_REPOS: frozenset[str] = frozenset({"Wan-AI/Wan2.2-I2V-A14B-Diffusers"})


def _repo(variant: dict[str, Any]) -> str:
    return str(variant.get("repo") or "")


def accepts_image_input(variant: dict[str, Any]) -> bool:
    """True when the model can start from an image (and end on one)."""
    from backend_service.ltx2_mlx_runtime import _is_ltx2_mlx_repo

    repo = _repo(variant)
    return _is_ltx2_mlx_repo(repo) or repo in WAN_I2V_REPOS


def requires_start_image(variant: dict[str, Any]) -> bool:
    """True when the model cannot make a clip from text alone."""
    return _repo(variant) in WAN_I2V_REPOS
