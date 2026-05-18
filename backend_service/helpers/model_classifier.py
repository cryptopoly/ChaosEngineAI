"""Keyword-based model-family classifiers.

Decides at discovery time whether a model directory is a draft / video /
image / chat checkpoint based on its name. Pure heuristics — used to keep
companion checkpoints (DFlash drafts) and Diffusers video pipelines out
of the chat-oriented My Models list, and to flag image diffusion repos
for the image catalog.

Extracted from ``backend_service/helpers/discovery.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.discovery`` so existing
``from backend_service.helpers.discovery import _looks_like_image_model``
imports keep working.
"""

from __future__ import annotations

from pathlib import Path


_IMAGE_MODEL_KEYWORDS = (
    "stable-diffusion", "sdxl", "flux.", "flux1", "flux-",
    "dall-e", "imagen", "kandinsky", "wuerstchen",
    "diffusion-pipe", "qwen-image", "qwen/qwen-image",
    "sana_sprint", "sana-sprint", "sana sprint", "sana_1600m", "sana-1600m",
    # Tracked latest seeds — keep aligned with the
    # ``LATEST_IMAGE_TRACKED_SEEDS`` repos in
    # ``backend_service/catalog/image_models.py`` so newly-tracked
    # families don't leak into the chat-oriented My Models list.
    "ernie-image",   # baidu/ERNIE-Image, baidu/ERNIE-Image-Turbo
    "nucleus-image", # NucleusAI/Nucleus-Image
    "z-image",       # Tongyi-MAI/Z-Image, Z-Image-Turbo
    "hidream",       # HiDream-ai/HiDream-I1-Full + siblings
    "glm-image",     # zai-org/GLM-Image
)


_DRAFT_MODEL_KEYWORDS = (
    "-dflash", "/dflash", "-draft", "-eagle",
)


# Video diffusion pipelines. Keep keywords specific enough that they don't
# collide with chat LLMs or image diffusion checkpoints — e.g. "hunyuanvideo"
# not "hunyuan" (which would catch the Hunyuan image model), "wan2" not "wan"
# (too generic), "mochi-1" not "mochi". New video families added to
# ``backend_service/catalog/video_models.py`` should also get a keyword here.
_VIDEO_MODEL_KEYWORDS = (
    "hunyuanvideo",
    "wan-ai/",
    "wan2.",
    "wan2-",
    "-t2v-",
    "-i2v-",
    "-v2v-",
    "mochi-1",
    "cogvideo",
    "ltx-video",
    "ltx-2",
    "zeroscope",
    "animatediff",
)


def _looks_like_draft_model(name: str) -> bool:
    """Return True if this looks like a speculative decoding draft model.

    Draft models (DFlash, EAGLE, etc.) are companion checkpoints, not
    standalone LLMs.  They should not appear in the model picker.
    """
    lower = name.lower()
    return any(kw in lower for kw in _DRAFT_MODEL_KEYWORDS)


def _looks_like_video_model(name: str) -> bool:
    """Return True if this looks like a video diffusion pipeline.

    Video models (LTX-Video, Wan, HunyuanVideo, Mochi, CogVideo, …) are
    Diffusers pipelines with much larger VRAM footprints than LLMs and
    their own dedicated Studio/Discover UI under the Video section. They
    should be excluded from the chat-oriented My Models list.

    Detection is keyword-only here because video Diffusers pipelines share
    the ``model_index.json`` marker with image pipelines — we can't use that
    to discriminate. When a partial HF cache download hasn't yet produced
    ``model_index.json``, the name-based match is what keeps them out of
    the LLM list.
    """
    lower = name.lower()
    return any(kw in lower for kw in _VIDEO_MODEL_KEYWORDS)


def _looks_like_image_model(path: Path, name: str) -> bool:
    """Return True if this looks like a diffusion / image generation model."""
    lower_name = name.lower()
    if any(kw in lower_name for kw in _IMAGE_MODEL_KEYWORDS):
        return True
    # Diffusers models have model_index.json
    if (path / "model_index.json").exists():
        return True
    return False
