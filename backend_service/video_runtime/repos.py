"""Per-repo registries for the video runtime.

Pure data + tiny lookup helpers — pipeline class, GGUF / NF4 transformer
class, default-step / default-guidance / scheduler tables, and the
default-step tables. Kept out of ``__init__.py`` so
the engine + manager files don't drown in registry data.

Extracted from ``video_runtime/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

from typing import Any


# LTX-2.3 diffusers-format repos -> True for the distilled model (8 steps with a fixed sigma schedule, no guidance).
# Sizes on disk: the official bf16 repos are ~100 GB (a fp32 Gemma 3 12B text encoder is half of that); the NF4 pack
# is ~31 GB. The dev NF4 pack (`OzzyGT/LTX-2.3-bnb-nf4`) is left out: it holds every connector file twice and needs a
# partial download this catalog has not been run with. LTX-2.5 (gated, `Lightricks/LTX-2.5-Diffusers`, a Gemma 4 text
# encoder) is listed as its distilled pipeline only; its 4-bit-on-load branch lives in ``ltx25_loader``.
LTX2_DIFFUSERS_REPOS: dict[str, bool] = {
    "diffusers/LTX-2.3-Diffusers": False,
    "diffusers/LTX-2.3-Distilled-Diffusers": True,
    "OzzyGT/LTX-2.3-Distilled-bnb-nf4": True,
    "Lightricks/LTX-2.5-Diffusers": True,
}


def is_ltx2_diffusers_repo(repo: str) -> bool:
    return repo in LTX2_DIFFUSERS_REPOS


def is_ltx2_distilled_repo(repo: str) -> bool:
    return LTX2_DIFFUSERS_REPOS.get(repo, False)


# Maps a Hugging Face repo id to the diffusers pipeline class that loads it.
# The class name is looked up dynamically on the ``diffusers`` module so we
# don't blow up at import time if the installed diffusers is older than
# expected — users just see a clearer "unsupported pipeline" error at preload.
PIPELINE_REGISTRY: dict[str, dict[str, str]] = {
    "Lightricks/LTX-Video": {"class_name": "LTXPipeline", "task": "txt2video"},
    "genmo/mochi-1-preview": {"class_name": "MochiPipeline", "task": "txt2video"},
    # Wan 2.1 and 2.2 share the same pipeline class — the version difference
    # lives in the weights, not the pipeline code. We route to the `-Diffusers`
    # mirrors because the base Wan-AI repos ship in the native Wan format
    # (no `model_index.json`) which WanPipeline.from_pretrained can't load.
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {"class_name": "WanPipeline", "task": "txt2video"},
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": {"class_name": "WanPipeline", "task": "txt2video"},
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": {"class_name": "WanPipeline", "task": "txt2video"},
    # Wan 2.2 TI2V-5B is a dense text+image-to-video model — uses the
    # standard WanPipeline loader (no dual-expert routing like A14B).
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": {"class_name": "WanPipeline", "task": "txt2video"},
    # Community-maintained diffusers port of tencent/HunyuanVideo.
    "hunyuanvideo-community/HunyuanVideo": {"class_name": "HunyuanVideoPipeline", "task": "txt2video"},
    # CogVideoX 2B and 5B share the same diffusers pipeline class — the
    # transformer scales but the loader is the same. CogVideoX 1.5 5B
    # (catalog refresh, FU-019 round) uses the same class with refreshed
    # weights and a higher training resolution.
    "THUDM/CogVideoX-2b": {"class_name": "CogVideoXPipeline", "task": "txt2video"},
    "THUDM/CogVideoX-5b": {"class_name": "CogVideoXPipeline", "task": "txt2video"},
    "THUDM/CogVideoX1.5-5B": {"class_name": "CogVideoXPipeline", "task": "txt2video"},
    # LTX-2.3 on diffusers (video with sound). Official bf16 packs for large GPUs and 4-bit (bitsandbytes NF4) packs
    # for 24 GB cards; every one is loaded by the same class, the quantisation lives in the component configs.
    **{repo: {"class_name": "LTX2Pipeline", "task": "txt2video"} for repo in LTX2_DIFFUSERS_REPOS},
}


# Repos whose diffusers pipeline can start from a picture, and which class does it. A run with a start image loads
# this class (same weights as the text-to-video one, so the pipeline is rebuilt when the mode changes). ``endImage``
# says whether the class also takes a last frame: Wan 2.2 TI2V-5B expands its timesteps per frame and only follows
# the first. The LTX-2 MLX engine is not diffusers and is handled by ``image_input_support``.
IMAGE_TO_VIDEO_REGISTRY: dict[str, dict[str, Any]] = {
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": {"class_name": "WanImageToVideoPipeline", "endImage": False},
    # LTX-2: a start picture goes to the image-to-video class, a start and an end picture (or an end alone) to the
    # condition class, which takes the pictures as conditions at the first and last frame.
    **{
        repo: {
            "class_name": "LTX2ImageToVideoPipeline",
            "endImage": True,
            "endClassName": "LTX2ConditionPipeline",
        }
        for repo in LTX2_DIFFUSERS_REPOS
    },
}


def image_input_support(repo: str) -> tuple[bool, bool]:
    """``(takes a start image, takes an end image)`` for a video repo, on any engine this app drives."""
    from backend_service.ltx2_mlx_runtime import _is_ltx2_mlx_repo

    if _is_ltx2_mlx_repo(repo):
        return True, True
    entry = IMAGE_TO_VIDEO_REGISTRY.get(repo)
    return (True, bool(entry.get("endImage"))) if entry else (False, False)


# Maps a base repo to the diffusers transformer class used when loading
# GGUF-quantized DiT weights via ``from_single_file``. city96 currently
# ships LTX-Video, Wan, and HunyuanVideo GGUFs; CogVideoX uses a
# different loader we don't support here. Returning None leaves the
# pipeline on the standard fp16 / bf16 transformer path.
_GGUF_VIDEO_TRANSFORMER_CLASSES: dict[str, str] = {
    "Lightricks/LTX-Video": "LTXVideoTransformer3DModel",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": "WanTransformer3DModel",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": "WanTransformer3DModel",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "WanTransformer3DModel",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "WanTransformer3DModel",
    "hunyuanvideo-community/HunyuanVideo": "HunyuanVideoTransformer3DModel",
}


def _gguf_video_transformer_class_for_repo(repo: str) -> str | None:
    return _GGUF_VIDEO_TRANSFORMER_CLASSES.get(repo)


# Repos for which we know the diffusers transformer subfolder layout used
# by ``BitsAndBytesConfig + from_pretrained(subfolder="transformer")``.
# Same class mapping as GGUF — bnb is just a different quant scheme on
# the same DiT classes. Returning None means we don't have a verified
# NF4 path for this repo (the loader will surface a clear note rather
# than failing the run).
_BNB_NF4_VIDEO_TRANSFORMER_CLASSES: dict[str, str] = {
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": "WanTransformer3DModel",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": "WanTransformer3DModel",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "WanTransformer3DModel",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "WanTransformer3DModel",
    "hunyuanvideo-community/HunyuanVideo": "HunyuanVideoTransformer3DModel",
    "Lightricks/LTX-Video": "LTXVideoTransformer3DModel",
    # The LTX-2 family (video with sound). The pre-quantised 2.3 pack is already 4-bit and is not listed.
    "diffusers/LTX-2.3-Diffusers": "LTX2VideoTransformer3DModel",
    "diffusers/LTX-2.3-Distilled-Diffusers": "LTX2VideoTransformer3DModel",
    "Lightricks/LTX-2.5-Diffusers": "LTX2VideoTransformer3DModel",
}


def _bnb_nf4_transformer_class_for_repo(repo: str) -> str | None:
    return _BNB_NF4_VIDEO_TRANSFORMER_CLASSES.get(repo)


# Per-model sweet-spot inference defaults sourced from upstream model cards
# and reference workflows. The schema-level defaults
# (steps=50, guidance=3.0) are conservative blanks; without per-model
# substitution a Wan 2.1 generation comes out grey/washed because CFG=3
# is half the value the model was trained with. Values come from:
#   - LTX-Video: Lightricks model card recommends 30 steps CFG 3 for the
#     full model; distilled variants override to 8 steps CFG 1.
#   - Wan 2.1 / 2.2: Wan-AI model card recommendations, Uni-PC
#     scheduler with CFG 5.0 (2.1) or 4.0 (2.2 A14B; the card gives 4.0 and
#     3.0 per expert and the pipeline reuses one value for both). Wan 2.2
#     TI2V 5B uses the card's CFG 5.0: at 7.5 the clip came out over-bright
#     with stripes and colour blocks, the longer the clip the worse it got
#     (FU-122). The 2.1 and A14B values follow the cards and were not run
#     here (no A14B weights; 2.1 was 6.0 and 7.5 before).
#   - HunyuanVideo: tencent/HunyuanVideo recommends 50 steps CFG 6.
#   - Mochi: genmo/mochi-1-preview defaults from upstream pipeline.
#   - CogVideoX: THUDM model cards.
_VIDEO_PIPELINE_DEFAULTS: dict[str, dict[str, Any]] = {
    # LTX-Video pipeline calls ``set_timesteps(mu=...)`` which only
    # ``FlowMatchEulerDiscreteScheduler`` accepts. Older cached snapshots
    # have plain ``EulerDiscreteScheduler`` baked in, so force-swap on
    # every load to keep the pipeline call valid.
    "Lightricks/LTX-Video": {"steps": 30, "guidance": 3.0, "scheduler": "flow-euler"},
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {"steps": 30, "guidance": 5.0, "scheduler": "unipc"},
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": {"steps": 30, "guidance": 5.0, "scheduler": "unipc"},
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": {"steps": 20, "guidance": 5.0, "scheduler": "unipc"},
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": {"steps": 30, "guidance": 4.0, "scheduler": "unipc"},
    "hunyuanvideo-community/HunyuanVideo": {"steps": 50, "guidance": 6.0, "scheduler": None},
    "genmo/mochi-1-preview": {"steps": 64, "guidance": 4.5, "scheduler": None},
    "THUDM/CogVideoX-2b": {"steps": 50, "guidance": 6.0, "scheduler": None},
    "THUDM/CogVideoX-5b": {"steps": 50, "guidance": 7.0, "scheduler": None},
    # CogVideoX 1.5 5B inherits the 5B defaults — refreshed weights but
    # the same step / CFG sweet spot per upstream model card.
    "THUDM/CogVideoX1.5-5B": {"steps": 50, "guidance": 7.0, "scheduler": None},
    # LTX-2.3: the dev model runs 30 steps at CFG 3 (Lightricks card), the distilled one 8 steps at CFG 1 with its
    # fixed sigma schedule. The scheduler is the pipeline's own flow-match Euler: never swapped.
    **{
        repo: ({"steps": 8, "guidance": 1.0, "scheduler": None} if distilled else {"steps": 30, "guidance": 3.0, "scheduler": None})
        for repo, distilled in LTX2_DIFFUSERS_REPOS.items()
    },
}

# Schema-level defaults — must mirror ``VideoGenerationRequest`` in
# ``backend_service/models/__init__.py``. We only substitute model-tuned
# values when the user kept the schema defaults, so explicit slider
# tweaks survive untouched.
_REQUEST_DEFAULT_STEPS = 50
_REQUEST_DEFAULT_GUIDANCE = 3.0

# Lightricks' recommended negative-prompt template for LTX-Video. Applied
# only when the request's negativePrompt is empty (or the schema's softer
# default). LTX was trained with strong negative-prompt conditioning, so
# the template materially improves output quality vs an empty / generic
# negative. Reference: huggingface.co/Lightricks/LTX-Video model card +
# Lightricks LTX-Video reference defaults.
_LTX_DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted"
)
