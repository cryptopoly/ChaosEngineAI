"""Per-repo registries for the video runtime.

Pure data + tiny lookup helpers — pipeline class, GGUF / NF4 transformer
class, default-step / default-guidance / scheduler tables, and the
deterministic prompt-enhancement suffixes. Kept out of ``__init__.py`` so
the engine + manager files don't drown in registry data.

Extracted from ``video_runtime/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

from typing import Any


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
    "THUDM/CogVideoX-1.5-5b": {"class_name": "CogVideoXPipeline", "task": "txt2video"},
}


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
#     scheduler with CFG 6 (2.1) or 7.5 (2.2).
#   - HunyuanVideo: tencent/HunyuanVideo recommends 50 steps CFG 6.
#   - Mochi: genmo/mochi-1-preview defaults from upstream pipeline.
#   - CogVideoX: THUDM model cards.
_VIDEO_PIPELINE_DEFAULTS: dict[str, dict[str, Any]] = {
    # LTX-Video pipeline calls ``set_timesteps(mu=...)`` which only
    # ``FlowMatchEulerDiscreteScheduler`` accepts. Older cached snapshots
    # have plain ``EulerDiscreteScheduler`` baked in, so force-swap on
    # every load to keep the pipeline call valid.
    "Lightricks/LTX-Video": {"steps": 30, "guidance": 3.0, "scheduler": "flow-euler"},
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {"steps": 30, "guidance": 6.0, "scheduler": "unipc"},
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": {"steps": 30, "guidance": 6.0, "scheduler": "unipc"},
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": {"steps": 20, "guidance": 7.5, "scheduler": "unipc"},
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": {"steps": 30, "guidance": 7.5, "scheduler": "unipc"},
    "hunyuanvideo-community/HunyuanVideo": {"steps": 50, "guidance": 6.0, "scheduler": None},
    "genmo/mochi-1-preview": {"steps": 64, "guidance": 4.5, "scheduler": None},
    "THUDM/CogVideoX-2b": {"steps": 50, "guidance": 6.0, "scheduler": None},
    "THUDM/CogVideoX-5b": {"steps": 50, "guidance": 7.0, "scheduler": None},
    # CogVideoX 1.5 5B inherits the 5B defaults — refreshed weights but
    # the same step / CFG sweet spot per upstream model card.
    "THUDM/CogVideoX-1.5-5b": {"steps": 50, "guidance": 7.0, "scheduler": None},
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


# Phase E1 — Prompt enhancement.
#
# Diffusion video models train against highly-detailed prompts. Short user
# prompts ("cartoon llama eating straw" — 4 words) under-condition the
# model and produce drifty / blurry output. Reference flows ship a small
# captioning LLM (e.g. Florence-2) that auto-expands short prompts into
# the structured 50-100 word format the model was trained on.
#
# Until we wire a real LLM-based enhancer (Phase E follow-up — would
# require a small instruction model + extra runtime cost), we deterministic-
# ally append model-specific structural hints. This is much weaker than a
# real captioner, but provides immediate uplift for short prompts and
# costs zero extra inference time.
#
# Each entry is the suffix appended to the user's prompt — never replaces
# what the user wrote. The structure mirrors what each upstream model
# card recommends:
#   - LTX-Video: action + visual details + lighting + camera direction
#   - Wan: cinematic descriptors + lens / depth-of-field language
#   - HunyuanVideo: scene + lighting + motion descriptors
#   - Mochi / CogVideoX: high-fidelity descriptors
_PROMPT_ENHANCEMENT_SUFFIXES: dict[str, str] = {
    "Lightricks/LTX-Video": (
        ", smooth natural motion, soft cinematic lighting, shallow depth of "
        "field, gentle camera movement, high detail, 4k cinematic quality."
    ),
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": (
        ", cinematic composition, 35mm film look, shallow depth of field, "
        "soft natural lighting, smooth motion, high detail."
    ),
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": (
        ", cinematic composition, 35mm film look, shallow depth of field, "
        "soft natural lighting, smooth motion, high detail."
    ),
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": (
        ", cinematic composition, 35mm film look, shallow depth of field, "
        "soft natural lighting, smooth motion, high detail."
    ),
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": (
        ", cinematic composition, 35mm film look, shallow depth of field, "
        "soft natural lighting, smooth motion, high detail."
    ),
    "hunyuanvideo-community/HunyuanVideo": (
        ", cinematic scene, dramatic lighting, smooth realistic motion, "
        "high fidelity detail, 4k quality."
    ),
    "genmo/mochi-1-preview": (
        ", cinematic composition, smooth motion, soft natural lighting, "
        "high detail, 4k quality."
    ),
    "THUDM/CogVideoX-2b": (
        ", cinematic composition, smooth motion, soft natural lighting, "
        "high detail."
    ),
    "THUDM/CogVideoX-5b": (
        ", cinematic composition, smooth motion, soft natural lighting, "
        "high detail."
    ),
    # LTX-2 family (mlx-video on Apple Silicon). LTX-2 is a 19B model
    # with stronger structural understanding than LTX 0.9 — slightly
    # less hand-holding via suffix, more emphasis on motion + lighting.
    "prince-canuma/LTX-2-distilled": (
        ", cinematic composition, soft natural lighting, smooth fluid "
        "motion, gentle camera dolly, shallow depth of field, high "
        "fidelity detail."
    ),
    "prince-canuma/LTX-2-dev": (
        ", cinematic composition, soft natural lighting, smooth fluid "
        "motion, gentle camera dolly, shallow depth of field, high "
        "fidelity detail."
    ),
    "prince-canuma/LTX-2.3-distilled": (
        ", cinematic composition, soft natural lighting, smooth fluid "
        "motion, gentle camera dolly, shallow depth of field, high "
        "fidelity detail."
    ),
    "prince-canuma/LTX-2.3-dev": (
        ", cinematic composition, soft natural lighting, smooth fluid "
        "motion, gentle camera dolly, shallow depth of field, high "
        "fidelity detail."
    ),
}

# Word-count threshold under which auto-enhancement fires. Above this the
# user is assumed to have written a structured prompt already and we leave
# it alone.
_PROMPT_ENHANCE_MIN_WORDS = 25


def _enhance_prompt(repo: str, prompt: str) -> tuple[str, str | None]:
    """Append per-model structural hints to short prompts.

    Returns ``(enhanced_prompt, note)``. ``note`` is non-None iff the
    suffix was appended; the caller publishes it to the run log so the
    user sees what was sent to the pipeline.

    Idempotent — a second call on an already-enhanced prompt is a no-op
    (the suffix is detected via substring match). Caller-side word count
    threshold means a long custom prompt is never modified.
    """
    suffix = _PROMPT_ENHANCEMENT_SUFFIXES.get(repo)
    if not suffix:
        return prompt, None
    cleaned = prompt.strip()
    if not cleaned:
        return prompt, None
    if len(cleaned.split()) >= _PROMPT_ENHANCE_MIN_WORDS:
        return prompt, None
    if suffix.strip() in cleaned:
        return prompt, None
    enhanced = cleaned.rstrip(",.!? ") + suffix
    note = (
        f"Auto-enhanced short prompt with model-specific structural hints "
        f"(was {len(cleaned.split())} words, now {len(enhanced.split())} "
        f"words). Toggle off via ``enhancePrompt: false`` if you'd rather "
        f"send the prompt verbatim."
    )
    return enhanced, note
