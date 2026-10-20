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

import struct
from pathlib import Path
from typing import Any


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


# ---------------------------------------------------------------------------
# Non-chat models
#
# Anything that is not an image / video / draft model but still cannot answer
# a chat message: 3D generators, vision and text encoders, voice models,
# bare weight files. Discovery used to call all of these "text", so the Chat
# tab's My Models list filled up with Trellis / TripoSG / DINO components.
# ---------------------------------------------------------------------------

# A path component that marks a tool tree of non-chat parts (3D generators).
_NON_CHAT_PATH_KEYWORDS = (
    "trellis", "triposg", "triposr", "hunyuan3d", "hunyuan-3d",
    "instantmesh", "stable-fast-3d",
)

# Names of checkpoints that are never a chat model.
_NON_CHAT_NAME_KEYWORDS = (
    "text_encoder", "text-encoder", "kokoro", "whisper", "parakeet",
    "dinov", "siglip", "clip-vit",
)

# ``config.json`` model types that carry a "ConditionalGeneration" or
# similar head but are not chat models.
_NON_CHAT_CONFIG_MODEL_TYPES = frozenset({
    "whisper", "t5", "mt5", "umt5", "bart", "mbart", "pegasus",
    "speech-encoder-decoder", "musicgen", "bark", "vits",
    "clip", "siglip", "dinov2", "dinov3", "vit", "bert", "roberta",
    "xlm-roberta", "distilbert", "wav2vec2", "hubert",
})

# Architecture-name endings of models that generate text from a prompt.
_CHAT_ARCHITECTURE_SUFFIXES = (
    "CausalLM", "ConditionalGeneration", "LMHeadModel", "ChatModel",
)

# ``general.architecture`` values of GGUF files that are not language models.
_NON_CHAT_GGUF_ARCHS = frozenset({
    "flux", "sd", "sd1", "sd2", "sd3", "sdxl", "wan", "ltxv",
    "hunyuan_video", "qwen_image", "z_image",
    "clip", "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2",
    "jina-bert-v3", "modern-bert", "neo-bert", "t5", "t5encoder",
    "whisper", "wavtokenizer-dec",
})


def _gguf_architecture(path: Path) -> str | None:
    """``general.architecture`` of a GGUF file, read from its header (None if not found).

    The key is written first by every converter, so a few key/value pairs are
    enough; an array before it means we give up rather than parse the file.
    """
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != b"GGUF":
                return None
            handle.read(4)  # version
            _tensors, kv_count = struct.unpack("<QQ", handle.read(16))
            scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            for _ in range(min(kv_count, 12)):
                (key_len,) = struct.unpack("<Q", handle.read(8))
                key = handle.read(key_len).decode("utf-8", "replace")
                (value_type,) = struct.unpack("<I", handle.read(4))
                if value_type == 8:
                    (value_len,) = struct.unpack("<Q", handle.read(8))
                    value = handle.read(value_len).decode("utf-8", "replace")
                    if key == "general.architecture":
                        return value
                elif value_type in scalar_sizes:
                    handle.read(scalar_sizes[value_type])
                else:
                    return None
    except (OSError, struct.error):
        return None
    return None


def _first_gguf(path: Path) -> Path | None:
    """The model GGUF of a file or directory (companion mmproj files skipped)."""
    if path.is_file():
        return path if path.suffix.lower() == ".gguf" else None
    try:
        for candidate in sorted(path.rglob("*.gguf")):
            if "mmproj" not in candidate.name.lower():
                return candidate
    except OSError:
        return None
    return None


def _looks_like_non_chat_model(
    path: Path,
    name: str,
    file_format: str,
    source_kind: str,
    config: dict[str, Any] | None,
    has_chat_template: bool,
) -> bool:
    """True for a checkpoint that cannot answer a chat message.

    Evidence, strongest first: a bare ``.safetensors`` file (no config or
    tokenizer, so nothing can load it as a chat model); a GGUF whose
    architecture is not a language model; a ``config.json`` whose
    architecture is not a generator (unless the folder ships a chat
    template); and, for entries with no readable config, the path and name.
    """
    if source_kind == "File" and file_format.upper() == "SAFETENSORS":
        return True

    lowered_path = str(path).lower()
    if any(keyword in lowered_path for keyword in _NON_CHAT_PATH_KEYWORDS):
        return True
    lowered_name = name.lower()
    if any(keyword in lowered_name for keyword in _NON_CHAT_NAME_KEYWORDS):
        return True

    if file_format.upper() == "GGUF":
        gguf = _first_gguf(path)
        if gguf is not None and _gguf_architecture(gguf) in _NON_CHAT_GGUF_ARCHS:
            return True

    if config is not None:
        model_type = str(config.get("model_type") or "").lower()
        if model_type in _NON_CHAT_CONFIG_MODEL_TYPES:
            return True
        architectures = [a for a in (config.get("architectures") or []) if isinstance(a, str)]
        if architectures:
            if any(a.endswith(_CHAT_ARCHITECTURE_SUFFIXES) for a in architectures):
                return False
            return not has_chat_template
        if not model_type:
            # No architecture or model type at all: a diffusers component or
            # a voice model, not something mlx-lm / transformers would chat with.
            return not has_chat_template
    return False
