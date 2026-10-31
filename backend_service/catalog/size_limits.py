"""Per-model image and video size limits for the catalog.

Every variant carries three fields, written here so the choice and its source
sit in one place and a new family has to be given a row (``tests/test_size_limits.py``
fails on a family that has none):

``maxSide``
    Largest edge in pixels the size guard allows for the model.
``maxPixels``
    Largest area in pixels. ``None`` below means ``maxSide`` squared.
``nativeSide``
    The resolution the model is trained for. Images: the square side. Video:
    the long edge of the native frame (832 for 832x480).

A figure marked ``unverified`` has no statement in the model card or upstream
documentation; it is the generic ``DEFAULT_MAX_SIDE`` (4096) for a modern DiT
model, or a smaller value where the model is plainly smaller-scale. The memory
gate (``helpers/memory_gate.py``) stops a request the machine cannot hold, so a
generous limit here does not by itself let a small machine run out of memory.
``scripts/measure-image-ceiling.py`` measures the real ceiling on a machine.

A value written directly on a catalog variant wins over this table.
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple

from backend_service.helpers.generation_limits import DEFAULT_MAX_SIDE, DEFAULT_NATIVE_SIDE


class SizeRow(NamedTuple):
    max_side: int
    max_pixels: int | None
    native_side: int


_UNKNOWN_IMAGE = SizeRow(DEFAULT_MAX_SIDE, None, DEFAULT_NATIVE_SIDE)  # unverified

# BFL's FLUX.2 limits (help.bfl.ai, "What are the resolution limits?"): 64 px minimum,
# 4 megapixels maximum, multiples of 16, "at or below 2MP for optimal quality and speed",
# 1024x1024 by default. The page lists FLUX.2 as a family; klein and dev are not named.
_FLUX2 = SizeRow(4096, 4 * 1024 * 1024, 1024)

# Qwen-Image-2.1 and its Turbo (HF cards): the aspect-ratio table is 2048x2048 for 1:1 and
# 2400x1792 / 2752x1536 for the others (about 4.2 MP), and the examples render 2048x2048.
# Nothing is said about a maximum; 4096 is unverified.
_QWEN_IMAGE_2_1 = SizeRow(DEFAULT_MAX_SIDE, None, 2048)

# Family id -> limits.
IMAGE_FAMILY_LIMITS: dict[str, SizeRow] = {
    # FLUX.1 schnell / dev: the HF cards and BFL's resolution page do not cover FLUX.1 text-to-image
    # (the page lists FLUX.2 and Kontext only). 4096 is unverified; the diffusers example is 1024x1024.
    "flux-fast": SizeRow(DEFAULT_MAX_SIDE, None, 1024),
    "flux-dev": SizeRow(DEFAULT_MAX_SIDE, None, 1024),
    # SD 3.5 Medium (HF card): progressive training 256 -> 512 -> 768 -> 1024 -> 1440, and Stability's
    # launch notes give 0.25 to 2 megapixels. 1440x1440 is 2,073,600 pixels.
    "sd35-medium": SizeRow(2048, 1440 * 1440, 1024),
    # SD 3.5 Large Turbo: the card gives no figure; Stability rates SD 3.5 Large at 1 megapixel.
    # Unverified: capped at the SDXL-class edge with twice that area.
    "sd35-turbo": SizeRow(2048, 2 * 1024 * 1024, 1024),
    # SDXL base: trained on multi-aspect buckets of about 1024x1024 area (SDXL paper); the card states
    # no maximum. 2048 is the SD / SDXL-class limit chosen for this catalog.
    "sdxl-balanced": SizeRow(2048, None, 1024),
    # Qwen-Image 20B (HF card): recommended sizes run from 1328x1328 (1:1) to 1664x928, about 1.8 MP.
    # No maximum is stated; 4096 is unverified.
    "qwen-image": SizeRow(DEFAULT_MAX_SIDE, None, 1328),
    # Z-Image Turbo: the card shows 1024x1024 only. 4096 is unverified.
    "z-image": SizeRow(DEFAULT_MAX_SIDE, None, 1024),
    # Iris-3B (HF card): "generated at about one megapixel; for larger prints, use an upscaler". It has
    # no VAE, so memory grows with the pixel count. The card gives no maximum: 2048 is unverified.
    "iris-3b": SizeRow(2048, None, 1024),
    "flux-2": _FLUX2,
    # FLUX 3 Image Klein: the card names no resolution but the 1024x1024 benchmark. 4096 is unverified.
    "flux-3": SizeRow(DEFAULT_MAX_SIDE, None, 1024),
}

# Variant id -> limits, for rows whose model differs from the rest of the family.
IMAGE_VARIANT_LIMITS: dict[str, SizeRow] = {
    "Qwen/Qwen-Image-2.1-mflux": _QWEN_IMAGE_2_1,
    "Qwen/Qwen-Image-2.1-mflux-q8": _QWEN_IMAGE_2_1,
    "Qwen/Qwen-Image-2.1-Turbo-mflux": _QWEN_IMAGE_2_1,
    "leejet/Qwen-Image-2.1-GGUF-q4k": _QWEN_IMAGE_2_1,
}

# Tracked-latest seeds (discover only, not launchable) by repo. Anything not listed
# is unverified at DEFAULT_MAX_SIDE.
IMAGE_SEED_LIMITS: dict[str, SizeRow] = {
    "black-forest-labs/FLUX.2-dev": _FLUX2,
    "black-forest-labs/FLUX.2-klein-9B": _FLUX2,
    "black-forest-labs/FLUX.2-klein-4B": _FLUX2,
    "fal/FLUX.2-dev-Turbo": _FLUX2,  # distilled FLUX.2 dev: assumed to keep its limits
    "Qwen/Qwen-Image": SizeRow(DEFAULT_MAX_SIDE, None, 1328),
    "Qwen/Qwen-Image-2512": SizeRow(DEFAULT_MAX_SIDE, None, 1328),
    "Qwen/Qwen-Image-Edit": SizeRow(DEFAULT_MAX_SIDE, None, 1328),
    "Qwen/Qwen-Image-Edit-2511": SizeRow(DEFAULT_MAX_SIDE, None, 1328),
    # The repo names say 1024px; unverified, held at the SDXL-class edge.
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers": SizeRow(2048, None, 1024),
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers": SizeRow(2048, None, 1024),
}

_UNKNOWN_VIDEO = SizeRow(DEFAULT_MAX_SIDE, None, 1024)  # unverified

# Wan 2.1 / 2.2 (Wan-Video READMEs): T2V-14B and the 2.2 A14B models "support 480P and 720P"; the
# 1.3B model "is capable of generating videos at 720P", with less stable results, and 480P is
# recommended. 1280x720 is 921,600 pixels.
_WAN = SizeRow(1280, 1280 * 720, 832)

VIDEO_FAMILY_LIMITS: dict[str, SizeRow] = {
    # LTX-Video 0.9.x (HF card): sizes divisible by 32, and "works best on resolutions under 720 x 1280".
    "ltx-video": SizeRow(1280, 1280 * 720, 768),
    # LTX-2 / 2.3 / 2.5: the HF card says width and height divisible by 32 and names no maximum; the
    # ltx-2-mlx README shows 1280x720 and 1920x1024 runs; measured here: 1280x704 x 121 frames in 18 GB.
    # 4096 is unverified (Lightricks markets 4K, which the card does not state).
    "ltx-2": SizeRow(DEFAULT_MAX_SIDE, 4096 * 2304, 768),
    "wan-2-1": _WAN,
    "wan-2-2": _WAN,
    # HunyuanVideo (Tencent README): the supported settings are 720px1280px129f and 544px960px129f.
    "hunyuan-video": SizeRow(1280, 1280 * 720, 1280),
    # Mochi 1 preview (HF card): "generates videos at 480p today"; the example is 848x480.
    "mochi-1": SizeRow(848, 848 * 480, 848),
    # CogVideoX 2B / 5B (HF card): "720 x 480, no support for other resolutions". The 1.5 model
    # overrides this below.
    "cogvideox": SizeRow(720, 720 * 480, 720),
    # LongLive (hidden from the catalog): Wan 2.1 1.3B base, 832x480 output per our catalog note. Unverified.
    "longlive": SizeRow(832, 832 * 480, 832),
}

VIDEO_VARIANT_LIMITS: dict[str, SizeRow] = {
    # CogVideoX 1.5 (HF card): text-to-video 1360x768; image-to-video: shorter edge 768, longer 768 to 1360.
    "THUDM/CogVideoX1.5-5B": SizeRow(1360, 1360 * 768, 1360),
}


def _write(variant: dict[str, Any], row: SizeRow) -> None:
    variant.setdefault("maxSide", row.max_side)
    variant.setdefault("maxPixels", row.max_pixels if row.max_pixels else row.max_side * row.max_side)
    variant.setdefault("nativeSide", row.native_side)


def apply_image_size_limits(
    families: Iterable[dict[str, Any]], seeds: Iterable[dict[str, Any]] = ()
) -> None:
    """Add ``maxSide`` / ``maxPixels`` / ``nativeSide`` to every image variant and tracked seed."""
    for family in families:
        for variant in family["variants"]:
            row = (
                IMAGE_VARIANT_LIMITS.get(str(variant.get("id")))
                or IMAGE_FAMILY_LIMITS.get(str(variant.get("familyId") or family.get("id")))
                or _UNKNOWN_IMAGE
            )
            _write(variant, row)
    for seed in seeds:
        _write(seed, IMAGE_SEED_LIMITS.get(str(seed.get("repo"))) or _UNKNOWN_IMAGE)


def apply_video_size_limits(families: Iterable[dict[str, Any]]) -> None:
    """Add ``maxSide`` / ``maxPixels`` / ``nativeSide`` to every video variant."""
    for family in families:
        for variant in family["variants"]:
            row = (
                VIDEO_VARIANT_LIMITS.get(str(variant.get("id")))
                or VIDEO_FAMILY_LIMITS.get(str(variant.get("familyId") or family.get("id")))
                or _UNKNOWN_VIDEO
            )
            _write(variant, row)
