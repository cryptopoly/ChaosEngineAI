"""Live denoise thumbnail emit (FU-018 part 2).

Decodes the current ``callback_kwargs["latents"]`` tensor through the
TAESD / TAEHV preview VAE that ``maybe_apply_preview_vae`` swapped onto
``pipeline.vae``, scales the result down, base64-encodes a PNG, and
returns the string for ``ProgressTracker.set_thumbnail`` to publish.

Two helpers — one for image pipelines (latents shape ``(B, C, H, W)``)
and one for video pipelines (latents shape ``(B, C, F, H, W)`` —
TAEHV/TAEW reduce on the frame axis already, but for thumbnails we
just pick the middle frame). Both clamp to a max output size (default
192 px on the long edge) to keep base64 payloads cheap on the polled
``/api/{images,video}/progress`` endpoint.

Errors are intentionally swallowed and turned into a ``None`` return —
a thumbnail decode crash should never abort the actual generation. The
caller (``callback_on_step_end``) just clears the slot and the UI
shows the previous frame until the next successful decode.
"""

from __future__ import annotations

import base64
import io
from typing import Any

# Cap thumbnail size so a 1024px gen doesn't push 1.5 MB of PNG through
# the polling endpoint each step. 192 px on the long edge keeps PNGs
# under ~30 KB after compression on typical content.
_MAX_THUMB_SIDE = 192


def _to_pil_from_tensor(image_tensor: Any):
    """Map a torch / mlx tensor (single image, 3xHxW or HxWx3, [-1,1] or
    [0,1]) to a ``PIL.Image``. Returns ``None`` on shape mismatch."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return None

    if image_tensor is None:
        return None

    # Accept torch.Tensor or numpy.ndarray. Detach + cpu + numpy.
    array = image_tensor
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "to"):
        try:
            array = array.to("cpu")
        except Exception:
            pass
    if hasattr(array, "float"):
        try:
            array = array.float()
        except Exception:
            pass
    if hasattr(array, "numpy"):
        try:
            array = array.numpy()
        except Exception:
            return None
    if not hasattr(array, "shape"):
        return None

    # Squeeze to a single image. Common shapes:
    #   (1, 3, H, W) -> (3, H, W)
    #   (3, H, W)
    #   (H, W, 3)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        return None
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        # CHW -> HWC
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        return None

    # Normalise into [0, 255] uint8. Detect [-1, 1] vs [0, 1] from the
    # observed range — taking the min lets us cover both VAE-output
    # conventions without an explicit flag.
    arr_min = float(array.min())
    if arr_min < -0.05:
        array = (array + 1.0) * 0.5
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype("uint8")

    return Image.fromarray(array, mode="RGB")


def _scale_to_max_side(image, max_side: int):
    if image is None:
        return None
    w, h = image.size
    long_side = max(w, h)
    if long_side <= max_side:
        return image
    ratio = max_side / float(long_side)
    target_w = max(1, int(round(w * ratio)))
    target_h = max(1, int(round(h * ratio)))
    return image.resize((target_w, target_h))


def _pil_to_b64_png(image) -> str | None:
    if image is None:
        return None
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _unpack_flux_latents(pipeline: Any, latents: Any) -> Any:
    """Convert FLUX's packed 3D latent ``(B, seq_len, 64)`` back to the
    4D ``(B, 16, H/8, W/8)`` shape ``vae.decode`` expects.

    FLUX packs 2x2 patches of 16-channel latents into a single sequence
    token, so ``seq_len = (H/16) * (W/16)``. We assume square latents
    when reading dimensions — that covers every FLUX preset we ship and
    keeps the helper from poking at private pipeline state for size info.
    """
    try:
        import math
    except Exception:
        return None
    if latents is None or not hasattr(latents, "shape") or len(latents.shape) != 3:
        return None
    seq_len = latents.shape[1]
    side = int(round(math.sqrt(seq_len)))
    if side * side != seq_len:
        return None
    # Pixel dimensions: each token covers a 16x16 pixel patch (FLUX
    # patch_size=2 over a 8x VAE downsample → 16 pixel stride).
    pixel_side = side * 16
    unpack = getattr(pipeline, "_unpack_latents", None)
    if not callable(unpack):
        return None
    try:
        # Most FLUX pipelines expose ``vae_scale_factor`` directly; fall
        # back to 8 (the published default for AutoencoderKL on FLUX).
        vae_scale = int(getattr(pipeline, "vae_scale_factor", 8) or 8)
        return unpack(latents, pixel_side, pixel_side, vae_scale)
    except Exception:
        return None


def decode_image_latent_to_b64(
    pipeline: Any,
    latents: Any,
    *,
    max_side: int = _MAX_THUMB_SIDE,
) -> str | None:
    """Decode an image latent via ``pipeline.vae``, scale down, return
    base64 PNG. Handles both standard 4D ``(B, C, H, W)`` latents
    (SD1.5 / SDXL / SD3) and FLUX's packed 3D ``(B, seq_len, 64)``
    latents — we unpack via ``pipeline._unpack_latents`` before decode.
    Returns ``None`` on any failure."""
    vae = getattr(pipeline, "vae", None)
    if vae is None or latents is None:
        return None
    try:
        import torch
    except ImportError:
        return None

    try:
        # FLUX packed latents need an unpack pass before VAE decode.
        if hasattr(latents, "shape") and len(latents.shape) == 3:
            unpacked = _unpack_flux_latents(pipeline, latents)
            if unpacked is None:
                return None
            latents = unpacked

        with torch.no_grad():
            vae_config = getattr(vae, "config", None)
            scaling = float(getattr(vae_config, "scaling_factor", 1.0) or 1.0)
            shift = float(getattr(vae_config, "shift_factor", 0.0) or 0.0)
            latents_in = latents
            # Most diffusers image pipelines store ``latents * scaling_factor + shift``
            # in the noise space — invert that before VAE decode.
            if scaling != 1.0 or shift != 0.0:
                latents_in = (latents / scaling) + shift if shift else latents / scaling
            decoded = vae.decode(latents_in.to(vae.dtype)).sample
        # Pick first batch element only — single-image preview is enough.
        first = decoded[0:1] if decoded.ndim == 4 else decoded
        image = _to_pil_from_tensor(first)
        image = _scale_to_max_side(image, max_side)
        return _pil_to_b64_png(image)
    except Exception:
        return None


def decode_video_latent_to_b64(
    pipeline: Any,
    latents: Any,
    *,
    max_side: int = _MAX_THUMB_SIDE,
) -> str | None:
    """Decode a 5D video latent ``(B, C, F, H, W)`` via ``pipeline.vae``,
    pick the middle frame, scale down, return base64 PNG. Returns ``None``
    on any failure."""
    vae = getattr(pipeline, "vae", None)
    if vae is None or latents is None:
        return None
    try:
        import torch
    except ImportError:
        return None

    try:
        with torch.no_grad():
            scaling = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0) or 1.0)
            latents_in = latents
            if scaling != 1.0:
                latents_in = latents / scaling
            decoded = vae.decode(latents_in.to(vae.dtype)).sample
        # Video VAE returns ``(B, C, F, H, W)``. Pick the middle frame.
        if decoded.ndim == 5:
            frame_count = decoded.shape[2]
            mid = frame_count // 2
            frame = decoded[0, :, mid, :, :]
        elif decoded.ndim == 4:
            frame = decoded[0]
        else:
            return None
        image = _to_pil_from_tensor(frame)
        image = _scale_to_max_side(image, max_side)
        return _pil_to_b64_png(image)
    except Exception:
        return None
