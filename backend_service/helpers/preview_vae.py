"""TAESD / TAEHV preview-decode VAE swap (FU-018).

Tiny VAE for cheap decode each step. Preview-only by default — caller
toggles via the ``previewVae`` knob on the generation request. The full
generate path uses the swapped-in VAE so the user trades final fidelity
for wall-time. Real-time UI thumbnails would use this same swap with the
per-step callback hook (planned).

Per-family mapping (longest prefix wins):

- FLUX.1 family            → ``madebyollin/taef1``
- FLUX.2 family            → ``madebyollin/taef2``
- SD3 / SD3.5              → ``madebyollin/taesd3``
- SDXL                     → ``madebyollin/taesdxl``
- SD 1.x / 2.x             → ``madebyollin/taesd``
- Wan2.1 / Wan2.2 (any)    → ``madebyollin/taew2_2``
- LTX-Video / LTX-2 family → ``madebyollin/taeltx2_3_wide``
- HunyuanVideo             → ``madebyollin/taehv1_5``
- Qwen-Image family        → ``madebyollin/taeqwenimage``
- CogVideoX                → ``madebyollin/taecogvideox``
- Mochi                    → ``madebyollin/taemochi``

The helper tries ``AutoencoderTiny.from_pretrained(..., local_files_only=True)``
first, then falls back to a remote fetch. Anything that isn't cached and
isn't reachable is treated as a no-op with a runtimeNote so the caller
can show the user why the swap didn't apply.
"""

from __future__ import annotations

import importlib.util
from typing import Any


# Repo-prefix → preview VAE HF id. Order matters: longer / more-specific
# prefixes first so FLUX.2 doesn't trigger the FLUX.1 default.
_PREVIEW_VAE_MAP: list[tuple[str, str]] = [
    ("black-forest-labs/FLUX.2", "madebyollin/taef2"),
    ("black-forest-labs/FLUX.1", "madebyollin/taef1"),
    ("fal/FLUX.2", "madebyollin/taef2"),
    ("stabilityai/stable-diffusion-3", "madebyollin/taesd3"),
    ("stabilityai/stable-diffusion-xl", "madebyollin/taesdxl"),
    ("stabilityai/stable-diffusion-2", "madebyollin/taesd"),
    ("stabilityai/stable-diffusion-v1", "madebyollin/taesd"),
    ("runwayml/stable-diffusion-v1", "madebyollin/taesd"),
    ("Wan-AI/Wan2", "madebyollin/taew2_2"),
    ("QuantStack/Wan2", "madebyollin/taew2_2"),
    ("Lightricks/LTX-Video", "madebyollin/taeltx2_3_wide"),
    ("prince-canuma/LTX-2", "madebyollin/taeltx2_3_wide"),
    ("hunyuanvideo-community/HunyuanVideo", "madebyollin/taehv1_5"),
    ("tencent/HunyuanVideo", "madebyollin/taehv1_5"),
    ("THUDM/CogVideoX", "madebyollin/taecogvideox"),
    ("genmo/mochi", "madebyollin/taemochi"),
    ("Qwen/Qwen-Image", "madebyollin/taeqwenimage"),
]


def resolve_preview_vae_id(repo: str) -> str | None:
    """Map a base repo id to a preview VAE HF id, or ``None`` if unmapped."""
    for prefix, vae_id in _PREVIEW_VAE_MAP:
        if repo.startswith(prefix):
            return vae_id
    return None


def maybe_apply_preview_vae(
    pipeline: Any,
    *,
    repo: str,
    enabled: bool,
) -> str | None:
    """Swap ``pipeline.vae`` for the matching TAESD / TAEHV preview decoder.

    Returns a runtimeNote string when the swap applied (or attempted-but-failed
    visibly), or ``None`` when the toggle is off, no preview VAE is mapped
    for the repo, or diffusers itself is missing. Failures are non-fatal —
    caller continues with the stock VAE.
    """
    if not enabled:
        return None
    if importlib.util.find_spec("diffusers") is None:
        return None

    preview_id = resolve_preview_vae_id(repo)
    if preview_id is None:
        return None

    target_vae = getattr(pipeline, "vae", None)
    if target_vae is None:
        return "Preview VAE skipped: pipeline has no .vae attribute."

    target_dtype = getattr(target_vae, "dtype", None)

    try:
        from diffusers import AutoencoderTiny
    except ImportError as exc:
        return f"Preview VAE skipped: AutoencoderTiny unavailable ({exc})."

    kwargs: dict[str, Any] = {}
    if target_dtype is not None:
        kwargs["torch_dtype"] = target_dtype

    # Try the local cache first so offline use keeps working when the
    # preview VAE hasn't been downloaded yet. If it's not cached, fall
    # through to a remote attempt — preview VAEs are small (~5-30 MB)
    # so the download cost is negligible.
    preview_vae = None
    try:
        preview_vae = AutoencoderTiny.from_pretrained(
            preview_id, local_files_only=True, **kwargs
        )
    except Exception:
        try:
            preview_vae = AutoencoderTiny.from_pretrained(preview_id, **kwargs)
        except Exception as exc:
            return (
                f"Preview VAE {preview_id} not cached and download failed "
                f"({type(exc).__name__}: {exc}). Using stock VAE."
            )

    pipeline.vae = preview_vae
    return f"Preview VAE: {preview_id} (fast decode)."
