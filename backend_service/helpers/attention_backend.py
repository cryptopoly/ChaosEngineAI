"""Attention-backend selection for diffusers DiT pipelines.

FU-016. Diffusers 0.36+ exposes ``transformer.set_attention_backend(...)``
for picking between PyTorch SDPA, FlashAttention 2/3, xformers and
SageAttention. SageAttention 2/2++ (thu-ml) is an INT8 (Ampere+) /
FP8 (Hopper) attention kernel that drops attention wall time 2-3× and
end-to-end DiT latency 1.3-1.6× on FLUX/Wan/Hunyuan/CogVideoX with no
documented quality regression.

Platform gate:
- CUDA only (no MPS / Metal port as of May 2026).
- Requires the ``sageattention`` pip wheel (``pip install sageattention``)
  AND a diffusers ≥0.36 build that exposes ``set_attention_backend``.
- Skipped silently on macOS / CPU / unsupported pipelines so the call
  site can stay platform-neutral.

Stacks multiplicatively with First Block Cache (FU-015) — community
benchmarks (Wan2.1 720P I2V) report cumulative ~54% wall-time reduction
when SageAttention + FBCache are combined.

Reference: https://github.com/thu-ml/SageAttention
"""

from __future__ import annotations

import importlib.util
from typing import Any


def maybe_apply_sage_attention(pipeline: Any) -> str | None:
    """Switch ``pipeline.transformer`` to the SageAttention backend if available.

    Returns a short note for the per-image / per-video runtimeNote slot
    (e.g. ``"Attention: SageAttention"``) when the swap succeeded, or
    ``None`` when the backend isn't available, the device isn't CUDA,
    or the pipeline shape doesn't expose ``set_attention_backend``.

    Failure modes (import error, kernel mismatch on a non-SM80+ GPU,
    incompatible diffusers version) all return ``None`` so the caller
    can keep the stock SDPA path. The only thing that propagates is a
    bug in this helper itself.
    """
    # 1. CUDA gate. SageAttention has no MPS / Metal port; calling
    #    ``set_attention_backend("sage")`` on a non-CUDA pipeline raises.
    try:
        import torch  # type: ignore
    except Exception:
        return None
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    if not cuda_available:
        return None

    # 2. SageAttention package gate. Importable means the pip wheel
    #    matched the user's CUDA + Python combo at install time.
    if importlib.util.find_spec("sageattention") is None:
        return None

    # 3. Pipeline shape gate. Must be a DiT pipeline with a transformer
    #    that exposes the diffusers ≥0.36 attention-backend selector.
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        return None
    set_backend = getattr(transformer, "set_attention_backend", None)
    if not callable(set_backend):
        return None

    try:
        set_backend("sage")
    except Exception as exc:  # noqa: BLE001 — keep stock SDPA on any failure
        return f"SageAttention unavailable ({type(exc).__name__})"

    return "Attention: SageAttention"
