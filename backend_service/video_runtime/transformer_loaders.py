"""Quantised video transformer loaders + device probes.

Five stateless helpers lifted out of ``DiffusersVideoEngine``:

* ``try_load_gguf_transformer`` — single-file ``.gguf`` for video DiTs
  (Wan, HunyuanVideo, LTX-Video) via diffusers ``GGUFQuantizationConfig``.
* ``try_load_bnb_nf4_transformer`` — bitsandbytes NF4 4-bit (CUDA only).
* ``swap_distill_transformers`` — FU-019 lightx2v 4-step distill swap
  for Wan 2.2 A14B I2V (high + low noise experts).
* ``detect_device`` — torch device probe (CUDA → MPS → CPU) with the
  Windows-CUDA-missing diagnostic.
* ``preferred_torch_dtype`` — bf16 / fp16 / fp32 picker per device with
  the M1-MPS bf16 capability probe + ``CHAOSENGINE_VIDEO_MPS_BF16=0``
  env opt-out.

Extracted from ``backend_service/video_runtime/__init__.py`` as part
of the v0.8.0 refactor. ``DiffusersVideoEngine`` thin-wraps each one
so test surface + call sites are unchanged.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

from backend_service.video_runtime.device import _windows_cuda_unavailable_message
from backend_service.video_runtime.repos import (
    _bnb_nf4_transformer_class_for_repo,
    _gguf_video_transformer_class_for_repo,
)


def try_load_gguf_transformer(
    repo: str,
    gguf_repo: str,
    gguf_file: str,
    torch: Any,
) -> tuple[Any, str | None]:
    """Load a video DiT from a single ``.gguf`` file via diffusers.

    Mirrors the image-side loader: GGUF weights cover the DiT only;
    VAE and text encoders are loaded from the base ``repo`` snapshot.
    The helper itself only reports ``(None, note)`` on failure so tests
    can exercise each missing-dependency path. ``_ensure_pipeline``
    treats a requested GGUF variant as strict and raises with that note
    rather than silently loading the full fp16 / bf16 transformer.
    """
    if importlib.util.find_spec("gguf") is None:
        return None, (
            "gguf package missing — install it from the Setup page to "
            f"load {gguf_file}. Falling back to the standard transformer."
        )
    try:
        from diffusers import GGUFQuantizationConfig  # type: ignore
    except Exception as exc:
        return None, (
            f"Installed diffusers cannot load GGUFQuantizationConfig "
            f"({type(exc).__name__}: {exc}). Upgrade diffusers via the "
            "Setup page to use GGUF variants."
        )
    transformer_cls_name = _gguf_video_transformer_class_for_repo(repo)
    if transformer_cls_name is None:
        return None, (
            f"No GGUF transformer class registered for {repo}. "
            "Add it to _GGUF_VIDEO_TRANSFORMER_CLASSES."
        )
    try:
        import diffusers  # type: ignore
    except Exception:
        return None, "diffusers import failed — cannot load GGUF transformer."
    transformer_cls = getattr(diffusers, transformer_cls_name, None)
    if transformer_cls is None:
        return None, (
            f"{transformer_cls_name} not in installed diffusers — "
            "upgrade to use this GGUF variant."
        )

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        gguf_local_path = hf_hub_download(
            repo_id=gguf_repo,
            filename=gguf_file,
            local_files_only=True,
        )
        # ``from_single_file`` defaults the architecture config to the
        # transformer class's largest known variant. For Wan that is the
        # 14 B / A14B layout (cross-attn dim 5120). The TI2V 5B uses
        # cross-attn dim 3072, so loading its GGUF without an explicit
        # config raises:
        #     blocks.0.attn2.to_k.bias expected torch.Size([5120]),
        #     but got torch.Size([3072])
        # Pointing at the base diffusers repo's transformer subfolder
        # makes diffusers build the model from the matching
        # ``transformer/config.json`` before mapping in GGUF tensors,
        # which fixes Wan 2.2 5B and stays correct for every other
        # variant (the config dim happens to match the GGUF anyway).
        transformer = transformer_cls.from_single_file(
            gguf_local_path,
            quantization_config=GGUFQuantizationConfig(
                compute_dtype=torch.bfloat16,
            ),
            torch_dtype=torch.bfloat16,
            config=repo,
            subfolder="transformer",
        )
        return transformer, f"Transformer loaded from GGUF ({gguf_file})"
    except Exception as exc:  # noqa: BLE001 — any failure → fall back
        return None, (
            f"GGUF load failed ({type(exc).__name__}: {exc}) — "
            "falling back to the standard transformer."
        )


def try_load_bnb_nf4_transformer(
    repo: str,
    local_path: str,
    torch: Any,
    device: str,
) -> tuple[Any, str | None]:
    """Load a video DiT in NF4 4-bit via bitsandbytes.

    CUDA-only — bitsandbytes has no Metal/MPS backend, and the kernels
    wouldn't help on a 64 GB Mac anyway. Failure modes (non-CUDA host,
    missing bitsandbytes, old diffusers without ``BitsAndBytesConfig``,
    unmapped repo, broken snapshot subfolder) all return ``(None,
    note)`` so the caller falls back to the standard fp16 / bf16
    transformer.
    """
    if device != "cuda":
        return None, (
            "NF4 (bitsandbytes) requires CUDA. "
            "Falling back to the standard transformer."
        )
    if importlib.util.find_spec("bitsandbytes") is None:
        return None, (
            "bitsandbytes package missing — install it from the Setup "
            "page to enable NF4. Falling back to the standard transformer."
        )
    try:
        from diffusers import BitsAndBytesConfig  # type: ignore
    except ImportError:
        return None, (
            "Installed diffusers doesn't expose BitsAndBytesConfig. "
            "Upgrade diffusers via the Setup page to use NF4 variants."
        )
    transformer_cls_name = _bnb_nf4_transformer_class_for_repo(repo)
    if transformer_cls_name is None:
        return None, (
            f"No NF4 transformer class registered for {repo}. "
            "Add it to _BNB_NF4_VIDEO_TRANSFORMER_CLASSES."
        )
    try:
        import diffusers  # type: ignore
    except Exception:
        return None, "diffusers import failed — cannot load NF4 transformer."
    transformer_cls = getattr(diffusers, transformer_cls_name, None)
    if transformer_cls is None:
        return None, (
            f"{transformer_cls_name} not in installed diffusers — "
            "upgrade to use NF4 quantization."
        )

    try:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = transformer_cls.from_pretrained(
            local_path,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        return transformer, "Transformer loaded with NF4 (bitsandbytes)"
    except Exception as exc:  # noqa: BLE001 — any failure → fall back
        return None, (
            f"NF4 load failed ({type(exc).__name__}: {exc}) — "
            "falling back to the standard transformer."
        )


def swap_distill_transformers(
    pipeline: Any,
    *,
    repo: str,
    high_file: str,
    low_file: str,
    precision: str,
    torch: Any,
) -> str:
    """Swap ``pipeline.transformer`` + ``pipeline.transformer_2`` for
    the lightx2v 4-step distilled experts (Wan 2.2 A14B I2V).

    Wan 2.2 A14B is MoE: ``transformer`` is the high-noise expert and
    ``transformer_2`` is the low-noise expert. Distillation publishes
    both as standalone safetensors files; the swap is the load-bearing
    substitution that takes the pipeline from 30-step base to 4-step
    distilled. Returns a runtimeNote describing what happened. Failure
    is non-fatal — the stock transformers stay in place and the user
    sees the failure in the note.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        return (
            f"Distill swap skipped: huggingface_hub unavailable ({exc}). "
            "Pipeline continuing with stock Wan transformers."
        )

    try:
        from diffusers import WanTransformer3DModel
    except ImportError as exc:
        return (
            f"Distill swap skipped: WanTransformer3DModel unavailable "
            f"({exc}). Pipeline continuing with stock Wan transformers."
        )

    # FP8/INT8 distill weights ship pre-quantized; they need a torch
    # backend that exposes the matching kernels (CUDA SM 8.9+ for FP8,
    # CUDA / Metal for INT8). On platforms without those kernels we
    # load as bf16 and let diffusers do the dequant — quality holds
    # but the memory savings disappear. ``bf16`` (no quantization)
    # always loads at native precision.
    torch_dtype = torch.bfloat16
    if precision == "fp8_e4m3":
        torch_dtype = getattr(torch, "float8_e4m3fn", torch.bfloat16)

    try:
        high_local = hf_hub_download(
            repo_id=repo, filename=high_file, local_files_only=False
        )
        low_local = hf_hub_download(
            repo_id=repo, filename=low_file, local_files_only=False
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal
        return (
            f"Distill download failed ({type(exc).__name__}: {exc}). "
            "Pipeline continuing with stock Wan transformers."
        )

    try:
        high_transformer = WanTransformer3DModel.from_single_file(
            high_local, torch_dtype=torch_dtype
        )
        low_transformer = WanTransformer3DModel.from_single_file(
            low_local, torch_dtype=torch_dtype
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal
        return (
            f"Distill load failed ({type(exc).__name__}: {exc}). "
            "Pipeline continuing with stock Wan transformers."
        )

    if not hasattr(pipeline, "transformer"):
        return (
            "Distill swap skipped: pipeline has no .transformer attribute. "
            "This Wan distill path requires a WanPipeline-shaped object."
        )

    pipeline.transformer = high_transformer
    if hasattr(pipeline, "transformer_2"):
        pipeline.transformer_2 = low_transformer
    else:
        return (
            f"Distill: high-noise expert applied, but pipeline lacks "
            f"transformer_2 (low-noise expert). Verify base repo {repo} "
            "is the A14B MoE pipeline. Quality may be degraded."
        )

    return (
        f"Distill: swapped transformer + transformer_2 from {repo} "
        f"(precision={precision}, 4-step schedule)."
    )


def detect_device(torch: Any) -> str:
    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is not None:
        try:
            if getattr(cuda_module, "is_available", lambda: False)():
                return "cuda"
        except Exception:
            pass
    cuda_error = _windows_cuda_unavailable_message(torch)
    if cuda_error:
        raise RuntimeError(cuda_error)
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
        return "mps"
    return "cpu"


def preferred_torch_dtype(torch: Any, device: str) -> Any:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        # M2 and newer support bf16 on MPS; M1 silently downcasts to
        # fp16 inside operators which costs accuracy on long DiT
        # sequences. Probe the capability with a one-element tensor —
        # if MPS rejects it, fall back to fp16 cleanly. Honour an env
        # opt-out so we have a rollback lever if a future MPS update
        # regresses.
        if os.getenv("CHAOSENGINE_VIDEO_MPS_BF16") == "0":
            return torch.float16
        try:
            probe = torch.zeros(1, dtype=torch.bfloat16, device="mps")
            del probe
            return torch.bfloat16
        except (RuntimeError, NotImplementedError, TypeError):
            return torch.float16
    return torch.float32
