"""Quantised transformer loaders + device probes for the image engine.

Eight stateless helpers lifted out of ``DiffusersTextToImageEngine``:

* ``try_load_nf4_flux_transformer`` — bitsandbytes NF4 (CUDA, FLUX).
* ``try_load_int8wo_flux_transformer`` — TorchAO int8 weight-only (MPS,
  FLUX).
* ``try_load_gguf_transformer`` — single-file ``.gguf`` via diffusers'
  ``GGUFQuantizationConfig`` (cross-platform; Q2_K … Q8_0).
* ``try_load_nunchaku_transformer`` — FU-023 SVDQuant int4 (CUDA;
  FLUX / Qwen-Image / SD3 / SANA / PixArt-Σ).
* ``maybe_enable_fp8_layerwise`` — FU-024 ``enable_layerwise_casting``
  with family-correct fp8 dtype (E4M3 / E5M2) gated on SM ≥ 8.9.
* ``should_use_model_cpu_offload`` — predicate for FLUX-on-CUDA's
  whole-component swap path.
* ``detect_device`` — torch device probe (CUDA → MPS → CPU) with the
  Windows-CUDA-missing diagnostic.

Extracted from ``backend_service/image_runtime/__init__.py`` as part of
the v0.8.0 refactor. ``DiffusersTextToImageEngine`` thin-wraps each one
so test surface + call sites are unchanged.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from backend_service.image_runtime.device import _windows_cuda_unavailable_message
from backend_service.image_runtime.repos import (
    _gguf_transformer_class_for_repo,
    _is_flux_repo,
    _is_sdxl_repo,
    _nunchaku_transformer_class_for_repo,
)


def try_load_nf4_flux_transformer(
    local_path: str, torch: Any,
) -> tuple[Any, str | None]:
    """Load FLUX's transformer quantized to NF4 via bitsandbytes.

    NF4 (4-bit NormalFloat) drops the 12B FLUX transformer from ~24 GB
    (bf16) to ~7 GB with negligible visual quality loss — the exact
    pattern the FLUX community runs on 24 GB consumer GPUs. T5-XXL and
    the VAE are NOT quantized (they're small enough, and quantizing
    text encoders hurts prompt adherence more than it saves memory).

    Returns ``(transformer, note)``. A ``None`` transformer means the
    caller should fall back to the unquantized pipeline — typically
    because bitsandbytes isn't installed yet or the diffusers version
    predates the ``quantization_config`` plumbing. The note is a user-
    visible progress message explaining which path was taken.
    """
    if importlib.util.find_spec("bitsandbytes") is None:
        return None, (
            "bitsandbytes missing — FLUX will load in bf16. "
            "Install it from the Setup page to enable NF4 quantization "
            "(turns 8 min/step into ~10 s/step on a 24 GB GPU)."
        )
    try:
        from diffusers import BitsAndBytesConfig, FluxTransformer2DModel  # type: ignore
    except ImportError:
        return None, (
            "Installed diffusers doesn't expose BitsAndBytesConfig. "
            "Upgrade via the Setup page to use NF4 FLUX."
        )

    try:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            local_path,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        return transformer, "FLUX transformer loaded in NF4 (~7 GB VRAM)"
    except Exception as exc:  # noqa: BLE001 — any failure → fall back to bf16
        return None, (
            f"NF4 quantization failed ({type(exc).__name__}: {exc}) — "
            "falling back to bf16 transformer (slower on <32 GB GPUs)."
        )


def try_load_int8wo_flux_transformer(
    local_path: str, torch: Any,
) -> tuple[Any, str | None]:
    """Load FLUX's transformer with TorchAO int8 weight-only quant.

    int8wo is the Apple-Silicon counterpart to bitsandbytes NF4:
    bitsandbytes ships CUDA kernels only, so an MPS FLUX run would
    otherwise need 24 GB bf16 weights and pagefile-thrash on any
    Mac under 48 GB. int8wo drops that to ~12 GB — not as tight as
    NF4's ~7 GB but wide enough for 32 GB M-series machines.

    Returns ``(transformer, note)`` with the same contract as the
    NF4 helper: ``None`` transformer means the caller should fall
    back, note is a human-readable progress message.
    """
    if importlib.util.find_spec("torchao") is None:
        return None, (
            "torchao missing — FLUX will load in bf16 on MPS. "
            "Install it from the Setup page to enable int8 "
            "quantization (~24 GB → ~12 GB)."
        )
    try:
        from diffusers import FluxTransformer2DModel, TorchAoConfig  # type: ignore
    except ImportError:
        return None, (
            "Installed diffusers doesn't expose TorchAoConfig. "
            "Upgrade via the Setup page to use int8wo FLUX."
        )
    try:
        transformer = FluxTransformer2DModel.from_pretrained(
            local_path,
            subfolder="transformer",
            quantization_config=TorchAoConfig("int8wo"),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        return transformer, "FLUX transformer loaded in int8wo (~12 GB)"
    except Exception as exc:  # noqa: BLE001 — fall back to bf16
        return None, (
            f"int8wo quantization failed ({type(exc).__name__}: {exc}) — "
            "falling back to bf16."
        )


def try_load_gguf_transformer(
    repo: str,
    gguf_repo: str,
    gguf_file: str,
    torch: Any,
) -> tuple[Any, str | None]:
    """Load a transformer from a single ``.gguf`` file via diffusers.

    GGUF wins over NF4 for two reasons: it works on Apple Silicon / CPU
    (bitsandbytes is CUDA-only), and the community ships a spread of
    quant levels (Q2_K … Q8_0) so the user can trade quality for VRAM
    at a finer granularity than NF4's single 4-bit point.

    The VAE and text encoders still come from the base ``repo``
    snapshot — GGUF files only carry the transformer/DiT weights.

    Returns ``(transformer, note)``. A ``None`` transformer means the
    caller should fall back (NF4 or bf16). Any failure here is
    non-fatal: missing ``gguf`` pip package, an old diffusers without
    ``GGUFQuantizationConfig``, or an HF cache miss for the chosen
    quant file will all route to the standard pipeline.
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

    transformer_cls_name = _gguf_transformer_class_for_repo(repo)
    if transformer_cls_name is None:
        return None, (
            f"No GGUF transformer class registered for {repo}. "
            "Add a mapping in image_runtime._gguf_transformer_class_for_repo."
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
        # Pin the architecture config to the base repo's
        # ``transformer/config.json`` — without this hint
        # ``from_single_file`` falls back to the transformer class's
        # default layout, which is fine for the largest variant in a
        # family but breaks smaller variants (different cross-attn
        # dim, hidden size, layer count). Mirrors the video-side
        # loader. See ``backend_service/video_runtime.py``'s
        # ``_try_load_gguf_transformer`` for the Wan 2.2 5B repro
        # that motivated the fix.
        transformer = transformer_cls.from_single_file(
            gguf_local_path,
            quantization_config=GGUFQuantizationConfig(
                compute_dtype=torch.bfloat16,
            ),
            torch_dtype=torch.bfloat16,
            config=repo,
            subfolder="transformer",
        )
        return transformer, (
            f"Transformer loaded from GGUF ({gguf_file})"
        )
    except Exception as exc:  # noqa: BLE001 — any failure → fall back
        return None, (
            f"GGUF load failed ({type(exc).__name__}: {exc}) — "
            "falling back to the standard transformer."
        )


def try_load_nunchaku_transformer(
    repo: str,
    nunchaku_repo: str,
    nunchaku_file: str | None,
    torch: Any,
) -> tuple[Any, str | None]:
    """FU-023: load a Nunchaku SVDQuant transformer for FLUX / Qwen-Image
    / SD3.5 / SANA / PixArt-Σ. CUDA only.

    Nunchaku ships dedicated transformer subclasses
    (``NunchakuFluxTransformer2dModel``, ``NunchakuQwenImageTransformer2DModel``,
    etc.) that load precompiled INT4 SVDQuant weights and expose the
    same forward signature as the stock diffusers transformer, so the
    rest of ``_ensure_pipeline`` keeps working without further
    plumbing. ~3× perf over NF4 on FLUX.1-dev.

    Returns ``(transformer, note)`` matching the NF4 / GGUF helper
    contract — ``None`` transformer means the caller should fall back.
    """
    if importlib.util.find_spec("nunchaku") is None:
        return None, (
            "Nunchaku package not installed — install it from the Setup "
            "page to enable SVDQuant 4-bit on CUDA. Falling back to "
            "the standard transformer."
        )
    cls_name = _nunchaku_transformer_class_for_repo(repo)
    if cls_name is None:
        return None, (
            f"No Nunchaku transformer class registered for {repo}. "
            "Add a mapping in image_runtime._nunchaku_transformer_class_for_repo."
        )
    try:
        import nunchaku  # type: ignore
    except ImportError as exc:
        return None, (
            f"Nunchaku import failed ({exc}). Install nunchaku>=1.2.1 "
            "from the Setup page."
        )
    cls = getattr(nunchaku, cls_name, None)
    if cls is None:
        return None, (
            f"{cls_name} not in installed nunchaku — upgrade via the "
            "Setup page to use this Nunchaku variant."
        )

    try:
        from huggingface_hub import snapshot_download  # type: ignore
        local_dir = snapshot_download(
            repo_id=nunchaku_repo,
            local_files_only=True,
        )
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if nunchaku_file:
            kwargs["filename"] = nunchaku_file
        transformer = cls.from_pretrained(local_dir, **kwargs)
        note = (
            f"Nunchaku SVDQuant transformer loaded from {nunchaku_repo}"
            + (f"/{nunchaku_file}" if nunchaku_file else "")
            + " (CUDA INT4 — ~3× over NF4)."
        )
        return transformer, note
    except Exception as exc:  # noqa: BLE001 — fall through to NF4
        return None, (
            f"Nunchaku load failed ({type(exc).__name__}: {exc}) — "
            "falling back to NF4 / int8wo / bf16."
        )


def maybe_enable_fp8_layerwise(
    pipeline: Any,
    repo: str,
    torch: Any,
) -> str | None:
    """FU-024: call ``transformer.enable_layerwise_casting`` with the
    family-correct fp8 dtype. Caller has already gated to CUDA. Pre-Ada
    GPUs lack hardware fp8 support — the cast still runs but generation
    is slower than bf16, so we additionally check the compute capability
    (SM 8.9 = Ada Lovelace, SM 9.0 = Hopper, SM 10.0 = Blackwell).
    Returns a runtimeNote string, or ``None`` when the path no-ops
    cleanly.
    """
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return "FP8 layerwise skipped: torch.cuda.get_device_capability failed."
    if (major, minor) < (8, 9):
        return (
            f"FP8 layerwise skipped: SM {major}.{minor} pre-dates Ada — "
            "hardware fp8 unavailable. Use bf16 / NF4 / Nunchaku instead."
        )
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or not hasattr(transformer, "enable_layerwise_casting"):
        return (
            "FP8 layerwise skipped: pipeline.transformer.enable_layerwise_casting "
            "missing — pipeline is UNet-based or the diffusers version is old."
        )
    repo_lower = repo.lower()
    if "hunyuan" in repo_lower:
        storage_dtype = torch.float8_e5m2
        storage_label = "E5M2"
    else:
        storage_dtype = torch.float8_e4m3fn
        storage_label = "E4M3"
    try:
        transformer.enable_layerwise_casting(
            storage_dtype=storage_dtype,
            compute_dtype=torch.bfloat16,
        )
    except Exception as exc:
        return (
            f"FP8 layerwise enable failed ({type(exc).__name__}: {exc}) — "
            "running bf16."
        )
    return f"FP8 layerwise casting enabled ({storage_label}, compute=bf16)."


def should_use_model_cpu_offload(repo: str, device: str) -> bool:
    """True when the pipeline should load via enable_model_cpu_offload().

    Currently limited to FLUX on CUDA. FLUX.1-Dev is ~24 GB transformer
    plus ~9 GB T5-XXL text encoder in bf16; on any single consumer GPU
    (≤32 GB VRAM) a plain ``pipeline.to("cuda")`` either OOMs or, worse
    on Windows, silently falls back to pinned host memory + pagefile
    and runs at PCIe speeds. Other pipelines (SD 1.5 / SDXL /
    Qwen-Image) fit comfortably and stay on the legacy .to(device)
    path for best throughput.
    """
    if device != "cuda":
        return False
    return _is_flux_repo(repo)


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


def preferred_torch_dtype(
    torch: Any,
    repo: str,
    device: str,
    sdxl_vae_fix_available: bool = False,
) -> Any:
    """Pick the best dtype for a (repo, device) combination.

    CUDA: bfloat16 for FLUX (matches the upstream Black Forest recipe);
    fp16 elsewhere. MPS: stock SDXL needs fp32 to avoid the
    sigmoid-overflow black-image bug, but FU-017's
    ``madebyollin/sdxl-vae-fp16-fix`` snapshot lets us stay on fp16
    when cached. CPU: fp32 for all repos.
    """
    if device == "cuda":
        if _is_flux_repo(repo):
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        lowered_repo = repo.lower()
        if any(token in lowered_repo for token in ("stable-diffusion", "sdxl", "sd_xl")):
            if sdxl_vae_fix_available and _is_sdxl_repo(repo):
                return torch.float16
            return torch.float32
        return torch.float16
    return torch.float32


def preferred_execution_device(repo: str, detected_device: str) -> str:
    """Override the device probe for repos where the auto-detected
    device is known to mis-render. Today: Qwen-Image on MPS produces
    black outputs in fp16, so we route to CPU instead of silently
    returning placeholder frames.
    """
    lowered_repo = repo.lower()
    if detected_device == "mps" and "qwen-image" in lowered_repo:
        return "cpu"
    return detected_device
