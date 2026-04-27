"""Video runtime for ChaosEngineAI.

Mirrors the shape of ``image_runtime.py`` so the frontend's runtime-status
contract is identical. This phase ships:

- Dependency probe (reports torch / diffusers availability, detected device,
  and any missing packages — including the mp4 encoders needed later for
  ``generate()``).
- Preload / unload lifecycle for one active pipeline at a time.
- Registry routing for the four first-wave engines (LTX-Video, Mochi 1,
  Wan 2.2, HunyuanVideo) to the right diffusers pipeline class.

Generation is intentionally not implemented yet — the preload-to-generate
phase lands next. This keeps the surface area small and testable while
the UX wiring stabilises.
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import logging
import os
import platform
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from backend_service.helpers.gpu import nvidia_gpu_present
from backend_service.image_runtime import validate_local_diffusers_snapshot
from cache_compression import apply_diffusion_cache_strategy
from backend_service.progress import (
    GenerationCancelled,
    PHASE_DECODING,
    PHASE_DIFFUSING,
    PHASE_ENCODING,
    PHASE_LOADING,
    PHASE_SAVING,
    VIDEO_PROGRESS,
)


_LOG = logging.getLogger(__name__)


MAX_VIDEO_SEED = 2147483647


def _resolve_video_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return secrets.randbelow(MAX_VIDEO_SEED + 1)


# ---------------------------------------------------------------------------
# Torch warmup
# ---------------------------------------------------------------------------
# Importing torch for the first time is expensive (30-60s on a cold Windows
# SSD). Because probe() is a sync FastAPI route that calls ``import torch``,
# the first probe blew past the frontend's 30s fetch timeout and surfaced as
# "Video runtime did not respond" with every downstream endpoint cascading to
# "Failed to fetch". We warm torch on a background thread at sidecar startup
# so probe() can return a fast "initializing" status while the import is in
# flight, and an accurate status the moment it completes. The import lock
# means any in-flight probe still ends up serialized behind the warmup
# anyway — the fast-path here is purely to keep the probe route itself from
# blocking so the rest of the video API stays responsive.

_torch_warmup_lock = threading.Lock()
_torch_warmup_state: dict[str, Any] = {
    "status": "not_started",  # "not_started" | "in_progress" | "ready" | "failed"
    "error": None,  # exception message when status == "failed"
    "started_at": None,
}


def _torch_warmup_worker() -> None:
    try:
        import torch  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - import failure path
        with _torch_warmup_lock:
            _torch_warmup_state["status"] = "failed"
            _torch_warmup_state["error"] = f"{type(exc).__name__}: {exc}"
        return
    # Pre-warm anything else the first probe() call would otherwise pay for
    # inline. On Windows the nvidia-smi shell-out adds 1-2s per probe when
    # uncached, and importlib.util.find_spec on a cold NTFS volume with
    # antivirus scanning can be slow enough to push a probe past the
    # frontend's fetch timeout. Doing both here keeps probe() a hashmap
    # lookup in the common case.
    try:
        from backend_service.helpers.gpu import get_device_vram_total_gb
        get_device_vram_total_gb()
    except Exception:
        pass
    try:
        for _pkg, module_name in _CORE_DEPS + _VIDEO_OUTPUT_DEPS + _VIDEO_MODEL_DEPS:
            try:
                importlib.util.find_spec(module_name)
            except Exception:
                pass
    except Exception:
        pass
    with _torch_warmup_lock:
        _torch_warmup_state["status"] = "ready"
        _torch_warmup_state["error"] = None


def start_torch_warmup() -> None:
    """Kick off a one-shot background import of torch.

    Called from ``create_app()`` at sidecar startup. Safe to call repeatedly —
    only the first call spawns a thread. If torch is already importable
    cheaply (e.g. the interpreter has seen it before in this process), the
    worker finishes almost immediately.
    """
    with _torch_warmup_lock:
        if _torch_warmup_state["status"] != "not_started":
            return
        _torch_warmup_state["status"] = "in_progress"
        _torch_warmup_state["started_at"] = time.monotonic()
    thread = threading.Thread(
        target=_torch_warmup_worker,
        name="chaosengine-torch-warmup",
        daemon=True,
    )
    thread.start()


def torch_warmup_status() -> dict[str, Any]:
    """Snapshot of the warmup state. Used by ``probe()`` to avoid blocking."""
    with _torch_warmup_lock:
        return dict(_torch_warmup_state)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_video_python() -> str:
    override = os.getenv("CHAOSENGINE_MLX_PYTHON") or os.getenv("CHAOSENGINE_VIDEO_PYTHON")
    if override:
        return override
    candidate = WORKSPACE_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return os.getenv("PYTHON", "python3")


def _detect_device_memory_gb(device: str | None) -> float | None:
    """Best-effort read of how much memory the inference device has access to.

    - ``cuda``: dedicated VRAM from ``nvidia-smi`` (via ``get_gpu_metrics``).
    - ``mps`` / ``cpu`` on macOS: unified memory from ``sysctl hw.memsize``.
    - ``cpu`` on Linux/Windows: system RAM via psutil.

    Returns ``None`` when detection fails — the frontend safety heuristic
    treats ``None`` as "stay conservative" and falls back to its 16 GB-safe
    thresholds rather than risk over-scaling on an unknown device.

    Uses the cached fast path in ``helpers.gpu`` because total VRAM never
    changes for the life of a process. The first call shells out to
    ``nvidia-smi``/``sysctl``; every subsequent call is a dict lookup, which
    keeps the ``/api/video/runtime`` probe well inside the frontend's
    15s fetch budget on Windows.
    """
    try:
        from backend_service.helpers.gpu import get_device_vram_total_gb
    except Exception:
        return None
    try:
        return get_device_vram_total_gb()
    except Exception:
        return None


@dataclass(frozen=True)
class VideoRuntimeStatus:
    activeEngine: str
    realGenerationAvailable: bool
    message: str
    device: str | None = None
    # ``expectedDevice`` is the device we'll ask torch to use on the
    # next Generate click, predicted from nvidia-smi + platform checks
    # WITHOUT importing torch. Lets the Studio show "Device: cuda
    # (expected)" before anything has loaded, so users can confirm GPU
    # will be used before sinking 2+ GB of model download into it.
    # Mirrors ``ImageRuntimeStatus.expectedDevice``.
    expectedDevice: str | None = None
    pythonExecutable: str | None = None
    missingDependencies: list[str] = field(default_factory=list)
    loadedModelRepo: str | None = None
    # Total memory available to the inference device, in GB. Used by the
    # frontend safety heuristic (``assessVideoGenerationSafety``) to scale its
    # attention-budget thresholds — a 64 GB M4 Max should tolerate far more
    # frames than a 16 GB base M2, and a 24 GB RTX 4090 differs again. We
    # source this from ``backend_service.helpers.gpu.get_gpu_metrics`` which
    # already reads Apple Silicon unified memory via sysctl and NVIDIA VRAM
    # via nvidia-smi. ``None`` means we couldn't detect it — the frontend
    # falls back to its MPS-strict defaults in that case.
    deviceMemoryGb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _guess_video_expected_device() -> str | None:
    """Predict the device torch will bind to without importing torch.

    Importing torch in probe() would lock torch/lib/*.dll and block the
    GPU-bundle installer on Windows (same trap the image runtime hit).
    ``find_spec`` + ``nvidia_gpu_present`` are free of that side effect
    and accurate enough for the UI badge.
    """
    if importlib.util.find_spec("torch") is None:
        return None
    if nvidia_gpu_present():
        return "cuda"
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class VideoGenerationConfig:
    """Shape consumed by ``DiffusersVideoEngine.generate``."""
    modelId: str
    modelName: str
    repo: str
    prompt: str
    negativePrompt: str
    width: int
    height: int
    numFrames: int
    fps: int
    guidance: float
    steps: int = 50
    seed: int | None = None
    # GGUF quantization for video DiT transformers. When set, the
    # transformer is loaded from a single .gguf file while the VAE /
    # text encoders still come from the base ``repo`` snapshot. The
    # pipeline cache keys on (repo, ggufFile) so multiple quant levels
    # can coexist without evicting each other.
    ggufRepo: str | None = None
    ggufFile: str | None = None
    # Post-processing frame interpolation. Factor of 1 means disabled;
    # 2 or 4 insert interpolated frames between each generated frame
    # and bump the reported fps by the same factor, producing smoother
    # motion at higher frame rates without generating more DiT frames
    # (which is 10-50x more expensive than interpolation).
    interpolationFactor: int = 1
    # Optional diffusion cache strategy id, e.g. "teacache". Mirrors the
    # image_runtime field — video DiTs benefit even more from timestep
    # caching (Wan2.1 720P 30% faster, HunyuanVideo up to 2.1×). When the
    # strategy has no vendored patch for this pipeline the engine swallows
    # the NotImplementedError and falls back to the stock pipeline — the
    # UI shows the "Scaffold" badge so users know why.
    cacheStrategy: str | None = None
    cacheRelL1Thresh: float | None = None
    # Optional diffusers scheduler override. ``None`` (or ``"auto"``) keeps
    # whatever scheduler the per-model defaults table picks, which in turn
    # falls back to the pipeline's baked-in default. Recognised ids match
    # ``_SCHEDULER_CLASSES`` below — anything else logs a warning and
    # leaves the pipeline scheduler untouched.
    scheduler: str | None = None
    # bitsandbytes NF4 quantization for the video DiT transformer. CUDA
    # only; ignored on MPS / CPU. Brings Wan 2.1 14B from ~28 GB bf16 to
    # ~7 GB on the RTX 4090 with negligible quality loss for video DiTs
    # (NF4 is the same scheme bitsandbytes ships for QLoRA).
    useNf4: bool = False
    # LTX-Video two-stage spatial upscale. When True and the pipeline is
    # ``LTXPipeline``, the engine runs the base sampler at the requested
    # resolution then refines through ``LTXLatentUpsamplePipeline``
    # (Lightricks/LTX-Video-0.9.5-spatial-upscaler). Frame budget grows
    # ~1.5×; the ``runtimeNote`` surfaces the substitution to users.
    enableLtxRefiner: bool = False
    # Phase E1: opt-in template-based prompt enhancement for short prompts
    # (< 25 words). See ``_enhance_prompt`` for the per-model suffixes.
    enhancePrompt: bool = True
    # Phase E2: CFG decay schedule. Linear ramp from initial guidance_scale
    # at step 0 to 1.0 at the last step. Default-on for flow-match pipelines.
    cfgDecay: bool = True


@dataclass(frozen=True)
class GeneratedVideo:
    """A single rendered mp4. Mirrors ``GeneratedImage`` from image_runtime."""
    seed: int
    bytes: bytes
    extension: str
    mimeType: str
    durationSeconds: float
    frameCount: int
    fps: int
    width: int
    height: int
    runtimeLabel: str
    runtimeNote: str | None = None


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
    # transformer scales but the loader is the same.
    "THUDM/CogVideoX-2b": {"class_name": "CogVideoXPipeline", "task": "txt2video"},
    "THUDM/CogVideoX-5b": {"class_name": "CogVideoXPipeline", "task": "txt2video"},
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


# Rough bf16 footprint for the full pipeline (transformer + VAE + text
# encoders) keyed on repo. Used by the memory-saver gate — slicing and
# tiling cut quality, so we only enable them when there's actual memory
# pressure. Numbers come from the catalog ``sizeGb`` estimates for the
# stock variants; GGUF Q4/Q6/Q8 variants override at the call site.
_VIDEO_MODEL_FOOTPRINT_BF16_GB: dict[str, float] = {
    "Lightricks/LTX-Video": 14.0,
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": 9.0,
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": 28.0,
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": 11.0,
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": 28.0,
    "hunyuanvideo-community/HunyuanVideo": 26.0,
    "genmo/mochi-1-preview": 20.0,
    "THUDM/CogVideoX-2b": 10.0,
    "THUDM/CogVideoX-5b": 18.0,
}

# GGUF quant level → multiplier vs the bf16 footprint. Keys are matched as
# substrings in the gguf filename so future quant levels (e.g. ``Q5_K_M``)
# fall through to a sensible default.
_GGUF_QUANT_MULTIPLIERS: tuple[tuple[str, float], ...] = (
    ("Q4", 0.30),
    ("Q5", 0.36),
    ("Q6", 0.42),
    ("Q8", 0.55),
)


def _estimate_model_footprint_gb(
    repo: str, dtype_name: str, gguf_file: str | None = None
) -> float | None:
    """Cheap estimate of a video pipeline's GPU/MPS memory footprint in GB.

    Returns ``None`` if the repo is unrecognised — callers treat that as
    "stay safe" and enable slicing. The dtype name is the str of the
    torch dtype (``"torch.bfloat16"`` etc.); we treat fp16/bf16 as the
    catalog baseline and double for fp32.
    """
    base = _VIDEO_MODEL_FOOTPRINT_BF16_GB.get(repo)
    if base is None:
        return None
    if gguf_file:
        upper = gguf_file.upper()
        for marker, multiplier in _GGUF_QUANT_MULTIPLIERS:
            if marker in upper:
                base = base * multiplier
                break
    if "float32" in dtype_name and "bfloat16" not in dtype_name:
        base = base * 2.0
    return base


def _should_apply_memory_savers(
    device: str, total_memory_gb: float | None, estimated_footprint_gb: float | None
) -> bool:
    """Decide whether to enable attention slicing + VAE slicing/tiling.

    Slicing trades quality for VRAM. The reference workflows don't enable it —
    we used to do it unconditionally, which left a 64 GB Mac running a
    1.3B model in slicing mode for no reason. Heuristic:

    - ``CHAOSENGINE_VIDEO_FORCE_SLICING=1`` always wins (rollback lever).
    - Unknown memory or unknown footprint → stay safe, enable slicing.
    - CPU device → enable, system RAM is shared.
    - Footprint > 70% of device memory → enable.
    - Otherwise → leave the pipeline at full quality.
    """
    if os.getenv("CHAOSENGINE_VIDEO_FORCE_SLICING") == "1":
        return True
    if device == "cpu":
        return True
    if total_memory_gb is None or estimated_footprint_gb is None:
        return True
    if total_memory_gb <= 0:
        return True
    return (estimated_footprint_gb / total_memory_gb) > 0.7


# Diffusers scheduler classes we expose via the ``scheduler`` request field.
# Resolved on the ``diffusers`` module at runtime so an old install that
# lacks one of these classes degrades to a logged warning instead of an
# import-time crash.
_SCHEDULER_CLASSES: dict[str, str] = {
    "unipc": "UniPCMultistepScheduler",
    "euler": "EulerDiscreteScheduler",
    # ``FlowMatchEulerDiscreteScheduler`` is the only scheduler that
    # accepts the ``mu`` kwarg LTXPipeline passes to ``set_timesteps``.
    # Older cached LTX snapshots have plain ``EulerDiscreteScheduler``
    # baked in; we force-swap on LTX to keep the pipeline call valid.
    "flow-euler": "FlowMatchEulerDiscreteScheduler",
    "dpm++": "DPMSolverMultistepScheduler",
    "ddim": "DDIMScheduler",
}


def _align_wan_num_frames(repo: str, requested: int) -> tuple[int, str | None]:
    """Round Wan's ``num_frames`` to the nearest valid ``(4k + 1)`` value.

    Wan models compute ``(n_frames - 1) / 4 + 1`` latent frames internally;
    off-spec counts produce mostly-black/garbled output. We round down to
    the nearest valid count rather than up so we don't silently exceed the
    user's requested clip length and frame budget.

    Returns ``(aligned_count, note_or_None)``. The note is surface-ready
    text — if non-None the caller should publish it to ``VIDEO_PROGRESS``
    and the run log so the UI explains why the count changed.
    """
    if "Wan" not in repo:
        return requested, None
    if requested < 5:
        return 5, "Wan requires num_frames >= 5; clamped."
    aligned = ((requested - 1) // 4) * 4 + 1
    if aligned != requested:
        return aligned, (
            f"Aligned num_frames {requested} → {aligned} (Wan requires 4k+1)."
        )
    return aligned, None


def _resolve_video_defaults(
    repo: str, requested_steps: int, requested_guidance: float
) -> dict[str, Any]:
    """Substitute per-model sweet-spot values when the user kept schema defaults.

    Heuristic: if the request matches the schema defaults exactly (50 steps,
    CFG 3.0) we treat it as "user did not dial this in" and substitute the
    upstream-recommended values. Any explicit deviation is preserved.

    Returns the resolved dict with ``steps``, ``guidance``, ``scheduler``,
    and ``substituted`` (True when at least one value was rewritten).
    """
    overrides = _VIDEO_PIPELINE_DEFAULTS.get(repo, {})
    resolved_steps = requested_steps
    resolved_guidance = requested_guidance
    substituted = False
    if requested_steps == _REQUEST_DEFAULT_STEPS and "steps" in overrides:
        resolved_steps = int(overrides["steps"])
        substituted = substituted or resolved_steps != requested_steps
    if requested_guidance == _REQUEST_DEFAULT_GUIDANCE and "guidance" in overrides:
        resolved_guidance = float(overrides["guidance"])
        substituted = substituted or resolved_guidance != requested_guidance
    return {
        "steps": resolved_steps,
        "guidance": resolved_guidance,
        "scheduler": overrides.get("scheduler"),
        "substituted": substituted,
    }


def _interpolate_frames(frames: list[Any], factor: int) -> list[Any]:
    """Insert ``factor - 1`` blended frames between each source pair.

    This is a linear-blend (numpy-weighted average) frame interpolator —
    simpler and faster than RIFE but gives visibly smoother motion at
    2x/4x. Swap this for a RIFE model call when the weights ship — the
    pipeline shape (``list[np.ndarray]`` in RGB uint8) stays the same.

    A factor of 1 is a no-op. Factors above 1 produce
    ``(len - 1) * factor + 1`` frames so the endpoint timings align
    with the original clip.
    """
    if factor <= 1 or len(frames) < 2:
        return list(frames)
    try:
        import numpy as np  # type: ignore
    except Exception:
        return list(frames)

    def _to_array(frame: Any):
        if hasattr(frame, "shape"):
            return np.asarray(frame)
        return np.asarray(frame, dtype=np.uint8)

    interpolated: list[Any] = []
    total = len(frames)
    for index in range(total - 1):
        current = _to_array(frames[index])
        nxt = _to_array(frames[index + 1])
        if current.shape != nxt.shape:
            # Different shape → skip blending, just duplicate. Robust
            # against frames of mixed dtypes (list of PIL Images).
            interpolated.append(frames[index])
            for _ in range(factor - 1):
                interpolated.append(frames[index])
            continue
        interpolated.append(frames[index])
        for sub_index in range(1, factor):
            alpha = sub_index / factor
            blended = (current.astype(np.float32) * (1.0 - alpha)
                       + nxt.astype(np.float32) * alpha)
            interpolated.append(
                np.clip(blended, 0, 255).astype(current.dtype)
            )
    interpolated.append(frames[-1])
    return interpolated


# Core packages that gate ``realGenerationAvailable``. Without these, the
# runtime can't even preload a model.
_CORE_DEPS: tuple[tuple[str, str], ...] = (
    ("diffusers", "diffusers"),
    ("torch", "torch"),
    ("accelerate", "accelerate"),
    ("huggingface_hub", "huggingface_hub"),
    ("pillow", "PIL"),
)


# Packages required only to write the final mp4. Reported as missing so users
# know what's needed for generation, but we don't block preload on them.
_VIDEO_OUTPUT_DEPS: tuple[tuple[str, str], ...] = (
    ("imageio", "imageio"),
    ("imageio-ffmpeg", "imageio_ffmpeg"),
)


# Packages individual video pipelines pull in lazily — only at preload or
# generate time, depending on the tokenizer / text encoder. Diffusers itself
# imports cleanly without them, so they don't block the runtime, but a user
# who picks LTX-Video without ``tiktoken`` installed sees a runtime error
# mid-generate. Surfacing them in the probe lets the Studio offer a one-
# click install before the user wastes a slow preload.
#
# Coverage at the time of writing:
# - tiktoken: LTX-Video's T5 tokenizer ships in tiktoken format.
# - sentencepiece: Wan (UMT5-XXL), HunyuanVideo, CogVideoX, Mochi (T5).
# - protobuf: required by the SentencePiece-based tokenizers HF loads.
# - ftfy: text-prep utility some pipelines use during prompt encoding.
_VIDEO_MODEL_DEPS: tuple[tuple[str, str], ...] = (
    ("tiktoken", "tiktoken"),
    ("sentencepiece", "sentencepiece"),
    ("protobuf", "google.protobuf"),
    ("ftfy", "ftfy"),
)


def _find_missing(deps: tuple[tuple[str, str], ...]) -> list[str]:
    # ``importlib.util.find_spec`` raises ``ModuleNotFoundError`` (not returns
    # ``None``) when the parent of a dotted name is not importable. Concretely:
    # ``find_spec("google.protobuf")`` blows up with "No module named 'google'"
    # on a machine that never installed protobuf, instead of just reporting
    # that protobuf is missing. Without this guard the probe crashes with a
    # 500 and the Video Studio shows "runtime did not respond" forever.
    missing: list[str] = []
    for package, module_name in deps:
        try:
            spec = importlib.util.find_spec(module_name)
        except (ModuleNotFoundError, ValueError, ImportError):
            spec = None
        if spec is None:
            missing.append(package)
    return missing


class DiffusersVideoEngine:
    """Thin wrapper around diffusers video pipelines.

    Single-pipeline at a time; preload() evicts the previous pipeline before
    loading a new one to avoid OOM on unified-memory machines. Generation
    is not implemented in this phase — see ``generate()`` which raises.
    """

    runtime_label = "Diffusers video engine"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._loaded_repo: str | None = None
        self._loaded_path: str | None = None
        self._loaded_variant_key: str | None = None
        self._device: str | None = None

    # ---------- public API ----------

    def probe(self) -> VideoRuntimeStatus:
        # Deliberately does NOT ``import torch`` or trigger the warmup
        # thread. Importing torch loads torch/lib/*.dll into the backend
        # process handle table, and on Windows those locked DLLs block
        # /api/setup/install-gpu-bundle from overwriting them (pip rmtree
        # fails with WinError 5). find_spec answers "is it installable?"
        # without the side effects. Device detection + broken-import
        # checks are deferred to preload/generate where we're about to
        # actually use torch.
        missing_core = _find_missing(_CORE_DEPS)
        missing_output = _find_missing(_VIDEO_OUTPUT_DEPS)
        missing_model = _find_missing(_VIDEO_MODEL_DEPS)

        # All missing deps are reported so the UI can surface a clear install
        # hint, but only ``_CORE_DEPS`` block ``realGenerationAvailable``.
        # ``_VIDEO_MODEL_DEPS`` are pipeline-specific (tiktoken for LTX,
        # sentencepiece for Wan/T5 etc.) — not all of them are needed for
        # every model, but listing them lets the Studio install proactively.
        missing_optional = missing_output + missing_model
        missing_all = missing_core + missing_optional

        if missing_core:
            # Include the missing package names in the message so consumers
            # that only see the RuntimeError string (e.g. preload()'s 500
            # response) still know WHAT to install — missingDependencies is
            # on the structured status but isn't plumbed through every path.
            return VideoRuntimeStatus(
                activeEngine="placeholder",
                realGenerationAvailable=False,
                missingDependencies=missing_all,
                pythonExecutable=_resolve_video_python(),
                expectedDevice=_guess_video_expected_device(),
                message=(
                    f"Video runtime needs these packages: {', '.join(missing_core)}. "
                    "Click the 'Install GPU runtime' button above to install the full bundle."
                ),
                loadedModelRepo=self._loaded_repo,
            )

        if missing_output and missing_model:
            message = (
                "Video runtime is ready to load models, but mp4 encoding and tokenizer packages "
                f"are missing — run `pip install {' '.join(missing_optional)}` before generating videos."
            )
        elif missing_output:
            message = (
                "Video runtime is ready to load models, but mp4 encoding packages are missing — "
                "run `pip install imageio imageio-ffmpeg` before generating videos."
            )
        elif missing_model:
            message = (
                "Video runtime is ready, but some models need tokenizer packages that are not "
                f"installed: {', '.join(missing_model)}. Install them now and the affected "
                "models will load on next preload."
            )
        else:
            message = (
                "Real local video generation is available. Download a video model, then Video Studio "
                "will use the diffusers runtime."
            )

        # ``device`` mirrors the currently-loaded model's runtime context —
        # None until preload, because importing torch speculatively locks
        # DLLs on Windows and breaks /api/setup/install-gpu-bundle.
        #
        # ``deviceMemoryGb`` is resolved independently. It reads sysctl on
        # macOS and nvidia-smi on Linux/Windows — neither needs a loaded
        # model, and both are cheap (cached per-process). Gating it behind
        # ``device is not None`` used to leave the frontend safety heuristic
        # with no data until first load, which made it fall back to its
        # 16 GB MPS default and warn a 64 GB M4 Max user as if they were
        # on a base-model Mac.
        device = self._device
        device_memory_gb = _detect_device_memory_gb(device)

        return VideoRuntimeStatus(
            activeEngine="diffusers",
            realGenerationAvailable=True,
            device=device,
            expectedDevice=_guess_video_expected_device(),
            pythonExecutable=_resolve_video_python(),
            missingDependencies=missing_optional,
            message=message,
            loadedModelRepo=self._loaded_repo,
            deviceMemoryGb=device_memory_gb,
        )

    def preload(self, repo: str) -> VideoRuntimeStatus:
        self._ensure_pipeline(repo)
        return self.probe()

    def unload(self, repo: str | None = None) -> VideoRuntimeStatus:
        with self._lock:
            if repo and self._loaded_repo != repo:
                return self.probe()
            self._release_pipeline()
            return self.probe()

    def generate(self, config: VideoGenerationConfig) -> GeneratedVideo:
        """Run a single text-to-video generation and return the encoded mp4.

        The hot path:
            1. Ensure the right pipeline is loaded.
            2. Build per-model kwargs.
            3. Run the pipeline with a seeded generator.
            4. Encode frames to mp4 via imageio-ffmpeg.
            5. Return bytes + metadata.

        We split the diffusers invocation and mp4 encoding into narrow seams
        (``_invoke_pipeline``, ``_encode_frames_to_mp4``) so tests can stub
        them without needing real 10+GB video weights on disk.
        """
        config, finalize_notes = self._finalize_config(config)
        VIDEO_PROGRESS.begin(
            run_label=self._format_run_label(config),
            total_steps=max(1, int(config.steps)),
            phase=PHASE_LOADING,
            message=f"Preparing {config.modelName}",
        )
        for note in finalize_notes:
            VIDEO_PROGRESS.set_phase(PHASE_LOADING, message=note)
            _LOG.info("video.finalize: %s", note)
        try:
            # mp4 encoding needs imageio-ffmpeg. Check before we spend 60+ seconds
            # doing a full generation we then can't save anywhere.
            missing_output = _find_missing(_VIDEO_OUTPUT_DEPS)
            if missing_output:
                raise RuntimeError(
                    "Video generation requires the mp4 encoding packages: "
                    f"missing {', '.join(missing_output)}. "
                    "Run `pip install imageio imageio-ffmpeg` and retry."
                )

            pipeline = self._ensure_pipeline(
                config.repo,
                gguf_repo=config.ggufRepo,
                gguf_file=config.ggufFile,
                use_nf4=config.useNf4,
            )
            # Early-cancel check after model load — from_pretrained is a
            # blocking C-extension call we can't interrupt. If the user hit
            # Cancel during load we catch up here and bail before we sink
            # time into T5 encoding + the denoising loop.
            if VIDEO_PROGRESS.is_cancelled():
                raise GenerationCancelled("Video generation cancelled by user")

            scheduler_note = self._swap_scheduler(pipeline, config.scheduler)
            if scheduler_note:
                VIDEO_PROGRESS.set_phase(PHASE_LOADING, message=scheduler_note)
                _LOG.info("video.scheduler: %s", scheduler_note)
            torch = self._torch
            if torch is None:
                raise RuntimeError("PyTorch was not initialised for the video runtime.")

            VIDEO_PROGRESS.set_phase(PHASE_ENCODING, message="Encoding prompt")

            base_seed = _resolve_video_seed(config.seed)
            # MPS generators don't seed the same way as CUDA/CPU — follow the
            # diffusers docs and always build the generator on CPU for MPS.
            generator_device = "cpu" if self._device == "mps" else (self._device or "cpu")
            generator = torch.Generator(device=generator_device).manual_seed(base_seed)

            kwargs = self._build_pipeline_kwargs(config, generator)

            VIDEO_PROGRESS.set_phase(
                PHASE_DIFFUSING,
                message=f"Diffusing {config.numFrames} frames",
            )
            VIDEO_PROGRESS.set_step(0, total=max(1, int(config.steps)))

            # TeaCache / other diffusion caches hook here — pipeline is
            # loaded and num_inference_steps is final. Video DiTs are
            # where TeaCache pays off most (1.6–2.1× on HunyuanVideo,
            # ~1.3–2× on Wan). NotImplementedError is swallowed by the
            # helper when the pipeline class has no vendored patch yet;
            # see FU-007 in CLAUDE.md.
            apply_diffusion_cache_strategy(
                pipeline,
                strategy_id=config.cacheStrategy,
                num_inference_steps=int(config.steps),
                rel_l1_thresh=config.cacheRelL1Thresh,
                domain="video",
            )

            started = time.perf_counter()
            if config.enableLtxRefiner and config.repo == "Lightricks/LTX-Video":
                try:
                    frames = self._invoke_pipeline_with_ltx_refiner(
                        pipeline, kwargs, torch
                    )
                    VIDEO_PROGRESS.set_phase(
                        PHASE_DIFFUSING,
                        message="LTX two-stage spatial upscale applied.",
                    )
                except Exception as exc:  # noqa: BLE001 — refiner is best-effort
                    note = (
                        f"LTX refiner skipped ({type(exc).__name__}: {exc}) — "
                        "running base pipeline only."
                    )
                    _LOG.info("video.ltx_refiner: %s", note)
                    VIDEO_PROGRESS.set_phase(PHASE_DIFFUSING, message=note)
                    frames = self._invoke_pipeline(pipeline, kwargs)
            else:
                frames = self._invoke_pipeline(pipeline, kwargs)
            elapsed = max(0.1, time.perf_counter() - started)

            if not frames:
                raise RuntimeError(
                    f"The video pipeline returned zero frames for {config.repo}. "
                    "Try a smaller resolution or a different model."
                )

            interpolation_factor = max(1, int(config.interpolationFactor or 1))
            if interpolation_factor > 1:
                VIDEO_PROGRESS.set_phase(
                    PHASE_DECODING,
                    message=f"Interpolating {interpolation_factor}x frames",
                )
                frames = _interpolate_frames(frames, interpolation_factor)
            effective_fps = config.fps * interpolation_factor
            VIDEO_PROGRESS.set_phase(PHASE_DECODING, message="Encoding mp4")
            mp4_bytes = self._encode_frames_to_mp4(frames, effective_fps)
            if not mp4_bytes:
                raise RuntimeError(
                    "mp4 encoding produced an empty buffer. Check that imageio-ffmpeg is "
                    "installed and healthy — run `python -m imageio_ffmpeg` to verify."
                )

            VIDEO_PROGRESS.set_phase(PHASE_SAVING, message="Saving to gallery")
            return GeneratedVideo(
                seed=base_seed,
                bytes=mp4_bytes,
                extension="mp4",
                mimeType="video/mp4",
                durationSeconds=round(elapsed, 2),
                frameCount=len(frames),
                fps=effective_fps,
                width=config.width,
                height=config.height,
                runtimeLabel=f"{self.runtime_label} ({self._device or 'cpu'})",
            )
        finally:
            VIDEO_PROGRESS.finish()

    def _format_run_label(self, config: VideoGenerationConfig) -> str:
        return f"{config.modelName} · {config.numFrames}f @ {config.width}x{config.height}"

    # ---------- internals ----------

    def _finalize_config(
        self, config: VideoGenerationConfig
    ) -> tuple[VideoGenerationConfig, list[str]]:
        """Apply per-model defaults + frame alignment + scheduler resolution.

        Centralised here so VIDEO_PROGRESS, the cache strategy hook, and the
        pipeline invocation all see the same resolved values. Returns a new
        (frozen) config + a list of human-readable notes the caller publishes
        to the run log.
        """
        notes: list[str] = []
        resolved = _resolve_video_defaults(config.repo, config.steps, config.guidance)
        steps = int(resolved["steps"])
        guidance = float(resolved["guidance"])
        if resolved.get("substituted"):
            notes.append(
                f"Substituting model-tuned defaults for {config.modelName}: "
                f"steps {config.steps} → {steps}, CFG {config.guidance} → {guidance}."
            )

        aligned_frames, frame_note = _align_wan_num_frames(config.repo, config.numFrames)
        if frame_note:
            notes.append(frame_note)

        # Scheduler: explicit request > model default > leave alone.
        requested_scheduler = (config.scheduler or "").strip().lower() or None
        if requested_scheduler == "auto":
            requested_scheduler = None
        scheduler = requested_scheduler or resolved.get("scheduler")
        if scheduler and scheduler not in _SCHEDULER_CLASSES:
            notes.append(
                f"Unknown scheduler {scheduler!r} — keeping the pipeline default."
            )
            scheduler = None

        # LTX-Video: surface the auto-tuned decode params + frame_rate
        # conditioning so the user sees why output quality matches the
        # Lightricks reference even though we didn't expose new sliders.
        if config.repo == "Lightricks/LTX-Video":
            notes.append(
                f"LTX-Video auto-tuned to Lightricks reference defaults: "
                f"frame_rate={int(config.fps)} (model conditioning), "
                f"decode_timestep=0.05, decode_noise_scale=0.025, "
                f"guidance_rescale=0.7."
            )

        # Phase E1 — auto-enhance short prompts. Default-on; opt-out via
        # config.enhancePrompt=False. Only fires below the word-count
        # threshold so a long custom prompt is never modified.
        enhanced_prompt = config.prompt
        if config.enhancePrompt:
            enhanced_prompt, enhance_note = _enhance_prompt(config.repo, config.prompt)
            if enhance_note:
                notes.append(enhance_note)

        # Phase E2 — CFG decay note. Only surfaces when decay actually
        # has somewhere to ramp (initial CFG > 1.5 — the floor that
        # keeps classifier-free guidance enabled throughout the loop).
        _CFG_DECAY_FLOOR = 1.5
        if config.cfgDecay and guidance > _CFG_DECAY_FLOOR and steps > 1:
            notes.append(
                f"CFG decay enabled: linearly ramping guidance_scale from "
                f"{guidance:.2f} (step 0) to {_CFG_DECAY_FLOOR} (final step) — "
                f"flow-match video models oversaturate when CFG stays high "
                f"throughout sampling. Floor stays above 1.0 so classifier-"
                f"free guidance keeps running 2-batch end-to-end."
            )

        return (
            replace(
                config,
                prompt=enhanced_prompt,
                steps=steps,
                guidance=guidance,
                numFrames=aligned_frames,
                scheduler=scheduler,
            ),
            notes,
        )

    def _swap_scheduler(self, pipeline: Any, scheduler_id: str | None) -> str | None:
        """Replace the pipeline's scheduler with the requested class.

        Returns a status message (non-None) iff the swap actually happened
        or failed in a user-relevant way. ``None`` means "no swap requested
        or pipeline already on this scheduler" — silent path.
        """
        if not scheduler_id:
            return None
        cls_name = _SCHEDULER_CLASSES.get(scheduler_id)
        if cls_name is None:
            return None
        current_cls = type(getattr(pipeline, "scheduler", None)).__name__
        if current_cls == cls_name:
            return None
        try:
            diffusers = importlib.import_module("diffusers")
        except Exception:
            return f"Scheduler swap skipped: diffusers import failed."
        scheduler_cls = getattr(diffusers, cls_name, None)
        if scheduler_cls is None:
            return (
                f"Scheduler {scheduler_id!r} ({cls_name}) not available in the "
                "installed diffusers — keeping the pipeline default."
            )
        try:
            pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)
        except Exception as exc:  # noqa: BLE001
            return f"Scheduler swap to {scheduler_id!r} failed: {exc}"
        return f"Scheduler swapped to {scheduler_id} ({cls_name})."

    def _build_pipeline_kwargs(
        self,
        config: VideoGenerationConfig,
        generator: Any,
    ) -> dict[str, Any]:
        """Per-model kwarg shaping.

        Most diffusers video pipelines accept the same shape, but there are
        small variations — e.g. HunyuanVideoPipeline does not accept a
        ``negative_prompt`` argument in its canonical signature.
        """
        kwargs: dict[str, Any] = {
            "prompt": config.prompt,
            "width": config.width,
            "height": config.height,
            "num_frames": config.numFrames,
            "num_inference_steps": config.steps,
            "guidance_scale": config.guidance,
            "generator": generator,
            # Force PIL output so ``_encode_frames_to_mp4`` always receives
            # ``list[PIL.Image]``. WanPipeline defaults to ``"np"``, which
            # returns a 5D numpy array (B, F, H, W, C). Our frame
            # post-processing assumes the diffusers PIL convention; a raw
            # numpy tensor leaks through and ``PIL.Image.fromarray`` then
            # raises "Image must have 1, 2, 3 or 4 channels" because it
            # reads the first non-batch dim as height. LTXPipeline
            # already defaults to "pil"; setting it explicitly here is
            # a no-op for LTX and the fix for Wan / Hunyuan / Mochi /
            # CogVideoX (all default to "np").
            "output_type": "pil",
        }
        lowered_repo = config.repo.lower()
        if "hunyuanvideo" not in lowered_repo and config.negativePrompt.strip():
            kwargs["negative_prompt"] = config.negativePrompt

        # LTX-Video kwargs parity with Lightricks' reference defaults.
        # Without these, diffusers' LTXPipeline produces rainbow / blurry
        # output because (1) the model conditions on default frame_rate=25
        # while our exporter writes config.fps, (2) the VAE decodes from
        # final latent without the small denoise pass that cleans
        # compression artifacts, (3) flow-match models oversaturate
        # without rescale. Reference: Lightricks LTX-Video model card.
        pipeline_cls = type(self._pipeline).__name__ if self._pipeline is not None else ""
        if pipeline_cls == "LTXPipeline":
            kwargs["frame_rate"] = int(config.fps)
            kwargs["decode_timestep"] = 0.05
            kwargs["decode_noise_scale"] = 0.025
            kwargs["guidance_rescale"] = 0.7
            # Inject Lightricks' recommended negative-prompt template when
            # the user hasn't overridden — LTX was trained with strong
            # negative-prompt conditioning, so the schema's softer default
            # ("blurry, low quality") leaves quality on the table.
            if not kwargs.get("negative_prompt"):
                kwargs["negative_prompt"] = _LTX_DEFAULT_NEGATIVE_PROMPT
        # Private kwarg consumed by ``_invoke_pipeline`` — pop'd before
        # passing to the diffusers pipeline, so it never reaches the
        # underlying call. Lets the engine plumb decay through one
        # callback factory rather than threading state through self.
        kwargs["__cfg_decay"] = bool(config.cfgDecay)
        return kwargs

    def _make_step_callback(
        self,
        total_steps: int,
        initial_guidance: float,
        cfg_decay: bool,
    ) -> Any:
        """Build the per-step callback the pipeline calls during sampling.

        Wires three concerns into one callback:
          1. Progress reporting via ``VIDEO_PROGRESS.set_step``.
          2. Cooperative cancel — raise ``GenerationCancelled`` when the
             user hits Cancel on the modal.
          3. Phase E2 CFG decay — linearly ramp ``pipeline.guidance_scale``
             from ``initial_guidance`` at step 0 toward 1.0 at the last
             step. Flow-match video models (LTX, Wan, HunyuanVideo) tend
             to oversaturate when CFG is held high through the whole
             schedule; decaying lets the early steps lock semantics
             (high CFG) while late steps preserve fine detail (low CFG).
        """
        # Floor MUST stay strictly above 1.0 so the pipeline's
        # ``do_classifier_free_guidance`` property (``_guidance_scale > 1.0``)
        # does not flip to False mid-loop. If it did, the pipeline would
        # switch from 2-batch (cond+uncond) to 1-batch on the last step
        # while the embeddings + scheduler state are still 2-batch shape,
        # crashing with shape mismatches or producing garbled frames
        # ("Image must have 1, 2, 3 or 4 channels" on Wan, batch
        # dimension errors on LTX).
        decay_floor = 1.5
        decay_active = cfg_decay and total_steps > 1 and initial_guidance > decay_floor

        def _on_step_end(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]):
            VIDEO_PROGRESS.set_step(step + 1, total=max(1, total_steps))
            if VIDEO_PROGRESS.is_cancelled():
                try:
                    _pipeline._interrupt = True
                except Exception:
                    pass
                raise GenerationCancelled("Video generation cancelled by user")
            if decay_active:
                # Step `step` just finished (step uses scale set BEFORE it).
                # Set the scale for step `step+1`. Linear ramp from initial
                # at step 0 to decay_floor at step total_steps-1.
                next_step = step + 1
                progress = min(1.0, next_step / max(1, total_steps - 1))
                next_scale = initial_guidance * (1.0 - progress) + decay_floor * progress
                try:
                    _pipeline.guidance_scale = float(next_scale)
                except Exception:
                    pass
            return callback_kwargs

        return _on_step_end

    def _invoke_pipeline(self, pipeline: Any, kwargs: dict[str, Any]) -> list[Any]:
        """Run the diffusers pipeline and return the first batch's frames.

        Carved out as a seam so tests can stub it without loading real
        weights. Diffusers video pipelines return an output with a
        ``.frames`` attribute shaped like ``list[list[PIL.Image]]`` — one
        inner list per batch item. We only ever render batchSize=1, so
        we return ``result.frames[0]``.

        Wires the diffusers per-step callback into ``VIDEO_PROGRESS`` so the
        UI bar tracks denoising in real time. Falls back to a callback-free
        invocation on older diffusers versions that don't expose the kwarg.
        """
        total_steps = int(kwargs.get("num_inference_steps") or 0)
        initial_guidance = float(kwargs.get("guidance_scale") or 1.0)
        # Phase E2: CFG decay flag is plumbed via a private kwarg the
        # caller pops before passing to the pipeline. Default-on when
        # absent so existing call sites pick up the schedule.
        cfg_decay = bool(kwargs.pop("__cfg_decay", True))
        callback = self._make_step_callback(total_steps, initial_guidance, cfg_decay)
        kwargs.setdefault("callback_on_step_end", callback)

        try:
            result = pipeline(**kwargs)
        except TypeError as exc:
            message = str(exc)
            # Older diffusers / pipelines that don't accept ``callback_on_step_end``.
            if "callback_on_step_end" in message:
                kwargs = {k: v for k, v in kwargs.items() if k != "callback_on_step_end"}
                try:
                    result = pipeline(**kwargs)
                except TypeError as inner:
                    if "negative_prompt" in str(inner) and "negative_prompt" in kwargs:
                        kwargs = {k: v for k, v in kwargs.items() if k != "negative_prompt"}
                        result = pipeline(**kwargs)
                    else:
                        raise
            elif "negative_prompt" in message and "negative_prompt" in kwargs:
                # Some pipelines reject ``negative_prompt`` even when given a
                # non-empty value. Fall back once without it rather than crashing
                # the whole generation.
                kwargs = {key: value for key, value in kwargs.items() if key != "negative_prompt"}
                result = pipeline(**kwargs)
            else:
                raise

        frames = getattr(result, "frames", None)
        if frames is None:
            raise RuntimeError(
                "Video pipeline result is missing a `.frames` attribute. "
                "This usually means the installed diffusers version returns a "
                "different output shape. Upgrade diffusers: pip install -U diffusers"
            )
        if isinstance(frames, (list, tuple)) and frames and isinstance(frames[0], (list, tuple)):
            return list(frames[0])
        return list(frames)

    def _invoke_pipeline_with_ltx_refiner(
        self, pipeline: Any, kwargs: dict[str, Any], torch: Any
    ) -> list[Any]:
        """Run LTX base + LTXLatentUpsamplePipeline spatial 2× upscale.

        Mirrors the upstream Lightricks LTX-Video two-stage pattern:
        sample latents through ``LTXPipeline`` then refine through
        ``LTXLatentUpsamplePipeline`` loaded from the
        ``Lightricks/LTX-Video-0.9.5-spatial-upscaler`` snapshot. Both
        snapshots must be locally cached — we never auto-download from
        within ``generate``. Failure modes (snapshot missing, diffusers
        too old, decode error) propagate to the caller which falls back
        to the base pipeline.
        """
        from huggingface_hub import snapshot_download  # type: ignore

        diffusers = importlib.import_module("diffusers")
        upscaler_cls = getattr(diffusers, "LTXLatentUpsamplePipeline", None)
        if upscaler_cls is None:
            raise RuntimeError(
                "Installed diffusers does not expose LTXLatentUpsamplePipeline."
            )
        upscaler_repo = "Lightricks/LTX-Video-0.9.5-spatial-upscaler"
        upscaler_path = snapshot_download(
            repo_id=upscaler_repo,
            local_files_only=True,
            resume_download=True,
        )

        base_kwargs = dict(kwargs)
        base_kwargs["output_type"] = "latent"
        base_result = pipeline(**base_kwargs)
        latents = getattr(base_result, "frames", None)
        if latents is None:
            raise RuntimeError("LTX base pipeline returned no latents.")

        device = self._device or "cpu"
        dtype = self._preferred_torch_dtype(torch, device)
        upscaler = upscaler_cls.from_pretrained(
            upscaler_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        if device != "cpu":
            try:
                upscaler = upscaler.to(device)
            except (RuntimeError, MemoryError):
                if hasattr(upscaler, "enable_sequential_cpu_offload"):
                    upscaler.enable_sequential_cpu_offload()
                else:
                    raise

        try:
            refined = upscaler(latents=latents)
        finally:
            del upscaler
            gc.collect()

        frames = getattr(refined, "frames", None)
        if frames is None:
            raise RuntimeError("LTX refiner returned no frames.")
        if isinstance(frames, (list, tuple)) and frames and isinstance(frames[0], (list, tuple)):
            return list(frames[0])
        return list(frames)

    def _encode_frames_to_mp4(self, frames: list[Any], fps: int) -> bytes:
        """Encode a list of PIL.Image frames to an mp4 byte buffer.

        Carved out as a seam so tests can stub it. We use ``imageio`` +
        ``imageio-ffmpeg`` via the ``diffusers.utils.export_to_video`` helper
        when available (it handles the numpy conversion), and fall back to a
        direct ``imageio`` writer if diffusers hasn't exposed the helper on
        the installed version.
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            tmp_path = handle.name
        try:
            export_to_video = None
            try:
                from diffusers.utils import export_to_video as _export  # type: ignore
                export_to_video = _export
            except Exception:
                export_to_video = None

            if export_to_video is not None:
                export_to_video(frames, tmp_path, fps=fps)
            else:
                # Minimal fallback — avoids tying us to diffusers' helper
                # layout. Uses the same pyav backend imageio-ffmpeg ships.
                import numpy as np  # type: ignore
                import imageio  # type: ignore

                writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
                try:
                    for frame in frames:
                        array = np.asarray(frame)
                        if array.ndim == 2:
                            array = np.stack([array] * 3, axis=-1)
                        writer.append_data(array.astype("uint8"))
                finally:
                    writer.close()

            return Path(tmp_path).read_bytes()
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    def _pipeline_class(self, repo: str) -> Any:
        entry = PIPELINE_REGISTRY.get(repo)
        if entry is None:
            raise RuntimeError(
                f"No diffusers pipeline is registered for repo '{repo}'. "
                f"Supported repos: {sorted(PIPELINE_REGISTRY.keys())}"
            )
        class_name = entry["class_name"]
        diffusers = importlib.import_module("diffusers")
        pipeline_cls = getattr(diffusers, class_name, None)
        if pipeline_cls is None:
            raise RuntimeError(
                f"The installed diffusers version does not expose '{class_name}'. "
                "Upgrade diffusers: pip install -U diffusers"
            )
        return pipeline_cls

    def _ensure_pipeline(
        self,
        repo: str,
        gguf_repo: str | None = None,
        gguf_file: str | None = None,
        use_nf4: bool = False,
    ) -> Any:
        with self._lock:
            variant_suffix = ""
            if gguf_file:
                variant_suffix = f"::{gguf_file}"
            elif use_nf4:
                variant_suffix = "::nf4"
            variant_key = f"{repo}{variant_suffix}" if variant_suffix else repo
            if self._pipeline is not None and self._loaded_variant_key == variant_key:
                return self._pipeline

            # Loading a video pipeline can read 10+ GB from disk on cold cache.
            # Publish the phase so the UI explicitly says "Loading model" while
            # snapshot_download + from_pretrained run.
            VIDEO_PROGRESS.set_phase(PHASE_LOADING, message=f"Loading {repo}")

            if self._pipeline is not None and self._loaded_variant_key != variant_key:
                self._release_pipeline()

            import torch  # type: ignore
            from huggingface_hub import snapshot_download  # type: ignore

            pipeline_cls = self._pipeline_class(repo)

            local_path = snapshot_download(
                repo_id=repo,
                local_files_only=True,
                resume_download=True,
            )
            local_root = Path(local_path)
            validation_error = validate_local_diffusers_snapshot(local_root, repo)
            if validation_error is not None:
                raise RuntimeError(validation_error)

            device = self._detect_device(torch)
            dtype = self._preferred_torch_dtype(torch, device)

            pipeline_kwargs: dict[str, Any] = {}
            if gguf_file:
                VIDEO_PROGRESS.set_phase(
                    PHASE_LOADING,
                    message=f"Loading GGUF transformer {gguf_file}",
                )
                quantized_transformer, gguf_note = self._try_load_gguf_transformer(
                    repo=repo,
                    gguf_repo=gguf_repo or repo,
                    gguf_file=gguf_file,
                    torch=torch,
                )
                if quantized_transformer is not None:
                    pipeline_kwargs["transformer"] = quantized_transformer
                if gguf_note:
                    VIDEO_PROGRESS.set_phase(PHASE_LOADING, message=gguf_note)
            elif use_nf4:
                VIDEO_PROGRESS.set_phase(
                    PHASE_LOADING,
                    message="Loading NF4 transformer (bitsandbytes)",
                )
                nf4_transformer, nf4_note = self._try_load_bnb_nf4_transformer(
                    repo=repo,
                    local_path=local_path,
                    torch=torch,
                    device=device,
                )
                if nf4_transformer is not None:
                    pipeline_kwargs["transformer"] = nf4_transformer
                if nf4_note:
                    VIDEO_PROGRESS.set_phase(PHASE_LOADING, message=nf4_note)

            pipeline = pipeline_cls.from_pretrained(
                local_path,
                torch_dtype=dtype,
                local_files_only=True,
                **pipeline_kwargs,
            )

            if hasattr(pipeline, "set_progress_bar_config"):
                pipeline.set_progress_bar_config(disable=True)

            # Memory-saving knobs. Slicing + tiling are quality-lossy and
            # Reference workflows don't enable them by default — only flip them on
            # when there's real pressure. See ``_should_apply_memory_savers``
            # for the decision matrix.
            total_memory_gb = _detect_device_memory_gb(device)
            estimated_footprint_gb = _estimate_model_footprint_gb(
                repo, str(dtype), gguf_file=gguf_file
            )
            if _should_apply_memory_savers(device, total_memory_gb, estimated_footprint_gb):
                _LOG.info(
                    "video.memory_savers: enabled (device=%s, total_gb=%s, "
                    "estimated_gb=%s)",
                    device,
                    total_memory_gb,
                    estimated_footprint_gb,
                )
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing()
                vae = getattr(pipeline, "vae", None)
                if vae is not None:
                    if hasattr(vae, "enable_slicing"):
                        vae.enable_slicing()
                    if hasattr(vae, "enable_tiling"):
                        vae.enable_tiling()
            else:
                _LOG.info(
                    "video.memory_savers: skipped (device=%s, total_gb=%s, "
                    "estimated_gb=%s) — full quality path.",
                    device,
                    total_memory_gb,
                    estimated_footprint_gb,
                )

            if device != "cpu":
                # MoE pipelines (Wan 2.2 A14B has both ``transformer`` and
                # ``transformer_2``) cannot fit two 28 GB experts in unified
                # memory on a 64 GB Mac. Skip the full-device placement path
                # and engage sequential CPU offload directly so the active
                # expert lives on-device while the inactive one swaps to
                # CPU. Without this, ``.to("mps")`` would raise mid-copy
                # and the user would see a hard crash.
                is_moe = (
                    hasattr(pipeline, "transformer_2")
                    and getattr(pipeline, "transformer_2", None) is not None
                )
                if is_moe and hasattr(pipeline, "enable_sequential_cpu_offload"):
                    _LOG.info(
                        "video.placement: MoE pipeline detected (transformer + transformer_2) — "
                        "engaging enable_sequential_cpu_offload() proactively to keep peak under "
                        "device memory."
                    )
                    pipeline.enable_sequential_cpu_offload()
                else:
                    # Try full-device placement first; fall back to sequential
                    # CPU offload if the model is too big to fit.
                    try:
                        pipeline = pipeline.to(device)
                    except (RuntimeError, MemoryError):
                        if hasattr(pipeline, "enable_sequential_cpu_offload"):
                            pipeline.enable_sequential_cpu_offload()
                        else:
                            raise

            self._pipeline = pipeline
            self._torch = torch
            self._loaded_repo = repo
            self._loaded_path = local_path
            self._loaded_variant_key = variant_key
            self._device = device
            return pipeline

    def _try_load_gguf_transformer(
        self,
        repo: str,
        gguf_repo: str,
        gguf_file: str,
        torch: Any,
    ) -> tuple[Any, str | None]:
        """Load a video DiT from a single ``.gguf`` file via diffusers.

        Mirrors the image-side loader: GGUF weights cover the DiT only;
        VAE and text encoders are loaded from the base ``repo`` snapshot.
        All failure modes are non-fatal — a missing ``gguf`` package, an
        old diffusers without ``GGUFQuantizationConfig``, or an HF cache
        miss falls back to the standard fp16 / bf16 transformer path.
        """
        if importlib.util.find_spec("gguf") is None:
            return None, (
                "gguf package missing — install it from the Setup page to "
                f"load {gguf_file}. Falling back to the standard transformer."
            )
        try:
            from diffusers import GGUFQuantizationConfig  # type: ignore
        except ImportError:
            return None, (
                "Installed diffusers doesn't expose GGUFQuantizationConfig. "
                "Upgrade diffusers via the Setup page to use GGUF variants."
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
            transformer = transformer_cls.from_single_file(
                gguf_local_path,
                quantization_config=GGUFQuantizationConfig(
                    compute_dtype=torch.bfloat16,
                ),
                torch_dtype=torch.bfloat16,
            )
            return transformer, f"Transformer loaded from GGUF ({gguf_file})"
        except Exception as exc:  # noqa: BLE001 — any failure → fall back
            return None, (
                f"GGUF load failed ({type(exc).__name__}: {exc}) — "
                "falling back to the standard transformer."
            )

    def _try_load_bnb_nf4_transformer(
        self,
        repo: str,
        local_path: str,
        torch: Any,
        device: str,
    ) -> tuple[Any, str | None]:
        """Load a video DiT in NF4 4-bit via bitsandbytes.

        CUDA-only — bitsandbytes has no Metal/MPS backend, and the kernels
        wouldn't help on a 64 GB Mac anyway. Failure modes (non-CUDA host,
        missing bitsandbytes, old diffusers without ``BitsAndBytesConfig``,
        unmapped repo, broken snapshot subfolder) all return ``(None, note)``
        so the caller falls back to the standard fp16 / bf16 transformer.

        The transformer subfolder pattern (``from_pretrained(local_path,
        subfolder="transformer", quantization_config=...)``) matches the
        Wan / HunyuanVideo / LTX-Video diffusers snapshots — VAE and text
        encoders still load via the parent pipeline ``from_pretrained`` on
        the same snapshot root.
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

    def _release_pipeline(self) -> None:
        pipeline = self._pipeline
        torch = self._torch
        device = self._device
        self._pipeline = None
        self._torch = None
        self._loaded_repo = None
        self._loaded_path = None
        self._loaded_variant_key = None
        self._device = None
        if pipeline is not None:
            del pipeline
        gc.collect()
        if torch is not None:
            try:
                if device == "cuda" and getattr(torch.cuda, "is_available", lambda: False)():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                mps_backend = getattr(getattr(torch, "mps", None), "empty_cache", None)
                if device == "mps" and callable(mps_backend):
                    mps_backend()
            except Exception:
                pass

    def _detect_device(self, torch: Any) -> str:
        if getattr(torch.cuda, "is_available", lambda: False)():
            return "cuda"
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
            return "mps"
        return "cpu"

    def _preferred_torch_dtype(self, torch: Any, device: str) -> Any:
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


def _is_longlive_repo(repo: str | None) -> bool:
    """Route LongLive repos to the subprocess engine, everything else to diffusers.

    LongLive is not a diffusers pipeline — it ships as a torchrun-launched
    script with its own CUDA-specific deps that we keep in an isolated
    venv (see ``backend_service.longlive_engine``). Routing happens by
    repo prefix so the rest of the video stack doesn't need to know
    there's a second engine behind the manager.
    """
    if not repo:
        return False
    return repo.startswith("NVlabs/LongLive")


class VideoRuntimeManager:
    """State-level facade that mirrors ``ImageRuntimeManager``."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._engine = DiffusersVideoEngine()
        # Lazy-constructed so the LongLive import (and its probe, which
        # shells out to nvidia-smi) doesn't run on every sidecar start —
        # only when a LongLive repo is actually selected.
        self._longlive: Any | None = None
        # Same pattern for mlx-video (FU-009). Probe-only in this phase
        # — generate() raises, preload/generate are not routed through
        # the manager yet. See ``mlx_video_runtime`` module docstring.
        self._mlx_video: Any | None = None
        # sd.cpp video engine (FU-008). Scaffold only: probe + preload
        # routed; generate() raises NotImplementedError so the manager
        # falls through to diffusers until the CLI subprocess lands.
        self._sdcpp_video: Any | None = None

    def _get_longlive(self) -> Any:
        if self._longlive is None:
            from backend_service.longlive_engine import LongLiveEngine
            self._longlive = LongLiveEngine()
        return self._longlive

    def _get_mlx_video(self) -> Any:
        if self._mlx_video is None:
            from backend_service.mlx_video_runtime import MlxVideoEngine
            self._mlx_video = MlxVideoEngine()
        return self._mlx_video

    def _is_mlx_video_repo(self, repo: str | None) -> bool:
        """Routing predicate for mlx-video. Avoids importing the engine
        module unless the repo prefix actually matches."""
        if not repo:
            return False
        from backend_service.mlx_video_runtime import _is_mlx_video_repo
        return _is_mlx_video_repo(repo)

    def _get_sdcpp_video(self) -> Any:
        if self._sdcpp_video is None:
            from backend_service.sdcpp_video_runtime import SdCppVideoEngine
            self._sdcpp_video = SdCppVideoEngine()
        return self._sdcpp_video

    def _is_sdcpp_video_repo(self, repo: str | None) -> bool:
        if not repo:
            return False
        from backend_service.sdcpp_video_runtime import _is_sdcpp_video_repo
        return _is_sdcpp_video_repo(repo)

    def capabilities(self) -> dict[str, Any]:
        return self._engine.probe().to_dict()

    def longlive_capabilities(self) -> dict[str, Any]:
        """Probe the LongLive engine separately so the Studio can surface install state."""
        return self._get_longlive().probe().to_dict()

    def mlx_video_capabilities(self) -> dict[str, Any]:
        """Probe the mlx-video engine so Setup can surface install state.

        On Apple Silicon with mlx-video installed, the manager routes
        ``prince-canuma/LTX-2-*`` repos here before falling through to
        diffusers — see ``generate``. Wan paths still use diffusers MPS
        until the mlx-video Wan conversion step is bundled.
        """
        return self._get_mlx_video().probe().to_dict()

    def sdcpp_video_capabilities(self) -> dict[str, Any]:
        """Probe the sd.cpp engine so Setup/Studio can surface staging state.

        Scaffold today: ``realGenerationAvailable`` is always ``False``
        because ``generate()`` is unwired. Probe still reports binary
        presence so the UI can prompt the user to stage `sd` ahead of
        the FU-008 generation cutover.
        """
        return self._get_sdcpp_video().probe().to_dict()

    def preload(self, repo: str) -> dict[str, Any]:
        with self._lock:
            if _is_longlive_repo(repo):
                engine = self._get_longlive()
                status = engine.probe()
                if not status.realGenerationAvailable:
                    raise RuntimeError(status.message)
                return engine.preload(repo).to_dict()
            if self._is_mlx_video_repo(repo):
                mlx = self._get_mlx_video()
                status = mlx.probe()
                if not status.realGenerationAvailable:
                    raise RuntimeError(status.message)
                return mlx.preload(repo).to_dict()
            status = self._engine.probe()
            if not status.realGenerationAvailable:
                raise RuntimeError(status.message)
            return self._engine.preload(repo).to_dict()

    def unload(self, repo: str | None = None) -> dict[str, Any]:
        with self._lock:
            if _is_longlive_repo(repo):
                return self._get_longlive().unload(repo).to_dict()
            if self._is_mlx_video_repo(repo):
                return self._get_mlx_video().unload(repo).to_dict()
            return self._engine.unload(repo).to_dict()

    def generate(self, config: VideoGenerationConfig) -> tuple[GeneratedVideo, dict[str, Any]]:
        """Run a single video generation, returning (video, runtime_status).

        Unlike the image manager, there is no placeholder fallback — video is
        heavy enough that a silent fake clip would waste the user's time. If
        the runtime isn't ready, raise a clear error so the route can return
        a proper 4xx.
        """
        if _is_longlive_repo(config.repo):
            engine = self._get_longlive()
            status = engine.probe()
            if not status.realGenerationAvailable:
                raise RuntimeError(status.message)
            with self._lock:
                video = engine.generate(config)
                runtime = engine.probe().to_dict()
            return video, runtime

        if self._is_mlx_video_repo(config.repo):
            mlx = self._get_mlx_video()
            status = mlx.probe()
            if status.realGenerationAvailable:
                with self._lock:
                    video = mlx.generate(config)
                    runtime = mlx.probe().to_dict()
                return video, runtime
            # mlx-video not available (Intel Mac, missing package, etc.) —
            # fall through to diffusers so the supported repo doesn't dead-
            # end. Diffusers won't actually load LTX-2-* (no compatible
            # pipeline yet), so this branch effectively only covers the
            # "supported repo on a non-Apple-Silicon host" edge case.

        status = self._engine.probe()
        if not status.realGenerationAvailable:
            raise RuntimeError(status.message)
        with self._lock:
            video = self._engine.generate(config)
            runtime = self._engine.probe().to_dict()
        return video, runtime
