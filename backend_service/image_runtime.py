from __future__ import annotations

import json
import importlib.util
import io
import os
import platform
import textwrap
import time
import gc
import secrets

from backend_service.helpers.gpu import nvidia_gpu_present as _nvidia_gpu_present
from colorsys import hsv_to_rgb
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from backend_service.progress import (
    GenerationCancelled,
    IMAGE_PROGRESS,
    PHASE_DECODING,
    PHASE_DIFFUSING,
    PHASE_ENCODING,
    PHASE_LOADING,
    PHASE_SAVING,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGE_SEED = 2147483647


def _snapshot_retry_guidance(repo: str | None = None) -> str:
    guidance = "Re-download the model and keep ChaosEngineAI open until the download completes."
    if repo:
        guidance += (
            f" If this model is gated, accept access on https://huggingface.co/{repo} if prompted, "
            "add HF_TOKEN in Settings, then retry."
        )
    return guidance


def _snapshot_visible_label(local_root: Path) -> str:
    try:
        visible_files = sorted(
            candidate.name
            for candidate in local_root.iterdir()
            if not candidate.name.startswith(".")
        )
    except OSError:
        visible_files = []
    return ", ".join(visible_files[:6]) if visible_files else "no files"


def validate_local_diffusers_snapshot(local_root: Path, repo: str | None = None) -> str | None:
    model_index_path = local_root / "model_index.json"
    if not model_index_path.exists():
        visible_label = _snapshot_visible_label(local_root)
        return (
            "The local snapshot is incomplete and cannot be opened as a diffusers pipeline "
            f"(missing model_index.json; found {visible_label}). {_snapshot_retry_guidance(repo)}"
        )

    # Verify each component listed in model_index.json actually has its folder
    # on disk with a recognisable config file. Diffusers will otherwise raise a
    # cryptic "no file named config.json found in directory <snapshot_root>"
    # error from inside ``from_pretrained`` that points at the snapshot root,
    # which is hard to action without knowing which subfolder is missing.
    # This typically happens when a download started before allow_patterns was
    # applied — HF queues the legacy root-level safetensors first and the user
    # tries to load before the per-component folders finish landing.
    try:
        pipeline_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            "The local snapshot's model_index.json could not be read "
            f"({exc}). {_snapshot_retry_guidance(repo)}"
        )

    missing_components: list[str] = []
    if isinstance(pipeline_index, dict):
        # Any of these names being present in a subfolder is enough to call it
        # a real component directory — diffusers picks the right one based on
        # the class type at load time.
        component_config_names = (
            "config.json",
            "scheduler_config.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
        )
        for component_name, descriptor in pipeline_index.items():
            if component_name.startswith("_"):
                continue  # ``_class_name`` / ``_diffusers_version`` metadata
            if not isinstance(descriptor, (list, tuple)) or len(descriptor) < 2:
                continue
            # Pipelines list ``[null, null]`` for optional components that the
            # checkpoint deliberately omits (e.g. safety_checker on community
            # models). Skip those — they aren't expected on disk.
            if descriptor[0] is None or descriptor[1] is None:
                continue
            component_dir = local_root / component_name
            if not component_dir.is_dir():
                missing_components.append(component_name)
                continue
            if not any((component_dir / name).exists() for name in component_config_names):
                missing_components.append(component_name)

    if missing_components:
        label = ", ".join(missing_components[:4])
        if len(missing_components) > 4:
            label += f" (+{len(missing_components) - 4} more)"
        return (
            "The local snapshot is incomplete and cannot be opened as a diffusers pipeline "
            f"(missing components: {label}). {_snapshot_retry_guidance(repo)}"
        )

    broken_links: list[str] = []
    weight_index_paths: list[Path] = []
    try:
        for candidate in local_root.rglob("*"):
            if candidate.is_dir():
                continue
            if candidate.is_symlink() and not candidate.exists():
                broken_links.append(candidate.relative_to(local_root).as_posix())
            if candidate.name.endswith(".index.json"):
                weight_index_paths.append(candidate)
    except OSError as exc:
        return (
            "The local snapshot could not be inspected before loading "
            f"({exc}). {_snapshot_retry_guidance(repo)}"
        )

    if broken_links:
        label = ", ".join(broken_links[:3])
        if len(broken_links) > 3:
            label += f" (+{len(broken_links) - 3} more)"
        return (
            "The local snapshot is incomplete and cannot be opened as a diffusers pipeline "
            f"(missing local files: {label}). {_snapshot_retry_guidance(repo)}"
        )

    missing_shards: list[str] = []
    for index_path in weight_index_paths:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rel_path = index_path.relative_to(local_root).as_posix()
            return (
                "The local snapshot is incomplete and cannot be opened as a diffusers pipeline "
                f"(could not read {rel_path}: {exc}). {_snapshot_retry_guidance(repo)}"
            )
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            continue
        shard_names = sorted({value for value in weight_map.values() if isinstance(value, str) and value})
        for shard_name in shard_names:
            shard_path = index_path.parent / shard_name
            if shard_path.exists():
                continue
            missing_shards.append(shard_path.relative_to(local_root).as_posix())

    if missing_shards:
        label = ", ".join(missing_shards[:3])
        if len(missing_shards) > 3:
            label += f" (+{len(missing_shards) - 3} more)"
        return (
            "The local snapshot is incomplete and cannot be opened as a diffusers pipeline "
            f"(missing weight shards: {label}). {_snapshot_retry_guidance(repo)}"
        )

    return None


def _resolve_image_python() -> str:
    override = os.getenv("CHAOSENGINE_MLX_PYTHON")
    if override:
        return override
    candidate = WORKSPACE_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return os.getenv("PYTHON", "python3")


def _guess_expected_device() -> str | None:
    """Best-effort prediction of what device diffusers will bind to on
    the next Generate click, computed WITHOUT importing torch.

    Importing torch here would lock torch/lib/*.dll in the backend
    process and block /api/setup/install-gpu-bundle on Windows (same
    trap we hit before). find_spec + nvidia_gpu_present are free.
    Returns ``None`` when torch isn't installed — caller surfaces
    the probe's ``missingDependencies`` list instead.

    Predicted device is provisional; the actual device used at
    generate time is what ``_detect_device`` decides once torch is
    imported. Mismatch is rare (driver missing, torch was CPU-only)
    and gets corrected in ``device`` once a model is loaded.
    """
    if importlib.util.find_spec("torch") is None:
        return None
    if _nvidia_gpu_present():
        return "cuda"
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "mps"
    return "cpu"


# FU-017: madebyollin's SDXL VAE fp16 fix. The stock SDXL VAE silently
# decodes to NaN at fp16 on MPS and on consumer CUDA fp16 paths — the
# image_runtime currently sidesteps the bug by forcing fp32 on MPS for
# SDXL repos, which doubles wall time. The fp16-fix VAE is a drop-in
# replacement (same architecture, weights re-quantised to avoid NaN
# overflow on fp16 sigmoid) so swapping it in lets MPS / CUDA stay on
# fp16 without producing black images.
#
# We only attempt the swap when the snapshot is already in the user's
# HF cache (``local_files_only=True``) — the runtime never triggers a
# surprise download. Users who haven't fetched the fix repo see the
# original fp32 fallback path.
_SDXL_VAE_FIX_REPO = "madebyollin/sdxl-vae-fp16-fix"


def _is_sdxl_repo(repo: str) -> bool:
    """Match SDXL family repos (Stability XL base, refiner, community fine-tunes).

    Matches loosely on substring — a false positive would attempt the
    VAE swap on a non-SDXL repo, but the fp16-fix VAE only loads
    successfully against an SDXL pipeline because the encoder/decoder
    shape has to match. ``AutoencoderKL.from_pretrained`` raises on
    mismatch and the swap silently no-ops, so an over-broad match is
    self-correcting.
    """
    lower = repo.lower()
    return "stable-diffusion-xl" in lower or "sdxl" in lower or "sd_xl" in lower


def _locate_sdxl_vae_fix_snapshot() -> str | None:
    """Return the local path to ``madebyollin/sdxl-vae-fp16-fix`` if cached.

    Uses ``snapshot_download(local_files_only=True)`` so a missing snapshot
    returns ``None`` rather than triggering a download mid-generate. Users
    who want the fp16-fix path opt in by downloading the repo from the
    Setup page (or via ``huggingface-cli download``); until then the
    runtime stays on the existing fp32-on-MPS fallback for SDXL.
    """
    if importlib.util.find_spec("huggingface_hub") is None:
        return None
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception:
        return None
    try:
        return snapshot_download(
            repo_id=_SDXL_VAE_FIX_REPO,
            local_files_only=True,
            resume_download=True,
        )
    except Exception:
        return None


def _is_flux_repo(repo: str) -> bool:
    """Does this HF repo look like a FLUX.1 family model?

    FLUX family checkpoints are published under the
    ``black-forest-labs/FLUX.1-*`` namespace (Dev, Schnell, Kontext, etc.)
    plus a long tail of community fine-tunes that keep "flux" in their
    repo name. We match loosely by lowercased substring — the
    consequence of a false positive (using bf16 + cpu-offload on a non-
    FLUX model) is "slower than optimal on this machine", not incorrect
    output, so erring wide is fine.
    """
    lowered = repo.lower()
    return "flux" in lowered


def _is_flow_matching_repo(repo: str) -> bool:
    """Flow-matching pipelines (FLUX, SD3, Qwen-Image) ship locked
    schedulers — swapping to DDIM/Euler/DPM++ silently produces noise
    because the model was trained against a flow-matching ODE, not
    epsilon/v-prediction. Gate the sampler dropdown on this so the UI
    only shows it for SD1.5 / SDXL / SD2 where scheduler swap is safe.
    """
    lowered = repo.lower()
    return (
        _is_flux_repo(repo)
        or "stable-diffusion-3" in lowered
        or "sd3" in lowered
        or "qwen-image" in lowered
        or "sana" in lowered
        or "hidream" in lowered
    )


def _gguf_transformer_class_for_repo(repo: str) -> str | None:
    """Map a base repo to the diffusers transformer class used for GGUF.

    GGUF ``.from_single_file`` needs the right class — FLUX and SD3 both
    ship their own MMDiT/FluxTransformer variants, and loading a FLUX GGUF
    into ``SD3Transformer2DModel`` produces garbage. Returns ``None`` for
    families we don't ship GGUF variants for (SD1.5/SDXL use UNets, which
    have a different loading path that we don't support yet).
    """
    lowered = repo.lower()
    if _is_flux_repo(repo):
        return "FluxTransformer2DModel"
    if "stable-diffusion-3" in lowered or "sd3" in lowered:
        return "SD3Transformer2DModel"
    if "hidream" in lowered:
        return "HiDreamImageTransformer2DModel"
    return None


# FU-020: Align Your Steps (AYS) — NVIDIA's hand-optimised 10-step
# timestep schedules for SD1.5, SDXL and SVD. At 7-10 steps the AYS
# arrays preserve substantially more detail than DPM++ 2M Karras —
# the user study cited in the paper shows a 2× preference at low step
# counts. Numbers are the *timesteps* (not sigmas) the scheduler
# should sample at, not the count itself; passing them via
# ``pipeline(timesteps=...)`` overrides the standard
# ``num_inference_steps`` path.
#
# Reference: NVIDIA AYS project page,
# https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/
_AYS_TIMESTEPS: dict[str, list[int]] = {
    "sd15": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24],
    "sdxl": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13],
    # SVD reserved for the video runtime; not exposed in the image
    # sampler dropdown today but registered here so the same
    # ``_ays_family`` token works if/when we surface it on a video
    # path.
    "svd":  [999, 963, 911, 833, 720, 562, 387, 219, 90, 8],
}


# Maps a stable UI-facing sampler id to (diffusers scheduler class name,
# optional from_config kwargs). The class is imported lazily from
# ``diffusers`` so the runtime doesn't pay the import cost unless a user
# actually picks a non-default sampler. Kwargs let us configure the
# Karras/SDE variants without adding separate classes.
#
# The ``_ays_family`` key is a private marker consumed by
# ``_apply_scheduler`` — when present it pops out of the kwargs (so it
# never reaches diffusers' ``from_config``) and stashes the matching
# AYS timestep array on the pipeline for ``_build_pipeline_kwargs`` to
# pass via the ``timesteps=`` arg.
_SAMPLER_REGISTRY: dict[str, tuple[str, dict[str, Any]]] = {
    "dpmpp_2m": ("DPMSolverMultistepScheduler", {}),
    "dpmpp_2m_karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "dpmpp_sde": ("DPMSolverSinglestepScheduler", {}),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
    "ays_dpmpp_2m_sd15": ("DPMSolverMultistepScheduler", {"_ays_family": "sd15"}),
    "ays_dpmpp_2m_sdxl": ("DPMSolverMultistepScheduler", {"_ays_family": "sdxl"}),
}


def _apply_scheduler(pipeline: Any, sampler_id: str | None) -> str | None:
    """Swap ``pipeline.scheduler`` to the sampler chosen by the user.

    Returns a short human-readable note on what was applied (or why
    nothing was), to surface in ``GeneratedImage.runtimeNote``. Silent
    failure modes (missing scheduler class on old diffusers, pipeline
    with no ``scheduler`` attribute) fall back to the model default.

    FU-020: when the registry entry includes the ``_ays_family`` private
    marker, the matching AYS timestep array is stashed on
    ``pipeline._chaosengine_ays_timesteps`` so
    ``_build_pipeline_kwargs`` can pass it via the ``timesteps=`` arg
    instead of the usual ``num_inference_steps``.
    """
    if not sampler_id:
        return None
    entry = _SAMPLER_REGISTRY.get(sampler_id)
    if entry is None:
        return f"Unknown sampler '{sampler_id}' — using model default."
    if not hasattr(pipeline, "scheduler") or pipeline.scheduler is None:
        return None
    class_name, registry_kwargs = entry
    try:
        import diffusers  # type: ignore
    except Exception:
        return None
    scheduler_cls = getattr(diffusers, class_name, None)
    if scheduler_cls is None:
        return f"Sampler '{sampler_id}' not available in installed diffusers."
    # Pop private markers (e.g. ``_ays_family``) before passing to
    # ``from_config`` — diffusers rejects unknown kwargs.
    extra_kwargs = dict(registry_kwargs)
    ays_family = extra_kwargs.pop("_ays_family", None)
    try:
        pipeline.scheduler = scheduler_cls.from_config(
            pipeline.scheduler.config, **extra_kwargs,
        )
    except Exception as exc:
        return f"Sampler swap to '{sampler_id}' failed: {type(exc).__name__}. Using model default."
    if ays_family:
        timesteps = _AYS_TIMESTEPS.get(ays_family)
        if timesteps:
            try:
                pipeline._chaosengine_ays_timesteps = list(timesteps)  # type: ignore[attr-defined]
            except Exception:
                # Pipeline objects are usually attribute-friendly, but
                # if a future diffusers version locks slots we swallow
                # and keep the swap-only behaviour rather than failing
                # the run.
                pass
        return f"Sampler: {sampler_id} ({len(timesteps or [])}-step AYS)"
    # Clear any stale stash from a previous AYS-using generate so a
    # later non-AYS run doesn't reuse the timestep array.
    if hasattr(pipeline, "_chaosengine_ays_timesteps"):
        try:
            delattr(pipeline, "_chaosengine_ays_timesteps")
        except Exception:
            pass
    return f"Sampler: {sampler_id}"


def _stable_hash(value: str) -> int:
    acc = 0
    for index, char in enumerate(value):
        acc = (acc + ord(char) * (index + 17)) % 0xFFFFFF
    return acc


def _resolve_base_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return secrets.randbelow(MAX_IMAGE_SEED + 1)


def _mix_channel(left: int, right: int, factor: float) -> int:
    return max(0, min(255, round((left * (1 - factor)) + (right * factor))))


def _rgb_from_hsv(hue: int, saturation: float, value: float) -> tuple[int, int, int]:
    red, green, blue = hsv_to_rgb((hue % 360) / 360.0, saturation, value)
    return (round(red * 255), round(green * 255), round(blue * 255))


@dataclass(frozen=True)
class ImageRuntimeStatus:
    activeEngine: str
    realGenerationAvailable: bool
    message: str
    device: str | None = None
    # ``expectedDevice`` is the device we'll ask torch to use on the
    # next Generate click, computed from nvidia-smi + find_spec without
    # importing torch. Lets the UI show "will use cuda" before any
    # model has actually been loaded. Kept separate from ``device`` so
    # consumers can distinguish "expected at load time" from "actually
    # bound right now".
    expectedDevice: str | None = None
    pythonExecutable: str | None = None
    missingDependencies: list[str] = field(default_factory=list)
    loadedModelRepo: str | None = None
    # Total memory available to the inference device, in GB. Populated via
    # ``backend_service.helpers.gpu.get_device_vram_total_gb`` — NVIDIA VRAM
    # from nvidia-smi on CUDA, unified memory from sysctl on Apple Silicon,
    # system RAM on CPU Linux/Windows. Used by the frontend image-safety
    # heuristic (``assessImageGenerationSafety``) to scale its memory-
    # budget thresholds — a 64 GB M4 Max tolerates far more than a 16 GB
    # base M2. ``None`` means detection failed; the frontend falls back
    # to MPS-strict defaults.
    deviceMemoryGb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageGenerationConfig:
    modelId: str
    modelName: str
    repo: str
    prompt: str
    negativePrompt: str
    width: int
    height: int
    steps: int
    guidance: float
    batchSize: int
    seed: int | None = None
    qualityPreset: str | None = None
    sampler: str | None = None
    # GGUF quantization: when set, the transformer is loaded from a single
    # .gguf file (e.g. city96/FLUX.1-dev-gguf / flux1-dev-Q4_K_M.gguf) while
    # the VAE and text encoders come from the base ``repo`` snapshot. The
    # pipeline cache keys on (repo, ggufFile) so multiple quant levels of
    # the same model can coexist without stomping on each other.
    ggufRepo: str | None = None
    ggufFile: str | None = None
    # Runtime selector. Default (None / "diffusers") uses the
    # cross-platform diffusers pipeline; "mflux" routes to the native
    # Apple Silicon MLX path for FLUX, which is noticeably faster on
    # M-series Macs and avoids MPS fp16 corner cases.
    runtime: str | None = None
    # Optional diffusion cache strategy id, e.g. "teacache". When set to a
    # strategy that reports ``applies_to()`` including "image", the engine
    # calls the strategy's ``apply_diffusers_hook`` before the first pipeline
    # forward. Unknown / inapplicable ids are ignored quietly — the caller
    # sees the same result as not passing anything.
    cacheStrategy: str | None = None
    # Threshold knob for TeaCache-style rel-L1 caches. ``None`` means the
    # strategy's default (0.4 for TeaCache → ~1.8× speedup). See
    # ``TeaCacheStrategy.recommended_thresholds()`` for presets.
    cacheRelL1Thresh: float | None = None
    # FU-021: CFG decay schedule, mirroring the video runtime knob. When
    # True and the model is flow-match (FLUX/SD3/Qwen-Image/Sana/HiDream),
    # the engine ramps ``guidance_scale`` linearly from the user's
    # setting at step 0 toward 1.5 (the floor that keeps
    # ``do_classifier_free_guidance`` True end-to-end). Default off:
    # image users typically want consistent CFG; turning on the knob is
    # opt-in. Non-flow-match repos (SD1.5/SDXL) ignore the flag because
    # CFG decay on UNet-based ε-prediction pipelines doesn't carry the
    # same oversaturation benefit.
    cfgDecay: bool = False
    # FU-018: TAESD / TAEHV preview-decode VAE swap. Preview-only quality
    # knob — when True the engine swaps ``pipeline.vae`` for the matching
    # tiny VAE before the first denoise so each step decodes in a fraction
    # of the wall-time. Final output goes through the same fast VAE; users
    # trade fidelity for iteration speed. Default off.
    previewVae: bool = False
    # FU-019 distill LoRAs: when the catalog variant pins a LoRA
    # (Hyper-SD FLUX, alimama FLUX.1-Turbo-Alpha, lightx2v Wan
    # CausVid), the engine fuses it into the pipeline at load time so
    # subsequent generates run at the LoRA's lower step count without
    # re-loading. ``loraRepo`` is the HF repo id, ``loraFile`` is the
    # specific weight name within that repo (LoRAs commonly ship
    # multiple step variants), ``loraScale`` is the fuse strength
    # (Hyper-SD recommends 0.125, alimama Turbo wants 1.0, lightx2v
    # CausVid wants 1.0).
    loraRepo: str | None = None
    loraFile: str | None = None
    loraScale: float | None = None
    # Variant-declared step / CFG defaults. Used by
    # ``_generate_image_artifacts`` in app.py to substitute the schema
    # defaults when the user hasn't moved the sliders — distill LoRAs
    # have very different optimal points (4-8 steps, CFG 1.0-3.5)
    # than the schema defaults (24 steps, CFG 5.5).
    defaultSteps: int | None = None
    cfgOverride: float | None = None


@dataclass(frozen=True)
class GeneratedImage:
    seed: int
    bytes: bytes
    extension: str
    mimeType: str
    durationSeconds: float
    runtimeLabel: str
    runtimeNote: str | None = None


class PlaceholderImageEngine:
    runtime_label = "Placeholder image engine"

    def generate(
        self,
        config: ImageGenerationConfig,
        *,
        runtime_note: str | None = None,
    ) -> list[GeneratedImage]:
        base_seed = _resolve_base_seed(config.seed)
        duration_base = max(1.2, (config.steps / 14.0) + 1.5)
        return [
            GeneratedImage(
                seed=base_seed + index,
                bytes=self._render_image_bytes(config, base_seed + index),
                extension="svg",
                mimeType="image/svg+xml",
                durationSeconds=round(duration_base + index * 0.35, 1),
                runtimeLabel=self.runtime_label,
                runtimeNote=runtime_note,
            )
            for index in range(config.batchSize)
        ]

    def _render_image_bytes(self, config: ImageGenerationConfig, seed: int) -> bytes:
        width = max(256, config.width)
        height = max(256, config.height)
        hash_value = _stable_hash(f"{config.modelName}:{config.prompt}:{seed}")
        hue_a = hash_value % 360
        hue_b = (hash_value * 7) % 360
        hue_c = (hash_value * 13) % 360
        base_a = _rgb_from_hsv(hue_a, 0.72, 0.94)
        base_b = _rgb_from_hsv(hue_b, 0.68, 0.62)
        accent = _rgb_from_hsv(hue_c, 0.55, 0.88)
        title_y = max(40, height - 170)
        prompt_lines = textwrap.wrap(
            config.prompt.strip() or "Generated image preview",
            width=max(24, width // 18),
        )[:3]
        footer = f"seed {seed} | {width}x{height} | {config.steps} steps"

        def _rgb(rgb: tuple[int, int, int]) -> str:
            return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

        def _rgba(rgb: tuple[int, int, int], alpha: float) -> str:
            safe_alpha = max(0.0, min(1.0, alpha))
            return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {safe_alpha:.3f})"

        def _escape(text: str) -> str:
            return (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        line_markup = []
        for index in range(7):
            offset = ((seed >> (index * 2)) % 120) - 30
            y1 = height * (0.12 + index * 0.1)
            y2 = height * (0.06 + index * 0.1) + offset
            line_markup.append(
                f'<line x1="{width * 0.05:.1f}" y1="{y1:.1f}" '
                f'x2="{width * 0.95:.1f}" y2="{y2:.1f}" '
                f'stroke="rgba(255,255,255,0.120)" stroke-width="{max(1, round(width * 0.004))}" '
                'stroke-linecap="round" />'
            )

        prompt_markup = []
        for index, line in enumerate(prompt_lines):
            prompt_markup.append(
                f'<text x="48" y="{title_y + 34 + (index * 22)}" '
                'font-size="16" fill="rgba(232,239,255,0.92)" '
                'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">'
                f"{_escape(line)}</text>"
            )

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{_rgb(base_a)}" />
      <stop offset="100%" stop-color="{_rgb(base_b)}" />
    </linearGradient>
    <radialGradient id="glowA" cx="22%" cy="24%" r="38%">
      <stop offset="0%" stop-color="{_rgba(accent, 0.65)}" />
      <stop offset="100%" stop-color="{_rgba(accent, 0.0)}" />
    </radialGradient>
    <radialGradient id="glowB" cx="74%" cy="58%" r="42%">
      <stop offset="0%" stop-color="rgba(255,255,255,0.22)" />
      <stop offset="100%" stop-color="rgba(255,255,255,0.0)" />
    </radialGradient>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#bg)" />
  <rect width="{width}" height="{height}" fill="rgba(7, 10, 18, 0.10)" />
  <circle cx="{width * 0.24:.1f}" cy="{height * 0.26:.1f}" r="{min(width, height) * 0.22:.1f}" fill="url(#glowA)" />
  <circle cx="{width * 0.74:.1f}" cy="{height * 0.58:.1f}" r="{min(width, height) * 0.26:.1f}" fill="url(#glowB)" />
  {''.join(line_markup)}
  <rect x="28" y="{max(24, height - 180)}" width="{max(140, width - 56)}" height="{min(152, height - max(24, height - 180) - 28)}"
        rx="28" fill="rgba(10,14,24,0.58)" stroke="rgba(255,255,255,0.16)" />
  <text x="48" y="{title_y}" font-size="18" font-weight="700" fill="rgba(255,255,255,0.96)"
        font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif">{_escape(config.modelName)}</text>
  {''.join(prompt_markup)}
  <text x="48" y="{height - 48}" font-size="14" fill="rgba(205,214,232,0.82)"
        font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">{_escape(footer)}</text>
</svg>
"""
        return svg.encode("utf-8")


class DiffusersTextToImageEngine:
    runtime_label = "Diffusers local engine"

    def __init__(self) -> None:
        self._lock = RLock()
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._loaded_repo: str | None = None
        self._loaded_path: str | None = None
        self._loaded_variant_key: str | None = None
        self._device: str | None = None
        # FU-017 / FU-019 / FU-016: notes accumulated during pipeline load
        # (VAE swap, LoRA fuse, attention backend). Surfaced as part of
        # ``runtimeNote`` on every GeneratedImage produced by the loaded
        # pipeline so the user sees what was applied without polling
        # capabilities mid-batch. Reset on each pipeline load.
        self._load_notes: list[str] = []

    def probe(self) -> ImageRuntimeStatus:
        # Deliberately does NOT ``import torch`` — that would load
        # torch/lib/*.dll into the backend process handle table, and on
        # Windows those locked DLLs break /api/setup/install-gpu-bundle
        # (pip's rmtree can't remove files another process has open).
        # find_spec answers "is it installable?" without triggering the
        # import side effects. Device detection (cuda vs cpu) is deferred
        # to preload/generate where we're about to import torch anyway.
        #
        # ``invalidate_caches`` matters when the GPU bundle install has
        # finished mid-process: pip writes the new packages into the
        # extras dir (already on ``sys.path`` from process start), but
        # ``importlib`` keeps a per-finder cache of negative lookups, so
        # the find_spec calls below would still report None even though
        # the .dist-info folders are sitting on disk. Calling
        # ``invalidate_caches`` first re-walks the path entries so the
        # newly installed packages are picked up without a process
        # restart.
        importlib.invalidate_caches()
        missing = [
            package
            for package, module_name in (
                ("diffusers", "diffusers"),
                ("torch", "torch"),
                ("accelerate", "accelerate"),
                ("huggingface_hub", "huggingface_hub"),
                ("pillow", "PIL"),
            )
            if importlib.util.find_spec(module_name) is None
        ]
        if missing:
            message = (
                "Install the GPU image runtime packages to enable real local generation. "
                "Click the 'Install GPU runtime' button above."
            )
            return ImageRuntimeStatus(
                activeEngine="placeholder",
                realGenerationAvailable=False,
                missingDependencies=missing,
                pythonExecutable=_resolve_image_python(),
                message=message,
                loadedModelRepo=self._loaded_repo,
            )

        message = (
            "Real local generation is available. Download an image model locally, then Image Studio "
            "will use the diffusers runtime instead of the placeholder engine."
        )
        device_memory_gb: float | None = None
        try:
            from backend_service.helpers.gpu import get_device_vram_total_gb
            device_memory_gb = get_device_vram_total_gb()
        except Exception:
            device_memory_gb = None
        return ImageRuntimeStatus(
            activeEngine="diffusers",
            realGenerationAvailable=True,
            # ``device`` is the *currently-loaded* model's device, or None
            # if no model is loaded. We no longer speculatively import
            # torch just to report cuda/mps/cpu availability in the empty
            # case — users find out on first Generate which is cheap.
            device=self._device,
            expectedDevice=_guess_expected_device(),
            pythonExecutable=_resolve_image_python(),
            message=message,
            loadedModelRepo=self._loaded_repo,
            deviceMemoryGb=device_memory_gb,
        )

    def generate(self, config: ImageGenerationConfig) -> list[GeneratedImage]:
        # Begin reporting progress before we touch the pipeline. ``_ensure_pipeline``
        # publishes its own ``loading`` phase if it actually has to materialise
        # the pipeline, but we still want a tracker entry from the moment the
        # request lands so the UI's first poll has something to render.
        IMAGE_PROGRESS.begin(
            run_label=self._format_run_label(config),
            total_steps=max(1, int(config.steps)),
            phase=PHASE_LOADING,
            message=f"Preparing {config.modelName}",
        )
        try:
            pipeline = self._ensure_pipeline(
                config.repo,
                gguf_repo=config.ggufRepo,
                gguf_file=config.ggufFile,
                lora_repo=config.loraRepo,
                lora_file=config.loraFile,
                lora_scale=config.loraScale,
                preview_vae=config.previewVae,
            )
            # Early-cancel check: the load phase is blocking (from_pretrained
            # is a C-extension call we can't interrupt), so if the user hit
            # Cancel during it we catch up here and bail before kicking off
            # the T5/VAE passes.
            if IMAGE_PROGRESS.is_cancelled():
                raise GenerationCancelled("Image generation cancelled by user")
            # Apply the user's sampler choice (SD1.5/SDXL only). Flow-matching
            # models (FLUX, SD3, Qwen-Image, Sana, HiDream) ship locked
            # schedulers — silently ignore the sampler there rather than
            # producing noise. The returned note lands on GeneratedImage
            # so users see which sampler was applied.
            sampler_note: str | None = None
            if config.sampler and not _is_flow_matching_repo(config.repo):
                sampler_note = _apply_scheduler(pipeline, config.sampler)
            torch = self._torch
            if torch is None:
                raise RuntimeError("PyTorch was not initialised for the diffusers runtime.")
            IMAGE_PROGRESS.set_phase(PHASE_ENCODING, message="Encoding prompt")
            generator_device = "cpu" if self._device == "mps" else (self._device or "cpu")
            base_seed = _resolve_base_seed(config.seed)
            generators = [
                torch.Generator(device=generator_device).manual_seed(base_seed + index)
                for index in range(config.batchSize)
            ]

            kwargs = self._build_pipeline_kwargs(config, generators if len(generators) > 1 else generators[0])
            lowered_repo = config.repo.lower()
            if "flux" in lowered_repo:
                kwargs.pop("negative_prompt", None)
                kwargs["num_inference_steps"] = min(config.steps, 8)
            if "turbo" in lowered_repo:
                kwargs["num_inference_steps"] = min(config.steps, 8)
                kwargs["guidance_scale"] = min(config.guidance, 2.5)

            # Wire the diffusers per-step callback so the UI sees the bar move
            # in lockstep with denoising, which is the bulk of the wall time on
            # most models. ``callback_on_step_end`` is the non-deprecated name
            # in modern diffusers (>=0.27); some pipelines also accept the
            # legacy ``callback`` arg, but we prefer the new one.
            # AYS path passes ``timesteps=[...]`` instead of
            # ``num_inference_steps`` — derive the step count from the
            # array length so the progress bar / decay schedule still
            # report the right total.
            if isinstance(kwargs.get("timesteps"), list):
                total_steps = len(kwargs["timesteps"])
            else:
                total_steps = int(kwargs.get("num_inference_steps", config.steps) or config.steps)
            IMAGE_PROGRESS.set_phase(
                PHASE_DIFFUSING,
                message=self._diffuse_message(config),
            )
            # Re-publish the totalSteps in case ``num_inference_steps`` was
            # clamped above (Flux/Turbo cap at 8).
            IMAGE_PROGRESS.set_step(0, total=max(1, total_steps))

            # TeaCache / other diffusion cache strategies hook here: the
            # pipeline is loaded, num_inference_steps is final, and we
            # haven't kicked off the forward pass yet. If the selected
            # strategy isn't applicable to images or hasn't landed a patch
            # for this pipeline yet we swallow NotImplementedError and run
            # the stock pipeline — the UI surfaces the "Scaffold" badge so
            # users know why speedup didn't appear.
            from cache_compression import apply_diffusion_cache_strategy

            cache_note = apply_diffusion_cache_strategy(
                pipeline,
                strategy_id=config.cacheStrategy,
                num_inference_steps=total_steps,
                rel_l1_thresh=config.cacheRelL1Thresh,
                domain="image",
            )
            if cache_note:
                # Surface for log only; sampler_note already owns the
                # runtime_note slot on GeneratedImage. Adding cache noise
                # to every image's metadata would flood the gallery UI.
                pass

            # FU-021: CFG decay schedule for flow-match image pipelines.
            # Same shape as the video-runtime ramp — linear from initial
            # guidance to a 1.5 floor that keeps
            # ``do_classifier_free_guidance`` True for the entire schedule
            # (dropping below 1.0 mid-loop swaps the pipeline from
            # 2-batch to 1-batch shape and produces shape-mismatch
            # crashes; 1.5 is the documented floor we use on video).
            # Gated to flow-match so SD1.5 / SDXL stay on constant CFG.
            decay_floor = 1.5
            initial_guidance = float(kwargs.get("guidance_scale", config.guidance) or config.guidance)
            decay_active = (
                config.cfgDecay
                and _is_flow_matching_repo(config.repo)
                and total_steps > 1
                and initial_guidance > decay_floor
            )

            def _on_step_end(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]):
                # Diffusers calls this *after* step ``step`` finishes, so step
                # 0 means "one step done". Convert to the 1-indexed value the
                # UI wants to display.
                IMAGE_PROGRESS.set_step(step + 1, total=max(1, total_steps))
                # Cooperative cancel: the Cancel button on the modal sets
                # IMAGE_PROGRESS.request_cancel(); we honor it at the next
                # step boundary by setting ``_interrupt``, which makes
                # diffusers stop the denoising loop cleanly at the next
                # iteration. We also raise here so the outer handler can
                # see a cancellation came from the user (not a pipeline
                # crash) and return the right response.
                if IMAGE_PROGRESS.is_cancelled():
                    try:
                        _pipeline._interrupt = True
                    except Exception:
                        pass
                    raise GenerationCancelled("Image generation cancelled by user")
                if decay_active:
                    next_step = step + 1
                    progress = min(1.0, next_step / max(1, total_steps - 1))
                    next_scale = (
                        initial_guidance * (1.0 - progress)
                        + decay_floor * progress
                    )
                    try:
                        _pipeline.guidance_scale = float(next_scale)
                    except Exception:
                        pass
                return callback_kwargs

            kwargs.setdefault("callback_on_step_end", _on_step_end)

            started = time.perf_counter()
            try:
                result = pipeline(**kwargs)
            except TypeError as exc:
                # Older diffusers versions don't accept ``callback_on_step_end``
                # — drop it and retry once before bubbling the original error.
                if "callback_on_step_end" in str(exc):
                    kwargs.pop("callback_on_step_end", None)
                    try:
                        result = pipeline(**kwargs)
                    except TypeError:
                        kwargs.pop("negative_prompt", None)
                        result = pipeline(**kwargs)
                else:
                    kwargs.pop("negative_prompt", None)
                    result = pipeline(**kwargs)
            elapsed = max(0.1, time.perf_counter() - started)

            IMAGE_PROGRESS.set_phase(PHASE_DECODING, message="Decoding pixels")

            artifacts: list[GeneratedImage] = []
            for index, image in enumerate(getattr(result, "images", []) or []):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                if image.getbbox() is None:
                    raise RuntimeError(
                        "The image runtime returned an all-black frame instead of a real image. "
                        f"Model: {config.repo}. Device: {self._device or 'cpu'}. "
                        "Try restarting the backend and generating again. If this keeps happening on Apple Silicon, "
                        "the model likely needs a safer precision path."
                    )
                buffer = io.BytesIO()
                image.save(buffer, format="PNG", optimize=True)
                # Combine all per-load notes (VAE swap, LoRA fuse,
                # attention backend) with the per-generate sampler note.
                # Joined with " · " so the UI can show a single line.
                note_parts: list[str] = list(self._load_notes)
                if sampler_note:
                    note_parts.append(sampler_note)
                if cache_note:
                    note_parts.append(cache_note)
                runtime_note = " · ".join(note_parts) if note_parts else None
                artifacts.append(
                    GeneratedImage(
                        seed=base_seed + index,
                        bytes=buffer.getvalue(),
                        extension="png",
                        mimeType="image/png",
                        durationSeconds=round(elapsed / max(1, config.batchSize), 1),
                        runtimeLabel=f"{self.runtime_label} ({self._device or 'cpu'})",
                        runtimeNote=runtime_note,
                    )
                )
            if not artifacts:
                raise RuntimeError("Diffusers returned no images.")
            IMAGE_PROGRESS.set_phase(PHASE_SAVING, message="Saving to gallery")
            return artifacts
        finally:
            IMAGE_PROGRESS.finish()

    def _diffuse_message(self, config: ImageGenerationConfig) -> str:
        if config.batchSize > 1:
            return f"Diffusing {config.batchSize} images"
        return "Diffusing image"

    def _format_run_label(self, config: ImageGenerationConfig) -> str:
        return f"{config.modelName} · {config.width}x{config.height}"

    def preload(self, repo: str) -> ImageRuntimeStatus:
        self._ensure_pipeline(repo)
        return self.probe()

    def unload(self, repo: str | None = None) -> ImageRuntimeStatus:
        with self._lock:
            if repo and self._loaded_repo != repo:
                return self.probe()
            self._release_pipeline()
            return self.probe()

    def _ensure_pipeline(
        self,
        repo: str,
        gguf_repo: str | None = None,
        gguf_file: str | None = None,
        lora_repo: str | None = None,
        lora_file: str | None = None,
        lora_scale: float | None = None,
        preview_vae: bool = False,
    ) -> Any:
        with self._lock:
            # Variant key folds LoRA identity in too — switching LoRAs
            # on the same base repo must rebuild the pipeline because
            # ``fuse_lora`` mutates the transformer weights in place.
            # ``preview_vae`` joins the same key set so toggling the
            # FU-018 preview-decode knob triggers a clean rebuild.
            variant_parts = [repo]
            if gguf_file:
                variant_parts.append(f"gguf={gguf_file}")
            if lora_repo and lora_file:
                variant_parts.append(f"lora={lora_repo}/{lora_file}@{lora_scale or 1.0}")
            if preview_vae:
                variant_parts.append("preview_vae")
            variant_key = "::".join(variant_parts)
            if self._pipeline is not None and self._loaded_variant_key == variant_key:
                return self._pipeline

            # Loading a pipeline can take 10-60s on cold disk. Surface that
            # explicitly to the UI so the progress bar stops sitting at 0%
            # while we read 5GB of weights from the SSD.
            IMAGE_PROGRESS.set_phase(PHASE_LOADING, message=f"Loading {repo}")

            if self._pipeline is not None and self._loaded_variant_key != variant_key:
                self._release_pipeline()

            import torch  # type: ignore
            from diffusers import AutoPipelineForText2Image  # type: ignore
            from huggingface_hub import snapshot_download  # type: ignore

            local_path = snapshot_download(
                repo_id=repo,
                local_files_only=True,
                resume_download=True,
            )
            local_root = Path(local_path)
            validation_error = validate_local_diffusers_snapshot(local_root, repo)
            if validation_error is not None:
                raise RuntimeError(validation_error)
            detected_device = self._detect_device(torch)
            device = self._preferred_execution_device(repo, detected_device)
            # FU-017: probe the SDXL fp16-fix VAE before deciding dtype so
            # SDXL on MPS can stay on fp16 when the fix snapshot is cached.
            # Probe only fires for SDXL repos on devices that actually
            # benefit (MPS / CUDA) — CPU stays on fp32 regardless.
            sdxl_vae_fix_path: str | None = None
            if _is_sdxl_repo(repo) and device in ("mps", "cuda"):
                sdxl_vae_fix_path = _locate_sdxl_vae_fix_snapshot()
            dtype = self._preferred_torch_dtype(
                torch, repo, device,
                sdxl_vae_fix_available=sdxl_vae_fix_path is not None,
            )
            use_cpu_offload = self._should_use_model_cpu_offload(repo, device)
            # Clear load notes on each pipeline (re)load so stale entries
            # from a previously-loaded model don't bleed into new outputs.
            self._load_notes = []

            # Three transformer-loading strategies, in preference order:
            #   1. GGUF (cross-platform, any quant level the user picked)
            #   2. NF4 via bitsandbytes (CUDA-only, FLUX-only, ~7 GB)
            #   3. Full-precision transformer bundled into the base pipeline
            # GGUF wins when the variant asked for it because the user's
            # quant choice is explicit; NF4 remains the default for FLUX
            # on CUDA when no GGUF file was specified.
            pipeline_kwargs: dict[str, Any] = {}
            gguf_note: str | None = None
            if gguf_file:
                IMAGE_PROGRESS.set_phase(
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
                    IMAGE_PROGRESS.set_phase(PHASE_LOADING, message=gguf_note)
            if (
                "transformer" not in pipeline_kwargs
                and device == "mps"
                and _is_flux_repo(repo)
            ):
                # MPS has no bitsandbytes/NF4 path — int8wo is the
                # cross-platform fallback that still halves FLUX's
                # memory footprint on Apple Silicon.
                IMAGE_PROGRESS.set_phase(
                    PHASE_LOADING,
                    message=f"Quantizing {repo} transformer to int8",
                )
                quantized_transformer, note = self._try_load_int8wo_flux_transformer(
                    local_path, torch,
                )
                if quantized_transformer is not None:
                    pipeline_kwargs["transformer"] = quantized_transformer
                if note:
                    IMAGE_PROGRESS.set_phase(PHASE_LOADING, message=note)
            if "transformer" not in pipeline_kwargs and use_cpu_offload:
                IMAGE_PROGRESS.set_phase(
                    PHASE_LOADING, message=f"Quantizing {repo} transformer to NF4",
                )
                quantized_transformer, note = self._try_load_nf4_flux_transformer(
                    local_path, torch,
                )
                if quantized_transformer is not None:
                    pipeline_kwargs["transformer"] = quantized_transformer
                if note:
                    IMAGE_PROGRESS.set_phase(PHASE_LOADING, message=note)

            pipeline = AutoPipelineForText2Image.from_pretrained(
                local_path,
                torch_dtype=dtype,
                local_files_only=True,
                **pipeline_kwargs,
            )
            # The safety checker adds extra vision-model dependencies and can
            # fail on tiny or oddly shaped test pipelines. For the local app
            # MVP we prioritize generation reliability over post-filtering.
            if hasattr(pipeline, "safety_checker"):
                pipeline.safety_checker = None
            if hasattr(pipeline, "feature_extractor"):
                pipeline.feature_extractor = None
            if hasattr(pipeline, "requires_safety_checker"):
                pipeline.requires_safety_checker = False
            if hasattr(pipeline, "set_progress_bar_config"):
                pipeline.set_progress_bar_config(disable=True)

            # FU-017: swap in madebyollin's SDXL VAE fp16-fix when the
            # snapshot is cached. The pipeline already loaded with fp16
            # weights (decided above) so the VAE swap is the load-bearing
            # piece — without it the stock SDXL VAE silently NaN-overflows
            # on the fp16 sigmoid and outputs black images on MPS / consumer
            # CUDA. Failure modes (corrupt snapshot, dtype mismatch) fall
            # back to the original VAE so the user still gets *some* image.
            if sdxl_vae_fix_path and getattr(pipeline, "vae", None) is not None:
                try:
                    from diffusers import AutoencoderKL  # type: ignore
                    fix_vae = AutoencoderKL.from_pretrained(
                        sdxl_vae_fix_path,
                        torch_dtype=torch.float16,
                        local_files_only=True,
                    )
                    pipeline.vae = fix_vae
                    self._load_notes.append("VAE: SDXL fp16-fix")
                except Exception as exc:  # noqa: BLE001 — fall back to stock VAE
                    self._load_notes.append(
                        f"SDXL VAE fp16-fix swap failed ({type(exc).__name__}); using stock VAE."
                    )

            # FU-016: SageAttention CUDA backend. No-op on MPS / CPU and
            # when the pipeline lacks ``transformer.set_attention_backend``.
            # Stacks multiplicatively with FBCache. Must run *before*
            # placement so the kernel selection is locked in before the
            # first forward pass.
            try:
                from backend_service.helpers.attention_backend import (
                    maybe_apply_sage_attention,
                )
                sage_note = maybe_apply_sage_attention(pipeline)
                if sage_note:
                    self._load_notes.append(sage_note)
            except Exception:
                # Helper is wrapped in its own try/except; any leakage
                # here is a bug in the helper, not a runtime concern.
                pass

            # FU-018: TAESD preview-decode VAE swap. No-op when toggle
            # is off or no preview VAE is mapped for this repo. Runs
            # before LoRA fuse so the LoRA's adapter modules don't trip
            # the VAE swap (they target the transformer, not the VAE,
            # but ordering keeps the swap close to other VAE-touching
            # code like the SDXL fp16-fix above).
            try:
                from backend_service.helpers.preview_vae import (
                    maybe_apply_preview_vae,
                )
                preview_note = maybe_apply_preview_vae(
                    pipeline, repo=repo, enabled=preview_vae
                )
                if preview_note:
                    self._load_notes.append(preview_note)
            except Exception:
                pass

            # FU-019: distill LoRAs (Hyper-SD FLUX, alimama FLUX.1-Turbo,
            # lightx2v Wan CausVid). Load + fuse at pipeline build time
            # so subsequent ``pipeline(...)`` calls run with the LoRA
            # baked into the transformer — no per-generate fuse cost.
            # ``unload_lora_weights`` after fuse drops the un-fused
            # state dict from RAM (the fused weights live in the
            # transformer itself).
            if lora_repo and lora_file:
                try:
                    pipeline.load_lora_weights(
                        lora_repo,
                        weight_name=lora_file,
                        local_files_only=True,
                    )
                    effective_scale = (
                        float(lora_scale) if lora_scale is not None else 1.0
                    )
                    pipeline.fuse_lora(lora_scale=effective_scale)
                    try:
                        pipeline.unload_lora_weights()
                    except Exception:
                        # Best-effort cleanup — older diffusers don't
                        # always succeed at unloading after fuse, and
                        # the fused transformer is correct either way.
                        pass
                    self._load_notes.append(
                        f"LoRA: {lora_repo}/{lora_file} @ scale {effective_scale:.3f}"
                    )
                except Exception as exc:  # noqa: BLE001 — non-fatal
                    self._load_notes.append(
                        f"LoRA load failed ({type(exc).__name__}: {exc}). "
                        "Pipeline continuing without LoRA."
                    )

            if use_cpu_offload:
                # Diffusers' stock recipe for FLUX on <32 GB VRAM: keep only
                # the active component (T5, then transformer, then VAE) on
                # GPU, transferring at component boundaries. Do NOT combine
                # with attention/VAE slicing or .to(device) — slicing issues
                # many tiny kernel launches that saturate PCIe when the
                # active weights are already being DMA'd in, and .to(device)
                # would pin all 33 GB of FLUX weights in VRAM at once
                # (exceeds even a 4090) causing fallback-to-pagefile thrash.
                # Real-world signature of doing it wrong: GPU at 97% util
                # but step 0/8 never completing.
                pipeline.enable_model_cpu_offload()
            else:
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing()
                vae = getattr(pipeline, "vae", None)
                if vae is not None and hasattr(vae, "enable_slicing"):
                    vae.enable_slicing()
                # VAE tiling is a no-op at low resolution (diffusers only
                # activates it when the latent exceeds the VAE's sample_size),
                # so enabling it unconditionally costs nothing at 1024px but
                # prevents the VAE decode from OOM-ing at 1536/2048px on
                # MPS / 8-12 GB CUDA cards. Same pattern as video_runtime.
                if vae is not None and hasattr(vae, "enable_tiling"):
                    vae.enable_tiling()
                if device != "cpu":
                    pipeline = pipeline.to(device)

            self._pipeline = pipeline
            self._torch = torch
            self._loaded_repo = repo
            self._loaded_path = local_path
            self._loaded_variant_key = variant_key
            self._device = device
            return pipeline

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

    def _preferred_torch_dtype(
        self,
        torch: Any,
        repo: str,
        device: str,
        sdxl_vae_fix_available: bool = False,
    ) -> Any:
        if device == "cuda":
            # FLUX was trained and validated in bfloat16. Loading it as
            # float16 produces slightly off saturations and occasional
            # NaN-propagation on long prompts — not catastrophic, but the
            # official Black Forest recipe is bfloat16 and we should match
            # it so output quality is on-spec.
            if _is_flux_repo(repo):
                return torch.bfloat16
            return torch.float16
        if device == "mps":
            lowered_repo = repo.lower()
            # SDXL / Stable Diffusion on MPS can silently decode to black
            # images in fp16 due to the stock SDXL VAE overflowing the
            # fp16 sigmoid. FU-017: when madebyollin/sdxl-vae-fp16-fix is
            # cached locally we swap that VAE in and stay on fp16 (≈2×
            # faster than fp32). Without the fix snapshot we keep the
            # safe fp32 fallback so users still get correct images.
            if any(token in lowered_repo for token in ("stable-diffusion", "sdxl", "sd_xl")):
                if sdxl_vae_fix_available and _is_sdxl_repo(repo):
                    return torch.float16
                return torch.float32
            return torch.float16
        return torch.float32

    def _preferred_execution_device(self, repo: str, detected_device: str) -> str:
        lowered_repo = repo.lower()
        # Qwen-Image's official quick start uses CUDA+bfloat16, otherwise CPU+float32.
        # On Apple MPS, users report black outputs with the naive fp16 path, so prefer
        # the safer CPU execution path instead of silently returning placeholder frames.
        if detected_device == "mps" and "qwen-image" in lowered_repo:
            return "cpu"
        return detected_device

    def _try_load_nf4_flux_transformer(
        self, local_path: str, torch: Any,
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
            # Any error here (missing subfolder, CUDA kernel mismatch,
            # bitsandbytes CPU-only wheel) falls back to the unquantized
            # path rather than breaking image generation entirely.
            return None, (
                f"NF4 quantization failed ({type(exc).__name__}: {exc}) — "
                "falling back to bf16 transformer (slower on <32 GB GPUs)."
            )

    def _try_load_int8wo_flux_transformer(
        self, local_path: str, torch: Any,
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

    def _try_load_gguf_transformer(
        self,
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
        except ImportError:
            return None, (
                "Installed diffusers doesn't expose GGUFQuantizationConfig. "
                "Upgrade diffusers via the Setup page to use GGUF variants."
            )

        # Pick the transformer class from the base repo. Most flow-matching
        # image models expose a dedicated DiT class; for SD1.5/SDXL the
        # GGUF community uses the UNet path which we don't support here —
        # those pipelines stay on the standard loader.
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

    def _should_use_model_cpu_offload(self, repo: str, device: str) -> bool:
        """True when the pipeline should load via enable_model_cpu_offload().

        Currently limited to FLUX on CUDA. FLUX.1-Dev is ~24 GB transformer
        plus ~9 GB T5-XXL text encoder in bf16; on any single consumer GPU
        (≤32 GB VRAM) a plain ``pipeline.to("cuda")`` either OOMs or, worse
        on Windows, silently falls back to pinned host memory + pagefile
        and runs at PCIe speeds — which is what "GPU at 97% but step 0/8
        never completes" looks like. enable_model_cpu_offload swaps whole
        components (not layers) at module boundaries, which is the
        diffusers-recommended pattern for FLUX on consumer hardware.

        Other pipelines (SD 1.5 / SDXL / Qwen-Image) fit comfortably and
        stay on the legacy .to(device) path for best throughput.
        """
        if device != "cuda":
            return False
        return _is_flux_repo(repo)

    def _build_pipeline_kwargs(self, config: ImageGenerationConfig, generator: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "prompt": config.prompt,
            "width": config.width,
            "height": config.height,
            "num_inference_steps": config.steps,
            "guidance_scale": config.guidance,
            "num_images_per_prompt": config.batchSize,
            "generator": generator,
        }
        # FU-020: when the user picked an AYS sampler,
        # ``_apply_scheduler`` stashed the precomputed timestep array on
        # the pipeline. Diffusers accepts ``timesteps=`` as an explicit
        # override; when present it takes precedence over
        # ``num_inference_steps`` so we drop the latter to avoid the
        # "got both" warning.
        pipeline = self._pipeline
        if pipeline is not None:
            ays_timesteps = getattr(pipeline, "_chaosengine_ays_timesteps", None)
            if ays_timesteps:
                kwargs["timesteps"] = list(ays_timesteps)
                kwargs.pop("num_inference_steps", None)
        lowered_repo = config.repo.lower()
        if "qwen-image" in lowered_repo:
            kwargs.pop("guidance_scale", None)
            kwargs["true_cfg_scale"] = config.guidance
            # Qwen-Image expects a negative prompt value, even if it is intentionally blank.
            kwargs["negative_prompt"] = config.negativePrompt if config.negativePrompt else " "
            return kwargs
        if config.negativePrompt.strip():
            kwargs["negative_prompt"] = config.negativePrompt
        return kwargs

    def _detect_device(self, torch: Any) -> str:
        if getattr(torch.cuda, "is_available", lambda: False)():
            return "cuda"
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
            return "mps"
        return "cpu"


class MfluxImageEngine:
    """Native Apple Silicon FLUX runtime via the ``mflux`` package.

    Only loaded for variants that set ``runtime="mflux"`` in the
    catalog. Compared to diffusers+MPS:

      * 2-3x faster on M-series Macs (native MLX kernels vs the
        PyTorch MPS backend).
      * No fp16 black-image hazard — MLX handles precision cleanly.
      * Limited to FLUX (schnell, dev) — not a diffusers replacement.

    The engine is a quiet no-op on non-Apple platforms: ``probe()``
    reports unavailability, and the manager routes to diffusers
    automatically.
    """

    runtime_label = "mflux (MLX native)"

    def __init__(self) -> None:
        self._flux: Any = None
        self._loaded_name: str | None = None

    def probe(self) -> dict[str, Any]:
        if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
            return {
                "available": False,
                "reason": "mflux runs on Apple Silicon only.",
            }
        if importlib.util.find_spec("mflux") is None:
            return {
                "available": False,
                "reason": (
                    "mflux not installed — add it from the Setup page to "
                    "enable the native Apple Silicon FLUX runtime."
                ),
            }
        return {"available": True, "reason": None}

    def generate(self, config: ImageGenerationConfig) -> list[GeneratedImage]:
        probe = self.probe()
        if not probe["available"]:
            raise RuntimeError(probe["reason"] or "mflux unavailable")

        # Map our repo ids to the names mflux expects. Anything else
        # falls back to the diffusers path.
        flux_name = _mflux_name_for_repo(config.repo)
        if flux_name is None:
            raise RuntimeError(
                f"mflux doesn't support {config.repo} — only FLUX.1-schnell "
                "and FLUX.1-dev are available via the native MLX runtime."
            )

        import mflux  # type: ignore
        started = time.perf_counter()
        if self._flux is None or self._loaded_name != flux_name:
            self._flux = mflux.Flux1.from_name(flux_name)
            self._loaded_name = flux_name
        seed = _resolve_base_seed(config.seed)
        result_image = self._flux.generate_image(
            seed=seed,
            prompt=config.prompt,
            config=mflux.Config(
                num_inference_steps=config.steps,
                height=config.height,
                width=config.width,
                guidance=config.guidance,
            ),
        )
        elapsed = max(0.1, time.perf_counter() - started)

        pil_image = getattr(result_image, "image", result_image)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG", optimize=True)
        return [
            GeneratedImage(
                seed=seed,
                bytes=buffer.getvalue(),
                extension="png",
                mimeType="image/png",
                durationSeconds=round(elapsed, 1),
                runtimeLabel=self.runtime_label,
                runtimeNote=f"MLX native FLUX ({flux_name})",
            )
        ]


def _mflux_name_for_repo(repo: str) -> str | None:
    lowered = repo.lower()
    if "flux.1-schnell" in lowered or "flux-schnell" in lowered:
        return "schnell"
    if "flux.1-dev" in lowered or "flux-dev" in lowered:
        return "dev"
    return None


class ImageRuntimeManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._placeholder = PlaceholderImageEngine()
        self._diffusers = DiffusersTextToImageEngine()
        self._mflux = MfluxImageEngine()
        # FU-008 image subset: sd.cpp engine. Wired lazily so the import
        # cost (small) is paid only when the manager is actually
        # constructed. Engine probe is cheap; full binary check happens
        # at generate time.
        from backend_service.sdcpp_image_runtime import SdCppImageEngine
        self._sdcpp = SdCppImageEngine()

    def capabilities(self) -> dict[str, Any]:
        return self._diffusers.probe().to_dict()

    def preload(self, repo: str) -> dict[str, Any]:
        with self._lock:
            status = self._diffusers.probe()
            if not status.realGenerationAvailable:
                raise RuntimeError(status.message)
            return self._diffusers.preload(repo).to_dict()

    def unload(self, repo: str | None = None) -> dict[str, Any]:
        with self._lock:
            return self._diffusers.unload(repo).to_dict()

    def generate(self, config: ImageGenerationConfig) -> tuple[list[GeneratedImage], dict[str, Any]]:
        # mflux path: Apple Silicon native FLUX via MLX. Routed only
        # when the catalog variant declared runtime="mflux". Any
        # failure (missing package, unsupported repo, runtime error)
        # falls through to the diffusers path below so the user still
        # gets an image.
        if (config.runtime or "").lower() == "mflux":
            probe = self._mflux.probe()
            if probe.get("available"):
                try:
                    images = self._mflux.generate(config)
                    status = self._diffusers.probe().to_dict()
                    status["activeEngine"] = "mflux"
                    status["message"] = "Generated via mflux (MLX native)."
                    return images, status
                except Exception as exc:
                    status = self._diffusers.probe()
                    note = (
                        f"mflux failed ({type(exc).__name__}: {exc}) — "
                        "falling back to diffusers."
                    )
                    # fall through, but annotate status later
                    _mflux_fallback_note = note
                else:
                    _mflux_fallback_note = None
            else:
                _mflux_fallback_note = probe.get("reason") or "mflux unavailable"
        else:
            _mflux_fallback_note = None

        # FU-008 image subset: sd.cpp path. Routed when the catalog
        # variant declares ``engine="sdcpp"`` (which app.py threads onto
        # ``config.runtime``). Failure modes (missing binary, unsupported
        # repo, missing GGUF, subprocess error) fall through to the
        # diffusers path below and surface a runtimeNote so the user
        # still gets an image rendered.
        if (config.runtime or "").lower() == "sdcpp":
            probe = self._sdcpp.probe()
            if probe.get("available"):
                try:
                    images = self._sdcpp.generate(config)
                    status = self._diffusers.probe().to_dict()
                    status["activeEngine"] = "sd.cpp"
                    status["message"] = "Generated via stable-diffusion.cpp subprocess."
                    return images, status
                except Exception as exc:
                    _sdcpp_fallback_note = (
                        f"sd.cpp failed ({type(exc).__name__}: {exc}) — "
                        "falling back to diffusers."
                    )
                else:
                    _sdcpp_fallback_note = None
            else:
                _sdcpp_fallback_note = probe.get("reason") or "sd.cpp unavailable"
            # Combine mflux + sdcpp fallback notes if both fired (rare but
            # possible if a variant lists ``engine="sdcpp"`` AND the user
            # has overridden the runtime selector to ``"mflux"`` somehow).
            if _sdcpp_fallback_note:
                if _mflux_fallback_note:
                    _mflux_fallback_note = (
                        f"{_mflux_fallback_note} {_sdcpp_fallback_note}"
                    )
                else:
                    _mflux_fallback_note = _sdcpp_fallback_note

        status = self._diffusers.probe()
        if status.realGenerationAvailable:
            try:
                images = self._diffusers.generate(config)
                result_status = self._diffusers.probe().to_dict()
                if _mflux_fallback_note:
                    result_status["message"] = (
                        f"{_mflux_fallback_note} {result_status.get('message', '')}".strip()
                    )
                return images, result_status
            except Exception as exc:
                fallback_note = (
                    "The diffusers runtime failed, so ChaosEngineAI fell back to the placeholder engine for this run. "
                    f"Details: {exc}"
                )
                fallback_status = ImageRuntimeStatus(
                    activeEngine="placeholder",
                    realGenerationAvailable=False,
                    device=status.device,
                    pythonExecutable=status.pythonExecutable,
                    missingDependencies=[],
                    loadedModelRepo=status.loadedModelRepo,
                    message=fallback_note,
                )
                return self._placeholder.generate(config, runtime_note=fallback_note), fallback_status.to_dict()

        return self._placeholder.generate(config, runtime_note=status.message), status.to_dict()
