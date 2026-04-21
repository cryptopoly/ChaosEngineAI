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
        self._device: str | None = None

    def probe(self) -> ImageRuntimeStatus:
        # Deliberately does NOT ``import torch`` — that would load
        # torch/lib/*.dll into the backend process handle table, and on
        # Windows those locked DLLs break /api/setup/install-gpu-bundle
        # (pip's rmtree can't remove files another process has open).
        # find_spec answers "is it installable?" without triggering the
        # import side effects. Device detection (cuda vs cpu) is deferred
        # to preload/generate where we're about to import torch anyway.
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
            pipeline = self._ensure_pipeline(config.repo)
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
            total_steps = int(kwargs.get("num_inference_steps", config.steps) or config.steps)
            IMAGE_PROGRESS.set_phase(
                PHASE_DIFFUSING,
                message=self._diffuse_message(config),
            )
            # Re-publish the totalSteps in case ``num_inference_steps`` was
            # clamped above (Flux/Turbo cap at 8).
            IMAGE_PROGRESS.set_step(0, total=max(1, total_steps))

            def _on_step_end(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]):
                # Diffusers calls this *after* step ``step`` finishes, so step
                # 0 means "one step done". Convert to the 1-indexed value the
                # UI wants to display.
                IMAGE_PROGRESS.set_step(step + 1, total=max(1, total_steps))
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
                artifacts.append(
                    GeneratedImage(
                        seed=base_seed + index,
                        bytes=buffer.getvalue(),
                        extension="png",
                        mimeType="image/png",
                        durationSeconds=round(elapsed / max(1, config.batchSize), 1),
                        runtimeLabel=f"{self.runtime_label} ({self._device or 'cpu'})",
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

    def _ensure_pipeline(self, repo: str) -> Any:
        with self._lock:
            if self._pipeline is not None and self._loaded_repo == repo:
                return self._pipeline

            # Loading a pipeline can take 10-60s on cold disk. Surface that
            # explicitly to the UI so the progress bar stops sitting at 0%
            # while we read 5GB of weights from the SSD.
            IMAGE_PROGRESS.set_phase(PHASE_LOADING, message=f"Loading {repo}")

            if self._pipeline is not None and self._loaded_repo != repo:
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
            dtype = self._preferred_torch_dtype(torch, repo, device)
            pipeline = AutoPipelineForText2Image.from_pretrained(
                local_path,
                torch_dtype=dtype,
                local_files_only=True,
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
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            vae = getattr(pipeline, "vae", None)
            if vae is not None and hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            if device != "cpu":
                pipeline = pipeline.to(device)

            self._pipeline = pipeline
            self._torch = torch
            self._loaded_repo = repo
            self._loaded_path = local_path
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

    def _preferred_torch_dtype(self, torch: Any, repo: str, device: str) -> Any:
        if device == "cuda":
            return torch.float16
        if device == "mps":
            lowered_repo = repo.lower()
            # SDXL / Stable Diffusion on MPS can silently decode to black
            # images in fp16. Favor correctness over speed for those repos.
            if any(token in lowered_repo for token in ("stable-diffusion", "sdxl", "sd_xl")):
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


class ImageRuntimeManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._placeholder = PlaceholderImageEngine()
        self._diffusers = DiffusersTextToImageEngine()

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
        status = self._diffusers.probe()
        if status.realGenerationAvailable:
            try:
                images = self._diffusers.generate(config)
                return images, self._diffusers.probe().to_dict()
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
