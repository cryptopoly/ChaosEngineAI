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
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend_service.image_runtime import validate_local_diffusers_snapshot


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_video_python() -> str:
    override = os.getenv("CHAOSENGINE_MLX_PYTHON") or os.getenv("CHAOSENGINE_VIDEO_PYTHON")
    if override:
        return override
    candidate = WORKSPACE_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return os.getenv("PYTHON", "python3")


@dataclass(frozen=True)
class VideoRuntimeStatus:
    activeEngine: str
    realGenerationAvailable: bool
    message: str
    device: str | None = None
    pythonExecutable: str | None = None
    missingDependencies: list[str] = field(default_factory=list)
    loadedModelRepo: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoGenerationConfig:
    """Shape for a future generate() call. Not wired yet — kept here so the
    routes can validate requests against the same structure once generation
    lands."""
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
    seed: int | None = None


# Maps a Hugging Face repo id to the diffusers pipeline class that loads it.
# The class name is looked up dynamically on the ``diffusers`` module so we
# don't blow up at import time if the installed diffusers is older than
# expected — users just see a clearer "unsupported pipeline" error at preload.
PIPELINE_REGISTRY: dict[str, dict[str, str]] = {
    "Lightricks/LTX-Video": {"class_name": "LTXPipeline", "task": "txt2video"},
    "genmo/mochi-1-preview": {"class_name": "MochiPipeline", "task": "txt2video"},
    # Wan 2.1 and 2.2 share the same pipeline class — the version difference
    # lives in the weights, not the pipeline code.
    "Wan-AI/Wan2.1-T2V-1.3B": {"class_name": "WanPipeline", "task": "txt2video"},
    "Wan-AI/Wan2.1-T2V-14B": {"class_name": "WanPipeline", "task": "txt2video"},
    "Wan-AI/Wan2.2-T2V-A14B": {"class_name": "WanPipeline", "task": "txt2video"},
    "tencent/HunyuanVideo": {"class_name": "HunyuanVideoPipeline", "task": "txt2video"},
}


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


def _find_missing(deps: tuple[tuple[str, str], ...]) -> list[str]:
    return [package for package, module_name in deps if importlib.util.find_spec(module_name) is None]


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
        self._device: str | None = None

    # ---------- public API ----------

    def probe(self) -> VideoRuntimeStatus:
        missing_core = _find_missing(_CORE_DEPS)
        missing_output = _find_missing(_VIDEO_OUTPUT_DEPS)

        # All missing deps are reported so the UI can surface a clear install
        # hint, but only ``_CORE_DEPS`` block ``realGenerationAvailable``.
        missing_all = missing_core + missing_output

        if missing_core:
            return VideoRuntimeStatus(
                activeEngine="placeholder",
                realGenerationAvailable=False,
                missingDependencies=missing_all,
                pythonExecutable=_resolve_video_python(),
                message=(
                    "Install the optional video runtime packages to enable local generation: "
                    "pip install 'diffusers[torch]' accelerate imageio imageio-ffmpeg"
                ),
                loadedModelRepo=self._loaded_repo,
            )

        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - torch import side effects
            return VideoRuntimeStatus(
                activeEngine="placeholder",
                realGenerationAvailable=False,
                missingDependencies=["torch"] + missing_output,
                pythonExecutable=_resolve_video_python(),
                message=f"PyTorch could not be imported cleanly: {exc}",
                loadedModelRepo=self._loaded_repo,
            )

        device = self._detect_device(torch)

        if missing_output:
            message = (
                "Video runtime is ready to load models, but mp4 encoding packages are missing — "
                "run `pip install imageio imageio-ffmpeg` before generating videos."
            )
        else:
            message = (
                "Real local video generation is available. Download a video model, then Video Studio "
                "will use the diffusers runtime."
            )

        return VideoRuntimeStatus(
            activeEngine="diffusers",
            realGenerationAvailable=True,
            device=device,
            pythonExecutable=_resolve_video_python(),
            missingDependencies=missing_output,
            message=message,
            loadedModelRepo=self._loaded_repo,
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

    def generate(self, config: VideoGenerationConfig) -> Any:
        raise NotImplementedError(
            "Video generation is not implemented yet. Preload works, but generation "
            "lands in a later phase (frame decoding + mp4 encoding + save pipeline)."
        )

    # ---------- internals ----------

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

    def _ensure_pipeline(self, repo: str) -> Any:
        with self._lock:
            if self._pipeline is not None and self._loaded_repo == repo:
                return self._pipeline

            if self._pipeline is not None and self._loaded_repo != repo:
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

            pipeline = pipeline_cls.from_pretrained(
                local_path,
                torch_dtype=dtype,
                local_files_only=True,
            )

            # Memory-saving knobs. Video pipelines are hungry on unified-memory
            # and consumer GPUs alike — slice everything we reasonably can.
            if hasattr(pipeline, "set_progress_bar_config"):
                pipeline.set_progress_bar_config(disable=True)
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            vae = getattr(pipeline, "vae", None)
            if vae is not None:
                if hasattr(vae, "enable_slicing"):
                    vae.enable_slicing()
                if hasattr(vae, "enable_tiling"):
                    vae.enable_tiling()

            if device != "cpu":
                # Try full-device placement first; fall back to sequential CPU
                # offload if the model is too big to fit.
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
            return torch.float16
        return torch.float32


class VideoRuntimeManager:
    """State-level facade that mirrors ``ImageRuntimeManager``."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._engine = DiffusersVideoEngine()

    def capabilities(self) -> dict[str, Any]:
        return self._engine.probe().to_dict()

    def preload(self, repo: str) -> dict[str, Any]:
        with self._lock:
            status = self._engine.probe()
            if not status.realGenerationAvailable:
                raise RuntimeError(status.message)
            return self._engine.preload(repo).to_dict()

    def unload(self, repo: str | None = None) -> dict[str, Any]:
        with self._lock:
            return self._engine.unload(repo).to_dict()
