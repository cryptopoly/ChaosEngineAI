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
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend_service.image_runtime import validate_local_diffusers_snapshot
from backend_service.progress import (
    PHASE_DECODING,
    PHASE_DIFFUSING,
    PHASE_ENCODING,
    PHASE_LOADING,
    PHASE_SAVING,
    VIDEO_PROGRESS,
)


MAX_VIDEO_SEED = 2147483647


def _resolve_video_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return secrets.randbelow(MAX_VIDEO_SEED + 1)


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
    """
    try:
        from backend_service.helpers.gpu import get_gpu_metrics
    except Exception:
        return None
    try:
        snapshot = get_gpu_metrics()
    except Exception:
        return None
    total = snapshot.get("vram_total_gb")
    if isinstance(total, (int, float)) and total > 0:
        return float(total)
    return None


@dataclass(frozen=True)
class VideoRuntimeStatus:
    activeEngine: str
    realGenerationAvailable: bool
    message: str
    device: str | None = None
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
    # Community-maintained diffusers port of tencent/HunyuanVideo.
    "hunyuanvideo-community/HunyuanVideo": {"class_name": "HunyuanVideoPipeline", "task": "txt2video"},
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
        device_memory_gb = _detect_device_memory_gb(device)

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
        VIDEO_PROGRESS.begin(
            run_label=self._format_run_label(config),
            total_steps=max(1, int(config.steps)),
            phase=PHASE_LOADING,
            message=f"Preparing {config.modelName}",
        )
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

            pipeline = self._ensure_pipeline(config.repo)
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

            started = time.perf_counter()
            frames = self._invoke_pipeline(pipeline, kwargs)
            elapsed = max(0.1, time.perf_counter() - started)

            if not frames:
                raise RuntimeError(
                    f"The video pipeline returned zero frames for {config.repo}. "
                    "Try a smaller resolution or a different model."
                )

            VIDEO_PROGRESS.set_phase(PHASE_DECODING, message="Encoding mp4")
            mp4_bytes = self._encode_frames_to_mp4(frames, config.fps)
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
                fps=config.fps,
                width=config.width,
                height=config.height,
                runtimeLabel=f"{self.runtime_label} ({self._device or 'cpu'})",
            )
        finally:
            VIDEO_PROGRESS.finish()

    def _format_run_label(self, config: VideoGenerationConfig) -> str:
        return f"{config.modelName} · {config.numFrames}f @ {config.width}x{config.height}"

    # ---------- internals ----------

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
        }
        lowered_repo = config.repo.lower()
        if "hunyuanvideo" not in lowered_repo and config.negativePrompt.strip():
            kwargs["negative_prompt"] = config.negativePrompt
        return kwargs

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

        def _on_step_end(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]):
            VIDEO_PROGRESS.set_step(step + 1, total=max(1, total_steps))
            return callback_kwargs

        kwargs.setdefault("callback_on_step_end", _on_step_end)

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

    def _ensure_pipeline(self, repo: str) -> Any:
        with self._lock:
            if self._pipeline is not None and self._loaded_repo == repo:
                return self._pipeline

            # Loading a video pipeline can read 10+ GB from disk on cold cache.
            # Publish the phase so the UI explicitly says "Loading model" while
            # snapshot_download + from_pretrained run.
            VIDEO_PROGRESS.set_phase(PHASE_LOADING, message=f"Loading {repo}")

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

    def generate(self, config: VideoGenerationConfig) -> tuple[GeneratedVideo, dict[str, Any]]:
        """Run a single video generation, returning (video, runtime_status).

        Unlike the image manager, there is no placeholder fallback — video is
        heavy enough that a silent fake clip would waste the user's time. If
        the runtime isn't ready, raise a clear error so the route can return
        a proper 4xx.
        """
        status = self._engine.probe()
        if not status.realGenerationAvailable:
            raise RuntimeError(status.message)
        with self._lock:
            video = self._engine.generate(config)
            runtime = self._engine.probe().to_dict()
        return video, runtime
