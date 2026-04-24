"""LongLive subprocess engine for real-time long video generation.

LongLive (NVlabs, ICLR 2026, Apache 2.0) extends Wan 2.1 T2V 1.3B with
causal long-video generation — up to 240s @ ~20 FPS on a single H100.
Unlike the rest of our video backends it ships as a torchrun-launched
script, not a diffusers pipeline, and it expects a very specific Python
environment (diffusers==0.31.0, torchao, flash-attn optional) that we do
not want to force on the host venv.

This module wraps the LongLive install (see ``scripts/install-longlive.sh``)
as a subprocess so the mismatched deps stay quarantined. It exposes the
same shape as ``DiffusersVideoEngine`` (``probe`` / ``generate``) so the
video runtime dispatch can route to it without the rest of the app
caring about torchrun.

CUDA-only. The engine reports unavailable on macOS and when the install
marker is missing, which lets the Studio surface a one-click install
rather than blowing up at generate-time.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from backend_service.video_runtime import (
    GeneratedVideo,
    VideoGenerationConfig,
    VideoRuntimeStatus,
)


DEFAULT_LONGLIVE_ROOT = Path(
    os.environ.get(
        "CHAOSENGINE_LONGLIVE_ROOT",
        str(Path.home() / ".chaosengine" / "longlive"),
    )
)

# Output FPS Wan 2.1 VAE decodes at — LongLive inherits this from the
# base Wan model. Used to compute latent-frame counts from the user's
# requested clip duration.
LONGLIVE_OUTPUT_FPS = 16

# VAE temporal compression factor. Each latent frame decodes into this
# many pixel frames. Latent counts must also satisfy LongLive's block
# constraint (``num_output_frames % num_frame_per_block == 0``).
LATENT_TO_PIXEL_FRAMES = 4
DEFAULT_FRAMES_PER_BLOCK = 3

MODEL_ID = "NVlabs/LongLive-1.3B"
MODEL_DISPLAY_NAME = "LongLive 1.3B (long-form, causal)"


@dataclass(frozen=True)
class LongLiveInstallInfo:
    """Filesystem view of a LongLive install."""

    root: Path
    repo_dir: Path
    venv_python: Path
    weights_dir: Path
    wan_base_dir: Path
    marker: Path

    @property
    def ready(self) -> bool:
        return (
            self.marker.exists()
            and self.repo_dir.is_dir()
            and self.venv_python.exists()
        )


def resolve_install(root: Path | None = None) -> LongLiveInstallInfo:
    base = Path(root) if root else DEFAULT_LONGLIVE_ROOT
    venv_python = base / "venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
    if os.name == "nt":
        venv_python = venv_python.with_suffix(".exe")
    return LongLiveInstallInfo(
        root=base,
        repo_dir=base / "repo",
        venv_python=venv_python,
        weights_dir=base / "longlive_models" / "models",
        wan_base_dir=base / "wan_base",
        marker=base / "ready.marker",
    )


def _cuda_available() -> bool:
    """Cheap CUDA presence check that does not import torch.

    We can't call torch.cuda.is_available() during probe() — importing
    torch in the main sidecar process would lock its DLLs on Windows and
    block the installer. Fall back to nvidia-smi / platform like the rest
    of the video runtime does.
    """
    if platform.system() == "Darwin":
        return False
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                timeout=5,
                text=True,
            )
            return result.returncode == 0 and "GPU" in result.stdout
        except (subprocess.SubprocessError, OSError):
            return False
    return False


def compute_latent_frames(
    duration_seconds: float,
    frames_per_block: int = DEFAULT_FRAMES_PER_BLOCK,
) -> int:
    """Map a requested clip duration to a LongLive ``num_output_frames``.

    LongLive's inference pipeline asserts
    ``num_output_frames % num_frame_per_block == 0`` so we round the
    derived latent count up to the nearest block. Short requests still
    get at least one block.
    """
    pixel_frames = max(1.0, float(duration_seconds)) * LONGLIVE_OUTPUT_FPS
    latent_frames = max(1, int(round(pixel_frames / LATENT_TO_PIXEL_FRAMES)))
    remainder = latent_frames % frames_per_block
    if remainder:
        latent_frames += frames_per_block - remainder
    return latent_frames


def pixel_frames_for_latents(latent_frames: int) -> int:
    return latent_frames * LATENT_TO_PIXEL_FRAMES


class _ProgressSink(Protocol):
    def __call__(self, phase: str, message: str, fraction: float) -> None: ...


class LongLiveEngine:
    """Subprocess-based engine for the NVlabs/LongLive long-video generator."""

    runtime_label = "LongLive (subprocess)"

    def __init__(self, install_root: Path | None = None) -> None:
        self._install = resolve_install(install_root)
        self._loaded_repo: str | None = None

    # ---------- public API ----------

    def probe(self) -> VideoRuntimeStatus:
        install = self._install
        cuda_ok = _cuda_available()

        if platform.system() == "Darwin":
            return VideoRuntimeStatus(
                activeEngine="longlive",
                realGenerationAvailable=False,
                expectedDevice=None,
                message=(
                    "LongLive requires CUDA — unsupported on macOS. Run on a "
                    "Windows or Linux machine with a recent NVIDIA GPU."
                ),
            )

        if not install.ready:
            return VideoRuntimeStatus(
                activeEngine="longlive",
                realGenerationAvailable=False,
                expectedDevice="cuda" if cuda_ok else None,
                message=(
                    f"LongLive is not installed. Run `scripts/install-longlive.sh` "
                    f"(or use the Studio install action) — expected install at "
                    f"{install.root}."
                ),
            )

        if not cuda_ok:
            return VideoRuntimeStatus(
                activeEngine="longlive",
                realGenerationAvailable=False,
                expectedDevice=None,
                message=(
                    "LongLive is installed but no CUDA GPU was detected. Connect "
                    "a supported NVIDIA GPU and retry."
                ),
                loadedModelRepo=self._loaded_repo,
            )

        return VideoRuntimeStatus(
            activeEngine="longlive",
            realGenerationAvailable=True,
            device="cuda",
            expectedDevice="cuda",
            message="LongLive ready — long-form generation up to ~240s supported.",
            loadedModelRepo=self._loaded_repo,
        )

    def preload(self, repo: str) -> VideoRuntimeStatus:
        # LongLive loads inside the subprocess so there's nothing to hold
        # in this process; we just remember the selection so the UI can
        # show "Loaded: LongLive" consistently with the diffusers engine.
        self._loaded_repo = repo
        return self.probe()

    def unload(self, repo: str | None = None) -> VideoRuntimeStatus:
        if repo is None or self._loaded_repo == repo:
            self._loaded_repo = None
        return self.probe()

    def generate(
        self,
        config: VideoGenerationConfig,
        *,
        infinite: bool = False,
        on_progress: _ProgressSink | None = None,
    ) -> GeneratedVideo:
        """Run a single long-video generation via torchrun subprocess.

        The LongLive pipeline writes its output mp4 under
        ``<workspace>/videos/0.mp4`` (``save_with_index`` mode). We poll
        for that file to appear, read its bytes, and surface them as a
        ``GeneratedVideo`` so the rest of the video runtime is unaware a
        subprocess was involved.
        """
        probe = self.probe()
        if not probe.realGenerationAvailable:
            raise RuntimeError(probe.message)

        if on_progress:
            on_progress("loading", "Preparing LongLive workspace", 0.0)

        workspace = Path(tempfile.mkdtemp(prefix="longlive-run-"))
        try:
            prompt_file = workspace / "prompt.txt"
            prompt_file.write_text(config.prompt.strip() + "\n", encoding="utf-8")

            latent_frames = compute_latent_frames(
                config.numFrames / max(1, config.fps)
            )
            pixel_frames = pixel_frames_for_latents(latent_frames)

            config_path = workspace / "run.yaml"
            config_path.write_text(
                _render_longlive_yaml(
                    install=self._install,
                    prompt_file=prompt_file,
                    output_dir=workspace / "videos",
                    num_output_frames=latent_frames,
                    seed=config.seed if config.seed is not None else 0,
                    infinite=infinite,
                ),
                encoding="utf-8",
            )

            if on_progress:
                on_progress("diffusing", f"Rendering {pixel_frames} frames", 0.05)

            self._launch_torchrun(workspace, config_path, on_progress)

            output_mp4 = _find_output_mp4(workspace / "videos")
            if output_mp4 is None:
                raise RuntimeError(
                    "LongLive finished but no mp4 was produced. "
                    f"Check {workspace} for logs."
                )

            data = output_mp4.read_bytes()
            return GeneratedVideo(
                seed=config.seed if config.seed is not None else 0,
                bytes=data,
                extension="mp4",
                mimeType="video/mp4",
                durationSeconds=pixel_frames / LONGLIVE_OUTPUT_FPS,
                frameCount=pixel_frames,
                fps=LONGLIVE_OUTPUT_FPS,
                width=config.width,
                height=config.height,
                runtimeLabel=self.runtime_label,
                runtimeNote="LongLive subprocess (torchrun)",
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    # ---------- internals ----------

    def _launch_torchrun(
        self,
        workspace: Path,
        config_path: Path,
        on_progress: _ProgressSink | None,
    ) -> None:
        """Run torchrun against the LongLive repo.

        Split out so tests can stub it without needing torchrun or CUDA.
        """
        install = self._install
        cmd = [
            str(install.venv_python),
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--master_port=29500",
            "inference.py",
            "--config_path",
            str(config_path),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Point the LongLive pipeline at our downloaded Wan base weights
        # so WanDiffusionWrapper doesn't try to fetch them at runtime.
        env["WAN_MODEL_DIR"] = str(install.wan_base_dir)

        start = time.monotonic()
        process = subprocess.Popen(
            cmd,
            cwd=str(install.repo_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                if on_progress and "kv_cache_size" in line:
                    on_progress("diffusing", line.strip()[:120], 0.1)
                elif on_progress and "block" in line.lower() and "/" in line:
                    on_progress("diffusing", line.strip()[:120], 0.5)
        finally:
            return_code = process.wait()

        duration = time.monotonic() - start
        if return_code != 0:
            raise RuntimeError(
                f"LongLive torchrun exited with code {return_code} after "
                f"{duration:.1f}s. See stdout above."
            )


def _render_longlive_yaml(
    *,
    install: LongLiveInstallInfo,
    prompt_file: Path,
    output_dir: Path,
    num_output_frames: int,
    seed: int,
    infinite: bool,
) -> str:
    """Emit a LongLive inference YAML matching ``configs/longlive_inference*.yaml``.

    We keep the architecture knobs identical to upstream (denoising
    schedule, frames_per_block, LoRA config) and only override the
    runtime fields the user controls (prompt, frame count, seed, output
    dir). Infinite mode enables ``use_infinite_attention`` for clips
    longer than the local-attention window can cover in one shot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    infinity_flag = "true" if infinite else "false"
    generator_ckpt = install.weights_dir / "longlive_base.pt"
    lora_ckpt = install.weights_dir / "lora.pt"
    return textwrap.dedent(
        f"""\
        denoising_step_list:
        - 1000
        - 750
        - 500
        - 250
        warp_denoising_step: true
        num_frame_per_block: {DEFAULT_FRAMES_PER_BLOCK}
        model_name: Wan2.1-T2V-1.3B
        model_kwargs:
          local_attn_size: 12
          timestep_shift: 5.0
          sink_size: 3
          use_infinite_attention: {infinity_flag}

        data_path: {prompt_file}
        output_folder: {output_dir}
        inference_iter: -1
        num_output_frames: {num_output_frames}
        use_ema: false
        seed: {seed}
        num_samples: 1
        save_with_index: true
        global_sink: true
        context_noise: 0

        generator_ckpt: {generator_ckpt}
        lora_ckpt: {lora_ckpt}

        adapter:
          type: "lora"
          rank: 256
          alpha: 256
          dropout: 0.0
          dtype: "bfloat16"
          verbose: false
        """
    )


def _find_output_mp4(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    candidates = sorted(output_dir.glob("*.mp4"))
    return candidates[0] if candidates else None
