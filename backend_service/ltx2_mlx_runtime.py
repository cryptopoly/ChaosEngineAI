"""Native Apple Silicon video runtime for LTX-2.3 and LTX-2.5 through ltx-2-mlx.

ltx-2-mlx (MIT) is a pure MLX port of Lightricks LTX-2.3 / 2.5: text-to-video
with stereo sound on the Apple GPU, no PyTorch. It lives in its own venv
(``~/.chaosengine/ltx2-venv``, built by ``scripts/install-ltx2-mlx.sh``) and the
engine runs one ``ltx-2-mlx generate --distilled`` command per clip, so the
weights are freed when the command exits and a stuck run can be killed without
touching the backend.

Only the distilled pipeline is wired: eight denoise passes at half size, a
neural 2x upscale and three refine passes, no CFG. Guidance, steps and the
negative prompt do not apply to it and are reported as such.

The weights come from ``dgrauet/ltx-2.3-mlx-q4`` (the Download button fetches
only the files this pipeline reads) and the text encoder from
``mlx-community/gemma-3-12b-it-4bit``. An LTX-2.5 pack (``dgrauet/ltx-2.5-mlx-q4``,
gated: the licence is accepted on Hugging Face) carries its own Gemma 4 text
encoder, so it needs no companion repo. Generation never downloads anything.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from backend_service.video_runtime.types import (
    GeneratedVideo,
    VideoGenerationConfig,
    VideoRuntimeStatus,
)

LTX2_VENV_DIR = Path.home() / ".chaosengine" / "ltx2-venv"
LTX2_VERSION_FILE = Path.home() / ".chaosengine" / "bin" / "ltx2-mlx.version"
_FFMPEG_SHIM_DIR = Path.home() / ".chaosengine" / "bin" / "ffmpeg-shim"

LTX25_REPOS: frozenset[str] = frozenset({"dgrauet/ltx-2.5-mlx-q4"})
SUPPORTED_REPOS: frozenset[str] = frozenset({
    "dgrauet/ltx-2.3-mlx-q4",
    "dgrauet/ltx-2.3-mlx-q8",
}) | LTX25_REPOS
# LTX-2.3 packs read their text encoder from this separate repo; 2.5 packs carry their own.
GEMMA_REPO = "mlx-community/gemma-3-12b-it-4bit"

# The distilled pipeline denoises 8 steps at half size, then 3 after the upscale.
DISTILLED_STEPS = 11
# Half size must be a multiple of the VAE's 32 px, so the full size is a multiple of 64.
SIZE_MULTIPLE = 64

# What `generate --distilled` reads from a pack; the Download button fetches
# exactly these (catalog ``downloadFiles``).
REQUIRED_FILES: tuple[str, ...] = (
    "transformer-distilled-1.1.safetensors",
    "connector.safetensors",
    "vae_decoder.safetensors",
    "vae_encoder.safetensors",
    "audio_vae.safetensors",
    "vocoder.safetensors",
    "spatial_upscaler_x2_v1_1.safetensors",
)
# The LTX-2.5 pack names its files differently (conv video VAE, v1.0 upscaler)
# and bundles the Gemma 4 text encoder.
REQUIRED_FILES_25: tuple[str, ...] = (
    "transformer-distilled.safetensors",
    "connector.safetensors",
    "vae_decoder_conv.safetensors",
    "vae_encoder_conv.safetensors",
    "audio_vae.safetensors",
    "vocoder.safetensors",
    "spatial_upscaler_x2_v1_0.safetensors",
    "text_encoder.safetensors",
)

_LAST_OUTPUT_LINES = 60
# tqdm prints ``Denoising (stage 1):  25%|##5 | 2/8 [00:11<00:33, 11.23s/it]``.
_STEP_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")
_PHASE_RE = re.compile(r"^\[(.+?)\] \.\.\.$")


def _apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def is_installed() -> bool:
    """Cheap check: the version file is written only after the install verified."""
    return LTX2_VERSION_FILE.exists() and (LTX2_VENV_DIR / "bin" / "ltx-2-mlx").exists()


def read_version() -> str | None:
    try:
        lines = LTX2_VERSION_FILE.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return None
    return lines[0].strip() if lines and lines[0].strip() else None


def _is_ltx2_mlx_repo(repo: str | None) -> bool:
    return bool(repo) and repo in SUPPORTED_REPOS


def is_ltx25_repo(repo: str | None) -> bool:
    return bool(repo) and repo in LTX25_REPOS


def required_files(repo: str | None) -> tuple[str, ...]:
    return REQUIRED_FILES_25 if is_ltx25_repo(repo) else REQUIRED_FILES


def gemma_repo(repo: str | None) -> str | None:
    """Repo holding the text encoder, or None when the pack bundles it."""
    return None if is_ltx25_repo(repo) else GEMMA_REPO


def generation_label(repo: str | None) -> str:
    return "LTX-2.5" if is_ltx25_repo(repo) else "LTX-2.3"


def missing_pack_files(root: Path, repo: str | None = None) -> list[str]:
    return [name for name in required_files(repo) if not (root / name).exists()]


def _local_snapshot(repo: str) -> Path | None:
    try:
        from backend_service.helpers.hf_local import local_snapshot_path

        return Path(local_snapshot_path(repo))
    except Exception:  # noqa: BLE001 - not downloaded, or an unreadable cache
        return None


def align_size(width: int, height: int) -> tuple[int, int]:
    """Round a size to this pipeline's multiple of 64 (never below 64)."""
    def snap(value: int) -> int:
        return max(SIZE_MULTIPLE, int((value + SIZE_MULTIPLE // 2) // SIZE_MULTIPLE) * SIZE_MULTIPLE)

    return snap(width), snap(height)


def _ffmpeg_path_dir() -> str | None:
    """A directory holding an ``ffmpeg`` for the child process, or None when the system has one.

    ltx-2-mlx encodes with the ``ffmpeg`` command. A Mac without Homebrew has
    none, so the static binary that ships with imageio-ffmpeg (a dependency of
    the app's own video export) is linked into a small directory on PATH.
    """
    if shutil.which("ffmpeg"):
        return None
    try:
        import imageio_ffmpeg

        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001
        return None
    try:
        _FFMPEG_SHIM_DIR.mkdir(parents=True, exist_ok=True)
        link = _FFMPEG_SHIM_DIR / "ffmpeg"
        if link.is_symlink() or link.exists():
            if link.resolve() == exe.resolve():
                return str(_FFMPEG_SHIM_DIR)
            link.unlink()
        link.symlink_to(exe)
    except OSError:
        return None
    return str(_FFMPEG_SHIM_DIR)


class Ltx2MlxVideoEngine:
    runtime_label = "LTX-2.3 (MLX native)"

    def probe(self) -> VideoRuntimeStatus:
        if not _apple_silicon():
            return VideoRuntimeStatus(
                activeEngine="ltx2-mlx",
                realGenerationAvailable=False,
                message="LTX-2.3 through ltx-2-mlx runs on Apple Silicon only.",
            )
        if not is_installed():
            return VideoRuntimeStatus(
                activeEngine="ltx2-mlx",
                realGenerationAvailable=False,
                message="The LTX-2.3 MLX engine is not installed. Install it from the Video Studio to run this model.",
            )
        return VideoRuntimeStatus(
            activeEngine="ltx2-mlx",
            realGenerationAvailable=True,
            message=f"LTX-2.3 MLX engine ready (ltx-2-mlx {read_version() or '?'}).",
            device="mlx",
        )

    def preload(self, repo: str) -> VideoRuntimeStatus:
        # One process per clip: nothing stays resident between runs.
        return self.probe()

    def unload(self, repo: str | None = None) -> VideoRuntimeStatus:
        return self.probe()

    def generate(self, config: VideoGenerationConfig) -> GeneratedVideo:
        status = self.probe()
        if not status.realGenerationAvailable:
            raise RuntimeError(status.message)
        if config.repo not in SUPPORTED_REPOS:
            raise RuntimeError(f"ltx-2-mlx does not support {config.repo}. Supported: {', '.join(sorted(SUPPORTED_REPOS))}.")

        pack = _local_snapshot(config.repo)
        if pack is None:
            raise RuntimeError(f"{config.modelName} is not downloaded. Download it from Video Discover first.")
        missing = missing_pack_files(pack, config.repo)
        if missing:
            raise RuntimeError(
                f"{config.modelName} is incomplete (missing {', '.join(missing)}). Download it again from Video Discover."
            )
        gemma: Path | None = None
        text_encoder_repo = gemma_repo(config.repo)
        if text_encoder_repo is not None:
            gemma = _local_snapshot(text_encoder_repo)
            if gemma is None or not (gemma / "model.safetensors.index.json").exists():
                raise RuntimeError(
                    f"The text encoder {text_encoder_repo} is not downloaded. Download {config.modelName} again from Video Discover."
                )

        from backend_service.progress import PHASE_LOADING, VIDEO_PROGRESS
        from backend_service.video_runtime import _resolve_video_seed

        width, height = align_size(config.width, config.height)
        seed = _resolve_video_seed(config.seed)
        label = generation_label(config.repo)
        notes = [f"{label} distilled ({DISTILLED_STEPS} passes, no guidance; steps and guidance do not apply)"]
        if (width, height) != (config.width, config.height):
            notes.append(f"size {config.width}x{config.height} changed to {width}x{height} (multiples of {SIZE_MULTIPLE})")
        if config.negativePrompt.strip():
            notes.append("the negative prompt is ignored by the distilled pipeline")

        VIDEO_PROGRESS.begin(
            run_label=f"{config.modelName} · {config.numFrames}f @ {width}x{height} · ltx-2-mlx",
            total_steps=DISTILLED_STEPS,
            phase=PHASE_LOADING,
            message=f"Loading {config.modelName}",
        )
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="chaosengine-ltx2-") as tmpdir:
                output = Path(tmpdir) / "clip.mp4"
                args = self._build_command(config, pack, gemma, width, height, seed, output)
                self._run(args, output)
                data = output.read_bytes()
        finally:
            VIDEO_PROGRESS.finish()

        return GeneratedVideo(
            seed=seed,
            bytes=data,
            extension="mp4",
            mimeType="video/mp4",
            durationSeconds=round(max(0.1, time.perf_counter() - started), 2),
            frameCount=config.numFrames,
            fps=config.fps,
            width=width,
            height=height,
            runtimeLabel=f"{label} (MLX native)",
            runtimeNote="; ".join(notes) + ".",
            effectiveSteps=DISTILLED_STEPS,
            effectiveGuidance=1.0,
        )

    @staticmethod
    def _build_command(
        config: VideoGenerationConfig,
        pack: Path,
        gemma: Path | None,
        width: int,
        height: int,
        seed: int,
        output: Path,
    ) -> list[str]:
        gemma_args = ["--gemma", str(gemma)] if gemma is not None else []
        return [
            str(LTX2_VENV_DIR / "bin" / "ltx-2-mlx"),
            "generate",
            "--distilled",
            "--model", str(pack),
            *gemma_args,
            "--prompt", config.prompt,
            "-H", str(height),
            "-W", str(width),
            "--frames", str(config.numFrames),
            "--frame-rate", f"{config.fps:g}",
            "--seed", str(seed),
            "--output", str(output),
        ]

    @staticmethod
    def _child_env() -> dict[str, str]:
        from backend_service.inference._utils import _isolated_child_env

        env = dict(_isolated_child_env())
        # Generation reads local files only; a missing file must fail, not download.
        env["HF_HUB_OFFLINE"] = "1"
        shim = _ffmpeg_path_dir()
        if shim:
            env["PATH"] = f"{shim}{os.pathsep}{env.get('PATH', '')}"
        return env

    def _run(self, args: list[str], output: Path) -> None:
        """Run one command and stream its progress into ``VIDEO_PROGRESS``."""
        from backend_service.progress import (
            PHASE_DECODING,
            PHASE_DIFFUSING,
            PHASE_ENCODING,
            VIDEO_PROGRESS,
            GenerationCancelled,
        )

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._child_env(),
        )
        tail: list[str] = []
        completed_steps = 0
        last_n = 0
        last_total = 0
        try:
            assert proc.stdout is not None
            # tqdm redraws with carriage returns; text mode turns those into lines.
            for line in proc.stdout:
                stripped = line.rstrip()
                if not stripped:
                    continue
                tail.append(stripped)
                del tail[:-_LAST_OUTPUT_LINES]
                step = _STEP_RE.search(stripped)
                if step:
                    n, total = int(step.group(1)), int(step.group(2))
                    # The second stage starts a new bar; carry the finished steps over.
                    if total != last_total or n < last_n:
                        completed_steps += last_n if last_total else 0
                        VIDEO_PROGRESS.set_phase(PHASE_DIFFUSING, message="Denoising")
                    last_n, last_total = n, total
                    VIDEO_PROGRESS.set_step(min(DISTILLED_STEPS, completed_steps + n), total=DISTILLED_STEPS)
                else:
                    phase = _PHASE_RE.match(stripped)
                    if phase:
                        label = phase.group(1).lower()
                        if ("encod" in label and "text" in label) or "gemma" in label:
                            VIDEO_PROGRESS.set_phase(PHASE_ENCODING, message="Encoding prompt")
                        elif "decod" in label or "vae" in label or "audio" in label:
                            VIDEO_PROGRESS.set_phase(PHASE_DECODING, message="Decoding video")
                if VIDEO_PROGRESS.is_cancelled():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise GenerationCancelled("Video generation cancelled by user")
            code = proc.wait()
        except BaseException:
            if proc.poll() is None:
                proc.kill()
            raise
        if code != 0:
            raise RuntimeError(f"ltx-2-mlx exited with code {code}.\nLast output:\n" + "\n".join(tail[-15:]))
        if not output.exists():
            raise RuntimeError("ltx-2-mlx finished without writing a video.\nLast output:\n" + "\n".join(tail[-10:]))


def status_payload() -> dict[str, Any]:
    """Lightweight probe for the setup route."""
    return {
        "installed": is_installed(),
        "version": read_version() if is_installed() else None,
        "supported": _apple_silicon(),
        "repos": sorted(SUPPORTED_REPOS),
    }
