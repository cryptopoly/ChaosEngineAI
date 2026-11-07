"""Native Apple Silicon video runtime for LTX-2.3 and LTX-2.5 through ltx-2-mlx.

ltx-2-mlx (MIT) is a pure MLX port of Lightricks LTX-2.3 / 2.5: text-to-video
with stereo sound on the Apple GPU, no PyTorch. It lives in its own venv
(``~/.chaosengine/ltx2-venv``, built by ``scripts/install-ltx2-mlx.sh``). On a
Mac with 64 GB or more the pipeline stays loaded between clips in a warm worker
process (``ltx2_worker.py``, V1); otherwise, when ``CHAOSENGINE_WARM_WORKERS=0``
or when the worker fails, the engine runs one ``ltx-2-mlx generate --distilled``
command per clip. Either way the weights live outside the backend, and a stuck
run is killed without touching it.

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
import threading
import time
import uuid
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
# ltx-2-mlx closes each phase with ``[Encoding prompt] done in 3.7s``.
_PHASE_DONE_RE = re.compile(r"^\[(.+?)\] done in ([\d.]+)s$")

_WORKER_SCRIPT = Path(__file__).resolve().parent / "ltx2_worker.py"
_WORKER_LOG = Path.home() / ".chaosengine" / "logs" / "ltx2-worker.log"
# The resident pipeline (LTX-2.5 q4) holds about 29 GB between clips and peaked at 38.6 GB of Metal
# memory decoding 768x448x49 (the CLI peaks near 17 GB; v-speed.md), so it is kept only on Macs that
# report about 64 GB, and a new worker starts only with 40 GB free; otherwise the clip runs as a CLI process.
WARM_MIN_MEMORY_GB = 60.0
WARM_START_AVAILABLE_GB = 40.0
_WORKER_TIMEOUT_S = 7200.0
_WORKER_STALL_S = 900.0  # load and VAE decode send no step lines
_WORKER_HARD_CANCEL_S = 20.0  # a stage-2 step at full size can take ~15 s before the next write


class Ltx2OutputParser:
    """Reads ltx-2-mlx output lines (CLI stdout or the warm worker's ``line`` events): progress into
    ``VIDEO_PROGRESS`` and the C1 timing split from the ``[phase] done in Xs`` lines."""

    def __init__(self) -> None:
        self.tail: list[str] = []
        self._completed_steps = 0
        self._last_n = 0
        self._last_total = 0
        self.phase_seconds = {"load": 0.0, "encode": 0.0, "decode": 0.0}

    def feed(self, stripped: str) -> None:
        from backend_service.progress import PHASE_DECODING, PHASE_DIFFUSING, PHASE_ENCODING, VIDEO_PROGRESS

        self.tail.append(stripped)
        del self.tail[:-_LAST_OUTPUT_LINES]
        step = _STEP_RE.search(stripped)
        if step:
            n, total = int(step.group(1)), int(step.group(2))
            # The second stage starts a new bar; carry the finished steps over.
            if total != self._last_total or n < self._last_n:
                self._completed_steps += self._last_n if self._last_total else 0
                VIDEO_PROGRESS.set_phase(PHASE_DIFFUSING, message="Denoising")
            self._last_n, self._last_total = n, total
            VIDEO_PROGRESS.set_step(min(DISTILLED_STEPS, self._completed_steps + n), total=DISTILLED_STEPS)
            return
        done = _PHASE_DONE_RE.match(stripped)
        if done:
            label, seconds = done.group(1).lower(), float(done.group(2))
            if label.startswith("loading"):
                self.phase_seconds["load"] += seconds
            elif "encod" in label and "prompt" in label:
                self.phase_seconds["encode"] += seconds
            elif "decod" in label:
                self.phase_seconds["decode"] += seconds
            return
        phase = _PHASE_RE.match(stripped)
        if phase:
            label = phase.group(1).lower()
            if ("encod" in label and "text" in label) or "gemma" in label or ("encod" in label and "prompt" in label):
                VIDEO_PROGRESS.set_phase(PHASE_ENCODING, message="Encoding prompt")
            elif "decod" in label or "vae" in label or "audio" in label:
                VIDEO_PROGRESS.set_phase(PHASE_DECODING, message="Decoding video")

    def timings(self, total_s: float, startup_s: float = 0.0, warm: bool = False) -> dict[str, Any]:
        """``startup_s``: process start and imports before the first line (CLI only), counted as load.
        Denoise is the rest: both stages, the latent upscale and the image conditioning."""
        load = self.phase_seconds["load"] + startup_s
        encode, decode = self.phase_seconds["encode"], self.phase_seconds["decode"]
        return {
            "loadSeconds": round(load, 2),
            "encodeSeconds": round(encode, 2),
            "denoiseSeconds": round(max(0.0, total_s - load - encode - decode), 2),
            "decodeSeconds": round(decode, 2),
            "totalSeconds": round(total_s, 2),
            "warm": warm,
        }


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


class WarmFallback(RuntimeError):
    """The warm worker could not make this clip; the CLI path makes it instead."""


_WORKER: Any = None
_WORKER_LOCK = threading.Lock()


def warm_worker() -> Any:
    """The one warm ltx-2-mlx worker of this backend (created on first use, started on first clip)."""
    global _WORKER
    from backend_service.warm_worker_client import WarmWorker

    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = WarmWorker(
                name="ltx-2-mlx",
                kind="video",
                command=lambda: [str(LTX2_VENV_DIR / "bin" / "python"), str(_WORKER_SCRIPT)],
                env=Ltx2MlxVideoEngine._child_env,
                log_path=_WORKER_LOG,
                hard_cancel_after_s=_WORKER_HARD_CANCEL_S,
            )
        return _WORKER


def warm_worker_wanted() -> bool:
    from backend_service.warm_worker_client import can_start, total_memory_gb, warm_workers_enabled

    return (
        warm_workers_enabled()
        and total_memory_gb() >= WARM_MIN_MEMORY_GB
        and _WORKER_SCRIPT.exists()
        and can_start(_WORKER, WARM_START_AVAILABLE_GB)
    )


def release_warm_worker(reason: str = "unloaded") -> bool:
    """Stop the warm worker if it idles (never kills a running clip)."""
    return _WORKER.stop_if_idle(reason) if _WORKER is not None else False


_WARM_REPO: str | None = None


def warm_loaded_repo() -> str | None:
    """The repo whose pipeline the warm worker holds, so memory gates count it as resident."""
    return _WARM_REPO if _WORKER is not None and _WORKER.running() else None


def _remember_warm_repo(repo: str) -> None:
    global _WARM_REPO
    _WARM_REPO = repo


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

    # Timing split of the last CLI run (C1), filled by ``_run``.
    _cli_timings: dict[str, Any] | None = None

    def preload(self, repo: str) -> VideoRuntimeStatus:
        # The warm worker loads on the first clip (its load is the first clip's own phase).
        return self.probe()

    def unload(self, repo: str | None = None) -> VideoRuntimeStatus:
        release_warm_worker("unloaded")
        return self.probe()

    def release(self, reason: str = "unloaded") -> bool:
        return release_warm_worker(reason)

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
        timings: dict[str, Any] | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="chaosengine-ltx2-") as tmpdir:
                output = Path(tmpdir) / "clip.mp4"
                args = self._build_command(config, pack, gemma, width, height, seed, output)
                if warm_worker_wanted():
                    try:
                        timings = self._run_warm(args, output)
                        _remember_warm_repo(config.repo)
                    except WarmFallback as exc:
                        notes.append(f"the warm worker failed ({exc}), so this clip ran as a one-off process")
                        timings = None
                if timings is None:
                    self._run(args, output)
                    timings = dict(self._cli_timings or {}) or None
                elif timings.get("warm"):
                    notes.append("pipeline kept loaded from the previous clip")
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
            timings=timings,
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
        # Image-to-video: --image PATH FRAME_IDX STRENGTH, repeated for the start and the end frame.
        image_args: list[str] = []
        if config.startImagePath:
            image_args += ["--image", config.startImagePath, "0", f"{config.imageStrength:g}"]
        if config.endImagePath:
            image_args += ["--image", config.endImagePath, str(max(1, config.numFrames - 1)), f"{config.imageStrength:g}"]
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
            *image_args,
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
        from backend_service.progress import VIDEO_PROGRESS, GenerationCancelled

        started = time.perf_counter()
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._child_env(),
        )
        parser = Ltx2OutputParser()
        tail = parser.tail
        first_line_at: float | None = None
        self._cli_timings = None
        try:
            assert proc.stdout is not None
            # tqdm redraws with carriage returns; text mode turns those into lines.
            for line in proc.stdout:
                stripped = line.rstrip()
                if not stripped:
                    continue
                if first_line_at is None:
                    first_line_at = time.perf_counter()
                parser.feed(stripped)
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
        total = time.perf_counter() - started
        self._cli_timings = parser.timings(total, startup_s=(first_line_at or started) - started)

    def _run_warm(self, args: list[str], output: Path) -> dict[str, Any]:
        """One clip through the warm worker; returns its timing split. ``WarmFallback`` = use the CLI."""
        from backend_service.progress import VIDEO_PROGRESS, GenerationCancelled
        from backend_service.warm_worker_client import WarmWorkerCancelled

        parser = Ltx2OutputParser()

        def on_event(event: dict[str, Any]) -> None:
            if event.get("event") == "line" and event.get("text"):
                parser.feed(str(event["text"]))

        worker = warm_worker()
        started = time.perf_counter()
        try:
            reply = worker.request(
                "generate",
                timeout=_WORKER_TIMEOUT_S,
                stall_s=_WORKER_STALL_S,
                on_event=on_event,
                cancel_check=VIDEO_PROGRESS.is_cancelled,
                job=f"vid-{uuid.uuid4().hex[:12]}",
                argv=args[1:],
                keep_resident=True,
            )
        except WarmWorkerCancelled as exc:
            raise GenerationCancelled("Video generation cancelled by user") from exc
        except Exception as exc:  # noqa: BLE001 - any worker failure falls back to the CLI
            if VIDEO_PROGRESS.is_cancelled():
                raise GenerationCancelled("Video generation cancelled by user") from exc
            worker.stop_if_idle("it failed a request")
            raise WarmFallback(f"{type(exc).__name__}: {exc}") from exc
        if not output.exists():
            worker.stop_if_idle("it wrote no video")
            raise WarmFallback("the worker wrote no video")
        worker.after_job(reply)
        return parser.timings(time.perf_counter() - started, warm=bool(reply.get("warm")))


def status_payload() -> dict[str, Any]:
    """Lightweight probe for the setup route."""
    return {
        "installed": is_installed(),
        "version": read_version() if is_installed() else None,
        "supported": _apple_silicon(),
        "repos": sorted(SUPPORTED_REPOS),
    }
