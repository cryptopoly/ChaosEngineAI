"""Lip sync: a start still and a vocal clip become a clip whose mouth follows the audio (LTX-2.3 audio to video).

NVIDIA (D-103 route b, measured on an RTX 4090): our own denoise loop around the 4-bit LTX-2.3 diffusers row. The
clip's log-mel spectrogram is encoded with the pack's audio VAE and held fixed on every transformer call, so the
picture is denoised against clean audio. It runs in ``lipsync_worker.py``, a warm worker process started from the
backend's own Python (the weights never load into the FastAPI process; a stuck run is killed without touching it).
Apple Silicon: ltx-2-mlx's ``a2v --distilled`` command on the LTX-2.3 MLX packs, run as one process per clip like the
existing ltx-2-mlx path. **The Mac path has never been run** (no Mac was available when it was wired).

Either way the clip's audio is first made into the shot's exact sound: stereo (mono is up-mixed), 48 kHz, trimmed or
padded with silence to ``frames / fps`` seconds, where the frames follow LTX's 8k+1 rule and the shot lasts 3 to 8 s.
That file is what the model hears and what is muxed into the finished mp4; the sound the pipeline decodes is dropped.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

CUDA_REPOS: frozenset[str] = frozenset({"OzzyGT/LTX-2.3-Distilled-bnb-nf4"})
# LTX-2.3 packs only: the landscape study found 2.5 worse at following outside vocals.
MLX_REPOS: frozenset[str] = frozenset({"dgrauet/ltx-2.3-mlx-q4"})
AUDIO_EXTENSIONS = frozenset({".wav", ".flac"})
MIN_SECONDS, MAX_SECONDS = 3.0, 8.0
SAMPLE_RATE = 48000
STEPS = 8  # distilled schedule (CUDA); the Mac path runs 8 + 3 refine passes

# Measured in the spike (docs/research/lip-sync-spike-windows-2026-11.md) on an RTX 4090 24 GB, 64 GB RAM, 768x512,
# 121 frames: whole-card GPU peak 21.7-21.8 GB above an idle 1.6 GB (22.5 GB on the phase-1 cold shot), host RSS peak
# 36.4-37.0 GB. The catalog row's
# ``lipSync`` block carries these; the constants are the fallback.
DEFAULT_NEEDS = {"cardGb": 24.0, "freeVramGb": 22.0, "hostRamGb": 37.0}
# A running worker already holds its CUDA context and its host copy of the weights.
_WARM_FREE_VRAM_ALLOWANCE_GB = 1.5

_WORKER_LOG = Path.home() / ".chaosengine" / "logs" / "lipsync-worker.log"
_WORKER_TIMEOUT_S = 3600.0
_WORKER_STALL_S = 600.0  # the load (35 s from the page cache, minutes from a cold disk) and the decode send no steps
_WORKER_HARD_CANCEL_S = 30.0


# Sung vocals (``lipSyncSinging``): measured in docs/research/lip-sync-singing-experiments-2026-11.md (experiment 4),
# naming the singing in the prompt removed the takes where the face listens with a shut mouth instead of singing
# (0 of 24 takes with it, 3 of 12 without) and raised the mouth/vocal correlation. Off by default: it is wrong for
# speech.
SINGING_CUE = "singing, mouth opening and closing with each syllable"


def singing_prompt(prompt: str) -> str:
    """``prompt`` with the singing cue appended, unless it already asks for the mouth to follow the syllables."""
    if "syllable" in prompt.lower():
        return prompt
    return f"{prompt.strip().rstrip('.')}, {SINGING_CUE}."


class LipSyncError(RuntimeError):
    """A request lip sync cannot run: ``status`` is the HTTP code the route answers with."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# what a request may ask for
# ---------------------------------------------------------------------------


def engine_for(variant: dict[str, Any] | None) -> str | None:
    """``"cuda"``, ``"mlx"`` or None for a catalog row."""
    repo = str((variant or {}).get("repo") or "")
    if repo in CUDA_REPOS:
        return "cuda"
    if repo in MLX_REPOS:
        return "mlx"
    return None


def supports(variant: dict[str, Any] | None) -> bool:
    return engine_for(variant) is not None


def needs(variant: dict[str, Any] | None) -> dict[str, float]:
    block = (variant or {}).get("lipSync") or {}
    return {key: float(block.get(key) or default) for key, default in DEFAULT_NEEDS.items()}


def resolve_audio(raw: str) -> Path:
    """The real path of the vocal clip (a local file or an ``upload:`` handle), WAV or FLAC."""
    from backend_service.storyboard.media import MediaError, resolve_source

    try:
        path = resolve_source(raw, {"audio"})
    except MediaError as exc:
        raise LipSyncError(str(exc)) from exc
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise LipSyncError(f"Lip sync takes a WAV or FLAC file, not '{path.suffix}': {path.name}")
    return path


def shot_frames(frames: int, fps: int) -> int:
    """The frame count to render: LTX's 8k+1 rule (rounded down), checked against the 3 to 8 s shot range."""
    if fps <= 0:
        raise LipSyncError("The frame rate must be above 0.")
    aligned = max(9, ((int(frames) - 1) // 8) * 8 + 1)
    seconds = (aligned - 1) / fps
    if seconds < MIN_SECONDS - 1e-6 or seconds > MAX_SECONDS + 1e-6:
        low = ((int(MIN_SECONDS * fps) + 7) // 8) * 8 + 1
        high = (int(MAX_SECONDS * fps) // 8) * 8 + 1
        raise LipSyncError(
            f"A lip-sync shot lasts {MIN_SECONDS:g} to {MAX_SECONDS:g} s: at {fps} fps that is {low} to {high} frames "
            f"(8k+1), and {frames} frames is {seconds:.2f} s. Split a longer vocal into several shots."
        )
    return aligned


_LOAD_SECONDS = 35.0  # the spike's measured load from the page cache, part of every fresh-process shot


def estimate_seconds(variant: dict[str, Any] | None, width: int, height: int, frames: int) -> float | None:
    """Seconds for one shot from the row's measured ``lipSync`` figures: the load plus the rest scaled by pixels x
    frames against the measured shot. None when the row has no measurement (the Mac path)."""
    block = (variant or {}).get("lipSync") or {}
    per_shot, ref = block.get("secondsPerShot"), block.get("estimateRef") or {}
    if not isinstance(per_shot, (int, float)) or not ref:
        return None
    scale = (width * height * frames) / float(ref["width"] * ref["height"] * ref["frames"])
    return round(_LOAD_SECONDS + (float(per_shot) - _LOAD_SECONDS) * scale, 1)


def memory_refusal(
    *,
    need: dict[str, float],
    card_gb: float | None,
    free_vram_gb: float | None,
    available_ram_gb: float | None,
    worker_loaded: bool,
) -> str | None:
    """A plain message naming what is short, or None when the run fits. Unreadable figures never block."""
    short: list[str] = []
    if card_gb is not None and card_gb < need["cardGb"] * 0.95:
        return (
            f"Lip sync with the 4-bit LTX-2.3 model needs a {need['cardGb']:.0f} GB NVIDIA card; this one has "
            f"{card_gb:.1f} GB. It cannot run on this GPU."
        )
    want_vram = need["freeVramGb"] - (_WARM_FREE_VRAM_ALLOWANCE_GB if worker_loaded else 0.0)
    if free_vram_gb is not None and free_vram_gb < want_vram:
        short.append(
            f"about {want_vram:.1f} GB of free GPU memory (the whole card); {free_vram_gb:.1f} GB is free, "
            f"{want_vram - free_vram_gb:.1f} GB short. Close other programs that use the GPU or unload other models"
        )
    if not worker_loaded and available_ram_gb is not None and available_ram_gb < need["hostRamGb"]:
        short.append(
            f"about {need['hostRamGb']:.0f} GB of free system memory while it runs; {available_ram_gb:.1f} GB is "
            f"free, {need['hostRamGb'] - available_ram_gb:.1f} GB short. Close other apps or unload other models"
        )
    if not short:
        return None
    return "Lip sync needs " + "; and ".join(short) + "."


# ---------------------------------------------------------------------------
# audio in, audio out
# ---------------------------------------------------------------------------


def _ffmpeg() -> str:
    from backend_service.storyboard import ffmpeg_bin

    found = ffmpeg_bin.find_ffmpeg(allow_path=True)
    if found is None:
        raise LipSyncError("ffmpeg was not found, so the vocal clip cannot be prepared. Install the video tools in Setup.", 409)
    return found[0]


def prepare_audio_command(exe: str, src: Path, dst: Path, start: float, seconds: float) -> list[str]:
    """ffmpeg arguments that make ``dst``: stereo 48 kHz 16-bit WAV, ``seconds`` long from ``start`` (padded with
    silence when the clip is shorter). ffmpeg reads the type from the content, not the name, so a ``.wav`` holding a
    playlist could fetch URLs: inputs are held to local files like every Storyboard ffmpeg call (``PROTOCOLS``)."""
    from backend_service.storyboard.ffmpeg_bin import PROTOCOLS

    return [
        exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-protocol_whitelist", PROTOCOLS,
        "-ss", f"{max(0.0, start):.3f}", "-i", str(src),
        "-map", "0:a:0", "-ac", "2", "-ar", str(SAMPLE_RATE),
        "-af", "apad", "-t", f"{seconds:.6f}",
        "-c:a", "pcm_s16le", str(dst),
    ]


def prepare_audio(src: Path, dst: Path, start: float, seconds: float) -> None:
    command = prepare_audio_command(_ffmpeg(), src, dst, start, seconds)
    run = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if run.returncode != 0 or not dst.is_file() or dst.stat().st_size <= 44:
        last = (run.stderr or "").strip().splitlines()[-1:] or ["no output"]
        raise LipSyncError(f"The vocal clip could not be read: {last[0][:200]}")


def mux_command(exe: str, video: Path, wav: Path, out: Path) -> list[str]:
    from backend_service.storyboard.ffmpeg_bin import PROTOCOLS

    return [
        exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-protocol_whitelist", PROTOCOLS,
        "-i", str(video), "-i", str(wav),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        str(out),
    ]


def mux(video: Path, wav: Path, out: Path) -> None:
    run = subprocess.run(mux_command(_ffmpeg(), video, wav, out), capture_output=True, text=True, timeout=600)
    if run.returncode != 0 or not out.is_file():
        last = (run.stderr or "").strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"ffmpeg could not add the vocal to the clip: {last[0][:200]}")


# ---------------------------------------------------------------------------
# Mac: ltx-2-mlx a2v (UNTESTED)
# ---------------------------------------------------------------------------


def mlx_a2v_command(
    *, pack: Path, gemma: Path | None, prompt: str, width: int, height: int, frames: int, fps: int, seed: int,
    audio: Path, image: Path, output: Path,
) -> list[str]:
    """``ltx-2-mlx a2v --distilled`` (ltx-2-mlx 0.18.0, ``ltx_pipelines_mlx.cli``): the distilled two-stage flow with
    the input audio frozen in both stages and the still pinned on frame 0. Never run on a Mac yet."""
    from backend_service.ltx2_mlx_runtime import LTX2_VENV_DIR

    gemma_args = ["--gemma", str(gemma)] if gemma is not None else []
    return [
        str(LTX2_VENV_DIR / "bin" / "ltx-2-mlx"),
        "a2v",
        "--distilled",
        "--model", str(pack),
        *gemma_args,
        "--prompt", prompt,
        "-H", str(height),
        "-W", str(width),
        "--frames", str(frames),
        "--frame-rate", f"{fps:g}",
        "--seed", str(seed),
        "--audio", str(audio),
        "--image", str(image), "0", "1.0",
        "--output", str(output),
    ]


def _run_mlx(*, repo: str, model_name: str, prompt: str, width: int, height: int, frames: int, fps: int, seed: int,
             audio: Path, image: Path, output: Path) -> dict[str, Any]:
    from backend_service import ltx2_mlx_runtime as ltx2

    engine = ltx2.Ltx2MlxVideoEngine()
    status = engine.probe()
    if not status.realGenerationAvailable:
        raise LipSyncError(status.message, 409)
    pack = ltx2._local_snapshot(repo)
    if pack is None or ltx2.missing_pack_files(pack, repo):
        raise LipSyncError(f"{model_name} is not downloaded completely. Download it from Video Discover.", 409)
    gemma = ltx2._local_snapshot(ltx2.GEMMA_REPO)
    if gemma is None or not (gemma / "model.safetensors.index.json").exists():
        raise LipSyncError(f"The text encoder {ltx2.GEMMA_REPO} is not downloaded. Download {model_name} again.", 409)
    width, height = ltx2.align_size(width, height)
    args = mlx_a2v_command(pack=pack, gemma=gemma, prompt=prompt, width=width, height=height, frames=frames, fps=fps,
                           seed=seed, audio=audio, image=image, output=output)
    engine._run(args, output)  # streams progress into VIDEO_PROGRESS, honours Cancel
    return {"width": width, "height": height, "timings": engine._cli_timings, "runtimeLabel": "LTX-2.3 a2v (MLX native)"}


# ---------------------------------------------------------------------------
# NVIDIA: the warm worker
# ---------------------------------------------------------------------------

_WORKER: Any = None
_WORKER_LOCK = threading.Lock()
_LOADED_REPO: str | None = None


def _worker_env() -> dict[str, str]:
    from backend_service.inference._utils import _isolated_child_env

    env = dict(_isolated_child_env())
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["HF_HUB_OFFLINE"] = "1"  # generation reads local files only
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def worker() -> Any:
    global _WORKER
    from backend_service.warm_worker_client import WarmWorker

    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = WarmWorker(
                name="lip-sync",
                kind="video",
                command=lambda: [sys.executable, "-m", "backend_service.lipsync_worker"],
                env=_worker_env,
                log_path=_WORKER_LOG,
                hard_cancel_after_s=_WORKER_HARD_CANCEL_S,
            )
        return _WORKER


def loaded_repo() -> str | None:
    """The repo whose pipeline the worker holds (counted as resident by the gate), or None."""
    return _LOADED_REPO if _WORKER is not None and _WORKER.running() else None


def release_worker(reason: str = "unloaded") -> bool:
    """Stop the worker when it idles (never kills a running shot); frees its ~37 GB of RAM and its GPU context."""
    global _LOADED_REPO
    stopped = _WORKER.stop_if_idle(reason) if _WORKER is not None else False
    if stopped:
        _LOADED_REPO = None
    return stopped


def _run_cuda(*, repo: str, prompt: str, width: int, height: int, frames: int, fps: int, seed: int,
              audio: Path, image: Path, output: Path) -> dict[str, Any]:
    global _LOADED_REPO
    from backend_service.helpers.hf_local import local_snapshot_path
    from backend_service.progress import PHASE_DECODING, PHASE_DIFFUSING, PHASE_ENCODING, VIDEO_PROGRESS, GenerationCancelled
    from backend_service.warm_worker_client import WarmWorkerCancelled, WarmWorkerError

    try:
        snapshot = Path(local_snapshot_path(repo))
    except Exception as exc:  # noqa: BLE001 - not downloaded, or an unreadable cache
        raise LipSyncError(f"{repo} is not downloaded. Download it from Video Discover.", 409) from exc

    def on_event(event: dict[str, Any]) -> None:
        if event.get("event") == "step":
            VIDEO_PROGRESS.set_step(int(event.get("step") or 0), total=int(event.get("total") or STEPS))
        elif event.get("event") == "stage":
            stage = event.get("stage")
            if stage == "encode":
                VIDEO_PROGRESS.set_phase(PHASE_ENCODING, message="Encoding the vocal clip")
            elif stage == "denoise":
                VIDEO_PROGRESS.set_phase(PHASE_DIFFUSING, message="Denoising against the vocal")
            elif stage == "decode":
                VIDEO_PROGRESS.set_phase(PHASE_DECODING, message="Decoding video")

    if _LOADED_REPO not in (None, repo):
        release_worker("another model was asked for")  # unload before reload: never two pipelines at once
    started = time.perf_counter()
    try:
        reply = worker().request(
            "generate",
            timeout=_WORKER_TIMEOUT_S,
            stall_s=_WORKER_STALL_S,
            on_event=on_event,
            cancel_check=VIDEO_PROGRESS.is_cancelled,
            job=f"lip-{uuid.uuid4().hex[:12]}",
            snapshot=str(snapshot), repo=repo, prompt=prompt, width=width, height=height, frames=frames, fps=fps,
            seed=seed, audio=str(audio), image=str(image), output=str(output), steps=STEPS,
        )
    except WarmWorkerCancelled as exc:
        raise GenerationCancelled("Lip sync cancelled by user") from exc
    except WarmWorkerError as exc:
        _LOADED_REPO = None
        raise RuntimeError(f"Lip sync failed: {exc}") from exc
    _LOADED_REPO = repo
    if not output.is_file():
        raise RuntimeError("The lip-sync worker finished without writing a video.")
    total = round(time.perf_counter() - started, 2)
    timings = {
        "loadSeconds": reply.get("load_s"), "encodeSeconds": reply.get("encode_s"),
        "denoiseSeconds": reply.get("denoise_s"), "decodeSeconds": reply.get("decode_s"),
        "totalSeconds": total, "warm": bool(reply.get("warm")),
        "gpuPeakGb": reply.get("vram_peak_gb"), "hostPeakGb": reply.get("host_peak_gb"),
    }
    return {"width": width, "height": height, "timings": timings, "runtimeLabel": "LTX-2.3 audio to video (diffusers, 4-bit)"}


# ---------------------------------------------------------------------------
# one shot
# ---------------------------------------------------------------------------


def generate(body: Any, variant: dict[str, Any]) -> dict[str, Any]:
    """Make one lip-sync shot for a checked request (``audioPath`` and ``startImagePath`` resolved by the route).

    Returns the artifact fields the video gallery saves (``videoBytes`` included). Raises ``LipSyncError``,
    ``GenerationCancelled`` or ``RuntimeError``.
    """
    from backend_service.helpers.generation_limits import normalize_video_size, size_limits
    from backend_service.helpers.video_audio import mp4_has_audio_bytes
    from backend_service.progress import PHASE_LOADING, VIDEO_PROGRESS
    from backend_service.video_runtime.device import _resolve_video_seed

    engine = engine_for(variant)
    if engine is None:
        raise LipSyncError(f"{variant.get('name')} cannot lip-sync. Use LTX-2.3 distilled 4-bit (NVIDIA) or LTX-2.3 MLX (Mac).")
    repo = str(variant["repo"])
    frames = shot_frames(body.numFrames, body.fps)
    width, height, size_note = normalize_video_size(body.width, body.height, repo, model_name=str(variant.get("name") or ""),
                                                    **size_limits(variant).as_kwargs())
    seconds = frames / body.fps
    seed = _resolve_video_seed(body.seed)
    notes = [note for note in (size_note,) if note]
    if frames != body.numFrames:
        notes.append(f"frames {body.numFrames} changed to {frames} (8k+1)")
    notes.append(f"lip sync: the vocal clip from {body.audioStartSeconds:g} s, {seconds:.2f} s long, held fixed while "
                 f"the picture is made; it is the clip's sound")
    prompt = body.prompt
    if getattr(body, "lipSyncSinging", False):
        prompt = singing_prompt(body.prompt)
        if prompt != body.prompt:
            notes.append(f"sung vocal: '{SINGING_CUE}' added to the prompt")
    if engine == "mlx":
        notes.append("Mac lip sync (ltx-2-mlx a2v) has not been tested yet")

    VIDEO_PROGRESS.begin(
        run_label=f"{variant['name']} · lip sync · {frames}f @ {width}x{height}",
        total_steps=STEPS if engine == "cuda" else 11,
        phase=PHASE_LOADING,
        message=f"Loading {variant['name']}",
    )
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="chaosengine-lipsync-") as tmp:
            folder = Path(tmp)
            wav, silent, final = folder / "vocal.wav", folder / "silent.mp4", folder / "shot.mp4"
            prepare_audio(Path(body.audioPath), wav, body.audioStartSeconds, seconds)
            run = _run_cuda if engine == "cuda" else _run_mlx
            extra = {} if engine == "cuda" else {"model_name": str(variant["name"])}
            result = run(repo=repo, prompt=prompt, width=width, height=height, frames=frames, fps=body.fps,
                         seed=seed, audio=wav, image=Path(body.startImagePath), output=silent, **extra)
            mux(silent, wav, final)
            data = final.read_bytes()
    finally:
        VIDEO_PROGRESS.finish()
    from datetime import datetime, timezone

    return {
        "artifactId": f"vid-{uuid.uuid4().hex[:12]}",
        "modelId": body.modelId,
        "modelName": variant["name"],
        "prompt": body.prompt,
        "negativePrompt": "",
        "width": result["width"],
        "height": result["height"],
        "numFrames": frames,
        "fps": body.fps,
        "steps": STEPS,
        "guidance": 1.0,
        "seed": seed,
        "createdAt": datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z",
        "durationSeconds": round(time.perf_counter() - started, 2),
        "timings": result.get("timings"),
        "clipDurationSeconds": round(seconds, 3),
        "videoBytes": data,
        "videoMimeType": "video/mp4",
        "videoExtension": "mp4",
        "runtimeLabel": result["runtimeLabel"],
        "runtimeNote": "; ".join(notes) + ".",
        "hasAudio": mp4_has_audio_bytes(data),
        "startImagePath": body.startImagePath,
        "audioPath": body.audioPath,
        "lipSync": True,
    }
