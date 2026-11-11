"""The voice venv: where speech packages live, and the job that builds it.

Speech packages install into ``~/.chaosengine/voice-venv``, never into the
backend's own environment or the extras overlay: mlx-audio, misaki and spaCy
pull versions of transformers / numpy / mlx that must not shadow the app's
(FU-098). The venv is created from the Python the backend runs on, so it works
the same on macOS, Windows and Linux without a system Python.

Two lanes:

* ``mlx`` (Apple Silicon): Parakeet and Whisper on MLX for speech-to-text,
  Kokoro through mlx-audio for speech.
* ``portable`` (Windows, Linux, Intel Macs): faster-whisper (CUDA when an
  NVIDIA GPU and its libraries are present, CPU otherwise) and kokoro-onnx.

The Kokoro quick voices are an add-on (D-75), ticked by default in the install
form: the install downloads their files only when ``"kokoro"`` is among the
extras, and ``installed_extras`` reports Kokoro from its files on disk.

``CHAOSENGINE_VOICE_LANE=portable`` forces the portable lane on a Mac, which is
how it gets tested there.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# CHAOSENGINE_VOICE_VENV relocates it (tests, and trying the portable lane on a Mac).
VOICE_VENV_DIR = Path(os.environ.get("CHAOSENGINE_VOICE_VENV") or Path.home() / ".chaosengine" / "voice-venv")
VOICE_DATA_DIR = Path.home() / ".chaosengine" / "voice"
_MARKER = VOICE_VENV_DIR / "chaosengine-voice.json"
_SCHEMA = 1

KOKORO_MLX_REPO = "prince-canuma/Kokoro-82M"

# Silero VAD (MIT, 2 MB) for conversation mode's turn-taking.
VAD_REPO = "onnx-community/silero-vad"
VAD_REVISION = "e71cae966052b992a7eca6b17738916ce0eca4ec"
VAD_FILE = "onnx/model.onnx"

# Speaker labels (diarization) through sherpa-onnx: pyannote segmentation 3.0
# (MIT, 6 MB) finds who-speaks-when, 3D-Speaker CAM++ (Apache-2.0, 30 MB)
# tells the voices apart. ONNX on every platform.
DIARIZE_FILES = {
    "segmentation": (
        "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
        "9403a6902bb58e3d5ae8c7e77c3422de279db2e0",
        "model.onnx",
    ),
    "embedding": (
        "csukuangfj/speaker-embedding-models",
        "0743f301363dec56491a490f6d6cbc9d67f9a3bf",
        "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    ),
}

# Hidden watermark for cloned-voice output (D-47): Meta's AudioSeal, code and weights MIT
# (checked 2026-11-07: PyPI classifier and README "full MIT license (including the license for the
# model weights)", Hub card licence mit, not gated). Installed into the voice venv on demand, the
# first time a cloning model is downloaded; never part of the base pack.
WATERMARK_PACKAGES = ["audioseal==0.2.1", "omegaconf>=2.3,<2.4", "einops>=0.8,<0.9"]
WATERMARK_REPO = "facebook/audioseal"
WATERMARK_REVISION = "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba"
WATERMARK_FILES = {"generator": "generator_base.pth", "detector": "detector_base.pth"}

# What this version of the pack provides; an older pack lacks some and the
# Models page offers an update. "speech-wave2": mlx-audio 0.6.4, the version the
# VoxCPM2, Chatterbox and Orpheus loaders were checked against (D-70).
FEATURES = ["vad", "speakers", "speech-wave2"]
# Features that only mean something on some lanes; a pack on another lane is not asked to update for them.
_LANE_FEATURES = {"speech-wave2": ("mlx",)}

# spaCy's small English model: misaki's English front-end downloads it at
# first use otherwise, through pip, which a packaged app cannot rely on.
_EN_CORE_WEB_SM = (
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)

# Versions checked together on an M4 Max (2026-10-26); bump deliberately.
_MLX_PACKAGES = [
    "mlx-audio>=0.6.4,<0.7",
    "misaki[en]>=0.9.4,<0.10",
    "parakeet-mlx>=0.5.3,<0.6",
    "mlx-whisper>=0.4.3,<0.5",
    "av>=13",
    "onnxruntime>=1.20",
    "sherpa-onnx>=2.0.1,<2.1",
    _EN_CORE_WEB_SM,
]
_PORTABLE_PACKAGES = [
    "faster-whisper>=1.2.1,<1.3",
    "kokoro-onnx>=0.6.1,<0.7",
    "av>=13",
    "onnxruntime>=1.20",
    "sherpa-onnx>=2.0.1,<2.1",
]
# ctranslate2 4.x runs on CUDA 12 with cuDNN 9; pip ships both libraries.
_CUDA_PACKAGES = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12>=9,<10"]

# Optional extras: name -> (lanes it applies to, packages, label). "speech-gpu" installs with its own commands
# (``speech_gpu_commands``), so its package list here is empty.
# "kokoro" has no packages either: its files are a download (the MLX repo, or the kokoro-onnx files), and its
# installed state is those files on disk (``kokoro_installed``), not the marker, so packs from before D-75 count it.
EXTRAS: dict[str, tuple[tuple[str, ...], list[str], str]] = {
    "kokoro": (("mlx", "portable"), [], "Kokoro quick voices"),
    "chinese": (("mlx",), ["misaki[zh]>=0.9.4,<0.10"], "Mandarin Chinese voices"),
    "speech-gpu": (("portable",), [], "More speech models on the NVIDIA GPU (Chatterbox, VoxCPM2, Orpheus)"),
}

# Speech GPU add-on (D-70 Phase 6, FU-149). NOT VERIFIED: written on a Mac; the RTX 4090 session runs it
# (docs/plans/rtx4090-test.md section 2). The two model packages go in with --no-deps: chatterbox-tts pins torch 2.6,
# transformers 5.2, gradio and numpy<2, and voxcpm pulls gradio, funasr, modelscope, datasets and matplotlib, none of
# which inference needs. What their model code imports is listed by hand instead, every licence checked (MIT,
# Apache-2.0, BSD, ISC). Left out on purpose: pykakasi (GPL-3.0; Chatterbox uses it for Japanese readings, so fugashi + unidic-lite
# stand in for it, see voice_ja.py) and the onnxruntime-gpu build (the SNAC decoder for Orpheus is small and runs on
# the CPU onnxruntime the pack already has). torch 2.9 with CUDA 12.8 wheels covers the RTX 40 and 50 series.
SPEECH_GPU_TORCH = ["torch==2.9.1", "torchaudio==2.9.1"]
SPEECH_GPU_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
SPEECH_GPU_PACKAGES = [
    "transformers==5.2.0",
    "diffusers==0.29.0",
    "safetensors==0.5.3",
    "librosa==0.11.0",
    "conformer==0.3.2",
    "s3tokenizer==0.3.0",
    "resemble-perth==1.0.1",
    "pyloudnorm==0.2.0",
    "omegaconf>=2.3,<2.4",
    "spacy-pkuseg==1.0.2",
    "einops>=0.8,<0.9",
    "inflect>=7.6,<8",
    "addict==2.4.0",
    "wetext==0.2.1",
    "soundfile>=0.13,<0.15",
    "torchcodec==0.8.1",
    "fugashi==1.5.2",  # Japanese readings for Chatterbox (backend_service/voice_ja.py), instead of GPL pykakasi
    "unidic-lite==1.0.8",
    "pydantic>=2,<3",
]
# chatterbox-tts: PyPI's newest (0.1.7) only knows the v2 Multilingual model; V3 (`from_local(..., t3_model="v3")`,
# weights `t3_mtl23ls_v3.safetensors`) is only in the upstream repository, from commit 65b1843 (2026-06-10). Pinned to
# a source archive of the commit the RTX 4090 run tested (same 0.1.7 version string, MIT). Move back to a PyPI pin
# once a release carries V3.
CHATTERBOX_REF = "ab55923c27757c90cfcc407baa57dc25a7d25217"
CHATTERBOX_PACKAGE = f"chatterbox-tts @ https://github.com/resemble-ai/chatterbox/archive/{CHATTERBOX_REF}.zip"
SPEECH_GPU_NO_DEPS = [CHATTERBOX_PACKAGE, "voxcpm==2.0.3"]
_SPEECH_GPU_CHECK = (
    "import torch, chatterbox.tts_turbo, chatterbox.mtl_tts, voxcpm; "
    "assert torch.cuda.is_available(), 'PyTorch cannot see the NVIDIA GPU'; print('speech gpu ok')"
)


def speech_gpu_supported() -> tuple[bool, str]:
    """Whether the speech GPU add-on can install here, and why not."""
    if lane() != "portable":
        return False, "The speech GPU add-on is for Windows and Linux; on a Mac these models run on MLX."
    if not has_nvidia_gpu():
        return False, "The speech GPU add-on needs an NVIDIA GPU (nvidia-smi was not found)."
    if platform.machine().lower() in ("aarch64", "arm64"):
        return False, "The speech GPU add-on is not checked on Linux ARM64 yet."
    return True, ""


def speech_gpu_commands(python: str) -> list[list[str]]:
    """The add-on's install commands, in order, then an import check that also asks PyTorch for the GPU."""
    pip = [python, "-m", "pip", "install"]
    return [
        [*pip, *SPEECH_GPU_TORCH, "--index-url", SPEECH_GPU_TORCH_INDEX],
        [*pip, *SPEECH_GPU_PACKAGES],
        # --force-reinstall: the pinned chatterbox archive reports the same version (0.1.7) as the PyPI wheel an earlier
        # install left behind, and pip keeps the installed copy of an equal version without it (RTX 4090 run).
        [*pip, "--no-deps", "--force-reinstall", *SPEECH_GPU_NO_DEPS],
        [python, "-I", "-c", _SPEECH_GPU_CHECK],
    ]


def speech_gpu_installed() -> bool:
    return is_installed() and "speech-gpu" in installed_extras()


def speech_lanes() -> list[str]:
    """Catalog lanes that run here: ``mlx`` on Apple Silicon, ``cuda`` with the speech GPU add-on and its GPU."""
    if lane() == "mlx":
        return ["mlx"]
    return ["cuda"] if speech_gpu_installed() and speech_gpu_supported()[0] else []

_VERIFY_IMPORTS = {
    "mlx": [
        "av", "numpy", "mlx_audio", "misaki.en", "parakeet_mlx", "mlx_whisper", "en_core_web_sm",
        "onnxruntime", "sherpa_onnx",
    ],
    "portable": ["av", "numpy", "faster_whisper", "kokoro_onnx", "onnxruntime", "sherpa_onnx"],
}


def lane() -> str:
    forced = os.environ.get("CHAOSENGINE_VOICE_LANE", "").strip().lower()
    if forced in ("mlx", "portable"):
        return forced
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "mlx"
    return "portable"


def _base_python() -> str:
    """The interpreter the voice venv is built from.

    A venv points at its base interpreter. The packaged app's Python lives in
    a temporary extraction folder whose name changes with every app version,
    so a venv built on it would break at the next update. That interpreter is
    relocatable (about 55 MB), so it is copied to a stable folder first.
    """
    base = Path(sys.base_prefix)
    if "chaosengine-embedded-runtime" not in str(base):
        return sys.executable
    version = ".".join(map(str, sys.version_info[:3]))
    stable = Path.home() / ".chaosengine" / "voice-python" / version
    if not stable.exists():
        partial = stable.with_name(stable.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        shutil.copytree(base, partial, symlinks=True)
        partial.rename(stable)
    for candidate in (stable / "bin" / "python3", stable / "python.exe"):
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(f"No Python interpreter found in {stable}.")


def _is_windows() -> bool:
    return os.name == "nt"


def venv_python() -> Path:
    if _is_windows():
        return VOICE_VENV_DIR / "Scripts" / "python.exe"
    return VOICE_VENV_DIR / "bin" / "python"


def has_nvidia_gpu() -> bool:
    if platform.system() == "Darwin":
        return False
    return shutil.which("nvidia-smi") is not None


def voice_uses_cuda() -> bool:
    """Whether the portable lane runs speech on the GPU.

    PyPI's aarch64 CTranslate2 wheel has no CUDA, so on Linux ARM64 (DGX Spark)
    faster-whisper runs on the CPU and the CUDA 12 libraries would be dead weight.
    """
    return has_nvidia_gpu() and platform.machine().lower() not in ("aarch64", "arm64")


def packages_for(lane_name: str, extras: list[str] | None = None) -> list[str]:
    specs = list(_MLX_PACKAGES if lane_name == "mlx" else _PORTABLE_PACKAGES)
    if lane_name == "portable" and voice_uses_cuda():
        specs += _CUDA_PACKAGES
    for name in extras or []:
        lanes, extra_specs, _label = EXTRAS[name]
        if lane_name in lanes:
            specs += extra_specs
    return specs


def read_marker() -> dict[str, Any] | None:
    try:
        data = json.loads(_MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_installed() -> bool:
    marker = read_marker()
    return bool(
        marker
        and marker.get("schema") == _SCHEMA
        and marker.get("lane") == lane()
        and venv_python().exists()
    )


def installed_extras() -> list[str]:
    marker = read_marker() or {}
    names = [name for name in marker.get("extras", []) if name in EXTRAS and name != "kokoro"]
    return ["kokoro", *names] if kokoro_installed() else names


def kokoro_installed() -> bool:
    """The Kokoro quick voices' files are on disk for this lane (D-75: an add-on, left out or removed at will)."""
    return tts_data_installed()


def kokoro_onnx_dir() -> Path:
    return VOICE_DATA_DIR / "kokoro"


def tts_source() -> tuple[str, str]:
    """(engine, source) the worker loads Kokoro from on this lane."""
    if lane() == "mlx":
        return "mlx-audio", KOKORO_MLX_REPO
    return "kokoro-onnx", str(kokoro_onnx_dir())


def vad_model_path() -> Path | None:
    """The Silero VAD file in the Hugging Face cache, if downloaded."""
    from backend_service.helpers.hf_cache_paths import _hf_repo_cache_dir

    path = _hf_repo_cache_dir(VAD_REPO) / "snapshots" / VAD_REVISION / VAD_FILE
    return path if path.is_file() else None


def vad_installed() -> bool:
    """Conversation mode needs onnxruntime (voice packs from before it lack
    it; the marker records it) and the Silero model file."""
    marker = read_marker() or {}
    return is_installed() and "vad" in marker.get("features", []) and vad_model_path() is not None


def diarize_model_paths() -> dict[str, Path] | None:
    """The speaker-label model files in the Hugging Face cache, if downloaded."""
    from backend_service.helpers.hf_cache_paths import _hf_repo_cache_dir

    paths = {
        role: _hf_repo_cache_dir(repo) / "snapshots" / revision / filename
        for role, (repo, revision, filename) in DIARIZE_FILES.items()
    }
    return paths if all(p.is_file() for p in paths.values()) else None


def diarize_installed() -> bool:
    marker = read_marker() or {}
    return is_installed() and "speakers" in marker.get("features", []) and diarize_model_paths() is not None


def missing_features() -> list[str]:
    """Features this pack version provides that the installed pack lacks."""
    if not is_installed():
        return []
    have = set((read_marker() or {}).get("features", []))
    here = lane()
    return [name for name in FEATURES if name not in have and here in _LANE_FEATURES.get(name, (here,))]


def watermark_model_paths() -> dict[str, Path] | None:
    """The pinned AudioSeal generator and detector in the Hugging Face cache, if downloaded."""
    from backend_service.helpers.hf_cache_paths import _hf_repo_cache_dir

    base = _hf_repo_cache_dir(WATERMARK_REPO) / "snapshots" / WATERMARK_REVISION
    paths = {role: base / name for role, name in WATERMARK_FILES.items()}
    return paths if all(p.is_file() for p in paths.values()) else None


def _watermark_package_present() -> bool:
    """The pinned AudioSeal is in the venv (checked on disk, so an install by any route counts)."""
    version = WATERMARK_PACKAGES[0].split("==")[1]
    patterns = (f"lib/python3*/site-packages/audioseal-{version}.dist-info", f"Lib/site-packages/audioseal-{version}.dist-info")
    return any(any(VOICE_VENV_DIR.glob(pattern)) for pattern in patterns)


def watermark_installed() -> bool:
    return is_installed() and _watermark_package_present() and watermark_model_paths() is not None


def install_watermark(cancel: Any = None) -> None:
    """Install AudioSeal into the voice venv and fetch its pinned weights; records it in the marker.

    Blocking (a few minutes on a slow line); called from the cloning model download thread. ``cancel`` is an
    optional ``threading.Event``: when set, the pip step is stopped and ``InterruptedError`` is raised.
    """
    if not is_installed():
        raise RuntimeError("The voice runtime is not installed. Install it from Voice > Models.")
    if not _watermark_package_present():
        process = subprocess.Popen(
            [str(venv_python()), "-m", "pip", "install", "--quiet", *WATERMARK_PACKAGES],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            env=worker_env(),
        )
        # Wait in short steps so Cancel can stop pip; the output is read after it exits (it is quiet).
        while True:
            try:
                output, _ = process.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    process.terminate()
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                    raise InterruptedError("cancelled") from None
        if process.returncode != 0:
            raise RuntimeError(f"Installing the watermark component failed: {(output or '')[-800:]}")
    from huggingface_hub import hf_hub_download

    for name in WATERMARK_FILES.values():
        if cancel is not None and cancel.is_set():
            raise InterruptedError("cancelled")
        hf_hub_download(WATERMARK_REPO, name, revision=WATERMARK_REVISION)
    marker = read_marker() or {}
    marker["watermark"] = WATERMARK_PACKAGES[0]
    _MARKER.write_text(json.dumps(marker), encoding="utf-8")


def tts_data_installed() -> bool:
    if lane() == "mlx":
        from backend_service.helpers.hf_cache_paths import _hf_repo_snapshot_dir

        return _hf_repo_snapshot_dir(KOKORO_MLX_REPO) is not None
    from backend_service import voice_kokoro_onnx

    return voice_kokoro_onnx.is_installed()


def download_kokoro(lane_name: str) -> None:
    """Fetch the Kokoro quick voices for ``lane_name`` (the install job, when the add-on is ticked)."""
    if lane_name == "mlx":
        from huggingface_hub import snapshot_download

        snapshot_download(KOKORO_MLX_REPO)
        return
    from backend_service import voice_kokoro_onnx

    if not voice_kokoro_onnx.is_installed():
        voice_kokoro_onnx._download_worker()
        state = voice_kokoro_onnx.download_status()
        if state.get("state") == "failed":
            raise RuntimeError(f"Kokoro voice files: {state.get('error')}")


def worker_env() -> dict[str, str]:
    """Environment for the voice worker and the install commands."""
    from backend_service.inference._utils import _isolated_child_env

    extra: dict[str, str] = {"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    if platform.system() == "Linux":
        # ctranslate2 loads cuBLAS / cuDNN with dlopen; point it at the pip copies.
        lib_dirs = [str(p) for p in VOICE_VENV_DIR.glob("lib/python3*/site-packages/nvidia/*/lib")]
        if lib_dirs:
            current = os.environ.get("LD_LIBRARY_PATH", "")
            extra["LD_LIBRARY_PATH"] = os.pathsep.join([*lib_dirs, current] if current else lib_dirs)
    return _isolated_child_env(extra)


# ---------------------------------------------------------------------------
# Install job
# ---------------------------------------------------------------------------

_PHASES: dict[str, str] = {
    "preflight": "Checking this machine",
    "creating-venv": "Creating the voice environment",
    "installing": "Installing speech packages",
    "downloading-voice": "Downloading the voice models",
    "verifying": "Checking the install",
}


@dataclass
class _Job:
    phase: str = "idle"
    message: str = ""
    error: str | None = None
    percent: float = 0.0
    log: list[str] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    extras: list[str] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)
    process: subprocess.Popen | None = None

    def to_dict(self) -> dict[str, Any]:
        phases = list(_PHASES)
        index = phases.index(self.phase) + 1 if self.phase in phases else 0
        return {
            "phase": self.phase,
            "phaseLabel": _PHASES.get(self.phase, self.phase),
            "phaseIndex": index,
            "phaseTotal": len(phases),
            "percent": round(self.percent, 1),
            "message": self.message,
            "error": self.error,
            "log": self.log[-40:],
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "running": self.phase not in ("idle", "done", "error", "cancelled"),
            "extras": list(self.extras),
        }


_JOB = _Job()
_JOB_LOCK = threading.Lock()


def job_status() -> dict[str, Any]:
    return _JOB.to_dict()


def start_install(extras: list[str] | None = None) -> dict[str, Any]:
    wanted = sorted({name for name in (extras or []) if name in EXTRAS} | set(installed_extras()))
    with _JOB_LOCK:
        if _JOB.to_dict()["running"]:
            return _JOB.to_dict()
        _JOB.__init__()  # type: ignore[misc]
        _JOB.phase = "preflight"
        _JOB.started_at = time.time()
        _JOB.extras = wanted
    threading.Thread(target=_run_install, args=(wanted,), daemon=True).start()
    return _JOB.to_dict()


def cancel_install() -> dict[str, Any]:
    _JOB.cancel.set()
    process = _JOB.process
    if process is not None and process.poll() is None:
        process.terminate()
    return _JOB.to_dict()


def _set_phase(name: str) -> None:
    _JOB.phase = name
    phases = list(_PHASES)
    _JOB.percent = phases.index(name) / len(phases) * 100
    _JOB.message = _PHASES[name]


def _run(command: list[str]) -> None:
    """Run one install command, streaming its output into the job log."""
    if _JOB.cancel.is_set():
        raise InterruptedError("Install cancelled.")
    _JOB.log.append("$ " + " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=worker_env(),
    )
    _JOB.process = process
    for line in process.stdout or []:
        line = line.rstrip()
        if line:
            _JOB.log.append(line)
            del _JOB.log[:-400]
    process.wait()
    _JOB.process = None
    if _JOB.cancel.is_set():
        raise InterruptedError("Install cancelled.")
    if process.returncode != 0:
        raise RuntimeError(f"`{' '.join(command[:4])} ...` exited with code {process.returncode}.")


def _run_install(extras: list[str]) -> None:
    lane_name = lane()
    try:
        if sys.version_info < (3, 10):
            raise RuntimeError("The voice runtime needs Python 3.10 or newer.")
        free_gb = shutil.disk_usage(Path.home()).free / 1e9
        if free_gb < 4:
            raise RuntimeError(f"Only {free_gb:.1f} GB free; the voice runtime needs about 4 GB.")

        _set_phase("creating-venv")
        if VOICE_VENV_DIR.exists() and not venv_python().exists():
            shutil.rmtree(VOICE_VENV_DIR, ignore_errors=True)  # a broken earlier attempt
        if not venv_python().exists():
            _run([_base_python(), "-m", "venv", str(VOICE_VENV_DIR)])
        _run([str(venv_python()), "-m", "pip", "install", "--upgrade", "--quiet", "pip"])

        _set_phase("installing")
        _run([str(venv_python()), "-m", "pip", "install", "--upgrade", *packages_for(lane_name, extras)])
        if "speech-gpu" in extras and lane_name == "portable":
            ok, why = speech_gpu_supported()
            if not ok:
                raise RuntimeError(why)
            for command in speech_gpu_commands(str(venv_python())):
                _run(command)

        _set_phase("downloading-voice")
        if "kokoro" in extras:
            download_kokoro(lane_name)
        from huggingface_hub import hf_hub_download

        hf_hub_download(VAD_REPO, VAD_FILE, revision=VAD_REVISION)
        for repo, revision, filename in DIARIZE_FILES.values():
            hf_hub_download(repo, filename, revision=revision)

        _set_phase("verifying")
        modules = ", ".join(_VERIFY_IMPORTS[lane_name])
        _run([str(venv_python()), "-I", "-c", f"import {modules}; print('voice pack ok')"])

        _MARKER.write_text(
            json.dumps(
                {
                    "schema": _SCHEMA,
                    "lane": lane_name,
                    "extras": extras,
                    "python": ".".join(map(str, sys.version_info[:3])),
                    "installedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "cuda": lane_name == "portable" and voice_uses_cuda(),
                    "features": FEATURES,
                }
            ),
            encoding="utf-8",
        )
        _JOB.phase, _JOB.percent, _JOB.message = "done", 100.0, "Voice runtime installed."
    except InterruptedError as exc:
        _JOB.phase, _JOB.error = "cancelled", str(exc)
    except Exception as exc:  # noqa: BLE001 - shown to the user with the log
        _JOB.phase, _JOB.error = "error", str(exc)
    finally:
        _JOB.finished_at = time.time()
        _JOB.process = None
