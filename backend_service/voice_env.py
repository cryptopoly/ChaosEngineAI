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

# spaCy's small English model: misaki's English front-end downloads it at
# first use otherwise, through pip, which a packaged app cannot rely on.
_EN_CORE_WEB_SM = (
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)

# Versions checked together on an M4 Max (2026-10-26); bump deliberately.
_MLX_PACKAGES = [
    "mlx-audio>=0.6.2,<0.7",
    "misaki[en]>=0.9.4,<0.10",
    "parakeet-mlx>=0.5.3,<0.6",
    "mlx-whisper>=0.4.3,<0.5",
    "av>=13",
    _EN_CORE_WEB_SM,
]
_PORTABLE_PACKAGES = [
    "faster-whisper>=1.2.1,<1.3",
    "kokoro-onnx>=0.6.1,<0.7",
    "av>=13",
]
# ctranslate2 4.x runs on CUDA 12 with cuDNN 9; pip ships both libraries.
_CUDA_PACKAGES = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12>=9,<10"]

# Optional extras: name -> (lanes it applies to, packages, label).
EXTRAS: dict[str, tuple[tuple[str, ...], list[str], str]] = {
    "chinese": (("mlx",), ["misaki[zh]>=0.9.4,<0.10"], "Mandarin Chinese voices"),
}

_VERIFY_IMPORTS = {
    "mlx": ["av", "numpy", "mlx_audio", "misaki.en", "parakeet_mlx", "mlx_whisper", "en_core_web_sm"],
    "portable": ["av", "numpy", "faster_whisper", "kokoro_onnx"],
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


def venv_python() -> Path:
    if os.name == "nt":
        return VOICE_VENV_DIR / "Scripts" / "python.exe"
    return VOICE_VENV_DIR / "bin" / "python"


def has_nvidia_gpu() -> bool:
    if platform.system() == "Darwin":
        return False
    return shutil.which("nvidia-smi") is not None


def packages_for(lane_name: str, extras: list[str] | None = None) -> list[str]:
    specs = list(_MLX_PACKAGES if lane_name == "mlx" else _PORTABLE_PACKAGES)
    if lane_name == "portable" and has_nvidia_gpu():
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
    return [name for name in marker.get("extras", []) if name in EXTRAS]


def kokoro_onnx_dir() -> Path:
    return VOICE_DATA_DIR / "kokoro"


def tts_source() -> tuple[str, str]:
    """(engine, source) the worker loads Kokoro from on this lane."""
    if lane() == "mlx":
        return "mlx-audio", KOKORO_MLX_REPO
    return "kokoro-onnx", str(kokoro_onnx_dir())


def tts_data_installed() -> bool:
    if lane() == "mlx":
        from backend_service.helpers.hf_cache_paths import _hf_repo_snapshot_dir

        return _hf_repo_snapshot_dir(KOKORO_MLX_REPO) is not None
    from backend_service import voice_kokoro_onnx

    return voice_kokoro_onnx.is_installed()


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
    "downloading-voice": "Downloading the Kokoro voice model",
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
            raise RuntimeError("The voice pack needs Python 3.10 or newer.")
        free_gb = shutil.disk_usage(Path.home()).free / 1e9
        if free_gb < 4:
            raise RuntimeError(f"Only {free_gb:.1f} GB free; the voice pack needs about 4 GB.")

        _set_phase("creating-venv")
        if VOICE_VENV_DIR.exists() and not venv_python().exists():
            shutil.rmtree(VOICE_VENV_DIR, ignore_errors=True)  # a broken earlier attempt
        if not venv_python().exists():
            _run([_base_python(), "-m", "venv", str(VOICE_VENV_DIR)])
        _run([str(venv_python()), "-m", "pip", "install", "--upgrade", "--quiet", "pip"])

        _set_phase("installing")
        _run([str(venv_python()), "-m", "pip", "install", "--upgrade", *packages_for(lane_name, extras)])

        _set_phase("downloading-voice")
        if lane_name == "mlx":
            from huggingface_hub import snapshot_download

            snapshot_download(KOKORO_MLX_REPO)
        else:
            from backend_service import voice_kokoro_onnx

            if not voice_kokoro_onnx.is_installed():
                voice_kokoro_onnx._download_worker()
                state = voice_kokoro_onnx.download_status()
                if state.get("state") == "failed":
                    raise RuntimeError(f"Kokoro voice files: {state.get('error')}")

        _set_phase("verifying")
        modules = ", ".join(_VERIFY_IMPORTS[lane_name])
        _run([str(venv_python()), "-c", f"import {modules}; print('voice pack ok')"])

        _MARKER.write_text(
            json.dumps(
                {
                    "schema": _SCHEMA,
                    "lane": lane_name,
                    "extras": extras,
                    "python": ".".join(map(str, sys.version_info[:3])),
                    "installedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "cuda": lane_name == "portable" and has_nvidia_gpu(),
                }
            ),
            encoding="utf-8",
        )
        _JOB.phase, _JOB.percent, _JOB.message = "done", 100.0, "Voice pack installed."
    except InterruptedError as exc:
        _JOB.phase, _JOB.error = "cancelled", str(exc)
    except Exception as exc:  # noqa: BLE001 - shown to the user with the log
        _JOB.phase, _JOB.error = "error", str(exc)
    finally:
        _JOB.finished_at = time.time()
        _JOB.process = None
