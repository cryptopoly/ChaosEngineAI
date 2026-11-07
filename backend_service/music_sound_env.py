"""The sounds pack: the isolated venv Stable Audio Open runs in, the job that builds it, the weights layout.

Sound effects (D-27) run in a venv of their own (``~/.chaosengine/sound-venv``, ``CHAOSENGINE_SOUND_VENV``),
never in the music venv or the backend: ``stable-audio-tools`` and the music engine pin different libraries,
and keeping the Stability code and weights apart keeps the licence notice honest. Same recipe as the music
pack in ``music_env``: plain pip from the backend's own Python (3.11 or 3.12) from ``sound_requirements.txt``,
then the pinned ``stable-audio-tools`` release with ``--no-deps`` (its own metadata asks for Python 3.10 and
exact old pins, so its dependencies are listed in the requirements file instead). The same ``music_worker``
runs in that venv with the Stable Audio engine file selected by environment. There is no MLX build of Stable
Audio Open Small we can pin (the community port has no licence and no package), so it runs on PyTorch: the
Apple GPU (MPS) when present, else the CPU. The model and its T5 text encoder are mirrored from the Hub cache
into ``<music data>/checkpoints`` (the worker may only read inside the music data folder).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend_service import music_env

SOUND_VENV_DIR = Path(os.environ.get("CHAOSENGINE_SOUND_VENV") or Path.home() / ".chaosengine" / "sound-venv")
SAT_VERSION = "0.0.20"
ENGINE_VERSION = f"stable-audio-tools {SAT_VERSION}"
REQUIREMENTS = Path(__file__).with_name("sound_requirements.txt")
ENGINE_FILE = Path(__file__).with_name("music_worker_stableaudio.py")
_SCHEMA = 1
_MIN_FREE_GB = 6.0

_VERIFY_CODE = (
    "import json, sys, torch, soundfile, transformers, stable_audio_tools\n"
    "from stable_audio_tools.inference.generation import generate_diffusion_cond\n"
    "from stable_audio_tools.models.factory import create_model_from_config\n"
    "print(json.dumps({'torch': torch.__version__, 'transformers': transformers.__version__, "
    "'mps': bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()), "
    "'python': '.'.join(map(str, sys.version_info[:3]))}))\n"
)


def lane() -> str:
    """``mps`` on Apple Silicon; ``unsupported`` elsewhere (the CPU and CUDA lanes are later work)."""
    return "mps" if music_env.lane() == "mlx" else "unsupported"


def lane_unavailable_reason() -> str | None:
    if lane() == "mps":
        return None
    return "Sound effects run on Apple Silicon in this version. The Windows and Linux lane is not built yet."


def _marker_path() -> Path:
    return music_env.MUSIC_DATA_DIR / "sound-pack.json"


def venv_python() -> Path:
    if os.name == "nt":
        return SOUND_VENV_DIR / "Scripts" / "python.exe"
    return SOUND_VENV_DIR / "bin" / "python"


def read_marker() -> dict[str, Any] | None:
    try:
        data = json.loads(_marker_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_installed() -> bool:
    """A verified pack for this release, built for this venv folder, whose Python still exists."""
    marker = read_marker()
    return bool(
        marker
        and marker.get("schema") == _SCHEMA
        and marker.get("version") == SAT_VERSION
        and marker.get("venv") == str(SOUND_VENV_DIR)
        and venv_python().exists()
    )


def worker_env() -> dict[str, str]:
    """Environment for the sound worker and the install commands."""
    from backend_service.inference._utils import _isolated_child_env

    return _isolated_child_env({
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "CHAOSENGINE_MUSIC_DATA": str(music_env.MUSIC_DATA_DIR),
        "CHAOSENGINE_MUSIC_ENGINE_MODULE": str(ENGINE_FILE),
        "CHAOSENGINE_MUSIC_ENGINE_VERSION": ENGINE_VERSION,
    })


# ---------------------------------------------------------------------------
# Weights layout
# ---------------------------------------------------------------------------

def model_dir(variant: dict[str, Any]) -> Path:
    return music_env.checkpoints_dir() / str(variant["checkpointSubdir"])


def text_encoder_dir(variant: dict[str, Any]) -> Path:
    return music_env.checkpoints_dir() / str(variant["textEncoder"]["checkpointSubdir"])


def _encoder_snapshot(variant: dict[str, Any]) -> Path | None:
    """A cache snapshot of the text encoder holding every file it needs (the pinned revision first)."""
    from backend_service.helpers.hf_cache_paths import _hf_repo_cache_dir

    spec = variant["textEncoder"]
    snapshots = _hf_repo_cache_dir(str(spec["repo"])) / "snapshots"
    names = [str(spec["revision"])] + sorted(p.name for p in snapshots.glob("*") if p.name != spec["revision"]) \
        if snapshots.is_dir() else []
    for name in names:
        folder = snapshots / name
        if folder.is_dir() and all((folder / f).exists() for f in spec["files"]):
            return folder
    return None


def text_encoder_ready(variant: dict[str, Any]) -> bool:
    """The T5 text encoder is in the layout already, or in the Hub cache (mirrored on first use)."""
    folder = text_encoder_dir(variant)
    if all((folder / f).exists() for f in variant["textEncoder"]["files"]):
        return True
    return _encoder_snapshot(variant) is not None


def prepare_text_encoder(variant: dict[str, Any]) -> bool:
    """Mirror the text encoder into the layout; returns True when it is complete there afterwards."""
    snapshot = _encoder_snapshot(variant)
    if snapshot is not None:
        music_env.mirror_tree(snapshot, text_encoder_dir(variant))
    folder = text_encoder_dir(variant)
    return all((folder / f).exists() for f in variant["textEncoder"]["files"])


# ---------------------------------------------------------------------------
# Install job
# ---------------------------------------------------------------------------

_PHASES: dict[str, str] = {
    "preflight": "Checking this machine",
    "creating-venv": "Creating the sounds environment",
    "installing": "Installing sound packages",
    "installing-engine": "Installing Stable Audio tools",
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
    cancel: threading.Event = field(default_factory=threading.Event)
    process: subprocess.Popen | None = None

    def to_dict(self) -> dict[str, Any]:
        phases = list(_PHASES)
        return {
            "phase": self.phase,
            "phaseLabel": _PHASES.get(self.phase, self.phase),
            "phaseIndex": phases.index(self.phase) + 1 if self.phase in phases else 0,
            "phaseTotal": len(phases),
            "percent": round(self.percent, 1),
            "message": self.message,
            "error": self.error,
            "log": self.log[-40:],
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "running": self.phase not in ("idle", "done", "error", "cancelled"),
        }


_JOB = _Job()
_JOB_LOCK = threading.Lock()


def job_status() -> dict[str, Any]:
    return _JOB.to_dict()


def start_install() -> dict[str, Any]:
    with _JOB_LOCK:
        if _JOB.to_dict()["running"]:
            return _JOB.to_dict()
        _JOB.__init__()  # type: ignore[misc]
        _JOB.phase = "preflight"
        _JOB.started_at = time.time()
    threading.Thread(target=_run_install, daemon=True).start()
    return _JOB.to_dict()


def cancel_install() -> dict[str, Any]:
    _JOB.cancel.set()
    process = _JOB.process
    if process is not None and process.poll() is None:
        process.terminate()
    return _JOB.to_dict()


def _set_phase(name: str) -> None:
    _JOB.phase = name
    _JOB.percent = list(_PHASES).index(name) / len(_PHASES) * 100
    _JOB.message = _PHASES[name]


def _run(command: list[str]) -> str:
    """Run one install command, streaming its output into the job log; returns the output."""
    if _JOB.cancel.is_set():
        raise InterruptedError("Install cancelled.")
    _JOB.log.append("$ " + " ".join(command))
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", env=worker_env(),
    )
    _JOB.process = process
    lines: list[str] = []
    for line in process.stdout or []:
        line = line.rstrip()
        if line:
            lines.append(line)
            _JOB.log.append(line)
            del _JOB.log[:-400]
    process.wait()
    _JOB.process = None
    if _JOB.cancel.is_set():
        raise InterruptedError("Install cancelled.")
    if process.returncode != 0:
        raise RuntimeError(f"`{' '.join(command[:4])} ...` exited with code {process.returncode}.")
    return "\n".join(lines)


def _verify() -> dict[str, Any]:
    output = _run([str(venv_python()), "-c", _VERIFY_CODE])
    for line in reversed(output.splitlines()):
        try:
            found = json.loads(line)
        except ValueError:
            continue
        if isinstance(found, dict):
            return found
    raise RuntimeError("The sounds pack check printed no versions.")


def _write_marker(versions: dict[str, Any]) -> None:
    music_env.MUSIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _marker_path().write_text(json.dumps({
        "schema": _SCHEMA,
        "version": SAT_VERSION,
        "venv": str(SOUND_VENV_DIR),
        "python": ".".join(map(str, sys.version_info[:3])),
        "installedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **{k: str(v) for k, v in versions.items()},
    }), encoding="utf-8")


def _run_install() -> None:
    try:
        reason = lane_unavailable_reason()
        if reason:
            raise RuntimeError(reason)
        if not music_env._python_supported():  # noqa: SLF001 - same Python rule as the music pack
            raise RuntimeError("The sounds pack needs Python 3.11 or 3.12; the backend runs on "
                               f"{sys.version_info[0]}.{sys.version_info[1]}.")
        for folder in (music_env.checkpoints_dir(), music_env.library_dir(), music_env.tmp_dir()):
            folder.mkdir(parents=True, exist_ok=True)
        if venv_python().exists():
            _set_phase("verifying")
            try:
                _write_marker(_verify())
                _JOB.phase, _JOB.percent, _JOB.message = "done", 100.0, "Sounds pack ready."
                return
            except RuntimeError:
                _JOB.log.append("The existing sounds environment is incomplete; installing.")
        free_gb = shutil.disk_usage(music_env.MUSIC_DATA_DIR).free / 1e9
        if free_gb < _MIN_FREE_GB:
            raise RuntimeError(f"Only {free_gb:.1f} GB free; the sounds pack needs about {_MIN_FREE_GB:.0f} GB.")

        _set_phase("creating-venv")
        if SOUND_VENV_DIR.exists() and not venv_python().exists() and any(SOUND_VENV_DIR.iterdir()):
            if not (SOUND_VENV_DIR / "pyvenv.cfg").is_file():
                raise RuntimeError(f"{SOUND_VENV_DIR} is not empty and is not a Python environment, so it was left "
                                   "alone. Point CHAOSENGINE_SOUND_VENV at a new or empty folder.")
            shutil.rmtree(SOUND_VENV_DIR, ignore_errors=True)
        if not venv_python().exists():
            from backend_service.voice_env import _base_python

            _run([_base_python(), "-m", "venv", str(SOUND_VENV_DIR)])
        _run([str(venv_python()), "-m", "pip", "install", "--upgrade", "--quiet", "pip"])

        _set_phase("installing")
        _run([str(venv_python()), "-m", "pip", "install", "-r", str(REQUIREMENTS)])

        _set_phase("installing-engine")
        _run([str(venv_python()), "-m", "pip", "install", "--no-deps", "--ignore-requires-python",
              f"stable-audio-tools=={SAT_VERSION}"])

        _set_phase("verifying")
        _write_marker(_verify())
        _JOB.phase, _JOB.percent, _JOB.message = "done", 100.0, "Sounds pack installed."
    except InterruptedError as exc:
        _JOB.phase, _JOB.error = "cancelled", str(exc)
    except Exception as exc:  # noqa: BLE001 - shown to the user with the log
        _JOB.phase, _JOB.error = "error", str(exc)
    finally:
        _JOB.finished_at = time.time()
        _JOB.process = None
