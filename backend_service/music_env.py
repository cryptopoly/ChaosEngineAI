"""The music venv and data folder: where ACE-Step lives, the job that builds it, the weights layout.

ACE-Step 1.5 installs into its own venv (``~/.chaosengine/music-venv``, ``CHAOSENGINE_MUSIC_VENV``),
never into the backend's environment: it pins torch, mlx, transformers and diffusers versions the app
must not see. The venv is built with plain pip from the backend's own Python (3.11 or 3.12, which the
spike proved both work), from ``music_requirements.txt`` (exported once from the pinned ``uv.lock``)
and the pinned ACE-Step commit. Weights come through the app's Hugging Face downloader into the standard
cache; ``prepare_checkpoints`` mirrors them into ``<data>/checkpoints`` in the layout the library reads
(``ACESTEP_CHECKPOINTS_DIR``): small files are copied (the library rewrites its model code there, which
must never reach the cache), weight files are linked. A dangling or wrong link is re-made, and
``checkpoints/.mirror.json`` records the revision mirrored per repo so a catalog revision bump re-mirrors.
Only the MLX lane (Apple Silicon) exists today.
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

MUSIC_VENV_DIR = Path(os.environ.get("CHAOSENGINE_MUSIC_VENV") or Path.home() / ".chaosengine" / "music-venv")
MUSIC_DATA_DIR = Path(os.environ.get("CHAOSENGINE_MUSIC_DATA") or Path.home() / ".chaosengine" / "music")

ACE_REPO = "https://github.com/ace-step/ACE-Step-1.5"
ACE_REF = "ca1e85fe9430179831e6bc6be790c332190a3866"
ENGINE_VERSION = ACE_REF[:7]
ACE_ARCHIVE = f"{ACE_REPO}/archive/{ACE_REF}.zip"
REQUIREMENTS = Path(__file__).with_name("music_requirements.txt")
_SCHEMA = 1
_MIN_FREE_GB = 4.0
# Files at least this large are linked into the checkpoints layout, smaller ones copied.
_LINK_MIN_BYTES = 16 * 1024 * 1024

_VERIFY_CODE = (
    "import json, sys, torch, soundfile, mlx.core as mx, acestep.handler, acestep.inference, "
    "acestep.models.mlx.dit_generate as dg\n"
    "assert hasattr(dg, 'tqdm'), 'dit_generate.tqdm missing'\n"
    "print(json.dumps({'torch': torch.__version__, 'mlx': mx.__version__, "
    "'python': '.'.join(map(str, sys.version_info[:3]))}))\n"
)


def lane() -> str:
    """``mlx`` on Apple Silicon; ``unsupported`` elsewhere (the CUDA lane is later work)."""
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "mlx"
    return "unsupported"


def lane_unavailable_reason() -> str | None:
    if lane() == "mlx":
        return None
    return (
        "Music runs on Apple Silicon (MLX) in this version. The Windows and Linux (CUDA) lane is "
        "not built yet."
    )


def checkpoints_dir() -> Path:
    return MUSIC_DATA_DIR / "checkpoints"


def library_dir() -> Path:
    return MUSIC_DATA_DIR / "library"


def tmp_dir() -> Path:
    return MUSIC_DATA_DIR / "tmp"


def _marker_path() -> Path:
    return MUSIC_DATA_DIR / "music-pack.json"


def venv_python() -> Path:
    if os.name == "nt":
        return MUSIC_VENV_DIR / "Scripts" / "python.exe"
    return MUSIC_VENV_DIR / "bin" / "python"


def _python_supported(version: tuple[int, ...] | None = None) -> bool:
    major_minor = tuple((version or sys.version_info)[:2])
    return (3, 11) <= major_minor <= (3, 12)


def read_marker() -> dict[str, Any] | None:
    try:
        data = json.loads(_marker_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_installed() -> bool:
    """A verified pack for this commit, built for this venv folder, whose Python still exists."""
    marker = read_marker()
    return bool(
        marker
        and marker.get("schema") == _SCHEMA
        and marker.get("ref") == ACE_REF
        and marker.get("venv") == str(MUSIC_VENV_DIR)
        and venv_python().exists()
    )


def worker_env() -> dict[str, str]:
    """Environment for the music worker (spike section 4) and the install commands."""
    from backend_service.inference._utils import _isolated_child_env

    return _isolated_child_env({
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "ACESTEP_MLX_VAE_CHUNK": "256",
        "ACESTEP_LM_BACKEND": "mlx",
        "HF_HUB_OFFLINE": "1",
        "ACESTEP_CHECKPOINTS_DIR": str(checkpoints_dir()),
        "ACESTEP_PROJECT_ROOT": str(MUSIC_DATA_DIR),
        "CHAOSENGINE_MUSIC_DATA": str(MUSIC_DATA_DIR),
        "CHAOSENGINE_MUSIC_ENGINE_VERSION": ENGINE_VERSION,
    })


# ---------------------------------------------------------------------------
# Weights layout
# ---------------------------------------------------------------------------

def _place(source: Path, target: Path, refresh: bool = False) -> bool:
    """Link a large file (symlink, else hard link, else copy); copy a small one. Returns True when placed.

    An existing file is kept (the library rewrites its small model files in the layout), except: a symlink
    that dangles or points anywhere but this snapshot file is re-linked, and ``refresh`` (the pinned
    revision changed) replaces everything.
    """
    real = source.resolve()
    if target.is_symlink():
        try:
            current = target.resolve(strict=True)
        except (OSError, RuntimeError):
            current = None  # dangling: the cache was cleared or downloaded again
        if current == real and not refresh:
            return False
        target.unlink()
    elif target.exists():
        if not refresh:
            return False
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if real.stat().st_size >= _LINK_MIN_BYTES:
        for link in (os.symlink, os.link):
            try:
                link(real, target)
                return True
            except OSError:
                continue
    shutil.copyfile(real, target)
    return True


def mirror_tree(source: Path, target: Path, refresh: bool = False) -> int:
    """Mirror ``source`` (a Hub snapshot) into ``target``; returns files placed. Hidden entries skipped."""
    placed = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part.startswith(".") for part in relative.parts) or not path.is_file():
            continue
        if _place(path, target / relative, refresh):
            placed += 1
    return placed


def _manifest_path() -> Path:
    return checkpoints_dir() / ".mirror.json"


def read_mirror_manifest() -> dict[str, str]:
    """repo -> the revision mirrored into the checkpoints layout."""
    try:
        data = json.loads(_manifest_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _write_mirror_manifest(manifest: dict[str, str]) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def mirrored_revision_ok(variant: dict[str, Any]) -> bool:
    """False when the layout holds another revision of this repo than the catalog pins (an older install
    without a manifest entry is accepted)."""
    found = read_mirror_manifest().get(str(variant["repo"]))
    return found is None or found == str(variant["revision"])


def has_weights(folder: Path) -> bool:
    try:
        return any(p.exists() for p in folder.glob("*.safetensors"))
    except OSError:
        return False


def hf_snapshot(repo: str, revision: str) -> Path | None:
    """The cache snapshot of ``repo`` at the pinned ``revision``, if downloaded."""
    from backend_service.helpers.hf_cache_paths import _hf_repo_cache_dir

    path = _hf_repo_cache_dir(repo) / "snapshots" / revision
    return path if path.is_dir() else None


def prepare_checkpoints(variants: list[dict[str, Any]]) -> list[str]:
    """Mirror every downloaded variant into the checkpoints layout; returns the repos placed."""
    placed: list[str] = []
    root = checkpoints_dir()
    manifest = read_mirror_manifest()
    for variant in variants:
        repo, revision = str(variant["repo"]), str(variant["revision"])
        snapshot = hf_snapshot(repo, revision)
        if snapshot is None:
            continue
        target = root / variant["checkpointSubdir"] if variant.get("checkpointSubdir") else root
        mirror_tree(snapshot, target, refresh=manifest.get(repo) not in (None, revision))
        if manifest.get(repo) != revision:
            manifest[repo] = revision
            _write_mirror_manifest(manifest)
        placed.append(repo)
    return placed


def remove_checkpoints(variant: dict[str, Any]) -> None:
    """Drop a variant's weight folders from the layout (its cache download is deleted separately)."""
    root = checkpoints_dir().resolve()
    for folder in variant.get("weightFolders") or []:
        target = (checkpoints_dir() / folder).resolve()
        if target != root and root in target.parents and target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
    manifest = read_mirror_manifest()
    if manifest.pop(str(variant.get("repo")), None) is not None:
        _write_mirror_manifest(manifest)


# ---------------------------------------------------------------------------
# Install job
# ---------------------------------------------------------------------------

_PHASES: dict[str, str] = {
    "preflight": "Checking this machine",
    "creating-venv": "Creating the music environment",
    "installing": "Installing music packages",
    "installing-engine": "Installing ACE-Step",
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
    """Import the engine in the venv; returns the versions it reports."""
    output = _run([str(venv_python()), "-I", "-c", _VERIFY_CODE])
    for line in reversed(output.splitlines()):
        try:
            found = json.loads(line)
        except ValueError:
            continue
        if isinstance(found, dict):
            return found
    raise RuntimeError("The music pack check printed no versions.")


def _write_marker(versions: dict[str, Any]) -> None:
    MUSIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _marker_path().write_text(json.dumps({
        "schema": _SCHEMA,
        "ref": ACE_REF,
        "venv": str(MUSIC_VENV_DIR),
        "python": ".".join(map(str, sys.version_info[:3])),
        "installedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **{k: str(v) for k, v in versions.items()},
    }), encoding="utf-8")


def _run_install() -> None:
    try:
        reason = lane_unavailable_reason()
        if reason:
            raise RuntimeError(reason)
        if not _python_supported():
            raise RuntimeError("ACE-Step needs Python 3.11 or 3.12; the backend runs on "
                               f"{sys.version_info[0]}.{sys.version_info[1]}.")
        for folder in (checkpoints_dir(), library_dir(), tmp_dir()):
            folder.mkdir(parents=True, exist_ok=True)
        if venv_python().exists():
            # Resume, or adopt a venv that already works: nothing is installed when it imports.
            _set_phase("verifying")
            try:
                _write_marker(_verify())
                _JOB.phase, _JOB.percent, _JOB.message = "done", 100.0, "Music pack ready."
                return
            except RuntimeError:
                _JOB.log.append("The existing music environment is incomplete; installing.")
        free_gb = shutil.disk_usage(MUSIC_DATA_DIR).free / 1e9
        if free_gb < _MIN_FREE_GB:
            raise RuntimeError(f"Only {free_gb:.1f} GB free; the music pack needs about {_MIN_FREE_GB:.0f} GB.")

        _set_phase("creating-venv")
        if MUSIC_VENV_DIR.exists() and not venv_python().exists() and any(MUSIC_VENV_DIR.iterdir()):
            if not (MUSIC_VENV_DIR / "pyvenv.cfg").is_file():
                # Not a venv (CHAOSENGINE_MUSIC_VENV may name a folder with other files): never delete it.
                raise RuntimeError(f"{MUSIC_VENV_DIR} is not empty and is not a Python environment, so it was left "
                                   "alone. Point CHAOSENGINE_MUSIC_VENV at a new or empty folder.")
            shutil.rmtree(MUSIC_VENV_DIR, ignore_errors=True)  # a broken earlier attempt
        if not venv_python().exists():
            from backend_service.voice_env import _base_python

            _run([_base_python(), "-m", "venv", str(MUSIC_VENV_DIR)])
        _run([str(venv_python()), "-m", "pip", "install", "--upgrade", "--quiet", "pip"])

        _set_phase("installing")
        _run([str(venv_python()), "-m", "pip", "install", "-r", str(REQUIREMENTS)])

        _set_phase("installing-engine")
        _run([str(venv_python()), "-m", "pip", "install", "--no-deps", ACE_ARCHIVE])

        _set_phase("verifying")
        _write_marker(_verify())
        _JOB.phase, _JOB.percent, _JOB.message = "done", 100.0, "Music pack installed."
    except InterruptedError as exc:
        _JOB.phase, _JOB.error = "cancelled", str(exc)
    except Exception as exc:  # noqa: BLE001 - shown to the user with the log
        _JOB.phase, _JOB.error = "error", str(exc)
    finally:
        _JOB.finished_at = time.time()
        _JOB.process = None
