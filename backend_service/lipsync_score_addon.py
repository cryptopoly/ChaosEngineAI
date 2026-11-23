"""The sync-scoring add-on (D-132): what lip-sync takes need to be scored, installed on demand into its own folder.

The score (``lip_sync_score.py``) reads each take with MediaPipe Face Landmarker and OpenCV, which the backend's
Python does not carry. The add-on puts them, with the face model, in ``~/.chaosengine/addons/lipsync-score``:
``site/`` holds the pinned packages (``pip install --target --no-deps``, like the speech-to-text GPU add-on's
``asr-site``) and ``face_landmarker.task`` is Google's float16 v1 model, fetched from its official address and
checked against a pinned SHA-256. The score then runs from the backend's own Python as ``python -I
lip_sync_score.py --site <site>``, so nothing here is ever imported into the backend. numpy, scipy, soundfile,
Pillow and packaging come from the backend; matplotlib (with its own dependencies) is here only because MediaPipe
imports it at start. Removing the folder removes the add-on.

Licences: MediaPipe, OpenCV, absl-py, flatbuffers and the face model Apache-2.0; matplotlib its own BSD-style
licence; contourpy, cycler and kiwisolver BSD-3-Clause; fonttools, pyparsing and six MIT; python-dateutil
Apache-2.0 and BSD-3-Clause.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

# Checked on the RTX 4090 (Windows, Python 3.12) on 2026-11-23: a research take scored the same as in the tools venv.
# mediapipe 1.2.3 is a py3-none wheel per OS (Windows x64 and ARM64, macOS 11+ arm64, Linux x86-64 and aarch64);
# opencv-python-headless 5.1.0.95 is abi3 and needs macOS 15 on a Mac.
PACKAGES = [
    "mediapipe==1.2.3", "opencv-python-headless==5.1.0.95", "absl-py==2.5.1", "flatbuffers==25.12.19",
    "matplotlib==3.11.2", "contourpy==1.4.0", "cycler==0.12.1", "fonttools==4.68.0", "kiwisolver==1.5.1",
    "pyparsing==3.3.3", "python-dateutil==2.9.0.post0", "six==1.17.0",
]
MEDIAPIPE_VERSION = "1.2.3"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                  "face_landmarker.task")
FACE_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
FACE_MODEL_BYTES = 3_758_596
DOWNLOAD_MB = 90  # the wheels (Windows x64: about 85 MB; Linux about 120 MB) plus the 3.8 MB face model
_CHECK = (
    "import sys; sys.path.insert(0, {site!r}); "
    "from importlib.metadata import version; import mediapipe, cv2, scipy.signal, soundfile; "
    "assert version('mediapipe') == {mp!r}, version('mediapipe'); print('lip-sync score ok')"
)


def addon_dir() -> Path:
    return Path.home() / ".chaosengine" / "addons" / "lipsync-score"


def site_dir() -> Path:
    return addon_dir() / "site"


def face_model_path() -> Path:
    return addon_dir() / "face_landmarker.task"


def site_installed() -> bool:
    site = site_dir()
    return ((site / "mediapipe").is_dir() and (site / "cv2").is_dir()
            and any(site.glob(f"mediapipe-{MEDIAPIPE_VERSION}.dist-info")))


def model_installed() -> bool:
    model = face_model_path()
    return model.is_file() and model.stat().st_size == FACE_MODEL_BYTES


def installed() -> bool:
    return site_installed() and model_installed()


def disk_mb() -> float:
    """Megabytes the add-on folder takes (0 when absent)."""
    root = addon_dir()
    if not root.is_dir():
        return 0.0
    return round(sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) / 1e6, 1)


def install_commands(python: str) -> list[list[str]]:
    """The pip install into ``site/`` (wheels only, no dependency resolution), then the import check."""
    site = str(site_dir())
    return [
        [python, "-m", "pip", "install", "--disable-pip-version-check", "--target", site, "--no-deps",
         "--only-binary=:all:", "--upgrade", *PACKAGES],
        [python, "-I", "-c", _CHECK.format(site=site, mp=MEDIAPIPE_VERSION)],
    ]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_face_model(target: Path, opener: Callable[..., Any] | None = None) -> None:
    """Fetch the face model to ``target`` (via a ``.part`` file) and keep it only when its SHA-256 matches the pin."""
    if opener is None:
        import urllib.request  # noqa: PLC0415

        opener = urllib.request.urlopen
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        with opener(FACE_MODEL_URL, timeout=60) as response, part.open("wb") as out:
            for block in iter(lambda: response.read(1 << 20), b""):
                out.write(block)
                digest.update(block)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download the face model from {FACE_MODEL_URL} ({exc}).") from exc
    if digest.hexdigest() != FACE_MODEL_SHA256:
        part.unlink(missing_ok=True)
        raise RuntimeError("The face model download does not match its pinned SHA-256; nothing was kept.")
    part.replace(target)


def install(progress: Callable[[str], None], python: str | None = None,
            runner: Callable[..., Any] = subprocess.run, opener: Callable[..., Any] | None = None) -> None:
    """Install the packages and the face model; raises ``RuntimeError`` with the reason when a step fails.

    A failed pip step removes ``site/`` so a half-installed folder never counts as installed."""
    python = python or sys.executable
    site = site_dir()
    site.parent.mkdir(parents=True, exist_ok=True)
    pip, check = install_commands(python)
    progress("packages")
    run = runner(pip, capture_output=True, text=True, timeout=1800)
    if run.returncode != 0:
        shutil.rmtree(site, ignore_errors=True)
        last = (run.stderr or run.stdout or "").strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"pip could not install the sync-score packages: {last[0][:240]}")
    progress("model")
    if not model_installed() or sha256_of(face_model_path()) != FACE_MODEL_SHA256:
        download_face_model(face_model_path(), opener)
    progress("check")
    run = runner(check, capture_output=True, text=True, timeout=300)
    if run.returncode != 0:
        last = (run.stderr or run.stdout or "").strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"The sync-score packages installed but do not import: {last[0][:240]}")


def status() -> dict[str, Any]:
    return {
        "installed": installed(),
        "packagesInstalled": site_installed(),
        "modelInstalled": model_installed(),
        "folder": str(addon_dir()),
        "downloadMb": DOWNLOAD_MB,
        "diskMb": disk_mb(),
        "packages": list(PACKAGES),
        "faceModelUrl": FACE_MODEL_URL,
    }
