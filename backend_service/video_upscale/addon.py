"""The video upscale add-on: spandrel and the Real-ESRGAN weights, installed on demand into their own folder.

Never at startup: only when the user presses Install (``routes/setup/video_upscale.py``). Layout under
``~/.chaosengine/addons/video-upscale``: ``site/`` holds the pinned packages (``pip install --target --no-deps``,
wheels only, like the sync-score add-on) and ``weights/`` the ``.pth`` files of the catalog rows
(``catalog/video_upscale_models.py``). torch, torchvision, safetensors and numpy come from the backend's own Python
(the GPU runtime); the worker puts ``site/`` on its path. Removing the folder removes the add-on.

A weight file is downloaded to a ``.part`` file and kept only when its byte size matches the row and, when the row
pins one, its SHA-256 matches. Rows without a pin yet (see the catalog) are kept on the size check alone and the
hash that was seen is written to ``weights.json`` so it can be pinned.

Licences: spandrel (chaiNNer-org) MIT, einops MIT, the Real-ESRGAN weights BSD-3-Clause (the repo's licence).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from backend_service.catalog.video_upscale_models import DEFAULT_MODEL_ID, VIDEO_UPSCALE_MODELS, find_model

SPANDREL_VERSION = "0.4.2"
PACKAGES = [f"spandrel=={SPANDREL_VERSION}", "einops==0.8.2"]
DOWNLOAD_MB = 1  # the two wheels; weights are counted per row
_CHECK = (
    "import sys; sys.path.insert(0, {site!r}); "
    "from importlib.metadata import version; import spandrel, einops; "
    "assert version('spandrel') == {v!r}, version('spandrel'); print('video upscale ok')"
)


def addon_dir() -> Path:
    return Path.home() / ".chaosengine" / "addons" / "video-upscale"


def site_dir() -> Path:
    return addon_dir() / "site"


def weights_dir() -> Path:
    return addon_dir() / "weights"


def weights_path(row: dict[str, Any]) -> Path:
    return weights_dir() / str(row["file"])


def site_installed() -> bool:
    site = site_dir()
    return (site / "spandrel").is_dir() and any(site.glob(f"spandrel-{SPANDREL_VERSION}.dist-info"))


def model_installed(row: dict[str, Any]) -> bool:
    path = weights_path(row)
    return path.is_file() and path.stat().st_size == int(row["sizeBytes"])


def installed(model_id: str | None = None) -> bool:
    """The packages and the weights of ``model_id`` (the default row when None)."""
    row = find_model(model_id or DEFAULT_MODEL_ID)
    return row is not None and site_installed() and model_installed(row)


def install_commands(python: str) -> list[list[str]]:
    site = str(site_dir())
    return [
        [python, "-m", "pip", "install", "--disable-pip-version-check", "--target", site, "--no-deps",
         "--only-binary=:all:", "--upgrade", *PACKAGES],
        [python, "-I", "-c", _CHECK.format(site=site, v=SPANDREL_VERSION)],
    ]


def _seen_hashes() -> dict[str, str]:
    try:
        data = json.loads((addon_dir() / "weights.json").read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def download_weights(row: dict[str, Any], opener: Callable[..., Any] | None = None) -> str:
    """Fetch one row's weights (via a ``.part`` file); returns the SHA-256 seen. Raises ``RuntimeError``."""
    if opener is None:
        import urllib.request  # noqa: PLC0415

        opener = urllib.request.urlopen
    target = weights_path(row)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    digest, size = hashlib.sha256(), 0
    try:
        with opener(str(row["url"]), timeout=60) as response, part.open("wb") as out:
            for block in iter(lambda: response.read(1 << 20), b""):
                out.write(block)
                digest.update(block)
                size += len(block)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {row['file']} ({exc}).") from exc
    seen = digest.hexdigest()
    if size != int(row["sizeBytes"]):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{row['file']} came down as {size} bytes, expected {row['sizeBytes']}; nothing was kept.")
    if row.get("sha256") and seen != row["sha256"]:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{row['file']} does not match its pinned SHA-256; nothing was kept.")
    part.replace(target)
    hashes = {**_seen_hashes(), str(row["file"]): seen}
    (addon_dir() / "weights.json").write_text(json.dumps(hashes, indent=1), encoding="utf-8")
    return seen


def install(progress: Callable[[str], None], model_ids: list[str] | None = None, python: str | None = None,
            runner: Callable[..., Any] = subprocess.run, opener: Callable[..., Any] | None = None) -> None:
    """Install the packages, then the weights of ``model_ids`` (the default row when None). Raises ``RuntimeError``.

    A failed pip step removes ``site/`` so a half-installed folder never counts as installed."""
    rows = [find_model(model_id) for model_id in (model_ids or [DEFAULT_MODEL_ID])]
    if any(row is None for row in rows):
        raise RuntimeError(f"Unknown video upscale model in {model_ids}.")
    python = python or sys.executable
    site = site_dir()
    site.parent.mkdir(parents=True, exist_ok=True)
    pip, check = install_commands(python)
    if not site_installed():
        progress("packages")
        run = runner(pip, capture_output=True, text=True, timeout=1800)
        if run.returncode != 0:
            shutil.rmtree(site, ignore_errors=True)
            last = (run.stderr or run.stdout or "").strip().splitlines()[-1:] or ["no output"]
            raise RuntimeError(f"pip could not install the video upscale packages: {last[0][:240]}")
    progress("weights")
    for row in rows:
        if not model_installed(row):
            download_weights(row, opener)  # type: ignore[arg-type]
    progress("check")
    run = runner(check, capture_output=True, text=True, timeout=300)
    if run.returncode != 0:
        last = (run.stderr or run.stdout or "").strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"The video upscale packages installed but do not import: {last[0][:240]}")


def disk_mb() -> float:
    root = addon_dir()
    if not root.is_dir():
        return 0.0
    return round(sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) / 1e6, 1)


def status() -> dict[str, Any]:
    seen = _seen_hashes()
    return {
        "installed": installed(),
        "packagesInstalled": site_installed(),
        "folder": str(addon_dir()),
        "diskMb": disk_mb(),
        "packages": list(PACKAGES),
        "models": [{"id": row["id"], "name": row["name"], "label": row["label"], "installed": model_installed(row),
                    "downloadMb": round(int(row["sizeBytes"]) / 1e6, 1), "pinned": bool(row.get("sha256")),
                    "sha256Seen": seen.get(str(row["file"]))} for row in VIDEO_UPSCALE_MODELS],
    }


NOT_INSTALLED = ("The fast video upscaler is not installed. Open Setup and choose Install video upscale (fast), "
                 "then try again.")
