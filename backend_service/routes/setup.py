from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()

_INSTALLABLE_PIP_PACKAGES: dict[str, str] = {
    "turboquant": "turboquant",
    "turboquant-mlx": "turboquant-mlx-full",
    "triattention": "triattention",
    "vllm": "vllm",
    "mlx": "mlx",
    "mlx-lm": "mlx-lm",
    # dflash-mlx ships only as a git tag (PyPI build is stale at 0.1.0 while
    # v0.1.4 on GitHub renamed the main entrypoint and removed the baseline
    # fallback). Install directly from the tagged commit.
    "dflash-mlx": "dflash-mlx @ git+https://github.com/bstnxbt/dflash-mlx.git@v0.1.4",
    "dflash": "dflash",
    # Video output encoding — diffusers can produce frames without these,
    # but exporting mp4/gif requires imageio + the ffmpeg plugin. The Video
    # Studio surfaces a one-click installer when they're missing.
    "imageio": "imageio",
    "imageio-ffmpeg": "imageio-ffmpeg",
}

_MANUAL_INSTALL_MESSAGES: dict[str, str] = {
    "chaosengine": (
        "ChaosEngine is not published on PyPI. Clone "
        "https://github.com/cryptopoly/ChaosEngine and install it into the "
        "backend runtime with: {python} -m pip install -e /path/to/ChaosEngine. "
        "Desktop release builds can also bundle a vendored vendor/ChaosEngine "
        "checkout automatically during npm run stage:runtime."
    ),
    "chaos-engine": (
        "ChaosEngine is not published on PyPI. Clone "
        "https://github.com/cryptopoly/ChaosEngine and install it into the "
        "backend runtime with: {python} -m pip install -e /path/to/ChaosEngine. "
        "Desktop release builds can also bundle a vendored vendor/ChaosEngine "
        "checkout automatically during npm run stage:runtime."
    ),
}

def _workspace_root() -> Path:
    from backend_service.app import WORKSPACE_ROOT
    return WORKSPACE_ROOT


_CHAOSENGINE_BIN_DIR = Path.home() / ".chaosengine" / "bin"

_TURBO_REPO = "https://github.com/TheTom/llama-cpp-turboquant.git"
_TURBO_BRANCH = "feature/turboquant-kv-cache"
_TURBO_VERSION_FILE = _CHAOSENGINE_BIN_DIR / "llama-server-turbo.version"

# Cached remote HEAD check (commit_hash, timestamp)
_turbo_remote_cache: tuple[str | None, float] = (None, 0.0)
_TURBO_REMOTE_CACHE_TTL = 3600.0  # 1 hour


def _installable_system_packages() -> dict[str, list[str]]:
    return {
        "llama.cpp": ["brew", "install", "llama.cpp"],
        "llama-server-turbo": [str(_workspace_root() / "scripts" / "build-llama-turbo.sh")],
    }


class InstallPackageRequest(BaseModel):
    package: str


@router.post("/api/setup/install-package")
def install_pip_package(request: Request, body: InstallPackageRequest) -> dict[str, Any]:
    """Install a whitelisted pip package into the backend's Python environment."""
    state = request.app.state.chaosengine
    pip_name = _INSTALLABLE_PIP_PACKAGES.get(body.package)
    if pip_name is None:
        manual_message = _MANUAL_INSTALL_MESSAGES.get(body.package)
        if manual_message is not None:
            raise HTTPException(
                status_code=400,
                detail=manual_message.format(python=state.runtime.capabilities.pythonExecutable),
            )
        raise HTTPException(status_code=400, detail=f"Package '{body.package}' is not in the allowed install list.")

    python = state.runtime.capabilities.pythonExecutable
    cmd = [python, "-m", "pip", "install", "--upgrade", pip_name]
    state.add_log("server", "info", f"Installing pip package: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = (result.stdout + "\n" + result.stderr).strip()
        ok = result.returncode == 0
    except subprocess.TimeoutExpired:
        output = "Installation timed out after 5 minutes."
        ok = False
    except OSError as exc:
        output = str(exc)
        ok = False

    # Re-probe capabilities after install
    state.runtime.refresh_capabilities(force=True)
    caps = state.runtime.capabilities.to_dict()
    state.add_log(
        "server", "info" if ok else "error",
        f"pip install {pip_name}: {'succeeded' if ok else 'failed'}",
    )
    return {"ok": ok, "output": output, "capabilities": caps}


@router.post("/api/setup/install-system-package")
def install_system_package(request: Request, body: InstallPackageRequest) -> dict[str, Any]:
    """Install a whitelisted system package (e.g. llama.cpp via brew)."""
    state = request.app.state.chaosengine
    cmd_template = _installable_system_packages().get(body.package)
    if cmd_template is None:
        raise HTTPException(status_code=400, detail=f"System package '{body.package}' is not in the allowed install list.")

    state.add_log("server", "info", f"Installing system package: {' '.join(cmd_template)}")
    try:
        result = subprocess.run(cmd_template, capture_output=True, text=True, timeout=600)
        output = (result.stdout + "\n" + result.stderr).strip()
        ok = result.returncode == 0
    except FileNotFoundError:
        output = f"'{cmd_template[0]}' is not installed. Install Homebrew first: https://brew.sh"
        ok = False
    except subprocess.TimeoutExpired:
        output = "Installation timed out after 10 minutes."
        ok = False
    except OSError as exc:
        output = str(exc)
        ok = False

    state.runtime.refresh_capabilities(force=True)
    caps = state.runtime.capabilities.to_dict()
    state.add_log(
        "server", "info" if ok else "error",
        f"System install {body.package}: {'succeeded' if ok else 'failed'}",
    )
    return {"ok": ok, "output": output, "capabilities": caps}


@router.post("/api/setup/refresh-capabilities")
def refresh_capabilities_endpoint(request: Request) -> dict[str, Any]:
    """Force re-probe all backend capabilities."""
    state = request.app.state.chaosengine
    caps = state.runtime.refresh_capabilities(force=True)
    return {"capabilities": caps.to_dict()}


# ------------------------------------------------------------------
# llama-server-turbo update check
# ------------------------------------------------------------------

def _read_turbo_version() -> tuple[str | None, str | None, str | None]:
    """Read the installed turbo binary version file.

    Returns ``(commit_hash, branch, build_date)`` or ``(None, None, None)``.
    """
    if not _TURBO_VERSION_FILE.exists():
        return None, None, None
    try:
        lines = _TURBO_VERSION_FILE.read_text().strip().splitlines()
        commit = lines[0].strip() if len(lines) > 0 else None
        branch = lines[1].strip() if len(lines) > 1 else None
        build_date = lines[2].strip() if len(lines) > 2 else None
        return commit, branch, build_date
    except OSError:
        return None, None, None


def _fetch_turbo_remote_head() -> str | None:
    """Get the latest commit hash on the turbo fork's branch via ``git ls-remote``.

    Results are cached for 1 hour to avoid excessive GitHub API calls.
    """
    global _turbo_remote_cache
    cached_hash, cached_at = _turbo_remote_cache
    if cached_hash is not None and (time.time() - cached_at) < _TURBO_REMOTE_CACHE_TTL:
        return cached_hash

    try:
        result = subprocess.run(
            ["git", "ls-remote", _TURBO_REPO, f"refs/heads/{_TURBO_BRANCH}"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            remote_hash = result.stdout.strip().split()[0]
            _turbo_remote_cache = (remote_hash, time.time())
            return remote_hash
    except (OSError, subprocess.TimeoutExpired):
        pass
    return cached_hash  # return stale cache on failure


@router.get("/api/setup/turbo-update-check")
def turbo_update_check() -> dict[str, Any]:
    """Check if llama-server-turbo is installed and whether an update is available."""
    installed_commit, branch, build_date = _read_turbo_version()
    installed = _TURBO_VERSION_FILE.exists() and (_CHAOSENGINE_BIN_DIR / "llama-server-turbo").exists()

    remote_commit = _fetch_turbo_remote_head() if installed else None
    update_available = (
        installed
        and installed_commit is not None
        and remote_commit is not None
        and installed_commit != remote_commit
    )

    return {
        "installed": installed,
        "installedCommit": installed_commit[:12] if installed_commit else None,
        "remoteCommit": remote_commit[:12] if remote_commit else None,
        "updateAvailable": update_available,
        "branch": branch,
        "buildDate": build_date,
    }
