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
    # Pipeline-specific tokenizer / text-encoder packages. Diffusers itself
    # imports without them, but individual video pipelines need one or more
    # at preload / generate time:
    #   - tiktoken: LTX-Video's T5 tokenizer ships in tiktoken format.
    #   - sentencepiece: Wan (UMT5-XXL), HunyuanVideo, CogVideoX, Mochi (T5).
    #   - protobuf: SentencePiece tokenizers HF loads.
    #   - ftfy: prompt-text preprocessing several pipelines use.
    "tiktoken": "tiktoken",
    "sentencepiece": "sentencepiece",
    "protobuf": "protobuf",
    "ftfy": "ftfy",
    # Core image / video runtime packages. Installed together via the
    # one-click button in Image Studio / Video Studio when the probe
    # reports the real engine as unavailable. Each is also individually
    # installable so we can retry a single failed package without redoing
    # the whole set.
    #
    # We deliberately do not pin versions here — the backend ships with
    # ``pyproject.toml`` extras that constrain them, and a bare ``pip
    # install diffusers`` resolves compatibly with whatever torch the user
    # already has. For a coordinated install of all of these, the Studio
    # calls this endpoint once per package in order so a single failure
    # doesn't abort the whole sequence.
    "diffusers": "diffusers",
    "torch": "torch",
    "accelerate": "accelerate",
    "huggingface_hub": "huggingface_hub",
    "pillow": "pillow",
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
# CUDA torch install (Windows/Linux NVIDIA fallback recovery)
# ------------------------------------------------------------------

# cu124 covers Python 3.9-3.13 and driver 525+. cu121 only ships wheels
# for Python up to 3.12, so fresh Windows installs (3.13) fail on it.
# The endpoint walks these in order and stops at the first success.
_CUDA_TORCH_INDEXES: list[str] = [
    "https://download.pytorch.org/whl/cu124",
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu128",
    "https://download.pytorch.org/whl/cu121",
]


@router.post("/api/setup/install-cuda-torch")
def install_cuda_torch(request: Request) -> dict[str, Any]:
    """Install a CUDA-enabled torch wheel into the backend runtime.

    The fresh-Windows-install case is Python 3.13 + system pip, which has
    no cu121 wheel at all — the install fails with "Could not find a
    version that satisfies the requirement torch". We try cu124 first
    (broadest Python 3.9-3.13 coverage), then cu126 / cu128 / cu121 in
    case the user's driver doesn't match the newest. Returns the output
    of the winning attempt so the UI can surface failures usefully.

    Torch already imported in this process stays CPU until the user
    restarts the backend — we flag ``requiresRestart`` in the response
    so the frontend can prompt appropriately.
    """
    state = request.app.state.chaosengine
    python = state.runtime.capabilities.pythonExecutable

    attempts: list[dict[str, Any]] = []
    ok = False
    winning_output = ""
    winning_index: str | None = None

    for index_url in _CUDA_TORCH_INDEXES:
        cmd = [
            python, "-m", "pip", "install",
            "--upgrade", "--force-reinstall",
            "--index-url", index_url,
            "torch>=2.4.0",
        ]
        state.add_log("server", "info", f"Installing CUDA torch from {index_url}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            output = (result.stdout + "\n" + result.stderr).strip()
            attempt_ok = result.returncode == 0
        except subprocess.TimeoutExpired:
            output = f"Install from {index_url} timed out after 15 minutes."
            attempt_ok = False
        except OSError as exc:
            output = f"{index_url}: {exc}"
            attempt_ok = False
        attempts.append({"indexUrl": index_url, "ok": attempt_ok, "output": output})
        if attempt_ok:
            ok = True
            winning_output = output
            winning_index = index_url
            break

    # Re-probe so the UI can refresh its capabilities view. Note: torch
    # already imported in this process is still the old module — the
    # live cuda check won't flip to True without a restart.
    state.runtime.refresh_capabilities(force=True)
    caps = state.runtime.capabilities.to_dict()
    state.add_log(
        "server", "info" if ok else "error",
        f"CUDA torch install: {'succeeded via ' + winning_index if ok else 'failed after all candidates'}",
    )
    return {
        "ok": ok,
        "output": winning_output or (attempts[-1]["output"] if attempts else ""),
        "indexUrl": winning_index,
        "attempts": attempts,
        "requiresRestart": ok,
        "capabilities": caps,
    }


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
