from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
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
    # PyPI build is stale at 0.1.0; the up-to-date code lives on GitHub.
    # The upstream removed all tags in April 2026, so we pin to a specific
    # commit on main instead — v0.1.4 no longer resolves and fresh clones
    # failed with "pathspec 'v0.1.4' did not match any file(s) known to
    # git". Bump the pin when we validate a newer main SHA.
    "dflash-mlx": "dflash-mlx @ git+https://github.com/bstnxbt/dflash-mlx.git@f825ffb268e50d531e8b6524413b0847334a14dd",
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
# The nightly index sometimes has wheels for very new Python (e.g. 3.14)
# before they land in stable — we try it last so users on bleeding-edge
# Python aren't stuck. The endpoint walks this list in order and stops
# at the first success.
_CUDA_TORCH_INDEXES: list[str] = [
    "https://download.pytorch.org/whl/cu124",
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu128",
    "https://download.pytorch.org/whl/cu121",
    "https://download.pytorch.org/whl/nightly/cu128",
]


def _read_python_version(python: str) -> str | None:
    """Return e.g. ``3.13.2`` for the given Python interpreter, or ``None``."""
    try:
        result = subprocess.run(
            [python, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _site_packages_for(python_executable: str) -> Path | None:
    """Return the site-packages directory for the given interpreter, or None."""
    try:
        result = subprocess.run(
            [
                python_executable, "-c",
                "import sysconfig; print(sysconfig.get_paths().get('purelib') or sysconfig.get_paths().get('platlib') or '')",
            ],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    path = (result.stdout or "").strip()
    return Path(path) if path else None


def _purge_broken_distributions(site_packages: Path) -> list[str]:
    """Delete ``~*`` stub directories pip leaves behind after an interrupted install.

    On Windows, pip atomically renames the old version of a package to ``~<name>``
    before unpacking the new one. If the process is killed mid-install (antivirus,
    a file lock, Ctrl-C) the stub is left behind. Subsequent ``pip install`` runs
    then print ``WARNING: Ignoring invalid distribution ~arkupsafe`` forever and
    sometimes refuse to heal the tree. Removing these stubs is cheap and safe —
    they contain no authoritative data.
    """
    if not site_packages.is_dir():
        return []
    removed: list[str] = []
    for entry in site_packages.iterdir():
        if not entry.name.startswith("~"):
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            if not entry.exists():
                removed.append(entry.name)
        except OSError:
            continue
    return removed


def _all_attempts_lack_wheel(attempts: list[dict[str, Any]]) -> bool:
    """True when pip reported 'No matching distribution' for every attempt.

    This is the signature of a Python version PyTorch doesn't ship wheels
    for (either too old or too new) — the fix is a different Python, not
    a different CUDA index. We surface that specifically to the UI so
    the user doesn't keep retrying.
    """
    if not attempts:
        return False
    for attempt in attempts:
        if attempt.get("ok"):
            return False
        text = (attempt.get("output") or "").lower()
        if "no matching distribution" not in text and "from versions: none" not in text:
            return False
    return True


@router.post("/api/setup/install-cuda-torch")
def install_cuda_torch(request: Request) -> dict[str, Any]:
    """Install a CUDA-enabled torch wheel into the backend runtime.

    The fresh-Windows-install case is Python 3.13 + system pip, which has
    no cu121 wheel at all — the install fails with "Could not find a
    version that satisfies the requirement torch". We try cu124 first
    (broadest Python 3.9-3.13 coverage), then cu126 / cu128 / cu121 in
    case the user's driver doesn't match the newest, and finally the
    nightly cu128 index for very-new Python (e.g. 3.14).

    If every attempt fails with "No matching distribution", we set
    ``noWheelForPython`` in the response — that means the user's Python
    version is the problem, not the CUDA index, so the UI can tell them
    to switch Python rather than keep retrying. The response always
    includes ``pythonVersion`` so the UI can show which interpreter this
    is targeting (important: it's the app's bundled venv, not the system
    pip the user might reach from a shell).

    Torch already imported in this process stays CPU until the user
    restarts the backend — we flag ``requiresRestart`` in the response
    so the frontend can prompt appropriately.
    """
    state = request.app.state.chaosengine
    python = state.runtime.capabilities.pythonExecutable
    python_version = _read_python_version(python)

    # Sweep pip's "~<pkg>" stub directories before attempting the install.
    # These are left behind by a prior interrupted install (common on Windows
    # where Defender briefly locks .pyd files), and they cause two problems:
    #   1. Noisy "WARNING: Ignoring invalid distribution ~arkupsafe" spam that
    #      confuses users reading install output.
    #   2. pip sometimes tries to repair them and fails with an "Access denied"
    #      write to a .pyd that the running backend process has loaded (e.g.
    #      markupsafe/_speedups.cp314-win_amd64.pyd via FastAPI -> Jinja2).
    # Removing the stubs is always safe — they hold no authoritative data.
    site_packages = _site_packages_for(python)
    purged: list[str] = []
    if site_packages is not None:
        purged = _purge_broken_distributions(site_packages)
        if purged:
            state.add_log(
                "server", "info",
                f"Removed {len(purged)} broken pip stub(s) from {site_packages}: {', '.join(purged)}",
            )

    attempts: list[dict[str, Any]] = []
    ok = False
    winning_output = ""
    winning_index: str | None = None

    for index_url in _CUDA_TORCH_INDEXES:
        # Two-pass install:
        #   Pass 1: --force-reinstall --no-deps swaps the torch wheel (CPU -> CUDA)
        #           without overwriting transitive deps like markupsafe. Those
        #           extensions are loaded into this Python process via FastAPI
        #           -> Jinja2; overwriting their .pyd / .so at runtime raises
        #           WinError 5 "Access is denied" and aborts the install.
        #   Pass 2: plain install (no --force) fills in any genuinely missing
        #           deps (e.g. nvidia-cublas-cu12 on Linux when swapping from
        #           CPU torch) without touching files that are already satisfied.
        cmd_swap = [
            python, "-m", "pip", "install",
            "--upgrade", "--force-reinstall", "--no-deps",
            "--index-url", index_url,
            "torch>=2.4.0",
        ]
        state.add_log("server", "info", f"Installing CUDA torch from {index_url}")
        try:
            result = subprocess.run(cmd_swap, capture_output=True, text=True, timeout=900)
            output = (result.stdout + "\n" + result.stderr).strip()
            attempt_ok = result.returncode == 0
        except subprocess.TimeoutExpired:
            output = f"Install from {index_url} timed out after 15 minutes."
            attempt_ok = False
        except OSError as exc:
            output = f"{index_url}: {exc}"
            attempt_ok = False

        if attempt_ok:
            cmd_deps = [
                python, "-m", "pip", "install",
                "--index-url", index_url,
                "torch>=2.4.0",
            ]
            try:
                dep_result = subprocess.run(cmd_deps, capture_output=True, text=True, timeout=900)
                dep_output = (dep_result.stdout + "\n" + dep_result.stderr).strip()
                output = f"{output}\n\n--- deps pass ---\n{dep_output}" if dep_output else output
            except (subprocess.TimeoutExpired, OSError):
                # Best-effort: torch itself swapped successfully, a missing
                # transitive dep will surface at runtime via an ImportError
                # the user can resolve from the Setup page.
                pass

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
    no_wheel_for_python = (not ok) and _all_attempts_lack_wheel(attempts)
    state.add_log(
        "server", "info" if ok else "error",
        f"CUDA torch install: {'succeeded via ' + winning_index if ok else 'failed after all candidates'}"
        + (f" (no wheel for Python {python_version})" if no_wheel_for_python and python_version else ""),
    )
    return {
        "ok": ok,
        "output": winning_output or (attempts[-1]["output"] if attempts else ""),
        "indexUrl": winning_index,
        "attempts": attempts,
        "requiresRestart": ok,
        "pythonExecutable": python,
        "pythonVersion": python_version,
        "noWheelForPython": no_wheel_for_python,
        "capabilities": caps,
    }


# ------------------------------------------------------------------
# GPU bundle install (torch + diffusers + video runtime deps)
# ------------------------------------------------------------------

# Packages installed by the one-click "Install GPU support" flow. Ordered
# so torch installs first — every other package below can defer to whatever
# torch version ended up on disk. If the user's Python has no CUDA wheel
# the job stops at torch (users get a clear "switch to Python 3.13" hint)
# rather than pressing on and installing diffusers against no torch.
_GPU_BUNDLE_PACKAGES: list[tuple[str, str]] = [
    ("torch", "torch>=2.4.0"),
    ("diffusers", "diffusers>=0.30.0"),
    ("accelerate", "accelerate>=0.34.0"),
    ("transformers", "transformers>=4.44.0"),
    ("safetensors", "safetensors>=0.4.5"),
    ("pillow", "pillow>=10.4.0"),
    ("huggingface-hub", "huggingface-hub>=0.26.0"),
    ("imageio", "imageio"),
    ("imageio-ffmpeg", "imageio-ffmpeg"),
    ("sentencepiece", "sentencepiece"),
    ("tiktoken", "tiktoken"),
    ("protobuf", "protobuf"),
    ("ftfy", "ftfy"),
]

# Rough total download size (torch CUDA dominates at ~2 GB; others sum to
# ~400 MB). We expose this to the UI so the install banner shows an
# honest "~2.5 GB, 1-3 min on broadband" instead of a silent multi-minute
# progress bar.
_GPU_BUNDLE_APPROX_DOWNLOAD_BYTES = 2_500_000_000

# Minimum free disk space we require before starting (download + extract +
# safety margin). Torch unpacks to ~2.5 GB, and pip holds both the wheel
# and the extracted copy during install, so we need ~5 GB of headroom.
_GPU_BUNDLE_REQUIRED_FREE_BYTES = 5_500_000_000


@dataclass
class _GpuBundleJobState:
    """In-memory status for the currently-running or most-recent install.

    Only one install runs at a time — a second POST while running returns
    the existing state. On completion the state sticks around so a late
    status poll sees the final outcome.
    """

    id: str = ""
    phase: str = "idle"  # idle | preflight | downloading | verifying | done | error
    message: str = ""
    package_current: str | None = None
    package_index: int = 0
    package_total: int = 0
    percent: float = 0.0
    target_dir: str | None = None
    index_url_used: str | None = None
    python_version: str | None = None
    no_wheel_for_python: bool = False
    cuda_verified: bool | None = None
    requires_restart: bool = False
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase,
            "message": self.message,
            "packageCurrent": self.package_current,
            "packageIndex": self.package_index,
            "packageTotal": self.package_total,
            "percent": round(self.percent, 1),
            "targetDir": self.target_dir,
            "indexUrlUsed": self.index_url_used,
            "pythonVersion": self.python_version,
            "noWheelForPython": self.no_wheel_for_python,
            "cudaVerified": self.cuda_verified,
            "requiresRestart": self.requires_restart,
            "error": self.error,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "attempts": self.attempts,
            "done": self.done,
        }


_GPU_BUNDLE_JOB = _GpuBundleJobState()
_GPU_BUNDLE_LOCK = threading.Lock()


def _extras_site_packages() -> Path | None:
    """Resolve the user-persistent extras site-packages dir.

    The Tauri shell sets ``CHAOSENGINE_EXTRAS_SITE_PACKAGES`` to a path
    like ``%LOCALAPPDATA%\\ChaosEngineAI\\extras\\site-packages`` (Windows)
    or ``~/Library/Application Support/ChaosEngineAI/extras/site-packages``
    (macOS). When the backend runs outside Tauri (``python -m backend_service``
    for dev / tests) we fall back to a predictable default so pip --target
    still has somewhere to write.
    """
    env_path = os.environ.get("CHAOSENGINE_EXTRAS_SITE_PACKAGES")
    if env_path:
        return Path(env_path)
    home = Path.home()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    return base / "ChaosEngineAI" / "extras" / "site-packages"


def _free_bytes(path: Path) -> int | None:
    """Return free disk space in bytes for the volume hosting ``path``.

    Returns None when the path doesn't exist yet AND no parent does — we
    can't check a drive we can't touch. ``shutil.disk_usage`` walks up
    until it hits an existing directory, so we mirror that.
    """
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return None
        probe = parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _run_pip_install(
    python: str,
    spec: str,
    target: Path,
    index_url: str | None,
    extra_flags: list[str],
) -> tuple[bool, str]:
    """Run ``pip install --target`` and return (ok, captured_output).

    Uses ``--upgrade`` so re-installs pick up newer versions and
    ``--target`` so we never touch the bundled site-packages (avoids the
    classic Windows WinError 5 from overwriting a loaded .pyd).
    """
    cmd = [
        python, "-m", "pip", "install",
        "--disable-pip-version-check",
        "--upgrade",
        "--target", str(target),
        *extra_flags,
    ]
    if index_url:
        cmd.extend(["--index-url", index_url])
    cmd.append(spec)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return False, f"pip install {spec} timed out after 30 minutes"
    except OSError as exc:
        return False, f"pip install {spec}: {exc}"
    output = ((result.stdout or "") + ("\n" + result.stderr if result.stderr else "")).strip()
    return result.returncode == 0, output


def _verify_cuda(python: str, extras_dir: Path) -> tuple[bool, str]:
    """Spawn a fresh Python to confirm ``torch.cuda.is_available()``.

    Uses a subprocess (not in-process import) because the backend may have
    already imported torch from the bundled CPU wheel; once sys.modules has
    a torch entry, ``import torch`` inside the running process returns the
    cached stale module. A fresh interpreter with PYTHONPATH pointing at
    extras sees the newly-installed wheel.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(extras_dir) + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import json, sys\n"
        "out = {'python': f'{sys.version_info.major}.{sys.version_info.minor}'}\n"
        "try:\n"
        "    import torch\n"
        "    out['torch'] = torch.__version__\n"
        "    out['cuda_build'] = str(getattr(torch.version, 'cuda', None))\n"
        "    out['cuda_available'] = bool(getattr(torch.cuda, 'is_available', lambda: False)())\n"
        "except Exception as exc:\n"
        "    out['error'] = str(exc).splitlines()[0][:200]\n"
        "print(json.dumps(out))\n"
    )
    try:
        result = subprocess.run(
            [python, "-c", script], capture_output=True, text=True, env=env, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"CUDA verification subprocess failed: {exc}"
    detail = (result.stdout or "").strip() + (("\n" + result.stderr) if result.stderr else "")
    # Consider the check passed only if the child exited cleanly AND said so.
    ok = result.returncode == 0 and '"cuda_available": true' in (result.stdout or "").lower()
    return ok, detail


def _install_torch_walking_indexes(
    python: str, extras_dir: Path, state: _GpuBundleJobState
) -> tuple[bool, str | None]:
    """Install torch walking the CUDA index list. First success wins."""
    for index_url in _CUDA_TORCH_INDEXES:
        state.message = f"Downloading torch from {index_url}"
        ok, output = _run_pip_install(
            python, "torch>=2.4.0", extras_dir, index_url, ["--no-deps"],
        )
        state.attempts.append({"indexUrl": index_url, "ok": ok, "output": output[-2000:]})
        if ok:
            # Second pass: install torch again with deps (no --no-deps) so
            # transitive nvidia-cublas / jinja2 / etc. land in the extras
            # tree. We keep --no-deps in the first pass to isolate the
            # winning CUDA index from transitive PyPI resolution noise.
            state.message = f"Resolving torch dependencies ({index_url})"
            dep_ok, dep_output = _run_pip_install(
                python, "torch>=2.4.0", extras_dir, index_url, [],
            )
            state.attempts.append({
                "indexUrl": index_url, "phase": "deps", "ok": dep_ok,
                "output": dep_output[-2000:],
            })
            return True, index_url
    return False, None


def _gpu_bundle_job_worker(python: str, extras_dir: Path) -> None:
    """Background-thread entry point for the GPU bundle install.

    Updates ``_GPU_BUNDLE_JOB`` as it progresses; the status endpoint reads
    that struct without locking (a stale read is fine — the field updates
    are each atomic assignments and the UI just polls again).
    """
    state = _GPU_BUNDLE_JOB
    try:
        state.phase = "preflight"
        state.message = "Checking disk space"
        free = _free_bytes(extras_dir)
        if free is not None and free < _GPU_BUNDLE_REQUIRED_FREE_BYTES:
            required_gb = _GPU_BUNDLE_REQUIRED_FREE_BYTES / 1_000_000_000
            free_gb = free / 1_000_000_000
            raise RuntimeError(
                f"Need at least {required_gb:.1f} GB free on the drive hosting "
                f"{extras_dir} — currently {free_gb:.1f} GB free. Free up space "
                "and try again."
            )

        # Sweep any broken ``~<pkg>`` stubs from a prior interrupted run —
        # they cause noisy pip warnings and occasionally block progress.
        purged = _purge_broken_distributions(extras_dir)
        if purged:
            state.message = f"Cleaned up {len(purged)} broken stub(s) from prior run"

        state.phase = "downloading"
        state.package_total = len(_GPU_BUNDLE_PACKAGES)

        # Package 1: torch (walks CUDA indexes).
        state.package_index = 1
        state.package_current = "torch"
        state.percent = 0.0
        ok, index_url = _install_torch_walking_indexes(python, extras_dir, state)
        if not ok:
            torch_attempts = [a for a in state.attempts if a.get("phase") != "deps"]
            state.no_wheel_for_python = _all_attempts_lack_wheel(torch_attempts)
            if state.no_wheel_for_python:
                raise RuntimeError(
                    f"PyTorch doesn't publish a CUDA wheel for Python {state.python_version} yet. "
                    "Rebuild ChaosEngineAI against Python 3.13 (most-widely-supported), "
                    "or set CHAOSENGINE_TORCH_INDEX_URL to a newer index before launching."
                )
            raise RuntimeError(
                "All CUDA index candidates failed. Check your internet connection, "
                "firewall, or see the attempts list for details."
            )
        state.index_url_used = index_url

        # Remaining packages: standard PyPI. Most are small — progress
        # advances quickly here so the UI doesn't look frozen.
        for idx, (label, spec) in enumerate(_GPU_BUNDLE_PACKAGES[1:], start=2):
            state.package_index = idx
            state.package_current = label
            state.percent = ((idx - 1) / len(_GPU_BUNDLE_PACKAGES)) * 100.0
            state.message = f"Installing {label}"
            ok, output = _run_pip_install(python, spec, extras_dir, None, [])
            state.attempts.append({"package": label, "ok": ok, "output": output[-2000:]})
            if not ok:
                # Individual package failure is non-fatal — torch + diffusers
                # are the must-haves and are earlier in the list. Surface
                # the warning but keep going so users aren't blocked on
                # e.g. a transient PyPI timeout for ftfy.
                state.message = (
                    f"{label} install failed (non-fatal — GPU generation will still work "
                    f"but some pipelines may need this package later)"
                )

        state.phase = "verifying"
        state.percent = 95.0
        state.package_current = None
        state.message = "Verifying CUDA availability"
        cuda_ok, detail = _verify_cuda(python, extras_dir)
        state.cuda_verified = cuda_ok
        state.attempts.append({"phase": "verify", "ok": cuda_ok, "output": detail[-2000:]})

        state.phase = "done"
        state.percent = 100.0
        state.done = True
        state.requires_restart = True
        state.finished_at = time.time()
        if cuda_ok:
            state.message = "GPU support installed. Restart the backend to activate."
        else:
            state.message = (
                "Install completed but CUDA isn't available — torch may have landed "
                "as the CPU wheel, or your NVIDIA driver doesn't match. See attempts "
                "for details."
            )
    except Exception as exc:  # noqa: BLE001 — surface ANY failure via status
        state.error = str(exc)
        state.phase = "error"
        state.message = str(exc)
        state.done = True
        state.finished_at = time.time()


@router.post("/api/setup/install-gpu-bundle")
def start_install_gpu_bundle(request: Request) -> dict[str, Any]:
    """Kick off a background install of the full GPU runtime bundle.

    Returns the current job state immediately. Poll
    ``/api/setup/install-gpu-bundle/status`` for progress. Calling this
    endpoint again while a job is running returns the running job's state
    rather than starting a new one.
    """
    state_chaosengine = request.app.state.chaosengine
    python = state_chaosengine.runtime.capabilities.pythonExecutable
    extras = _extras_site_packages()
    if extras is None:
        raise HTTPException(
            status_code=500,
            detail="Could not resolve the extras site-packages directory.",
        )
    extras.mkdir(parents=True, exist_ok=True)

    with _GPU_BUNDLE_LOCK:
        if _GPU_BUNDLE_JOB.phase in {"preflight", "downloading", "verifying"}:
            return _GPU_BUNDLE_JOB.to_dict()

        # Reset state for a fresh run.
        _GPU_BUNDLE_JOB.id = f"gpu-bundle-{int(time.time() * 1000)}"
        _GPU_BUNDLE_JOB.phase = "preflight"
        _GPU_BUNDLE_JOB.message = "Starting install"
        _GPU_BUNDLE_JOB.package_current = None
        _GPU_BUNDLE_JOB.package_index = 0
        _GPU_BUNDLE_JOB.package_total = len(_GPU_BUNDLE_PACKAGES)
        _GPU_BUNDLE_JOB.percent = 0.0
        _GPU_BUNDLE_JOB.target_dir = str(extras)
        _GPU_BUNDLE_JOB.index_url_used = None
        _GPU_BUNDLE_JOB.python_version = _read_python_version(python)
        _GPU_BUNDLE_JOB.no_wheel_for_python = False
        _GPU_BUNDLE_JOB.cuda_verified = None
        _GPU_BUNDLE_JOB.requires_restart = False
        _GPU_BUNDLE_JOB.error = None
        _GPU_BUNDLE_JOB.started_at = time.time()
        _GPU_BUNDLE_JOB.finished_at = 0.0
        _GPU_BUNDLE_JOB.attempts = []
        _GPU_BUNDLE_JOB.done = False

        thread = threading.Thread(
            target=_gpu_bundle_job_worker,
            args=(python, extras),
            name="chaosengine-gpu-bundle-install",
            daemon=True,
        )
        thread.start()

    state_chaosengine.add_log(
        "server", "info",
        f"GPU bundle install started (job={_GPU_BUNDLE_JOB.id}, target={extras})",
    )
    return _GPU_BUNDLE_JOB.to_dict()


@router.get("/api/setup/install-gpu-bundle/status")
def install_gpu_bundle_status() -> dict[str, Any]:
    """Snapshot of the current GPU bundle install job.

    Safe to poll at 1-2 Hz. Returns ``phase="idle"`` before any install
    has been started in this backend session.
    """
    return _GPU_BUNDLE_JOB.to_dict()


@router.get("/api/setup/gpu-bundle-info")
def gpu_bundle_info() -> dict[str, Any]:
    """Pre-install metadata for the install banner UI.

    Surfaces the extras target dir, approximate download size, free disk
    on the target volume, and the set of packages we intend to install so
    the frontend can render a clear "what you're about to do" confirmation.
    """
    extras = _extras_site_packages()
    extras_str = str(extras) if extras else None
    free = _free_bytes(extras) if extras else None
    return {
        "targetDir": extras_str,
        "approxDownloadBytes": _GPU_BUNDLE_APPROX_DOWNLOAD_BYTES,
        "requiredFreeBytes": _GPU_BUNDLE_REQUIRED_FREE_BYTES,
        "freeBytes": free,
        "packages": [{"label": label, "spec": spec} for label, spec in _GPU_BUNDLE_PACKAGES],
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
