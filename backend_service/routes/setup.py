from __future__ import annotations

import importlib
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

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()

_INSTALLABLE_PIP_PACKAGES: dict[str, str] = {
    "turboquant": "turboquant",
    "turboquant-mlx": "turboquant-mlx-full",
    # Not published on PyPI — install from git. Pairs with mlx_lm on macOS
    # or vllm on Linux/CUDA (see the cache_compression.triattention adapter).
    "triattention": "triattention @ git+https://github.com/WeianMao/triattention.git",
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
    # NF4 quantization for FLUX.1 Dev on consumer GPUs. Without this, the
    # 12B FLUX transformer fits in bf16 only on ≥32 GB VRAM cards; with
    # NF4 it drops to ~7 GB and runs comfortably on 4090-class hardware.
    # Windows wheels have shipped cleanly since 0.43.
    "bitsandbytes": "bitsandbytes",
    # GGUF transformer loading for FLUX, SD3, LTX-Video, HunyuanVideo, Wan.
    # Unlike bitsandbytes, gguf is pure-python + CPU-side — it works on
    # Apple Silicon and Windows without CUDA, so we ship it as the
    # cross-platform quantization option for image and video DiTs.
    "gguf": "gguf",
    # TorchAO int8 weight-only quantization. Works on CUDA and MPS — the
    # Apple Silicon FLUX path has no bitsandbytes (CUDA-only) equivalent,
    # so int8wo is how we drop the 12B transformer from ~24 GB bf16 to
    # ~12 GB on M-series Macs. Roughly half the memory saving of NF4
    # but twice the platform reach.
    "torchao": "torchao",
    # Native Apple Silicon FLUX runtime. mflux uses MLX directly instead
    # of diffusers+MPS, which is noticeably faster and doesn't hit the
    # MPS fp16-black-image edge cases. Apple Silicon only — installer
    # should hide this package on other platforms (handled upstream in
    # the capability check).
    "mflux": "mflux",
    # Apple Silicon MLX video runtime (Blaizzy/mlx-video, MIT). Subprocess
    # wrapper in backend_service.mlx_video_runtime routes Wan2.1/2.2/LTX-2
    #
    # IMPORTANT: install from git, not PyPI. The PyPI package named
    # ``mlx-video`` is an unrelated 0.1.0 utilities package (just `load`,
    # `normalize`, `resize`, `to_float`) — does NOT ship the LTX-2 / Wan
    # / HunyuanVideo generation entry points. Blaizzy's repo lives only
    # on GitHub; pin by branch so we pick up new model entries without
    # needing a PyPI release every time.
    # to native MLX kernels instead of diffusers+MPS. The capability probe
    # gates this package on Apple Silicon — installer hides it elsewhere.
    # See FU-009 in CLAUDE.md.
    "mlx-video": "mlx-video @ git+https://github.com/Blaizzy/mlx-video.git",
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


def _installable_system_packages(python_executable: str) -> dict[str, list[str]]:
    # LongLive's install runs a multi-minute clone + pip install + weight
    # download, so it needs the longer 10-minute system-install timeout
    # rather than the 5-minute pip path. We invoke it as a Python module
    # rather than a shell script so Windows hosts don't need Git Bash.
    # The installer itself rejects macOS (CUDA-only).
    return {
        "llama.cpp": ["brew", "install", "llama.cpp"],
        "llama-server-turbo": [str(_workspace_root() / "scripts" / "build-llama-turbo.sh")],
        "longlive": [python_executable, "-m", "backend_service.longlive_installer"],
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
    # Persist installs to the user-writable extras dir (mirrors GPU bundle).
    # Without --target, packaged builds install into the embedded Python's
    # site-packages inside the .app bundle, which gets reset on every app
    # rebuild/upgrade — users were losing mlx-video / triattention / etc.
    # between sessions. Tauri shell injects the same dir on PYTHONPATH, so
    # imports resolve at sidecar boot.
    extras_dir = _extras_site_packages()
    cmd = [python, "-m", "pip", "install", "--disable-pip-version-check", "--upgrade"]
    if extras_dir is not None:
        extras_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--target", str(extras_dir)])
    # ``mlx-video`` users may already have the unrelated PyPI 0.1.0
    # package on disk from before we switched to the git spec —
    # ``--upgrade`` won't always reach for a git URL when an existing
    # version is present in --target. ``--force-reinstall`` guarantees
    # the git source replaces whatever name-collides on disk.
    if body.package == "mlx-video":
        cmd.append("--force-reinstall")
    cmd.append(pip_name)
    state.add_log("server", "info", f"Installing pip package: {' '.join(cmd)}")
    cleaned_mlx_metadata: list[str] = []
    if body.package == "mlx-video" and extras_dir is not None:
        cleaned_mlx_metadata.extend(_cleanup_mlx_video_shadow_metadata(extras_dir))
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

    if body.package == "mlx-video" and extras_dir is not None:
        cleaned_mlx_metadata.extend(_cleanup_mlx_video_shadow_metadata(extras_dir))
        if cleaned_mlx_metadata:
            unique = sorted(set(cleaned_mlx_metadata))
            output = (
                f"{output}\n\nCleaned stale mlx-video metadata: "
                f"{', '.join(unique)}"
            ).strip()

    importlib.invalidate_caches()

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
    python_executable = state.runtime.capabilities.pythonExecutable
    cmd_template = _installable_system_packages(python_executable).get(body.package)
    if cmd_template is None:
        raise HTTPException(status_code=400, detail=f"System package '{body.package}' is not in the allowed install list.")

    state.add_log("server", "info", f"Installing system package: {' '.join(cmd_template)}")
    try:
        result = subprocess.run(cmd_template, capture_output=True, text=True, timeout=600)
        output = (result.stdout + "\n" + result.stderr).strip()
        ok = result.returncode == 0
    except FileNotFoundError:
        # The generic "install Homebrew" hint only makes sense when the
        # command actually starts with ``brew``; Windows LongLive installs
        # used to hit this branch and get a nonsense macOS error.
        missing = cmd_template[0]
        if missing == "brew":
            output = f"'{missing}' is not installed. Install Homebrew first: https://brew.sh"
        else:
            output = (
                f"'{missing}' is not available on PATH. "
                "Check that the backend runtime was staged correctly and retry."
            )
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


def _purge_stale_torch_from_extras(extras_dir: Path) -> list[str]:
    """Remove torch and its NVIDIA runtime deps from the extras dir.

    Reported failure mode: extras contained ``torch-2.6.0+cu124.dist-info``
    from an earlier CUDA install plus a ``torch-2.11.0+cpu`` folder from a
    later clobber. Python's importer couldn't resolve either cleanly, so
    ``import torch`` raised ``ModuleNotFoundError`` even though files were
    on disk. Wiping the family before a reinstall forces a known-clean
    slate.

    Matches by directory/file name prefix:
      - exactly ``torch`` (the package folder)
      - anything starting with ``torch-`` (dist-info, partial installs)
      - anything starting with ``nvidia_`` or ``nvidia-`` (CUDA runtime deps)

    Does NOT match sibling packages like ``torchvision`` or ``torchaudio`` —
    they start with ``torchv``/``torcha``, not ``torch-``, so the prefix
    check leaves them alone.
    """
    if not extras_dir.is_dir():
        return []
    removed: list[str] = []
    for entry in extras_dir.iterdir():
        name = entry.name
        lower = name.lower()
        is_torch = name == "torch" or lower.startswith("torch-")
        is_nvidia = lower.startswith("nvidia_") or lower.startswith("nvidia-")
        if not (is_torch or is_nvidia):
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            if not entry.exists():
                removed.append(name)
        except OSError:
            continue
    return removed


def _find_installed_torch_version(extras_dir: Path) -> str | None:
    """Return the torch version recorded in its dist-info METADATA, if any.

    Used after a successful CUDA torch install so we can pin torch in a
    constraints file for the subsequent gpu-bundle packages, preventing
    pip's resolver from silently swapping the CUDA wheel for a CPU one
    while installing diffusers/transformers/etc. from default PyPI.
    """
    if not extras_dir.is_dir():
        return None
    for entry in extras_dir.iterdir():
        lower = entry.name.lower()
        if not (lower.startswith("torch-") and lower.endswith(".dist-info")):
            continue
        metadata = entry / "METADATA"
        if not metadata.is_file():
            continue
        try:
            text = metadata.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            if line.lower().startswith("version:"):
                return line.split(":", 1)[1].strip() or None
    return None


def _write_torch_constraint(extras_dir: Path, torch_version: str) -> Path:
    """Pin torch in a constraints.txt so follow-up installs can't swap it.

    Without this pin, ``pip install diffusers --target extras/`` could let
    pip's resolver pull a newer torch from default PyPI (which ships only
    the CPU wheel) — silently replacing the CUDA wheel we just installed.
    With the pin, pip is forced to respect the exact version (including
    the ``+cu124`` local segment), and will error out if some package
    requires a strictly newer torch rather than swapping it for CPU.
    """
    path = extras_dir / ".chaosengine-torch-constraints.txt"
    path.write_text(f"torch=={torch_version}\n", encoding="utf-8")
    return path


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

    Installs land in ``extras_dir`` (the user-persistent extras tree on
    PYTHONPATH), NOT the bundled venv. The venv on packaged builds lives
    under paths that need admin to write, and a venv install would be
    wiped on the next app upgrade anyway. Extras is user-writable and
    persists across upgrades — it's also where the gpu-bundle flow
    installs, so both recovery paths agree on torch's location.

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

    extras_dir = _extras_site_packages()
    if extras_dir is None:
        raise HTTPException(
            status_code=500,
            detail="Could not resolve the extras site-packages directory.",
        )
    extras_dir.mkdir(parents=True, exist_ok=True)

    # Wipe any stale torch + nvidia-* runtime deps from extras first. A
    # prior half-installed wheel (dist-info without a matching package
    # folder, or vice versa) causes ``import torch`` to raise at runtime
    # with confusing "No module named torch" messages.
    purged_torch: list[str] = []
    try:
        purged_torch = _purge_stale_torch_from_extras(extras_dir)
    except OSError as exc:
        state.add_log("server", "warning", f"Could not purge stale torch from extras: {exc}")
    if purged_torch:
        state.add_log(
            "server", "info",
            f"Purged stale torch files from extras ({len(purged_torch)} entries)",
        )

    # Sweep pip's "~<pkg>" stub directories from the bundled site-packages.
    # These are left behind by a prior interrupted install (common on Windows
    # where Defender briefly locks .pyd files) and cause noisy "Ignoring
    # invalid distribution" warnings in future pip runs.
    site_packages = _site_packages_for(python)
    purged_stubs: list[str] = []
    if site_packages is not None:
        purged_stubs = _purge_broken_distributions(site_packages)
        if purged_stubs:
            state.add_log(
                "server", "info",
                f"Removed {len(purged_stubs)} broken pip stub(s) from {site_packages}: {', '.join(purged_stubs)}",
            )

    attempts: list[dict[str, Any]] = []
    ok = False
    winning_output = ""
    winning_index: str | None = None

    for index_url in _CUDA_TORCH_INDEXES:
        state.add_log("server", "info", f"Installing CUDA torch from {index_url}")
        # Two-pass install into extras (via --target in _run_pip_install):
        #   Pass 1: --force-reinstall --no-deps swaps the torch wheel even
        #           when a same-versioned CPU wheel is already present
        #           (PEP 440 treats 2.6.0+cpu == 2.6.0+cu124 as equal for
        #           upgrade purposes, so --force-reinstall is required).
        #   Pass 2: plain install (no --force) fills transitive deps like
        #           nvidia-cublas-cu12 without clobbering files held by
        #           the running backend process.
        swap_ok, swap_output = _run_pip_install(
            python, "torch>=2.4.0", extras_dir, index_url,
            ["--force-reinstall", "--no-deps"],
        )
        combined_output = swap_output
        if swap_ok:
            _dep_ok, dep_output = _run_pip_install(
                python, "torch>=2.4.0", extras_dir, index_url, [],
            )
            if dep_output:
                combined_output = f"{swap_output}\n\n--- deps pass ---\n{dep_output}"

        attempts.append({
            "indexUrl": index_url,
            "ok": swap_ok,
            "output": combined_output,
        })
        if swap_ok:
            ok = True
            winning_output = combined_output
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
        "targetDir": str(extras_dir),
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
    # NF4 quantization for FLUX's 12B transformer — shrinks it from ~24 GB
    # (bf16) to ~7 GB so it runs comfortably on 24 GB consumer GPUs. With
    # bf16 + cpu_offload alone, a 4090 is right at the edge of VRAM and
    # pays a heavy pagefile-thrash cost per step.
    ("bitsandbytes", "bitsandbytes>=0.43.0"),
    # GGUF loader for image/video DiT transformers. Cross-platform
    # quantization (works on CUDA, MPS, CPU) complementing the
    # CUDA-only bitsandbytes NF4 path.
    ("gguf", "gguf>=0.10.0"),
    # TorchAO int8wo — Apple Silicon's answer to NF4 for FLUX. Drops
    # the 12B transformer from ~24 GB to ~12 GB on MPS so FLUX fits in
    # 32 GB unified memory without pagefile thrash.
    ("torchao", "torchao>=0.6.0"),
]

# Apple Silicon: ship mlx-video alongside the diffusers GPU bundle so the
# MLX-native LTX-2 engine is available out of the box. Skipped on Intel
# Macs and non-Darwin hosts where mlx-video has no working backend.
if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
    _GPU_BUNDLE_PACKAGES.append((
        "mlx-video",
        "mlx-video @ git+https://github.com/Blaizzy/mlx-video.git",
    ))

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
    namespaced by Python ``major.minor`` (e.g.
    ``~/Library/Application Support/ChaosEngineAI/extras/cp312/site-packages``)
    so wheels compiled against one Python ABI can't shadow a different
    interpreter on the next launch — that bit users in v0.7.0-rc.4 when
    a switch from cp311 to cp312 left a dead pydantic_core wheel in
    place. When the backend runs outside Tauri (``python -m backend_service``
    for dev / tests) we fall back to a predictable default that uses the
    *current* interpreter's tag.
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
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return base / "ChaosEngineAI" / "extras" / tag / "site-packages"


def _cleanup_mlx_video_shadow_metadata(extras_dir: Path) -> list[str]:
    """Remove stale PyPI ``mlx-video`` dist-info folders from ``--target``.

    Blaizzy's generator package and the unrelated PyPI preprocessing package
    share the normalized project name ``mlx-video``. pip's ``--target`` mode
    can leave both ``mlx_video-*.dist-info`` folders behind after a forced git
    reinstall, which makes version/provenance checks ambiguous even when the
    importable package directory was correctly overwritten.
    """
    removed: list[str] = []
    if not extras_dir.exists():
        return removed
    for dist_info in extras_dir.glob("mlx_video-*.dist-info"):
        metadata_path = dist_info / "METADATA"
        try:
            metadata = metadata_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            metadata = ""
        if "github.com/Blaizzy/mlx-video" in metadata:
            continue
        shutil.rmtree(dist_info, ignore_errors=True)
        removed.append(dist_info.name)
    return removed


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


_DLL_LOCK_PATTERNS = (
    # Windows pip rmtree failure signatures when a torch DLL is held open by
    # another process (typically the ChaosEngineAI backend that eagerly
    # imported torch before the install started).
    "winerror 5",
    "permissionerror",
    "access is denied",
)


def _looks_like_dll_lock(output: str) -> bool:
    """Heuristic: does pip's stderr look like a locked-DLL rmtree failure?

    The backend eagerly imported torch at startup for warmup speed, and
    once torch/lib/*.dll is in the process handle table pip can't remove
    those files even with --force-reinstall --target. Detecting this
    specifically lets us surface a clear "restart backend, then retry"
    message instead of burying the root cause under a wall of pip trace.
    """
    lowered = output.lower()
    if "torch" not in lowered or ".dll" not in lowered:
        return False
    return any(marker in lowered for marker in _DLL_LOCK_PATTERNS)


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
        if not ok and _looks_like_dll_lock(output):
            # Stop walking indexes — no index will succeed until the DLLs
            # are released. Raise so the worker captures a clean error
            # with an actionable message instead of four duplicate
            # "WinError 5" attempt rows.
            raise RuntimeError(
                "Cannot overwrite existing torch files because they're locked by the running "
                "backend (likely a previous partial install). Click Restart Backend, wait for "
                "it to come back online, then click Install GPU runtime again. If the problem "
                "persists, quit ChaosEngineAI fully, delete "
                f"{extras_dir / 'torch'}, and reopen."
            )
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

    Failure handling:
      - Fatal: the worker raises, ``except`` block sets ``phase=error`` +
        ``error`` + ``message`` from the exception text.
      - Non-fatal (post-torch package install fails): the loop appends a
        FAIL attempt with full output and keeps going. At the end we sum
        non-fatal failures into a final message so the UI doesn't show
        ``done`` with a green tick when half the bundle didn't land.
    """
    state = _GPU_BUNDLE_JOB
    non_fatal_failures: list[str] = []
    # Apple Silicon hosts ship torch (MPS-enabled) inside the bundled venv —
    # walking CUDA indexes here would fail (no aarch64-darwin CUDA wheels)
    # and abort the rest of the bundle. Skip the torch step on macOS arm64
    # so diffusers + mlx-video still install. Mirror the cuda-verify skip
    # at the tail so the summary stays accurate.
    is_apple_silicon = (
        platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")
    )
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

        if is_apple_silicon:
            # Skip torch CUDA walk — torch is already in the bundled venv
            # (MPS-enabled). Mark the slot as accounted for and proceed to
            # diffusers + mlx-video.
            state.package_index = 1
            state.package_current = "torch"
            state.percent = 0.0
            state.attempts.append({
                "package": "torch",
                "ok": True,
                "output": "Apple Silicon: using bundled MPS torch (CUDA install skipped)",
            })
            ok = True
            index_url = None
        else:
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
            # Pull the most recent failure tail so the error message
            # itself is actionable (no blank "All indexes failed" toast).
            last_attempt = state.attempts[-1] if state.attempts else {}
            tail = (last_attempt.get("output") or "").splitlines()[-3:]
            tail_blob = " | ".join(line.strip() for line in tail if line.strip())[:300]
            raise RuntimeError(
                "All CUDA index candidates failed. Check your internet connection, "
                f"firewall, or proxy settings. Last pip output: {tail_blob or '(empty)'}"
            )
        state.index_url_used = index_url

        # Pin torch in a constraints file so the follow-up packages
        # (diffusers, transformers, etc.) can't cause pip to swap the
        # CUDA wheel for a CPU one from default PyPI. Without the pin,
        # the resolver occasionally decides a fresh torch satisfies some
        # transitive upper bound better than the installed CUDA wheel,
        # and silently overwrites it. Any package that strictly requires
        # a different torch version will now error out visibly against
        # the constraint instead of silently clobbering torch.
        constraint_path: Path | None = None
        torch_version = _find_installed_torch_version(extras_dir)
        if torch_version:
            try:
                constraint_path = _write_torch_constraint(extras_dir, torch_version)
                state.attempts.append({
                    "phase": "constraint",
                    "ok": True,
                    "output": f"Pinned torch=={torch_version} for subsequent packages",
                })
            except OSError as exc:
                # Non-fatal: we just lose the torch pin for this run. The
                # packages below might or might not clobber torch, but the
                # verify step at the end will detect that.
                state.attempts.append({
                    "phase": "constraint",
                    "ok": False,
                    "output": f"Could not write torch constraint: {exc}",
                })

        # Remaining packages: standard PyPI. Most are small — progress
        # advances quickly here so the UI doesn't look frozen.
        for idx, (label, spec) in enumerate(_GPU_BUNDLE_PACKAGES[1:], start=2):
            state.package_index = idx
            state.package_current = label
            state.percent = ((idx - 1) / len(_GPU_BUNDLE_PACKAGES)) * 100.0
            state.message = f"Installing {label}"
            extra_flags: list[str] = []
            if constraint_path is not None:
                extra_flags = ["--constraint", str(constraint_path)]
            ok, output = _run_pip_install(python, spec, extras_dir, None, extra_flags)
            if label == "mlx-video":
                cleaned = _cleanup_mlx_video_shadow_metadata(extras_dir)
                if cleaned:
                    output = (
                        f"{output}\n\nCleaned stale mlx-video metadata: "
                        f"{', '.join(sorted(set(cleaned)))}"
                    ).strip()
            state.attempts.append({"package": label, "ok": ok, "output": output[-2000:]})
            if not ok:
                # Individual package failure is non-fatal — torch + diffusers
                # are the must-haves and torch is earlier in the list. Track
                # the failure for the final summary so the UI doesn't show
                # a clean "done" when ftfy/sentencepiece/etc. didn't land.
                non_fatal_failures.append(label)
                state.message = (
                    f"{label} install failed (non-fatal — see install log; you can "
                    f"retry it individually after the bundle finishes)"
                )

        state.phase = "verifying"
        state.percent = 95.0
        state.package_current = None
        if is_apple_silicon:
            # No CUDA on Apple Silicon — bundled torch already gives us MPS.
            # Mark verify as a pass so the UI doesn't show a red verify badge
            # on a successful Apple Silicon install.
            state.message = "Apple Silicon — skipping CUDA verify (MPS via bundled torch)"
            cuda_ok = True
            detail = "skipped on Apple Silicon"
        else:
            state.message = "Verifying CUDA availability"
            cuda_ok, detail = _verify_cuda(python, extras_dir)
        state.cuda_verified = cuda_ok
        state.attempts.append({"phase": "verify", "ok": cuda_ok, "output": detail[-2000:]})

        state.phase = "done"
        state.percent = 100.0
        state.done = True
        state.requires_restart = True
        state.finished_at = time.time()
        if cuda_ok and not non_fatal_failures:
            state.message = "GPU support installed. Restart the backend to activate."
        elif cuda_ok and non_fatal_failures:
            # Surface the partial failure so users know to retry the
            # individual missing pieces (mp4 encoder, tokenizers) rather
            # than re-running the whole 2 GB torch install.
            state.message = (
                "GPU support installed and CUDA verified, but "
                f"{len(non_fatal_failures)} optional package(s) failed: "
                f"{', '.join(non_fatal_failures)}. Restart the backend to activate "
                "torch + diffusers; the failed packages can be retried individually."
            )
        else:
            verify_tail = (detail or "").splitlines()[-2:]
            verify_blob = " | ".join(line.strip() for line in verify_tail if line.strip())[:300]
            state.message = (
                "Install completed but CUDA isn't available. torch may have landed "
                "as the CPU wheel, or your NVIDIA driver doesn't match. "
                f"Verify subprocess said: {verify_blob or '(no output)'}. "
                "See the install log for the full attempts list."
            )
    except Exception as exc:  # noqa: BLE001 — surface ANY failure via status
        # Always set a non-empty message: ``str(exc)`` can be empty for
        # bare-Exception cases and that's exactly when the UI ends up
        # showing "failed without reason". Fall back to the exception
        # type name so users see SOMETHING actionable.
        message = str(exc) or f"{type(exc).__name__} (no message attached)"
        state.error = message
        state.phase = "error"
        state.message = message
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
# LongLive async install job
# ------------------------------------------------------------------
#
# Mirrors the GPU bundle job pattern above: kick off an install in a
# daemon thread, expose start + status endpoints, the frontend polls
# the status endpoint at ~1.5 Hz and renders the live terminal output
# via ``InstallLogPanel``.
#
# We can't run the LongLive install through the ordinary
# ``/api/setup/install-system-package`` route because:
#   1. That route blocks for the entire install with ``capture_output=True``,
#      so the user sees no progress for 10-20 minutes.
#   2. The 600 s subprocess timeout would SIGTERM real installs — fresh
#      installs on residential connections download ~8 GB of weights
#      after pip's ~30 packages, which routinely exceeds 10 minutes.
#
# The async job here uses ``LongLiveLogStreamer`` to write each line of
# the installer's stdout into the job's attempt buffer in real time. The
# frontend ``InstallLogPanel`` already supports phased attempts via
# ``GpuBundleAttempt``; we reuse the same shape.


@dataclass
class _LongLiveJobState:
    """In-memory status for the currently-running or most-recent LongLive install.

    Same single-job semantics as the GPU bundle: a second POST while
    running returns the running job's state. State sticks around after
    completion so a late status poll sees the final outcome.
    """

    id: str = ""
    phase: str = "idle"  # idle | preflight | downloading | verifying | done | error
    message: str = ""
    package_current: str | None = None
    package_index: int = 0
    package_total: int = 0
    percent: float = 0.0
    target_dir: str | None = None
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
            "error": self.error,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "attempts": self.attempts,
            "done": self.done,
        }


_LONGLIVE_JOB = _LongLiveJobState()
_LONGLIVE_LOCK = threading.Lock()


# Friendly labels for the phases declared in
# ``backend_service.longlive_installer.INSTALL_PHASES``. Used by the
# job worker to populate ``package_current`` so the UI shows
# "Step 4/9: Installing requirements" instead of "pip-requirements".
_LONGLIVE_PHASE_LABELS: dict[str, str] = {
    "clone": "Clone LongLive repo",
    "venv": "Create isolated venv",
    "pip-upgrade": "Upgrade pip / setuptools / wheel",
    "pip-requirements": "Install LongLive requirements",
    "flash-attn": "Build flash-attn (optional)",
    "pip-hub": "Install huggingface-hub",
    "weights-longlive": "Download LongLive checkpoints (~5 GB)",
    "weights-wan": "Download Wan 2.1 base (~3 GB)",
    "marker": "Write ready marker",
}


def _longlive_job_worker() -> None:
    """Run ``longlive_installer.install`` and stream its output into the job state.

    All exceptions land in ``job.error`` rather than escaping — the daemon
    thread has no parent to receive them. The job's ``phase`` flips to
    ``error`` on any failure and ``done`` on a clean run.
    """
    # Local import: ``backend_service.longlive_installer`` pulls in
    # ``longlive_engine`` which imports torch/transformers shims at module
    # load time on some paths. Deferring the import keeps the route module
    # importable in environments where those aren't installed (notably the
    # macOS test box and CI).
    from backend_service import longlive_installer  # noqa: PLC0415

    job = _LONGLIVE_JOB

    # Buffer of streamed log lines. We chunk the buffer into one
    # ``output`` field per phase so each row in the InstallLogPanel
    # holds the lines that landed during that phase. The current
    # phase's buffer accumulates until ``progress()`` fires.
    phase_buffer: list[str] = []
    current_phase: dict[str, object] = {"name": "preflight"}
    total_phases = len(longlive_installer.INSTALL_PHASES)

    def push_attempt(phase: str, ok: bool) -> None:
        """Emit one row for ``InstallLogPanel`` and reset the buffer."""
        job.attempts.append(
            {
                "phase": phase,
                "package": _LONGLIVE_PHASE_LABELS.get(phase, phase),
                "ok": ok,
                "output": "\n".join(phase_buffer)[-8000:],
            }
        )
        phase_buffer.clear()

    def stream_log(line: str) -> None:
        # Each line of installer output appends to the active phase's
        # buffer. ``InstallLogPanel`` tail-limits to 80 lines per
        # attempt anyway, but we cap the raw buffer at 8000 chars so
        # a misbehaving subprocess can't blow the response payload.
        phase_buffer.append(line)
        if len(phase_buffer) > 400:  # ~80 displayed + headroom for filter
            del phase_buffer[: len(phase_buffer) - 400]

    def report_progress(event: dict[str, object]) -> None:
        # Phase transition: flush the current phase's buffer as one
        # attempt row, advance the counter, set the new label.
        phase_name = str(event.get("phase") or "")
        ok = bool(event.get("ok"))
        push_attempt(phase_name, ok)
        if not ok:
            job.phase = "error"
            return
        try:
            idx = longlive_installer.INSTALL_PHASES.index(phase_name)
        except ValueError:
            return
        next_idx = idx + 1
        job.package_index = next_idx
        job.percent = (next_idx / total_phases) * 100.0
        if next_idx < total_phases:
            next_phase = longlive_installer.INSTALL_PHASES[next_idx]
            current_phase["name"] = next_phase
            job.package_current = _LONGLIVE_PHASE_LABELS.get(next_phase, next_phase)
            job.message = f"Running: {job.package_current}"

    job.phase = "downloading"  # mirror the GPU bundle phase name so the UI label matches
    job.message = "Starting LongLive install"
    job.package_current = _LONGLIVE_PHASE_LABELS.get("clone", "clone")
    job.package_total = total_phases

    try:
        longlive_installer.install(
            logger=stream_log,
            progress=report_progress,
        )
    except longlive_installer.LongLiveInstallError as exc:
        # Flush whatever was buffered during the failing phase, then
        # mark the job errored. ``current_phase['name']`` is the phase
        # that was running when the exception fired.
        if phase_buffer:
            push_attempt(str(current_phase["name"]), ok=False)
        job.phase = "error"
        job.error = str(exc)
        job.message = f"LongLive install failed: {exc}"
    except Exception as exc:  # noqa: BLE001 — surface unexpected failures
        if phase_buffer:
            push_attempt(str(current_phase["name"]), ok=False)
        job.phase = "error"
        job.error = f"Unexpected error: {exc}"
        job.message = job.error
    else:
        job.phase = "done"
        job.percent = 100.0
        job.package_index = total_phases
        job.package_current = None
        job.message = "LongLive install complete."
    finally:
        job.finished_at = time.time()
        job.done = True


@router.post("/api/setup/install-longlive")
def start_install_longlive(request: Request) -> dict[str, Any]:
    """Kick off a background LongLive install.

    Returns the current job state immediately. Poll
    ``/api/setup/install-longlive/status`` for progress. Calling this
    endpoint again while a job is running returns the running job's
    state rather than starting a new one.
    """
    state_chaosengine = request.app.state.chaosengine

    # Resolve the install root the same way the installer does so the
    # UI can show it in the ``InstallLogPanel`` meta line. We import
    # locally for the same reason as the worker above.
    from backend_service.longlive_engine import resolve_install  # noqa: PLC0415

    info = resolve_install()

    with _LONGLIVE_LOCK:
        if _LONGLIVE_JOB.phase in {"preflight", "downloading", "verifying"}:
            return _LONGLIVE_JOB.to_dict()

        # Reset state for a fresh run.
        from backend_service import longlive_installer  # noqa: PLC0415

        _LONGLIVE_JOB.id = f"longlive-{int(time.time() * 1000)}"
        _LONGLIVE_JOB.phase = "preflight"
        _LONGLIVE_JOB.message = "Starting install"
        _LONGLIVE_JOB.package_current = _LONGLIVE_PHASE_LABELS["clone"]
        _LONGLIVE_JOB.package_index = 0
        _LONGLIVE_JOB.package_total = len(longlive_installer.INSTALL_PHASES)
        _LONGLIVE_JOB.percent = 0.0
        _LONGLIVE_JOB.target_dir = str(info.root)
        _LONGLIVE_JOB.error = None
        _LONGLIVE_JOB.started_at = time.time()
        _LONGLIVE_JOB.finished_at = 0.0
        _LONGLIVE_JOB.attempts = []
        _LONGLIVE_JOB.done = False

        thread = threading.Thread(
            target=_longlive_job_worker,
            name="chaosengine-longlive-install",
            daemon=True,
        )
        thread.start()

    state_chaosengine.add_log(
        "server", "info",
        f"LongLive install started (job={_LONGLIVE_JOB.id}, target={info.root})",
    )
    return _LONGLIVE_JOB.to_dict()


@router.get("/api/setup/install-longlive/status")
def install_longlive_status() -> dict[str, Any]:
    """Snapshot of the current LongLive install job. Safe to poll at 1-2 Hz."""
    return _LONGLIVE_JOB.to_dict()


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
