from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend_service.routes.setup._install_helpers import (
    _CUDA_TORCH_INDEXES,
    _all_attempts_lack_wheel,
    _cleanup_mlx_video_shadow_metadata,
    _extras_site_packages,
    _find_installed_torch_version,
    _is_cuda_torch_version,
    _purge_broken_distributions,
    _purge_stale_torch_from_extras,
    _read_python_version,
    _run_pip_install,
    _site_packages_for,
    _write_torch_constraint,
)

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
    # huggingface_hub imports PyYAML at module load. A partially-extracted
    # wheel in the user-local extras dir ships error.py / __init__.py that
    # don't agree on submodule layout, surfacing as
    # ``ModuleNotFoundError: No module named 'yaml.error'`` when the download
    # subprocess imports snapshot_download. Exposing pyyaml as an installable
    # package lets users repair this without reinstalling the whole bundle.
    "pyyaml": "pyyaml",
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
    # SageAttention CUDA fast-attention kernels. Wired through
    # ``backend_service/helpers/attention_backend.py`` (FU-016). Pin to 2.2.0
    # (SageAttention2++) — PyPI's default resolves to the stale 1.0.6
    # (2024-11) which lacks the SA2++ kernels. SageAttention3 lives on the
    # ``sageattention3_blackwell`` branch (Blackwell SM10.0 only) and is
    # not yet on PyPI; install path here always pulls the released SA2++
    # kernels regardless of GPU generation. No-op on macOS / CPU / non-DiT
    # pipelines — the helper guards before invoking.
    "sageattention": "sageattention==2.2.0",
    # FU-023 Nunchaku / SVDQuant — 4-bit weight quantization for FLUX
    # family + Qwen-Image + SD3.5 on CUDA. ~3× over NF4 on FLUX.1-dev.
    # CUDA only; Apple Silicon / Linux-CPU installs no-op at runtime
    # because the Nunchaku transformer subclasses fall back to the
    # stock diffusers transformer when the import fails. v1.2.1 is the
    # current pin (2026-01-25) — covers FLUX dev/Schnell/Tools/Kontext/
    # Krea, Qwen-Image + Qwen-Image-Edit, Z-Image-Turbo, SANA, PixArt-Σ.
    "nunchaku": "nunchaku>=1.2.1",
    # FU-027 NVIDIA/kvpress — KV cache compression toolkit (Apache 2.0,
    # 26 releases as of v0.5.3 / 2026-04-09). HF transformers + multi-GPU
    # Accelerate hookups. CUDA-side complement to TurboQuant on Apple
    # Silicon. Hooks land separately under cache_compression/kvpress.py
    # — installable here so the Setup tab can pre-stage the wheel before
    # the integration code goes live.
    "kvpress": "kvpress>=0.5.3",
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

_MANUAL_INSTALL_MESSAGES: dict[str, str] = {}

def _workspace_root() -> Path:
    from backend_service.app import WORKSPACE_ROOT
    return WORKSPACE_ROOT




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
# Heavy installer endpoints — extracted to focused submodules
# ------------------------------------------------------------------
#
# Each installer is either a synchronous endpoint that runs a few short
# pip calls (``cuda_torch``, ``turbo``) or a single-job background pattern
# (``gpu_bundle``, ``longlive``, ``wan_install``: POST kicks off a daemon
# thread, GET polls status, frontend renders attempt rows via
# ``InstallLogPanel``). They live in dedicated submodules to keep this
# file focused; sub-routers are included on ``router`` at the bottom
# of the file.
#
#   * ``setup/cuda_torch.py`` — recovery installer that walks the
#     PyTorch CUDA download indexes (cu124 → nightly cu128).
#   * ``setup/gpu_bundle.py`` — one-click "Install GPU support" flow:
#     torch + diffusers + transformers + video runtime deps.
#   * ``setup/longlive.py`` — clone + venv + pip requirements +
#     ~8 GB weights download. Uses its own job-state singleton.
#   * ``setup/turbo.py`` — llama-server-turbo update-check.
#   * ``setup/wan_install.py`` — download raw HF Wan checkpoint +
#     convert weights to MLX. Same pattern, separate job state.
#
# The history of why these can't ride on
# ``/api/setup/install-system-package`` (blocking 600s subprocess
# timeout vs. a 10-20 minute install) lives in the docstrings of the
# extracted modules.



from backend_service.routes.setup.cuda_torch import router as _cuda_torch_router
from backend_service.routes.setup.gpu_bundle import (
    _GPU_BUNDLE_JOB,
    _GpuBundleJobState,
    _install_torch_walking_indexes,
    _looks_like_dll_lock,
)
from backend_service.routes.setup.gpu_bundle import router as _gpu_bundle_router
from backend_service.routes.setup.longlive import router as _longlive_router
from backend_service.routes.setup.turbo import router as _turbo_router
from backend_service.routes.setup.wan_install import router as _wan_install_router

router.include_router(_cuda_torch_router)
router.include_router(_gpu_bundle_router)
router.include_router(_longlive_router)
router.include_router(_turbo_router)
router.include_router(_wan_install_router)
