"""Probe helpers for CUDA-side accelerator packages (FU-056 Phase 1).

Lazy importability + version probes for the five accelerators the
Setup tab + per-feature install panels expose:

- **nunchaku** — SVDQuant 4-bit transformers for FLUX / SD3.5 / Qwen-Image
  (FU-023). Pulled in by ``ImageStudio`` when a DiT pipeline loads with
  ``nunchakuRepo`` pinned. CUDA-only at runtime, but the import itself
  succeeds on any platform so the capability flag tracks "package usable"
  rather than "package will accelerate this machine".
- **sageattention** — fast attention kernels for DiT pipelines on CUDA
  (FU-016). Stacks multiplicatively with FBCache / Nunchaku. No-op on
  Apple Silicon and on UNet pipelines.
- **dflash CUDA** — PyTorch/CUDA half of the speculative decoding family
  (FU-031, FU-048). ``dflash.is_vllm_available()`` already exists in the
  local ``dflash/__init__.py`` wrapper and inspects the ``dflash.model``
  submodule, so we delegate to it rather than re-detecting here.
- **triattention** — vLLM compressor used by FU-003 LongLive on CUDA
  and FU-002 on Apple Silicon. The pip name + import name agree
  (``triattention``).
- **kvpress** — NVIDIA KV cache compression toolkit (FU-027). Already
  registered in ``_INSTALLABLE_PIP_PACKAGES`` but had no capability flag
  before this phase; integration code arrives in a later phase, but the
  install button needs the flag to gate "Installed ✓" state.

Plus a Windows-specific ``wsl2_available()`` helper used by the future
Phase 8 vLLM-via-WSL bridge. On macOS/Linux it's always ``False`` — the
flag only carries weight on Windows where ``vllm`` has no native wheels.

Probes are deliberately lazy: every ``import`` lives inside a function
body so ``python -X importtime backend_service.app`` stays under the 2 s
cold-start budget (per CLAUDE.md performance guidelines). The companion
``_version`` helpers return ``None`` if the package isn't installed —
callers don't need a separate availability check before reading them.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys


def _spec_exists(module_name: str) -> bool:
    """``importlib.util.find_spec`` wrapper that swallows ModuleNotFoundError.

    ``find_spec`` can raise on partially-broken installs (e.g. a torch
    directory that exists on disk but has no ``__init__.py``) — see the
    Windows torch install bug investigated 2026-05-17. We treat any raise
    as "not available" so the capability resolver never crashes on a half-
    installed package.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _safe_version(module_name: str) -> str | None:
    """Read ``__version__`` without crashing on broken installs.

    Mirrors the half-broken-install resilience of ``_spec_exists``: a
    package that registers an import spec but has no Python source (the
    Windows ``torch/`` failure mode) raises on attribute access, not on
    ``find_spec``. Catching here keeps the capability payload honest.
    """
    if not _spec_exists(module_name):
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


# ---------------------------------------------------------------------------
# Nunchaku — FU-023
# ---------------------------------------------------------------------------

def nunchaku_available() -> bool:
    return _spec_exists("nunchaku")


def nunchaku_version() -> str | None:
    return _safe_version("nunchaku")


# ---------------------------------------------------------------------------
# SageAttention — FU-016
# ---------------------------------------------------------------------------

def sageattention_available() -> bool:
    return _spec_exists("sageattention")


def sageattention_version() -> str | None:
    return _safe_version("sageattention")


# ---------------------------------------------------------------------------
# DFlash — FU-031 (MLX side) + FU-048 (CUDA side)
#
# Two flags here because the two backends live in two separate pip
# packages with two import names (``dflash_mlx`` for Apple Silicon,
# ``dflash.model`` for CUDA). The shared ``dflash`` integration module
# already exposes detection helpers; reuse them so the wrapping stays
# in one place if the upstream package layout changes.
# ---------------------------------------------------------------------------

def dflash_mlx_available() -> bool:
    """``dflash_mlx`` (Apple Silicon) — the MLX-native draft runner."""
    try:
        from dflash import is_mlx_available
    except ImportError:
        return False
    try:
        return bool(is_mlx_available())
    except Exception:
        return False


def dflash_cuda_available() -> bool:
    """``dflash`` PyPI package (CUDA) — the PyTorch/CUDA draft runner.

    Uses the integration module's existing helper, which checks for the
    ``dflash.model`` submodule specifically (the local ``dflash/`` wrapper
    in this repo shadows the bare ``dflash`` import, so the submodule
    check is what disambiguates "real upstream package" from "our shim").
    """
    try:
        from dflash import is_vllm_available
    except ImportError:
        return False
    try:
        return bool(is_vllm_available())
    except Exception:
        return False


def dflash_mlx_version() -> str | None:
    return _safe_version("dflash_mlx")


def dflash_cuda_version() -> str | None:
    """The CUDA wheel exposes its version via ``dflash.model.__version__``
    when installed, but our local wrapper ``dflash/__init__.py`` shadows
    the bare name. Probe the submodule path the upstream package owns.
    """
    if not dflash_cuda_available():
        return None
    return _safe_version("dflash.model")


# ---------------------------------------------------------------------------
# TriAttention — FU-002 (MLX) + FU-003 LongLive (CUDA)
# ---------------------------------------------------------------------------

def triattention_available() -> bool:
    return _spec_exists("triattention")


def triattention_version() -> str | None:
    return _safe_version("triattention")


# ---------------------------------------------------------------------------
# kvpress — FU-027 (capability flag now; integration in a later phase)
# ---------------------------------------------------------------------------

def kvpress_available() -> bool:
    return _spec_exists("kvpress")


def kvpress_version() -> str | None:
    return _safe_version("kvpress")


# ---------------------------------------------------------------------------
# WSL2 — Windows-only bridge for vLLM (FU-056 Phase 8)
#
# Pure no-op on macOS / Linux. On Windows we shell ``wsl --status`` with
# a tight timeout. The two-second timeout covers cold WSL service starts
# without hanging the capability probe — repeated calls are throttled by
# the capability cache, so a slow first probe doesn't compound.
# ---------------------------------------------------------------------------

def wsl2_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        result = subprocess.run(
            ["wsl", "--status"],
            capture_output=True,
            timeout=2.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0
