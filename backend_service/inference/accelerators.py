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


# ---------------------------------------------------------------------------
# WSL2 vLLM bridge probes (FU-056 Phase 8)
#
# vLLM ships no native Windows wheels; the practical path on a Windows +
# CUDA box is to install vLLM inside a WSL2 Ubuntu distro and run it
# there. These three probes feed the Setup tab's WSL bridge panel +
# the future engine-routing layer:
#
#   - ``wsl_default_distro()`` → string name reported by ``wsl --status``
#     ("Ubuntu-24.04" on the dev box). The install + run paths anchor
#     on this so a user with multiple distros gets predictable
#     behaviour (always use the default).
#   - ``wsl_cuda_available()`` → ``nvidia-smi -L`` returns exit 0 from
#     inside WSL, proving CUDA passthrough works. False on stock WSL
#     installs without the NVIDIA WSL driver kicker.
#   - ``wsl_vllm_available()`` / ``wsl_vllm_version()`` → the isolated
#     venv at ``~/.chaosengine/vllm-venv`` can ``import vllm``.
#
# Every probe is gated on ``sys.platform == "win32"`` so macOS / Linux
# hosts pay zero subprocess cost for these checks (the WSL bridge has
# no meaning there). The same fallthrough pattern as ``wsl2_available``.
# ---------------------------------------------------------------------------

# Timeout sized for cold WSL service start. The first call after a
# Windows reboot can take 3-5 s while LxssManager spins up. After
# that, subsequent calls return in <100 ms.
_WSL_PROBE_TIMEOUT_SEC = 5.0

# Persistent isolated venv inside WSL. The path is rooted at the WSL
# user's $HOME — ``wsl`` resolves the leading ``~`` per-distro. This
# keeps it out of the Windows filesystem (where CUDA torch on
# ``/mnt/c/...`` would be 10x slower than on the ext4-backed home).
_WSL_VLLM_VENV_PATH = "~/.chaosengine/vllm-venv"


def _run_wsl(args: list[str], timeout: float = _WSL_PROBE_TIMEOUT_SEC) -> subprocess.CompletedProcess[bytes] | None:
    """Helper: invoke ``wsl <args>`` with a tight timeout, swallow failures.

    Returns ``None`` when the subprocess can't even start (missing
    ``wsl.exe``, host isn't Windows, etc.) so callers can branch on
    ``is None`` rather than re-handling FileNotFoundError everywhere.
    """
    if sys.platform != "win32":
        return None
    try:
        return subprocess.run(
            ["wsl", *args],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def wsl_default_distro() -> str | None:
    """Return the WSL default-distro name from ``wsl --status``.

    The line we want looks like ``Default Distribution: Ubuntu-24.04``.
    Windows emits ``wsl --status`` output as UTF-16 LE with a BOM
    (a Windows-style legacy), so we decode permissively + filter NUL
    bytes that survive a flawed decoding.
    """
    result = _run_wsl(["--status"])
    if result is None or result.returncode != 0:
        return None
    # ``wsl --status`` output is UTF-16 LE. ``decode("utf-16", errors="ignore")``
    # handles the BOM cleanly; ``replace("\x00", "")`` is a belt-and-braces
    # guard for hosts that emit raw UTF-16 without a marker.
    try:
        text = result.stdout.decode("utf-16", errors="ignore").replace("\x00", "")
    except UnicodeDecodeError:
        text = result.stdout.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        normalized = line.strip()
        if normalized.lower().startswith("default distribution"):
            # "Default Distribution: Ubuntu-24.04" → "Ubuntu-24.04"
            _, _, value = normalized.partition(":")
            distro = value.strip()
            return distro or None
    return None


def wsl_cuda_available() -> bool:
    """True when CUDA passthrough into WSL is functional.

    ``nvidia-smi -L`` lists installed GPUs and exits 0 when the NVIDIA
    WSL driver kicker is present. Without that kicker the binary
    typically isn't even reachable inside the distro, so this also
    catches the "user installed WSL but skipped the NVIDIA driver" case.
    """
    result = _run_wsl(["--", "nvidia-smi", "-L"])
    if result is None or result.returncode != 0:
        return False
    # A successful nvidia-smi line looks like ``GPU 0: NVIDIA GeForce RTX 4090``.
    # The ``-L`` output is ASCII so we don't need the UTF-16 dance.
    return b"GPU " in (result.stdout or b"")


def wsl_vllm_available() -> bool:
    """True when the WSL isolated venv has ``vllm`` importable.

    Runs the import inside the venv's python so we don't accidentally
    pick up a system-Python install of vllm — only the venv we
    manage. Same hygiene as the MTPLX detector that checks the
    dedicated ``~/.chaosengine/mtplx-venv`` path.
    """
    result = _run_wsl(
        [
            "--",
            "bash",
            "-c",
            (
                f"test -x {_WSL_VLLM_VENV_PATH}/bin/python && "
                f"{_WSL_VLLM_VENV_PATH}/bin/python -c 'import vllm' 2>/dev/null"
            ),
        ],
        timeout=8.0,
    )
    return result is not None and result.returncode == 0


def wsl_vllm_version() -> str | None:
    """Read ``vllm.__version__`` from the WSL isolated venv, or ``None``.

    Two-shot: skips the import probe if ``wsl_vllm_available()`` already
    returned False so we don't pay for a duplicate WSL roundtrip on
    machines where the venv isn't there.
    """
    if not wsl_vllm_available():
        return None
    result = _run_wsl(
        [
            "--",
            "bash",
            "-c",
            (
                f"{_WSL_VLLM_VENV_PATH}/bin/python -c "
                "'import vllm; print(getattr(vllm, \"__version__\", \"\"))'"
            ),
        ],
        timeout=8.0,
    )
    if result is None or result.returncode != 0:
        return None
    version = result.stdout.decode("utf-8", errors="ignore").strip()
    return version or None
