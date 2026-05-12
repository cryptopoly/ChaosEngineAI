"""Torch wheel + CUDA / MPS availability probe — without importing torch.

Importing torch in the long-running backend process maps
``torch/lib/*.dll`` (asmjit, cublas, cudnn, ...) into the process handle
table. Once those handles are open, pip's ``--upgrade --target extras``
install can't ``rmtree`` the existing package directories on Windows —
every retry fails with ``PermissionError: [WinError 5] Access is denied``.

So everything in this module reads ``torch/version.py`` from disk via
``importlib.util.find_spec`` for warnings, and runs the live CUDA / MPS
probe in a short-lived subprocess for the dashboard banner. The result
is cached for the backend lifetime — wheels on disk don't change without
a restart.

Extracted from ``backend_service/helpers/gpu.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.gpu`` so existing imports
keep working.
"""

from __future__ import annotations

import json
import platform
import subprocess
import threading
from typing import Any


_CUDA_WHEEL_HINT = (
    "Click \"Install CUDA torch\" in this banner, or run: "
    "pip install --upgrade --force-reinstall torch "
    "--index-url https://download.pytorch.org/whl/cu124"
)

# Cached torch availability — see ``gpu_status_snapshot``. Cleared after a
# successful GPU bundle install via ``reset_torch_status_cache``.
_TORCH_STATUS_LOCK = threading.Lock()
_TORCH_STATUS_CACHE: dict[str, dict[str, bool]] = {}


def torch_install_warning() -> str | None:
    """Detect a torch wheel/host mismatch WITHOUT importing torch.

    Three failure modes that all silently sandbag generation onto CPU:

      1. NVIDIA GPU present but torch isn't installed at all -- the GPU
         bundle never ran, so even the "Real engine ready" badge would
         be misleading.
      2. NVIDIA GPU present but the installed torch wheel is the +cpu
         build -- the bundle ran but pip resolved the CPU wheel instead
         of a CUDA one. Studio shows "Device: cuda (expected)" because
         nvidia-smi is on PATH, but generation runs on CPU because
         torch is literally CPU-only.
      3. Apple Silicon host but no torch installed -- mirrors case 1.

    Returns a one-line warning string when a mismatch is detected,
    ``None`` when everything looks fine. Importing torch would lock
    torch DLLs in the backend process and break the GPU-bundle install
    flow on Windows, so we read ``torch/version.py`` from disk instead.
    """
    import importlib.util
    from pathlib import Path

    from backend_service.helpers.gpu import nvidia_gpu_present

    spec = importlib.util.find_spec("torch")
    torch_installed = spec is not None
    torch_local_version: str | None = None
    torch_version_str: str | None = None

    if spec is not None and spec.origin:
        try:
            version_path = Path(spec.origin).with_name("version.py")
            if version_path.is_file():
                text = version_path.read_text(errors="ignore")
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("__version__"):
                        for quote in ("'", '"'):
                            if quote in stripped:
                                _, _, rest = stripped.partition(quote)
                                value, _, _ = rest.partition(quote)
                                if value:
                                    torch_version_str = value
                                    break
                        break
                if torch_version_str and "+" in torch_version_str:
                    torch_local_version = "+" + torch_version_str.split("+", 1)[1]
        except OSError:
            pass

    nvidia_present = nvidia_gpu_present()
    on_apple_silicon = (
        platform.system() == "Darwin"
        and platform.machine() in ("arm64", "aarch64")
    )

    if nvidia_present and torch_installed and torch_local_version:
        if torch_local_version.lower().startswith("+cpu"):
            return (
                f"torch is installed as a CPU-only wheel ({torch_version_str}) "
                "even though an NVIDIA GPU is present. Generation will run "
                "on CPU at a fraction of GPU speed. Open Settings > Setup "
                "and click Install CUDA torch, then Restart Backend."
            )
    if nvidia_present and not torch_installed:
        return (
            "torch is not installed but an NVIDIA GPU is present. Open "
            "Settings > Setup and click Install GPU runtime."
        )
    if on_apple_silicon and not torch_installed:
        return (
            "torch is not installed. Open Settings > Setup and click "
            "Install GPU runtime to enable Apple Silicon (MPS) generation."
        )
    return None


def _probe_torch_status_subprocess() -> dict[str, bool]:
    """Probe torch availability via a short-lived subprocess.

    See module docstring — we must NOT ``import torch`` in the backend
    process. Spawning a child Python lets us answer "is torch importable
    / does it see CUDA / MPS?" without poisoning the long-running
    backend.
    """
    from backend_service.helpers.gpu import _monitor, _SUBPROCESS_KWARGS

    executable = _monitor._resolve_python_executable()
    if executable is None:
        return {"torchImported": False, "cudaAvailable": False, "mpsAvailable": False}

    script = (
        "import json, sys\n"
        "out = {'torchImported': False, 'cudaAvailable': False, 'mpsAvailable': False}\n"
        "try:\n"
        "    import torch\n"
        "    out['torchImported'] = True\n"
        "    try:\n"
        "        out['cudaAvailable'] = bool(getattr(torch.cuda, 'is_available', lambda: False)())\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        mps = getattr(torch.backends, 'mps', None)\n"
        "        if mps is not None:\n"
        "            out['mpsAvailable'] = bool(getattr(mps, 'is_available', lambda: False)())\n"
        "    except Exception:\n"
        "        pass\n"
        "except Exception:\n"
        "    pass\n"
        "json.dump(out, sys.stdout)\n"
    )

    try:
        result = subprocess.run(
            [executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            **_SUBPROCESS_KWARGS,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"torchImported": False, "cudaAvailable": False, "mpsAvailable": False}

    if result.returncode != 0:
        return {"torchImported": False, "cudaAvailable": False, "mpsAvailable": False}

    payload = (result.stdout or "").strip()
    if not payload:
        return {"torchImported": False, "cudaAvailable": False, "mpsAvailable": False}

    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        return {"torchImported": False, "cudaAvailable": False, "mpsAvailable": False}

    return {
        "torchImported": bool(data.get("torchImported")),
        "cudaAvailable": bool(data.get("cudaAvailable")),
        "mpsAvailable": bool(data.get("mpsAvailable")),
    }


def reset_torch_status_cache() -> None:
    """Clear the cached torch status.

    Called after a successful GPU bundle install so the next health probe
    re-runs the subprocess and picks up the freshly-installed wheel rather
    than serving the pre-install "torch not importable" snapshot.
    """
    with _TORCH_STATUS_LOCK:
        _TORCH_STATUS_CACHE.clear()


def gpu_status_snapshot() -> dict[str, Any]:
    """Unified GPU status for the frontend warning banner.

    Returns a dict with the host platform, whether an NVIDIA driver is
    visible, whether torch can reach CUDA / MPS, and a recommendation string
    when torch falls back to CPU on a machine with an NVIDIA GPU. All fields
    are optional so this can be called before torch has been imported without
    failing.

    Critical: this MUST stay out-of-process. The torch availability probe
    runs in a short-lived subprocess (see ``_probe_torch_status_subprocess``)
    and the result is cached for the backend's lifetime — wheels on disk
    don't change without a restart, and importing torch into this process
    locks DLLs/PYDs that block ``/api/setup/install-gpu-bundle``.
    """
    from backend_service.helpers.gpu import nvidia_gpu_present

    system = platform.system()
    nvidia_present = nvidia_gpu_present()

    with _TORCH_STATUS_LOCK:
        cached = _TORCH_STATUS_CACHE.get("value")
    if cached is None:
        cached = _probe_torch_status_subprocess()
        with _TORCH_STATUS_LOCK:
            _TORCH_STATUS_CACHE["value"] = cached

    torch_imported = cached["torchImported"]
    cuda_available = cached["cudaAvailable"]
    mps_available = cached["mpsAvailable"]

    if system in ("Windows", "Linux") and nvidia_present and torch_imported and not cuda_available:
        recommendation = (
            "torch was imported but CUDA is unavailable — generation will run on CPU "
            "(expect minutes per step). Reinstall the CUDA wheel: "
            + _CUDA_WHEEL_HINT
        )
        warn = True
    else:
        recommendation = None
        warn = False

    return {
        "platform": system,
        "nvidiaGpuDetected": nvidia_present,
        "torchImported": torch_imported,
        "torchCudaAvailable": cuda_available,
        "torchMpsAvailable": mps_available,
        "cpuFallbackWarning": warn,
        "recommendation": recommendation,
    }
