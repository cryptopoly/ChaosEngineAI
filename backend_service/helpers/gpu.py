"""GPU monitoring utilities for ChaosEngineAI.

Provides a unified interface for querying GPU metrics across platforms:
- macOS: Apple Silicon via sysctl + psutil
- Linux/Windows: NVIDIA GPUs via nvidia-smi
- Fallback: system RAM via psutil
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import threading
from typing import Any


# Windows: prevent every nvidia-smi / sysctl invocation from flashing a
# console window. Without this, FastAPI worker threads on Windows pop a
# brief cmd.exe window per probe — and on slower disks the spawn alone
# can add 1-2s of latency to ``/api/video/runtime``, blowing past the
# frontend's 15s fetch timeout and surfacing as "Failed to fetch".
_SUBPROCESS_KWARGS: dict[str, Any] = {}
if hasattr(subprocess, "CREATE_NO_WINDOW"):
    _SUBPROCESS_KWARGS["creationflags"] = subprocess.CREATE_NO_WINDOW


class GPUMonitor:
    """Cross-platform GPU/accelerator monitor."""

    def __init__(self) -> None:
        self._system = platform.system()

    def snapshot(self) -> dict[str, Any]:
        """Return a dict with current GPU / accelerator metrics.

        Keys:
            gpu_name, vram_total_gb, vram_used_gb,
            utilization_pct, temperature_c, power_w
        """
        if self._system == "Darwin":
            return self._snapshot_macos()
        else:
            return self._snapshot_nvidia()

    # ------------------------------------------------------------------
    # macOS (Apple Silicon unified memory)
    # ------------------------------------------------------------------

    def _snapshot_macos(self) -> dict[str, Any]:
        gpu_name = "Apple Silicon"
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, timeout=5,
                **_SUBPROCESS_KWARGS,
            ).strip()
            if chip:
                gpu_name = chip
        except Exception:
            pass

        vram_total_gb = 0.0
        vram_used_gb = 0.0
        try:
            total_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True, timeout=5,
                **_SUBPROCESS_KWARGS,
            ).strip())
            vram_total_gb = round(total_bytes / (1024 ** 3), 2)
        except Exception:
            pass

        try:
            import psutil
            mem = psutil.virtual_memory()
            vram_used_gb = round(mem.used / (1024 ** 3), 2)
            if vram_total_gb == 0:
                vram_total_gb = round(mem.total / (1024 ** 3), 2)
        except ImportError:
            pass

        utilization_pct: float | None = None
        try:
            out = subprocess.check_output(
                ["ioreg", "-r", "-d", "1", "-c", "AppleARMIODevice"],
                text=True, timeout=5,
                **_SUBPROCESS_KWARGS,
            )
            # Best-effort — ioreg doesn't reliably expose GPU util on all chips
        except Exception:
            pass

        return {
            "gpu_name": gpu_name,
            "vram_total_gb": vram_total_gb,
            "vram_used_gb": vram_used_gb,
            "utilization_pct": utilization_pct,
            "temperature_c": None,
            "power_w": None,
        }

    # ------------------------------------------------------------------
    # NVIDIA via nvidia-smi
    # ------------------------------------------------------------------

    def _snapshot_nvidia(self) -> dict[str, Any]:
        # Try torch.cuda first — when the GPU bundle is installed it reads
        # the right total VRAM via the CUDA driver without shelling out,
        # and works even if ``nvidia-smi`` isn't on PATH (common on Windows
        # when the user installs the driver but not the CUDA toolkit).
        torch_snapshot = self._snapshot_torch_cuda()
        if torch_snapshot is not None:
            return torch_snapshot

        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
                **_SUBPROCESS_KWARGS,
            )
            parts = [p.strip() for p in out.strip().split(",")]
            if len(parts) >= 6:
                return {
                    "gpu_name": parts[0],
                    "vram_total_gb": round(float(parts[1]) / 1024, 2),
                    "vram_used_gb": round(float(parts[2]) / 1024, 2),
                    "utilization_pct": float(parts[3]),
                    "temperature_c": float(parts[4]),
                    "power_w": float(parts[5]),
                }
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            pass

        # No GPU detected — return a None-VRAM dict rather than reporting
        # system RAM as if it were VRAM. The image / video safety
        # estimators downstream treat ``vram_total_gb is None`` as
        # "unknown" and skip the crash warning, which is the correct
        # behaviour when we genuinely don't know the card's capacity.
        return self._no_gpu_detected()

    def _snapshot_torch_cuda(self) -> dict[str, Any] | None:
        """Read total + used VRAM from torch.cuda via a short-lived subprocess.

        We deliberately do NOT ``import torch`` in the backend process.
        On Windows, importing torch loads ``torch/lib/*.dll`` (asmjit,
        cublas, cudnn, ...) into the backend's process handle table,
        and pip's ``--target`` install of a fresh torch then fails with
        ``[WinError 5] Access is denied`` when ``shutil.rmtree`` tries
        to delete the locked DLLs:

            PermissionError: [WinError 5] Access is denied:
            '...\\extras\\cp312\\site-packages\\torch\\lib\\asmjit.dll'

        The fix is to query torch in a child Python process that exits
        as soon as it has printed the JSON — the OS releases the DLL
        handles, and the next ``Install GPU runtime`` click can swap
        torch in place.

        Returns ``None`` if torch isn't installed, has no CUDA build,
        no CUDA device is visible, or the subprocess errors. The caller
        then falls through to ``nvidia-smi``.
        """
        # Skip on macOS — Apple Silicon has no torch.cuda; ``_snapshot_macos``
        # owns the unified-memory path.
        if self._system == "Darwin":
            return None

        executable = self._resolve_python_executable()
        if executable is None:
            return None

        script = (
            "import json, sys\n"
            "try:\n"
            "    import torch\n"
            "except Exception:\n"
            "    sys.exit(0)\n"
            "if not getattr(torch, 'cuda', None) or not torch.cuda.is_available():\n"
            "    sys.exit(0)\n"
            "device = torch.cuda.current_device()\n"
            "props = torch.cuda.get_device_properties(device)\n"
            "total = int(props.total_memory)\n"
            "try:\n"
            "    free, _ = torch.cuda.mem_get_info(device)\n"
            "    used = max(0, total - int(free))\n"
            "except Exception:\n"
            "    used = 0\n"
            "json.dump({'gpu_name': props.name, 'total': total, 'used': used}, sys.stdout)\n"
        )

        try:
            result = subprocess.run(
                [executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=15,
                **_SUBPROCESS_KWARGS,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        payload = (result.stdout or "").strip()
        if not payload:
            return None
        try:
            data = json.loads(payload)
            total_bytes = int(data["total"])
            used_bytes = int(data.get("used") or 0)
            gpu_name = str(data.get("gpu_name") or "NVIDIA GPU")
        except (ValueError, KeyError, TypeError):
            return None
        return {
            "gpu_name": gpu_name,
            "vram_total_gb": round(total_bytes / (1024 ** 3), 2),
            "vram_used_gb": round(used_bytes / (1024 ** 3), 2),
            "utilization_pct": None,
            "temperature_c": None,
            "power_w": None,
        }

    def _resolve_python_executable(self) -> str | None:
        """Pick a Python interpreter for the torch.cuda subprocess probe.

        Prefers the embedded sidecar Python (the same one pip writes the
        GPU bundle wheels to) so ``import torch`` resolves the freshly
        installed wheel. Falls back to the running interpreter if the
        embed override isn't set.
        """
        candidates: list[str] = []
        embed = os.environ.get("CHAOSENGINE_EMBED_PYTHON_BIN")
        if embed:
            candidates.append(embed)
        candidates.append(sys.executable)
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    def _no_gpu_detected(self) -> dict[str, Any]:
        return {
            "gpu_name": "No GPU detected",
            "vram_total_gb": None,
            "vram_used_gb": None,
            "utilization_pct": None,
            "temperature_c": None,
            "power_w": None,
        }

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_psutil(self) -> dict[str, Any]:
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "gpu_name": "System RAM (no GPU detected)",
                "vram_total_gb": round(mem.total / (1024 ** 3), 2),
                "vram_used_gb": round(mem.used / (1024 ** 3), 2),
                "utilization_pct": mem.percent,
                "temperature_c": None,
                "power_w": None,
            }
        except ImportError:
            return {
                "gpu_name": "Unknown (psutil not installed)",
                "vram_total_gb": 0,
                "vram_used_gb": 0,
                "utilization_pct": None,
                "temperature_c": None,
                "power_w": None,
            }


# Convenience wrapper
_monitor = GPUMonitor()


def get_gpu_metrics() -> dict[str, Any]:
    """Return a snapshot of current GPU / accelerator metrics."""
    return _monitor.snapshot()


# VRAM total never changes for the life of a process — caching it lets the
# video runtime probe stay snappy even when nvidia-smi takes a second or two
# to spawn on Windows. Cleared by ``reset_vram_total_cache()`` for tests.
_VRAM_TOTAL_LOCK = threading.Lock()
_VRAM_TOTAL_CACHE: dict[str, float | None] = {}


def get_device_vram_total_gb() -> float | None:
    """Return total device memory in GB, cached for the process lifetime.

    Hot path for ``backend_service.video_runtime._detect_device_memory_gb``.
    The full ``snapshot()`` call shells out to ``nvidia-smi``/``sysctl`` every
    time, which is fine for the metrics endpoint (live readings) but wasteful
    for the video runtime probe (which only needs total VRAM, and a value
    that is fixed per machine). On Windows the subprocess startup cost was
    blowing past the frontend's 15s fetch timeout under load.
    """
    with _VRAM_TOTAL_LOCK:
        if "value" in _VRAM_TOTAL_CACHE:
            return _VRAM_TOTAL_CACHE["value"]

    try:
        snapshot = _monitor.snapshot()
    except Exception:
        snapshot = {}

    total = snapshot.get("vram_total_gb")
    value: float | None = float(total) if isinstance(total, (int, float)) and total > 0 else None

    with _VRAM_TOTAL_LOCK:
        _VRAM_TOTAL_CACHE["value"] = value
    return value


def reset_vram_total_cache() -> None:
    """Clear the cached VRAM total. Used by tests."""
    with _VRAM_TOTAL_LOCK:
        _VRAM_TOTAL_CACHE.clear()


def nvidia_gpu_present() -> bool:
    """Cheap, side-effect-free check for an NVIDIA GPU on Linux/Windows.

    We only look for ``nvidia-smi`` on ``PATH`` — invoking it is deliberately
    avoided because some locked-down laptops and WSL installs without the
    driver shim hang on the first call. Presence on ``PATH`` is a
    reliable-enough signal for the "you probably wanted CUDA" diagnostic the
    image/video runtimes surface when torch falls back to CPU.
    """
    return shutil.which("nvidia-smi") is not None


_CUDA_WHEEL_HINT = (
    "Click \"Install CUDA torch\" in this banner, or run: "
    "pip install --upgrade --force-reinstall torch "
    "--index-url https://download.pytorch.org/whl/cu124"
)


def gpu_status_snapshot() -> dict[str, Any]:
    """Unified GPU status for the frontend warning banner.

    Returns a dict with the host platform, whether an NVIDIA driver is
    visible, whether torch can reach CUDA / MPS, and a recommendation string
    when torch falls back to CPU on a machine with an NVIDIA GPU. All fields
    are optional so this can be called before torch has been imported without
    failing.
    """
    system = platform.system()
    nvidia_present = nvidia_gpu_present()

    torch_imported = False
    cuda_available = False
    mps_available = False
    try:
        import torch  # type: ignore
    except Exception:
        torch_module = None
    else:
        torch_module = torch
        torch_imported = True

    if torch_module is not None:
        try:
            cuda_available = bool(getattr(torch_module.cuda, "is_available", lambda: False)())
        except Exception:
            cuda_available = False
        try:
            mps_module = getattr(torch_module.backends, "mps", None)
            if mps_module is not None:
                mps_available = bool(getattr(mps_module, "is_available", lambda: False)())
        except Exception:
            mps_available = False

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
