"""GPU monitoring utilities for ChaosEngineAI.

Provides a unified interface for querying GPU metrics across platforms:
- macOS: Apple Silicon via sysctl + psutil
- Linux/Windows: NVIDIA GPUs via nvidia-smi
- Fallback: system RAM via psutil
"""
from __future__ import annotations

import platform
import subprocess
import json
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

        # Fallback: system RAM via psutil
        return self._fallback_psutil()

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
