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
from typing import Any


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
