"""What this machine can hold: one place for "how much model fits here".

The fit badges, the recommended Studio settings and the size limits all ask the same
question, so they share this answer. ``usable_gb`` is the memory a model can occupy, not
what happens to be free right now: the Metal working-set limit on a Mac (macOS lets the GPU
wire about three quarters of unified memory, two thirds on 36 GB and smaller, unless
``iogpu.wired_limit_mb`` raises it), the card's VRAM on CUDA, and 60 % of RAM on a CPU-only
host. ``bucket`` names the tier the recommendation tables are keyed by.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HardwareClass:
    kind: str          # "apple" | "cuda" | "cpu"
    bucket: str        # "apple-16", "apple-64", "cuda-24", "cpu"
    label: str         # "Apple Silicon, 64 GB"
    total_gb: float
    usable_gb: float


def _wired_limit_gb() -> float | None:
    """``iogpu.wired_limit_mb`` when the user raised it (0 means the macOS default)."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True, timeout=2)
        value = int(out.stdout.strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return value / 1024 if value > 0 else None


def _nearest(value: float, tiers: tuple[int, ...]) -> int:
    return min(tiers, key=lambda tier: abs(tier - value))


def hardware_class(system: dict[str, Any]) -> HardwareClass:
    """Classify the machine from the system snapshot (``totalMemoryGb``, ``gpuVramTotalGb``, platform)."""
    total = float(system.get("totalMemoryGb") or 0.0)
    vram = system.get("gpuVramTotalGb")
    platform_name = str(system.get("platform") or sys.platform).lower()
    machine = str(system.get("machine") or system.get("arch") or "").lower()
    if vram:
        vram = float(vram)
        tier = _nearest(vram, (6, 8, 10, 12, 16, 20, 24, 32, 48, 80, 96))
        return HardwareClass("cuda", f"cuda-{tier}", f"NVIDIA GPU, {vram:.0f} GB VRAM", total, vram)
    if platform_name.startswith("darwin") or machine in {"arm64", "aarch64"} and platform_name.startswith("mac"):
        tier = _nearest(total, (8, 16, 24, 32, 36, 48, 64, 96, 128, 192, 256))
        fraction = 0.67 if total <= 36 else 0.75
        usable = _wired_limit_gb() or round(total * fraction, 1)
        return HardwareClass("apple", f"apple-{tier}", f"Apple Silicon, {total:.0f} GB", total, usable)
    return HardwareClass("cpu", "cpu", f"CPU only, {total:.0f} GB RAM", total, round(total * 0.6, 1))
