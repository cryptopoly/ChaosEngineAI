"""Peak memory bandwidth of this machine, for the decode-speed estimate.

Generating a token reads the active weights (and the KV cache) once, so on
every platform this app supports decode speed is set by memory bandwidth far
more than by core count. Figures are the vendors' published peaks; the
estimate applies its own efficiency factor.
"""
from __future__ import annotations

import platform
import re
import subprocess
from functools import lru_cache

# Apple Silicon: (generation, tier) -> GB/s, with binned GPU variants below.
_APPLE: dict[tuple[int, str], float] = {
    (1, ""): 68, (1, "Pro"): 200, (1, "Max"): 400, (1, "Ultra"): 800,
    (2, ""): 100, (2, "Pro"): 200, (2, "Max"): 400, (2, "Ultra"): 800,
    (3, ""): 100, (3, "Pro"): 150, (3, "Max"): 400, (3, "Ultra"): 819,
    (4, ""): 120, (4, "Pro"): 273, (4, "Max"): 546,
    (5, ""): 153,
}
# Lower-binned chips with fewer GPU cores also have a narrower memory bus.
_APPLE_BINNED: dict[tuple[int, str, int], float] = {
    (3, "Max", 30): 300,
    (4, "Max", 32): 410,
}

# NVIDIA GeForce / workstation / datacentre cards, by name fragment -> GB/s.
_NVIDIA: list[tuple[str, float]] = [
    ("RTX 5090", 1792), ("RTX 5080", 960), ("RTX 5070 Ti", 896), ("RTX 5070", 672), ("RTX 5060 Ti", 448),
    ("RTX 4090", 1008), ("RTX 4080", 717), ("RTX 4070 Ti SUPER", 672), ("RTX 4070 Ti", 504),
    ("RTX 4070", 504), ("RTX 4060 Ti", 288), ("RTX 4060", 272),
    ("RTX 3090", 936), ("RTX 3080 Ti", 912), ("RTX 3080", 760), ("RTX 3070", 448), ("RTX 3060", 360),
    ("RTX 6000 Ada", 960), ("RTX A6000", 768), ("L40S", 864), ("A100", 1555), ("H100", 3350), ("H200", 4800),
]

# System RAM on a PC without a supported GPU: dual-channel DDR5 is about this.
_CPU_FALLBACK = 60.0


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _apple_gpu_cores() -> int | None:
    try:
        out = subprocess.run(
            ["ioreg", "-rc", "AGXAccelerator", "-d", "1"], capture_output=True, text=True, timeout=3,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out)
    return int(match.group(1)) if match else None


def apple_bandwidth(chip: str, gpu_cores: int | None) -> float | None:
    """Bandwidth for an Apple chip name such as ``Apple M4 Max``."""
    match = re.search(r"\bM(\d+)(?:\s+(Pro|Max|Ultra))?\b", chip)
    if not match:
        return None
    generation, tier = int(match.group(1)), match.group(2) or ""
    if gpu_cores is not None and (generation, tier, gpu_cores) in _APPLE_BINNED:
        return _APPLE_BINNED[(generation, tier, gpu_cores)]
    return _APPLE.get((generation, tier))


def nvidia_bandwidth(gpu_name: str) -> float | None:
    upper = gpu_name.upper()
    for fragment, gbps in _NVIDIA:  # longer, more specific names come first
        if fragment.upper() in upper:
            return gbps
    return None


@lru_cache(maxsize=1)
def memory_bandwidth() -> tuple[float, str, bool]:
    """``(GB/s, label, known)`` for this machine; ``known`` is False for a guess."""
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        chip = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
        cores = _apple_gpu_cores()
        gbps = apple_bandwidth(chip, cores)
        label = chip.replace("Apple ", "") + (f" ({cores}-core GPU)" if cores else "")
        if gbps:
            return gbps, label, True
        return 120.0, label, False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        out = []
    if out:
        gbps = nvidia_bandwidth(out[0])
        if gbps:
            return gbps, out[0].strip(), True
        return 400.0, out[0].strip(), False
    return _CPU_FALLBACK, "system RAM", False
