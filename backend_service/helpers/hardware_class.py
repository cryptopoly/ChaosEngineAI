"""What this machine can hold: one place for "how much model fits here".

The fit badges, the recommended Studio settings and the size limits all ask the same
question, so they share this answer. ``usable_gb`` is the memory a model can occupy, not
what happens to be free right now: the Metal working-set limit on a Mac (read once from
the GPU through a short subprocess, because the backend never imports mlx; macOS lets the
GPU wire about three quarters of unified memory, two thirds on 36 GB and smaller, when
that read is not available), the card's VRAM on CUDA, and 60 % of RAM on a CPU-only host.
``bucket`` names the tier the recommendation tables are keyed by.

Reads that describe this very machine (the Metal limit, ``iogpu.wired_limit_mb``) are used
only when the system dict being classified is this machine, so a test or a remote report
with another memory size gets the plain heuristic.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HardwareClass:
    kind: str          # "apple" | "cuda" | "cpu"
    bucket: str        # "apple-16", "apple-64", "cuda-24", "cpu"
    label: str         # "Apple Silicon, 64 GB"
    total_gb: float
    usable_gb: float


_GIB = 1024 ** 3
# Total memory of the described system may differ from this machine's by rounding.
_HOST_MATCH_TOLERANCE_GB = 0.6
_METAL_PROBE_TIMEOUT_SECONDS = 8.0
# How long a caller waits for the read before using the heuristic for this call.
_METAL_WAIT_SECONDS = 6.0
# A failed probe (slow cold import, busy machine) is retried after this long.
_METAL_FAILURE_TTL_SECONDS = 300.0
_HOST_CLASS_TTL_SECONDS = 60.0

_METAL_PROBE_CODE = (
    "import json, mlx.core as mx\n"
    "info = (getattr(mx, 'device_info', None) or mx.metal.device_info)()\n"
    "print(json.dumps({'workingSetBytes': info.get('max_recommended_working_set_size'),"
    " 'memoryBytes': info.get('memory_size')}))\n"
)

_metal_lock = threading.Lock()
_metal_cache: dict[str, Any] = {}


def _on_macos() -> bool:
    return sys.platform == "darwin"


def _wired_limit_gb() -> float | None:
    """``iogpu.wired_limit_mb`` when the user raised it (0 means the macOS default)."""
    if not _on_macos():
        return None
    try:
        out = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True, timeout=2)
        value = int(out.stdout.strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return value / 1024 if value > 0 else None


def _read_metal_working_set_gb() -> tuple[float, float] | None:
    """``(memory GB, working-set GB)`` from one throwaway subprocess, or None.

    The GPU's own recommended working-set limit is the number MLX and llama.cpp stay
    under; it is 51.8 GB on a 64 GB M4 Max, not the 48 GB a flat 0.75 share gives. Only
    device info is read, no model is loaded. Never raises.
    """
    try:
        from backend_service.inference.binaries import _json_subprocess, _resolve_mlx_python

        code, payload, _message = _json_subprocess(
            [_resolve_mlx_python(), "-c", _METAL_PROBE_CODE],
            timeout=_METAL_PROBE_TIMEOUT_SECONDS,
        )
        if code != 0 or not isinstance(payload, dict):
            return None
        working_set = float(payload.get("workingSetBytes") or 0)
        memory = float(payload.get("memoryBytes") or 0)
        if working_set <= 0 or memory <= 0 or working_set > memory:
            return None
        return memory / _GIB, working_set / _GIB
    except Exception:
        return None


def _probe_metal_in_background() -> None:
    value = _read_metal_working_set_gb()
    with _metal_lock:
        if value is None:
            _metal_cache["failedAt"] = time.monotonic()
        else:
            _metal_cache["value"] = value
            _metal_cache.pop("failedAt", None)
        _metal_cache.pop("thread", None)
    # A class worked out while the read was still running used the heuristic.
    with _host_lock:
        _host_cache.clear()


def _metal_info() -> tuple[float, float] | None:
    """Cached ``_read_metal_working_set_gb``: kept for the process on success, 5 minutes on failure.

    The read runs on a background thread and a caller waits at most ``_METAL_WAIT_SECONDS`` for it,
    so a cold interpreter start never stalls a catalog request; until the read lands the caller
    gets None (the heuristic) and the next call sees the real value.
    """
    with _metal_lock:
        cached = _metal_cache.get("value")
        if cached is not None:
            return cached
        failed_at = _metal_cache.get("failedAt")
        if failed_at is not None and time.monotonic() - failed_at < _METAL_FAILURE_TTL_SECONDS:
            return None
        thread = _metal_cache.get("thread")
        if thread is None:
            thread = threading.Thread(target=_probe_metal_in_background, name="metal-limit-probe", daemon=True)
            _metal_cache["thread"] = thread
            thread.start()
    thread.join(_METAL_WAIT_SECONDS)
    with _metal_lock:
        return _metal_cache.get("value")


def reset_metal_cache() -> None:
    """Forget the cached Metal read and the host class. Used by tests."""
    with _metal_lock:
        _metal_cache.clear()
    with _host_lock:
        _host_cache.clear()


def _host_memory_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / _GIB
    except Exception:
        return None


def _is_this_machine(total_gb: float) -> bool:
    host = _host_memory_gb()
    return host is not None and abs(host - total_gb) <= _HOST_MATCH_TOLERANCE_GB


def _apple_usable_gb(total_gb: float) -> float:
    fraction = 0.67 if total_gb <= 36 else 0.75
    heuristic = round(total_gb * fraction, 1)
    if not _on_macos() or not _is_this_machine(total_gb):
        return heuristic
    info = _metal_info()
    if info is not None and abs(info[0] - total_gb) <= _HOST_MATCH_TOLERANCE_GB:
        return round(info[1], 1)
    wired = _wired_limit_gb()
    return round(wired, 1) if wired else heuristic


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
    # An Intel Mac has neither MLX nor CUDA; a snapshot without an arch is a current Mac.
    if platform_name.startswith(("darwin", "mac")) and machine not in {"x86_64", "amd64", "i386"}:
        tier = _nearest(total, (8, 16, 24, 32, 36, 48, 64, 96, 128, 192, 256))
        return HardwareClass("apple", f"apple-{tier}", f"Apple Silicon, {total:.0f} GB", total, _apple_usable_gb(total))
    return HardwareClass("cpu", "cpu", f"CPU only, {total:.0f} GB RAM", total, round(total * 0.6, 1))


_host_lock = threading.Lock()
_host_cache: dict[str, Any] = {}


def host_system_dict() -> dict[str, Any]:
    """The few system-snapshot fields ``hardware_class`` reads, without building the full snapshot.

    The image and video catalog routes have no system snapshot (it lists processes and probes
    the runtimes); memory, platform and the GPU's VRAM are all the classification needs.
    """
    vram = None
    if not (platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")):
        try:
            from backend_service.helpers.gpu import get_device_vram_total_gb

            vram = get_device_vram_total_gb()
        except Exception:
            vram = None
    return {
        "platform": platform.system(),
        "arch": platform.machine(),
        "totalMemoryGb": round((_host_memory_gb() or 0.0), 2),
        "gpuVramTotalGb": vram,
    }


def host_hardware_class() -> HardwareClass:
    """``hardware_class`` of this machine, kept for a minute (the wired limit can be changed live)."""
    with _host_lock:
        stored = _host_cache.get("value")
        if stored is not None and time.monotonic() - stored[0] < _HOST_CLASS_TTL_SECONDS:
            return stored[1]
    value = hardware_class(host_system_dict())
    with _host_lock:
        _host_cache["value"] = (time.monotonic(), value)
    return value
