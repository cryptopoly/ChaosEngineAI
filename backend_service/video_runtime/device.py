"""Device + Python-runtime probe helpers for the video runtime.

Mirrors ``backend_service/image_runtime/device.py`` so the two runtimes
share the same probe shape on every supported platform.

Extracted from ``video_runtime/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import secrets
from pathlib import Path
from typing import Any

from backend_service.helpers.gpu import nvidia_gpu_present


MAX_VIDEO_SEED = 2147483647


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _resolve_video_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return secrets.randbelow(MAX_VIDEO_SEED + 1)


def _resolve_video_python() -> str:
    override = os.getenv("CHAOSENGINE_MLX_PYTHON") or os.getenv("CHAOSENGINE_VIDEO_PYTHON")
    if override:
        return override
    candidate = WORKSPACE_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return os.getenv("PYTHON", "python3")


def _detect_device_memory_gb(device: str | None) -> float | None:
    """Best-effort read of how much memory the inference device has access to.

    - ``cuda``: dedicated VRAM from ``nvidia-smi`` (via ``get_gpu_metrics``).
    - ``mps`` / ``cpu`` on macOS: unified memory from ``sysctl hw.memsize``.
    - ``cpu`` on Linux/Windows: system RAM via psutil.

    Returns ``None`` when detection fails — the frontend safety heuristic
    treats ``None`` as "stay conservative" and falls back to its 16 GB-safe
    thresholds rather than risk over-scaling on an unknown device.

    Uses the cached fast path in ``helpers.gpu`` because total VRAM never
    changes for the life of a process. The first call shells out to
    ``nvidia-smi``/``sysctl``; every subsequent call is a dict lookup, which
    keeps the ``/api/video/runtime`` probe well inside the frontend's
    15s fetch budget on Windows.
    """
    try:
        from backend_service.helpers.gpu import get_device_vram_total_gb
    except Exception:
        return None
    try:
        return get_device_vram_total_gb()
    except Exception:
        return None


def _guess_video_expected_device() -> str | None:
    """Predict the device torch will bind to without importing torch.

    Importing torch in probe() would lock torch/lib/*.dll and block the
    GPU-bundle installer on Windows (same trap the image runtime hit).
    ``find_spec`` + ``nvidia_gpu_present`` are free of that side effect
    and accurate enough for the UI badge.
    """
    if importlib.util.find_spec("torch") is None:
        return None
    if nvidia_gpu_present():
        return "cuda"
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "mps"
    return "cpu"


def _windows_cuda_unavailable_message(torch: Any) -> str | None:
    if platform.system() != "Windows" or not nvidia_gpu_present():
        return None
    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is None:
        return (
            "CUDA torch is unavailable on this Windows NVIDIA host: torch imports "
            "but has no torch.cuda module. Open Settings > Setup and click "
            "Install CUDA torch, then Restart Backend."
        )
    try:
        cuda_available = bool(getattr(cuda_module, "is_available", lambda: False)())
    except Exception as exc:
        return (
            "CUDA torch is unavailable on this Windows NVIDIA host: "
            f"torch.cuda.is_available failed ({type(exc).__name__}: {exc}). "
            "Open Settings > Setup and click Install CUDA torch, then Restart Backend."
        )
    if not cuda_available:
        return (
            "CUDA torch is unavailable on this Windows NVIDIA host. Open Settings > "
            "Setup and click Install CUDA torch, then Restart Backend."
        )
    return None
