"""Device + Python interpreter detection for the image runtime.

Four small helpers used by the engine to pick where to run + diagnose
common Windows-CUDA install gotchas:

- ``_resolve_image_python`` — locate the python interpreter the
  embedded venv uses (or whichever is in PATH). Used by setup probes
  and the placeholder engine's missing-deps message.
- ``_guess_expected_device`` — predict torch's device WITHOUT importing
  torch. Importing torch here would lock torch/lib/*.dll on Windows
  and block ``/api/setup/install-gpu-bundle``; ``find_spec`` +
  ``nvidia-smi`` are free.
- ``_windows_cuda_unavailable_message`` — translate a missing
  ``torch.cuda`` (or ``torch.cuda.is_available()`` returning False) on
  a Windows host with an NVIDIA card into the actionable "Install CUDA
  torch from Settings > Setup" message.
- ``_is_cuda_torch_unavailable_error`` — predicate the load path uses
  to decide whether a runtime error came from the helper above.

Extracted from ``image_runtime.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import importlib.util
import os
import platform
from pathlib import Path
from typing import Any

from backend_service.helpers.gpu import nvidia_gpu_present as _nvidia_gpu_present


def _resolve_image_python(workspace_root: Path) -> str:
    override = os.getenv("CHAOSENGINE_MLX_PYTHON")
    if override:
        return override
    candidate = workspace_root / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return os.getenv("PYTHON", "python3")


def _guess_expected_device() -> str | None:
    """Best-effort prediction of what device diffusers will bind to on
    the next Generate click, computed WITHOUT importing torch.

    Importing torch here would lock torch/lib/*.dll in the backend
    process and block /api/setup/install-gpu-bundle on Windows (same
    trap we hit before). find_spec + nvidia_gpu_present are free.
    Returns ``None`` when torch isn't installed — caller surfaces
    the probe's ``missingDependencies`` list instead.

    Predicted device is provisional; the actual device used at
    generate time is what ``_detect_device`` decides once torch is
    imported. Mismatch is rare (driver missing, torch was CPU-only)
    and gets corrected in ``device`` once a model is loaded.
    """
    if importlib.util.find_spec("torch") is None:
        return None
    if _nvidia_gpu_present():
        return "cuda"
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        return "mps"
    return "cpu"


def _windows_cuda_unavailable_message(torch: Any) -> str | None:
    if platform.system() != "Windows" or not _nvidia_gpu_present():
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


def _is_cuda_torch_unavailable_error(exc: Exception) -> bool:
    return "CUDA torch is unavailable on this Windows NVIDIA host" in str(exc)
