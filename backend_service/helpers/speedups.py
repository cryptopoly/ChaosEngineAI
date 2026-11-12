"""Opt-in diffusion speed-ups applied once after a pipeline is placed (W2 of the RTX 4090 pass).

``CHAOSENGINE_SPEEDUPS`` is a comma list for measuring; each entry is independent:

- ``vae_tiling`` / ``vae_slicing``: diffusers' memory-saving decode modes (slower, smaller peak).
- ``compile``: regional ``torch.compile`` of the repeated transformer blocks (needs Triton).
- ``time_decode``: log how long ``vae.decode`` takes per call, to see its share of a run.

Nothing here runs unless the variable is set, and every step reports a note instead of raising.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

_LOG = logging.getLogger(__name__)


def requested_speedups() -> frozenset[str]:
    raw = os.environ.get("CHAOSENGINE_SPEEDUPS", "")
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def compile_blocker() -> str | None:
    """Why ``torch.compile`` cannot work on this machine, or None.

    Inductor's Triton kernels need Triton and a C compiler for their launcher stubs. A Windows PC
    without the Visual Studio C++ tools fails on the first step with "Failed to find C compiler".
    """
    import importlib.util
    import shutil
    import sys

    if importlib.util.find_spec("triton") is None:
        return "Triton is not installed"
    if sys.platform == "win32" and not (os.environ.get("CC") or shutil.which("cl")):
        return "no C compiler: install the Visual Studio C++ build tools or set CC"
    if sys.platform.startswith("linux"):
        if not (os.environ.get("CC") or any(shutil.which(name) for name in ("gcc", "cc", "clang"))):
            return "no C compiler: install one (Ubuntu or Debian: sudo apt install build-essential) or set CC"
        # Triton builds its CUDA launcher against Python.h, which a python3-dev-less system lacks.
        import sysconfig
        from pathlib import Path

        include_dirs = [sysconfig.get_paths().get("include") or ""]
        include_dirs += [part for part in (os.environ.get("C_INCLUDE_PATH") or "").split(os.pathsep) if part]
        if not any(directory and (Path(directory) / "Python.h").is_file() for directory in include_dirs):
            return "the Python headers are missing (Ubuntu or Debian: sudo apt install python3-dev)"
    return None


def apply_speedups(pipeline: Any) -> list[str]:
    """Apply the requested speed-ups to ``pipeline``; returns notes for the runtime note."""
    wanted = requested_speedups()
    if not wanted:
        return []
    notes: list[str] = []
    vae = getattr(pipeline, "vae", None)
    if "vae_tiling" in wanted and vae is not None and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
        notes.append("VAE tiling on")
    if "vae_slicing" in wanted and vae is not None and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
        notes.append("VAE slicing on")
    if "time_decode" in wanted and vae is not None and hasattr(vae, "decode"):
        original = vae.decode

        def timed(*args: Any, **kwargs: Any) -> Any:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = original(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _LOG.warning("speedups.decode_seconds=%.2f", time.perf_counter() - started)
            return result

        vae.decode = timed
    if "compile" in wanted:
        transformer = getattr(pipeline, "transformer", None)
        missing = compile_blocker()
        if missing:
            notes.append(f"torch.compile not applied ({missing})")
        elif transformer is not None and hasattr(transformer, "compile_repeated_blocks"):
            try:
                transformer.compile_repeated_blocks(fullgraph=True)
                notes.append("torch.compile on the repeated blocks")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"torch.compile not applied ({type(exc).__name__}: {exc})")
        else:
            notes.append("torch.compile not applied (transformer has no compile_repeated_blocks)")
    return notes
