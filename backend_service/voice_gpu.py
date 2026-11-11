"""Whether the NVIDIA GPU in this computer can run a speech row: its memory and its compute capability, read from
the driver, never assumed (8 GB cards up to an RTX 5090 and unified-memory boards such as the DGX Spark).

A row opts in with ``minVramGb`` (measured peak plus a margin) and ``minComputeCap``. Facts that cannot be read
(no ``nvidia-smi``, a board that reports no memory total) never block a row: the worker then fails or succeeds on
its own and says why.
"""

from __future__ import annotations

import subprocess
from typing import Any

_CAP: dict[str, float | None] = {}


def compute_capability() -> float | None:
    """The first GPU's compute capability (8.9 for an RTX 4090), cached; None when it cannot be read."""
    if "value" not in _CAP:
        value: float | None = None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if out.returncode == 0:
                value = float(out.stdout.strip().splitlines()[0].strip())
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            value = None
        _CAP["value"] = value
    return _CAP["value"]


def vram_total_gb() -> float | None:
    from backend_service.helpers.gpu import get_device_vram_total_gb  # noqa: PLC0415

    return get_device_vram_total_gb()


def reset() -> None:
    """Forget what was read (tests)."""
    _CAP.clear()


def problem(row: dict[str, Any]) -> str | None:
    """Plain words for why this GPU cannot run ``row``, or None. Only rows that declare a need are checked."""
    need_cap = row.get("minComputeCap")
    if need_cap:
        cap = compute_capability()
        if cap is not None and cap < float(need_cap):
            return (f"{row['name']} needs an NVIDIA GPU of compute capability {float(need_cap):g} or newer; "
                    f"this one is {cap:g}.")
    need_vram = row.get("minVramGb")
    if need_vram:
        total = vram_total_gb()
        if total is not None and total < float(need_vram):
            return (f"{row['name']} needs about {float(need_vram):g} GB of GPU memory; "
                    f"this GPU has {total:.0f} GB.")
    return None
