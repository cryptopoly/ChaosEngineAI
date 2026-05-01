"""Thermal-pressure read helpers for the runaway-watchdog stack.

Phase 2.0.5-I: surface OS-level thermal warnings so the chat stream loop
can pause / warn when the host is throttling. On macOS we shell out to
`pmset -g therm` (works without sudo, returns a thermal warning level
string when one is recorded). Linux and Windows return None today —
both expose thermal data via vendor-specific paths that can be wired in
later when there's a per-OS UX story (NVML on NVIDIA, ACPI on Intel /
AMD, etc.).

The function is best-effort. Any subprocess error or unparseable output
returns None so the caller can decide how to handle missing data
(usually: continue uninterrupted).
"""

from __future__ import annotations

import platform
import subprocess
from typing import Literal


ThermalState = Literal["nominal", "moderate", "critical"]


def read_thermal_state() -> ThermalState | None:
    """Return the current thermal state, or None when unknown.

    macOS: parses `pmset -g therm`. The command emits one or more lines
    in the form `<Name> = <value>`; specifically `CPU_Scheduler_Limit`
    and `CPU_Available_CPUs` reflect throttling. We classify based on
    the warning levels reported in the same output:
    - "Thermal warning level set to 0" → nominal
    - 1-2 → moderate
    - 3+ → critical

    Other platforms: returns None (cross-platform thermal probes are
    intentionally out of scope for Phase 2.0.5-I; revisit when we wire
    the substrate-telemetry strip in Phase 3.5).
    """
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return _classify_pmset_output(result.stdout)


def _classify_pmset_output(output: str) -> ThermalState | None:
    """Pure helper for tests — classifies a pmset stdout string.

    `pmset -g therm` reports the highest-severity thermal warning the
    kernel has recorded since boot, plus CPU scheduler / available-CPU
    limits when active throttling is in effect. We map the reported
    warning level to our three-state space.
    """
    if not output:
        return None
    lower = output.lower()
    # Explicit "no thermal warning level" — the host is fine.
    if "no thermal warning level has been recorded" in lower:
        return "nominal"
    # "Thermal warning level set to N" lines.
    for line in lower.splitlines():
        if "thermal warning level set to" in line:
            tail = line.rsplit("set to", 1)[-1].strip().rstrip(".")
            try:
                level = int(tail.split()[0])
            except (ValueError, IndexError):
                continue
            if level <= 0:
                return "nominal"
            if level <= 2:
                return "moderate"
            return "critical"
    # CPU_Scheduler_Limit lower than 100 means active throttling — call
    # that "moderate" so the watchdog at least surfaces a hint.
    for line in lower.splitlines():
        if "cpu_scheduler_limit" in line:
            tail = line.split("=", 1)[-1].strip().rstrip(".")
            try:
                limit = int(tail.split()[0])
            except (ValueError, IndexError):
                continue
            if limit < 100:
                return "moderate"
            return "nominal"
    return None
