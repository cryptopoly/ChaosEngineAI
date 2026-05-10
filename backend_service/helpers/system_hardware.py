"""Hardware probe helpers — chip / OS summary, version, GPU + battery + memory.

Cross-platform fallbacks (``_safe_run``, ``_generic_hardware_summary``,
``_resolve_app_version``) sit alongside macOS-specific probes
(``_apple_hardware_summary``, ``_get_compressed_memory_gb``,
``_get_battery_info``) so the dashboard snapshot can render correctly
without an extra capabilities round-trip.

Extracted from ``backend_service/helpers/system.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.system`` so existing
imports keep working.
"""

from __future__ import annotations

import json
import platform
import subprocess
import tomllib
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _safe_run(command: list[str], timeout: float = 1.5) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _resolve_app_version() -> str:
    pyproject_path = WORKSPACE_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return "0.0.0"
    try:
        with pyproject_path.open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])
    except Exception:
        return "0.0.0"


def _apple_hardware_summary(total_memory_gb: float) -> str | None:
    if platform.system() != "Darwin":
        return None
    payload = _safe_run(["system_profiler", "SPHardwareDataType", "-json"], timeout=2.5)
    if not payload:
        return None
    try:
        hardware = json.loads(payload)["SPHardwareDataType"][0]
    except Exception:
        return None

    chip = hardware.get("chip_type") or hardware.get("cpu_type")
    model = hardware.get("machine_model") or hardware.get("machine_name")
    parts = [part for part in [chip, model] if part]
    if not parts:
        return None
    return f"{' / '.join(parts)} / {total_memory_gb:.0f} GB unified memory"


def _generic_hardware_summary(total_memory_gb: float) -> str:
    system_name = platform.system()
    machine = platform.machine()
    processor = platform.processor() or machine
    return f"{processor} / {system_name} / {total_memory_gb:.0f} GB memory"


def _runtime_label(capabilities: dict[str, Any] | None = None) -> str:
    from backend_service.inference import get_backend_capabilities
    native = capabilities or get_backend_capabilities().to_dict()
    on_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if on_apple_silicon and native.get("mlxUsable"):
        return "MLX + ChaosEngine"
    if native.get("ggufAvailable"):
        return "llama.cpp + GGUF sidecar"
    return "Python sidecar"


def _detect_gpu_utilization() -> float | None:
    return None


def _get_compressed_memory_gb() -> float:
    """Parse macOS vm_stat for compressed memory (no sudo)."""
    if platform.system() != "Darwin":
        return 0.0
    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2,
        )
        page_size = 16384  # Apple Silicon default
        pages_compressed = 0
        for line in result.stdout.split("\n"):
            if "page size of" in line:
                # "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
                try:
                    page_size = int(line.split("page size of")[1].split("bytes")[0].strip())
                except (ValueError, IndexError):
                    pass
            elif "Pages occupied by compressor" in line:
                try:
                    pages_compressed = int(line.split(":")[1].strip().rstrip("."))
                except (ValueError, IndexError):
                    pass
        return round((pages_compressed * page_size) / (1024 ** 3), 1)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return 0.0


def _get_battery_info() -> dict[str, Any] | None:
    """Parse pmset -g batt for battery state (no sudo). Returns None on desktops."""
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=2,
        )
        output = result.stdout
        # First line: "Now drawing from 'AC Power'" or "'Battery Power'"
        power_source = "AC"
        if "Battery Power" in output:
            power_source = "Battery"
        # Subsequent line: " -InternalBattery-0 ... 85%; discharging; ..."
        if "InternalBattery" not in output:
            return None  # No battery (desktop Mac)
        percent = 100
        charging = False
        for line in output.split("\n"):
            if "InternalBattery" in line:
                # Extract "85%"
                if "%" in line:
                    try:
                        parts = line.split("%")[0].split()
                        percent = int(parts[-1])
                    except (ValueError, IndexError):
                        pass
                if "charging" in line.lower() and "discharging" not in line.lower():
                    charging = True
                elif "charged" in line.lower():
                    charging = False
                break
        return {
            "percent": percent,
            "powerSource": power_source,
            "charging": charging,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
