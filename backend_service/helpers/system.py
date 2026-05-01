"""System monitoring: hardware snapshots, GPU, battery, memory, processes."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any

import psutil

from backend_service.helpers.formatting import _bytes_to_gb


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


def _get_disk_usage_for_models(settings: dict[str, Any]) -> dict[str, float] | None:
    """Return disk usage of the first enabled model directory."""
    dirs = settings.get("modelDirectories") or []
    for entry in dirs:
        if not entry.get("enabled", True):
            continue
        raw_path = str(entry.get("path") or "").strip()
        if not raw_path:
            continue
        try:
            expanded = Path(os.path.expanduser(raw_path))
            if not expanded.exists():
                continue
            usage = psutil.disk_usage(str(expanded))
            return {
                "totalGb": _bytes_to_gb(usage.total),
                "usedGb": _bytes_to_gb(usage.used),
                "freeGb": _bytes_to_gb(usage.free),
                "path": str(expanded),
            }
        except (OSError, PermissionError):
            continue
    # Fall back to home directory
    try:
        usage = psutil.disk_usage(str(Path.home()))
        return {
            "totalGb": _bytes_to_gb(usage.total),
            "usedGb": _bytes_to_gb(usage.used),
            "freeGb": _bytes_to_gb(usage.free),
            "path": str(Path.home()),
        }
    except OSError:
        return None


def _parse_top_mem_value(mem_str: str) -> float | None:
    normalized = mem_str.strip().rstrip("+-")
    if not normalized:
        return None
    try:
        if normalized.endswith("T"):
            return float(normalized[:-1]) * 1024
        if normalized.endswith("G"):
            return float(normalized[:-1])
        if normalized.endswith("M"):
            return float(normalized[:-1]) / 1024
        if normalized.endswith("K"):
            return float(normalized[:-1]) / (1024 * 1024)
        return float(normalized) / (1024 ** 3)
    except ValueError:
        return None


def _get_top_memory_map() -> dict[int, float]:
    """Use macOS `top` to get real memory (including GPU/compressed) per PID.

    psutil's RSS misses Metal GPU memory used by MLX models. macOS `top`
    reports the full footprint that matches Activity Monitor.
    """
    try:
        result = subprocess.run(
            ["top", "-l", "1", "-stats", "pid,mem", "-o", "mem", "-n", "120"],
            capture_output=True, text=True, timeout=10,
        )
        mem_map: dict[int, float] = {}
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            parsed = _parse_top_mem_value(parts[1])
            if parsed is None:
                continue
            mem_map[pid] = parsed
        return mem_map
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}


def _get_top_memory_for_pid(pid: int) -> float | None:
    """Query a single PID via macOS `top` for a more reliable live footprint."""
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["top", "-l", "1", "-stats", "pid,mem", "-pid", str(int(pid))],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            current_pid = int(parts[0])
        except ValueError:
            continue
        if current_pid != int(pid):
            continue
        return _parse_top_mem_value(parts[1])
    return None


def _describe_process(
    pid: int,
    *,
    kind_hint: str | None = None,
    owner_hint: str | None = None,
    top_mem: dict[int, float] | None = None,
) -> dict[str, Any] | None:
    """Describe a single process for dashboard display.

    ``kind_hint`` and ``owner_hint`` let callers surface runtime-managed workers
    even when the generic LLM process scan missed them.
    """
    try:
        process = psutil.Process(int(pid))
        name = (process.name() or "").lower()
        cmdline_parts = process.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, OSError, ValueError):
        return None

    cmdline = " ".join(cmdline_parts).lower()
    haystack = f"{name} {cmdline}"
    binary_path = (cmdline_parts[0] if cmdline_parts else "").lower()

    if any(
        marker in haystack
        for marker in ("chaosengine", "backend_service.mlx_worker", "chaosengine-embedded")
    ):
        owner = "ChaosEngineAI"
    else:
        owner = owner_hint or "System"

    kind = "other"
    if "mlx_worker" in cmdline or "backend_service.mlx_worker" in cmdline:
        kind = "mlx_worker"
    elif "llama-server" in name or "llama-server" in cmdline:
        kind = "llama_server"
    elif "backend_service.app" in cmdline:
        kind = "backend"
    elif kind_hint:
        kind = kind_hint

    try:
        rss_gb = _bytes_to_gb(process.memory_info().rss)
    except (psutil.Error, AttributeError, OSError):
        rss_gb = 0.0
    mem_map = top_mem if top_mem is not None else (_get_top_memory_map() if platform.system() == "Darwin" else {})
    top_mem_gb = mem_map.get(int(pid))
    if platform.system() == "Darwin" and (top_mem_gb is None or top_mem_gb <= 0):
        top_mem_gb = _get_top_memory_for_pid(int(pid))
    mem_gb = round(top_mem_gb if top_mem_gb is not None and top_mem_gb > 0 else rss_gb, 1)

    try:
        cpu_percent = round(float(process.cpu_percent() or 0.0), 1)
    except (psutil.Error, OSError):
        cpu_percent = 0.0

    return {
        "pid": int(pid),
        "name": name or "process",
        "owner": owner,
        "memoryGb": mem_gb,
        "cpuPercent": cpu_percent,
        "kind": kind,
    }


def _list_llm_processes(limit: int = 12) -> list[dict[str, Any]]:
    # Process-name keywords that indicate an LLM-related process.
    # Intentionally excludes the desktop app name itself, which is too broad
    # and can match the shell/UI process instead of the actual model worker.
    name_keywords = ("mlx", "llama-server", "llama-cli", "openclaw")
    # Match real model workers by their command line too so bundled workers
    # still show up even if their executable name is not literally "python".
    cmdline_markers = ("backend_service.mlx_worker", "mlx_worker", "llama-server", "llama-cli", "openclaw")
    # Get real memory from top (includes GPU/Metal memory on macOS)
    top_mem = _get_top_memory_map() if platform.system() == "Darwin" else {}

    matches: list[dict[str, Any]] = []
    try:
        for process in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "cpu_percent", "ppid"]):
            try:
                name = (process.info.get("name") or "").lower()
                cmdline_parts = process.info.get("cmdline") or []
                cmdline = " ".join(cmdline_parts).lower()
                haystack = f"{name} {cmdline}"

                # Check if this is an LLM process by name
                is_llm = any(keyword in name for keyword in name_keywords)
                if not is_llm:
                    is_llm = any(marker in cmdline for marker in cmdline_markers)
                pid = process.info["pid"]

                if not is_llm:
                    continue

                described = _describe_process(pid, top_mem=top_mem)
                if described is not None:
                    matches.append(described)
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, OSError):
                continue
    except (psutil.Error, PermissionError, OSError):
        return []

    matches.sort(key=lambda item: (item["memoryGb"], item["cpuPercent"]), reverse=True)
    return matches[:limit]


def _capabilities_payload(capabilities: Any | None = None) -> dict[str, Any]:
    if capabilities is None:
        from backend_service.inference import get_backend_capabilities
        return get_backend_capabilities().to_dict()
    to_dict = getattr(capabilities, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return dict(capabilities)


def _build_system_snapshot(
    app_version: str,
    app_started_at: float,
    *,
    capabilities: Any | None = None,
) -> dict[str, Any]:
    native = _capabilities_payload(capabilities)
    memory = psutil.virtual_memory()
    try:
        swap = psutil.swap_memory()
        swap_used = swap.used
        swap_total = swap.total
    except OSError:
        swap_used = 0
        swap_total = 0
    total_memory_gb = _bytes_to_gb(memory.total)
    available_memory_gb = _bytes_to_gb(memory.available)
    used_memory_gb = _bytes_to_gb(memory.used)
    swap_used_gb = _bytes_to_gb(swap_used)
    swap_total_gb = _bytes_to_gb(swap_total)
    spare_headroom_gb = round(max(0.0, available_memory_gb - 6.0), 1)
    hardware_summary = _apple_hardware_summary(total_memory_gb) or _generic_hardware_summary(total_memory_gb)

    compressed_memory_gb = _get_compressed_memory_gb()
    battery = _get_battery_info()
    # Memory pressure: used + compressed + swap as a fraction of total
    pressure_numerator = used_memory_gb + compressed_memory_gb + swap_used_gb
    memory_pressure_percent = (
        round(min(100.0, (pressure_numerator / total_memory_gb) * 100), 1)
        if total_memory_gb > 0 else 0.0
    )

    def _get_cache_strategies():
        from cache_compression import registry
        return registry.available()

    def _get_dflash_info():
        try:
            from dflash import availability_info
            return availability_info()
        except (ImportError, AttributeError):
            local_integration = WORKSPACE_ROOT / "dflash" / "__init__.py"
            if local_integration.exists():
                try:
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "_chaosengine_dflash_integration",
                        local_integration,
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return module.availability_info()
                except Exception:
                    pass
            return {"available": False, "mlxAvailable": False, "vllmAvailable": False, "ddtreeAvailable": False, "supportedModels": []}

    return {
        "platform": platform.system(),
        "arch": platform.machine(),
        "hardwareSummary": hardware_summary,
        "backendLabel": _runtime_label(native),
        "appVersion": app_version,
        "availableCacheStrategies": _get_cache_strategies(),
        "dflash": _get_dflash_info(),
        "vllmAvailable": native.get("vllmAvailable", False),
        "vllmVersion": native.get("vllmVersion"),
        "mlxAvailable": native["mlxAvailable"],
        "mlxLmAvailable": native["mlxLmAvailable"],
        "mlxUsable": native["mlxUsable"],
        "ggufAvailable": native["ggufAvailable"],
        "converterAvailable": native["converterAvailable"],
        "nativePython": native["pythonExecutable"],
        "llamaServerPath": native["llamaServerPath"],
        "llamaServerTurboPath": native.get("llamaServerTurboPath"),
        "llamaCliPath": native["llamaCliPath"],
        "nativeRuntimeMessage": native["mlxMessage"],
        "totalMemoryGb": total_memory_gb,
        "availableMemoryGb": available_memory_gb,
        "usedMemoryGb": used_memory_gb,
        "swapUsedGb": swap_used_gb,
        "swapTotalGb": swap_total_gb,
        "compressedMemoryGb": compressed_memory_gb,
        "memoryPressurePercent": memory_pressure_percent,
        "cpuUtilizationPercent": round(psutil.cpu_percent(interval=None), 1),
        "gpuUtilizationPercent": _detect_gpu_utilization(),
        "spareHeadroomGb": spare_headroom_gb,
        "battery": battery,
        "runningLlmProcesses": _list_llm_processes(),
        "uptimeMinutes": round((time.time() - app_started_at) / 60, 1),
    }


def _best_fit_recommendation(system_stats: dict[str, Any]) -> dict[str, Any]:
    memory_gb = system_stats["totalMemoryGb"]
    is_macos_mlx = (
        system_stats["platform"] == "Darwin"
        and system_stats["arch"] == "arm64"
        and bool(system_stats.get("mlxUsable", False))
    )

    if memory_gb >= 64:
        model_size = "70B"
        cache_label = "Native f16"
        headroom_percent = 68
    elif memory_gb >= 48:
        model_size = "70B"
        cache_label = "Native f16"
        headroom_percent = 65
    elif memory_gb >= 36:
        model_size = "32B"
        cache_label = "Native f16"
        headroom_percent = 54
    elif memory_gb >= 24:
        model_size = "14B"
        cache_label = "Native f16"
        headroom_percent = 49
    else:
        model_size = "7B"
        cache_label = "Native f16"
        headroom_percent = 43

    if is_macos_mlx:
        title = f"Recommended target: {model_size} class @ {cache_label}"
        detail = (
            f"This forecast is relative to a recommended {model_size} class local target on "
            f"{system_stats['hardwareSummary']}, not a currently selected chat model."
        )
    else:
        title = f"Recommended target: {model_size} GGUF"
        detail = (
            "Cross-platform mode will prefer llama.cpp GGUF for broad hardware support."
        )

    return {
        "title": title,
        "detail": detail,
        "targetModel": model_size,
        "cacheLabel": cache_label,
        "headroomPercent": headroom_percent,
    }
