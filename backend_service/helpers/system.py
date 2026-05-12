"""System monitoring: hardware snapshots, GPU, battery, memory, processes."""
from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

import psutil

from backend_service.helpers.formatting import _bytes_to_gb
from backend_service.helpers.system_hardware import (
    _apple_hardware_summary,
    _detect_gpu_utilization,
    _generic_hardware_summary,
    _get_battery_info,
    _get_compressed_memory_gb,
    _resolve_app_version,
    _runtime_label,
    _safe_run,
)
from backend_service.helpers.system_processes import (
    _describe_process,
    _get_top_memory_for_pid,
    _get_top_memory_map,
    _list_llm_processes,
    _parse_top_mem_value,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]




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

    # Discrete GPU VRAM (CUDA cards on Windows/Linux). Apple Silicon shares
    # unified memory with the CPU so this stays None there -- the chat /
    # video safety estimators already treat unified memory as a single pool.
    # The chat-side cache-fit warning needs this number because llama.cpp
    # places the KV cache on the GPU when ngl=999, so a 60 GB cache on a
    # 24 GB 4090 fails far worse than the system-RAM check would suggest.
    try:
        from backend_service.helpers.gpu import get_device_vram_total_gb
        gpu_vram_total_gb_raw = get_device_vram_total_gb()
    except Exception:
        gpu_vram_total_gb_raw = None
    if (
        platform.system() == "Darwin"
        and platform.machine() in ("arm64", "aarch64")
    ):
        # On Apple Silicon get_device_vram_total_gb returns the unified
        # memory total (== totalMemoryGb). Reporting it as a separate
        # "GPU VRAM" field would double-count and confuse the cache-fit
        # message ("60 GB > 24 GB VRAM" on a 64 GB Mac). Leave it None
        # so the consumer falls back to the unified totalMemoryGb.
        gpu_vram_total_gb: float | None = None
    else:
        gpu_vram_total_gb = gpu_vram_total_gb_raw


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
        "gpuVramTotalGb": gpu_vram_total_gb,
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
        # FU-042: structured i18n keys + payload for frontend translation.
        title_key = "recommendation.mlxTitle"
        detail_key = "recommendation.mlxDetail"
        payload = {
            "modelSize": model_size,
            "cacheLabel": cache_label,
            "hardware": system_stats["hardwareSummary"],
        }
    else:
        title = f"Recommended target: {model_size} GGUF"
        detail = (
            "Cross-platform mode will prefer llama.cpp GGUF for broad hardware support."
        )
        title_key = "recommendation.ggufTitle"
        detail_key = "recommendation.ggufDetail"
        payload = {"modelSize": model_size}

    return {
        "title": title,
        "detail": detail,
        "targetModel": model_size,
        "cacheLabel": cache_label,
        "headroomPercent": headroom_percent,
        "titleKey": title_key,
        "detailKey": detail_key,
        "payload": payload,
    }
