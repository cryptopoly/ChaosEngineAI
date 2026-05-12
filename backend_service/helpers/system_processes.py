"""System process inspection — Activity Monitor-style memory + LLM scan.

Two-step approach for memory accounting:

1. ``_get_top_memory_map`` shells out to macOS ``top`` for an
   Activity-Monitor-accurate per-PID footprint. ``psutil``'s ``rss``
   misses Metal GPU memory used by MLX models — ``top`` reports the
   full footprint that matches the user's system monitor.
2. ``_describe_process`` uses ``psutil`` for naming / cmdline / CPU and
   falls back to the rss reading when ``top`` doesn't return a value
   (Linux / Windows).

``_list_llm_processes`` walks ``psutil.process_iter`` filtering by
LLM-related keywords (``mlx_worker``, ``llama-server``, ``llama-cli``,
``openclaw``) and returns up to ``limit`` rows ranked by memory + CPU.

Extracted from ``backend_service/helpers/system.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.system`` so existing
``from backend_service.helpers.system import _describe_process``
imports keep working.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Any

import psutil

from backend_service.helpers.formatting import _bytes_to_gb


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
