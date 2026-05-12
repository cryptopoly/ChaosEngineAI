"""Workspace + server-status payload renderers for ``ChaosEngineState``.

Two payload builders lifted out of ``state/__init__.py``:

* ``server_status`` — `/api/server/status` shape. Reports whether the
  OpenAI-compatible API is bindable, the loaded model name, recent
  log lines, plus the loading-stage breakdown the UI uses to render
  the model-load progress bar.
* ``workspace`` — `/api/workspace` aggregate. Composes
  ``system_snapshot`` + library scan + recommendation + featured
  models + runtime status + benchmark history + log/activity tails
  + cache-preview math into the single payload the dashboard reads.
  The heavy lifting is the per-process annotation pass that joins
  ``runningLlmProcesses`` against the runtime's active + warm
  engines so the UI can show which loaded/warm model each PID
  belongs to.

Both take the ``ChaosEngineState`` instance as their first argument.

Extracted as part of the v0.8.0 Phase 1a-7 refactor.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from backend_service.helpers.discovery import _model_family_payloads
from backend_service.helpers.network import _local_ipv4_addresses
from backend_service.helpers.system import (
    _best_fit_recommendation,
    _describe_process,
    _get_disk_usage_for_models,
)


if TYPE_CHECKING:
    from backend_service.state import ChaosEngineState


def server_status(state: ChaosEngineState) -> dict[str, Any]:
    from backend_service.app import DEFAULT_HOST

    runtime = state.runtime.status(
        active_requests=state.active_requests,
        requests_served=state.requests_served,
    )
    loaded = runtime["loadedModel"]
    recent_orphaned_workers = runtime.get("recentOrphanedWorkers") or []
    recent_server_logs = [
        entry["message"]
        for entry in list(state.logs)
        if entry["source"] in {"runtime", "chat", "server"}
    ][:3]
    status = "running" if runtime["serverReady"] else "idle"
    remote_enabled = DEFAULT_HOST != "127.0.0.1"
    localhost_url = f"http://127.0.0.1:{state.server_port}/v1"
    lan_urls = (
        [f"http://{address}:{state.server_port}/v1" for address in _local_ipv4_addresses()]
        if remote_enabled
        else []
    )
    base_url = localhost_url
    preferred_port = state.settings["preferredServerPort"]
    port_note = (
        f"Preferred API port is {preferred_port}. Restart the API service to apply it."
        if preferred_port != state.server_port
        else (
            "Remote access is enabled for local-network clients. Allow incoming connections in your firewall if prompted."
            if remote_enabled
            else "Third-party tools on this machine can target the displayed localhost URL."
        )
    )
    loading = None
    if state._loading_state is not None:
        elapsed = time.time() - state._loading_state["startedAt"]
        loading = {
            "modelName": state._loading_state["modelName"],
            "stage": state._loading_state["stage"],
            "elapsedSeconds": round(elapsed, 1),
            "progress": state._loading_state.get("progress"),
            "progressPercent": state._loading_state.get("progressPercent"),
            "progressPhase": state._loading_state.get("progressPhase"),
            "progressMessage": state._loading_state.get("progressMessage"),
            "recentLogLines": list(state._loading_state.get("recentLogLines") or []),
        }

    return {
        "status": status,
        "baseUrl": base_url,
        "localhostUrl": localhost_url,
        "lanUrls": lan_urls,
        "bindHost": DEFAULT_HOST,
        "remoteAccessActive": remote_enabled,
        "port": state.server_port,
        "activeConnections": runtime["activeRequests"],
        "concurrentRequests": runtime["activeRequests"],
        "requestsServed": runtime["requestsServed"],
        "loadedModelName": loaded["name"] if loaded else None,
        "loading": loading,
        "recentOrphanedWorkers": recent_orphaned_workers,
        "logTail": recent_server_logs or [
            "Load a model to make the OpenAI-compatible local API ready for external tools.",
            "Ports and concurrency are configurable in Settings.",
            port_note,
        ],
    }


def workspace(state: ChaosEngineState) -> dict[str, Any]:
    from backend_service.app import compute_cache_preview

    system_stats = state._system_snapshot()
    try:
        loaded_name = state.runtime.loaded_model.name if state.runtime.loaded_model else None
        loaded_engine = state.runtime.engine.engine_name if state.runtime.engine else None
        warm_entries = [
            (engine.engine_name, info.name)
            for engine, info in state.runtime._warm_pool.values()
        ]
        procs = list(system_stats.get("runningLlmProcesses") or [])
        seen_pids = {
            int(pid)
            for pid in (proc.get("pid") for proc in procs)
            if isinstance(pid, int)
        }
        proc_by_pid = {
            int(proc["pid"]): proc
            for proc in procs
            if isinstance(proc.get("pid"), int)
        }

        tracked_runtime_processes: list[tuple[int, str]] = []
        active_pid_getter = (
            getattr(state.runtime.engine, "process_pid", None)
            if state.runtime.engine
            else None
        )
        active_pid = active_pid_getter() if callable(active_pid_getter) else None
        if isinstance(active_pid, int):
            active_kind = "other"
            if loaded_engine == "mlx":
                active_kind = "mlx_worker"
            elif loaded_engine == "llama.cpp":
                active_kind = "llama_server"
            tracked_runtime_processes.append((active_pid, active_kind))

        for engine, _info in state.runtime._warm_pool.values():
            pid_getter = getattr(engine, "process_pid", None)
            warm_pid = pid_getter() if callable(pid_getter) else None
            if not isinstance(warm_pid, int) or warm_pid in seen_pids:
                continue
            warm_kind = "other"
            if engine.engine_name == "mlx":
                warm_kind = "mlx_worker"
            elif engine.engine_name == "llama.cpp":
                warm_kind = "llama_server"
            tracked_runtime_processes.append((warm_pid, warm_kind))

        for pid, kind in tracked_runtime_processes:
            described = _describe_process(pid, kind_hint=kind, owner_hint="ChaosEngineAI")
            if described is None:
                continue
            existing = proc_by_pid.get(pid)
            if existing is not None:
                model_name = existing.get("modelName")
                model_status = existing.get("modelStatus")
                existing.clear()
                existing.update(described)
                if model_name:
                    existing["modelName"] = model_name
                if model_status:
                    existing["modelStatus"] = model_status
                continue
            procs.append(described)
            seen_pids.add(pid)
            proc_by_pid[pid] = described

        mlx_workers = [p for p in procs if p.get("kind") == "mlx_worker"]
        llama_servers = [p for p in procs if p.get("kind") == "llama_server"]

        assigned_loaded = False
        if loaded_name and loaded_engine == "mlx" and mlx_workers:
            mlx_workers[0]["modelName"] = loaded_name
            mlx_workers[0]["modelStatus"] = "active"
            assigned_loaded = True
        elif loaded_name and loaded_engine == "llama.cpp" and llama_servers:
            llama_servers[0]["modelName"] = loaded_name
            llama_servers[0]["modelStatus"] = "active"
            assigned_loaded = True

        if loaded_name and not assigned_loaded:
            for proc in procs:
                if proc.get("owner") == "ChaosEngineAI" and not proc.get("modelName"):
                    proc["modelName"] = loaded_name
                    proc["modelStatus"] = "active"
                    break

        warm_mlx = [
            name for engine, name in warm_entries
            if engine == "mlx" and name != loaded_name
        ]
        warm_llama = [
            name for engine, name in warm_entries
            if engine == "llama.cpp" and name != loaded_name
        ]
        for proc in mlx_workers[1:]:
            if warm_mlx and not proc.get("modelName"):
                proc["modelName"] = warm_mlx.pop(0)
                proc["modelStatus"] = "warm"
        for proc in llama_servers[1:]:
            if warm_llama and not proc.get("modelName"):
                proc["modelName"] = warm_llama.pop(0)
                proc["modelStatus"] = "warm"

        def _proc_rank(proc: dict[str, Any]) -> tuple[int, float, float]:
            status = proc.get("modelStatus")
            if status == "active":
                priority = 0
            elif status == "warm":
                priority = 1
            else:
                priority = 2
            return (
                priority,
                -float(proc.get("memoryGb", 0.0)),
                -float(proc.get("cpuPercent", 0.0)),
            )

        procs.sort(key=_proc_rank)
        system_stats["runningLlmProcesses"] = procs[:12]
    except Exception:
        pass

    try:
        disk_info = _get_disk_usage_for_models(state.settings)
        if disk_info:
            system_stats["diskFreeGb"] = disk_info["freeGb"]
            system_stats["diskTotalGb"] = disk_info["totalGb"]
            system_stats["diskUsedGb"] = disk_info["usedGb"]
            system_stats["diskPath"] = disk_info.get("path")
    except Exception:
        pass
    library = state._library()
    recommendation = _best_fit_recommendation(system_stats)
    launch_preferences = state._launch_preferences()
    return {
        "system": system_stats,
        "recommendation": recommendation,
        "featuredModels": _model_family_payloads(system_stats, library),
        "library": library,
        "libraryStatus": "ready" if state._library_scan_done.is_set() else "scanning",
        "settings": state._settings_payload(library),
        "chatSessions": state.chat_sessions,
        "runtime": state.runtime.status(
            active_requests=state.active_requests,
            requests_served=state.requests_served,
        ),
        "server": state.server_status(),
        "benchmarks": state.benchmark_runs,
        "logs": [entry for entry in state.logs if entry.get("level") != "debug"],
        "activity": list(state.activity),
        "preview": compute_cache_preview(
            bits=launch_preferences["cacheBits"],
            fp16_layers=launch_preferences["fp16Layers"],
            context_tokens=launch_preferences["contextTokens"],
            system_stats=system_stats,
        ),
        "quickActions": [
            "Online Models",
            "New Thread",
            "Start Server",
            "Convert to MLX",
            "Run Benchmark",
            "Open Logs",
        ],
    }
