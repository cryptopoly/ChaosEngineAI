"""Diagnostics endpoint for in-app troubleshooting.

Surfaces a structured snapshot of the host OS, hardware, bundled runtime,
GPU state, and backend log tail — everything we'd otherwise ask users to
fetch via PowerShell when diagnosing a failed install or a restart loop.
The frontend (Settings → Diagnostics) renders this and offers a one-click
"Copy to clipboard" button so users can paste the payload into a support
thread without piecing it together by hand.

Design notes:
- Purely READ-ONLY for the snapshot. ``/reextract-runtime`` is the one
  action endpoint and it only deletes the ephemeral %TEMP% extraction;
  the persistent ``extras/`` tree is not touched.
- Redacts API tokens / HF tokens so users don't accidentally leak them.
- Caps log tail + attempts list size so the snapshot stays paste-friendly.
- Does NOT import torch — uses ``find_spec`` + a subprocess for version
  info, matching the rest of the backend's no-eager-import rule.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


# Cap log tail at a sensible ceiling so the snapshot payload stays
# copy-paste-able. 500 lines is ~40 KB of text — plenty for diagnosis,
# still fits in a single clipboard paste without choking the receiving
# chat UI.
_LOG_TAIL_MAX_LINES = 500
_LOG_TAIL_DEFAULT_LINES = 200

# Environment variables we redact before returning. The diagnostics
# payload is designed to be shared with support; anything here could
# reveal an auth secret, billing identity, or hijack-able session.
_REDACTED_ENV_SUBSTRINGS = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
)

# Environment variables we explicitly surface (empty or not) so users
# can see what the backend inherited. Anything else matching
# ``CHAOSENGINE_*`` or ``PYTHON*`` is also included.
_PINNED_ENV_VARS = (
    "CHAOSENGINE_HOST",
    "CHAOSENGINE_PORT",
    "CHAOSENGINE_EMBEDDED_RUNTIME",
    "CHAOSENGINE_EMBED_PYTHON_BIN",
    "CHAOSENGINE_MLX_PYTHON",
    "CHAOSENGINE_LLAMA_SERVER",
    "CHAOSENGINE_LLAMA_SERVER_TURBO",
    "CHAOSENGINE_LLAMA_CLI",
    "CHAOSENGINE_LLAMA_BIN_DIR",
    "CHAOSENGINE_EXTRAS_SITE_PACKAGES",
    "CHAOSENGINE_BACKEND_ROOT",
    "CHAOSENGINE_VENDOR_PATH",
    "CHAOSENGINE_TORCH_INDEX_URL",
    "CHAOSENGINE_DEBUG_EMBEDDED",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PATH",
    "VIRTUAL_ENV",
    "CONDA_DEFAULT_ENV",
)


@router.get("/api/diagnostics/snapshot")
def diagnostics_snapshot(request: Request) -> dict[str, Any]:
    """Comprehensive environment snapshot for troubleshooting.

    Structured so each section can be collapsed in the UI and copied
    verbatim into a support message. No side effects.

    Defensive contract: every section builder is wrapped so a single
    section's failure can't blow up the whole endpoint. If a builder
    raises, we fall back to ``{"error": "<type>: <msg>"}`` for that
    section and move on. The Diagnostics panel exists specifically to
    diagnose broken installs — it would be deeply ironic (and was, per
    user report) for the panel to fetch-fail when something's off.
    """
    state = request.app.state.chaosengine
    return {
        "generatedAt": time.time(),
        "app": _guarded("app", lambda: _app_info(state)),
        "os": _guarded("os", _os_info),
        "hardware": _guarded("hardware", _hardware_info),
        "python": _guarded("python", _python_info),
        "runtime": _guarded("runtime", lambda: _runtime_info(state)),
        "gpu": _guarded("gpu", _gpu_info),
        "extras": _guarded("extras", _extras_info),
        "environment": _guarded("environment", _env_vars),
        "logs": _guarded("logs", _log_info),
    }


def _guarded(section: str, fn):
    """Run ``fn()``; on exception return an error sentinel instead of
    propagating. Keeps the snapshot endpoint shipping a response even
    when a single section fails (psutil on a weird Windows mountpoint,
    subprocess hang on a stuck nvidia-smi, etc.).
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — this IS the error-barrier
        return {
            "error": f"{type(exc).__name__}: {str(exc)[:400]}",
            "section": section,
        }


@router.get("/api/diagnostics/log-tail")
def diagnostics_log_tail(lines: int = _LOG_TAIL_DEFAULT_LINES) -> dict[str, Any]:
    """Return the last N lines of the current backend log file.

    Useful for the Diagnostics panel's live tail view. Clamped to
    ``_LOG_TAIL_MAX_LINES`` so the frontend's copy-paste flow stays
    fast and the response body stays small.
    """
    lines = max(1, min(int(lines or _LOG_TAIL_DEFAULT_LINES), _LOG_TAIL_MAX_LINES))
    path = _active_log_path()
    if path is None:
        return {"path": None, "lines": [], "lineCount": 0}
    tail = _read_log_tail(path, lines)
    return {
        "path": str(path),
        "lines": tail,
        "lineCount": len(tail),
    }


@router.post("/api/diagnostics/reextract-runtime")
def reextract_runtime(request: Request) -> dict[str, Any]:
    """Delete the ephemeral embedded-runtime extraction cache.

    On the NEXT backend launch, the Tauri shell re-extracts the runtime
    tarball from scratch. Handy when the extraction tree is corrupted
    (e.g. a mid-install crash). Does not touch the persistent
    ``extras/`` tree or any user data — just nukes the staging dir that
    lives under ``%TEMP%``.

    Returns the deleted path and whether the delete succeeded. Callers
    typically follow up with ``restart_backend_sidecar`` (the Tauri
    command) so the next respawn triggers a fresh extract.
    """
    state = request.app.state.chaosengine
    target = _runtime_extraction_root()
    deleted = False
    error: str | None = None
    if target is None:
        error = "Could not resolve the runtime extraction root on this platform."
    elif not target.exists():
        # Nothing to delete is a soft-success — next bootstrap extracts anyway.
        deleted = False
    else:
        try:
            shutil.rmtree(target)
            deleted = True
        except Exception as exc:  # noqa: BLE001 — surface any OS error verbatim
            error = f"{type(exc).__name__}: {exc}"
    state.add_log(
        "server",
        "info" if error is None else "error",
        f"Diagnostics re-extract requested: path={target} deleted={deleted} error={error}",
    )
    return {
        "path": str(target) if target else None,
        "deleted": deleted,
        "error": error,
    }


# ------------------------------------------------------------------
# Section builders
# ------------------------------------------------------------------


def _app_info(state: Any) -> dict[str, Any]:
    from backend_service.app import WORKSPACE_ROOT, app_version

    return {
        "appVersion": app_version,
        "workspaceRoot": str(WORKSPACE_ROOT),
        "logCount": len(getattr(state, "logs", [])),
        "activeRequests": int(getattr(state, "active_requests", 0)),
        "requestsServed": int(getattr(state, "requests_served", 0)),
    }


def _os_info() -> dict[str, Any]:
    # ``platform.version()`` is more useful than ``release()`` on Windows
    # (includes the build number), but on Linux it's the kernel's
    # ``uname -v`` which contains a timestamp and changes on every
    # release. Including both lets us match on whichever the user's
    # platform makes more stable.
    try:
        uname = platform.uname()
        uname_info = {
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
        }
    except Exception:  # noqa: BLE001
        uname_info = {}
    info: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "uname": uname_info,
    }
    if sys.platform == "win32":
        info["windowsEdition"] = _safe(lambda: platform.win32_edition())
        info["windowsVersion"] = _safe(lambda: platform.win32_ver())
    elif sys.platform == "darwin":
        info["macVersion"] = _safe(lambda: platform.mac_ver())
    else:
        info["libcVersion"] = _safe(lambda: platform.libc_ver())
    return info


def _hardware_info() -> dict[str, Any]:
    try:
        import psutil
    except ImportError:
        return {"error": "psutil not available"}

    mem = _safe(lambda: psutil.virtual_memory())
    swap = _safe(lambda: psutil.swap_memory())
    cpu_freq = _safe(lambda: psutil.cpu_freq())
    return {
        "cpu": {
            "logicalCount": _safe(lambda: psutil.cpu_count(logical=True)),
            "physicalCount": _safe(lambda: psutil.cpu_count(logical=False)),
            "frequencyMhz": {
                "current": cpu_freq.current if cpu_freq else None,
                "max": cpu_freq.max if cpu_freq else None,
            } if cpu_freq else None,
            "utilizationPercent": _safe(lambda: psutil.cpu_percent(interval=0.1)),
        },
        "memory": {
            "totalGb": _bytes_to_gb(mem.total) if mem else None,
            "availableGb": _bytes_to_gb(mem.available) if mem else None,
            "usedGb": _bytes_to_gb(mem.used) if mem else None,
            "percent": mem.percent if mem else None,
        },
        "swap": {
            "totalGb": _bytes_to_gb(swap.total) if swap else None,
            "usedGb": _bytes_to_gb(swap.used) if swap else None,
        },
        "disks": _disk_usage(),
        "gpu": _gpu_hardware(),
    }


def _gpu_hardware() -> dict[str, Any]:
    """GPU metrics scraped from nvidia-smi / sysctl without importing torch.

    ``get_gpu_metrics`` already has the cross-platform adapter we need —
    reuse it instead of re-implementing nvidia-smi parsing here.
    """
    from backend_service.helpers.gpu import get_gpu_metrics, nvidia_gpu_present

    metrics = _safe(lambda: get_gpu_metrics()) or {}
    return {
        "nvidiaSmiOnPath": nvidia_gpu_present(),
        "gpuName": metrics.get("gpu_name"),
        "vramTotalGb": metrics.get("vram_total_gb"),
        "vramUsedGb": metrics.get("vram_used_gb"),
        "utilizationPercent": metrics.get("utilization_pct"),
        "temperatureC": metrics.get("temperature_c"),
        "powerW": metrics.get("power_w"),
        "driverVersion": _nvidia_driver_version(),
        "systemCudaVersion": _nvidia_cuda_version(),
    }


def _python_info() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version.splitlines()[0] if sys.version else None,
        "versionTuple": list(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "prefix": sys.prefix,
        "basePrefix": sys.base_prefix,
        "platform": sys.platform,
        # ``sys.path`` can be long; truncate at a sensible ceiling so the
        # payload stays readable. Any entry longer than the cap is kept
        # so we don't silently drop the important ones.
        "sysPath": list(sys.path)[:40],
        "cwd": _safe(lambda: os.getcwd()),
    }


def _runtime_info(state: Any) -> dict[str, Any]:
    runtime = getattr(state, "runtime", None)
    engine = getattr(runtime, "engine", None) if runtime is not None else None
    loaded_model = getattr(runtime, "loaded_model", None) if runtime is not None else None
    return {
        "engineName": getattr(engine, "engine_name", None) if engine else None,
        "engineLabel": getattr(engine, "engine_label", None) if engine else None,
        "loadedModel": loaded_model.to_dict() if loaded_model is not None else None,
        "warmPoolCount": _safe(
            lambda: len(getattr(runtime, "_warm_pool", {}) or {})
        ),
        "llamaServerPath": os.environ.get("CHAOSENGINE_LLAMA_SERVER"),
        "llamaServerTurboPath": os.environ.get("CHAOSENGINE_LLAMA_SERVER_TURBO"),
        "llamaCliPath": os.environ.get("CHAOSENGINE_LLAMA_CLI"),
    }


def _gpu_info() -> dict[str, Any]:
    """GPU-runtime presence check — find_spec only, no actual imports.

    For the torch version / CUDA availability reports we shell out to a
    fresh Python subprocess so the backend process itself never imports
    torch (which would lock DLLs on Windows and block the GPU bundle
    install flow).
    """
    return {
        "torchFindSpec": _has_module("torch"),
        "diffusersFindSpec": _has_module("diffusers"),
        "accelerateFindSpec": _has_module("accelerate"),
        "transformersFindSpec": _has_module("transformers"),
        "imageioFindSpec": _has_module("imageio"),
        "ffmpegFindSpec": _has_module("imageio_ffmpeg"),
        "sentencepieceFindSpec": _has_module("sentencepiece"),
        "tiktokenFindSpec": _has_module("tiktoken"),
        "protobufFindSpec": _has_module("google.protobuf"),
        "ftfyFindSpec": _has_module("ftfy"),
        # Actual torch + CUDA status requires a real import, which we do
        # in a subprocess so DLL state in the backend process is
        # untouched. Null when torch isn't installed or the probe fails.
        "torchSubprocess": _probe_torch_subprocess(),
    }


def _has_module(name: str) -> bool:
    """``importlib.util.find_spec`` on a dotted name raises
    ``ModuleNotFoundError`` when the parent namespace is missing
    (``google.protobuf`` on a machine without any ``google`` packages
    being the classic offender). That crashed the whole _gpu_info
    section on the user's Windows VM and broke the Diagnostics panel.
    Catching the raise treats "parent missing" as "child missing",
    which is the semantic the caller wants.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


def _extras_info() -> dict[str, Any]:
    """State of the persistent user-local ``extras/site-packages`` dir.

    Users see this as "where GPU runtime libs live". Surfaces existence,
    size, contents, and free disk so diagnosis covers both "is it
    installed?" and "is there room to install?"
    """
    from backend_service.routes.setup import _extras_site_packages, _free_bytes

    extras = _extras_site_packages()
    if extras is None:
        return {"error": "Could not resolve extras site-packages path"}
    exists = extras.exists()
    free = _free_bytes(extras) if extras else None
    entries: list[str] = []
    size_bytes: int | None = None
    if exists:
        try:
            entries = sorted(p.name for p in extras.iterdir())[:60]
        except OSError:
            entries = []
        size_bytes = _safe(lambda: _dir_size(extras))
    return {
        "path": str(extras),
        "exists": exists,
        "freeBytes": free,
        "sizeBytes": size_bytes,
        "topLevelEntries": entries,
    }


def _env_vars() -> dict[str, Any]:
    """Filtered + redacted environment variables.

    Includes pinned CHAOSENGINE_* / PYTHON* / VIRTUAL_ENV vars always
    (even when empty) so the reader can confirm their expected values.
    Any other env var whose name contains a secret-ish substring is
    redacted to prevent accidental leaks in support pastes.
    """
    seen: dict[str, str | None] = {}
    for key in _PINNED_ENV_VARS:
        seen[key] = os.environ.get(key)
    for key, value in os.environ.items():
        if key in seen:
            continue
        if not (key.startswith("CHAOSENGINE_") or key.startswith("PYTHON")):
            continue
        seen[key] = value
    redacted = {}
    for key, value in seen.items():
        if value is None:
            redacted[key] = None
            continue
        if any(marker in key.lower() for marker in _REDACTED_ENV_SUBSTRINGS):
            redacted[key] = "***redacted***"
            continue
        redacted[key] = value
    return redacted


def _log_info() -> dict[str, Any]:
    path = _active_log_path()
    return {
        "path": str(path) if path else None,
        "tailLines": _read_log_tail(path, _LOG_TAIL_DEFAULT_LINES) if path else [],
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _safe(fn):
    """Call ``fn()``; return None on any exception. Keeps the snapshot
    endpoint from 500ing when a single metric is unavailable.
    """
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024 ** 3), 2)


def _dir_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _disk_usage() -> list[dict[str, Any]]:
    try:
        import psutil
    except ImportError:
        return []
    out: list[dict[str, Any]] = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue
        out.append(
            {
                "mount": part.mountpoint,
                "device": part.device,
                "fstype": part.fstype,
                "totalGb": _bytes_to_gb(usage.total),
                "freeGb": _bytes_to_gb(usage.free),
                "usedPercent": usage.percent,
            }
        )
        if len(out) >= 6:
            break  # don't dump hundreds of loop devices on Linux
    return out


def _nvidia_driver_version() -> str | None:
    """Parse driver version from ``nvidia-smi --query-gpu=driver_version``.

    Returns None when nvidia-smi isn't on PATH or the call fails.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            **_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip().splitlines()[0] if result.stdout else None


def _nvidia_cuda_version() -> str | None:
    """CUDA runtime version reported by the driver (``CUDA Version: X.Y``).

    This is the DRIVER's capability, not the torch wheel's build CUDA —
    the torch CUDA version comes from ``_probe_torch_subprocess``.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5, **_subprocess_kwargs()
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in (result.stdout or "").splitlines():
        idx = line.find("CUDA Version:")
        if idx >= 0:
            return line[idx + len("CUDA Version:") :].strip().split()[0]
    return None


def _probe_torch_subprocess() -> dict[str, Any] | None:
    """Run a fresh Python subprocess to check torch / CUDA status.

    Never imports torch into the backend process — that would load
    torch/lib/*.dll into the handle table and break the GPU bundle
    install flow on Windows.
    """
    if importlib.util.find_spec("torch") is None:
        return None
    script = (
        "import json, sys\n"
        "out = {}\n"
        "try:\n"
        "    import torch\n"
        "    out['version'] = getattr(torch, '__version__', None)\n"
        "    out['cudaBuild'] = str(getattr(torch.version, 'cuda', None))\n"
        "    out['cudaAvailable'] = bool(getattr(torch.cuda, 'is_available', lambda: False)())\n"
        "    try:\n"
        "        out['deviceCount'] = int(torch.cuda.device_count())\n"
        "    except Exception as exc:\n"
        "        out['deviceCountError'] = str(exc)[:200]\n"
        "    try:\n"
        "        if out.get('cudaAvailable'):\n"
        "            out['deviceName'] = torch.cuda.get_device_name(0)\n"
        "    except Exception as exc:\n"
        "        out['deviceNameError'] = str(exc)[:200]\n"
        "    try:\n"
        "        out['mpsAvailable'] = bool(getattr(torch.backends.mps, 'is_available', lambda: False)())\n"
        "    except Exception:\n"
        "        out['mpsAvailable'] = False\n"
        "except Exception as exc:\n"
        "    out['importError'] = f'{type(exc).__name__}: {str(exc)[:300]}'\n"
        "print(json.dumps(out))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            **_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    stdout = (result.stdout or "").strip()
    if not stdout:
        return {"error": "empty stdout", "stderrTail": (result.stderr or "")[-400:]}
    try:
        import json
        return json.loads(stdout)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"parse failure: {exc}", "rawStdout": stdout[-400:]}


def _active_log_path() -> Path | None:
    """Find the backend log file for the currently-bound port.

    Tauri's sidecar launcher names the log ``chaosengine-backend-<port>.log``
    under ``%TEMP%`` / ``$TMPDIR``. We match on the current process's
    port when we can detect it, otherwise return the most recently
    modified log file.
    """
    port = os.environ.get("CHAOSENGINE_PORT") or "8876"
    candidates: list[Path] = []
    try:
        import tempfile
        base = Path(tempfile.gettempdir())
    except Exception:
        return None
    exact = base / f"chaosengine-backend-{port}.log"
    if exact.exists():
        return exact
    # Fall back to the most recently touched backend log if the exact
    # name didn't match — useful when the port rolled over after a
    # conflict.
    try:
        for match in glob.glob(str(base / "chaosengine-backend-*.log")):
            p = Path(match)
            if p.is_file():
                candidates.append(p)
    except Exception:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return candidates[0]


def _read_log_tail(path: Path | None, max_lines: int) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            # Simple approach: read the whole file and take the tail. Log
            # files cap around a few MB in practice (uvicorn request log),
            # so full-reads are fine — if this becomes a hotspot we can
            # swap in a seek-from-end reader.
            lines = handle.readlines()
    except OSError:
        return []
    tail = lines[-max_lines:]
    # Trim trailing newlines so the JSON payload renders cleanly on the
    # frontend without requiring string-split gymnastics.
    return [line.rstrip("\r\n") for line in tail]


def _runtime_extraction_root() -> Path | None:
    """Same path as ``ensure_embedded_runtime_extracted`` in lib.rs.

    Duplicated here because Python doesn't have a way to call into the
    Rust shell to ask. Kept in sync by convention — if the Rust side
    ever changes the path, update this too.
    """
    try:
        import tempfile
        base = Path(tempfile.gettempdir())
    except Exception:
        return None
    arch = platform.machine().lower()
    system = sys.platform
    if arch in ("amd64", "x86_64"):
        arch = "x86_64"
    elif arch in ("arm64", "aarch64"):
        arch = "aarch64"
    plat = (
        "windows" if system == "win32"
        else "darwin" if system == "darwin"
        else "linux"
    )
    tag = f"{plat}-{arch}"
    return base / "chaosengine-embedded-runtime" / tag


def _subprocess_kwargs() -> dict[str, Any]:
    """Windows-only: suppress the cmd.exe flash for short subprocesses."""
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
