"""Backend capability probe + cache.

Probes MLX (via subprocess so torch / mlx import cost lands in a child
process), llama-server availability, and vLLM importability. The result
is cached for ``CAPABILITY_CACHE_TTL_SECONDS`` so the FastAPI capability
endpoint stays cheap.

Extracted from ``inference/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import time
from threading import RLock

from pathlib import Path

from backend_service.inference._constants import CAPABILITY_CACHE_TTL_SECONDS
from backend_service.inference.base import BackendCapabilities
from backend_service.inference.binaries import (
    _json_subprocess,
    _resolve_llama_cli,
    _resolve_llama_server,
    _resolve_llama_server_turbo,
    _resolve_mlx_python,
)


_MTPLX_VENV = Path.home() / ".chaosengine" / "mtplx-venv"
_MTPLX_VERSION_FILE = Path.home() / ".chaosengine" / "bin" / "mtplx.version"

_capability_cache: tuple[float, BackendCapabilities] | None = None
_capability_lock = RLock()


def _detect_mtplx() -> tuple[bool, str | None]:
    """Return (available, python_path) for the MTPLX isolated venv.

    Cheap file-existence check — no subprocess spawn.  The version file is
    written by install-mtplx.sh on clean install; its presence together with
    the venv python binary is sufficient to confirm a usable install.
    """
    python = _MTPLX_VENV / "bin" / "python"
    if _MTPLX_VERSION_FILE.exists() and python.exists():
        return True, str(python)
    return False, None


def _initial_backend_capabilities() -> BackendCapabilities:
    """Cheap capability placeholder used while the real probe runs.

    The full probe imports/spawns MLX and checks vLLM, which can add seconds
    to cold start. These path checks are safe enough for initial UI rendering;
    load_model() still refreshes capabilities synchronously before selecting
    an engine.
    """
    python_executable = _resolve_mlx_python()
    llama_server_path = _resolve_llama_server()
    llama_server_turbo_path = _resolve_llama_server_turbo()
    llama_cli_path = _resolve_llama_cli()
    mtplx_available, mtplx_python = _detect_mtplx()
    return BackendCapabilities(
        pythonExecutable=python_executable,
        mlxAvailable=False,
        mlxLmAvailable=False,
        mlxUsable=False,
        mlxMessage="Native backend detection is still running.",
        ggufAvailable=bool(llama_server_path) or bool(llama_server_turbo_path),
        llamaCliPath=llama_cli_path,
        llamaServerPath=llama_server_path,
        llamaServerTurboPath=llama_server_turbo_path,
        converterAvailable=False,
        vllmAvailable=False,
        vllmVersion=None,
        mtplxAvailable=mtplx_available,
        mtplxPythonPath=mtplx_python,
        probing=True,
    )


def _probe_native_backends() -> BackendCapabilities:
    python_executable = _resolve_mlx_python()
    llama_server_path = _resolve_llama_server()
    llama_server_turbo_path = _resolve_llama_server_turbo()
    llama_cli_path = _resolve_llama_cli()

    code, payload, message = _json_subprocess(
        [python_executable, "-m", "backend_service.mlx_worker", "probe"],
        timeout=12.0,
    )

    if payload is None:
        payload = {}

    mlx_available = bool(payload.get("mlxAvailable", False))
    mlx_lm_available = bool(payload.get("mlxLmAvailable", False))
    mlx_usable = bool(payload.get("mlxUsable", False) and code == 0)
    probe_message = payload.get("message")
    if probe_message is None and code != 0:
        probe_message = message or f"probe exited with code {code}"

    from backend_service.vllm_engine import _vllm_importable, _vllm_version

    mtplx_available, mtplx_python = _detect_mtplx()

    # FU-047: detect whether the resolved llama-server advertises
    # --spec-type (PR #22673 merged 2026-05-16). Probe the standard
    # binary first, fall back to the turbo fork. Either is sufficient
    # because the same flag was implemented in both upstream branches.
    gguf_mtp_available = False
    if llama_server_path or llama_server_turbo_path:
        from backend_service.inference.llama_cpp_engine import _llama_server_supports
        gguf_mtp_available = bool(
            (llama_server_path and _llama_server_supports(llama_server_path, "--spec-type"))
            or (llama_server_turbo_path and _llama_server_supports(llama_server_turbo_path, "--spec-type"))
        )

    return BackendCapabilities(
        pythonExecutable=python_executable,
        mlxAvailable=mlx_available,
        mlxLmAvailable=mlx_lm_available,
        mlxUsable=mlx_usable,
        mlxVersion=payload.get("mlxVersion"),
        mlxLmVersion=payload.get("mlxLmVersion"),
        mlxMessage=probe_message,
        ggufAvailable=bool(llama_server_path) or bool(llama_server_turbo_path),
        llamaCliPath=llama_cli_path,
        llamaServerPath=llama_server_path,
        llamaServerTurboPath=llama_server_turbo_path,
        converterAvailable=mlx_usable,
        vllmAvailable=_vllm_importable(),
        vllmVersion=_vllm_version(),
        mtplxAvailable=mtplx_available,
        mtplxPythonPath=mtplx_python,
        ggufMtpAvailable=gguf_mtp_available,
    )


def get_backend_capabilities(*, force: bool = False) -> BackendCapabilities:
    global _capability_cache
    with _capability_lock:
        now = time.time()
        if not force and _capability_cache is not None:
            cached_at, cached = _capability_cache
            if (now - cached_at) < CAPABILITY_CACHE_TTL_SECONDS:
                return cached

        capabilities = _probe_native_backends()
        _capability_cache = (now, capabilities)
        return capabilities
