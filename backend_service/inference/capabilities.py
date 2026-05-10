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

from backend_service.inference._constants import CAPABILITY_CACHE_TTL_SECONDS
from backend_service.inference.base import BackendCapabilities
from backend_service.inference.binaries import (
    _json_subprocess,
    _resolve_llama_cli,
    _resolve_llama_server,
    _resolve_llama_server_turbo,
    _resolve_mlx_python,
)


_capability_cache: tuple[float, BackendCapabilities] | None = None
_capability_lock = RLock()


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
