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
from backend_service.inference.accelerators import (
    dflash_cuda_available,
    dflash_cuda_version,
    dflash_mlx_available,
    dflash_mlx_version,
    kvpress_available,
    kvpress_version,
    nunchaku_available,
    nunchaku_version,
    sageattention_available,
    sageattention_version,
    triattention_available,
    triattention_version,
    wsl2_available,
    wsl_cuda_available,
    wsl_default_distro,
    wsl_vllm_available,
    wsl_vllm_version,
)
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
    # FU-056 Phase 1: prime accelerator flags during the placeholder phase
    # too. The probes are cheap (single ``find_spec`` per package, no
    # imports) so the UI gets accurate "Install" vs "Installed" state on
    # first render without waiting for the full MLX subprocess probe.
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
        nunchakuAvailable=nunchaku_available(),
        nunchakuVersion=nunchaku_version(),
        sageattentionAvailable=sageattention_available(),
        sageattentionVersion=sageattention_version(),
        dflashMlxAvailable=dflash_mlx_available(),
        dflashMlxVersion=dflash_mlx_version(),
        dflashCudaAvailable=dflash_cuda_available(),
        dflashCudaVersion=dflash_cuda_version(),
        triattentionAvailable=triattention_available(),
        triattentionVersion=triattention_version(),
        kvpressAvailable=kvpress_available(),
        kvpressVersion=kvpress_version(),
        wsl2Available=wsl2_available(),
        # FU-056 Phase 8: WSL-detail probes deferred to the full probe
        # below. They shell out to ``wsl --`` subprocesses which can
        # take 5-8 s each on a cold service start — too slow for the
        # placeholder path that primes the first UI render.
        probing=True,
    )


def _probe_native_backends() -> BackendCapabilities:
    python_executable = _resolve_mlx_python()
    llama_server_path = _resolve_llama_server()
    llama_server_turbo_path = _resolve_llama_server_turbo()
    llama_cli_path = _resolve_llama_cli()

    code, payload, message = _json_subprocess(
        [python_executable, "-m", "backend_service.mlx_worker", "probe"],
        # FU-068: cold ``mlx_lm + mlx + mlx_vlm`` import has crept to
        # ~12.4 s on M4 Max / Python 3.11 (measured 2026-05-25 v0.9.3),
        # blowing the original 12.0 s ceiling and causing intermittent
        # E2E Phase 1 fails on a freshly-booted backend. Bump to 20 s
        # for ~60% headroom over today's cold-boot envelope.
        timeout=20.0,
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

    # FU-056 Phase 8: WSL2 + vLLM-bridge probes. ``wsl2_available`` is
    # cheap (``wsl --status`` returns in <100ms on warm LxssManager);
    # the three detail probes shell out via ``wsl --`` and can take a
    # few seconds on a cold service start, so they're gated behind the
    # ``wsl2_active`` short-circuit to avoid paying that cost on hosts
    # that have no WSL at all.
    wsl2_active = wsl2_available()
    wsl_distro = wsl_default_distro() if wsl2_active else None
    wsl_cuda = wsl_cuda_available() if wsl2_active else False
    wsl_vllm = wsl_vllm_available() if wsl2_active else False
    wsl_vllm_ver = wsl_vllm_version() if wsl2_active and wsl_vllm else None

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
        # FU-056 Phase 1: per-accelerator import + version probes.
        nunchakuAvailable=nunchaku_available(),
        nunchakuVersion=nunchaku_version(),
        sageattentionAvailable=sageattention_available(),
        sageattentionVersion=sageattention_version(),
        dflashMlxAvailable=dflash_mlx_available(),
        dflashMlxVersion=dflash_mlx_version(),
        dflashCudaAvailable=dflash_cuda_available(),
        dflashCudaVersion=dflash_cuda_version(),
        triattentionAvailable=triattention_available(),
        triattentionVersion=triattention_version(),
        kvpressAvailable=kvpress_available(),
        kvpressVersion=kvpress_version(),
        # FU-056 Phase 8 WSL bridge state (see note above).
        wsl2Available=wsl2_active,
        wslDistroName=wsl_distro,
        wslCudaAvailable=wsl_cuda,
        wslVllmAvailable=wsl_vllm,
        wslVllmVersion=wsl_vllm_ver,
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
