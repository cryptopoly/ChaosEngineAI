"""Binary + Python-runtime resolution for the inference layer.

Where to find ``llama-server`` / ``llama-server-turbo`` / ``llama-cli``,
which Python interpreter to use for the MLX worker, and a small JSON
subprocess wrapper used by the capability probe and other helpers.

Extracted from ``inference/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from backend_service.inference._constants import WORKSPACE_ROOT


def _json_subprocess(
    command: list[str],
    *,
    timeout: float = 15.0,
    cwd: Path = WORKSPACE_ROOT,
) -> tuple[int, dict[str, Any] | None, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (-1, None, str(exc))

    payload: dict[str, Any] | None = None
    stdout = completed.stdout.decode("utf-8", errors="replace").strip()
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
    return (completed.returncode, payload, stderr or stdout)


def _resolve_mlx_python() -> str:
    override = os.getenv("CHAOSENGINE_MLX_PYTHON")
    if override:
        return override
    candidate = WORKSPACE_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


# Common install locations for llama.cpp binaries that may not be in PATH
# when launched from a GUI app (Tauri doesn't inherit the user's shell profile).
_CHAOSENGINE_BIN_DIR = str(Path.home() / ".chaosengine" / "bin")

_LLAMA_FALLBACK_DIRS = [
    _CHAOSENGINE_BIN_DIR,        # ChaosEngineAI-managed binaries
    "/opt/homebrew/bin",         # macOS ARM Homebrew
    "/usr/local/bin",            # macOS Intel Homebrew / manual
    "/usr/bin",                  # system
    str(Path.home() / ".local" / "bin"),  # pip --user installs
]


def _which_with_fallbacks(name: str) -> str | None:
    """Like shutil.which but also checks common install directories."""
    found = shutil.which(name)
    if found:
        return found
    for d in _LLAMA_FALLBACK_DIRS:
        candidate = os.path.join(d, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _resolve_llama_server() -> str | None:
    override = os.getenv("CHAOSENGINE_LLAMA_SERVER")
    if override:
        return override
    return _which_with_fallbacks("llama-server")


def _resolve_llama_server_turbo() -> str | None:
    """Resolve the TurboQuant fork of llama-server (``llama-server-turbo``).

    This fork supports all standard cache types **plus** iso/planar/turbo
    cache types required by RotorQuant and TurboQuant strategies.
    """
    override = os.getenv("CHAOSENGINE_LLAMA_SERVER_TURBO")
    if override:
        return override
    return _which_with_fallbacks("llama-server-turbo")


def _resolve_llama_cli() -> str | None:
    override = os.getenv("CHAOSENGINE_LLAMA_CLI")
    if override:
        return override
    return _which_with_fallbacks("llama-cli")
