from __future__ import annotations

import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()

_INSTALLABLE_PIP_PACKAGES: dict[str, str] = {
    "turboquant": "turboquant",
    "turboquant-mlx": "turboquant-mlx",
    "triattention": "triattention",
    "vllm": "vllm",
    "mlx": "mlx",
    "mlx-lm": "mlx-lm",
    "dflash-mlx": "dflash-mlx",
    "dflash": "dflash",
}

_MANUAL_INSTALL_MESSAGES: dict[str, str] = {
    "chaosengine": (
        "ChaosEngine is not published on PyPI. Clone "
        "https://github.com/cryptopoly/ChaosEngine and install it into the "
        "backend runtime with: {python} -m pip install -e /path/to/ChaosEngine. "
        "Desktop release builds can also bundle a vendored vendor/ChaosEngine "
        "checkout automatically during npm run stage:runtime."
    ),
    "chaos-engine": (
        "ChaosEngine is not published on PyPI. Clone "
        "https://github.com/cryptopoly/ChaosEngine and install it into the "
        "backend runtime with: {python} -m pip install -e /path/to/ChaosEngine. "
        "Desktop release builds can also bundle a vendored vendor/ChaosEngine "
        "checkout automatically during npm run stage:runtime."
    ),
}

_INSTALLABLE_SYSTEM_PACKAGES: dict[str, list[str]] = {
    "llama.cpp": ["brew", "install", "llama.cpp"],
}


class InstallPackageRequest(BaseModel):
    package: str


@router.post("/api/setup/install-package")
def install_pip_package(request: Request, body: InstallPackageRequest) -> dict[str, Any]:
    """Install a whitelisted pip package into the backend's Python environment."""
    state = request.app.state.chaosengine
    pip_name = _INSTALLABLE_PIP_PACKAGES.get(body.package)
    if pip_name is None:
        manual_message = _MANUAL_INSTALL_MESSAGES.get(body.package)
        if manual_message is not None:
            raise HTTPException(
                status_code=400,
                detail=manual_message.format(python=state.runtime.capabilities.pythonExecutable),
            )
        raise HTTPException(status_code=400, detail=f"Package '{body.package}' is not in the allowed install list.")

    python = state.runtime.capabilities.pythonExecutable
    cmd = [python, "-m", "pip", "install", "--upgrade", pip_name]
    state.add_log("server", "info", f"Installing pip package: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = (result.stdout + "\n" + result.stderr).strip()
        ok = result.returncode == 0
    except subprocess.TimeoutExpired:
        output = "Installation timed out after 5 minutes."
        ok = False
    except OSError as exc:
        output = str(exc)
        ok = False

    # Re-probe capabilities after install
    state.runtime.refresh_capabilities(force=True)
    caps = state.runtime.capabilities.to_dict()
    state.add_log(
        "server", "info" if ok else "error",
        f"pip install {pip_name}: {'succeeded' if ok else 'failed'}",
    )
    return {"ok": ok, "output": output, "capabilities": caps}


@router.post("/api/setup/install-system-package")
def install_system_package(request: Request, body: InstallPackageRequest) -> dict[str, Any]:
    """Install a whitelisted system package (e.g. llama.cpp via brew)."""
    state = request.app.state.chaosengine
    cmd_template = _INSTALLABLE_SYSTEM_PACKAGES.get(body.package)
    if cmd_template is None:
        raise HTTPException(status_code=400, detail=f"System package '{body.package}' is not in the allowed install list.")

    state.add_log("server", "info", f"Installing system package: {' '.join(cmd_template)}")
    try:
        result = subprocess.run(cmd_template, capture_output=True, text=True, timeout=600)
        output = (result.stdout + "\n" + result.stderr).strip()
        ok = result.returncode == 0
    except FileNotFoundError:
        output = f"'{cmd_template[0]}' is not installed. Install Homebrew first: https://brew.sh"
        ok = False
    except subprocess.TimeoutExpired:
        output = "Installation timed out after 10 minutes."
        ok = False
    except OSError as exc:
        output = str(exc)
        ok = False

    state.runtime.refresh_capabilities(force=True)
    caps = state.runtime.capabilities.to_dict()
    state.add_log(
        "server", "info" if ok else "error",
        f"System install {body.package}: {'succeeded' if ok else 'failed'}",
    )
    return {"ok": ok, "output": output, "capabilities": caps}


@router.post("/api/setup/refresh-capabilities")
def refresh_capabilities_endpoint(request: Request) -> dict[str, Any]:
    """Force re-probe all backend capabilities."""
    state = request.app.state.chaosengine
    caps = state.runtime.refresh_capabilities(force=True)
    return {"capabilities": caps.to_dict()}
