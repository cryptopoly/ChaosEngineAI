"""llama-server status + upgrade-hint endpoint (FU-047 follow-up).

Reads the installed llama-server binary's ``--version`` and ``--help``
output, reports the build number + whether it advertises the
``draft-mtp`` spec-decoding type that PR #22673 added on 2026-05-16,
and supplies a platform-aware upgrade command the UI can surface as a
one-click action or copy-to-clipboard hint.

We deliberately don't ship a background-job installer here (yet). The
upgrade path is platform-specific and best owned by the OS package
manager: ``brew upgrade llama.cpp`` on macOS, ``apt`` / ``pacman`` /
ggml-org tarball on Linux, MSYS / scoop / tarball on Windows. Trying
to wrap all of those in a "click to update" button is a footgun until
we vendor our own llama.cpp build (planned for v1.0). For now this
endpoint is read-only: tell the user exactly what to run, and let them
run it.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter()


_HOMEBREW_PATHS = [
    Path("/opt/homebrew/bin/llama-server"),
    Path("/usr/local/bin/llama-server"),
]


def _resolve_llama_server() -> Path | None:
    """Find the standard llama-server binary (not the turbo fork)."""
    for candidate in _HOMEBREW_PATHS:
        if candidate.exists() and candidate.is_file():
            return candidate
    which = shutil.which("llama-server")
    return Path(which) if which else None


def _version_string(binary: Path) -> tuple[str | None, str | None]:
    """Return (full version line, build number) from llama-server --version."""
    try:
        proc = subprocess.run(
            [str(binary), "--version"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    # llama-server emits ``version: 9150 (15f786e65)`` on stderr in
    # recent builds, on stdout in older builds. Search both.
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = re.search(r"version:\s*(\d+)\s*\(([a-f0-9]+)\)", blob)
    if match:
        return f"{match.group(1)} ({match.group(2)})", match.group(1)
    # Fallback — return the first non-empty line containing 'version'
    for line in blob.splitlines():
        if "version" in line.lower():
            return line.strip(), None
    return None, None


def _supports_draft_mtp(binary: Path) -> bool:
    """True when ``--help`` lists ``draft-mtp`` among --spec-type values."""
    try:
        proc = subprocess.run(
            [str(binary), "--help"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # Look for the --spec-type line and check its value list.
    for line in blob.splitlines():
        if "--spec-type" in line and "draft-mtp" in line:
            return True
    return False


def _upgrade_command() -> dict[str, Any]:
    """Platform-aware upgrade hint."""
    system = platform.system()
    if system == "Darwin":
        return {
            "platform": "macOS",
            "manager": "homebrew",
            "command": "brew upgrade llama.cpp",
            "manualInstall": "brew install llama.cpp",
            "note": (
                "Homebrew bottles trail ggml-org master by a few hours. "
                "If draft-mtp is missing after upgrade, wait for the next "
                "bottle (usually <24h) or build from source: "
                "git clone https://github.com/ggml-org/llama.cpp && "
                "cd llama.cpp && cmake -B build -DGGML_METAL=ON && "
                "cmake --build build --target llama-server"
            ),
        }
    if system == "Linux":
        return {
            "platform": "Linux",
            "manager": "tarball",
            "command": (
                "curl -L https://github.com/ggml-org/llama.cpp/releases/latest/"
                "download/llama-bin-linux-x64.tar.gz | tar xz -C ~/.chaosengine/bin/"
            ),
            "manualInstall": "Build from source: cmake -B build && cmake --build build --target llama-server",
            "note": "Pre-built tarballs ship from the ggml-org releases page within hours of a tag cut.",
        }
    if system == "Windows":
        return {
            "platform": "Windows",
            "manager": "scoop",
            "command": "scoop update llama.cpp",
            "manualInstall": (
                "Download the latest Windows zip from "
                "https://github.com/ggml-org/llama.cpp/releases/latest"
            ),
            "note": "Or use winget / chocolatey if you installed via those.",
        }
    return {
        "platform": system,
        "manager": "unknown",
        "command": None,
        "manualInstall": "See https://github.com/ggml-org/llama.cpp for build instructions.",
        "note": None,
    }


@router.get("/api/setup/llama-server-status")
def llama_server_status() -> dict[str, Any]:
    """Report whether the resolved llama-server supports MTP + how to upgrade."""
    binary = _resolve_llama_server()
    if binary is None:
        return {
            "installed": False,
            "path": None,
            "version": None,
            "build": None,
            "supportsDraftMtp": False,
            "upgrade": _upgrade_command(),
            "message": (
                "llama-server not found on PATH or in /opt/homebrew/bin/. "
                "Install it to enable GGUF generation + MTP speculative decoding."
            ),
        }

    version_line, build = _version_string(binary)
    supports_mtp = _supports_draft_mtp(binary)
    message: str | None = None
    if not supports_mtp:
        message = (
            f"Installed llama-server (build {build or '?'}) predates ggml-org/llama.cpp "
            "PR #22673 (merged 2026-05-16). Upgrade to enable GGUF MTP speculative "
            "decoding for the MTP-bearing Qwen3.6 GGUFs."
        )

    return {
        "installed": True,
        "path": str(binary),
        "version": version_line,
        "build": build,
        "supportsDraftMtp": supports_mtp,
        "upgrade": _upgrade_command(),
        "message": message,
    }
