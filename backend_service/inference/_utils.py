"""Shared helpers used by inference engines + the controller.

Lives in its own module so submodules (engines, controller pieces) can
import without circling through ``backend_service.inference``'s
``__init__``. Functions here are deliberately small + dependency-free
(stdlib only) so any submodule can import this without dragging the
full package along with it.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.request
from pathlib import Path
from typing import Any


def _now_label() -> str:
    """ISO-ish timestamp for ``LoadedModelInfo.loadedAt`` and similar fields."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_message_content(content: Any) -> str:
    """Coerce whatever shape the chat API received into a single text string.

    The OpenAI-style payloads can be a plain string, a list of content
    parts (``[{"type": "text", "text": "…"}, …]``), or even ``None``.
    Concrete engines all want a single string for downstream prompt
    assembly, so do that conversion in one place.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return " ".join(parts)
    return str(content or "")


def _read_text_tail(path: Path | None, limit: int = 40) -> str:
    """Tail the last ``limit`` lines of a log file. Returns empty string on miss."""
    if path is None or not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])


def _append_runtime_note(existing: str | None, extra: str) -> str:
    """Append ``extra`` to ``existing``, deduping if it's already present."""
    if not existing:
        return extra
    if extra in existing:
        return existing
    return f"{existing} {extra}"


def _http_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST or GET a JSON request and return the parsed response."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if payload is not None else "GET"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _find_open_port() -> int:
    """Bind ephemeral port + return the chosen number for downstream subprocess use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _looks_like_gguf(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return lowered.endswith(".gguf") or "gguf" in lowered


def _resolve_gguf_path(path: str | None, runtime_target: str | None) -> str | None:
    """Find a concrete .gguf file from a path or HF-cache directory.

    When a user loads an HF-cache GGUF repo, the path points to the repo
    directory (e.g. ``models--lmstudio-community--Qwen3.5-9B-GGUF``), not a
    specific file.  We scan for the best .gguf file inside it, excluding
    vision projectors (mmproj) and picking the largest non-projector file.
    """
    for candidate in (path, runtime_target):
        if not candidate:
            continue
        p = Path(candidate)
        if p.is_file() and p.suffix.lower() == ".gguf":
            if "mmproj" in p.name.lower():
                continue
            return str(p)
        if p.is_dir():
            gguf_files = sorted(p.rglob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
            # Filter out vision projector files
            model_files = [f for f in gguf_files if "mmproj" not in f.name.lower()]
            if model_files:
                return str(model_files[0])
    return None


def _is_local_target(candidate: str | None) -> bool:
    if not candidate:
        return False
    expanded = os.path.expanduser(candidate)
    path = Path(expanded)
    return (
        path.exists()
        or expanded.startswith(("/", "~/", "./", "../"))
        or bool(re.match(r"^[A-Za-z]:[\\/]", expanded))
    )
