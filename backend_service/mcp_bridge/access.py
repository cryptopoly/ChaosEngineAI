"""The Agent access switch and the token agents send. Stored beside settings.json, off by default."""

from __future__ import annotations

import json
import os
import secrets
import threading
from pathlib import Path
from typing import Any

OFF_MESSAGE = (
    "Agent access is off. Turn it on in ChaosEngineAI: Settings > Integrations > Agent access."
)
_LOCK = threading.Lock()


def _path() -> Path:
    from backend_service.app import SETTINGS_DIR

    return Path(SETTINGS_DIR) / "agent_access.json"


def _read() -> dict[str, Any]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(data: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def is_enabled() -> bool:
    return bool(_read().get("enabled", False))


def set_enabled(enabled: bool) -> None:
    with _LOCK:
        data = _read()
        data["enabled"] = bool(enabled)
        if not data.get("token"):
            data["token"] = secrets.token_urlsafe(32)
        _write(data)


def get_token() -> str:
    """The agent token; created on first use so a copied config line survives app restarts."""
    with _LOCK:
        data = _read()
        if not data.get("token"):
            data["token"] = secrets.token_urlsafe(32)
            _write(data)
        return str(data["token"])


def rotate_token() -> str:
    with _LOCK:
        data = _read()
        data["token"] = secrets.token_urlsafe(32)
        _write(data)
        return str(data["token"])


def token_ok(provided: str) -> bool:
    provided = (provided or "").strip()
    return bool(provided) and secrets.compare_digest(provided, get_token())
