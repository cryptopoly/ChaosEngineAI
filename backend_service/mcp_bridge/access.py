"""The Agent access switch and the token agents send. Stored beside settings.json, off by default."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OFF_MESSAGE = (
    "Agent access is off. Turn it on in ChaosEngineAI: Settings > Integrations > Agent access."
)
_LOCK = threading.Lock()

# What a token may do (A4). ``read`` is the floor: status, lists, job status. ``generate`` makes media and
# edits projects; ``files`` moves uploads in and outputs out; ``download`` (a confirmed model download) and
# ``clone`` (voice cloning) are never in a token unless the user ticks them.
SCOPES = ("read", "generate", "files", "download", "clone")
DEFAULT_SCOPES = ("read", "generate", "files")
ALL_SCOPES = frozenset(SCOPES)


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


# ---------------------------------------------------------------------------
# scopes
# ---------------------------------------------------------------------------


def _clean_scopes(values: Any) -> list[str]:
    wanted = {str(v) for v in values or []} & ALL_SCOPES
    return [s for s in SCOPES if s in wanted or s == "read"]


def get_scopes() -> list[str]:
    """Scopes of the main token. A token made before scopes existed gets read, generate and files."""
    with _LOCK:
        data = _read()
        if "scopes" not in data:
            data["scopes"] = list(DEFAULT_SCOPES)
            if data.get("token"):
                _write(data)  # migrate once so the file shows what the token can do
        return _clean_scopes(data["scopes"])


def set_scopes(scopes: Any) -> list[str]:
    with _LOCK:
        data = _read()
        data["scopes"] = _clean_scopes(scopes)
        _write(data)
        return list(data["scopes"])


def remote_clone_allowed() -> bool:
    return bool(_read().get("remoteClone", False))


def set_remote_clone(allowed: bool) -> None:
    with _LOCK:
        data = _read()
        data["remoteClone"] = bool(allowed)
        _write(data)


def get_read_token() -> str | None:
    token = _read().get("readToken")
    return str(token) if token else None


def make_read_token() -> str:
    """A second token that only has ``read``; making it again replaces the old one."""
    with _LOCK:
        data = _read()
        data["readToken"] = secrets.token_urlsafe(32)
        _write(data)
        return str(data["readToken"])


def revoke_read_token() -> None:
    with _LOCK:
        data = _read()
        data.pop("readToken", None)
        _write(data)


@dataclass(frozen=True)
class Principal:
    scopes: frozenset[str]
    owner: str  # short id of the token, for the per-token job cap


def _owner(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:10]


def principal_for(provided: str, loopback: bool = True) -> Principal | None:
    """Who holds ``provided``, or None when it is not an agent token.

    A caller that is not on this computer never gets ``clone`` unless the user ticked that box.
    """
    provided = (provided or "").strip()
    if not provided:
        return None
    scopes: frozenset[str] | None = None
    if secrets.compare_digest(provided, get_token()):
        scopes = frozenset(get_scopes())
    else:
        read_token = get_read_token()
        if read_token and secrets.compare_digest(provided, read_token):
            scopes = frozenset({"read"})
    if scopes is None:
        return None
    if not loopback and not remote_clone_allowed():
        scopes -= {"clone"}
    return Principal(scopes, _owner(provided))
