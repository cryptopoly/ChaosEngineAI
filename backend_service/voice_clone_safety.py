"""Guards every cloned voice passes through: who may clone, the AI tag, the watermark hook, the audit log.

* ``clone_allowed(request)``: the app's own window on this computer may clone; an agent (``/mcp`` or the
  Agent access token) or a caller from another machine may only when the person turned on
  ``allowAgentCloning`` in voice settings (off by default). The agent bridge can wire its token scopes
  to this one helper later.
* ``tag_wav``: a RIFF ``LIST/INFO`` chunk with ``ICMT=chaosengine:{json}`` (``machine_generated`` and
  ``clonedVoice`` always true) and ``ISFT``. Cloned output always gets it; there is no switch.
* ``apply_watermark(path)``: a deliberate no-op until the owner decides D-47 (PerTh or AudioSeal).
  Every cloned output already passes through it, so turning it on is a change to this one function.
* ``audit``: one JSON line per event in ``~/.chaosengine/voice/clone-audit.jsonl``. Local only, never
  uploaded; it records ids, hashes and lengths, never the text spoken.
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
from pathlib import Path
from typing import Any

from backend_service import voice_env

TAG_PREFIX = "chaosengine:"
_LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}
_AUDIT_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Who may clone
# ---------------------------------------------------------------------------

def request_origin(request: Any) -> str:
    """``agent`` (``/mcp`` or the Agent access token), ``remote`` (another machine) or ``ui``."""
    path = str(getattr(getattr(request, "url", None), "path", "") or "")
    if path.startswith("/mcp"):
        return "agent"
    headers = getattr(request, "headers", None) or {}
    header = str(headers.get("authorization", "") or "").strip()
    provided = header[7:].strip() if header.lower().startswith("bearer ") else str(headers.get("x-chaosengine-token", "") or "").strip()
    if provided:
        try:
            from backend_service.mcp_bridge import access  # noqa: PLC0415

            if access.token_ok(provided):
                return "agent"
        except Exception:  # noqa: BLE001 - no settings dir in some tests: not an agent token
            pass
    client = getattr(request, "client", None)
    host = str(getattr(client, "host", "") or "")
    if host and host not in _LOOPBACK:
        return "remote"
    return "ui"


def agent_cloning_enabled() -> bool:
    from backend_service.voice_runtime import load_settings  # noqa: PLC0415

    return bool(load_settings().get("allowAgentCloning"))


def clone_allowed(request: Any) -> bool:
    return request_origin(request) == "ui" or agent_cloning_enabled()


REFUSED_MESSAGE = (
    "Cloned voices can only be used from the ChaosEngineAI window on this computer. "
    "To let agents use them, turn on Voice > Settings > Allow agents to use cloned voices."
)


# ---------------------------------------------------------------------------
# AI tag in WAV files
# ---------------------------------------------------------------------------

def _chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos, chunks = 12, []
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        body = data[pos + 8:pos + 8 + size]
        chunks.append((cid, body))
        pos += 8 + size + (size & 1)
    return chunks


def _info_chunk(fields: dict[bytes, str]) -> bytes:
    out = bytearray(b"INFO")
    for key, value in fields.items():
        raw = value.encode("utf-8") + b"\x00"
        out += key + struct.pack("<I", len(raw)) + raw + (b"\x00" if len(raw) & 1 else b"")
    return bytes(out)


def tag_settings(fields: dict[str, Any]) -> dict[str, Any]:
    """The tag JSON: given fields plus ``machine_generated`` and ``clonedVoice``, always true."""
    settings = {k: v for k, v in fields.items() if v not in (None, "")}
    settings["machine_generated"] = True
    settings["clonedVoice"] = True
    return settings


def tag_wav(data: bytes, fields: dict[str, Any], software: str) -> bytes:
    """``data`` with any old ``LIST`` chunk replaced by ours, placed before ``data``."""
    chunks = [(cid, body) for cid, body in _chunks(data) if cid != b"LIST"]
    comment = TAG_PREFIX + json.dumps(tag_settings(fields), ensure_ascii=False, separators=(",", ":"))
    info = (b"LIST", _info_chunk({b"ISFT": software, b"ICMT": comment}))
    at = next((i for i, (cid, _b) in enumerate(chunks) if cid == b"data"), len(chunks))
    chunks.insert(at, info)
    body = bytearray(b"WAVE")
    for cid, chunk in chunks:
        body += cid + struct.pack("<I", len(chunk)) + chunk + (b"\x00" if len(chunk) & 1 else b"")
    return b"RIFF" + struct.pack("<I", len(body)) + bytes(body)


def read_wav_tag(source: Path | bytes) -> dict[str, Any] | None:
    """The ``chaosengine:`` JSON from a WAV file's ``LIST/INFO/ICMT``, or ``None``."""
    try:
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        chunks = _chunks(data)
    except (OSError, ValueError, struct.error):
        return None
    for cid, body in chunks:
        if cid != b"LIST" or body[:4] != b"INFO":
            continue
        pos = 4
        while pos + 8 <= len(body):
            key = body[pos:pos + 4]
            (size,) = struct.unpack_from("<I", body, pos + 4)
            value = body[pos + 8:pos + 8 + size].rstrip(b"\x00").decode("utf-8", "replace")
            pos += 8 + size + (size & 1)
            if key == b"ICMT" and value.startswith(TAG_PREFIX):
                try:
                    parsed = json.loads(value[len(TAG_PREFIX):])
                except ValueError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


# ---------------------------------------------------------------------------
# Watermark hook (D-47 pending)
# ---------------------------------------------------------------------------

def apply_watermark(path: Path) -> bool:
    """Hook for an imperceptible watermark on cloned-voice audio. A no-op today; returns False.

    The owner has not yet chosen whether to add one (D-47: PerTh or AudioSeal, cloned voices only).
    Every cloned output is written to ``path`` and passed here before it is tagged and saved, so the
    watermark lands in one place when it is approved.
    """
    return False


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def audit_path() -> Path:
    return voice_env.VOICE_DATA_DIR / "clone-audit.jsonl"


def audit(event: str, **fields: Any) -> None:
    """Append one line; a failure to write never blocks the action."""
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
    try:
        path = audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
