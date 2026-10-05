"""Persisted voice gallery — saved transcripts and synthesized audio clips.

Each item is a JSON sidecar under ``~/.chaosengine/voice/gallery/``; audio
items additionally write a sibling ``.wav`` file. IDs are generated here
(timestamp + short uuid) and never accepted from the caller, so path
construction from an item id is always safe.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

_ID_PATTERN = re.compile(r"^\d+_[0-9a-f]{8}$")


def _gallery_dir() -> Path:
    path = Path.home() / ".chaosengine" / "voice" / "gallery"
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_valid_id(item_id: str) -> bool:
    # fullmatch: ``$`` alone also matches before a trailing newline.
    return bool(_ID_PATTERN.fullmatch(item_id))


def _new_id() -> str:
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}"


def save_transcript(text: str) -> dict[str, Any]:
    item_id = _new_id()
    meta: dict[str, Any] = {"id": item_id, "kind": "transcript", "createdAt": int(time.time()), "text": text}
    (_gallery_dir() / f"{item_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def save_audio(text: str, voice: str, audio_bytes: bytes) -> dict[str, Any]:
    item_id = _new_id()
    (_gallery_dir() / f"{item_id}.wav").write_bytes(audio_bytes)
    meta: dict[str, Any] = {
        "id": item_id,
        "kind": "audio",
        "createdAt": int(time.time()),
        "text": text,
        "voice": voice,
    }
    (_gallery_dir() / f"{item_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def list_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in _gallery_dir().glob("*.json"):
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    items.sort(key=lambda meta: meta.get("createdAt", 0), reverse=True)
    return items


def delete_item(item_id: str) -> bool:
    json_path = _gallery_dir() / f"{item_id}.json"
    if not json_path.is_file():
        return False
    audio_path = _gallery_dir() / f"{item_id}.wav"
    if audio_path.is_file():
        audio_path.unlink()
    json_path.unlink()
    return True


def get_audio_path(item_id: str) -> Path | None:
    audio_path = _gallery_dir() / f"{item_id}.wav"
    return audio_path if audio_path.is_file() else None
