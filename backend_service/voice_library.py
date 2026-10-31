"""Voice Library: saved transcripts and generated speech.

Each item is ``~/.chaosengine/voice/library/<id>.json``, plus ``<id><ext>``
when it has audio (the recording or file a transcript came from, or the
generated speech). IDs are made here and never taken from the caller, and
audio extensions come from a fixed list, so file paths built from an item
are always inside the library folder.

Items saved by the old Gallery (``voice/gallery``) move here on first use.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from backend_service import voice_env

_ID_PATTERN = re.compile(r"^\d+_[0-9a-f]{8}$")
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".webm",
    ".mp4", ".mov", ".mkv", ".m4v", ".aiff", ".aif", ".caf",
}
KINDS = ("transcript", "speech")
_EDITABLE = ("title", "text", "segments", "speakers")
_PREVIEW_CHARS = 300
_LOCK = threading.Lock()


def library_dir() -> Path:
    path = voice_env.VOICE_DATA_DIR / "library"
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_valid_id(item_id: str) -> bool:
    # fullmatch: ``$`` alone also matches before a trailing newline.
    return bool(_ID_PATTERN.fullmatch(item_id))


def _new_id() -> str:
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _meta_path(item_id: str) -> Path:
    return library_dir() / f"{item_id}.json"


def _write(meta: dict[str, Any]) -> None:
    path = _meta_path(meta["id"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.replace(path)


def _title_from(text: str, fallback: str) -> str:
    line = " ".join(text.split())[:60].strip()
    return line or fallback


def _migrate_gallery() -> None:
    """Move items from the old Gallery folder into the library, once."""
    old = voice_env.VOICE_DATA_DIR / "gallery"
    if not old.is_dir():
        return
    for meta_file in old.glob("*.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            item_id = str(meta.get("id", ""))
            if not is_valid_id(item_id):
                continue
            audio = old / f"{item_id}.wav"
            kind = "speech" if meta.get("kind") == "audio" else "transcript"
            text = str(meta.get("text") or "")
            item = {
                "id": item_id,
                "kind": kind,
                "title": _title_from(text, kind.title()),
                "text": text,
                "createdAt": int(meta.get("createdAt") or time.time()),
                "updatedAt": int(meta.get("createdAt") or time.time()),
                "audio": None,
                "voice": meta.get("voice"),
                "segments": [],
                "speakers": {},
            }
            if audio.is_file():
                shutil.move(str(audio), str(library_dir() / f"{item_id}.wav"))
                item["audio"] = ".wav"
            _write(item)
            meta_file.unlink()
        except (OSError, ValueError):
            continue
    try:
        old.rmdir()
    except OSError:
        pass


def create(
    kind: str,
    fields: dict[str, Any],
    audio_file: Path | None = None,
    audio_ext: str | None = None,
) -> dict[str, Any]:
    """Save a new item; ``audio_file`` is moved into the library."""
    if kind not in KINDS:
        raise ValueError(f"Unknown item kind {kind!r}.")
    ext = (audio_ext or "").lower()
    if audio_file is not None and ext not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio type {ext!r}.")
    now = int(time.time())
    text = str(fields.get("text") or "")
    item: dict[str, Any] = {
        "id": _new_id(),
        "kind": kind,
        "title": str(fields.get("title") or _title_from(text, kind.title()))[:200],
        "text": text,
        "createdAt": now,
        "updatedAt": now,
        "audio": None,
        "segments": fields.get("segments") or [],
        "speakers": fields.get("speakers") or {},
    }
    for key in ("language", "model", "voice", "durationS", "source", "task", "speed"):
        if fields.get(key) is not None:
            item[key] = fields[key]
    with _LOCK:
        if audio_file is not None:
            shutil.move(str(audio_file), str(library_dir() / f"{item['id']}{ext}"))
            item["audio"] = ext
        _write(item)
    return item


def get(item_id: str) -> dict[str, Any] | None:
    try:
        return json.loads(_meta_path(item_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _summary(meta: dict[str, Any]) -> dict[str, Any]:
    summary = {k: v for k, v in meta.items() if k not in ("segments", "text")}
    text = str(meta.get("text") or "")
    summary["preview"] = text[:_PREVIEW_CHARS]
    summary["speakerCount"] = len({s.get("speaker") for s in meta.get("segments", []) if s.get("speaker") is not None})
    return summary


def list_items(query: str = "", kind: str = "") -> list[dict[str, Any]]:
    """Summaries, newest first; ``query`` matches title and text, case-insensitive."""
    _migrate_gallery()
    needle = query.strip().lower()
    items = []
    for path in library_dir().glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if kind and meta.get("kind") != kind:
            continue
        if needle and needle not in f"{meta.get('title', '')}\n{meta.get('text', '')}".lower():
            continue
        items.append(_summary(meta))
    items.sort(key=lambda m: m.get("createdAt", 0), reverse=True)
    return items


def update(item_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    with _LOCK:
        meta = get(item_id)
        if meta is None:
            return None
        for key in _EDITABLE:
            if patch.get(key) is not None:
                meta[key] = patch[key]
        meta["updatedAt"] = int(time.time())
        _write(meta)
    return meta


def audio_path(item_id: str) -> Path | None:
    meta = get(item_id)
    ext = (meta or {}).get("audio")
    if not ext or ext not in AUDIO_EXTENSIONS:
        return None
    path = library_dir() / f"{item_id}{ext}"
    return path if path.is_file() else None


def delete(item_id: str) -> bool:
    with _LOCK:
        meta_path = _meta_path(item_id)
        if not meta_path.is_file():
            return False
        path = audio_path(item_id)
        if path is not None:
            path.unlink(missing_ok=True)
        meta_path.unlink()
    return True
