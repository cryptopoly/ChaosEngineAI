"""Music Library: generated tracks, one JSON file and one audio file per item.

Each item is ``<music data>/library/<id>.json`` plus ``<id>.flac``. IDs are made here and checked with
``is_valid_id`` before any path is built, audio extensions come from a fixed list, and the only way
in is ``create`` (which moves a file the backend itself produced), so every path built from an item is
inside the library folder. List, get, rename, tag and delete, like the Voice Library.
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

from backend_service import music_env

_ID_PATTERN = re.compile(r"^\d+_[0-9a-f]{8}$")
_TAG_PATTERN = re.compile(r"^[^\x00-\x1f]{1,40}$")
AUDIO_EXTENSIONS = (".flac", ".wav")
KINDS = ("track",)
MAX_TAGS = 20
_LOCK = threading.Lock()


def library_dir() -> Path:
    path = music_env.library_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_valid_id(item_id: str) -> bool:
    return bool(_ID_PATTERN.fullmatch(str(item_id)))


def _new_id() -> str:
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}"


def _meta_path(item_id: str) -> Path:
    if not is_valid_id(item_id):
        raise ValueError("Invalid item id.")
    return library_dir() / f"{item_id}.json"


def _write(meta: dict[str, Any]) -> None:
    path = _meta_path(meta["id"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.replace(path)


def clean_tags(tags: list[Any]) -> list[str]:
    """Trimmed, de-duplicated, at most ``MAX_TAGS`` tags of up to 40 printable characters."""
    seen: list[str] = []
    for raw in tags:
        tag = " ".join(str(raw).split())
        if tag and _TAG_PATTERN.fullmatch(tag) and tag.lower() not in {t.lower() for t in seen}:
            seen.append(tag)
    return seen[:MAX_TAGS]


def _title_from(text: str) -> str:
    return " ".join(text.split())[:60].strip() or "Untitled track"


def create(fields: dict[str, Any], audio_file: Path, audio_ext: str = ".flac") -> dict[str, Any]:
    """Save a new track; ``audio_file`` (written by the worker) is moved into the library."""
    ext = audio_ext.lower()
    if ext not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio type {ext!r}.")
    now = int(time.time())
    caption = str(fields.get("caption") or "")
    item: dict[str, Any] = {
        "id": _new_id(),
        "kind": "track",
        "title": str(fields.get("title") or _title_from(caption))[:200],
        "caption": caption,
        "lyrics": str(fields.get("lyrics") or ""),
        "tags": clean_tags(list(fields.get("tags") or [])),
        "createdAt": now,
        "updatedAt": now,
        "audio": ext,
    }
    for key in ("model", "modelName", "engine", "seed", "durationS", "sampleRate", "channels", "bpm", "key",
                "instrumental", "variation", "jobSeconds", "peakFootprintGb", "source", "sizeBytes", "tagged"):
        if fields.get(key) is not None:
            item[key] = fields[key]
    with _LOCK:
        target = library_dir() / f"{item['id']}{ext}"
        shutil.move(str(audio_file), str(target))
        item["sizeBytes"] = target.stat().st_size
        _write(item)
    return item


def get(item_id: str) -> dict[str, Any] | None:
    if not is_valid_id(item_id):
        return None
    try:
        meta = json.loads(_meta_path(item_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and meta.get("id") == item_id else None


def list_items(query: str = "", tag: str = "") -> list[dict[str, Any]]:
    """Items newest first; ``query`` matches title, caption and tags; ``tag`` must be one of its tags."""
    needle = query.strip().lower()
    wanted_tag = tag.strip().lower()
    items = []
    for path in library_dir().glob("*.json"):
        if not is_valid_id(path.stem):
            continue
        meta = get(path.stem)
        if meta is None:
            continue
        tags = [str(t) for t in meta.get("tags") or []]
        if wanted_tag and wanted_tag not in {t.lower() for t in tags}:
            continue
        haystack = f"{meta.get('title', '')}\n{meta.get('caption', '')}\n{' '.join(tags)}".lower()
        if needle and needle not in haystack:
            continue
        items.append(meta)
    items.sort(key=lambda m: (m.get("createdAt", 0), m.get("id", "")), reverse=True)
    return items


def update(item_id: str, *, title: str | None = None, tags: list[Any] | None = None) -> dict[str, Any] | None:
    """Rename and / or replace the tags."""
    with _LOCK:
        meta = get(item_id)
        if meta is None:
            return None
        if title is not None:
            meta["title"] = " ".join(title.split())[:200] or meta.get("title") or "Untitled track"
        if tags is not None:
            meta["tags"] = clean_tags(tags)
        meta["updatedAt"] = int(time.time())
        _write(meta)
    return meta


def audio_path(item_id: str) -> Path | None:
    meta = get(item_id)
    ext = (meta or {}).get("audio")
    if ext not in AUDIO_EXTENSIONS:
        return None
    path = library_dir() / f"{item_id}{ext}"
    return path if path.is_file() else None


def delete(item_id: str) -> bool:
    with _LOCK:
        if get(item_id) is None:
            return False
        path = audio_path(item_id)
        if path is not None:
            path.unlink(missing_ok=True)
        _meta_path(item_id).unlink(missing_ok=True)
    return True


def disk_bytes() -> int:
    return sum(p.stat().st_size for p in library_dir().iterdir() if p.is_file())
