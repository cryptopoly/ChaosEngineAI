"""Files in and out of the agent bridge, for an agent on another machine.

Upload: an image goes into an inbox folder next to settings.json and comes back as a handle
(``upload:<id>``) that image and video tools accept wherever they accept a path. Download: a
finished job's output can be fetched over HTTP, but only files inside the app's own output
folders (images, videos, voice library, music library, storyboard exports), never an arbitrary path. Inbox files are removed after a day.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
import urllib.parse
from pathlib import Path
from typing import Any

from backend_service.helpers.media_input import IMAGE_EXTENSIONS, MAX_INPUT_IMAGE_BYTES, InputImageError

HANDLE_PREFIX = "upload:"
INBOX_MAX_AGE_SECONDS = 24 * 3600
_ID_RE = re.compile(r"^[0-9a-f]{12}$")
# (magic bytes, extension)
_MAGIC = ((b"\x89PNG\r\n\x1a\n", ".png"), (b"\xff\xd8\xff", ".jpg"))


def inbox_dir() -> Path:
    from backend_service.app import SETTINGS_DIR

    path = Path(SETTINGS_DIR) / "agent-inbox"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extension_for(data: bytes) -> str | None:
    for magic, extension in _MAGIC:
        if data.startswith(magic):
            return extension
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


def prune_inbox(now: float | None = None) -> None:
    cutoff = (now or time.time()) - INBOX_MAX_AGE_SECONDS
    for path in inbox_dir().iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def save_upload(data: bytes) -> dict[str, Any]:
    """Store an uploaded image; the type comes from its bytes, never from a file name."""
    if not data:
        raise InputImageError("The upload is empty.")
    if len(data) > MAX_INPUT_IMAGE_BYTES:
        raise InputImageError(f"The upload is larger than {MAX_INPUT_IMAGE_BYTES // (1024 * 1024)} MB.")
    extension = _extension_for(data)
    if extension is None or extension not in IMAGE_EXTENSIONS:
        raise InputImageError("The upload is not a PNG, JPEG or WebP image.")
    prune_inbox()
    upload_id = secrets.token_hex(6)
    (inbox_dir() / f"{upload_id}{extension}").write_bytes(data)
    return {"handle": f"{HANDLE_PREFIX}{upload_id}", "bytes": len(data)}


def resolve_handle(raw: str) -> Path:
    """The inbox file behind ``upload:<id>``, or ``InputImageError``."""
    upload_id = raw[len(HANDLE_PREFIX):].strip()
    if not _ID_RE.match(upload_id):
        raise InputImageError(f"'{raw}' is not a valid upload handle.")
    for extension in sorted(IMAGE_EXTENSIONS):
        candidate = inbox_dir() / f"{upload_id}{extension}"
        if candidate.is_file():
            return candidate.resolve()
    raise InputImageError(f"Upload not found (handles last 24 hours): {raw}")


def output_roots() -> list[Path]:
    """Folders a finished job's files may be fetched from."""
    from backend_service import music_env, voice_library
    from backend_service.app import _current_image_outputs_dir, _current_video_outputs_dir

    roots = [
        _current_image_outputs_dir(),
        _current_video_outputs_dir(),
        voice_library.library_dir(),
        music_env.library_dir(),
        *_storyboard_exports(),
    ]
    resolved: list[Path] = []
    for root in roots:
        try:
            resolved.append(Path(root).resolve())
        except OSError:
            pass
    return resolved


def _storyboard_exports() -> list[Path]:
    """Each storyboard project's ``exports/`` folder (films and bundles); never its media or the projects root."""
    from backend_service.storyboard.models import PROJECT_ID_RE
    from backend_service.storyboard.store import storyboards_dir

    root = storyboards_dir()
    try:
        real_root = root.resolve()
        folders = [p / "exports" for p in root.iterdir() if PROJECT_ID_RE.match(p.name)] if root.is_dir() else []
    except OSError:
        return []
    return [f for f in folders if f.is_dir() and not f.is_symlink() and f.resolve().is_relative_to(real_root)]


def fetchable(path: str | Path) -> Path | None:
    """The real path when it is a file inside an output folder, else None."""
    try:
        real = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not real.is_file():
        return None
    return real if any(real.is_relative_to(root) for root in output_roots()) else None


def download_info(path: str | Path) -> dict[str, Any] | None:
    """``{url, bytes, sha256}`` for a fetchable output, else None. ``url`` is relative to the app's address."""
    real = fetchable(path)
    if real is None:
        return None
    digest = hashlib.sha256()
    with real.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {
        "url": "/api/agent/files?path=" + urllib.parse.quote(str(real), safe=""),
        "bytes": real.stat().st_size,
        "sha256": digest.hexdigest(),
    }
