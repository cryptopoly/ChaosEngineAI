"""Files this run's chat already made that the project does not hold yet (Keep going, FU-197).

In a rehearsal the model made two music tracks and a voice line, never attached them, and each nudge told it to
make music again. The nudge now lists what the chat's finished jobs wrote since the run started and the project
lacks (compared by file hash, as imports keep the hash of what they copy), so the model attaches it instead.
Read only; a file that went missing is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_KINDS = {
    "music": {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"},
    "voice": {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"},
    "image": {".png", ".jpg", ".jpeg", ".webp"},
    "video": {".mp4", ".mov", ".webm", ".mkv", ".m4v"},
}
_HASHES: dict[tuple[str, int, int], str] = {}  # (path, size, mtime_ns) -> sha256: a clip is hashed once


@dataclass
class Asset:
    kind: str  # music | voice | image | video (the job's kind)
    path: str


def _hash(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        from backend_service.storyboard.media import sha256_file

        try:
            _HASHES[key] = sha256_file(path)
        except OSError:
            return None
    return _HASHES[key]


def made_files(jobs: list[Any], since: float) -> list[Asset]:
    """Output files of the chat's finished jobs since ``since``, newest first, one per path."""
    out: list[Asset] = []
    seen: set[str] = set()
    for job in sorted(jobs, key=lambda j: getattr(j, "created_at", 0.0) or 0.0, reverse=True):
        kind = str(getattr(job, "kind", ""))
        output = getattr(job, "output", None)
        if (getattr(job, "state", "") != "done" or kind not in _KINDS or not isinstance(output, dict)
                or (getattr(job, "created_at", 0.0) or 0.0) < since):
            continue
        path = str(output.get("path") or "")
        if not path or path in seen or Path(path).suffix.lower() not in _KINDS[kind]:
            continue
        seen.add(path)
        out.append(Asset(kind, path))
    return out


def loose(assets: list[Asset], hashes: set[str], kinds: set[str]) -> list[Asset]:
    """The assets of ``kinds`` whose file is there and not in the project (``hashes``)."""
    out = []
    for asset in assets:
        if asset.kind not in kinds:
            continue
        digest = _hash(Path(asset.path))
        if digest is not None and digest not in hashes:
            out.append(asset)
    return out


def asset_line(assets: list[Asset], limit: int = 6) -> str:
    """The nudge's line about them, or "" when there are none."""
    if not assets:
        return ""
    items = "; ".join(f"{a.kind} {a.path}" for a in assets[:limit])
    more = f" (and {len(assets) - limit} more)" if len(assets) > limit else ""
    return (f"Already made in this chat and not in the project yet: {items}{more}. Attach with storyboard_audio "
            "add path=... (music, voice) or storyboard_shot import path=... (stills, clips); do not make it again.")
