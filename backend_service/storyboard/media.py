"""Media files for a storyboard project: check an incoming file, copy it in, hash it and measure it.

Incoming files are local paths (or ``upload:`` handles) that a user or agent names, so they
are resolved, must be regular files with an allowed extension and a size cap. Paths inside a project are
always resolved against the project folder and must stay inside it. Files enter ``media/`` by hard link
when the source is on the same disk, else by copy, and are recorded with a sha256. Durations come from
the pinned bundled ffmpeg (imported lazily), counted in frames for video so a 97-frame 24 fps clip
measures 4.0417 s and not the rounded container duration.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

from backend_service.storyboard.models import check_rel_path

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".m4v", ".webm"})
AUDIO_EXTS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"})
MAX_MEDIA_BYTES = 8 * 1024**3


class MediaError(ValueError):
    """A readable message for the user or agent."""


def kind_of(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return None


def resolve_source(raw: str, allowed: set[str]) -> Path:
    """The real path of a file to import, or ``MediaError``. ``allowed`` holds 'image', 'video', 'audio'."""
    text = str(raw or "").strip()
    if not text:
        raise MediaError("The file path is empty.")
    if text.startswith("upload:"):
        from backend_service.helpers.media_input import InputImageError
        from backend_service.mcp_bridge.files import UPLOAD_EXTENSIONS, resolve_handle

        try:
            text = str(resolve_handle(text, UPLOAD_EXTENSIONS))
        except InputImageError as exc:
            raise MediaError(str(exc)) from exc
    try:
        path = Path(text).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MediaError(f"File not found: {Path(text).name}") from exc
    if not path.is_file():
        raise MediaError(f"Not a file: {path.name}")
    kind = kind_of(path)
    if kind not in allowed:
        raise MediaError(f"'{path.suffix or 'no extension'}' is not a {' or '.join(sorted(allowed))} file: {path.name}")
    size = path.stat().st_size
    if size == 0 or size > MAX_MEDIA_BYTES:
        raise MediaError(f"File is empty or larger than {MAX_MEDIA_BYTES // 1024**3} GB: {path.name}")
    return path


def inside(folder: Path, rel: str, must_exist: bool = True) -> Path:
    """``folder / rel`` resolved, refusing any path form outside ``media/`` and anything that leaves ``folder``."""
    try:
        check_rel_path(rel)
    except ValueError as exc:
        raise MediaError(f"Refused path '{rel[:80]}': {exc}") from exc
    root = folder.resolve()
    try:
        real = (root / rel).resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise MediaError(f"Missing file: {rel}") from exc
    if not real.is_relative_to(root):
        raise MediaError(f"Refused path '{rel}': it leaves the project folder.")
    if must_exist and not real.is_file():
        raise MediaError(f"Not a file: {rel}")
    return real


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_or_copy(src: Path, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        tmp.unlink()
    linked = False
    try:
        if src.stat().st_dev == dest.parent.stat().st_dev:
            os.link(src, tmp)
            linked = True
    except OSError:
        linked = False
    if not linked:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def _media_dest(folder: Path, rel_dest: str) -> Path:
    dest = inside(folder, rel_dest, must_exist=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.parent.resolve().is_relative_to(folder.resolve()):
        raise MediaError(f"Refused path '{rel_dest}': it leaves the project folder.")
    return dest


def ingest(folder: Path, src: Path, rel_dest: str) -> str:
    """Hard-link (same disk) or copy ``src`` to ``folder/rel_dest``; returns its sha256."""
    dest = _media_dest(folder, rel_dest)
    _link_or_copy(src, dest)
    return sha256_file(dest)


def stage(folder: Path, src: Path) -> Path:
    """Link or copy ``src`` into ``folder/cache/incoming`` (the slow part of an import, done without the lock)."""
    incoming = folder / "cache" / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    dest = incoming / f"{os.urandom(8).hex()}{src.suffix.lower()}"
    _link_or_copy(src, dest)
    return dest


def adopt(folder: Path, staged: Path, rel_dest: str) -> Path:
    """Move a staged file to ``folder/rel_dest`` (a rename inside the project folder: quick)."""
    dest = _media_dest(folder, rel_dest)
    os.replace(staged, dest)
    return dest


def ffmpeg_exe() -> str | None:
    """The pinned bundled ffmpeg, never one on PATH (section 5.4)."""
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


_FRAME_RE = re.compile(r"frame=\s*(\d+)")
_FPS_RE = re.compile(r"Video:.*?(\d+(?:\.\d+)?) fps")
_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")


def probe_seconds(path: Path) -> float | None:
    """Measured duration in seconds (video: frames / fps; audio: decoded time), or None."""
    exe = ffmpeg_exe()
    if not exe:
        return None
    is_video = kind_of(path) == "video"
    args = [exe, "-hide_banner", "-nostdin", "-i", str(path)]
    args += ["-map", "0:v:0", "-c", "copy", "-f", "null", "-"] if is_video else ["-map", "0:a:0", "-f", "null", "-"]
    try:
        result = subprocess.run(args, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    text = result.stderr.decode("utf-8", "replace")
    if result.returncode != 0:
        return None
    if is_video:
        frames = _FRAME_RE.findall(text)
        fps = _FPS_RE.search(text)
        if frames and fps and float(fps.group(1)) > 0:
            return round(int(frames[-1]) / float(fps.group(1)), 4)
    times = _TIME_RE.findall(text)
    if times:
        h, m, s = times[-1]
        return round(int(h) * 3600 + int(m) * 60 + float(s), 4)
    return None


def scrub(text: str) -> str:
    """Drop folder names from a message (a stored error must not carry the user's paths)."""
    return re.sub(r"(?:[A-Za-z]:)?[\\/](?:[^\s\\/'\"]+[\\/])+([^\s\\/'\"]+)", r"\1", str(text))[:1000]
