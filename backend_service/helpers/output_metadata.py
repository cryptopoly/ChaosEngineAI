"""Write generation settings into the output files themselves.

A PNG gets one ``tEXt`` chunk (keyword ``chaosengine``) holding a JSON object; an MP4 gets the same
JSON in its ``comment`` tag. Anyone can then read prompt, model, seed, steps, guidance, size and app
version back from the file alone. Both writers are best effort: a failure never loses the output.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any

PNG_KEYWORD = "chaosengine"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Image and video artifacts use these keys; keep only the ones that exist.
_SETTING_KEYS = (
    "prompt",
    "negativePrompt",
    "modelId",
    "modelName",
    "seed",
    "steps",
    "guidance",
    "width",
    "height",
    "numFrames",
    "fps",
    "startImagePath",
    "endImagePath",
    "imageStrength",
    "referenceImagePaths",
    "createdAt",
)


def generation_settings(artifact: dict[str, Any]) -> dict[str, Any]:
    """Pick the reproducible settings from an artifact dict and add the app version."""
    settings = {
        key: artifact[key]
        for key in _SETTING_KEYS
        if artifact.get(key) not in (None, "")
    }
    try:
        from backend_service.helpers.system_hardware import _resolve_app_version

        settings["appVersion"] = _resolve_app_version()
    except Exception:
        pass
    return settings


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def embed_png_settings(png_bytes: bytes, settings: dict[str, Any]) -> bytes:
    """Return the PNG with a ``tEXt`` chunk inserted before IEND. Non-PNG bytes come back unchanged."""
    if not png_bytes.startswith(_PNG_SIGNATURE) or not settings:
        return png_bytes
    iend = png_bytes.rfind(b"IEND")
    if iend < 8:
        return png_bytes
    insert_at = iend - 4  # the chunk length field sits 4 bytes before the type
    # tEXt is Latin-1; ensure_ascii keeps the JSON in that range.
    payload = json.dumps(settings, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    chunk = _png_chunk(b"tEXt", PNG_KEYWORD.encode("latin-1") + b"\x00" + payload)
    return png_bytes[:insert_at] + chunk + png_bytes[insert_at:]


def read_png_settings(png_bytes: bytes) -> dict[str, Any] | None:
    """Read the settings back from a PNG written by ``embed_png_settings``."""
    if not png_bytes.startswith(_PNG_SIGNATURE):
        return None
    pos = 8
    while pos + 8 <= len(png_bytes):
        length, = struct.unpack(">I", png_bytes[pos : pos + 4])
        kind = png_bytes[pos + 4 : pos + 8]
        data = png_bytes[pos + 8 : pos + 8 + length]
        if kind == b"tEXt":
            key, _, value = data.partition(b"\x00")
            if key == PNG_KEYWORD.encode("latin-1"):
                try:
                    parsed = json.loads(value.decode("ascii"))
                except ValueError:
                    return None
                return parsed if isinstance(parsed, dict) else None
        if kind == b"IEND":
            break
        pos += 12 + length
    return None


def _ffmpeg_exe() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


def embed_mp4_settings(path: Path, settings: dict[str, Any]) -> bool:
    """Copy the streams into a new file with a ``comment`` tag, then replace the original.

    Returns False and leaves the original untouched when ffmpeg is missing or fails.
    """
    exe = _ffmpeg_exe()
    if not exe or not settings or not path.exists():
        return False
    tmp = path.with_name(path.stem + ".meta" + path.suffix)
    comment = json.dumps(settings, ensure_ascii=True, separators=(",", ":"))
    try:
        result = subprocess.run(
            [exe, "-y", "-loglevel", "error", "-i", str(path), "-map", "0", "-c", "copy",
             "-metadata", f"comment={PNG_KEYWORD}:{comment}", str(tmp)],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            return False
        os.replace(tmp, path)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def read_mp4_settings(path: Path) -> dict[str, Any] | None:
    """Read the ``comment`` tag back; the tag text is stored in the file as plain UTF-8."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    marker = (PNG_KEYWORD + ":{").encode("ascii")
    start = data.find(marker)
    if start < 0:
        return None
    start += len(PNG_KEYWORD) + 1
    depth = 0
    for index in range(start, len(data)):
        byte = data[index : index + 1]
        if byte == b"{":
            depth += 1
        elif byte == b"}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(data[start : index + 1].decode("utf-8"))
                except ValueError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None
