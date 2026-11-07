"""Generation settings inside a FLAC file's Vorbis comments, written and read in pure Python.

A music track gets ``ENCODER`` (app and engine) and ``COMMENT=chaosengine:{json}`` with the app version,
model, seed, duration, BPM and key when given, the style prompt and the D-34 label
``machine_generated: true``; never lyrics and never a local path. The block is rewritten in place
(STREAMINFO and every other block kept, the old comment and padding replaced), because an ffmpeg
stream copy renames ``COMMENT`` to ``DESCRIPTION`` and replaces ``ENCODER`` with its own. Writing is
best effort: a failure leaves the file as it was.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any

COMMENT_PREFIX = "chaosengine:"
_FLAC_MAGIC = b"fLaC"
_STREAMINFO, _PADDING, _VORBIS_COMMENT = 0, 1, 4
_PADDING_BYTES = 1024
# Keys a music tag may carry; anything else (lyrics, paths) is dropped.
TAG_KEYS = ("appVersion", "model", "modelName", "engine", "seed", "durationSeconds", "bpm", "key",
            "prompt", "instrumental", "variation", "createdAt", "machine_generated", "kind", "loop", "licence")


def _blocks(data: bytes) -> tuple[list[tuple[int, bytes]], int]:
    """``([(type, body)], offset of the first audio frame)``; raises ValueError for a non-FLAC file."""
    if not data.startswith(_FLAC_MAGIC):
        raise ValueError("not a FLAC file")
    pos, blocks = 4, []
    while True:
        if pos + 4 > len(data):
            raise ValueError("truncated FLAC metadata")
        header = data[pos]
        length = int.from_bytes(data[pos + 1:pos + 4], "big")
        body = data[pos + 4:pos + 4 + length]
        if len(body) != length:
            raise ValueError("truncated FLAC metadata block")
        blocks.append((header & 0x7F, body))
        pos += 4 + length
        if header & 0x80:
            return blocks, pos


def _vorbis_body(vendor: str, comments: list[tuple[str, str]]) -> bytes:
    out = bytearray()
    raw_vendor = vendor.encode("utf-8")
    out += struct.pack("<I", len(raw_vendor)) + raw_vendor + struct.pack("<I", len(comments))
    for key, value in comments:
        entry = f"{key}={value}".encode("utf-8")
        out += struct.pack("<I", len(entry)) + entry
    return bytes(out)


def _parse_vorbis(body: bytes) -> list[tuple[str, str]]:
    (vendor_len,) = struct.unpack_from("<I", body, 0)
    pos = 4 + vendor_len
    (count,) = struct.unpack_from("<I", body, pos)
    pos += 4
    comments = []
    for _ in range(count):
        (length,) = struct.unpack_from("<I", body, pos)
        entry = body[pos + 4:pos + 4 + length].decode("utf-8", "replace")
        pos += 4 + length
        key, _, value = entry.partition("=")
        comments.append((key, value))
    return comments


def music_tag_settings(fields: dict[str, Any]) -> dict[str, Any]:
    """The allowed keys only, with ``machine_generated`` always true."""
    settings = {k: fields[k] for k in TAG_KEYS if fields.get(k) not in (None, "")}
    settings["machine_generated"] = True
    return settings


def embed_flac_tags(path: Path, settings: dict[str, Any], encoder: str) -> bool:
    """Replace the file's Vorbis comment block with ``ENCODER`` and the ``COMMENT`` JSON. False on failure."""
    try:
        data = path.read_bytes()
        blocks, audio_at = _blocks(data)
    except (OSError, ValueError):
        return False
    if not blocks or blocks[0][0] != _STREAMINFO:
        return False
    comment = COMMENT_PREFIX + json.dumps(music_tag_settings(settings), ensure_ascii=False, separators=(",", ":"))
    kept = [(kind, body) for kind, body in blocks if kind not in (_PADDING, _VORBIS_COMMENT)]
    kept.append((_VORBIS_COMMENT, _vorbis_body("ChaosEngineAI", [("ENCODER", encoder), ("COMMENT", comment)])))
    kept.append((_PADDING, b"\x00" * _PADDING_BYTES))
    out = bytearray(_FLAC_MAGIC)
    for index, (kind, body) in enumerate(kept):
        last = 0x80 if index == len(kept) - 1 else 0
        out += bytes([last | kind]) + len(body).to_bytes(3, "big") + body
    out += data[audio_at:]
    tmp = path.with_name(path.name + ".tag")
    try:
        tmp.write_bytes(bytes(out))
        os.replace(tmp, path)
        return True
    except OSError:
        tmp.unlink(missing_ok=True)
        return False


def read_flac_tags(path: Path) -> dict[str, str]:
    """Every Vorbis comment as ``{KEY: value}`` (keys upper-cased), ``{}`` when there is none."""
    try:
        blocks, _ = _blocks(Path(path).read_bytes())
    except (OSError, ValueError):
        return {}
    for kind, body in blocks:
        if kind == _VORBIS_COMMENT:
            try:
                return {key.upper(): value for key, value in _parse_vorbis(body)}
            except struct.error:
                return {}
    return {}


def read_audio_settings(path: Path) -> dict[str, Any] | None:
    """The ``chaosengine:`` JSON from a FLAC file's ``COMMENT`` tag, or ``None``."""
    comment = read_flac_tags(path).get("COMMENT", "")
    if not comment.startswith(COMMENT_PREFIX):
        return None
    try:
        parsed = json.loads(comment[len(COMMENT_PREFIX):])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
