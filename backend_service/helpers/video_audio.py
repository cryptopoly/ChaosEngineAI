"""Tell whether an mp4 carries an audio track, without ffprobe.

LTX-2 clips are written with stereo sound; every other video model makes
silent files. The gallery and the result window show a sound badge and play
the clip with sound only when this says there is a track. The check walks the
top-level boxes by header (seeking over the media data), then ``moov`` ->
``trak`` -> ``mdia`` -> ``hdlr`` and looks for the ``soun`` handler type, so
it reads a few kilobytes even from a large file.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

_CONTAINERS = (b"moov", b"trak", b"mdia")


def _has_sound_track(stream: BinaryIO, end: int) -> bool:
    while stream.tell() + 8 <= end:
        start = stream.tell()
        head = stream.read(8)
        if len(head) < 8:
            return False
        size = int.from_bytes(head[:4], "big")
        kind = head[4:8]
        header = 8
        if size == 1:
            extended = stream.read(8)
            if len(extended) < 8:
                return False
            size = int.from_bytes(extended, "big")
            header = 16
        elif size == 0:
            size = end - start
        if size < header:
            return False
        box_end = min(end, start + size)
        if kind in _CONTAINERS:
            if _has_sound_track(stream, box_end):
                return True
        elif kind == b"hdlr":
            payload = stream.read(12)
            if len(payload) == 12 and payload[8:12] == b"soun":
                return True
        stream.seek(box_end)
    return False


def mp4_has_audio_bytes(data: bytes) -> bool:
    try:
        return _has_sound_track(io.BytesIO(data), len(data))
    except (OSError, ValueError):
        return False


def mp4_has_audio_file(path: str | Path) -> bool:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, io.SEEK_END)
            end = handle.tell()
            handle.seek(0)
            return _has_sound_track(handle, end)
    except (OSError, ValueError):
        return False
