"""Waveform peaks for one project audio file (or the sound of a video clip), for the Mixer timeline.

The pinned bundled ffmpeg decodes the file to mono 8 kHz float32 on a pipe; the samples are folded into
a fixed number of buckets (the largest absolute value in each) and scaled so the loudest bucket is 1.0.
The result is cached as a small JSON file under the project's ``cache/peaks/``, keyed by the file's
sha256 and the bucket count, so a file that is replaced or relinked never shows an old waveform.
Nothing here accepts a path from the caller: the route resolves it inside the project first.
"""

from __future__ import annotations

import array
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from backend_service.storyboard import ffmpeg_bin
from backend_service.storyboard.media import sha256_file

SAMPLE_RATE = 8000
MIN_BUCKETS = 32
MAX_BUCKETS = 4000
MAX_SECONDS = 36000
DECODE_TIMEOUT_S = 180
_CHUNK_BYTES = 1 << 18

_hash_memo: dict[tuple[str, int, int], str] = {}
_hash_lock = threading.Lock()


class PeaksError(ValueError):
    """A readable message for the user."""


def clamp_buckets(buckets: int) -> int:
    return max(MIN_BUCKETS, min(MAX_BUCKETS, int(buckets)))


def _file_hash(path: Path) -> str:
    """sha256 of the file, remembered per (path, size, mtime) so repeated requests do not re-read it."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _hash_lock:
        known = _hash_memo.get(key)
    if known:
        return known
    digest = sha256_file(path)
    with _hash_lock:
        if len(_hash_memo) > 512:
            _hash_memo.clear()
        _hash_memo[key] = digest
    return digest


def fold_samples(chunks: Iterable[Any], total_samples: int, buckets: int) -> list[float]:
    """Largest absolute sample per bucket. ``chunks`` yields ``array('f')`` blocks; ``total_samples`` sizes the buckets."""
    per_bucket = max(1, -(-total_samples // buckets))  # ceil
    peaks = [0.0] * buckets
    index = 0
    for block in chunks:
        position = 0
        length = len(block)
        while position < length:
            bucket = min(buckets - 1, index // per_bucket)
            take = length - position
            if bucket < buckets - 1:
                take = min(take, per_bucket * (bucket + 1) - index)
            segment = block[position:position + take]
            peak = max(max(segment), -min(segment))
            if peak > peaks[bucket]:
                peaks[bucket] = peak
            position += take
            index += take
    return peaks


def _decode_chunks(exe: str, path: Path) -> Iterable[Any]:
    args = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-protocol_whitelist", ffmpeg_bin.PROTOCOLS,
            "-i", str(path), "-vn", "-map", "0:a:0", "-t", str(MAX_SECONDS), "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "f32le", "pipe:1"]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + DECODE_TIMEOUT_S
    try:
        assert proc.stdout is not None
        leftover = b""
        while True:
            if time.monotonic() > deadline:
                raise PeaksError("Reading the audio took too long.")
            data = proc.stdout.read(_CHUNK_BYTES)
            if not data:
                break
            data = leftover + data
            usable = len(data) - len(data) % 4
            leftover = data[usable:]
            block = array.array("f")
            block.frombytes(data[:usable])
            if sys.byteorder == "big":
                block.byteswap()
            if len(block):
                yield block
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()


def compute_peaks(path: Path, buckets: int) -> dict[str, Any]:
    """``{peaks, durationS}`` for a media file; ``peaks`` is empty when the file has no audio stream."""
    buckets = clamp_buckets(buckets)
    exe = ffmpeg_bin.ffmpeg_exe()
    try:
        info = ffmpeg_bin.probe(path)
    except ValueError as exc:
        raise PeaksError(str(exc)) from exc
    if not info["has_audio"]:
        return {"peaks": [], "durationS": round(float(info["duration"]), 4)}
    duration = min(float(info["duration"]), MAX_SECONDS)
    total = max(1, int(round(duration * SAMPLE_RATE)))
    peaks = fold_samples(_decode_chunks(exe, path), total, buckets)
    top = max(peaks) if peaks else 0.0
    scale = 1.0 / top if top > 1e-6 else 0.0
    return {"peaks": [round(min(1.0, p * scale), 4) for p in peaks], "durationS": round(duration, 4)}


def peaks_for(folder: Path, path: Path, buckets: int) -> dict[str, Any]:
    """Cached peaks for ``path`` (already resolved inside ``folder``)."""
    buckets = clamp_buckets(buckets)
    cache_dir = folder / "cache" / "peaks"
    cache_file = cache_dir / f"{_file_hash(path)}-{buckets}.json"
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if isinstance(cached.get("peaks"), list) and isinstance(cached.get("durationS"), (int, float)):
            return {"peaks": cached["peaks"], "durationS": cached["durationS"]}
    except (OSError, ValueError):
        pass
    result = compute_peaks(path, buckets)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_name(cache_file.name + ".part")
        tmp.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
        tmp.replace(cache_file)
    except OSError:
        pass  # a read-only cache only costs a re-decode next time
    return result
