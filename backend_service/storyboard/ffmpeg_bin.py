"""The pinned ffmpeg binary for the Storyboard, its filter list, its version, and a stderr-based probe.

Storyboard renders always use the ffmpeg shipped by ``imageio-ffmpeg`` (a known build with known
filters), never one found on ``PATH``. That build has no ``ffprobe``, so ``probe`` runs
``ffmpeg -i <file>`` and reads the stream summary ffmpeg prints on stderr.
"""

from __future__ import annotations

import functools
import re
import subprocess
from pathlib import Path
from typing import Any

PROTOCOLS = "file,pipe"

_INSTALL_HINT = (
    "The bundled ffmpeg is missing. Install the 'ffmpeg' extra (pip install 'chaosengine-ai[ffmpeg]', "
    "which provides imageio-ffmpeg) and try again."
)


def ffmpeg_exe() -> str:
    """Path to the pinned imageio-ffmpeg binary. Raises RuntimeError with an install hint when absent."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    try:
        exe = str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # imageio-ffmpeg raises RuntimeError when its binary is not on disk
        raise RuntimeError(_INSTALL_HINT) from exc
    if not Path(exe).is_file():
        raise RuntimeError(_INSTALL_HINT)
    return exe


_FILTER_LINE = re.compile(r"^\s*[T.][S.][C.]\s+(\w+)\s+\S*->\S*")


@functools.lru_cache(maxsize=4)
def _filters_for(exe: str) -> frozenset[str]:
    result = subprocess.run(
        [exe, "-hide_banner", "-nostdin", "-filters"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    names = set()
    for line in result.stdout.splitlines():
        match = _FILTER_LINE.match(line)
        if match:
            names.add(match.group(1))
    return frozenset(names)


def ffmpeg_filters() -> frozenset[str]:
    """Names of every filter the pinned binary was built with (cached per binary)."""
    return _filters_for(ffmpeg_exe())


@functools.lru_cache(maxsize=4)
def _version_for(exe: str) -> str:
    result = subprocess.run(
        [exe, "-hide_banner", "-nostdin", "-version"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    match = re.search(r"ffmpeg version (\S+)", result.stdout)
    return match.group(1) if match else "unknown"


def ffmpeg_version() -> str:
    return _version_for(ffmpeg_exe())


def missing_filters(required: set[str] | frozenset[str]) -> list[str]:
    return sorted(set(required) - ffmpeg_filters())


_DURATION = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_STREAM = re.compile(r"^\s*Stream #\d+:\d+.*?: (Video|Audio): (.*)$", re.MULTILINE)
_SIZE = re.compile(r"(?:^|[ ,])(\d{2,5})x(\d{2,5})(?:[ ,\[]|$)")
_FPS = re.compile(r"(\d+(?:\.\d+)?) fps")


def parse_probe(stderr: str) -> dict[str, Any]:
    """Read duration, audio presence, size and fps from the text ``ffmpeg -i`` prints."""
    match = _DURATION.search(stderr)
    if not match:
        raise ValueError("ffmpeg reported no duration for this file")
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    info: dict[str, Any] = {"duration": duration, "has_audio": False, "has_video": False,
                            "width": None, "height": None, "fps": None}
    for kind, rest in _STREAM.findall(stderr):
        if kind == "Audio":
            info["has_audio"] = True
        elif kind == "Video" and not info["has_video"]:
            # Cover art in an audio file shows up as an attached-picture video stream; it is not picture.
            if "(attached pic)" in rest:
                continue
            info["has_video"] = True
            size = _SIZE.search(rest)
            if size:
                info["width"], info["height"] = int(size.group(1)), int(size.group(2))
            fps = _FPS.search(rest)
            if fps:
                info["fps"] = float(fps.group(1))
    return info


def probe(path: str | Path) -> dict[str, Any]:
    """``{duration, has_audio, has_video, width, height, fps}`` for one local media file."""
    exe = ffmpeg_exe()
    result = subprocess.run(
        [exe, "-hide_banner", "-nostdin", "-protocol_whitelist", PROTOCOLS, "-i", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    # With no output file ffmpeg always exits non-zero; the summary on stderr is what we want.
    try:
        return parse_probe(result.stderr)
    except ValueError as exc:
        tail = result.stderr.strip().splitlines()[-1:] or ["no output"]
        raise ValueError(f"Could not read media file {Path(path).name}: {tail[0]}") from exc
