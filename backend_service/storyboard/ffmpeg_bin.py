"""The one ffmpeg lookup, its filter list, its version, and a stderr-based probe.

Lookup order (``find_ffmpeg``): ``CHAOSENGINE_FFMPEG`` (dev), a file the user chose in Setup, the app-managed
download (``~/.chaosengine/bin/ffmpeg/<version>/``), then ``imageio-ffmpeg`` from the Setup install. A
program on ``PATH`` is used only when a caller asks for it (``allow_path``); Storyboard renders never do,
because they depend on a known build with known filters. When nothing is found ``ffmpeg_exe`` raises
``FfmpegMissing`` and the routes answer with a code that the app turns into its "Video tools are needed" prompt.
The ltx-2-mlx venv keeps its own libx264 ffmpeg and never goes through here.

``imageio-ffmpeg`` ships no ``ffprobe``, so ``probe`` runs ``ffmpeg -i <file>`` and reads the stream summary
ffmpeg prints on stderr.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from backend_service.storyboard import encoders

PROTOCOLS = "file,pipe"
INSTALL_ACTION = "install-ffmpeg"

_INSTALL_HINT = (
    "Video tools are needed. Open Setup and choose Install video tools (or choose an ffmpeg file), then try again."
)


class FfmpegMissing(RuntimeError):
    """No usable ffmpeg. ``action`` tells the app which prompt to show."""

    action = INSTALL_ACTION

    def __init__(self, message: str = _INSTALL_HINT) -> None:
        super().__init__(message)


def managed_dir() -> Path:
    """Where the app keeps its ffmpeg: ``<dir>/<version>/ffmpeg``, the ``current`` pointer, ``custom`` choice."""
    override = os.environ.get("CHAOSENGINE_FFMPEG_DIR")
    return Path(override) if override else Path.home() / ".chaosengine" / "bin" / "ffmpeg"


def binary_name() -> str:
    return "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"


def downloaded_exe() -> Path | None:
    """The app-managed download named by the ``current`` pointer, when its file is there."""
    root = managed_dir()
    try:
        version = (root / "current").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not version or "/" in version or "\\" in version or version.startswith("."):
        return None
    exe = root / version / binary_name()
    return exe if exe.is_file() else None


def custom_exe() -> Path | None:
    """An ffmpeg file the user chose in Setup (``custom`` holds its path)."""
    try:
        text = (managed_dir() / "custom").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    path = Path(text) if text else None
    return path if path is not None and path.is_file() else None


def _imageio_exe() -> str | None:
    try:
        import imageio_ffmpeg

        exe = str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # not installed, or its binary is not on disk
        return None
    return exe if Path(exe).is_file() else None


def find_ffmpeg(*, allow_path: bool = False) -> tuple[str, str] | None:
    """``(path, source)`` of the first ffmpeg found, or None. Source: env, custom, download, imageio, path."""
    env = os.environ.get("CHAOSENGINE_FFMPEG")
    if env and Path(env).is_file():
        return env, "env"
    custom = custom_exe()
    if custom is not None:
        return str(custom), "custom"
    downloaded = downloaded_exe()
    if downloaded is not None:
        return str(downloaded), "download"
    imageio = _imageio_exe()
    if imageio:
        return imageio, "imageio"
    if allow_path:
        found = shutil.which("ffmpeg")
        if found:
            return found, "path"
    return None


def optional_ffmpeg_exe(*, allow_path: bool = True) -> str | None:
    """The ffmpeg path or None, for callers that can do without (metadata tagging, last-frame grabs)."""
    found = find_ffmpeg(allow_path=allow_path)
    return found[0] if found else None


def ffmpeg_exe() -> str:
    """Path to the ffmpeg Storyboard renders use (never PATH). Raises ``FfmpegMissing`` when absent."""
    found = find_ffmpeg()
    if found is None:
        raise FfmpegMissing()
    return found[0]


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


# ---- encoder choice -------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _encoders_for(exe: str) -> frozenset[str]:
    result = subprocess.run(
        [exe, "-hide_banner", "-nostdin", "-encoders"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    return encoders.parse_encoders(result.stdout)


def ffmpeg_encoders() -> frozenset[str]:
    return _encoders_for(ffmpeg_exe())


def _trial_encode(exe: str, encoder: str) -> bool:
    """One tiny frame through ``encoder``: a GPU encoder can be listed on a machine without the GPU."""
    with tempfile.TemporaryDirectory() as folder:
        result = subprocess.run(
            [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=c=black:s=256x256:r=10:d=0.2", "-frames:v", "1",
             *encoders.codec_args(encoder, False, width=256, height=256, fps=10),
             "-f", "null", "-"],
            capture_output=True, text=True, errors="replace", timeout=60, cwd=folder,
        )
    return result.returncode == 0


@functools.lru_cache(maxsize=4)
def _h264_for(exe: str, override: str) -> str | None:
    return encoders.choose_h264(_encoders_for(exe), sys.platform, lambda name: _trial_encode(exe, name),
                                override or None)


def h264_encoder() -> str:
    """The H.264 encoder for this binary (libx264 when present, else a platform one). Cached per binary.

    ``CHAOSENGINE_VIDEO_ENCODER`` (an encoder name, default ``auto``) overrides the choice.
    """
    exe = ffmpeg_exe()
    name = _h264_for(exe, os.environ.get("CHAOSENGINE_VIDEO_ENCODER", "").strip())
    if name is None:
        raise RuntimeError(
            "This ffmpeg has no usable H.264 encoder. Install video tools from Setup (the imageio-ffmpeg build "
            "includes libx264), or choose a different ffmpeg file."
        )
    return name


def h264_args(preview: bool, *, width: int | None = None, height: int | None = None,
              fps: int | None = None) -> list[str]:
    return encoders.codec_args(h264_encoder(), preview, width=width, height=height, fps=fps)


def ffmpeg_info() -> dict[str, Any]:
    """Status for the Setup row: where ffmpeg came from, its version, licence line, and H.264 encoder."""
    found = find_ffmpeg()
    if found is None:
        return {"installed": False, "path": None, "source": None, "version": None, "licence": None, "encoder": None}
    exe, source = found
    info: dict[str, Any] = {"installed": True, "path": exe, "source": source, "version": _version_for(exe),
                            "licence": _licence_for(exe), "encoder": None}
    try:
        info["encoder"] = _h264_for(exe, os.environ.get("CHAOSENGINE_VIDEO_ENCODER", "").strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


@functools.lru_cache(maxsize=4)
def _licence_for(exe: str) -> str | None:
    """"LGPL" or "GPL" from ``-L`` (the licence text ffmpeg prints), None when it cannot be read."""
    try:
        result = subprocess.run([exe, "-hide_banner", "-nostdin", "-L"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    text = result.stdout
    if "Lesser General Public" in text[:400]:
        return "LGPL"
    if "General Public License" in text[:400]:
        return "GPL"
    return None
