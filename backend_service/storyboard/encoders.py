"""H.264 encoder choice for the Storyboard: pure parsing and ordering, plus the quality presets.

The downloaded LGPL ffmpeg and imageio-ffmpeg differ: the second ships ``libx264``, the first does not.
``choose_h264`` takes the text of ``ffmpeg -encoders`` and picks libx264 when it is there, else the first
platform encoder that is listed (and, when a ``trial`` is given, actually works: a GPU encoder can be
listed on a machine with no GPU). ``codec_args`` gives the ffmpeg arguments for one encoder, for the
``final`` or the ``preview`` quality.
"""

from __future__ import annotations

import re
from typing import Callable

LIBX264 = "libx264"

# After libx264 (software, best quality per byte) these are tried in order. libopenh264 is last
# everywhere: it works on any CPU but needs a bitrate, not a quality level. VAAPI is not offered
# (it needs a device and an upload filter).
PLATFORM_ORDER: dict[str, tuple[str, ...]] = {
    "darwin": ("h264_videotoolbox", "libopenh264"),
    "win32": ("h264_nvenc", "h264_amf", "h264_qsv", "libopenh264"),
    "linux": ("h264_nvenc", "h264_qsv", "libopenh264"),
}

_ENCODER_LINE = re.compile(r"^\s*V[F.][S.][X.][B.][D.]\s+([A-Za-z0-9]\S*)\s")


def parse_encoders(text: str) -> frozenset[str]:
    """Names of the video encoders listed by ``ffmpeg -hide_banner -encoders``."""
    return frozenset(m.group(1) for line in text.splitlines() if (m := _ENCODER_LINE.match(line)))


def platform_key(sys_platform: str) -> str:
    if sys_platform.startswith("win"):
        return "win32"
    if sys_platform.startswith("darwin"):
        return "darwin"
    return "linux"


def candidates(available: frozenset[str] | set[str], sys_platform: str) -> list[str]:
    """Encoders to try, best first: libx264, then the platform order, each only if listed."""
    order = (LIBX264, *PLATFORM_ORDER[platform_key(sys_platform)])
    return [name for name in order if name in available]


def choose_h264(
    available: frozenset[str] | set[str],
    sys_platform: str,
    trial: Callable[[str], bool] | None = None,
    override: str | None = None,
) -> str | None:
    """The encoder to use, or None when the build has no H.264 encoder at all.

    ``override`` (an encoder name) wins when it is listed and passes the trial. Software libx264 is
    never trialled; hardware encoders are, when ``trial`` is given.
    """
    if override and override != "auto" and override in available:
        if override == LIBX264 or trial is None or trial(override):
            return override
    for name in candidates(available, sys_platform):
        if name == LIBX264 or trial is None or trial(name):
            return name
    return None


def codec_args(
    encoder: str,
    preview: bool,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
) -> list[str]:
    """ffmpeg arguments for ``encoder`` at final or preview quality (always yuv420p)."""
    pix = ["-pix_fmt", "yuv420p"]
    if encoder == LIBX264:
        return ["-c:v", encoder, *(["-preset", "ultrafast", "-crf", "28"] if preview
                                   else ["-preset", "medium", "-crf", "18"]), *pix]
    if encoder == "h264_videotoolbox":
        return ["-c:v", encoder, *(["-q:v", "45", "-realtime", "1"] if preview else ["-q:v", "65"]), *pix]
    if encoder == "h264_nvenc":
        return ["-c:v", encoder, *(["-preset", "p1", "-rc", "vbr", "-cq", "28", "-b:v", "0"] if preview
                                   else ["-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]), *pix]
    if encoder == "h264_amf":
        return ["-c:v", encoder, *(["-quality", "speed", "-rc", "cqp", "-qp_i", "28", "-qp_p", "30"] if preview
                                   else ["-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20"]), *pix]
    if encoder == "h264_qsv":
        return ["-c:v", encoder, "-global_quality", "30" if preview else "20", *pix]
    if encoder == "libopenh264":
        return ["-c:v", encoder, "-b:v", _openh264_bitrate(preview, width, height, fps),
                "-allow_skip_frames", "0", *pix]
    raise ValueError(f"No quality preset for encoder '{encoder}'")


def _openh264_bitrate(preview: bool, width: int | None, height: int | None, fps: int | None) -> str:
    """Bits per pixel per frame (0.12 final, 0.06 preview) times the frame rate, kept within 500k to 40M."""
    pixels = (width or 1920) * (height or 1080)
    bits = pixels * (fps or 30) * (0.06 if preview else 0.12)
    return f"{int(min(max(bits, 500_000), 40_000_000))}"
