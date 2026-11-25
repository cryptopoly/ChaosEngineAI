"""Shared fakes for the video upscale tests: a pixel-doubling upscaler and tiny generated clips (ffmpeg lavfi, CPU)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


class DoublePixels:
    """Repeats every pixel twice across and down: a 2x 'model' with a known answer."""

    scale = 2

    def __init__(self) -> None:
        self.calls = 0
        self.frames = 0

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.frames += int(frames.shape[0])
        return frames.repeat(2, axis=1).repeat(2, axis=2)


def find_ffmpeg() -> str | None:
    try:
        from backend_service.storyboard import ffmpeg_bin

        return ffmpeg_bin.ffmpeg_exe()
    except Exception:  # noqa: BLE001 - absent binary: the tests skip
        return None


def make_clip(exe: str, path: Path, *, seconds: float = 2.0, fps: int = 10, size: str = "96x64",
              audio: bool = True, extra: list[str] | None = None) -> Path:
    """A tiny test clip: testsrc picture (and a 440 Hz tone), H.264 when the build has libx264, else mpeg4."""
    args = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
    codec = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    run = subprocess.run([*args, *codec, *(extra or []), *(["-c:a", "aac"] if audio else []), str(path)],
                         capture_output=True, text=True)
    if run.returncode != 0:
        run = subprocess.run([*args, "-c:v", "mpeg4", "-q:v", "3", *(extra or []), *(["-c:a", "aac"] if audio else []),
                              str(path)], capture_output=True, text=True, check=True)
    return path


def count_frames(exe: str, path: Path) -> tuple[int, int, int]:
    """(frames, width, height) of the first video stream, decoded."""
    run = subprocess.run([exe, "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True)
    text = run.stderr.decode("utf-8", "replace")
    import re

    size = re.findall(r"Video: rawvideo.*?(\d{2,5})x(\d{2,5})", text)
    width, height = (int(size[-1][0]), int(size[-1][1])) if size else (0, 0)
    frames = len(run.stdout) // (width * height * 3) if width else 0
    return frames, width, height
