"""Room after the vocal in a lip-sync shot (FU-609): what to do when a hand-attached vocal ends on a sung syllable.

Measured (``docs/research/lip-sync-singing-experiments-2026-11.md``, follow-up of 2026-11-23): a sung line cut to end
on its last syllable makes the model keep the mouth shut more and leaves the sync score nothing to confirm (0 of 11
takes beat chance at 4 s); the same line with 0.7 s of silence after it was usable. So when the shot's vocal is still
sounding within ``NEAR_END`` of its end, the model gets about ``ROOM`` more of silence (whole 8k+1 frame steps, 0.3 to
0.7 s, never past the 8 s limit), the takes are made and scored on that longer window, and each finished clip is cut
back to the frames that were asked for, with the vocal as asked as its sound. Song to shots gives its slices this room
itself and asks the film to cut it off (``storyboard.song_vocals``), so it is not added twice.
"""

from __future__ import annotations

import wave
from pathlib import Path

NEAR_END = 0.2  # a vocal still sounding this close to the shot's end ends on a syllable
ROOM = 0.5
ROOM_MIN, ROOM_MAX = 0.3, 0.7


def ends_voiced(wav: Path, near_end: float = NEAR_END) -> bool:
    """True when the 16-bit WAV (as ``lipsync_runtime.prepare_audio`` writes it) sounds within ``near_end`` of its
    end, by the same voiced measure as song to shots (``song_vocals.voiced_frames``). False when it cannot be read."""
    import numpy as np

    from backend_service.storyboard.song_vocals import _FRAME_SECONDS, voiced_frames

    try:
        with wave.open(str(wav), "rb") as handle:
            channels, rate, width = handle.getnchannels(), handle.getframerate(), handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
    except (OSError, EOFError, wave.Error):
        return False  # an unreadable clip gets no room; preparing it has already said what is wrong
    if width != 2 or not raw:
        return False
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, channels).mean(axis=1) / 32768.0
    voiced = voiced_frames(samples, rate)
    sounding = np.flatnonzero(voiced)
    if not sounding.size:
        return False
    return (sounding[-1] + 1) * _FRAME_SECONDS >= samples.size / rate - near_end - 1e-9


def room_frames(frames: int, fps: int, max_frames: int) -> int | None:
    """The 8k+1 frame count to make instead of ``frames`` so about ``ROOM`` of silence follows (between ``ROOM_MIN``
    and ``ROOM_MAX``, the nearer to ``ROOM``, the longer on a tie), or None when no step fits under ``max_frames``."""
    choices = [frames + 8 * step for step in range(1, int(ROOM_MAX * fps + 1e-6) // 8 + 1)
               if 8 * step >= ROOM_MIN * fps - 1e-6 and frames + 8 * step <= max_frames]
    if not choices:
        return None
    return min(choices, key=lambda count: (abs((count - frames) - ROOM * fps), -count))


def trim_command(exe: str, src: Path, dst: Path, frames: int, codec: list[str]) -> list[str]:
    """ffmpeg arguments that keep the first ``frames`` of the silent take ``src`` (re-encoded with ``codec``: cutting
    a stream copy by frame count can drop a reordered frame)."""
    from backend_service.storyboard.ffmpeg_bin import PROTOCOLS

    return [
        exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-protocol_whitelist", PROTOCOLS,
        "-i", str(src), "-map", "0:v:0", "-frames:v", str(int(frames)), *codec, "-an", str(dst),
    ]


def trim(exe: str, src: Path, dst: Path, frames: int, width: int, height: int, fps: int) -> None:
    import os
    import subprocess

    from backend_service.storyboard import encoders, ffmpeg_bin

    encoder = ffmpeg_bin._h264_for(exe, os.environ.get("CHAOSENGINE_VIDEO_ENCODER", "").strip())  # this binary's
    if encoder is None:
        raise RuntimeError("This ffmpeg has no usable H.264 encoder to cut the room after the vocal off the clip.")
    codec = encoders.codec_args(encoder, False, width=width, height=height, fps=fps)
    run = subprocess.run(trim_command(exe, src, dst, frames, codec), capture_output=True, text=True, timeout=600)
    if run.returncode != 0 or not dst.is_file():
        last = (run.stderr or "").strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"ffmpeg could not cut the room after the vocal off the clip: {last[0][:200]}")


def note(room: int, fps: int) -> str:
    return (f"the vocal ends on a syllable, so the model and the sync score got {room / fps:.2f} s of silence after "
            f"it; that room is cut off the clip")

