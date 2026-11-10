"""Put the sound a video pipeline made (LTX-2) into the mp4 it made the picture for.

The diffusers export helper for LTX-2 needs PyAV, which the app does not ship; ffmpeg is already there for the
Storyboard, so the finished silent mp4 and the waveform go through one list-form ffmpeg call (video copied, sound as
AAC). Nothing here touches torch: the waveform may be a tensor or an array.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

_MAX_SECONDS_FOR_FFMPEG = 600


def waveform_to_pcm16(waveform: Any) -> tuple[bytes, int]:
    """``(interleaved 16-bit little-endian samples, channels)`` for a waveform of any of the usual shapes.

    Accepts ``(batch, channels, samples)``, ``(channels, samples)``, ``(samples, channels)`` (two or fewer columns)
    and ``(samples,)``; torch tensors are moved to the CPU first. Values are clipped to [-1, 1].
    """
    import numpy as np  # noqa: PLC0415

    if hasattr(waveform, "detach"):
        waveform = waveform.detach().float().cpu().numpy()
    data = np.asarray(waveform, dtype=np.float32)
    while data.ndim > 2:
        data = data[0]
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] > 2 and data.shape[1] <= 2:
        data = data.T  # (samples, channels) -> (channels, samples)
    channels = int(data.shape[0])
    interleaved = np.clip(data.T, -1.0, 1.0)
    return (interleaved * 32767.0).astype("<i2").tobytes(), channels


def mux_audio_into_mp4(mp4: bytes, waveform: Any, sample_rate: int) -> bytes:
    """The mp4 with ``waveform`` as its AAC sound track. Raises ``RuntimeError`` with ffmpeg's last line on failure."""
    from backend_service.storyboard import ffmpeg_bin  # noqa: PLC0415

    found = ffmpeg_bin.find_ffmpeg(allow_path=True)
    if found is None:
        raise RuntimeError("ffmpeg was not found, so the sound could not be added to the clip.")
    pcm, channels = waveform_to_pcm16(waveform)
    if not pcm or sample_rate <= 0:
        raise RuntimeError("the pipeline returned no usable sound.")
    with tempfile.TemporaryDirectory(prefix="chaosengine-mux-") as tmp:
        folder = Path(tmp)
        (folder / "in.mp4").write_bytes(mp4)
        (folder / "in.pcm").write_bytes(pcm)
        command = [
            found[0], "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(folder / "in.mp4"),
            "-f", "s16le", "-ar", str(int(sample_rate)), "-ac", str(channels), "-i", str(folder / "in.pcm"),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(folder / "out.mp4"),
        ]
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=_MAX_SECONDS_FOR_FFMPEG)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"ffmpeg could not add the sound: {type(exc).__name__}") from exc
        if run.returncode != 0 or not (folder / "out.mp4").is_file():
            last = (run.stderr or "").strip().splitlines()[-1:] or ["no output"]
            raise RuntimeError(f"ffmpeg could not add the sound: {last[0][:200]}")
        return (folder / "out.mp4").read_bytes()
