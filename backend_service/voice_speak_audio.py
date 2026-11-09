"""Audio steps for spoken text: WAV facts, joining chunks, lifting quiet speech, speed and pitch with ffmpeg.

Split out of ``voice_speak`` (which re-exports every name here). Works on 16-bit PCM WAV bytes at the file's own
rate, so 24 kHz and VoxCPM2's 48 kHz need nothing special.
"""

from __future__ import annotations

import array
import io
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

SENTENCE_GAP_S = 0.15
# A model copies the loudness of what it is given: a cloned voice from a quiet sample (peak near -28 dBFS on the
# 2026-11-08 run) speaks just as quietly. Speech whose peak is below QUIET_PEAK is lifted to LIFT_PEAK, by at most
# MAX_LIFT times; louder speech is left alone.
QUIET_PEAK = 0.2
LIFT_PEAK = 0.5
MAX_LIFT = 12.0

_NO_FFMPEG = "Changing speed or pitch for this voice needs ffmpeg. Install it from Settings, or leave both at their defaults."


def wav_info(data: bytes) -> tuple[int, int, int, float]:
    """``(channels, sample width, sample rate, seconds)`` of a PCM WAV."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        rate = handle.getframerate()
        return handle.getnchannels(), handle.getsampwidth(), rate, handle.getnframes() / float(rate or 1)


def join_wavs(parts: list[bytes], gap_s: float = SENTENCE_GAP_S) -> bytes:
    """One WAV from several with the same format, a short silence between them."""
    if len(parts) == 1:
        return parts[0]
    frames: list[bytes] = []
    params = None
    for data in parts:
        with wave.open(io.BytesIO(data), "rb") as handle:
            current = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
            if params is None:
                params = current
            elif current != params:
                raise ValueError("The speech chunks came back in different audio formats.")
            frames.append(handle.readframes(handle.getnframes()))
    assert params is not None
    silence = b"\x00" * (int(gap_s * params[2]) * params[0] * params[1])
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(params[0])
        handle.setsampwidth(params[1])
        handle.setframerate(params[2])
        handle.writeframes(silence.join(frames))
    return out.getvalue()


def lift_quiet(wav: bytes) -> bytes:
    """Raise very quiet 16-bit speech to a normal level (see ``QUIET_PEAK``); anything else is returned as is."""
    with wave.open(io.BytesIO(wav), "rb") as handle:
        params, frames = handle.getparams(), handle.readframes(handle.getnframes())
    if params.sampwidth != 2 or not frames:
        return wav
    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    peak = max(max(samples), -min(samples)) / 32768.0
    if peak <= 0.0 or peak >= QUIET_PEAK:
        return wav
    gain = min(LIFT_PEAK / peak, MAX_LIFT)
    lifted = array.array("h", (max(-32768, min(32767, int(v * gain))) for v in samples))
    if sys.byteorder == "big":
        lifted.byteswap()
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setparams(params)
        handle.writeframes(lifted.tobytes())
    return out.getvalue()


def atempo_chain(factor: float) -> list[str]:
    """``atempo`` filters whose product is ``factor``; each one is limited to 0.5 to 2.0."""
    if factor <= 0:
        raise ValueError("Tempo must be above zero.")
    filters: list[str] = []
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        filters.append("atempo=0.5")
        factor /= 0.5
    if abs(factor - 1.0) > 1e-6:
        filters.append(f"atempo={factor:.6f}")
    return filters


def filter_graph(sample_rate: int, speed: float, pitch: float) -> str:
    """The ffmpeg audio filter for a speed factor and a pitch shift in semitones; empty when both are neutral."""
    filters: list[str] = []
    if abs(pitch) > 1e-6:
        ratio = 2.0 ** (pitch / 12.0)
        filters.append(f"asetrate={sample_rate * ratio:.3f}")
        filters.append(f"aresample={sample_rate}")
        filters.extend(atempo_chain(1.0 / ratio))
    filters.extend(atempo_chain(speed))
    return ",".join(filters)


def process_audio(wav: bytes, speed: float = 1.0, pitch: float = 0.0) -> bytes:
    """Speed (pitch kept) and pitch (tempo kept) with ffmpeg. 1.0 and 0 return the audio untouched."""
    _channels, _width, rate, _seconds = wav_info(wav)
    graph = filter_graph(rate, speed, pitch)
    if not graph:
        return wav
    from backend_service.storyboard import ffmpeg_bin  # noqa: PLC0415

    exe = ffmpeg_bin.optional_ffmpeg_exe()
    if exe is None:
        raise RuntimeError(_NO_FFMPEG)
    with tempfile.TemporaryDirectory(prefix="chaosengine-speak-") as folder:
        source, target = Path(folder) / "in.wav", Path(folder) / "out.wav"
        source.write_bytes(wav)
        result = subprocess.run(
            [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
             "-af", graph, "-c:a", "pcm_s16le", str(target)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, check=False,
        )
        if result.returncode != 0 or not target.is_file():
            raise RuntimeError(f"ffmpeg could not change the speed or pitch: {result.stderr.strip()[-300:]}")
        return target.read_bytes()
