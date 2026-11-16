"""The song a music video is cut to (S-M1): check the file, decode it, and estimate tempo, beats and sections.

Only the bundled ffmpeg and numpy (both ship with the backend; numpy is imported inside the calls) are used: no
audio library. ffmpeg decodes the file to mono 11025 Hz float32 on a pipe. From short frames the analysis takes a
spectral-flux onset curve (how much new sound starts in each frame), loudness, and the share of energy in four
bands. Tempo is the onset curve's autocorrelation peak between 60 and 200 BPM (weighted towards 120 so a half or
double tempo is not picked for no reason); beats come from dynamic-programming beat tracking on that tempo.
Sections are the peaks of a novelty curve over a self-similarity matrix of half-second loudness and band features,
snapped to the nearest beat; parts that sound alike share a letter, a quiet short first or last part is
``intro`` / ``outro``. All of it is an estimate and the result says so (``rough``, ``note``).

The source is a file the user named or a Music library item, so it goes through ``media.resolve_source`` and its
first bytes must be a WAV, FLAC, MP3, OGG or MP4 audio header (an extension alone is not trusted).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from backend_service.storyboard import ffmpeg_bin, media
from backend_service.storyboard.models import MAX_BEATS, MAX_SECTIONS, MAX_SONG_SECONDS

SAMPLE_RATE = 11025
N_FFT = 1024
HOP = 256
FPS = SAMPLE_RATE / HOP  # onset frames per second (about 43)
MIN_SONG_SECONDS = 5.0
BLOCK_SECONDS = 0.5
MIN_SECTION_SECONDS = 8.0
MIN_ONSET_SHARE = 0.02  # the strongest onset must add this share of a frame's sound to count as a hit
DECODE_TIMEOUT_S = 300
ROUGH_NOTE = "Tempo, beats and sections are estimated from onsets and loudness; check them by ear and edit the sections if needed."
NO_BEAT_NOTE = "No steady beat was found, so there are no beats; cuts follow the sections only."
HINT_NOTE = "The tempo is the one the Music library track was made at; beats are placed on it."


class SongError(ValueError):
    """A readable refusal (not audio, too short or long, unreadable)."""


def looks_like_audio(path: Path) -> bool:
    """True when the first bytes are a WAV, FLAC, MP3 (ID3 or a frame sync), OGG or MP4 audio header."""
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError:
        return False
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return True
    if head[:4] in (b"fLaC", b"OggS") or head[:3] == b"ID3":
        return True
    if head[4:8] == b"ftyp":  # .m4a / .aac in an MP4 box
        return True
    return len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0  # MP3 frame or ADTS AAC sync


def check_source(raw: str) -> Path:
    """The real path of an audio file to attach, or ``SongError`` (path rules of ``media.resolve_source``, then bytes)."""
    try:
        path = media.resolve_source(raw, {"audio"})
    except media.MediaError as exc:
        raise SongError(str(exc)) from exc
    if not looks_like_audio(path):
        raise SongError(f"Not an audio file (WAV, FLAC, MP3, OGG or M4A) despite its extension: {path.name}")
    return path


def library_source(item_id: str) -> tuple[Path, dict[str, Any]]:
    """The audio file and metadata of a Music library track, or ``SongError``."""
    from backend_service import music_library

    if not music_library.is_valid_id(item_id):
        raise SongError("Unknown Music library item.")
    meta = music_library.get(item_id)
    path = music_library.audio_path(item_id)
    if meta is None or path is None:
        raise SongError("That Music library item is gone or has no audio file.")
    return check_source(str(path)), meta


def lyric_lines(text: str) -> list[str]:
    """Lyrics as caption lines: one per line, structure tags such as ``[verse]`` and blank lines dropped."""
    lines = []
    for raw in str(text or "").splitlines():
        line = " ".join(raw.split())
        if not line or (line.startswith("[") and line.endswith("]")) or (line.startswith("(") and line.endswith(")")):
            continue
        lines.append(line[:200])
    return lines


# ---- decoding ----
def decode(path: Path) -> Any:
    """Mono float32 samples at ``SAMPLE_RATE`` (numpy array). Raises ``SongError`` or ``FfmpegMissing``."""
    import numpy as np

    exe = ffmpeg_bin.ffmpeg_exe()
    try:
        info = ffmpeg_bin.probe(path)
    except ValueError as exc:
        raise SongError(str(exc)) from exc
    if not info.get("has_audio"):
        raise SongError("The file has no sound.")
    duration = float(info.get("duration") or 0.0)
    if duration > MAX_SONG_SECONDS + 1:
        raise SongError(f"The song is {duration / 60:.0f} minutes long; a project takes songs up to {MAX_SONG_SECONDS / 60:.0f} minutes.")
    args = [exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-protocol_whitelist", ffmpeg_bin.PROTOCOLS,
            "-i", str(path), "-vn", "-map", "0:a:0", "-t", str(MAX_SONG_SECONDS), "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "f32le", "pipe:1"]
    try:
        result = subprocess.run(args, capture_output=True, timeout=DECODE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise SongError("Reading the song took too long.") from exc
    except OSError as exc:
        raise SongError("ffmpeg could not be started to read the song.") from exc
    if result.returncode != 0:
        raise SongError("ffmpeg could not read the song.")
    data = result.stdout[: len(result.stdout) - len(result.stdout) % 4]
    samples = np.frombuffer(data, dtype="<f4").astype(np.float32)
    if samples.size < MIN_SONG_SECONDS * SAMPLE_RATE:
        raise SongError(f"The song is shorter than {MIN_SONG_SECONDS:.0f} seconds.")
    return samples


# ---- features ----
def features(samples: Any) -> dict[str, Any]:
    """Per frame: onset strength (spectral flux), loudness (rms) and the energy share of four bands."""
    import numpy as np

    n_frames = 1 + max(0, (samples.size - N_FFT) // HOP)
    window = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    edges = [0, 200, 800, 3000, SAMPLE_RATE / 2 + 1]
    band_bins = [(freqs >= lo) & (freqs < hi) for lo, hi in zip(edges, edges[1:])]
    flux = np.zeros(n_frames, dtype=np.float32)
    rms = np.zeros(n_frames, dtype=np.float32)
    bands = np.zeros((n_frames, 4), dtype=np.float32)
    prev = None
    level = 0.0  # summed log magnitude per frame, so onsets are judged against how full the sound is
    chunk = 2048
    for first in range(0, n_frames, chunk):
        count = min(chunk, n_frames - first)
        idx = (first + np.arange(count))[:, None] * HOP + np.arange(N_FFT)[None, :]
        frames = samples[idx] * window
        mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
        logmag = np.log1p(100.0 * mag)
        start = logmag[:1] if prev is None else prev
        diff = np.diff(np.vstack([start, logmag]), axis=0)
        flux[first:first + count] = np.maximum(diff, 0).sum(axis=1)
        prev = logmag[-1:]
        level += float(logmag.sum())
        rms[first:first + count] = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        power = mag * mag
        for b, mask in enumerate(band_bins):
            bands[first:first + count, b] = power[:, mask].sum(axis=1)
    smooth = max(1, int(round(0.5 * FPS)))
    local = np.convolve(flux, np.ones(smooth) / smooth, mode="same")
    onset = np.maximum(flux - local, 0)
    top = float(onset.max()) if onset.size else 0.0
    if top < MIN_ONSET_SHARE * level / max(1, n_frames):
        onset = np.zeros_like(onset)  # only the ripple of steady tones: nothing starts, so there is no beat
    elif top > 0:
        onset = onset / top
    return {"onset": onset, "rms": rms, "bands": bands}


def tempo(onset: Any, hint: float | None = None) -> tuple[float | None, float]:
    """(bpm, strength 0..1) from the onset autocorrelation; ``hint`` (a known tempo) is checked, not trusted blindly."""
    import numpy as np

    if hint is not None and 30 <= hint <= 400:
        return float(hint), 1.0
    x = onset - onset.mean()
    if not np.any(x):
        return None, 0.0
    size = 1 << int(np.ceil(np.log2(2 * x.size)))
    spec = np.fft.rfft(x, size)
    ac = np.fft.irfft(spec * np.conj(spec), size)[: x.size]
    if ac[0] <= 0:
        return None, 0.0
    ac = ac / ac[0]
    lo, hi = int(60 * FPS / 200), int(np.ceil(60 * FPS / 60))
    if hi + 2 >= ac.size:
        return None, 0.0
    lags = np.arange(lo, hi + 1)
    bpms = 60 * FPS / lags
    weight = np.exp(-0.5 * (np.log2(bpms / 120.0) / 1.0) ** 2)
    scored = ac[lo:hi + 1] * weight
    best = int(np.argmax(scored))
    lag = float(lags[best])
    if 0 < best < len(scored) - 1:  # parabolic interpolation around the peak
        a, b, c = ac[lags[best] - 1], ac[lags[best]], ac[lags[best] + 1]
        denom = a - 2 * b + c
        if denom != 0:
            lag += 0.5 * (a - c) / denom
    strength = float(max(0.0, ac[int(round(lag))] - np.median(ac[lo:hi + 1])))
    return round(float(60 * FPS / lag), 1), min(1.0, strength * 2)


def track_beats(onset: Any, bpm: float, tightness: float = 100.0) -> list[float]:
    """Beat times in seconds by dynamic programming: strong onsets about one beat period apart."""
    import numpy as np

    period = 60.0 * FPS / bpm
    n = onset.size
    if n < 2 * period:
        return []
    score = onset.astype(np.float64).copy()
    back = np.full(n, -1, dtype=np.int64)
    window = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    penalty = -tightness * np.log(-window / period) ** 2
    for t in range(n):
        prev = t + window
        ok = prev >= 0
        if not ok.any():
            continue
        candidates = score[prev[ok]] + penalty[ok]
        k = int(np.argmax(candidates))
        if candidates[k] > 0:
            score[t] = onset[t] + candidates[k]
            back[t] = prev[ok][k]
    tail = int(round(period))
    t = int(n - tail + np.argmax(score[n - tail:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = int(back[t])
    beats.reverse()
    return [round(b / FPS, 3) for b in beats][:MAX_BEATS]


def _blocks(feats: dict[str, Any]) -> Any:
    """Half-second blocks of z-scored features: loudness in dB, the four band shares, onset density."""
    import numpy as np

    per = max(1, int(round(BLOCK_SECONDS * FPS)))
    n = feats["rms"].size // per
    if n < 2:
        return np.zeros((max(n, 1), 6)), per
    rms = feats["rms"][: n * per].reshape(n, per).mean(axis=1)
    bands = feats["bands"][: n * per].reshape(n, per, 4).sum(axis=1)
    share = bands / (bands.sum(axis=1, keepdims=True) + 1e-12)
    onset = feats["onset"][: n * per].reshape(n, per).mean(axis=1)
    loud = 20 * np.log10(rms + 1e-6)
    table = np.column_stack([loud, share, onset])
    std = table.std(axis=0)
    std[std < 1e-6] = 1.0
    table = (table - table.mean(axis=0)) / std
    kernel = np.ones(3) / 3
    table = np.column_stack([np.convolve(table[:, i], kernel, mode="same") for i in range(table.shape[1])])
    return table, per


def novelty(table: Any) -> Any:
    """Foote novelty: a tapered checkerboard kernel slid along the diagonal of the self-similarity matrix."""
    import numpy as np

    n = table.shape[0]
    half = max(2, min(int(MIN_SECTION_SECONDS / BLOCK_SECONDS), n // 4))
    padded = np.pad(table, ((half, half), (0, 0)), mode="edge")  # the ends repeat, so they add no false change
    dist = ((padded[:, None, :] - padded[None, :, :]) ** 2).sum(axis=2)
    scale = float(np.percentile(dist, 90)) or 1.0  # wide: texture inside a part should still look alike
    sim = np.exp(-dist / scale)
    axis = np.arange(-half, half) + 0.5
    taper = np.exp(-0.5 * (axis / (half / 2)) ** 2)
    checker = np.sign(axis)[:, None] * np.sign(axis)[None, :] * taper[:, None] * taper[None, :]
    out = np.zeros(n)
    for i in range(n):
        c = i + half
        out[i] = float((sim[c - half:c + half, c - half:c + half] * checker).sum()) / checker.size
    return np.maximum(out, 0)


def boundaries(nov: Any, seconds: float) -> list[float]:
    """Section starts (seconds, the first is 0): the highest novelty peaks at least ``MIN_SECTION_SECONDS`` apart."""
    import numpy as np

    if not nov.size or float(nov.max()) <= 0:
        return [0.0]
    gap = int(MIN_SECTION_SECONDS / BLOCK_SECONDS)
    floor = max(float(nov.mean() + 0.5 * nov.std()), 0.3 * float(nov.max()))  # a real change, not texture
    picked: list[int] = []
    for i in np.argsort(nov)[::-1]:
        if nov[i] < floor or len(picked) >= min(MAX_SECTIONS - 1, int(seconds / MIN_SECTION_SECONDS)):
            break
        t = int(i)
        if not (nov[max(0, t - 1)] <= nov[t] >= nov[min(nov.size - 1, t + 1)]):
            continue  # the shoulder of a peak, not the peak
        if t * BLOCK_SECONDS < MIN_SECTION_SECONDS / 2 or seconds - t * BLOCK_SECONDS < MIN_SECTION_SECONDS / 2:
            continue
        if all(abs(t - p) >= gap for p in picked):
            picked.append(t)
    return [0.0] + sorted(round(p * BLOCK_SECONDS, 3) for p in picked)


def _snap(times: list[float], beats: list[float]) -> list[float]:
    if not beats:
        return times
    out = [times[0]]
    for t in times[1:]:
        nearest = min(beats, key=lambda b: abs(b - t))
        out.append(round(nearest if abs(nearest - t) < 1.0 else t, 3))
    return sorted(set(out))


def label_sections(starts: list[float], seconds: float, table: Any, per_block_seconds: float,
                   loud_db: Any) -> list[dict[str, Any]]:
    """Sections with a letter (alike parts share it), ``intro`` / ``outro`` for a quiet short first or last part,
    and energy 0..1 (the part's loudness between the quietest and loudest part)."""
    import numpy as np

    ends = starts[1:] + [seconds]
    spans = [(a, b) for a, b in zip(starts, ends) if b - a > 0.05]
    means, levels = [], []
    for a, b in spans:
        lo, hi = int(a / per_block_seconds), max(int(a / per_block_seconds) + 1, int(b / per_block_seconds))
        means.append(table[lo:hi].mean(axis=0) if hi <= table.shape[0] and lo < table.shape[0] else table[-1])
        levels.append(float(loud_db[lo:hi].mean()) if lo < loud_db.size else float(loud_db[-1]))
    span_db = (max(levels) - min(levels)) if levels else 0.0
    energy = [round((v - min(levels)) / span_db, 3) if span_db > 0.5 else 1.0 for v in levels]
    letters: list[tuple[str, Any]] = []
    labels = []
    for vector in means:
        found = next((name for name, rep in letters if float(np.sqrt(((vector - rep) ** 2).mean())) < 0.6), None)
        if found is None:
            found = chr(ord("A") + len(letters)) if len(letters) < 26 else "Z"
            letters.append((found, vector))
        labels.append(found)
    median = float(np.median(energy)) if energy else 0.0
    if len(spans) >= 3:
        if spans[0][1] - spans[0][0] < 0.2 * seconds and energy[0] < median:
            labels[0] = "intro"
        if spans[-1][1] - spans[-1][0] < 0.2 * seconds and energy[-1] < median:
            labels[-1] = "outro"
    order: dict[str, str] = {}
    for label in labels:  # letters again from A, in order of first use, once intro and outro took theirs
        if label not in ("intro", "outro") and label not in order:
            order[label] = chr(ord("A") + len(order)) if len(order) < 26 else "Z"
    labels = [order.get(label, label) for label in labels]
    return [{"start": round(a, 3), "end": round(min(b, seconds), 3), "label": label, "energy": e}
            for (a, b), label, e in zip(spans, labels, energy)]


def analyse_samples(samples: Any, bpm_hint: float | None = None) -> dict[str, Any]:
    """Tempo, beats and sections of decoded samples, with ``rough`` and a ``note`` saying how it was found."""
    import numpy as np

    seconds = round(samples.size / SAMPLE_RATE, 3)
    feats = features(samples)
    notes = [ROUGH_NOTE]
    bpm, strength = tempo(feats["onset"], bpm_hint)
    if bpm_hint is not None and bpm is not None:
        notes.append(HINT_NOTE)
    if bpm is None or strength < 0.05:
        bpm, beats = None, []
        notes.append(NO_BEAT_NOTE)
    else:
        beats = track_beats(feats["onset"], bpm)
    table, per = _blocks(feats)
    starts = _snap(boundaries(novelty(table), seconds), beats) if table.shape[0] >= 8 else [0.0]
    loud = 20 * np.log10(feats["rms"][: table.shape[0] * per].reshape(-1, per).mean(axis=1) + 1e-6) \
        if table.shape[0] * per <= feats["rms"].size else np.zeros(table.shape[0])
    sections = label_sections(starts, seconds, table, BLOCK_SECONDS, loud)
    return {"seconds": seconds, "bpm": bpm, "beats": beats, "sections": sections, "rough": True, "note": " ".join(notes)}


def analyse(path: Path, bpm_hint: float | None = None) -> dict[str, Any]:
    """Decode ``path`` (already checked) and analyse it."""
    return analyse_samples(decode(path), bpm_hint)
