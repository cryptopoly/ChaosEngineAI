"""Automatic lip-sync score for one clip: does the mouth open when the voice sounds and shut when it is silent?

Mouth opening per frame comes from MediaPipe Face Landmarker (Apache-2.0): the inner-lip gap (landmarks 13 and 14)
over the face height (10 to 152), median-filtered over 3 frames against landmark jitter. The voice comes from the
reference audio (best: the vocal stem of the shot's window, whatever the model was given): RMS per video frame after
a 100-4000 Hz band-pass. Numbers reported (FU-334, measured in ``docs/research/lip-sync-singing-experiments-2026-11.md``):

- ``r``: Pearson correlation of mouth opening and voice envelope, best lag within +-``max_lag`` frames (the mouth
  usually leads the sound by a frame or two), with ``chance_p95``: the 95th percentile of r over circular shifts
  of the envelope. ``r`` above ``chance_p95`` means the mouth follows this audio, not just any audio.
- ``closed_while_voiced``: share of voiced frames with the mouth shut (the singing failure: a held line sung with a
  closed mouth). ``open_while_silent``: share of silent frames with the mouth open (gaps between words count as
  silent, so even good speech scores 0.3-0.5 here).
- ``balanced``: mean of the two complements, 0.5 for a mouth that ignores the audio, 1.0 for perfect agreement.
- ``score``: ``0.5 * balanced + 0.5 * max(0, r)``, the one number used to rank takes of the same shot. It is a
  ranking aid, not a quality grade: compare it between takes of one shot, not across different songs.

The maths (``score_series``) is plain numpy and scipy. Reading a video needs ``cv2`` and ``mediapipe``, which the app's
venv does not carry; they are imported only in ``mouth_series`` (the experiments ran them from a tools venv).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

UPPER_LIP, LOWER_LIP, FOREHEAD, CHIN = 13, 14, 10, 152
BAND_HZ = (100.0, 4000.0)
ENV_RATE = 16000
VOICED_FRACTION = 0.15  # voiced when the envelope is above 15 % of its 95th percentile
OPEN_THRESHOLD = 0.02  # inner-lip gap / face height above which the mouth counts as open
MAX_LAG = 3
CHANCE_SHIFTS = 200
MIN_SHIFT = 12


@dataclass
class SyncScore:
    frames: int
    face_frames: int
    voiced_frames: int
    r: float
    r_lag0: float
    lag: int
    chance_p95: float
    chance_mean: float
    beats_chance: bool
    closed_while_voiced: float
    open_while_silent: float
    balanced: float
    score: float
    mouth_mean: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def envelope(samples: Any, sample_rate: int, n_frames: int, fps: float) -> Any:
    """Voice-band RMS of ``samples`` (mono or ``[n, channels]``) in ``n_frames`` windows of one video frame each."""
    import numpy as np  # noqa: PLC0415
    from scipy.signal import butter, resample_poly, sosfiltfilt  # noqa: PLC0415

    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != ENV_RATE:
        audio = resample_poly(audio, ENV_RATE, int(sample_rate))
    audio = sosfiltfilt(butter(4, BAND_HZ, btype="band", fs=ENV_RATE, output="sos"), audio)
    hop = ENV_RATE / float(fps)
    out = np.zeros(n_frames)
    for i in range(n_frames):
        seg = audio[int(i * hop): int((i + 1) * hop)]
        out[i] = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
    return out


def _pearson(x: Any, y: Any) -> float:
    import numpy as np  # noqa: PLC0415

    keep = ~(np.isnan(x) | np.isnan(y))
    if keep.sum() < 10:
        return float("nan")
    a, b = x[keep] - x[keep].mean(), y[keep] - y[keep].mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def _lagged(mouth: Any, env: Any, lag: int) -> float:
    """r of mouth[t] against env[t + lag]: a positive lag means the mouth moves before the sound."""
    if lag > 0:
        return _pearson(mouth[:-lag], env[lag:])
    if lag < 0:
        return _pearson(mouth[-lag:], env[:lag])
    return _pearson(mouth, env)


def score_series(mouth: Any, env: Any, *, max_lag: int = MAX_LAG, open_threshold: float = OPEN_THRESHOLD,
                 seed: int = 0) -> SyncScore:
    """Score a mouth-opening series (NaN where no face was found) against a voice envelope of the same length."""
    import numpy as np  # noqa: PLC0415

    mouth = np.asarray(mouth, dtype=np.float64)
    env = np.asarray(env, dtype=np.float64)
    if mouth.shape != env.shape:
        raise ValueError(f"mouth has {mouth.size} frames and the envelope {env.size}")
    n = mouth.size
    lags = {lag: _lagged(mouth, env, lag) for lag in range(-max_lag, max_lag + 1)}
    finite = {k: v for k, v in lags.items() if not np.isnan(v)}
    best = max(finite, key=finite.get) if finite else 0
    r = finite.get(best, float("nan"))
    chance = np.array([])
    if n > 2 * MIN_SHIFT + 1:
        rng = np.random.default_rng(seed)
        shifts = rng.integers(MIN_SHIFT, n - MIN_SHIFT, CHANCE_SHIFTS)
        # the same best-of-lags search on shuffled audio, so chance gets the same advantage
        chance = np.array([max((_lagged(mouth, np.roll(env, int(s)), k) for k in range(-max_lag, max_lag + 1)),
                               key=lambda v: -2.0 if np.isnan(v) else v) for s in shifts])
        chance = chance[~np.isnan(chance)]
    p95 = float(np.percentile(chance, 95)) if chance.size else float("nan")
    ref = np.percentile(env, 95) if n else 0.0
    voiced = env > VOICED_FRACTION * ref if ref > 0 else np.zeros(n, dtype=bool)
    face = ~np.isnan(mouth)
    opened = np.nan_to_num(mouth, nan=0.0) > open_threshold
    v, s = voiced & face, ~voiced & face
    closed_v = float((v & ~opened).sum() / v.sum()) if v.sum() else 0.0
    open_s = float((s & opened).sum() / s.sum()) if s.sum() else 0.0
    balanced = 0.5 * ((1.0 - closed_v) + (1.0 - open_s))
    r_safe = 0.0 if np.isnan(r) else r
    return SyncScore(
        frames=int(n),
        face_frames=int(face.sum()),
        voiced_frames=int(voiced.sum()),
        r=round(r_safe, 4),
        r_lag0=round(lags[0], 4) if not np.isnan(lags[0]) else 0.0,
        lag=int(best),
        chance_p95=round(p95, 4),
        chance_mean=round(float(chance.mean()), 4) if chance.size else float("nan"),
        beats_chance=bool(chance.size and r_safe > p95),
        closed_while_voiced=round(closed_v, 4),
        open_while_silent=round(open_s, 4),
        balanced=round(balanced, 4),
        score=round(0.5 * balanced + 0.5 * max(0.0, r_safe), 4),
        mouth_mean=round(float(np.nanmean(mouth)), 4) if face.any() else 0.0,
    )


def smooth(mouth: Any) -> Any:
    """3-frame median, NaN-aware (a missing face stays missing)."""
    import numpy as np  # noqa: PLC0415

    m = np.asarray(mouth, dtype=np.float64)
    out = m.copy()
    for i in range(m.size):
        win = m[max(0, i - 1): i + 2]
        win = win[~np.isnan(win)]
        if not np.isnan(m[i]) and win.size:
            out[i] = float(np.median(win))
    return out


def mouth_series(video: Path, model: Path) -> tuple[Any, float, list[Any]]:
    """Per-frame mouth opening of the first face, the video's fps, and the decoded BGR frames (for a strip)."""
    import cv2  # noqa: PLC0415
    import mediapipe as mp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames: list[Any] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    values: list[float] = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            found = landmarker.detect_for_video(image, int(i * 1000 / fps))
            if not found.face_landmarks:
                values.append(float("nan"))
                continue
            p = found.face_landmarks[0]
            gap = np.hypot(p[UPPER_LIP].x - p[LOWER_LIP].x, p[UPPER_LIP].y - p[LOWER_LIP].y)
            height = np.hypot(p[FOREHEAD].x - p[CHIN].x, p[FOREHEAD].y - p[CHIN].y)
            values.append(float(gap / height) if height > 0 else float("nan"))
    return smooth(values), fps, frames


def score_files(video: Path, audio: Path, model: Path) -> tuple[SyncScore, dict[str, Any]]:
    """Score ``video`` against the voice in ``audio`` (trimmed or zero-padded to the video's length)."""
    import soundfile as sf  # noqa: PLC0415

    mouth, fps, _frames = mouth_series(video, model)
    samples, rate = sf.read(str(audio), always_2d=True, dtype="float64")
    env = envelope(samples, rate, len(mouth), fps)
    result = score_series(mouth, env)
    return result, {"mouth": [None if v != v else round(float(v), 4) for v in mouth],
                    "env": [round(float(v), 5) for v in env], "fps": fps}


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("audio", type=Path, help="the voice to score against (a vocal stem is best)")
    parser.add_argument("--model", type=Path, required=True, help="MediaPipe face_landmarker.task")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    result, series = score_files(args.video, args.audio, args.model)
    print(json.dumps(result.as_dict(), indent=1))
    if args.json:
        args.json.write_text(json.dumps({**result.as_dict(), **series}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
