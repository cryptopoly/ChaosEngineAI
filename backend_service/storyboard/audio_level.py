"""The gain a new Storyboard voice lane starts with, from the narration's own level (FU-218).

A fixed boost cannot suit every narration: one voice file sits near -23 dB RMS and another near -29 dB. So when a
voice lane is created the file is measured once with the bundled ffmpeg and the lane gets the gain that brings its
RMS to ``TARGET_RMS_DB``, never below 0 dB, never above ``MAX_GAIN_DB`` and never so much that the file's peak would
pass ``PEAK_CEILING_DB``.

The target is the level FU-216 calibrated: its fixed +8 dB on a -22.6 dB narration put it at about -14.6 dB, which
measured 10 dB or more over the ducked music bed. The peak ceiling is a backstop, not a clip guard: the export's
loudness normaliser (on by default) limits peaks, and a ceiling near full scale would hold a normal narration back to
+2.7 dB, which measured 4.5 dB over the bed. The result is stored in the lane's ``gainDb`` like any value the user sets; nothing is
measured when the film is rendered. A file that cannot be measured gets ``FALLBACK_GAIN_DB``.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from backend_service.storyboard import media

TARGET_RMS_DB = -14.0
MAX_GAIN_DB = 12.0  # the most a lane's gainDb can hold (AudioClip.gainDb)
PEAK_CEILING_DB = 6.0
FALLBACK_GAIN_DB = 8.0
_TIMEOUT_SECONDS = 20

# Digital silence (below -80 dB) is dropped first so a padded file is not read as quieter than its speech.
_FILTER = ("silenceremove=start_periods=1:start_threshold=-80dB:stop_periods=-1:stop_threshold=-80dB:stop_duration=0.1,"
           "astats=metadata=0:measure_perchannel=none:measure_overall=Peak_level+RMS_level")
_RMS = re.compile(r"RMS level dB:\s*(-?\d+(?:\.\d+)?|-inf)")
_PEAK = re.compile(r"Peak level dB:\s*(-?\d+(?:\.\d+)?|-inf)")


@dataclass(frozen=True)
class Level:
    rms_db: float
    peak_db: float


def measure(path: Path) -> Level | None:
    """The file's overall RMS and peak in dBFS, or None when ffmpeg is missing, the file cannot be read or it is silent."""
    exe = media.ffmpeg_exe()
    if not exe:
        return None
    cmd = [exe, "-hide_banner", "-nostdin", "-protocol_whitelist", media.PROTOCOLS, "-i", str(path),
           "-map", "0:a:0", "-af", _FILTER, "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    text = result.stderr.decode("utf-8", "replace")
    rms, peak = _RMS.findall(text), _PEAK.findall(text)
    if not rms or not peak or "-inf" in (rms[-1], peak[-1]):
        return None
    return Level(float(rms[-1]), float(peak[-1]))


def gain_for(level: Level | None) -> float:
    """The starting gain for a voice lane whose file measured ``level`` (``FALLBACK_GAIN_DB`` when it did not)."""
    if level is None:
        return FALLBACK_GAIN_DB
    gain = min(max(TARGET_RMS_DB - level.rms_db, 0.0), MAX_GAIN_DB)
    gain = min(gain, max(PEAK_CEILING_DB - level.peak_db, 0.0))
    return round(gain, 1)


def voice_gain_db(path: Path) -> float:
    return gain_for(measure(path))
