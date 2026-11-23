"""Lip-sync takes (D-110): how many takes a shot makes, and which one is kept.

A sung shot makes 3 takes by default, speech 1, and 4 is the best-quality setting. Each take runs with its own seed
(the seed decides whether the face performs at all, ``docs/research/lip-sync-singing-experiments-2026-11.md``), is
scored by ``lip_sync_score`` against the vocal it followed, and ``lip_sync_score.pick_best`` keeps one: usable takes
first, then the highest score. Measured usable rate: 0.72 with 1 take, 0.93 with 3, 0.98 with 4.

The score reads the video with MediaPipe Face Landmarker and OpenCV, which the backend's Python does not carry. It
runs as its own process (``python -I lip_sync_score.py``) from the first Python that has both: the one named by
``CHAOSENGINE_LIPSYNC_SCORE_PYTHON``, else the backend's own. The face model (``face_landmarker.task``) is read from
``CHAOSENGINE_LIPSYNC_FACE_MODEL`` or ``~/.chaosengine/models/lipsync/face_landmarker.task``. Without either the takes
still run, every one is kept in the gallery, and the first is used; the clip's note says the takes were not scored.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TAKES_SPEECH = 1
TAKES_SUNG = 3
TAKES_BEST = 4  # the best-quality setting
MAX_TAKES = TAKES_BEST
SCORE_SECONDS = 10.0  # MediaPipe on the CPU for a 5 s take, process start included (measured 7.5-10 s)
_SCORE_TIMEOUT_S = 300

SCORE_PYTHON_ENV = "CHAOSENGINE_LIPSYNC_SCORE_PYTHON"
FACE_MODEL_ENV = "CHAOSENGINE_LIPSYNC_FACE_MODEL"
DEFAULT_FACE_MODEL = Path.home() / ".chaosengine" / "models" / "lipsync" / "face_landmarker.task"
_SCRIPT = Path(__file__).resolve().with_name("lip_sync_score.py")


def take_count(requested: int | None, singing: bool) -> int:
    """The takes to make: what was asked (1 to 4), else 3 for a sung vocal and 1 for speech."""
    if requested is None:
        return TAKES_SUNG if singing else TAKES_SPEECH
    return max(1, min(MAX_TAKES, int(requested)))


def take_seeds(seed: int, takes: int) -> list[int]:
    """One seed per take: the request's seed for take 1, the next ones after it."""
    return [(int(seed) + index) % 2**31 for index in range(takes)]


def _has_score_modules() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("mediapipe", "cv2"))


def scorer() -> tuple[str, Path] | str:
    """(Python to run the score with, face model) or the reason the takes cannot be scored."""
    model = Path(os.environ.get(FACE_MODEL_ENV) or DEFAULT_FACE_MODEL).expanduser()
    configured = os.environ.get(SCORE_PYTHON_ENV, "").strip()
    if configured:
        python = Path(configured).expanduser()
        if not python.is_file():
            return f"{SCORE_PYTHON_ENV} names a Python that does not exist"
        chosen = str(python)
    elif _has_score_modules():
        chosen = sys.executable
    else:
        return "the sync score needs MediaPipe and OpenCV, which are not installed"
    if not model.is_file():
        return "the face model for the sync score (face_landmarker.task) is not installed"
    return chosen, model


def score(found: tuple[str, Path], video: Path, audio: Path) -> tuple[Any | None, str | None]:
    """(``SyncScore``, None) for one take, or (None, reason) when the scoring process failed."""
    from backend_service.lip_sync_score import SyncScore

    python, model = found
    with tempfile.TemporaryDirectory(prefix="chaosengine-lipscore-") as tmp:
        out = Path(tmp) / "score.json"
        command = [python, "-I", str(_SCRIPT), str(video), str(audio), "--model", str(model), "--json", str(out)]
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=_SCORE_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"the sync score could not run ({type(exc).__name__})"
        if run.returncode != 0 or not out.is_file():
            last = (run.stderr or "").strip().splitlines()[-1:] or ["no output"]
            return None, f"the sync score failed: {last[0][:160]}"
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
            fields = {name: data[name] for name in SyncScore.__dataclass_fields__}
        except (OSError, ValueError, KeyError) as exc:
            return None, f"the sync score wrote no result ({type(exc).__name__})"
    return SyncScore(**fields), None


def pick(scores: list[Any | None]) -> int:
    """Index of the take to keep: ``pick_best`` over the scored takes; the first take when none was scored."""
    from backend_service.lip_sync_score import pick_best

    scored = [(index, value) for index, value in enumerate(scores) if value is not None]
    if not scored:
        return 0
    return scored[pick_best([value for _index, value in scored])][0]


def summary(scores: list[Any | None], seeds: list[int], kept: int, reason: str | None) -> str:
    """The runtime note's sentence about the takes."""
    takes = len(seeds)
    if takes == 1:
        return ""
    if all(value is None for value in scores):
        return (f"{takes} takes (seeds {', '.join(map(str, seeds))}) were not scored ({reason or 'no score'}); "
                f"kept take 1, the others are in the Video Gallery")
    value = scores[kept]
    state = "usable" if value is not None and value.usable else "none was usable"
    return (f"{takes} takes; the sync score kept take {kept + 1} (seed {seeds[kept]}, score "
            f"{value.score:.2f}, {state}); the others are in the Video Gallery")
