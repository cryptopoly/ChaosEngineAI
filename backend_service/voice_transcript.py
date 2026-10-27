"""Put speaker labels on a transcript.

The speech-to-text model gives segments (with word timings when asked); the
speaker model gives turns of who speaks when. Each word takes the speaker
whose turn overlaps it most (or the nearest turn, for a word in a gap), and a
segment is split wherever the speaker changes inside it.
"""

from __future__ import annotations

from typing import Any

# Languages written without spaces between words.
_NO_SPACE_LANGUAGES = {"zh", "ja", "ko", "th", "lo", "my"}


def _speaker_for(start: float, end: float, turns: list[dict[str, Any]]) -> int | None:
    if not turns:
        return None
    best, best_overlap = None, 0.0
    for turn in turns:
        overlap = min(end, turn["end"]) - max(start, turn["start"])
        if overlap > best_overlap:
            best, best_overlap = turn["speaker"], overlap
    if best is not None:
        return int(best)
    middle = (start + end) / 2
    nearest = min(turns, key=lambda t: min(abs(middle - t["start"]), abs(middle - t["end"])))
    return int(nearest["speaker"])


def _join(words: list[dict[str, Any]], language: str | None) -> str:
    separator = "" if (language or "").split("-")[0] in _NO_SPACE_LANGUAGES else " "
    return separator.join(w["text"] for w in words).strip()


def assign_speakers(
    segments: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Segments with a ``speaker`` number each, split where the speaker changes."""
    labelled: list[dict[str, Any]] = []
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            speaker = _speaker_for(segment["start"], segment["end"], turns)
            labelled.append({**segment, "speaker": speaker})
            continue
        runs: list[tuple[int | None, list[dict[str, Any]]]] = []
        for word in words:
            speaker = _speaker_for(word["start"], word["end"], turns)
            if runs and runs[-1][0] == speaker:
                runs[-1][1].append(word)
            else:
                runs.append((speaker, [word]))
        if len(runs) == 1:
            labelled.append({**segment, "speaker": runs[0][0]})
            continue
        for speaker, run in runs:
            labelled.append({
                "start": run[0]["start"],
                "end": run[-1]["end"],
                "text": _join(run, language),
                "words": run,
                "speaker": speaker,
            })
    return labelled
