"""What a Storyboard project still lacks, read from the saved project (Keep going, FU-191).

The run controller asks this after every Studio turn: which shots have no still or no clip, which audio lanes
are missing, whether the final film is rendered and current. ``FilmState`` is plain data so the nudge text, the
"goal looks complete" check and the progress snapshot all read the same facts. Nothing here writes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Words in the brief that ask for a voice lane or captions. Music is always expected (an ad film has a bed).
_VOICE_WORDS = re.compile(r"\b(voice|voiceover|voice-over|narrat\w*|speech|spoken|speak\w*|dialogue|presenter)\b", re.I)
_CAPTION_WORDS = re.compile(r"\b(caption\w*|subtitle\w*)\b", re.I)
# A brief that asks for a subtitle file accepts captions written as an .srt instead of drawn into the picture.
_SRT_WORDS = re.compile(r"\b(srt|subtitle file|caption file)\b", re.I)


# "2-minute film", "two minutes", "about 120 s": the length the brief asks for. Short numbers (a 4 s shot) are
# not a film length; the largest figure of 15 s or more is.
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_LENGTH = re.compile(r"\b(\d+(?:\.\d+)?|" + "|".join(_WORD_NUMBERS) + r")\s*-?\s*(minutes?|mins?|seconds?|secs?|s)\b", re.I)
_MIN_FILM_SECONDS = 15.0
SHORT_FILM_RATIO = 0.85  # a film under this share of the asked length is "too short"


def target_seconds(goal: str) -> float | None:
    """The film length the brief asks for, in seconds, or None when it names none."""
    best: float | None = None
    for number, unit in _LENGTH.findall(goal or ""):
        value = _WORD_NUMBERS.get(number.lower()) if number.lower() in _WORD_NUMBERS else float(number)
        seconds = value * 60 if unit.lower().startswith("min") else value
        if seconds >= _MIN_FILM_SECONDS and (best is None or seconds > best):
            best = float(seconds)
    return best


def wants_voice(goal: str) -> bool:
    return bool(_VOICE_WORDS.search(goal or ""))


def wants_captions(goal: str) -> bool:
    return bool(_CAPTION_WORDS.search(goal or ""))


@dataclass
class FilmState:
    project_id: str
    name: str
    revision: int
    shots: int
    seconds: float = 0.0  # the shots' lengths added up
    needs_still: list[int] = field(default_factory=list)  # shot numbers, 1-based in film order
    needs_clip: list[int] = field(default_factory=list)
    needs_redo: list[int] = field(default_factory=list)  # stale (inputs changed) or failed
    missing_files: list[int] = field(default_factory=list)
    takes: int = 0
    music: bool = False
    voice: bool = False
    captions: int = 0
    render: str = "none"  # none | running | failed | preview | stale | done
    render_error: str | None = None
    render_finished_at: float | None = None
    batch_running: bool = False
    batch_paused: str | None = None  # id of a paused batch a resume would finish
    captions_burn: bool = False  # export setting: captions drawn into the picture
    captions_srt: bool = False  # export setting: captions written as an .srt beside the film
    hashes: set[str] = field(default_factory=set)  # sha256 of every still, clip and audio file in the project
    made: set[str] = field(default_factory=set)  # audio this run's chat made ("music", "voice"); set by the adapter

    def captions_ok(self, goal: str) -> bool:
        """Captions that exist show in the film: burned in, or an .srt when the brief asks for a file."""
        return self.captions == 0 or self.captions_burn or (self.captions_srt and bool(_SRT_WORDS.search(goal or "")))

    def wants_voice(self, goal: str) -> bool:
        return wants_voice(goal) or "voice" in self.made

    @property
    def shots_done(self) -> bool:
        return self.shots > 0 and not (self.needs_still or self.needs_clip or self.needs_redo or self.missing_files)

    def too_short(self, goal: str) -> float | None:
        """The length the brief asks for when the shots add up to clearly less than it, else None."""
        target = target_seconds(goal)
        return target if target and self.shots and self.seconds > 0 and self.seconds < target * SHORT_FILM_RATIO else None

    def extra_shots(self, target: float) -> int:
        average = self.seconds / self.shots if self.shots else 4.0
        return max(1, round((target - self.seconds) / max(average, 0.5)))

    def lacks(self, goal: str) -> list[str]:
        """What is still missing, as short phrases, in the order the work is done."""
        out: list[str] = []
        if self.shots == 0:
            out.append("no shots yet")
        target = self.too_short(goal)
        if target:
            out.append(f"film length ({self.seconds:.0f} s of the {target:.0f} s the brief asks for: about "
                       f"{self.extra_shots(target)} more shots)")
        if self.needs_still:
            out.append(f"still for shot {_numbers(self.needs_still)}")
        if self.needs_clip:
            out.append(f"clip for shot {_numbers(self.needs_clip)}")
        if self.needs_redo:
            out.append(f"new take for shot {_numbers(self.needs_redo)} (changed or failed)")
        if self.missing_files:
            out.append(f"files missing for shot {_numbers(self.missing_files)}")
        if not self.music:
            out.append("music")
        if self.wants_voice(goal) and not self.voice:
            out.append("voice")
        if wants_captions(goal) and self.captions == 0:
            out.append("captions")
        elif not self.captions_ok(goal):
            out.append("captions burned into the picture")
        if self.render != "done":
            out.append({"stale": "a new render (the film changed after the last one)",
                        "preview": "the final render (only a preview exists)",
                        "failed": "a render that works (the last one failed)",
                        "running": "the render to finish"}.get(self.render, "the final render"))
        return out

    def complete(self, goal: str) -> bool:
        """Every shot has a clip, the lanes the brief asks for are in and the final film is rendered after the
        last change, as far as the saved project can tell."""
        return self.shots_done and not self.lacks(goal)

    def next_step(self, goal: str, made: dict[str, str] | None = None) -> str:
        """One concrete next step, naming the tool and the arguments that matter. ``made`` maps "music" and
        "voice" to a file this chat already made and the project lacks: attach it rather than make it again."""
        pid = self.project_id
        made = made or {}
        if self.batch_running:
            return "A generate batch is running: call wait_for_job on its job id, then check the project again."
        if self.batch_paused:
            return f"Resume the paused batch: storyboard_batch action resume, project_id {pid}, batch_id {self.batch_paused}."
        if self.shots == 0:
            return f"Add the shots from the brief with storyboard_shot add (project_id {pid})."
        target = self.too_short(goal)
        if target:
            return (f"The film is {self.seconds:.0f} s, the brief asks for about {target:.0f} s. Add about "
                    f"{self.extra_shots(target)} more shots with storyboard_shot add (project_id {pid}) to fill the "
                    "story, then make them.")
        if self.missing_files:
            return (f"Shot {_numbers(self.missing_files)} lost its files: storyboard_generate with shot_ids for "
                    "those shots, or ask the user.")
        if self.needs_still:
            return (f"storyboard_generate with project_id {pid}, what \"still\" (shots {_numbers(self.needs_still)}), "
                    "then wait_for_job.")
        if self.needs_clip or self.needs_redo:
            shots = sorted(set(self.needs_clip) | set(self.needs_redo))
            return f"storyboard_generate with project_id {pid}, what \"clip\" (shots {_numbers(shots)}), then wait_for_job."
        if not self.music and made.get("music"):
            return (f'storyboard_audio add with project_id {pid}, kind "music", path "{made["music"]}" (the track '
                    "this chat already made; do not make new music).")
        if not self.music:
            return (f"make_music for the film's length, wait_for_job, then storyboard_audio add kind \"music\" with the "
                    f"track's path (project_id {pid}).")
        if self.wants_voice(goal) and not self.voice and made.get("voice"):
            return (f'storyboard_audio add with project_id {pid}, kind "voice", path "{made["voice"]}" (the voice '
                    "line this chat already made; do not speak it again).")
        if self.wants_voice(goal) and not self.voice:
            return f"speak the voiceover, wait_for_job, then storyboard_audio add kind \"voice\" (project_id {pid})."
        if wants_captions(goal) and self.captions == 0:
            return f"storyboard_captions to make the captions (project_id {pid}), then wait_for_job."
        if not self.captions_ok(goal):
            return (f'storyboard_project edit with project_id {pid} and fields {{"export": {{"captionsBurn": true}}}} '
                    "so the captions show in the film, then render.")
        if self.render == "running":
            return "The render is running: call wait_for_job on its job id."
        if self.render != "done":
            return f"storyboard_render with project_id {pid} and preview false, then wait_for_job."
        return (f"Everything looks done and the film is rendered ({self.seconds:.0f} s long). Do not render again. "
                "Say in one line that it is finished, with its real length.")

    def summary(self, goal: str) -> str:
        head = f'Project "{self.name}" (id {self.project_id}, revision {self.revision}): {self.shots} shots.'
        lacks = self.lacks(goal)
        return f"{head} Still missing: {'; '.join(lacks)}." if lacks else f"{head} Nothing is missing."

    def snapshot(self) -> dict[str, Any]:
        """What counts as progress between two nudges: an edit (revision), a take, an audio clip, captions, a
        finished render."""
        return {"project": self.project_id, "revision": self.revision, "takes": self.takes,
                "music": self.music, "voice": self.voice, "captions": self.captions,
                "render": self.render_finished_at}


def _numbers(values: list[int]) -> str:
    """1, 2, 3, 5 -> '1-3, 5' (short for a 24-shot film)."""
    nums = sorted(set(values))
    parts: list[str] = []
    start = prev = None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = n
    if start is not None:
        parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def _render_state(folder: Path, newest_change: float) -> tuple[str, str | None, float | None]:
    """The final render's state from its manifest: done only when it finished after the newest change and its
    file is there. A preview never counts as the final film."""
    work = folder / "cache" / "render"

    def read(name: str) -> dict[str, Any] | None:
        try:
            data = json.loads((work / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    export = read("manifest-export.json")
    if export is not None and (export.get("preview") or Path(str(export.get("out") or "")).name == "preview.mp4"):
        export = None  # a preview is never the finished film, whatever file it came from
    if export is None:
        return ("preview" if (read("manifest-preview.json") or {}).get("state") == "done" else "none"), None, None
    state = str(export.get("state") or "")
    finished = export.get("finishedAt")
    finished_at = float(finished) if isinstance(finished, (int, float)) else None
    if state == "running":
        return "running", None, finished_at
    if state != "done":
        return "failed", (str(export.get("error")) if export.get("error") else None), finished_at
    out = str(export.get("out") or "")
    if not out or not (folder / "exports" / Path(out).name).is_file():
        return "none", None, finished_at
    if finished_at is None or finished_at + 1.0 < newest_change:
        return "stale", None, finished_at
    return "done", None, finished_at


def film_state(project: Any, folder: Path) -> FilmState:
    """Read one loaded project (``storyboard.models.Project``) and its folder."""
    shots = sorted(project.shots, key=lambda s: s.order)
    state = FilmState(project_id=project.id, name=project.name, revision=project.revision, shots=len(shots),
                      seconds=sum(float(s.seconds) for s in shots))
    newest = 0.0
    for number, shot in enumerate(shots, start=1):
        state.takes += len(shot.takes)
        newest = max([newest, *(t.createdAt for t in shot.takes)])
        state.hashes.update(h for t in shot.takes for h in t.hashes.values())
        take = shot.active()
        if shot.status == "missing":
            state.missing_files.append(number)
        elif shot.status in ("stale", "failed") or shot.redo:
            state.needs_redo.append(number)
        elif take is None or not take.still and not take.clip:
            (state.needs_still if shot.startFrom == "generate" else state.needs_clip).append(number)
        elif not take.clip:
            state.needs_clip.append(number)
    ok_audio = [a for a in project.audio if a.status == "ok"]
    state.music = any(a.kind == "music" for a in ok_audio) or project.song is not None
    state.voice = any(a.kind == "voice" for a in ok_audio)
    state.captions = len(project.captions)
    state.captions_burn = bool(project.export.captionsBurn)
    state.captions_srt = bool(project.export.captionsSrt)
    state.hashes.update(a.source.hash for a in project.audio if a.source.hash)
    if project.song is not None and project.song.hash:
        state.hashes.add(project.song.hash)
    for batch in project.batches:
        if batch.state in ("queued", "running"):
            state.batch_running = True
        elif batch.state == "paused":
            state.batch_paused = batch.id
    try:
        newest = max(newest, (folder / "project.json").stat().st_mtime)
    except OSError:
        pass
    state.render, state.render_error, state.render_finished_at = _render_state(folder, newest)
    return state


def read_film_state(project_id: str | None) -> FilmState | None:
    """The saved project's state, or None when there is no project or it cannot be read."""
    if not project_id:
        return None
    try:
        from backend_service.storyboard.store import default_store

        store = default_store()
        project = store.load(project_id)
        return film_state(project, store.folder(project.id))
    except Exception:  # noqa: BLE001 - an unreadable project is "no project" for the nudge; the model sees the error
        return None
