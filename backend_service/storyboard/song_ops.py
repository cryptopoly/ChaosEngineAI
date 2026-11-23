"""Song ops (S-M1): attach the analysed song, edit its sections, tempo and lyrics by hand, take it off.

Split out of ``ops`` like ``plan_ops``; ``ops`` imports this module at its end and registers these ops in ``OPS``.
``set_song`` is what ``POST .../song`` runs once the slow analysis is done (outside the project lock): the file is
staged by the ops file machinery (``SONG_FILE_ARGS``), copied to ``media/song/`` and, with ``addToMixer``, put on the
music lane at the film's start in the same step, so one Undo takes both off again. ``song`` is a diffed top-level key.
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Annotated

from pydantic import Field

from backend_service.storyboard.models import (
    MAX_BEATS,
    MAX_LYRICS,
    MAX_SECTIONS,
    MAX_SONG_SECONDS,
    AudioClip,
    AudioSource,
    Id,
    Line,
    Project,
    Song,
    SongSection,
    new_id,
)
from backend_service.storyboard.ops import OpError, _Args, _import, _seconds, _shot

Beat = Annotated[float, Field(ge=0, le=MAX_SONG_SECONDS)]


class SetSong(_Args):
    path: str = Field(min_length=1, max_length=4096)
    title: Line = ""
    seconds: float = Field(gt=0, le=MAX_SONG_SECONDS)
    bpm: float | None = Field(None, ge=30, le=400)
    beats: list[Beat] = Field(default_factory=list, max_length=MAX_BEATS)
    sections: list[SongSection] = Field(default_factory=list, max_length=MAX_SECTIONS)
    lyrics: list[Line] = Field(default_factory=list, max_length=MAX_LYRICS)
    libraryId: str | None = Field(None, max_length=40)
    note: str = Field("", max_length=600)
    addToMixer: bool = True


class EditSong(_Args):
    title: Line | None = None
    bpm: float | None = Field(None, ge=30, le=400)
    clearBpm: bool = False
    sections: list[SongSection] | None = Field(None, max_length=MAX_SECTIONS)
    lyrics: list[Line] | None = Field(None, max_length=MAX_LYRICS)


class ClearSong(_Args):
    removeClip: bool = True


class ShotVocal(_Args):
    shotId: Id
    audio: str = Field(min_length=1, max_length=4096)  # the slice made by ``song_vocals`` (a temporary WAV)
    lead: float = Field(ge=0, le=1)
    length: float = Field(gt=0, le=60)
    singing: bool = True


class SetShotVocals(_Args):
    """Song to lip-sync shots (FU-371): each shot's slice of the vocal, in one op (one Undo)."""

    items: list[ShotVocal] = Field(min_length=1, max_length=500)


def _need_song(p: Project) -> Song:
    if p.song is None:
        raise OpError("This project has no song. Attach one first.")
    return p.song


def _drop_clip(p: Project, clip_id: str | None) -> None:
    if clip_id and p.clip(clip_id) is not None:
        p.audio = [c for c in p.audio if c.id != clip_id]
        p.captions = [c for c in p.captions if c.clipId != clip_id]


def set_song(p: Project, folder: Path, a: SetSong) -> None:
    old = p.song
    rel, digest, _ = _import(folder, a.path, {"audio"}, f"media/song/song-{int(time.time() * 1000)}-{secrets.token_hex(3)}")
    clip_id = None
    if a.addToMixer:
        if old is not None:
            _drop_clip(p, old.clipId)  # the old song's clip and its lyric captions go with it
        clip_id = new_id("a")
        source = AudioSource(path=rel, hash=digest, measuredSeconds=_seconds(folder, rel) or a.seconds,
                             label=a.title or "Song")
        p.audio.append(AudioClip(id=clip_id, kind="music", source=source, start=0.0, duck=True))  # under voice (FU-212)
    # Without addToMixer an earlier song's clip stays on the mixer as an ordinary music clip.
    try:
        p.song = Song(path=rel, hash=digest, title=a.title, seconds=a.seconds, bpm=a.bpm, beats=list(a.beats),
                      sections=list(a.sections), lyrics=list(a.lyrics), clipId=clip_id, libraryId=a.libraryId,
                      rough=True, note=a.note, analysedAt=time.time())
    except ValueError as exc:
        raise OpError(f"song: {exc}") from exc


def edit_song(p: Project, folder: Path, a: EditSong) -> None:
    data = _need_song(p).model_dump()
    if a.title is not None:
        data["title"] = a.title
    if a.bpm is not None:
        data["bpm"] = a.bpm
    elif a.clearBpm:
        data["bpm"] = None
    if a.sections is not None:
        data["sections"] = [s.model_dump() for s in sorted(a.sections, key=lambda s: s.start)]
    if a.lyrics is not None:
        data["lyrics"] = list(a.lyrics)
    if data == p.song.model_dump():  # type: ignore[union-attr]
        raise OpError("Give title, bpm, clearBpm, sections or lyrics that change the song.")
    try:
        p.song = Song.model_validate(data)
    except ValueError as exc:
        raise OpError(f"song: {exc}") from exc


def clear_song(p: Project, folder: Path, a: ClearSong) -> None:
    song = _need_song(p)
    if a.removeClip:
        _drop_clip(p, song.clipId)
    p.song = None


def set_shot_vocals(p: Project, folder: Path, a: SetShotVocals) -> None:
    """Attach each slice as the shot's vocal; the shot grows by its lead-in, cut off again by ``trimIn``."""
    stamp = int(time.time() * 1000)
    for item in a.items:
        shot = _shot(p, item.shotId)
        rel, _digest, _src = _import(folder, item.audio, {"audio"}, f"media/{shot.id}/vocal-{stamp}")
        shot.vocalClip = rel
        shot.vocalStart = 0.0
        shot.vocalSinging = item.singing
        shot.seconds = round(item.lead + item.length, 3)
        shot.trimIn = item.lead
        shot.trimOut = round(item.lead + item.length, 3)


# name -> (argument model, change function, reversible); merged into ops.OPS
SONG_OPS = {
    "set_song": (SetSong, set_song, True),
    "edit_song": (EditSong, edit_song, True),
    "clear_song": (ClearSong, clear_song, True),
    "set_shot_vocals": (SetShotVocals, set_shot_vocals, True),
}
SONG_FILE_ARGS = {"set_song": (("path", {"audio"}),)}
