"""Storyboard shot length, audio fades at the film's end and music ducking under narration (FU-210 to FU-212).

The frame and timeline checks are pure. The render checks build a project folder with synthetic media made by the
bundled ffmpeg (lavfi sources, 160x90), turn it into a render plan with ``build_render_plan`` (so the timeline's
clamps are part of what is tested) and measure the film's sound with ``astats``. They skip when ffmpeg is missing.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from backend_service.helpers.generation_limits import shot_clip_frames
from backend_service.storyboard import ffmpeg_bin
from backend_service.storyboard.assemble import render
from backend_service.storyboard.models import Anchor, AudioClip, AudioSource, Defaults, Project, Shot, Take
from backend_service.storyboard.render_plan import RenderPlan
from backend_service.storyboard.timeline import build_render_plan, compute_timeline
from tests.storyboard_fixtures import StoreCase, fake_wav

LTX = "dgrauet/ltx-2.5-mlx-q4"


def _ffmpeg() -> str | None:
    try:
        return ffmpeg_bin.ffmpeg_exe()
    except RuntimeError:
        return None


def made_shot(sid: str, seconds: float, frames: int, fps: int = 24, origin: str = "batch", **fields) -> Shot:
    take = Take(id=f"t{sid}", clip=f"media/{sid}/clip.mp4", measuredSeconds=round(frames / fps, 4),
                hashes={"clip": (sid * 64)[:64]}, origin=origin)
    return Shot(id=sid, seconds=seconds, takes=[take], activeTake=take.id, status="done", **fields)


class FrameMathTests(unittest.TestCase):
    def test_ltx_frames_cover_the_shot(self):
        # 8n+1 frames, rounded up: never short of the shot (89 frames for 4 s was the rehearsal's defect).
        self.assertEqual(shot_clip_frames(LTX, 4, 24), 97)
        self.assertEqual(shot_clip_frames(LTX, 4, 25), 105)
        self.assertEqual(shot_clip_frames(LTX, 1, 24), 25)
        self.assertEqual(shot_clip_frames(LTX, 8, 24), 193)
        self.assertEqual(shot_clip_frames(LTX, 10, 24), 241)
        self.assertEqual(shot_clip_frames(LTX, 10, 25), 257)  # 250 rounds up to the cap
        self.assertEqual(shot_clip_frames(LTX, 20, 24), 257)  # past the cap: the longest clip the model makes
        for seconds, fps in ((4, 24), (4, 25), (1, 24), (8, 24), (10, 24)):
            frames = shot_clip_frames(LTX, seconds, fps)
            self.assertEqual((frames - 1) % 8, 0)
            self.assertGreaterEqual(frames / fps, seconds)

    def test_other_models_keep_their_own_rule(self):
        self.assertEqual(shot_clip_frames("Wan-AI/Wan2.2-TI2V-5B-Diffusers", 4, 24), 97)  # 4n+1
        self.assertEqual(shot_clip_frames("Wan-AI/Wan2.2-TI2V-5B-Diffusers", 1, 24), 25)
        self.assertEqual(shot_clip_frames("some/other-model", 4, 24), 96)  # no rule: the exact count

    def test_estimate_asks_for_the_frames_the_batch_asks_for(self):
        from backend_service.storyboard.estimate import _clip_frames

        self.assertEqual(_clip_frames({"repo": LTX}, Shot(id="a", seconds=4), 24), 97)


class ShotLengthTimelineTests(unittest.TestCase):
    def project(self, shots):
        return Project(id="0123456789abcdef", name="Film", shots=list(shots))

    def test_the_film_is_the_sum_of_the_shots_asked_for(self):
        t = compute_timeline(self.project([made_shot(s, 4, 97) for s in "abc"]))
        self.assertAlmostEqual(t.total, 12.0, places=6)
        self.assertEqual([round(s.duration, 6) for s in t.shots], [4.0, 4.0, 4.0])
        self.assertEqual([round(s.source_out, 6) for s in t.shots], [4.0, 4.0, 4.0])
        mixed = compute_timeline(self.project([made_shot("a", 1, 25), made_shot("b", 8, 193), made_shot("c", 10, 241)]))
        self.assertAlmostEqual(mixed.total, 19.0, places=6)
        self.assertEqual(mixed.problems, [])  # a clip longer than its shot is not a problem to report

    def test_a_short_clip_and_an_imported_clip_keep_their_own_length(self):
        t = compute_timeline(self.project([made_shot("a", 4, 89), made_shot("b", 4, 97, origin="upload")]))
        self.assertAlmostEqual(t.shots[0].duration, 3.7083, places=4)  # an older take made too short
        self.assertAlmostEqual(t.shots[1].duration, 97 / 24, places=4)  # an imported clip is not cut

    def test_trims_and_speed_still_apply(self):
        t = compute_timeline(self.project([made_shot("a", 4, 97, trimIn=1.0), made_shot("b", 4, 97, trimOut=2.0),
                                           made_shot("c", 4, 97, speed=2.0)]))
        self.assertEqual([round(s.duration, 6) for s in t.shots], [3.0, 2.0, 2.0])


class _Synth:
    """Synthetic media in a project folder, made once per test class."""

    @staticmethod
    def run(*args: str) -> None:
        subprocess.run([ffmpeg_bin.ffmpeg_exe(), "-v", "error", "-nostdin", "-y", *args], check=True)

    @classmethod
    def video(cls, path: Path, frames: int, fps: int = 24, colour: str = "black") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cls.run("-f", "lavfi", "-i", f"color=c={colour}:s=160x90:r={fps}", "-frames:v", str(frames),
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path))

    @classmethod
    def tone(cls, path: Path, freq: int, seconds: float, volume: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cls.run("-f", "lavfi", "-i", f"sine=f={freq}:d={seconds}:sample_rate=48000", "-af", f"volume={volume}", str(path))


def _folder(root: Path) -> Path:
    for sub in ("media/audio", "exports", "cache/render"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _render(project: Project, folder: Path) -> Path:
    plan = build_render_plan(project, preview=True, folder=folder)
    plan["loudness_lufs"] = None  # measure the mix itself, not a loudness-normalised copy
    return Path(render(RenderPlan.from_dict(plan)).path)


def rms_db(path: Path, start: float, seconds: float, band: str | None = None) -> float:
    """RMS level (dBFS) of the film's sound from ``start`` for ``seconds``, optionally through a filter first."""
    chain = f"{band}," if band else ""
    out = subprocess.run([ffmpeg_bin.ffmpeg_exe(), "-hide_banner", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}",
                          "-i", str(path), "-vn", "-af", f"{chain}astats=metadata=0", "-f", "null", "-"],
                         capture_output=True, text=True, check=True).stderr
    levels = re.findall(r"RMS level dB: (-?[\d.]+|-inf)", out)
    value = levels[-1] if levels else "-inf"
    return float("-inf") if value == "-inf" else float(value)


def frame_count(path: Path) -> int:
    out = subprocess.run([ffmpeg_bin.ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v",
                          "-f", "null", "-"], capture_output=True, text=True, check=True).stderr
    return int(re.findall(r"frame=\s*(\d+)", out)[-1])


@unittest.skipIf(_ffmpeg() is None, "the bundled ffmpeg is not installed")
class FilmRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.media = Path(cls._tmp.name).resolve() / "media"
        _Synth.video(cls.media / "c97.mp4", 97)
        _Synth.video(cls.media / "c264.mp4", 264)  # 11 s of silent picture
        _Synth.tone(cls.media / "music14.wav", 220, 14, 2.0)  # lavfi sine is 1/8 full scale: x2 is about -15 dB RMS
        _Synth.tone(cls.media / "music6.wav", 220, 6, 2.0)
        _Synth.tone(cls.media / "music1.wav", 220, 1, 2.0)
        _Synth.tone(cls.media / "voice3.wav", 2500, 3, 0.8)  # about -22.6 dB RMS, the rehearsal's narration level

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def project_folder(self) -> Path:
        folder = _folder(Path(tempfile.mkdtemp(dir=self._tmp.name)))
        for item in self.media.iterdir():
            target = folder / "media" / item.name
            target.write_bytes(item.read_bytes())
        return folder

    def project(self, shots, audio=()) -> Project:
        return Project(id="0123456789abcdef", name="Film", defaults=Defaults(width=160, height=90, fps=24),
                       shots=list(shots), audio=list(audio))

    def shot(self, sid: str, clip: str, seconds: float, frames: int) -> Shot:
        take = Take(id=f"t{sid}", clip=f"media/{clip}", measuredSeconds=round(frames / 24, 4),
                    hashes={"clip": (sid * 64)[:64]}, origin="batch")
        return Shot(id=sid, seconds=seconds, takes=[take], activeTake=take.id, status="done")

    def music(self, cid: str, name: str, seconds: float, **fields) -> AudioClip:
        return AudioClip(id=cid, kind="music", source=AudioSource(path=f"media/{name}", measuredSeconds=seconds),
                         start=fields.pop("start", 0.0), **fields)

    # ---- FU-210 ----
    def test_a_97_frame_clip_is_cut_to_the_shots_96_frames(self):
        folder = self.project_folder()
        film = _render(self.project([self.shot(s, "c97.mp4", 4, 97) for s in "abc"]), folder)
        self.assertEqual(frame_count(film), 3 * 96)
        self.assertAlmostEqual(ffmpeg_bin.probe(film)["duration"], 12.0, delta=0.05)

    # ---- FU-211 ----
    def test_a_music_track_longer_than_the_film_fades_out_by_the_films_end(self):
        folder = self.project_folder()
        film = _render(self.project([self.shot("a", "c264.mp4", 11, 264)],
                                    [self.music("m", "music14.wav", 14, fadeOut=3)]), folder)
        self.assertAlmostEqual(ffmpeg_bin.probe(film)["duration"], 11.0, delta=0.05)
        body = rms_db(film, 4.0, 1.0)
        self.assertGreater(body, -20)
        self.assertLess(rms_db(film, 10.5, 0.5), body - 15)  # the last half second is well into the fade
        self.assertLess(rms_db(film, 10.9, 0.1), -45)  # near silence at the cut, not the track's level
        self.assertLess(rms_db(film, 9.5, 0.5), body - 2)  # and the fade started before 11 s, not at 11 s

    def test_a_track_shorter_than_the_film_fades_on_its_own_end(self):
        folder = self.project_folder()
        film = _render(self.project([self.shot("a", "c264.mp4", 11, 264)],
                                    [self.music("m", "music6.wav", 6, fadeOut=2, start=1.0)]), folder)
        body = rms_db(film, 2.0, 1.0)
        self.assertGreater(body, -20)
        self.assertAlmostEqual(rms_db(film, 4.0, 0.5), body, delta=0.5)  # full level until the fade (5 to 7 s)
        self.assertLess(rms_db(film, 6.0, 0.5), body - 4)
        self.assertLess(rms_db(film, 6.85, 0.1), -35)
        self.assertEqual(rms_db(film, 8.0, 1.0), float("-inf"))  # silence after the track

    def test_a_fade_in_starts_at_the_clips_start(self):
        folder = self.project_folder()
        film = _render(self.project([self.shot("a", "c264.mp4", 11, 264)],
                                    [self.music("m", "music14.wav", 14, fadeIn=2)]), folder)
        self.assertLess(rms_db(film, 0.0, 0.1), -35)
        self.assertLess(rms_db(film, 0.5, 0.2), rms_db(film, 1.5, 0.2) - 3)
        self.assertGreater(rms_db(film, 3.0, 1.0), -20)

    def test_both_fades_on_a_clip_shorter_than_them(self):
        folder = self.project_folder()
        film = _render(self.project([self.shot("a", "c264.mp4", 11, 264)],
                                    [self.music("m", "music1.wav", 1, fadeIn=2, fadeOut=2, start=3.0)]), folder)
        self.assertLess(rms_db(film, 3.0, 0.05), -30)  # starts from silence
        self.assertLess(rms_db(film, 3.95, 0.05), -30)  # and ends in it
        self.assertGreater(rms_db(film, 3.4, 0.2), -40)  # with some sound between
        self.assertEqual(rms_db(film, 5.0, 1.0), float("-inf"))

    # ---- FU-212 ----
    def test_music_ducks_under_the_narration(self):
        folder = self.project_folder()
        voice = AudioClip(id="v", kind="voice", source=AudioSource(path="media/voice3.wav", measuredSeconds=3),
                          anchor=Anchor(shotId="a", offset=4.0))
        music = self.music("m", "music14.wav", 14, duck=True)
        film = _render(self.project([self.shot("a", "c264.mp4", 11, 264)], [music, voice]), folder)
        music_band = "bandpass=f=220:width_type=q:w=8"
        alone = rms_db(film, 1.0, 2.0, music_band)
        under = rms_db(film, 4.8, 1.8, music_band)
        # About 10 dB down while the narration (about -22.6 dB RMS) plays: the existing 10:1 sidechain duck.
        self.assertGreater(alone - under, 8, f"music alone {alone:.1f} dB, under the voice {under:.1f} dB")
        self.assertLess(alone - under, 16)
        self.assertAlmostEqual(rms_db(film, 8.5, 1.5, music_band), alone, delta=1.0)  # back up after the release
        # duck: false still turns it off, and the narration's lead over the bed shrinks by the duck amount.
        music.duck = False
        plain = _render(self.project([self.shot("a", "c264.mp4", 11, 264)], [music, voice]),
                        self.project_folder())
        speech_band = "bandpass=f=2500:width_type=q:w=8"
        lead_ducked = rms_db(film, 4.8, 1.8, speech_band) - under
        lead_plain = rms_db(plain, 4.8, 1.8, speech_band) - rms_db(plain, 4.8, 1.8, music_band)
        self.assertGreater(lead_ducked - lead_plain, 8)
        self.assertAlmostEqual(rms_db(plain, 1.0, 2.0, music_band), rms_db(plain, 4.8, 1.8, music_band), delta=1.0)


class DuckDefaultTests(StoreCase):
    def test_new_music_ducks_and_other_kinds_and_explicit_false_do_not(self):
        p = self.new(shots=1)
        p = self.op(p, "add_audio", kind="music", path=str(fake_wav(self.src / "m.wav")))
        p = self.op(p, "add_audio", kind="music", path=str(fake_wav(self.src / "m2.wav")), duck=False)
        p = self.op(p, "add_audio", kind="sfx", path=str(fake_wav(self.src / "s.wav")))
        p = self.op(p, "add_audio", kind="voice", path=str(fake_wav(self.src / "v.wav")))
        self.assertEqual([(c.kind, c.duck) for c in p.audio],
                         [("music", True), ("music", False), ("sfx", False), ("voice", False)])

    def test_a_stored_music_clip_keeps_its_setting(self):
        p = self.new(shots=1)
        p = self.op(p, "add_audio", kind="music", path=str(fake_wav(self.src / "m.wav")), duck=False)
        p = self.store.load(p.id)
        self.assertFalse(p.audio[0].duck)
        p = self.op(p, "edit_audio", clipId=p.audio[0].id, gainDb=-3)
        self.assertFalse(p.audio[0].duck)


if __name__ == "__main__":
    unittest.main()
