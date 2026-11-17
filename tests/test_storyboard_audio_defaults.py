"""Storyboard audio defaults: the last shot's own sound fades out at the film's end (FU-215) and a new voice lane
starts louder so the narration sits clearly over the ducked music bed (FU-216), by what its own file measures (FU-218).

The render checks build a project folder with synthetic media made by the bundled ffmpeg (lavfi sources, 160x90), turn
it into a render plan with ``build_render_plan`` and measure the film's sound with ``astats``. They skip when ffmpeg
is missing.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.storyboard import assemble_graphs, audio_level, ffmpeg_bin, ops
from backend_service.storyboard.assemble import render
from backend_service.storyboard.assemble_graphs import END_FADE_SECONDS, build_mix_graph, end_fade
from backend_service.storyboard.models import Anchor, AudioClip, AudioSource, Defaults, Project, Shot, Take, Transition
from backend_service.storyboard.audio_level import Level
from backend_service.storyboard.ops import MUSIC_GAIN_DB, VOICE_GAIN_DB
from backend_service.storyboard.render_plan import RenderPlan, ShotInput
from backend_service.storyboard.timeline import build_render_plan
from tests.storyboard_fixtures import StoreCase, fake_wav
from tests.test_storyboard_timing_mix import _Synth, _ffmpeg, _folder, rms_db

FLOOR = float("-inf")


def _loud_clip(path: Path, frames: int, volume: float = 3.0, fps: int = 24) -> None:
    """A black clip with a 440 Hz tone (lavfi sine is 1/8 full scale: x3 is about -12.5 dB RMS) for its whole length."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seconds = frames / fps
    _Synth.run("-f", "lavfi", "-i", f"color=c=black:s=160x90:r={fps}", "-f", "lavfi",
               "-i", f"sine=f=440:d={seconds + 0.5}:sample_rate=48000", "-af", f"volume={volume}", "-frames:v", str(frames),
               "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path))


def _plan(project: Project, folder: Path) -> dict:
    plan = build_render_plan(project, preview=True, folder=folder)
    plan["loudness_lufs"] = None  # measure the mix itself, not a loudness-normalised copy
    return plan


def _render(project: Project, folder: Path, **shot_fields) -> Path:
    plan = _plan(project, folder)
    for key, value in shot_fields.items():
        plan["shots"][-1][key] = value
    return Path(render(RenderPlan.from_dict(plan)).path)


class EndFadeMathTests(unittest.TestCase):
    def shot(self, db: float = 0.0) -> ShotInput:
        return ShotInput(id="a", video="/x", hash="h", native_sound_db=db)

    def test_the_fade_ends_where_the_shot_ends_and_is_clamped_to_its_length(self):
        self.assertEqual(end_fade(self.shot(), 8.0, 12.0), (12.0 - END_FADE_SECONDS, END_FADE_SECONDS))
        start, length = end_fade(self.shot(), 8.0, 8.4)
        self.assertAlmostEqual(length, 0.4)
        self.assertAlmostEqual(start, 8.0)

    def test_a_muted_shot_has_no_fade(self):
        self.assertIsNone(end_fade(self.shot(-math.inf), 8.0, 12.0))
        self.assertIsNone(end_fade(self.shot(), 8.0, 8.005))

    def test_the_fade_goes_on_the_shots_own_sound_only(self):
        graph, _ = build_mix_graph([], 48000 * 12, None, native_fade=(11.4, 0.6))
        self.assertIn("atrim=end_sample=576000,afade=t=out:st=11.4:d=0.6[nat]", graph)
        plain, _ = build_mix_graph([], 48000 * 12, None)
        self.assertNotIn("afade", plain)


@unittest.skipIf(_ffmpeg() is None, "the bundled ffmpeg is not installed")
class EndFadeRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.media = Path(cls._tmp.name).resolve() / "media"
        _loud_clip(cls.media / "loud97.mp4", 97)  # 4 s of tone
        _loud_clip(cls.media / "loud49.mp4", 49)  # 2 s
        _loud_clip(cls.media / "loud13.mp4", 13)  # half a second
        _Synth.video(cls.media / "silent97.mp4", 97)  # picture only
        _Synth.tone(cls.media / "music14.wav", 220, 14, 2.0)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def folder(self) -> Path:
        folder = _folder(Path(tempfile.mkdtemp(dir=self._tmp.name)))
        for item in self.media.iterdir():
            (folder / "media" / item.name).write_bytes(item.read_bytes())
        return folder

    def shot(self, sid: str, clip: str, seconds: float, frames: int, **fields) -> Shot:
        take = Take(id=f"t{sid}", clip=f"media/{clip}", measuredSeconds=round(frames / 24, 4),
                    hashes={"clip": (sid * 64)[:64]}, origin="batch")
        return Shot(id=sid, seconds=seconds, takes=[take], activeTake=take.id, status="done", **fields)

    def project(self, shots, audio=()) -> Project:
        return Project(id="0123456789abcdef", name="Film", defaults=Defaults(width=160, height=90, fps=24),
                       shots=list(shots), audio=list(audio))

    def assert_ends_near_silence(self, film: Path, end: float, body_at: float) -> None:
        body = rms_db(film, body_at, 0.3)
        self.assertGreater(body, -20, "the clip's tone is audible before the fade")
        # A linear 0.6 s fade is about 20 dB down over its last tenth of a second and gone at the cut.
        self.assertLess(rms_db(film, end - 0.1, 0.1), body - 18)
        self.assertLess(rms_db(film, end - 0.03, 0.03), body - 30)
        self.assertLess(rms_db(film, end - 0.4, 0.2), body - 2)  # the fade has started well before the end

    def test_a_loud_last_clip_ends_near_silence(self):
        film = _render(self.project([self.shot("a", "loud97.mp4", 4, 97), self.shot("b", "loud97.mp4", 4, 97)]), self.folder())
        self.assertAlmostEqual(ffmpeg_bin.probe(film)["duration"], 8.0, delta=0.05)
        self.assert_ends_near_silence(film, 8.0, 6.0)
        # the first shot's end is a cut, not a fade: its sound runs right up to the join
        self.assertGreater(rms_db(film, 3.8, 0.15), -20)

    def test_a_one_shot_film_works(self):
        film = _render(self.project([self.shot("a", "loud97.mp4", 4, 97)]), self.folder())
        self.assert_ends_near_silence(film, 4.0, 2.0)

    def test_a_speed_changed_last_shot(self):
        film = _render(self.project([self.shot("a", "loud97.mp4", 4, 97), self.shot("b", "loud97.mp4", 4, 97, speed=2.0)]),
                       self.folder())
        self.assertAlmostEqual(ffmpeg_bin.probe(film)["duration"], 6.0, delta=0.05)  # 4 s of source at 2x is 2 s
        self.assert_ends_near_silence(film, 6.0, 4.5)

    def test_a_trimmed_last_shot(self):
        film = _render(self.project([self.shot("a", "loud97.mp4", 4, 97),
                                     self.shot("b", "loud97.mp4", 2, 97, trimIn=0.5, trimOut=2.5)]), self.folder())
        self.assertAlmostEqual(ffmpeg_bin.probe(film)["duration"], 6.0, delta=0.05)
        self.assert_ends_near_silence(film, 6.0, 4.5)

    def test_a_last_shot_shorter_than_the_fade_fades_over_its_whole_length(self):
        film = _render(self.project([self.shot("a", "loud97.mp4", 4, 97), self.shot("b", "loud13.mp4", 0.5, 13)]),
                       self.folder())
        self.assertAlmostEqual(ffmpeg_bin.probe(film)["duration"], 4.5, delta=0.05)
        self.assertLess(rms_db(film, 4.4, 0.1), -30)  # a 0.5 s shot fades over its own 0.5 s
        self.assertGreater(rms_db(film, 3.8, 0.15), -20)  # the shot before it is not touched

    def test_a_crossfaded_last_shot_still_ends_near_silence(self):
        shots = [self.shot("a", "loud97.mp4", 4, 97), self.shot("b", "loud97.mp4", 4, 97, transition=Transition(type="crossfade", seconds=1.0))]
        film = _render(self.project(shots), self.folder())
        self.assertAlmostEqual(ffmpeg_bin.probe(film)["duration"], 7.0, delta=0.05)
        self.assert_ends_near_silence(film, 7.0, 5.0)

    def test_a_silent_last_shot_is_unchanged(self):
        project = self.project([self.shot("a", "loud97.mp4", 4, 97), self.shot("b", "silent97.mp4", 4, 97)])
        with_fade = _render(project, self.folder())
        with mock.patch.object(assemble_graphs, "END_FADE_SECONDS", 0.0):
            without = _render(project, self.folder())
        self.assertEqual(self.pcm(with_fade), self.pcm(without))
        self.assertEqual(rms_db(with_fade, 7.0, 0.5), FLOOR)

    def test_a_muted_last_shot_is_unchanged(self):
        project = self.project([self.shot("a", "loud97.mp4", 4, 97), self.shot("b", "loud97.mp4", 4, 97)])
        with_fade = _render(project, self.folder(), native_sound_db="-inf")
        with mock.patch.object(assemble_graphs, "END_FADE_SECONDS", 0.0):
            without = _render(project, self.folder(), native_sound_db="-inf")
        self.assertEqual(self.pcm(with_fade), self.pcm(without))
        self.assertEqual(rms_db(with_fade, 6.0, 1.0), FLOOR)

    def test_music_and_voice_lanes_keep_their_own_fades(self):
        # A music track with no fade of its own plays on to the cut: only the shot's sound is faded.
        music = AudioClip(id="m", kind="music", source=AudioSource(path="media/music14.wav", measuredSeconds=14), start=0.0)
        film = _render(self.project([self.shot("a", "silent97.mp4", 4, 97), self.shot("b", "silent97.mp4", 4, 97)], [music]),
                       self.folder())
        self.assertGreater(rms_db(film, 7.8, 0.15), -20)

    @staticmethod
    def pcm(film: Path) -> bytes:
        return subprocess.run([ffmpeg_bin.ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(film), "-vn", "-f", "s16le", "-"],
                              capture_output=True, check=True).stdout


class VoiceDefaultTests(StoreCase):
    def lanes(self, p):
        return [(c.kind, c.gainDb) for c in p.audio]

    def test_new_voice_lanes_start_louder_and_other_kinds_do_not(self):
        # The test files are not real audio, so they cannot be measured and the voice lane gets the fallback gain.
        p = self.new(shots=1)
        for kind in ("voice", "music", "sfx", "import"):
            p = self.op(p, "add_audio", kind=kind, path=str(fake_wav(self.src / f"{kind}.wav")))
        self.assertEqual(self.lanes(p), [("voice", VOICE_GAIN_DB), ("music", MUSIC_GAIN_DB), ("sfx", 0.0), ("import", 0.0)])
        self.assertGreater(VOICE_GAIN_DB, 0)

    def test_the_order_of_voice_and_music_does_not_matter(self):
        a = self.new(shots=1)
        a = self.op(a, "add_audio", kind="music", path=str(fake_wav(self.src / "m.wav")))
        a = self.op(a, "add_audio", kind="voice", path=str(fake_wav(self.src / "v.wav")))
        b = self.new(shots=1)
        b = self.op(b, "add_audio", kind="voice", path=str(fake_wav(self.src / "v2.wav")))
        b = self.op(b, "add_audio", kind="music", path=str(fake_wav(self.src / "m2.wav")))
        self.assertEqual(sorted(self.lanes(a)), sorted(self.lanes(b)))

    def test_explicit_values_win(self):
        p = self.new(shots=1)
        for n, gain in enumerate((0.0, -3.0, 12.0)):
            p = self.op(p, "add_audio", kind="voice", path=str(fake_wav(self.src / f"v{n}.wav")), gainDb=gain)
        self.assertEqual([c.gainDb for c in p.audio], [0.0, -3.0, 12.0])

    def test_a_voice_lane_already_in_a_project_keeps_its_gain(self):
        p = self.new(shots=1)
        p = self.op(p, "add_audio", kind="voice", path=str(fake_wav(self.src / "v.wav")), gainDb=0.0)
        old = self.store.load(p.id)
        self.assertEqual(old.audio[0].gainDb, 0.0)
        p = self.op(old, "add_audio", kind="music", path=str(fake_wav(self.src / "m.wav")))
        p = self.op(p, "edit_audio", clipId=p.audio[0].id, fadeOut=1.0)
        self.assertEqual(self.store.load(p.id).audio[0].gainDb, 0.0)

    def test_a_song_added_to_the_mixer_keeps_unity_gain(self):
        p = self.new(shots=1)
        p = self.op(p, "set_song", path=str(fake_wav(self.src / "song.wav")), title="Song", seconds=2.0, addToMixer=True)
        self.assertEqual([(c.kind, c.gainDb) for c in p.audio], [("music", 0.0)])


class VoiceGainMathTests(unittest.TestCase):
    """``gain_for``: the gain that brings a narration to the target, within the limits (FU-218)."""

    def gain(self, rms: float, peak: float) -> float:
        return audio_level.gain_for(Level(rms, peak))

    def test_a_quiet_file_is_boosted_more_than_a_louder_one(self):
        quiet, louder = self.gain(-24.0, -12.0), self.gain(-19.0, -6.0)
        self.assertGreater(quiet, louder)
        self.assertAlmostEqual(quiet, audio_level.TARGET_RMS_DB + 24.0, delta=0.1)
        self.assertAlmostEqual(louder, audio_level.TARGET_RMS_DB + 19.0, delta=0.1)

    def test_a_file_at_or_over_the_target_is_never_attenuated(self):
        self.assertEqual(self.gain(audio_level.TARGET_RMS_DB, -3.0), 0.0)
        self.assertEqual(self.gain(-8.0, -1.0), 0.0)

    def test_the_boost_is_capped(self):
        self.assertEqual(self.gain(-50.0, -40.0), audio_level.MAX_GAIN_DB)
        self.assertLessEqual(audio_level.MAX_GAIN_DB, 12.0)  # AudioClip.gainDb cannot hold more

    def test_the_peak_ceiling_limits_the_boost(self):
        peak = audio_level.PEAK_CEILING_DB - 3.0
        self.assertEqual(self.gain(-40.0, peak), 3.0)  # the target alone would call for the cap
        self.assertEqual(self.gain(-40.0, audio_level.PEAK_CEILING_DB + 2.0), 0.0)  # already over: no boost, no cut

    def test_a_file_that_could_not_be_measured_gets_the_fallback(self):
        self.assertEqual(audio_level.gain_for(None), audio_level.FALLBACK_GAIN_DB)
        self.assertEqual(VOICE_GAIN_DB, audio_level.FALLBACK_GAIN_DB)


@unittest.skipIf(_ffmpeg() is None, "the bundled ffmpeg is not installed")
class VoiceLevelMeasuredTests(StoreCase):
    """A new voice lane's gain comes from the real file, measured with the bundled ffmpeg when the lane is created."""

    def setUp(self):
        super().setUp()
        # lavfi sine is 1/8 full scale (-21 dB RMS at volume 1): 0.6 is a quiet narration, 3 a loud one
        for name, volume in (("quiet", 0.6), ("mid", 1.0), ("loud", 3.0)):
            _Synth.tone(self.src / f"{name}.wav", 440, 2.0, volume)

    def add(self, p, name, **fields):
        return self.op(p, "add_audio", kind="voice", path=str(self.src / f"{name}.wav"), **fields)

    def test_the_measurement_matches_the_file(self):
        level = audio_level.measure(self.src / "mid.wav")
        self.assertAlmostEqual(level.rms_db, -21.0, delta=0.3)
        self.assertAlmostEqual(level.peak_db, -18.0, delta=0.3)

    def test_a_quiet_file_gets_more_gain_than_a_louder_one_and_a_loud_one_none(self):
        p = self.new(shots=1)
        for name in ("quiet", "mid", "loud"):
            p = self.add(p, name)
        quiet, mid, loud = (c.gainDb for c in p.audio)
        self.assertGreater(quiet, mid)
        self.assertGreater(mid, 0)
        self.assertEqual(loud, 0.0)
        self.assertAlmostEqual(mid, audio_level.TARGET_RMS_DB + 21.0, delta=0.5)

    def test_a_clip_is_measured_once_at_creation_and_never_again(self):
        p = self.new(shots=1)
        p = self.add(p, "quiet")
        gain = p.audio[0].gainDb
        with mock.patch.object(audio_level, "measure", side_effect=AssertionError("measured again")):
            p = self.op(p, "edit_audio", clipId=p.audio[0].id, fadeOut=1.0)
            p = self.op(p, "edit_audio", clipId=p.audio[0].id, gainDb=-2.0)
            self.assertEqual(self.store.load(p.id).audio[0].gainDb, -2.0)
        self.assertGreater(gain, 0)

    def test_an_explicit_gain_wins_and_is_not_measured(self):
        p = self.new(shots=1)
        with mock.patch.object(audio_level, "measure", side_effect=AssertionError("measured")):
            for name, gain in (("quiet", 0.0), ("quiet", -3.0), ("loud", 6.0)):
                p = self.add(p, name, gainDb=gain)
        self.assertEqual([c.gainDb for c in p.audio], [0.0, -3.0, 6.0])

    def test_only_a_voice_lane_is_measured(self):
        p = self.new(shots=1)
        with mock.patch.object(audio_level, "measure", side_effect=AssertionError("measured")):
            p = self.op(p, "add_audio", kind="music", path=str(self.src / "quiet.wav"))
        self.assertEqual(p.audio[0].gainDb, MUSIC_GAIN_DB)

    def test_a_measurement_that_fails_falls_back_to_the_fixed_gain(self):
        p = self.new(shots=1)
        with mock.patch.object(audio_level, "measure", return_value=None):
            p = self.add(p, "quiet")
        self.assertEqual(p.audio[0].gainDb, VOICE_GAIN_DB)
        with mock.patch.object(ops.media, "ffmpeg_exe", return_value=None):  # no ffmpeg at all
            self.assertIsNone(audio_level.measure(self.src / "quiet.wav"))
            p = self.add(p, "quiet")
        self.assertEqual(p.audio[1].gainDb, VOICE_GAIN_DB)

    def test_a_silent_file_gets_the_fallback(self):
        _Synth.tone(self.src / "silent.wav", 440, 2.0, 0.0)
        self.assertIsNone(audio_level.measure(self.src / "silent.wav"))
        p = self.add(self.new(shots=1), "silent")
        self.assertEqual(p.audio[0].gainDb, VOICE_GAIN_DB)

    def test_the_measurement_is_quick(self):
        import time

        start = time.perf_counter()
        audio_level.measure(self.src / "quiet.wav")
        self.assertLess(time.perf_counter() - start, 2.0)


@unittest.skipIf(_ffmpeg() is None, "the bundled ffmpeg is not installed")
class NarrationOverBedRenderTests(unittest.TestCase):
    """The narration (a 2.5 kHz tone at about -22.6 dB RMS) over a music bed (220 Hz at about -15 dB RMS), through the
    export graph with the music ducking under the voice, as a new project's lanes would be mixed."""

    BED = "bandpass=f=220:width_type=q:w=8"
    SPEECH = "bandpass=f=2500:width_type=q:w=8"

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.media = Path(cls._tmp.name).resolve() / "media"
        _Synth.video(cls.media / "c264.mp4", 264)
        _Synth.tone(cls.media / "music14.wav", 220, 14, 2.0)
        _Synth.tone(cls.media / "voice3.wav", 2500, 3, 0.8)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def film(self, voice_gain: float, music_gain: float = 0.0, loudness: float | None = None) -> Path:
        folder = _folder(Path(tempfile.mkdtemp(dir=self._tmp.name)))
        for item in self.media.iterdir():
            (folder / "media" / item.name).write_bytes(item.read_bytes())
        take = Take(id="ta", clip="media/c264.mp4", measuredSeconds=11.0, hashes={"clip": "a" * 64}, origin="batch")
        shot = Shot(id="a", seconds=11, takes=[take], activeTake="ta", status="done")
        voice = AudioClip(id="v", kind="voice", source=AudioSource(path="media/voice3.wav", measuredSeconds=3),
                          anchor=Anchor(shotId="a", offset=4.0), gainDb=voice_gain)
        music = AudioClip(id="m", kind="music", source=AudioSource(path="media/music14.wav", measuredSeconds=14),
                          start=0.0, duck=True, gainDb=music_gain)
        project = Project(id="0123456789abcdef", name="Film", defaults=Defaults(width=160, height=90, fps=24),
                          shots=[shot], audio=[music, voice])
        plan = build_render_plan(project, preview=True, folder=folder)
        plan["loudness_lufs"] = loudness
        return Path(render(RenderPlan.from_dict(plan)).path)

    def lead(self, film: Path) -> float:
        """How far the narration sits over the ducked bed while it speaks (4.8 to 6.6 s), in dB."""
        return rms_db(film, 4.8, 1.8, self.SPEECH) - rms_db(film, 4.8, 1.8, self.BED)

    def default_gain(self) -> float:
        """What a new voice lane would get for this narration (it measures about -22.6 dB RMS)."""
        return audio_level.voice_gain_db(self.media / "voice3.wav")

    def test_the_new_voice_default_puts_the_narration_10_db_over_the_ducked_bed(self):
        before = self.lead(self.film(0.0))
        after = self.lead(self.film(self.default_gain()))
        self.assertLess(before, 10, f"unity gain already leads by {before:.1f} dB: the default would be needless")
        self.assertGreaterEqual(after, 10, f"narration leads the ducked bed by {after:.1f} dB (was {before:.1f} dB)")

    def test_the_lead_survives_the_loudness_normaliser(self):
        film = self.film(self.default_gain(), loudness=-16.0)
        self.assertGreaterEqual(self.lead(film), 10)
        # The bed away from the narration is not pushed down to nothing by the louder voice.
        self.assertGreater(rms_db(film, 1.0, 2.0, self.BED), -30)

    def test_a_music_only_film_is_as_loud_as_before(self):
        folder = _folder(Path(tempfile.mkdtemp(dir=self._tmp.name)))
        for item in self.media.iterdir():
            (folder / "media" / item.name).write_bytes(item.read_bytes())
        take = Take(id="ta", clip="media/c264.mp4", measuredSeconds=11.0, hashes={"clip": "a" * 64}, origin="batch")
        shot = Shot(id="a", seconds=11, takes=[take], activeTake="ta", status="done")
        music = AudioClip(id="m", kind="music", source=AudioSource(path="media/music14.wav", measuredSeconds=14), start=0.0, duck=True)
        plan = build_render_plan(Project(id="0123456789abcdef", name="Film", defaults=Defaults(width=160, height=90, fps=24),
                                         shots=[shot], audio=[music]), preview=True, folder=folder)
        plan["loudness_lufs"] = None
        film = Path(render(RenderPlan.from_dict(plan)).path)
        # The music's own level (a mono 220 Hz tone, -18.1 dB RMS once spread over two channels), as it always was.
        self.assertAlmostEqual(rms_db(film, 2.0, 2.0), -18.1, delta=0.5)


if __name__ == "__main__":
    unittest.main()
