"""Storyboard join-and-mix engine: render plan validation, pure graph maths, and real ffmpeg renders.

All media is generated with ffmpeg's lavfi sources (tiny 160x90 clips of 0.5 to 3 s), so the suite
needs only the pinned imageio-ffmpeg binary and stays fast. Render tests skip, not fail, when that
binary is missing or lacks a filter they need.
"""

from __future__ import annotations

import array
import math
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers.output_metadata import read_mp4_settings
from backend_service.storyboard import assemble, ffmpeg_bin
from backend_service.storyboard.assemble import (
    RenderCancelled,
    atempo_chain,
    build_mix_graph,
    build_shot_graph,
    compute_join_offsets,
    concat_list,
    parse_loudnorm,
    render,
)
from backend_service.storyboard.render_plan import (
    AudioInput,
    PlanError,
    RenderPlan,
    ShotInput,
    Transition,
)

SR = 48000


def _ffmpeg() -> str | None:
    try:
        return ffmpeg_bin.ffmpeg_exe()
    except RuntimeError:
        return None


class PureMathTests(unittest.TestCase):
    def test_cuts_add_up(self):
        timing = compute_join_offsets([2.0, 1.5, 1.0], [0, 0, 0])
        self.assertEqual(timing.starts, [0.0, 2.0, 3.5])
        self.assertAlmostEqual(timing.total, 4.5)

    def test_crossfade_offsets_use_the_shortened_film(self):
        # Shot 1 starts 0.5 s before shot 0 ends; shot 2's xfade offset is measured on the
        # already-shortened 3.0 s film, so it is 3.0 - 0.25, not 3.5 - 0.25.
        timing = compute_join_offsets([2.0, 1.5, 1.0], [0, 0.5, 0.25])
        self.assertEqual(timing.starts, [0.0, 1.5, 2.75])
        self.assertAlmostEqual(timing.total, 3.75)

    def test_mixed_cut_and_crossfade(self):
        timing = compute_join_offsets(
            [1.0, 1.0, 1.0, 1.0], [Transition(), Transition("crossfade", 0.2), Transition(), Transition("crossfade", 0.4)])
        self.assertEqual([round(s, 6) for s in timing.starts], [0.0, 0.8, 1.8, 2.4])
        self.assertAlmostEqual(timing.total, 3.4)

    def test_first_transition_is_ignored(self):
        self.assertAlmostEqual(compute_join_offsets([1.0, 1.0], [0.9, 0]).total, 2.0)

    def test_crossfades_longer_than_a_shot_are_refused(self):
        with self.assertRaises(ValueError):
            compute_join_offsets([1.0, 0.5], [0, 0.6])
        with self.assertRaises(ValueError):  # 0.3 in + 0.3 out > 0.5
            compute_join_offsets([1.0, 0.5, 1.0], [0, 0.3, 0.3])
        with self.assertRaises(ValueError):
            compute_join_offsets([1.0], [0, 0])

    def test_atempo_chain_stays_in_range(self):
        self.assertEqual(atempo_chain(1.0), [])
        self.assertEqual(atempo_chain(1.5), [1.5])
        self.assertEqual(atempo_chain(4.0), [2.0, 2.0])
        self.assertEqual(atempo_chain(0.25), [0.5, 0.5])
        for speed in (0.3, 0.7, 2.5, 3.0, 3.9):
            chain = atempo_chain(speed)
            self.assertTrue(all(0.5 <= f <= 2.0 for f in chain), chain)
            self.assertAlmostEqual(math.prod(chain), speed)
        with self.assertRaises(ValueError):
            atempo_chain(0)

    def test_concat_list_only_takes_plain_names(self):
        self.assertEqual(concat_list(["shot-ab.mov", "card-1.mov"]), "file 'shot-ab.mov'\nfile 'card-1.mov'\n")
        for bad in ("../x.mov", "/abs/x.mov", ".hidden.mov", "a'b.mov", "a b.mov", "x\nfile y"):
            with self.assertRaises(ValueError):
                concat_list([bad])

    def test_parse_probe(self):
        text = """Input #0, mov,mp4, from 'x.mp4':
  Duration: 00:01:02.50, start: 0.000000, bitrate: 39 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 160x90 [SAR 1:1 DAR 16:9], 30 kb/s, 23.98 fps, 24 tbr, 12288 tbn (default)
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, mono, fltp, 2 kb/s (default)
"""
        info = ffmpeg_bin.parse_probe(text)
        self.assertEqual(info, {"duration": 62.5, "has_audio": True, "has_video": True,
                                "width": 160, "height": 90, "fps": 23.98})
        with self.assertRaises(ValueError):
            ffmpeg_bin.parse_probe("x.mp4: Invalid data found when processing input")

    def test_parse_loudnorm(self):
        text = '[Parsed_loudnorm_9 @ 0x1]\n{\n\t"input_i" : "-20.52",\n\t"input_tp" : "-inf"\n}\n'
        self.assertEqual(parse_loudnorm(text), {"input_i": "-20.52", "input_tp": "-inf"})
        self.assertIsNone(parse_loudnorm("no json here"))


class _Media:
    """Tiny generated clips shared by the plan and render tests."""

    dir: Path

    @classmethod
    def make(cls, root: Path) -> None:
        exe = ffmpeg_bin.ffmpeg_exe()
        cls.dir = root

        def run(*args: str) -> None:
            subprocess.run([exe, "-v", "error", "-nostdin", "-y", *args], check=True)

        h264 = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
        run("-f", "lavfi", "-i", "testsrc2=s=160x90:r=24:d=2", "-f", "lavfi", "-i", "sine=f=440:d=2",
            *h264, "-c:a", "aac", "-shortest", str(root / "a.mp4"))
        run("-f", "lavfi", "-i", "color=c=red:s=120x90:r=30:d=1.5", *h264, str(root / "silent.mp4"))
        run("-f", "lavfi", "-i", "testsrc2=s=200x90:r=25:d=1", "-f", "lavfi", "-i", "sine=f=660:d=1",
            *h264, "-c:a", "aac", "-shortest", str(root / "c.mp4"))
        run("-f", "lavfi", "-i", "color=c=blue:s=160x90:r=20:d=0.5", *h264, str(root / "tiny.mp4"))
        run("-f", "lavfi", "-i", "color=c=black:s=160x90:r=24:d=3", *h264, str(root / "quiet3.mp4"))
        run("-f", "lavfi", "-i", "sine=f=220:d=3:sample_rate=44100", str(root / "music.wav"))
        run("-f", "lavfi", "-i", "sine=f=220:d=1", str(root / "music1s.wav"))
        # lavfi's sine has amplitude 1/8; the "voice" is raised to 0.5, a plain speaking level.
        run("-f", "lavfi", "-i", "sine=f=2500:d=1", "-af", "volume=4", str(root / "voice.wav"))
        (root / "caps.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")


@unittest.skipIf(_ffmpeg() is None, "the pinned imageio-ffmpeg binary is not installed")
class RenderPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name).resolve()
        _Media.make(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def plan(self, **over):
        data = {"width": 160, "height": 90, "fps": 24, "out_path": str(self.root / "o.mp4"),
                "work_dir": str(self.root / "w"),
                "shots": [{"id": "a", "video": str(self.root / "a.mp4"), "hash": "h"}]}
        data.update(over)
        return data

    def test_round_trip(self):
        plan = RenderPlan.from_dict(self.plan(
            audio=[{"id": "m", "kind": "music", "path": str(self.root / "music.wav")}],
            shots=[{"id": "a", "video": str(self.root / "a.mp4"), "hash": "h", "native_sound_db": "-inf",
                    "transition_in": {"type": "crossfade", "seconds": 0.5}}],
            end_card={"text": "AI-generated", "seconds": 2}))
        self.assertTrue(plan.audio[0].duck, "music ducks by default")
        self.assertEqual(plan.shots[0].native_sound_db, -math.inf)
        again = RenderPlan.from_dict(plan.to_dict())
        self.assertEqual(again, plan)

    def test_hostile_paths_rejected(self):
        bad = ["-i.mp4", str(self.root / "a.mp4") + "\nfile x", str(self.root / "missing.mp4"),
               "http://x", "file:/etc/passwd", "a.mp4", "concat:/a|/b", str(self.root)]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(PlanError):
                RenderPlan.from_dict(self.plan(shots=[{"id": "a", "video": value, "hash": "h"}]))
            with self.subTest(audio=value), self.assertRaises(PlanError):
                RenderPlan.from_dict(self.plan(audio=[{"id": "m", "kind": "music", "path": value}]))
        for out in ("-o.mp4", "rel.mp4", str(self.root / "o.mkv"), str(self.root / "nope" / "o.mp4"),
                    "http://x/o.mp4"):
            with self.subTest(out=out), self.assertRaises(PlanError):
                RenderPlan.from_dict(self.plan(out_path=out))
        with self.assertRaises(PlanError):
            RenderPlan.from_dict(self.plan(work_dir="relative/work"))
        with self.assertRaises(PlanError):
            RenderPlan.from_dict(self.plan(captions_srt=str(self.root / "a.mp4")))

    def test_bad_values_rejected(self):
        shot = {"id": "a", "video": str(self.root / "a.mp4"), "hash": "h"}
        cases = [
            {"width": 161}, {"fps": 0}, {"shots": []},
            {"shots": [dict(shot, speed=10)]}, {"shots": [dict(shot, trim_in=1, trim_out=0.5)]},
            {"shots": [dict(shot, hash="a;b")]}, {"shots": [dict(shot, id="x[0]")]},
            {"shots": [shot, shot]},
            {"shots": [dict(shot, transition_in={"type": "wipe", "seconds": 1})]},
            {"shots": [dict(shot, transition_in={"type": "crossfade", "seconds": 0})]},
            {"shots": [dict(shot, native_sound_db=float("nan"))]},
            {"audio": [{"id": "m", "kind": "drums", "path": str(self.root / "music.wav")}]},
            {"end_card": {"text": "bad\x1b[2J", "seconds": 2}}, {"end_card": {"text": "x" * 201}},
            {"end_card": {"text": "  "}}, {"metadata": {"k": 1}}, {"loudness_lufs": 3},
        ]
        for over in cases:
            with self.subTest(over=over), self.assertRaises(PlanError):
                RenderPlan.from_dict(self.plan(**over))


@unittest.skipIf(_ffmpeg() is None, "the pinned imageio-ffmpeg binary is not installed")
class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name).resolve()
        _Media.make(cls.root)
        cls.filters = ffmpeg_bin.ffmpeg_filters()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.case = Path(tempfile.mkdtemp(dir=self.root))

    def need(self, *names: str) -> None:
        missing = sorted(set(names) - set(self.filters))
        if missing:
            self.skipTest(f"pinned ffmpeg lacks {', '.join(missing)}")

    def shot(self, sid: str, name: str = "a.mp4", **over):
        return dict({"id": sid, "video": str(self.root / name), "hash": f"hash-{sid}-{name}"}, **over)

    def plan(self, shots, **over) -> RenderPlan:
        data = {"width": 160, "height": 90, "fps": 24, "preview": True, "loudness_lufs": None,
                "out_path": str(self.case / "film.mp4"), "work_dir": str(self.case / "work"),
                "shots": shots}
        data.update(over)
        return RenderPlan.from_dict(data)

    def decode(self, path: str | Path, audio_filter: str | None = None) -> array.array:
        """The film's audio as mono 48 kHz float samples."""
        args = [ffmpeg_bin.ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path), "-vn"]
        if audio_filter:
            args += ["-af", audio_filter]
        args += ["-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
        raw = subprocess.run(args, capture_output=True, check=True).stdout
        samples = array.array("f")
        samples.frombytes(raw[: len(raw) // 4 * 4])
        return samples

    @staticmethod
    def rms(samples: array.array, start: float, end: float) -> float:
        window = samples[int(start * SR): int(end * SR)]
        return math.sqrt(sum(x * x for x in window) / max(1, len(window)))

    def test_probe(self):
        info = ffmpeg_bin.probe(self.root / "a.mp4")
        self.assertAlmostEqual(info["duration"], 2.0, delta=0.05)
        self.assertTrue(info["has_audio"])
        self.assertEqual((info["width"], info["height"], info["fps"]), (160, 90, 24.0))
        self.assertFalse(ffmpeg_bin.probe(self.root / "silent.mp4")["has_audio"])
        music = ffmpeg_bin.probe(self.root / "music.wav")
        self.assertTrue(music["has_audio"])
        self.assertFalse(music["has_video"])
        with self.assertRaises(ValueError):
            ffmpeg_bin.probe(self.root / "caps.srt")

    def test_three_shot_cut(self):
        plan = self.plan([self.shot("a"), self.shot("b", "silent.mp4"), self.shot("c", "c.mp4")],
                         metadata={"app": "ChaosEngineAI", "kind": "storyboard"})
        stages = []
        result = render(plan, progress_cb=lambda stage, f: stages.append((stage, f)))
        self.assertTrue(Path(result.path).is_file())
        self.assertAlmostEqual(result.duration_s, 2.0 + 1.5 + 1.0, delta=0.1)
        info = ffmpeg_bin.probe(result.path)
        self.assertTrue(info["has_audio"])
        self.assertEqual((info["width"], info["height"]), (160, 90))
        self.assertEqual((result.shots_rendered, result.shots_cached), (3, 0))
        self.assertEqual(read_mp4_settings(Path(result.path)), {"app": "ChaosEngineAI", "kind": "storyboard"})
        self.assertEqual({s for s, _ in stages}, {"shots", "join", "mix"})
        self.assertTrue(all(0.0 <= f <= 1.0 for _, f in stages))
        self.assertIn(("mix", 1.0), stages)
        # Only cached intermediates stay in the work folder; scripts and lists are cleaned up.
        self.assertTrue(all(p.name.startswith("shot-") for p in Path(plan.work_dir).iterdir()))

    def test_crossfade_duration(self):
        self.need("xfade", "acrossfade")
        fade = {"type": "crossfade", "seconds": 0.5}
        plan = self.plan([self.shot("a"), self.shot("b", "silent.mp4", transition_in=fade),
                          self.shot("c", "c.mp4", transition_in={"type": "crossfade", "seconds": 0.25}),
                          self.shot("d", "a.mp4")])
        result = render(plan)
        expected = 2.0 + 1.5 + 1.0 + 2.0 - 0.5 - 0.25
        self.assertAlmostEqual(result.duration_s, expected, delta=0.1)
        self.assertAlmostEqual(len(self.decode(result.path)) / SR, expected, delta=0.1)

    def test_clip_without_audio_gets_silence(self):
        plan = self.plan([self.shot("b", "silent.mp4")])
        result = render(plan)
        self.assertTrue(ffmpeg_bin.probe(result.path)["has_audio"])
        samples = self.decode(result.path)
        self.assertAlmostEqual(len(samples) / SR, 1.5, delta=0.1)
        self.assertLess(self.rms(samples, 0, 1.5), 1e-4)

    def test_cache_reuse_and_single_shot_redo(self):
        shots = [self.shot("a"), self.shot("b", "silent.mp4"), self.shot("c", "c.mp4")]

        def timed(plan):
            marks = {}

            def cb(stage, fraction):
                marks.setdefault(stage, time.perf_counter())
            result = render(plan, progress_cb=cb)
            return result, marks["join"] - marks["shots"]

        first, first_build = timed(self.plan(shots))
        work = Path(self.case / "work")
        before = {p.name: p.stat().st_mtime_ns for p in work.iterdir()}
        second, second_build = timed(self.plan(shots))
        self.assertEqual((second.shots_rendered, second.shots_cached), (0, 3))
        self.assertLess(second_build, first_build)
        shots[1] = dict(shots[1], hash="redo-take-2")
        third = render(self.plan(shots))
        self.assertEqual((third.shots_rendered, third.shots_cached), (1, 2))
        after = {p.name: p.stat().st_mtime_ns for p in work.iterdir()}
        self.assertEqual(len(after), 4)
        for name, mtime in before.items():
            self.assertEqual(after[name], mtime, "untouched shots keep their intermediate bytes")
        self.assertAlmostEqual(third.duration_s, first.duration_s, delta=0.05)

    def test_music_ducks_under_voice_and_loudnorm_two_pass(self):
        self.need("sidechaincompress", "asplit", "loudnorm", "bandpass")
        audio = [{"id": "m", "kind": "music", "path": str(self.root / "music.wav"), "start": 0},
                 {"id": "v", "kind": "voice", "path": str(self.root / "voice.wav"), "start": 1.0}]
        plan = self.plan([self.shot("q", "quiet3.mp4")], audio=audio, preview=False, loudness_lufs=-16.0)
        result = render(plan)
        self.assertEqual(result.loudness["passes"], 2)
        self.assertIn("input_i", result.loudness["measured"])
        self.assertIsNotNone(result.loudness["normalization_type"])
        # Measure only the music: a narrow band-pass at 220 Hz removes the 2.5 kHz voice. Without ducking
        # the music level is the same in both windows (ratio about 1, checked below); a 10:1 compressor
        # keyed by a voice far above its threshold pulls it down by well over 6 dB, so ask for 2x (6 dB).
        music_only = "bandpass=f=220:width_type=q:w=8"
        samples = self.decode(result.path, music_only)
        alone, under = self.rms(samples, 0.3, 0.9), self.rms(samples, 1.3, 1.9)
        self.assertGreater(alone, 2.0 * under, f"music alone {alone:.4f}, under voice {under:.4f}")
        # Control: the same film without ducking keeps the music level.
        audio[0]["duck"] = False
        plain = render(self.plan([self.shot("q", "quiet3.mp4")], audio=audio, preview=False,
                                 loudness_lufs=-16.0, out_path=str(self.case / "plain.mp4")))
        samples = self.decode(plain.path, music_only)
        ratio = self.rms(samples, 0.3, 0.9) / self.rms(samples, 1.3, 1.9)
        self.assertAlmostEqual(ratio, 1.0, delta=0.2)

    def test_loop_and_trim(self):
        self.need("aloop")
        audio = [{"id": "m", "kind": "music", "path": str(self.root / "music1s.wav"), "loop": True},
                 {"id": "s", "kind": "sfx", "path": str(self.root / "voice.wav"), "start": 1.0,
                  "trim_in": 0.25, "duration": 0.5}]
        shots = [self.shot("a", trim_in=0.5, trim_out=1.5), self.shot("q", "quiet3.mp4", trim_out=2.0),
                 self.shot("z", "silent.mp4", speed=1.5)]
        result = render(self.plan(shots, audio=audio))
        self.assertAlmostEqual(result.duration_s, 1.0 + 2.0 + 1.0, delta=0.1)
        music = self.decode(result.path, "bandpass=f=220:width_type=q:w=8")
        # A 1 s music clip looped over 4 s: still playing in the last second.
        self.assertGreater(self.rms(music, 3.2, 3.9), 0.05)
        effect = self.decode(result.path, "highpass=f=2000,highpass=f=2000")
        self.assertGreater(self.rms(effect, 1.1, 1.4), 0.05)
        self.assertLess(self.rms(effect, 1.7, 3.9), 0.01)

    def test_cancel_leaves_no_output(self):
        event = threading.Event()

        def cb(stage, fraction):
            if stage == "mix":
                event.set()
        plan = self.plan([self.shot("a"), self.shot("c", "c.mp4")])
        with self.assertRaises(RenderCancelled):
            render(plan, progress_cb=cb, cancel_event=event)
        self.assertFalse(Path(plan.out_path).exists())
        self.assertEqual([p.name for p in self.case.iterdir()], ["work"])
        self.assertTrue(all(p.name.startswith("shot-") for p in (self.case / "work").iterdir()))

    def test_cancel_during_shots(self):
        event = threading.Event()
        event.set()
        with self.assertRaises(RenderCancelled):
            render(self.plan([self.shot("a")]), cancel_event=event)
        self.assertFalse((self.case / "film.mp4").exists())
        self.assertEqual(list((self.case / "work").iterdir()), [])

    def test_twenty_shots_use_script_files(self):
        self.need("xfade", "acrossfade")
        shots = []
        fades = 0
        for i in range(20):
            transition = {"type": "crossfade", "seconds": 0.1} if i % 2 else None
            fades += 1 if transition else 0
            shot = self.shot(f"s{i:02d}", "tiny.mp4" if i % 3 else "c.mp4")
            if transition:
                shot["transition_in"] = transition
            shots.append(shot)
        commands: list[list[str]] = []
        real_run = assemble._run

        def spy(args, **kwargs):
            commands.append(list(args))
            return real_run(args, **kwargs)
        plan = self.plan(shots, fps=20, audio=[{"id": "m", "kind": "music",
                                                "path": str(self.root / "music.wav"), "loop": True}])
        with mock.patch.object(assemble, "_run", side_effect=spy):
            result = render(plan)
        durations = [1.0 if i % 3 == 0 else 0.5 for i in range(20)]
        expected = sum(durations) - fades * 0.1
        self.assertAlmostEqual(result.duration_s, expected, delta=0.1)
        self.assertEqual(result.shots_rendered, 20)
        self.assertTrue(any("-filter_complex_script" in args for args in commands))
        for args in commands:
            self.assertNotIn("-filter_complex", args)
            self.assertFalse(any(";" in a or "xfade=" in a or "amix=" in a for a in args),
                             "graphs go in a script file, not argv")
            # Far under Windows' 32 KB command-line limit even with 20 shots.
            self.assertLess(sum(len(a) + 1 for a in args), 8000)

    def test_end_card(self):
        self.need("drawtext", "color")
        plan = self.plan([self.shot("a")], end_card={"text": "AI-generated: it's 100% {fake}", "seconds": 1})
        result = render(plan)
        self.assertEqual(result.warnings, [])
        self.assertAlmostEqual(result.duration_s, 3.0, delta=0.1)

    def test_end_card_skipped_without_drawtext(self):
        no_drawtext = frozenset(self.filters - {"drawtext"})
        plan = self.plan([self.shot("a")], end_card={"text": "AI-generated", "seconds": 1})
        with mock.patch.object(ffmpeg_bin, "ffmpeg_filters", return_value=no_drawtext):
            result = render(plan)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("drawtext", result.warnings[0])
        self.assertAlmostEqual(result.duration_s, 2.0, delta=0.1)

    def test_end_card_skipped_without_font(self):
        plan = self.plan([self.shot("a")], end_card={"text": "AI-generated", "seconds": 1})
        with mock.patch.object(assemble, "find_font", return_value=None):
            result = render(plan)
        self.assertIn("font", result.warnings[0])

    def test_missing_filter_is_a_readable_error(self):
        fade = {"type": "crossfade", "seconds": 0.5}
        plan = self.plan([self.shot("a"), self.shot("c", "c.mp4", transition_in=fade)])
        with mock.patch.object(ffmpeg_bin, "ffmpeg_filters", return_value=frozenset(self.filters - {"xfade"})):
            with self.assertRaisesRegex(assemble.RenderError, "xfade"):
                render(plan)

    def test_captions_burn_in(self):
        self.need("subtitles")
        result = render(self.plan([self.shot("a")], captions_srt=str(self.root / "caps.srt")))
        self.assertAlmostEqual(result.duration_s, 2.0, delta=0.1)

    def test_crossfade_longer_than_shot_fails_before_rendering(self):
        fade = {"type": "crossfade", "seconds": 1.2}
        plan = self.plan([self.shot("a"), self.shot("c", "c.mp4", transition_in=fade)])
        with self.assertRaisesRegex(ValueError, "shorter than its crossfades"):
            render(plan)
        self.assertEqual(list(Path(plan.work_dir).iterdir()), [])

    def test_out_path_must_not_be_an_input(self):
        shutil.copy(self.root / "a.mp4", self.case / "film.mp4")
        plan = self.plan([dict(self.shot("a"), video=str(self.case / "film.mp4"))])
        with self.assertRaises(assemble.RenderError):
            render(plan)


class GraphBuilderTests(unittest.TestCase):
    def test_silent_clip_uses_anullsrc_for_its_exact_length(self):
        shot = ShotInput(id="a", video="/x.mp4", hash="h")
        plan = mock.Mock(width=160, height=90, fps=24)
        graph = build_shot_graph(shot, plan, has_audio=False, frames=36, src_in=0, src_out=1.5)
        self.assertIn("anullsrc", graph)
        self.assertIn("atrim=end_sample=72000", graph)
        self.assertIn("trim=end_frame=36", graph)
        self.assertIn("settb=AVTB", graph)

    def test_speed_and_mute(self):
        shot = ShotInput(id="a", video="/x.mp4", hash="h", speed=3.0, native_sound_db=-math.inf)
        plan = mock.Mock(width=160, height=90, fps=24)
        graph = build_shot_graph(shot, plan, has_audio=True, frames=16, src_in=0, src_out=2)
        self.assertIn("setpts=PTS/3", graph)
        self.assertIn("atempo=2,atempo=1.5", graph)
        self.assertIn("volume=0", graph)

    def test_mix_graph_ducks_only_with_voice(self):
        def lane(kind, idx, duck=False):
            return assemble.Lane(AudioInput(id=f"{kind}{idx}", kind=kind, path="/x.wav", duck=duck), idx, 2.0)
        graph, _ = build_mix_graph([lane("music", 1, True), lane("voice", 2)], SR * 3, None)
        self.assertIn("sidechaincompress", graph)
        self.assertIn("asplit=2[vmix][vkey]", graph)
        self.assertIn("normalize=0", graph)
        graph, _ = build_mix_graph([lane("music", 1, True)], SR * 3, None)
        self.assertNotIn("sidechaincompress", graph)
        graph, warnings = build_mix_graph(
            [assemble.Lane(AudioInput(id="late", kind="sfx", path="/x.wav", start=5), 1, 1.0)], SR * 3, None)
        self.assertNotIn("[1:a]", graph)
        self.assertIn("late", warnings[0])


if __name__ == "__main__":
    unittest.main()
