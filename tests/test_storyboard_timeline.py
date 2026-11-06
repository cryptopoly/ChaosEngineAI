"""Storyboard timeline (pure) and the render plan it produces."""

import json
import tempfile
import unittest
from pathlib import Path

from backend_service.storyboard.models import AudioClip, AudioSource, Project, Shot, Take, Transition
from backend_service.storyboard.timeline import build_render_plan, compute_timeline

SHOT_KEYS = {"id", "video", "hash", "trim_in", "trim_out", "speed", "native_sound_db", "transition_in"}
AUDIO_KEYS = {"id", "kind", "path", "start", "duration", "trim_in", "gain_db", "fade_in", "fade_out", "duck", "loop"}
PLAN_KEYS = {"width", "height", "fps", "preview", "loudness_lufs", "out_path", "work_dir", "shots", "audio",
             "metadata", "end_card", "captions_srt"}


def shot(sid, measured=4.0, **fields):
    take = Take(id=f"t{sid}", clip=f"media/{sid}/t{sid}/clip.mp4", measuredSeconds=measured,
                hashes={"clip": "a" * 64}, settings={"model": "LTX-2.3"})
    return Shot(id=sid, takes=[take], activeTake=take.id, status="done", stillPrompt="SECRET PROMPT", **fields)


def voice(cid, measured=2.0, **fields):
    return AudioClip(id=cid, kind="voice", source=AudioSource(path=f"media/audio/{cid}.wav", measuredSeconds=measured), **fields)


def project(shots, audio=()):
    return Project(id="0123456789abcdef", name="My Film", shots=list(shots), audio=list(audio))


class TimelineTests(unittest.TestCase):
    def test_measured_97_frame_clip(self):
        t = compute_timeline(project([shot("a", 97 / 24)]))
        self.assertAlmostEqual(t.total, 4.0417, places=4)
        self.assertTrue(t.shots[0].measured)

    def test_trims_and_speed(self):
        t = compute_timeline(project([shot("a", 4.0, trimIn=1.0, trimOut=3.0, speed=2.0)]))
        span = t.shots[0]
        self.assertEqual((span.source_in, span.source_out), (1.0, 3.0))
        self.assertAlmostEqual(t.total, 1.0)

    def test_bad_trims_are_ignored_with_a_note(self):
        t = compute_timeline(project([shot("a", 4.0, trimIn=3.0, trimOut=1.0)]))
        self.assertAlmostEqual(t.total, 4.0)
        self.assertIn("trims", t.problems[0])

    def test_crossfades_overlap_and_shorten_the_film(self):
        fade = Transition(type="crossfade", seconds=0.5)
        t = compute_timeline(project([shot("a"), shot("b", transition=fade), shot("c", transition=fade)]))
        self.assertEqual([s.start for s in t.shots], [0.0, 3.5, 7.0])
        self.assertAlmostEqual(t.total, 11.0)
        first = compute_timeline(project([shot("a", transition=fade)]))
        self.assertEqual(first.shots[0].transition_type, "cut")  # nothing to fade from

    def test_crossfade_is_clamped_to_half_the_shorter_shot(self):
        t = compute_timeline(project([shot("a", 0.5), shot("b", 4.0, transition=Transition(type="crossfade", seconds=2))]))
        self.assertAlmostEqual(t.shots[1].transition_seconds, 0.25)
        self.assertTrue(any("crossfade" in p for p in t.problems))

    def test_anchored_voice_follows_its_shot(self):
        clips = [voice("v", anchor={"shotId": "b", "offset": 0.5}), voice("m", start=2.0)]
        before = compute_timeline(project([shot("a", 4.0), shot("b")], clips))
        after = compute_timeline(project([shot("a", 6.0), shot("b")], clips))
        self.assertAlmostEqual(before.audio[0].start, 4.5)
        self.assertAlmostEqual(after.audio[0].start, 6.5)
        self.assertEqual(before.audio[1].start, after.audio[1].start)

    def test_audio_length_loop_and_bounds(self):
        clips = [voice("long", 30.0, start=1.0), voice("loop", 1.0, start=2.0, loop=True),
                 voice("trim", 3.0, start=0.0, trimIn=1.0), voice("neg", 2.0, anchor={"shotId": "a", "offset": -0.5}),
                 voice("late", 1.0, start=50.0)]
        t = compute_timeline(project([shot("a", 4.0), shot("b", 4.0)], clips))
        spans = {a.id: a for a in t.audio}
        self.assertAlmostEqual(spans["long"].duration, 7.0)  # cut at the end of the picture
        self.assertAlmostEqual(spans["loop"].duration, 6.0)
        self.assertAlmostEqual(spans["trim"].duration, 2.0)
        self.assertEqual((spans["neg"].start, spans["neg"].trim_in), (0.0, 0.5))
        self.assertNotIn("late", spans)

    def test_unready_shots_and_unmeasured_takes(self):
        empty = Shot(id="e", seconds=3.0)
        t = compute_timeline(project([shot("a", None), empty]))
        self.assertAlmostEqual(t.total, 7.0)  # planned seconds stand in
        self.assertFalse(t.shots[0].measured)
        ready = compute_timeline(project([shot("a"), empty], [voice("v", anchor={"shotId": "e", "offset": 0})]), ready_only=True)
        self.assertEqual([s.id for s in ready.shots], ["a"])
        self.assertEqual(ready.audio, [])
        self.assertEqual(len(ready.problems), 2)


class RenderPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / "0123456789abcdef"

    def _files(self, p):
        for s in p.shots:
            path = self.folder / s.active().clip
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        for a in p.audio:
            path = self.folder / a.source.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        return p

    def test_plan_shape_paths_and_metadata(self):
        p = self._files(project([shot("a", 97 / 24), shot("b", transition=Transition(type="crossfade", seconds=0.5))],
                                [voice("v", anchor={"shotId": "b", "offset": 1.0}, duck=True)]))
        plan = build_render_plan(p, preview=False, folder=self.folder)
        self.assertEqual(set(plan), PLAN_KEYS)
        self.assertEqual(set(plan["shots"][0]), SHOT_KEYS)
        self.assertEqual(set(plan["audio"][0]), AUDIO_KEYS)
        for path in [plan["out_path"], plan["work_dir"], plan["shots"][0]["video"], plan["audio"][0]["path"]]:
            self.assertTrue(Path(path).is_absolute(), path)
            self.assertTrue(Path(path).resolve().is_relative_to(self.folder.resolve()))
        self.assertEqual(plan["shots"][1]["transition_in"], {"type": "crossfade", "seconds": 0.5})
        self.assertAlmostEqual(plan["audio"][0]["start"], 97 / 24 - 0.5 + 1.0, places=3)
        self.assertEqual((plan["width"], plan["height"], plan["fps"]), (1280, 720, 24))
        meta = json.dumps(plan["metadata"])
        self.assertTrue(plan["metadata"]["machineGenerated"])
        self.assertEqual(plan["metadata"]["models"], ["LTX-2.3"])
        for leak in ("SECRET PROMPT", "My Film", str(self.folder)):
            self.assertNotIn(leak, meta)
        self.assertIsNone(plan["end_card"])

    def test_ai_label_switch_and_cloned_voice_rule(self):
        p = self._files(project([shot("a")]))
        p.export.aiLabel.embedTag = False
        p.export.aiLabel.endCard = True
        plan = build_render_plan(p, preview=False, folder=self.folder)
        self.assertEqual(plan["metadata"], {})
        self.assertEqual(plan["end_card"]["text"], "Made with AI")
        cloned = AudioClip(id="c", kind="voice", start=0.0,
                           source=AudioSource(path="media/audio/c.wav", measuredSeconds=1.0, clonedVoice=True))
        p.audio.append(cloned)
        self._files(p)
        plan = build_render_plan(p, preview=False, folder=self.folder)
        self.assertTrue(plan["metadata"]["machineGenerated"])  # the switch is ignored
        self.assertTrue(plan["metadata"]["clonedVoice"])

    def test_preview_is_small_and_presets_reframe(self):
        p = self._files(project([shot("a")]))
        preview = build_render_plan(p, preview=True, folder=self.folder)
        self.assertEqual((preview["width"], preview["height"], preview["preview"]), (640, 360, True))
        self.assertTrue(preview["out_path"].endswith("preview.mp4"))
        p.defaults.exportPreset = "9:16"
        self.assertEqual(build_render_plan(p, preview=False, folder=self.folder)["width"], 720)
        p.defaults.exportPreset = "1:1"
        plan = build_render_plan(p, preview=False, folder=self.folder)
        self.assertEqual((plan["width"], plan["height"]), (720, 720))

    def test_nothing_to_render(self):
        with self.assertRaises(ValueError):
            build_render_plan(project([Shot(id="e")]), preview=True, folder=self.folder)


if __name__ == "__main__":
    unittest.main()
