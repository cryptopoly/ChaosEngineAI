"""scripts/measure-voice-models.py: the timing and memory maths, and the plan each row is measured with."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("measure_voice_models", ROOT / "scripts" / "measure-voice-models.py")
measure = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(measure)


class MathsTests(unittest.TestCase):
    def test_seconds_per_ten_seconds_and_cold_start(self) -> None:
        self.assertEqual(measure.seconds_per_10s(21.0, 100.0), 2.1)
        self.assertEqual(measure.seconds_per_10s(5.0, 2.5), 20.0)
        with self.assertRaises(ValueError):
            measure.seconds_per_10s(1.0, 0.0)
        self.assertEqual(measure.cold_start(4.2, 0.8), 3.4)
        self.assertEqual(measure.cold_start(0.5, 0.9), 0.0)

    def test_estimates_block_matches_the_catalog_shape(self) -> None:
        est = measure.estimates(measured_on="Apple M4 Max, 64 GB, 2026-11-09", cold=2.44, per_10s=2.104, short=0.83,
                                peak_gb=1.7234, idle_gb=1.7)
        self.assertEqual(est, {
            "basis": "measured", "measuredOn": "Apple M4 Max, 64 GB, 2026-11-09", "coldStartSeconds": 2.4,
            "secondsPer10sAudio": 2.1, "wallSecondsShortLine": 0.8, "peakGb": 1.72, "idleGb": 1.7,
        })

    def test_chat_estimates_take_the_median_warm_time(self) -> None:
        est = measure.chat_estimates(measured_on="M", cold_s=4.06, warm_s=[1.3, 0.6, 0.7], peak_gb=3.512)
        self.assertEqual(est, {"chatMeasuredOn": "M", "chatFirstAudioColdS": 4.1, "chatFirstAudioS": 0.7, "chatPeakGb": 3.51})
        self.assertEqual(measure.chat_plan("kokoro").kind, "kokoro")

    def test_rss_sampler_keeps_the_peak(self) -> None:
        readings = iter([100, 300, 200, 250] + [150] * 1000)
        sampler = measure.RssSampler(1, interval=0.001, reader=lambda _pid: next(readings))
        with sampler:
            pass
        self.assertGreaterEqual(sampler.peak, 100)
        sampler = measure.RssSampler(1, reader=lambda _pid: 0)
        for value in (5, 9, 3):
            sampler._reader = lambda _pid, v=value: v
            sampler.sample()
        self.assertEqual(sampler.peak, 9)

    def test_texts_are_fixed(self) -> None:
        self.assertEqual(len(measure.SHORT_LINE.split()), 13)
        self.assertTrue(590 <= len(measure.LONG_TEXT) <= 610)

    def test_machine_label(self) -> None:
        with mock.patch.object(measure.subprocess, "run", side_effect=OSError):
            label = measure.machine_label("2026-11-09")
        self.assertTrue(label.endswith("2026-11-09"))


class PlanTests(unittest.TestCase):
    def test_each_wave2_row_is_measured_its_usual_way(self) -> None:
        with mock.patch("backend_service.voice_clone.snapshot_dir", return_value=Path("/snap")), \
                mock.patch("backend_service.voice_clone.companion_paths", return_value={}), \
                mock.patch("backend_service.voice_clone.default_voice_path", return_value=Path("/c/conds.safetensors")):
            turbo = measure.plan_for_key("chatterbox-turbo")
            orpheus = measure.plan_for_key("orpheus-3b")
            vox = measure.plan_for_key("voxcpm2")
            mtl = measure.plan_for_key("chatterbox-mtl")
            base = measure.plan_for_key("qwen3-tts-0.6b", reference=Path("/ref.wav"), reference_text="Hi")
            with self.assertRaises(ValueError):
                measure.plan_for_key("qwen3-tts-0.6b")
        self.assertEqual((turbo.kind, turbo.voice), ("preset", "chatterbox-turbo"))
        self.assertEqual(orpheus.extra["speaker"], "tara")
        self.assertEqual((vox.kind, vox.extra["mode"]), ("design", "design"))
        self.assertEqual((mtl.voice, Path(mtl.extra["defaultVoicePath"]).as_posix(), mtl.extra["expression"]), ("chatterbox", "/c/conds.safetensors", 0.5))
        self.assertEqual((base.kind, Path(base.extra["refAudioPath"]).as_posix()), ("clone", "/ref.wav"))


if __name__ == "__main__":
    unittest.main()
