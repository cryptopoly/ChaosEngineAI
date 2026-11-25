"""Run E2E phase 25 (video upscale, B1316) in process: the real upscale routes and job manager with a pixel-doubling
stand-in for the model on a tiny generated clip, so the phase itself is checked without a backend, a model or a GPU."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.video_upscale_fakes import find_ffmpeg

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FFMPEG = find_ffmpeg()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while the file runs
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class VideoUpscalePhaseTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(SCRIPTS))
        self.addCleanup(sys.path.remove, str(SCRIPTS))
        self.suite_mod = _load("e2e_suite_for_test_video_upscale", SCRIPTS / "e2e_test_suite.py")
        self.phase_mod = _load("e2e_phase_video_upscale_for_test", SCRIPTS / "e2e_phases" / "video_upscale.py")
        self.suite = SimpleNamespace(PhaseResult=self.suite_mod.PhaseResult, CheckResult=self.suite_mod.CheckResult,
                                     _check=self.suite_mod._check, _voice_http=mock.Mock())

    def test_the_suite_registers_the_phase_under_a_key_nothing_else_uses(self):
        self.assertEqual(self.phase_mod.PHASE_KEY, 25)
        self.assertIn(25, self.suite_mod.PHASES)
        keys = [self.suite_mod.PHASES[k] for k in self.suite_mod.PHASES]
        self.assertEqual(len(keys), len(set(self.suite_mod.PHASES)))

    @unittest.skipUnless(FFMPEG, "ffmpeg is not available")
    def test_phase_passes_in_process_and_skips_the_live_check_without_a_backend(self):
        phase = self.phase_mod.run(SimpleNamespace(backend_reachable=False), self.suite)
        report = [(c.name, c.status, c.reason) for c in phase.checks]
        self.assertEqual(phase.status, "pass", report)
        self.assertEqual([c.status for c in phase.checks], ["pass", "skip"], report)
        self.assertEqual(phase.checks[0].detail["frames"], self.phase_mod.SOURCE_FRAMES)
        self.suite._voice_http.assert_not_called()

    @unittest.skipUnless(FFMPEG, "ffmpeg is not available")
    def test_the_live_check_skips_with_its_reason_when_the_addon_or_a_model_is_missing(self):
        def answer(models):
            self.suite._voice_http.return_value = (200, json.dumps(models).encode())
            return self.phase_mod._live(self.suite, FFMPEG)

        status, reason, _ = answer({"addonInstalled": False, "models": []})
        self.assertEqual(status, "skip")
        self.assertIn("add-on", reason)
        status, reason, _ = answer({"addonInstalled": True, "models": [{"id": "m", "installed": False}]})
        self.assertEqual(status, "skip")
        self.assertIn("model", reason)
        self.suite._voice_http.return_value = (404, b'{"detail": "Not Found"}')
        self.assertEqual(self.phase_mod._live(self.suite, FFMPEG)[0], "fail")

    @unittest.skipUnless(FFMPEG, "ffmpeg is not available")
    def test_phase_fails_when_frames_per_second_is_null(self):
        from backend_service.video_upscale import jobs

        real = jobs.UpscaleJob.to_dict

        def null_rate(job):
            return {**real(job), "framesPerSecond": None}

        with mock.patch.object(jobs.UpscaleJob, "to_dict", null_rate):
            phase = self.phase_mod.run(SimpleNamespace(backend_reachable=False), self.suite)
        self.assertEqual(phase.status, "fail")
        self.assertIn("framesPerSecond", phase.checks[0].reason)

    @unittest.skipUnless(FFMPEG, "ffmpeg is not available")
    def test_phase_fails_when_the_frame_total_is_duration_times_rate_again(self):
        from backend_service.video_upscale import pipeline

        with mock.patch.object(pipeline, "count_video_frames", return_value=None):
            phase = self.phase_mod.run(SimpleNamespace(backend_reachable=False), self.suite)
        self.assertEqual(phase.status, "fail")


if __name__ == "__main__":
    unittest.main()
