"""Run E2E phase 27 (the upscaler's quality tier) in process: the catalogue and gate check, and the segment loop with a
stand-in model on a generated clip; the live check skips without a backend."""

import importlib.util
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
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class VideoUpscaleQualityPhaseTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(SCRIPTS))
        self.addCleanup(sys.path.remove, str(SCRIPTS))
        self.suite_mod = _load("e2e_suite_for_test_upscale_quality", SCRIPTS / "e2e_test_suite.py")
        self.phase_mod = _load("e2e_phase_upscale_quality_for_test", SCRIPTS / "e2e_phases" / "video_upscale_quality.py")
        self.suite = SimpleNamespace(PhaseResult=self.suite_mod.PhaseResult, CheckResult=self.suite_mod.CheckResult,
                                     _check=self.suite_mod._check, _voice_http=mock.Mock())

    def test_registered_under_27(self):
        self.assertEqual(self.phase_mod.PHASE_KEY, 27)
        self.assertIn(27, self.suite_mod.PHASES)

    @unittest.skipUnless(FFMPEG, "ffmpeg is not available")
    def test_in_process_checks_pass_and_live_skips_without_a_backend(self):
        phase = self.phase_mod.run(SimpleNamespace(backend_reachable=False), self.suite)
        statuses = [check.status for check in phase.checks]
        self.assertEqual(statuses, ["pass", "pass", "skip"], [c.reason for c in phase.checks])
        self.assertEqual(phase.status, "pass")
        self.suite._voice_http.assert_not_called()

    def test_live_skips_where_the_tier_cannot_run(self):
        self.suite._voice_http.return_value = (200, b'{"available": false, "unavailableReason": "no NVIDIA card"}')
        status, reason, _ = self.phase_mod._live(self.suite, "ffmpeg")
        self.assertEqual(status, "skip")
        self.assertIn("no NVIDIA card", reason)


if __name__ == "__main__":
    unittest.main()
