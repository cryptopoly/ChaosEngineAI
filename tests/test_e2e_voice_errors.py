"""Run E2E phase 24 (voice error codes, UX audit change 6) in process: the real voice routes with a stubbed worker,
so the phase itself is checked without a running backend, a model or a GPU."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while the file runs
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class VoiceErrorsPhaseTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(SCRIPTS))
        self.addCleanup(sys.path.remove, str(SCRIPTS))
        self.suite_mod = _load("e2e_suite_for_test_voice_errors", SCRIPTS / "e2e_test_suite.py")
        self.phase_mod = _load("e2e_phase_voice_errors_for_test", SCRIPTS / "e2e_phases" / "voice_errors.py")
        self.suite = SimpleNamespace(PhaseResult=self.suite_mod.PhaseResult, CheckResult=self.suite_mod.CheckResult,
                                     _check=self.suite_mod._check, _voice_http=mock.Mock())

    def test_the_suite_registers_the_phase(self):
        self.assertIn(24, self.suite_mod.PHASES)

    def test_phase_passes_in_process_and_skips_the_live_check_without_a_backend(self):
        phase = self.phase_mod.run(SimpleNamespace(backend_reachable=False), self.suite)
        report = [(c.name, c.status, c.reason) for c in phase.checks]
        self.assertEqual(phase.status, "pass", report)
        self.assertEqual(len(phase.checks), 6)
        self.assertEqual([c.status for c in phase.checks].count("fail"), 0, report)
        self.assertEqual(phase.checks[-1].status, "skip")
        self.suite._voice_http.assert_not_called()

    def test_the_live_check_reads_the_running_backend(self):
        body = b'{"detail": "Unknown voice", "code": "unknown", "message": "m", "action": "none", "extra": null}'
        self.suite._voice_http.return_value = (400, body)
        phase = self.phase_mod.run(SimpleNamespace(backend_reachable=True), self.suite)
        self.assertEqual(phase.checks[-1].status, "pass", phase.checks[-1].reason)
        self.suite._voice_http.return_value = (400, b'{"detail": "Unknown voice"}')
        phase = self.phase_mod.run(SimpleNamespace(backend_reachable=True), self.suite)
        self.assertEqual((phase.checks[-1].status, phase.status), ("fail", "fail"))

    def test_phase_fails_when_a_response_loses_its_code(self):
        from backend_service import voice_errors

        with mock.patch.object(voice_errors, "classify", side_effect=lambda raw, status=None: voice_errors._info("unknown")):
            phase = self.phase_mod.run(SimpleNamespace(backend_reachable=False), self.suite)
        self.assertEqual(phase.status, "fail")


if __name__ == "__main__":
    unittest.main()
