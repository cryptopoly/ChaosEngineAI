"""Run E2E phase 23 (Storyboard guide, UX audit stream A, and the character lock B1315) against an in-process app,
so the phase itself is checked without a running backend. Projects go to a temporary storyboards folder."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while the file runs
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class StoryboardGuidePhaseTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "storyboards"
        env = mock.patch.dict(os.environ, {"CHAOSENGINE_STORYBOARDS_DIR": str(self.root)})
        env.start()
        self.addCleanup(env.stop)
        client = TestClient(create_app(api_token="t"))

        def http(method, path, *, body=None, content_type=None, timeout=0):
            headers = {"Authorization": "Bearer t", **({"Content-Type": content_type} if content_type else {})}
            response = client.request(method, path, content=body, headers=headers)
            return response.status_code, response.content

        sys.path.insert(0, str(SCRIPTS))
        self.addCleanup(sys.path.remove, str(SCRIPTS))
        suite_mod = _load("e2e_suite_for_test_guide", SCRIPTS / "e2e_test_suite.py")
        self.phase_mod = _load("e2e_phase_guide_for_test", SCRIPTS / "e2e_phases" / "storyboard_guide.py")
        self.suite_mod = suite_mod
        self.suite = SimpleNamespace(PhaseResult=suite_mod.PhaseResult, CheckResult=suite_mod.CheckResult,
                                     _check=suite_mod._check, _voice_http=http)

    def test_the_suite_registers_the_phase(self):
        self.assertIn(23, self.suite_mod.PHASES)

    def test_phase_passes_and_deletes_its_projects(self):
        phase = self.phase_mod.run(SimpleNamespace(backend_reachable=True), self.suite)
        report = [(c.name, c.status, c.reason) for c in phase.checks]
        self.assertEqual(phase.status, "pass", report)
        self.assertEqual(len(phase.checks), 6)
        self.assertEqual([c.status for c in phase.checks].count("fail"), 0, report)
        left = [p for p in self.root.iterdir() if p.is_dir()] if self.root.exists() else []
        self.assertEqual(left, [])

    def test_skips_without_a_backend(self):
        phase = self.phase_mod.run(SimpleNamespace(backend_reachable=False), self.suite)
        self.assertEqual(phase.status, "skip")


if __name__ == "__main__":
    unittest.main()
