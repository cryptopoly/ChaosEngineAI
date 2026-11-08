"""D-62: run E2E phase 19 (skills import) against an in-process app, so the phase itself is checked."""

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


class PhaseNineteenTests(unittest.TestCase):
    def test_phase_passes_and_leaves_no_skill(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        skills_dir = Path(tmp.name) / "skills"
        with mock.patch.dict(os.environ, {"CHAOSENGINE_SKILLS_DIR": str(skills_dir)}):
            client = TestClient(create_app(api_token="t"))

            def http(method, path, *, body=None, content_type=None, timeout=0):
                headers = {"Authorization": "Bearer t", **({"Content-Type": content_type} if content_type else {})}
                response = client.request(method, path, content=body, headers=headers)
                return response.status_code, response.content

            sys.path.insert(0, str(SCRIPTS))
            self.addCleanup(sys.path.remove, str(SCRIPTS))
            suite_mod = _load("e2e_suite_for_test", SCRIPTS / "e2e_test_suite.py")
            phase_mod = _load("e2e_phase_19_for_test", SCRIPTS / "e2e_phases" / "phase_19.py")
            suite = SimpleNamespace(PhaseResult=suite_mod.PhaseResult, CheckResult=suite_mod.CheckResult,
                                    _check=suite_mod._check, _voice_http=http)
            phase = phase_mod.run(SimpleNamespace(backend_reachable=True), suite)
            self.assertEqual(phase.status, "pass", [(c.name, c.status, c.reason) for c in phase.checks])
            self.assertEqual(len(phase.checks), 5)
            self.assertEqual([p.name for p in skills_dir.iterdir()], [])


if __name__ == "__main__":
    unittest.main()
