"""Review C: run E2E phase 213 (Prompts and Plugins) against an in-process app, so the phase itself is checked.

The prompt library is pointed at a temporary folder (left alone, the route opens the real data directory). The
real-model check skips here: the capability object lists no MLX model.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.helpers.prompts import PromptLibrary
from backend_service.plugins import plugin_registry
from backend_service.routes import prompts as prompts_module

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while the file runs
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class PhaseTwentyOnePromptsPluginsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data_dir = Path(tmp.name)
        prompts_module._library = PromptLibrary(self.data_dir)
        self.addCleanup(setattr, prompts_module, "_library", None)
        client = TestClient(create_app(api_token="t"))

        def http(method, path, *, body=None, content_type=None, timeout=0):
            headers = {"Authorization": "Bearer t", **({"Content-Type": content_type} if content_type else {})}
            response = client.request(method, path, content=body, headers=headers)
            return response.status_code, response.content

        sys.path.insert(0, str(SCRIPTS))
        self.addCleanup(sys.path.remove, str(SCRIPTS))
        suite_mod = _load("e2e_suite_for_test_p21", SCRIPTS / "e2e_test_suite.py")
        self.phase_mod = _load("e2e_phase_21_pp_for_test", SCRIPTS / "e2e_phases" / "phase_21_prompts_plugins.py")
        self.suite_mod = suite_mod
        self.suite = SimpleNamespace(
            PhaseResult=suite_mod.PhaseResult, CheckResult=suite_mod.CheckResult, _check=suite_mod._check,
            _voice_http=http, _anonymous_http=lambda path: client.get(path).status_code)
        self.cap = SimpleNamespace(backend_reachable=True, mlx_usable=False, local_mlx_models=[])

    def test_phase_passes_and_leaves_nothing_behind(self):
        before = {t["id"] for t in prompts_module._library.list_all()}
        phase = self.phase_mod.run(self.cap, self.suite)
        report = [(c.name, c.status, c.reason) for c in phase.checks]
        self.assertEqual(phase.status, "pass", report)
        self.assertEqual(len(phase.checks), 10)
        self.assertEqual([c.status for c in phase.checks].count("skip"), 1, report)  # the real-model check
        self.assertEqual(phase.checks[-1].reason, "no MLX text model on disk")
        self.assertEqual({t["id"] for t in prompts_module._library.list_all()}, before)
        self.assertTrue(all(manifest.enabled for manifest in plugin_registry.list_all()))

    def test_registered_under_213_with_the_other_review_phases(self):
        self.assertEqual(self.phase_mod.PHASE_KEY, 213)
        self.assertEqual(self.suite_mod.PHASES[213].__name__, "phase_21_prompts_plugins")
        self.assertNotIn(21, self.suite_mod.PHASES)  # each review stream has its own key, nothing is joined
        self.assertTrue({211, 212, 213, 214} <= set(self.suite_mod.PHASES))

    def test_backend_not_reachable_skips(self):
        phase = self.phase_mod.run(SimpleNamespace(backend_reachable=False), self.suite)
        self.assertEqual(phase.status, "skip")


if __name__ == "__main__":
    unittest.main()
