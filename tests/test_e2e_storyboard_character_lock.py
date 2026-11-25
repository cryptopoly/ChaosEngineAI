"""Run E2E phase 26 (Storyboard character lock on the NVIDIA lane, B1315) in process.

The in-process checks run for real (the FLUX 3 Klein catalogue row, the studio's still builder, a temporary store). The
live check runs against an in-process app whose image route's inner call is a stand-in that writes a PNG of the asked
size, so the phase's own logic (batch, files, control run, clean-up) is checked without a GPU or the weights."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.helpers.images import _find_image_variant

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
KLEIN = "leejet/flux-3-image-klein-GGUF-q8"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while the file runs
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class CharacterLockPhaseTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "storyboards"
        env = mock.patch.dict(os.environ, {"CHAOSENGINE_STORYBOARDS_DIR": str(self.root)})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app(api_token="t"))
        self.catalog_downloaded = True

        def http(method, path, *, body=None, content_type=None, timeout=0):
            if path == "/api/images/catalog":
                row = {"id": KLEIN, "availableLocally": self.catalog_downloaded}
                return 200, json.dumps({"families": [{"variants": [row]}]}).encode()
            headers = {"Authorization": "Bearer t", **({"Content-Type": content_type} if content_type else {})}
            response = self.client.request(method, path, content=body, headers=headers)
            return response.status_code, response.content

        sys.path.insert(0, str(SCRIPTS))
        self.addCleanup(sys.path.remove, str(SCRIPTS))
        self.suite_mod = _load("e2e_suite_for_test_character_lock", SCRIPTS / "e2e_test_suite.py")
        self.phase_mod = _load("e2e_phase_character_lock_for_test", SCRIPTS / "e2e_phases" / "storyboard_character_lock.py")
        self.suite = SimpleNamespace(PhaseResult=self.suite_mod.PhaseResult, CheckResult=self.suite_mod.CheckResult,
                                     _check=self.suite_mod._check, _voice_http=http)

    def _report(self, phase):
        return [(c.name[:40], c.status, c.reason) for c in phase.checks]

    def _fake_route(self, fail_with: str | None = None):
        from e2e_phases.phase_20 import png

        sent = []

        def stand_in(_request, body):
            sent.append(body)
            if fail_with:
                raise HTTPException(status_code=500, detail=fail_with)
            colour = (body.seed % 250, 40, 200 if body.referenceImagePaths else 20)
            path = Path(tempfile.mkdtemp()) / f"{len(sent)}.png"
            path.write_bytes(png(body.width, body.height, [colour]))
            return {"artifacts": [{"imagePath": str(path), "modelName": "stand-in", "seed": body.seed,
                                   "width": body.width, "height": body.height}]}

        return sent, stand_in

    def test_the_suite_registers_the_phase(self):
        self.assertIn(26, self.suite_mod.PHASES)

    def test_in_process_checks_pass_and_the_live_one_skips_without_a_backend(self):
        phase = self.phase_mod.run(SimpleNamespace(backend_reachable=False), self.suite)
        report = self._report(phase)
        self.assertEqual(phase.status, "pass", report)
        self.assertEqual([c.status for c in phase.checks], ["pass", "pass", "skip"], report)
        self.assertEqual(phase.checks[-1].reason, "backend not reachable")

    def test_the_request_check_fails_when_the_picture_is_dropped(self):
        from backend_service.storyboard import cast

        real = cast.apply_to_still

        def drop_pictures(project, shot, paths, slots, prompt):
            return real(project, shot, [], slots, prompt)

        with mock.patch.object(cast, "apply_to_still", side_effect=drop_pictures):
            phase = self.phase_mod.run(SimpleNamespace(backend_reachable=False), self.suite)
        self.assertEqual(phase.status, "fail", self._report(phase))
        self.assertEqual(phase.checks[1].status, "fail")

    def test_live_check_makes_three_stills_then_the_control_and_deletes_its_project(self):
        sent, stand_in = self._fake_route()
        from backend_service.storyboard import generators

        with mock.patch.object(generators, "_variant", return_value=_find_image_variant(KLEIN)), \
                mock.patch("backend_service.routes.images._generate_image_locked", side_effect=stand_in):
            phase = self.phase_mod.run(SimpleNamespace(backend_reachable=True), self.suite)
        report = self._report(phase)
        self.assertEqual(phase.status, "pass", report)
        self.assertEqual(phase.checks[-1].status, "pass", report)
        live = [b for b in sent if b.width == 512 and b.height == 320]
        self.assertEqual(len(live), 6)  # three with the character, three without
        self.assertEqual([bool(b.referenceImagePaths) for b in live], [True] * 3 + [False] * 3)
        self.assertEqual([b.seed for b in live[:3]], [b.seed for b in live[3:]])
        left = [p for p in self.root.iterdir() if p.is_dir()] if self.root.exists() else []
        self.assertEqual(left, [])

    def test_live_check_skips_when_the_weights_are_not_downloaded(self):
        self.catalog_downloaded = False
        phase = self.phase_mod.run(SimpleNamespace(backend_reachable=True), self.suite)
        self.assertEqual(phase.checks[-1].status, "skip", self._report(phase))
        self.assertIn("not downloaded", phase.checks[-1].reason)

    def test_live_check_skips_when_the_backend_has_no_sd_binary(self):
        _sent, stand_in = self._fake_route(fail_with="stable-diffusion.cpp binary not staged. Run the build script.")
        from backend_service.storyboard import generators

        with mock.patch.object(generators, "_variant", return_value=_find_image_variant(KLEIN)), \
                mock.patch("backend_service.routes.images._generate_image_locked", side_effect=stand_in):
            phase = self.phase_mod.run(SimpleNamespace(backend_reachable=True), self.suite)
        self.assertEqual(phase.checks[-1].status, "skip", self._report(phase))
        self.assertIn("binary not staged", phase.checks[-1].reason)
        left = [p for p in self.root.iterdir() if p.is_dir()] if self.root.exists() else []
        self.assertEqual(left, [])


if __name__ == "__main__":
    unittest.main()
