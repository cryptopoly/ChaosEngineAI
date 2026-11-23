"""The sync-scoring add-on (D-132): pins, install steps, the face model's hash check, status and the Setup routes.

No network and no pip: the runner and the URL opener are stand-ins."""

from __future__ import annotations

import hashlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_service import lipsync_score_addon as addon
from backend_service.routes.setup import lipsync_score as route


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _ok(*_args: object, **_kwargs: object) -> mock.Mock:
    return mock.Mock(returncode=0, stdout="", stderr="")


class _Home(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "addons" / "lipsync-score"
        patcher = mock.patch.object(addon, "addon_dir", return_value=self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def fake_site(self) -> None:
        site = addon.site_dir()
        for name in ("mediapipe", "cv2", f"mediapipe-{addon.MEDIAPIPE_VERSION}.dist-info"):
            (site / name).mkdir(parents=True, exist_ok=True)


class PinTests(unittest.TestCase):
    def test_every_package_is_pinned_exactly(self) -> None:
        for spec in addon.PACKAGES:
            self.assertRegex(spec, r"^[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*$")
        self.assertIn(f"mediapipe=={addon.MEDIAPIPE_VERSION}", addon.PACKAGES)

    def test_the_face_model_comes_from_its_official_versioned_address(self) -> None:
        self.assertTrue(addon.FACE_MODEL_URL.startswith("https://storage.googleapis.com/mediapipe-models/"))
        self.assertIn("/float16/1/", addon.FACE_MODEL_URL)
        self.assertRegex(addon.FACE_MODEL_SHA256, r"^[0-9a-f]{64}$")

    def test_the_folder_is_under_the_home_folder_on_every_platform(self) -> None:
        self.assertEqual(addon.addon_dir(), Path.home() / ".chaosengine" / "addons" / "lipsync-score")

    def test_pip_installs_wheels_only_into_the_site_folder_then_checks_isolated(self) -> None:
        pip, check = addon.install_commands("py")
        self.assertEqual(pip[:3], ["py", "-m", "pip"])
        self.assertEqual(pip[pip.index("--target") + 1], str(addon.site_dir()))
        self.assertIn("--no-deps", pip)
        self.assertIn("--only-binary=:all:", pip)
        self.assertEqual(check[:2], ["py", "-I"])
        self.assertIn(repr(str(addon.site_dir())), check[-1])


class StatusTests(_Home):
    def test_absent_add_on(self) -> None:
        status = addon.status()
        self.assertFalse(status["installed"])
        self.assertEqual(status["diskMb"], 0.0)
        self.assertEqual(status["downloadMb"], addon.DOWNLOAD_MB)

    def test_packages_without_the_model_are_not_installed(self) -> None:
        self.fake_site()
        self.assertTrue(addon.site_installed())
        self.assertFalse(addon.installed())
        addon.face_model_path().write_bytes(b"x" * addon.FACE_MODEL_BYTES)
        self.assertTrue(addon.installed())


class InstallTests(_Home):
    def test_install_runs_pip_fetches_the_model_and_checks(self) -> None:
        body = b"face model bytes"
        phases: list[str] = []
        runner = mock.Mock(side_effect=_ok)
        with mock.patch.object(addon, "FACE_MODEL_SHA256", hashlib.sha256(body).hexdigest()):
            addon.install(phases.append, python="py", runner=runner, opener=lambda *_a, **_k: _Response(body))
        self.assertEqual(phases, ["packages", "model", "check"])
        self.assertEqual(addon.face_model_path().read_bytes(), body)
        self.assertEqual(runner.call_count, 2)
        self.assertFalse(addon.face_model_path().with_name("face_landmarker.task.part").exists())

    def test_a_changed_face_model_is_refused_and_not_kept(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            addon.install(lambda _p: None, python="py", runner=_ok, opener=lambda *_a, **_k: _Response(b"other"))
        self.assertFalse(addon.face_model_path().exists())
        self.assertFalse(addon.face_model_path().with_name("face_landmarker.task.part").exists())

    def test_a_failed_pip_removes_the_half_installed_folder(self) -> None:
        self.fake_site()
        failed = mock.Mock(returncode=1, stdout="", stderr="ERROR: No matching distribution found for mediapipe")
        with self.assertRaisesRegex(RuntimeError, "No matching distribution"):
            addon.install(lambda _p: None, python="py", runner=lambda *_a, **_k: failed)
        self.assertFalse(addon.site_dir().exists())

    def test_a_network_error_is_a_plain_reason(self) -> None:
        def offline(*_a: object, **_k: object) -> _Response:
            raise OSError("no route to host")

        with self.assertRaisesRegex(RuntimeError, "Could not download the face model"):
            addon.install(lambda _p: None, python="py", runner=_ok, opener=offline)


class RouteTests(_Home):
    def setUp(self) -> None:
        super().setUp()
        app = FastAPI()
        app.include_router(route.router)
        self.client = TestClient(app)
        route._JOB.thread, route._JOB.phase, route._JOB.error = None, "idle", None

    def test_status_shape(self) -> None:
        data = self.client.get("/api/setup/lipsync-score/status").json()
        for key in ("installed", "packagesInstalled", "modelInstalled", "downloadMb", "diskMb", "folder", "job"):
            self.assertIn(key, data)
        self.assertEqual(data["job"]["phase"], "idle")

    def test_install_runs_in_the_background_and_reports_errors(self) -> None:
        with mock.patch.object(addon, "install", side_effect=RuntimeError("pip said no")):
            started = self.client.post("/api/setup/install-lipsync-score").json()
            self.assertIn(started["job"]["phase"], ("packages", "error"))
            for _ in range(100):
                job = self.client.get("/api/setup/lipsync-score/status").json()["job"]
                if job["done"]:
                    break
                time.sleep(0.02)
        self.assertEqual((job["phase"], job["error"]), ("error", "pip said no"))

    def test_a_finished_install_reports_done(self) -> None:
        with mock.patch.object(addon, "install", return_value=None):
            self.client.post("/api/setup/install-lipsync-score")
            route._JOB.thread.join(timeout=5)
        self.assertEqual(self.client.get("/api/setup/lipsync-score/status").json()["job"]["phase"], "done")


if __name__ == "__main__":
    unittest.main()
