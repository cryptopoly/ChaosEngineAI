"""Video upscale add-on: pins, the install steps with pip and the download faked, and its Setup routes."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_service.catalog import video_upscale_models as catalog
from backend_service.routes.setup import video_upscale as route
from backend_service.video_upscale import addon


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _ok(*_args: object, **_kwargs: object) -> mock.Mock:
    return mock.Mock(returncode=0, stdout="", stderr="")


def _row(data: bytes, sha: str | None) -> dict:
    return {"id": "tiny", "name": "Tiny", "label": "Fast", "file": "tiny.pth", "url": "https://example.invalid/tiny.pth",
            "sizeBytes": len(data), "sha256": sha}


class _Home(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "addons" / "video-upscale"
        patcher = mock.patch.object(addon, "addon_dir", return_value=self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def fake_site(self) -> None:
        for name in ("spandrel", f"spandrel-{addon.SPANDREL_VERSION}.dist-info"):
            (addon.site_dir() / name).mkdir(parents=True, exist_ok=True)


class PinTests(unittest.TestCase):
    def test_packages_are_pinned_exactly(self) -> None:
        for spec in addon.PACKAGES:
            self.assertRegex(spec, r"^[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*$")
        self.assertIn(f"spandrel=={addon.SPANDREL_VERSION}", addon.PACKAGES)

    def test_install_is_wheels_only_into_the_addon_site_without_dependencies(self) -> None:
        pip, check = addon.install_commands("python")
        self.assertIn("--no-deps", pip)
        self.assertIn("--only-binary=:all:", pip)
        self.assertEqual(pip[pip.index("--target") + 1], str(addon.site_dir()))
        self.assertEqual(check[:2], ["python", "-I"])

    def test_folder(self) -> None:
        self.assertEqual(addon.addon_dir(), Path.home() / ".chaosengine" / "addons" / "video-upscale")

    def test_nothing_here_imports_spandrel_or_torch_at_import_time(self) -> None:
        import subprocess
        import sys

        code = ("import sys; import backend_service.video_upscale.addon, backend_service.routes.setup.video_upscale; "
                "print('torch' in sys.modules, 'spandrel' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.split(), ["False", "False"])


class DownloadTests(_Home):
    def test_size_and_pin_are_checked(self) -> None:
        data = b"weights" * 100
        sha = hashlib.sha256(data).hexdigest()
        self.assertEqual(addon.download_weights(_row(data, sha), lambda *a, **k: _Response(data)), sha)
        self.assertTrue(addon.model_installed(_row(data, sha)))

    def test_a_wrong_hash_keeps_nothing(self) -> None:
        data = b"weights" * 100
        with self.assertRaises(RuntimeError):
            addon.download_weights(_row(data, "0" * 64), lambda *a, **k: _Response(data))
        self.assertEqual(list((self.root / "weights").iterdir()), [])

    def test_a_wrong_size_keeps_nothing(self) -> None:
        row = {**_row(b"x" * 10, None), "sizeBytes": 11}
        with self.assertRaises(RuntimeError):
            addon.download_weights(row, lambda *a, **k: _Response(b"x" * 10))
        self.assertFalse(addon.weights_path(row).exists())

    def test_an_unpinned_row_records_the_hash_it_saw(self) -> None:
        data = b"abc" * 50
        seen = addon.download_weights(_row(data, None), lambda *a, **k: _Response(data))
        self.assertEqual(json.loads((self.root / "weights.json").read_text())["tiny.pth"], seen)


class InstallTests(_Home):
    def test_install_runs_pip_then_downloads_the_default_row_then_checks(self) -> None:
        calls, phases = [], []
        row = dict(catalog.find_model(catalog.DEFAULT_MODEL_ID))
        data = b"\0" * row["sizeBytes"]
        row["sha256"] = hashlib.sha256(data).hexdigest()  # the pinned real hash does not match fake bytes

        def runner(cmd, **_kw):
            calls.append(cmd)
            if "--target" in cmd:
                self.fake_site()
            return _ok()

        with mock.patch.object(addon, "find_model", return_value=row):
            addon.install(phases.append, runner=runner, opener=lambda *a, **k: _Response(data))
        self.assertEqual(phases, ["packages", "weights", "check"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(addon.installed())
        self.assertFalse(addon.installed("realesrgan-x4plus"))

    def test_a_failed_pip_removes_the_site(self) -> None:
        def runner(cmd, **_kw):
            self.fake_site()
            return mock.Mock(returncode=1, stdout="", stderr="ERROR: no wheel")

        with self.assertRaises(RuntimeError) as caught:
            addon.install(lambda _p: None, runner=runner, opener=lambda *a, **k: _Response(b""))
        self.assertIn("no wheel", str(caught.exception))
        self.assertFalse(addon.site_dir().exists())

    def test_unknown_model(self) -> None:
        with self.assertRaises(RuntimeError):
            addon.install(lambda _p: None, ["nope"], runner=_ok)


class RouteTests(_Home):
    def setUp(self) -> None:
        super().setUp()
        app = FastAPI()
        app.include_router(route.router)
        self.client = TestClient(app)
        route._JOB.thread, route._JOB.phase, route._JOB.error = None, "idle", None

    def test_status_shape(self) -> None:
        data = self.client.get("/api/setup/video-upscale/status").json()
        for key in ("installed", "packagesInstalled", "folder", "diskMb", "models", "job"):
            self.assertIn(key, data)
        self.assertEqual([m["id"] for m in data["models"]], [r["id"] for r in catalog.VIDEO_UPSCALE_MODELS])
        self.assertFalse(data["installed"])

    def test_install_runs_in_the_background_with_the_models_asked_for(self) -> None:
        with mock.patch.object(addon, "install", return_value=None) as install:
            self.client.post("/api/setup/install-video-upscale", json={"models": ["realesrgan-x4plus"]})
            route._JOB.thread.join(timeout=5)
        install.assert_called_once()
        self.assertEqual(install.call_args.args[1], ["realesrgan-x4plus"])
        self.assertEqual(self.client.get("/api/setup/video-upscale/status").json()["job"]["phase"], "done")

    def test_errors_reach_the_status(self) -> None:
        with mock.patch.object(addon, "install", side_effect=RuntimeError("pip said no")):
            self.client.post("/api/setup/install-video-upscale")
            for _ in range(100):
                job = self.client.get("/api/setup/video-upscale/status").json()["job"]
                if job["done"]:
                    break
                time.sleep(0.02)
        self.assertEqual((job["phase"], job["error"]), ("error", "pip said no"))

    def test_unknown_model_is_refused(self) -> None:
        self.assertEqual(self.client.post("/api/setup/install-video-upscale", json={"models": ["x"]}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
