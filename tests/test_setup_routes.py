"""Tests for the setup routes — package installation, turbo update check."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.state import ChaosEngineState

TEST_API_TOKEN = "test-api-token"


def _fake_system_snapshot():
    return {
        "platform": "Darwin",
        "arch": "arm64",
        "hardwareSummary": "Test Machine",
        "backendLabel": "test",
        "appVersion": "0.0.0-test",
        "availableCacheStrategies": [],
        "dflash": {"available": False, "mlxAvailable": False, "vllmAvailable": False, "supportedModels": []},
        "vllmAvailable": False,
        "mlxAvailable": False,
        "mlxLmAvailable": False,
        "mlxUsable": False,
        "ggufAvailable": True,
        "converterAvailable": False,
        "nativePython": "/usr/bin/python3",
        "llamaServerPath": "/usr/local/bin/llama-server",
        "llamaServerTurboPath": None,
        "llamaCliPath": None,
        "nativeRuntimeMessage": None,
        "totalMemoryGb": 64,
        "availableMemoryGb": 32,
        "usedMemoryGb": 32,
        "swapUsedGb": 0,
        "swapTotalGb": 0,
        "compressedMemoryGb": 0,
        "memoryPressurePercent": 50.0,
        "cpuUtilizationPercent": 10.0,
        "gpuUtilizationPercent": None,
        "spareHeadroomGb": 26.0,
        "battery": None,
        "runningLlmProcesses": [],
        "uptimeMinutes": 1.0,
    }


class FakeRuntime:
    """Minimal runtime stub for setup route tests."""

    class _Caps:
        pythonExecutable = "/usr/bin/python3"

        def to_dict(self):
            return {"pythonExecutable": self.pythonExecutable, "ggufAvailable": True}

    capabilities = _Caps()

    def refresh_capabilities(self, *, force=False):
        return self.capabilities

    def status(self, **kwargs):
        return {"engine": "mock", "loadedModel": None, "nativeBackends": {}}


class SetupRouteTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            settings_path=Path(self.tempdir.name) / "settings.json",
            benchmarks_path=Path(self.tempdir.name) / "benchmarks.json",
            chat_sessions_path=Path(self.tempdir.name) / "chats.json",
        )
        state.runtime = FakeRuntime()
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})

    def tearDown(self):
        self.tempdir.cleanup()

    # ------------------------------------------------------------------
    # Pip package install
    # ------------------------------------------------------------------

    def test_install_pip_rejects_unknown_package(self):
        resp = self.client.post("/api/setup/install-package", json={"package": "evil-package"})
        self.assertEqual(resp.status_code, 400)

    def test_install_pip_returns_manual_message_for_chaosengine(self):
        resp = self.client.post("/api/setup/install-package", json={"package": "chaosengine"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not published on PyPI", resp.json()["detail"])

    def test_install_pip_accepts_whitelisted_package(self):
        with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="OK", stderr="")
            resp = self.client.post("/api/setup/install-package", json={"package": "dflash-mlx"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertIn("capabilities", body)

    def test_install_pip_reports_failure(self):
        with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="ERROR: No matching distribution")
            resp = self.client.post("/api/setup/install-package", json={"package": "dflash"})
        body = resp.json()
        self.assertFalse(body["ok"])
        self.assertIn("No matching distribution", body["output"])

    def test_install_pip_accepts_imageio(self):
        """Video Studio installs this directly when the mp4 encoder is missing."""
        with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="Successfully installed imageio", stderr="")
            resp = self.client.post("/api/setup/install-package", json={"package": "imageio"})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])
        # Confirm we actually invoked pip install with the right distribution name.
        cmd = mock_run.call_args[0][0]
        self.assertIn("imageio", cmd)

    def test_install_pip_accepts_imageio_ffmpeg(self):
        """The ffmpeg plugin is the other half of mp4 export — must also be whitelisted."""
        with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="Successfully installed imageio-ffmpeg", stderr="")
            resp = self.client.post("/api/setup/install-package", json={"package": "imageio-ffmpeg"})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])
        cmd = mock_run.call_args[0][0]
        self.assertIn("imageio-ffmpeg", cmd)

    def test_video_output_deps_are_whitelisted(self):
        """Regression guard — the Video Studio install button targets these exact keys.

        If someone renames or removes them we want the test suite to scream before
        the UI starts handing users a 400 on ``/api/setup/install-package``.
        """
        from backend_service.routes.setup import _INSTALLABLE_PIP_PACKAGES

        self.assertIn("imageio", _INSTALLABLE_PIP_PACKAGES)
        self.assertIn("imageio-ffmpeg", _INSTALLABLE_PIP_PACKAGES)
        # Distribution names should match the short keys so the UI doesn't need
        # its own translation table.
        self.assertEqual(_INSTALLABLE_PIP_PACKAGES["imageio"], "imageio")
        self.assertEqual(_INSTALLABLE_PIP_PACKAGES["imageio-ffmpeg"], "imageio-ffmpeg")

    def test_video_model_tokenizer_deps_are_whitelisted(self):
        """LTX-Video / Wan / Hunyuan / CogVideoX need these tokenizer packages.

        The Studio's "Install missing video dependencies" button targets these
        exact keys; if the install allow-list drops them, the user gets a 400
        and the button looks broken instead of unblocking generation.
        """
        from backend_service.routes.setup import _INSTALLABLE_PIP_PACKAGES

        for pkg in ("tiktoken", "sentencepiece", "protobuf", "ftfy"):
            self.assertIn(
                pkg,
                _INSTALLABLE_PIP_PACKAGES,
                f"{pkg} must be whitelisted so the Studio install button works",
            )
            self.assertEqual(_INSTALLABLE_PIP_PACKAGES[pkg], pkg)

    def test_install_pip_accepts_tiktoken(self):
        """LTX-Video's exact missing-dep error — the user-reported case."""
        with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0, stdout="Successfully installed tiktoken", stderr=""
            )
            resp = self.client.post("/api/setup/install-package", json={"package": "tiktoken"})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])
        cmd = mock_run.call_args[0][0]
        self.assertIn("tiktoken", cmd)

    # ------------------------------------------------------------------
    # System package install
    # ------------------------------------------------------------------

    def test_install_system_rejects_unknown_package(self):
        resp = self.client.post("/api/setup/install-system-package", json={"package": "evil"})
        self.assertEqual(resp.status_code, 400)

    def test_install_system_accepts_llama_server_turbo(self):
        with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="build complete", stderr="")
            resp = self.client.post("/api/setup/install-system-package", json={"package": "llama-server-turbo"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])

    def test_install_system_reports_timeout(self):
        import subprocess as _sp

        with mock.patch("backend_service.routes.setup.subprocess.run", side_effect=_sp.TimeoutExpired(["cmd"], 600)):
            resp = self.client.post("/api/setup/install-system-package", json={"package": "llama.cpp"})
        body = resp.json()
        self.assertFalse(body["ok"])
        self.assertIn("timed out", body["output"])

    # ------------------------------------------------------------------
    # Refresh capabilities
    # ------------------------------------------------------------------

    def test_refresh_capabilities_returns_dict(self):
        resp = self.client.post("/api/setup/refresh-capabilities")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("capabilities", resp.json())

    # ------------------------------------------------------------------
    # Turbo update check
    # ------------------------------------------------------------------

    def test_turbo_update_check_not_installed(self):
        with mock.patch("backend_service.routes.setup._TURBO_VERSION_FILE", Path("/nonexistent")):
            with mock.patch("backend_service.routes.setup._CHAOSENGINE_BIN_DIR", Path("/nonexistent")):
                resp = self.client.get("/api/setup/turbo-update-check")
        body = resp.json()
        self.assertFalse(body["installed"])
        self.assertIsNone(body["installedCommit"])
        self.assertFalse(body["updateAvailable"])

    def test_turbo_update_check_installed_up_to_date(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".version", delete=False) as f:
            f.write("abc123def456\nfeature/planarquant-kv-cache\n2026-01-01T00:00:00Z\n")
            version_path = Path(f.name)

        bin_dir = version_path.parent
        turbo_bin = bin_dir / "llama-server-turbo"
        turbo_bin.touch()

        try:
            with (
                mock.patch("backend_service.routes.setup._TURBO_VERSION_FILE", version_path),
                mock.patch("backend_service.routes.setup._CHAOSENGINE_BIN_DIR", bin_dir),
                mock.patch("backend_service.routes.setup._fetch_turbo_remote_head", return_value="abc123def456"),
            ):
                resp = self.client.get("/api/setup/turbo-update-check")
            body = resp.json()
            self.assertTrue(body["installed"])
            self.assertEqual(body["installedCommit"], "abc123def456")
            self.assertFalse(body["updateAvailable"])
        finally:
            version_path.unlink(missing_ok=True)
            turbo_bin.unlink(missing_ok=True)

    def test_turbo_update_check_update_available(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".version", delete=False) as f:
            f.write("oldcommithash1\nfeature/planarquant-kv-cache\n2026-01-01T00:00:00Z\n")
            version_path = Path(f.name)

        bin_dir = version_path.parent
        turbo_bin = bin_dir / "llama-server-turbo"
        turbo_bin.touch()

        try:
            with (
                mock.patch("backend_service.routes.setup._TURBO_VERSION_FILE", version_path),
                mock.patch("backend_service.routes.setup._CHAOSENGINE_BIN_DIR", bin_dir),
                mock.patch("backend_service.routes.setup._fetch_turbo_remote_head", return_value="newcommithash2"),
            ):
                resp = self.client.get("/api/setup/turbo-update-check")
            body = resp.json()
            self.assertTrue(body["installed"])
            self.assertTrue(body["updateAvailable"])
            self.assertNotEqual(body["installedCommit"], body["remoteCommit"])
        finally:
            version_path.unlink(missing_ok=True)
            turbo_bin.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
