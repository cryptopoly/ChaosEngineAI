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
    # CUDA torch install (Windows/Linux NVIDIA fallback)
    # ------------------------------------------------------------------

    def _patch_cuda_helpers(self):
        """Neutralise the pre-install helpers so tests focus on the pip loop.

        ``_site_packages_for`` and ``_purge_broken_distributions`` add
        their own subprocess and filesystem side effects that would
        confuse the subprocess.run mock below; patching them to no-op
        keeps each test asserting exactly the behaviour it names.
        """
        return (
            mock.patch("backend_service.routes.setup._site_packages_for", return_value=None),
            mock.patch("backend_service.routes.setup._purge_broken_distributions", return_value=[]),
        )

    def test_install_cuda_torch_stops_at_first_success(self):
        """First working index wins — we must not keep trying after success."""
        sp_patch, purge_patch = self._patch_cuda_helpers()
        with sp_patch, purge_patch, mock.patch(
            "backend_service.routes.setup._read_python_version", return_value="3.12.5"
        ):
            with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(
                    returncode=0, stdout="Successfully installed torch-2.5.0+cu124", stderr=""
                )
                resp = self.client.post("/api/setup/install-cuda-torch", json={})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["requiresRestart"])
        self.assertIsNotNone(body["indexUrl"])
        self.assertIn("cu124", body["indexUrl"])
        self.assertEqual(body["pythonVersion"], "3.12.5")
        self.assertFalse(body["noWheelForPython"])
        # Only one index should have been attempted — cu124 succeeded so
        # cu126 / cu128 / cu121 / nightly must not be tried. Each successful
        # attempt issues two pip calls: pass 1 swaps torch (--no-deps),
        # pass 2 fills in any missing transitive deps.
        self.assertEqual(len(body["attempts"]), 1)
        self.assertEqual(mock_run.call_count, 2)

    def test_install_cuda_torch_falls_through_to_later_indexes(self):
        """If cu124 has no wheel for the user's Python, move on to cu126."""
        call_results = [
            # cu124 swap fails → deps pass skipped
            mock.Mock(returncode=1, stdout="", stderr="ERROR: No matching distribution found for torch"),
            # cu126 swap succeeds → deps pass runs
            mock.Mock(returncode=0, stdout="Successfully installed torch-2.6.0+cu126", stderr=""),
            mock.Mock(returncode=0, stdout="Requirement already satisfied: sympy", stderr=""),
        ]
        sp_patch, purge_patch = self._patch_cuda_helpers()
        with sp_patch, purge_patch, mock.patch(
            "backend_service.routes.setup._read_python_version", return_value="3.13.1"
        ):
            with mock.patch("backend_service.routes.setup.subprocess.run", side_effect=call_results):
                resp = self.client.post("/api/setup/install-cuda-torch", json={})
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(len(body["attempts"]), 2)
        self.assertFalse(body["attempts"][0]["ok"])
        self.assertTrue(body["attempts"][1]["ok"])
        self.assertIn("cu126", body["indexUrl"])
        self.assertFalse(body["noWheelForPython"])

    def test_install_cuda_torch_reports_failure_after_all_attempts(self):
        """All indexes fail — surface the last error to the UI."""
        fail = mock.Mock(returncode=1, stdout="", stderr="ERROR: Install failed, disk full")
        sp_patch, purge_patch = self._patch_cuda_helpers()
        with sp_patch, purge_patch, mock.patch(
            "backend_service.routes.setup._read_python_version", return_value="3.12.5"
        ):
            with mock.patch("backend_service.routes.setup.subprocess.run", return_value=fail):
                resp = self.client.post("/api/setup/install-cuda-torch", json={})
        body = resp.json()
        self.assertFalse(body["ok"])
        self.assertFalse(body["requiresRestart"])
        self.assertIsNone(body["indexUrl"])
        from backend_service.routes.setup import _CUDA_TORCH_INDEXES
        self.assertEqual(len(body["attempts"]), len(_CUDA_TORCH_INDEXES))
        # Generic failure (not a wheel mismatch) must NOT be flagged as
        # noWheelForPython — the user can usefully retry.
        self.assertFalse(body["noWheelForPython"])

    def test_install_cuda_torch_flags_no_wheel_when_every_attempt_misses(self):
        """Python 3.14 case — every index returns "No matching distribution".

        The UI uses noWheelForPython to tell the user their Python version
        is the problem, not the CUDA index, so they stop retrying.
        """
        no_wheel = mock.Mock(
            returncode=1,
            stdout="",
            stderr="ERROR: Could not find a version that satisfies the requirement torch "
                   "(from versions: none)\nERROR: No matching distribution found for torch",
        )
        sp_patch, purge_patch = self._patch_cuda_helpers()
        with sp_patch, purge_patch, mock.patch(
            "backend_service.routes.setup._read_python_version", return_value="3.14.0"
        ):
            with mock.patch("backend_service.routes.setup.subprocess.run", return_value=no_wheel):
                resp = self.client.post("/api/setup/install-cuda-torch", json={})
        body = resp.json()
        self.assertFalse(body["ok"])
        self.assertTrue(body["noWheelForPython"])
        self.assertEqual(body["pythonVersion"], "3.14.0")

    def test_install_cuda_torch_sweeps_broken_stub_dists(self):
        """Broken ``~<pkg>`` dirs are removed before pip runs.

        Real-world trigger: a prior interrupted pip install leaves
        ``~arkupsafe/`` in site-packages. Without cleanup, subsequent
        installs print a "Ignoring invalid distribution" warning and
        sometimes fail mid-install trying to heal the stub.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            broken = tmp_path / "~arkupsafe"
            broken.mkdir()
            (broken / "stub.txt").write_text("leftover")

            sp_patch = mock.patch(
                "backend_service.routes.setup._site_packages_for", return_value=tmp_path,
            )
            with sp_patch, mock.patch(
                "backend_service.routes.setup._read_python_version", return_value="3.12.5"
            ):
                with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run:
                    mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
                    resp = self.client.post("/api/setup/install-cuda-torch", json={})

            self.assertEqual(resp.status_code, 200)
            self.assertFalse(broken.exists(), "broken ~arkupsafe stub should be removed")

    def test_install_cuda_torch_default_list_starts_with_cu124(self):
        """cu124 is the broadest 3.9-3.13 match; cu121 must not be first anymore."""
        from backend_service.routes.setup import _CUDA_TORCH_INDEXES
        self.assertTrue(_CUDA_TORCH_INDEXES[0].endswith("cu124"))
        self.assertIn("https://download.pytorch.org/whl/cu126", _CUDA_TORCH_INDEXES)
        self.assertIn("https://download.pytorch.org/whl/cu128", _CUDA_TORCH_INDEXES)
        # cu121 is still in the list for old Python + old driver combos,
        # just no longer leading.
        self.assertIn("https://download.pytorch.org/whl/cu121", _CUDA_TORCH_INDEXES)
        # The nightly index is our last-resort for bleeding-edge Python
        # (e.g. 3.14) — PyTorch sometimes ships nightly wheels before the
        # stable index catches up.
        self.assertIn("https://download.pytorch.org/whl/nightly/cu128", _CUDA_TORCH_INDEXES)
        self.assertEqual(_CUDA_TORCH_INDEXES[-1], "https://download.pytorch.org/whl/nightly/cu128")

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


class InstallGpuBundleTests(unittest.TestCase):
    """Tests for the async ``/api/setup/install-gpu-bundle`` endpoint.

    The endpoint kicks off a background thread and returns immediately;
    tests use a ``threading.Event`` to drive the worker to each state
    synchronously (no real polling / sleep), then assert the status
    endpoint reports what the UI needs to drive its progress bar.
    """

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

        # Target the bundle install at a throwaway temp dir so the test
        # doesn't scribble in a real user profile — the endpoint honours
        # CHAOSENGINE_EXTRAS_SITE_PACKAGES when set.
        self.extras_dir = Path(self.tempdir.name) / "extras" / "site-packages"
        self._env_patch = mock.patch.dict(
            "os.environ",
            {"CHAOSENGINE_EXTRAS_SITE_PACKAGES": str(self.extras_dir)},
        )
        self._env_patch.start()

        # Reset the module-global job state so tests don't see leftover
        # state from a previous case.
        import backend_service.routes.setup as setup_module
        self._setup_module = setup_module
        setup_module._GPU_BUNDLE_JOB = setup_module._GpuBundleJobState()

    def tearDown(self):
        self._env_patch.stop()
        self.tempdir.cleanup()

    def test_status_before_any_install_reports_idle(self):
        resp = self.client.get("/api/setup/install-gpu-bundle/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["phase"], "idle")
        self.assertFalse(body["done"])
        self.assertEqual(body["packageIndex"], 0)

    def test_bundle_info_exposes_target_and_packages(self):
        resp = self.client.get("/api/setup/gpu-bundle-info")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["targetDir"], str(self.extras_dir))
        self.assertGreaterEqual(len(body["packages"]), 5)
        # torch must come first so the walking CUDA-index logic runs before
        # other packages attempt to resolve against a missing torch.
        self.assertEqual(body["packages"][0]["label"], "torch")

    def test_start_install_happy_path_reaches_done(self):
        """End-to-end: start, let the worker run to completion, assert done."""
        # Mock the underlying pip install to always succeed.
        def fake_run(*args, **kwargs):
            return mock.Mock(returncode=0, stdout="Successfully installed", stderr="")

        # Mock _verify_cuda so we don't try to actually spawn a subprocess
        # that imports torch.
        with mock.patch("backend_service.routes.setup.subprocess.run", side_effect=fake_run), \
             mock.patch("backend_service.routes.setup._verify_cuda", return_value=(True, "cuda_available=true")), \
             mock.patch("backend_service.routes.setup._free_bytes", return_value=100_000_000_000), \
             mock.patch("backend_service.routes.setup._read_python_version", return_value="3.13.1"):
            start_resp = self.client.post("/api/setup/install-gpu-bundle", json={})
            self.assertEqual(start_resp.status_code, 200)
            # Wait for the background thread to finish. The worker is daemon=True
            # and the test env has no real network / disk — it exits in milliseconds.
            import time as _time
            deadline = _time.time() + 5.0
            while _time.time() < deadline:
                status = self.client.get("/api/setup/install-gpu-bundle/status").json()
                if status["done"]:
                    break
                _time.sleep(0.05)

        self.assertTrue(status["done"])
        self.assertEqual(status["phase"], "done")
        self.assertTrue(status["requiresRestart"])
        self.assertTrue(status["cudaVerified"])
        self.assertEqual(status["pythonVersion"], "3.13.1")
        self.assertIsNotNone(status["indexUrlUsed"])

    def test_start_install_fails_when_disk_space_low(self):
        """A preflight check stops the install before any pip calls."""
        with mock.patch("backend_service.routes.setup._free_bytes", return_value=1_000_000), \
             mock.patch("backend_service.routes.setup._read_python_version", return_value="3.13.1"):
            self.client.post("/api/setup/install-gpu-bundle", json={})
            import time as _time
            deadline = _time.time() + 3.0
            while _time.time() < deadline:
                status = self.client.get("/api/setup/install-gpu-bundle/status").json()
                if status["done"]:
                    break
                _time.sleep(0.05)

        self.assertTrue(status["done"])
        self.assertEqual(status["phase"], "error")
        self.assertIn("GB free", status["error"])

    def test_start_install_flags_no_wheel_for_python(self):
        """When every CUDA index returns 'no matching distribution', surface it."""
        def no_wheel_run(*args, **kwargs):
            return mock.Mock(
                returncode=1,
                stdout="",
                stderr="ERROR: Could not find a version that satisfies the requirement torch\n"
                       "ERROR: No matching distribution found for torch",
            )

        with mock.patch("backend_service.routes.setup.subprocess.run", side_effect=no_wheel_run), \
             mock.patch("backend_service.routes.setup._free_bytes", return_value=100_000_000_000), \
             mock.patch("backend_service.routes.setup._read_python_version", return_value="3.14.0"):
            self.client.post("/api/setup/install-gpu-bundle", json={})
            import time as _time
            deadline = _time.time() + 5.0
            while _time.time() < deadline:
                status = self.client.get("/api/setup/install-gpu-bundle/status").json()
                if status["done"]:
                    break
                _time.sleep(0.05)

        self.assertTrue(status["done"])
        self.assertEqual(status["phase"], "error")
        self.assertTrue(status["noWheelForPython"])
        self.assertIn("3.14.0", status["error"])

    def test_second_start_while_running_returns_existing_job(self):
        """Don't spawn two installers in parallel — UI gets whichever started first."""
        # Prime the job state as if an install is already running.
        import backend_service.routes.setup as setup_module
        setup_module._GPU_BUNDLE_JOB.phase = "downloading"
        setup_module._GPU_BUNDLE_JOB.id = "existing-job-id"
        setup_module._GPU_BUNDLE_JOB.done = False
        setup_module._GPU_BUNDLE_JOB.package_current = "torch"

        with mock.patch("backend_service.routes.setup._free_bytes", return_value=100_000_000_000), \
             mock.patch("backend_service.routes.setup._read_python_version", return_value="3.13.1"):
            resp = self.client.post("/api/setup/install-gpu-bundle", json={})
        body = resp.json()
        self.assertEqual(body["id"], "existing-job-id")
        self.assertEqual(body["phase"], "downloading")
        self.assertEqual(body["packageCurrent"], "torch")


class DllLockDetectionTests(unittest.TestCase):
    """``_looks_like_dll_lock`` is a Windows-specific heuristic that swaps a
    generic pip rmtree failure for an actionable 'restart backend, retry
    install' message. Pure-function unit tests because this path is hard
    to exercise end-to-end on a Linux CI runner.
    """

    def test_matches_winerror_5_on_torch_dll(self):
        from backend_service.routes.setup import _looks_like_dll_lock
        output = (
            "ERROR: Exception:\n"
            "shutil.rmtree(target_item_dir)\n"
            "PermissionError: [WinError 5] Access is denied: "
            "'C:\\\\Users\\\\Dan\\\\AppData\\\\Local\\\\ChaosEngineAI\\\\extras\\\\site-packages\\\\torch\\\\lib\\\\asmjit.dll'"
        )
        self.assertTrue(_looks_like_dll_lock(output))

    def test_does_not_match_unrelated_pip_failure(self):
        from backend_service.routes.setup import _looks_like_dll_lock
        # A generic "no matching distribution" failure (the Python 3.14
        # wheel-missing case) must NOT be misclassified as a DLL lock —
        # otherwise users on unsupported Python get the wrong advice.
        output = (
            "ERROR: Could not find a version that satisfies the requirement torch>=2.4.0\n"
            "ERROR: No matching distribution found for torch>=2.4.0"
        )
        self.assertFalse(_looks_like_dll_lock(output))

    def test_does_not_match_locked_non_torch_file(self):
        from backend_service.routes.setup import _looks_like_dll_lock
        # WinError 5 on a non-torch file (e.g. some editor-locked script)
        # shouldn't point users at the "restart backend" remedy, because
        # restarting the backend won't release unrelated file handles.
        output = "PermissionError: [WinError 5] Access is denied: 'C:\\\\something-else\\\\config.json'"
        self.assertFalse(_looks_like_dll_lock(output))


if __name__ == "__main__":
    unittest.main()
