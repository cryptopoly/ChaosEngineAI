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

    def test_install_pip_persists_to_extras_dir(self):
        """install-package must use ``--target`` so packaged builds keep
        the install across app rebuilds. Without this, mlx-video etc.
        get wiped from the embedded site-packages on every release.
        """
        with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run, \
             mock.patch("backend_service.routes.setup._extras_site_packages") as mock_extras:
            from pathlib import Path
            mock_extras.return_value = Path("/tmp/test-extras-site-packages")
            mock_run.return_value = mock.Mock(returncode=0, stdout="OK", stderr="")
            resp = self.client.post("/api/setup/install-package", json={"package": "mlx-video"})
        self.assertEqual(resp.status_code, 200)
        cmd = mock_run.call_args[0][0]
        self.assertIn("--target", cmd)
        target_idx = cmd.index("--target")
        self.assertEqual(cmd[target_idx + 1], "/tmp/test-extras-site-packages")
        # mlx-video is now pinned to the GitHub source (PyPI ships an
        # unrelated 0.1.0 utilities package). Match on the spec prefix.
        self.assertTrue(
            any("mlx-video" in arg for arg in cmd),
            f"mlx-video spec missing from pip cmd: {cmd}",
        )
        self.assertTrue(
            any("github.com/Blaizzy/mlx-video" in arg for arg in cmd),
            f"mlx-video spec must point to Blaizzy git URL: {cmd}",
        )

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

    def test_install_system_longlive_invokes_python_module(self):
        # LongLive used to shell out to install-longlive.sh, which broke on
        # Windows. Regression guard: the command must now be a python -m
        # invocation of backend_service.longlive_installer so it runs
        # cross-platform.
        with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
            resp = self.client.post("/api/setup/install-system-package", json={"package": "longlive"})
        self.assertEqual(resp.status_code, 200)
        cmd = mock_run.call_args.args[0]
        self.assertIn("-m", cmd)
        self.assertIn("backend_service.longlive_installer", cmd)

    def test_install_system_missing_binary_does_not_suggest_brew_for_non_brew(self):
        # Before the fix, *any* FileNotFoundError from a system-package
        # install emitted "Install Homebrew first: https://brew.sh" — even
        # on Windows, even for LongLive. The user-visible hint must now be
        # scoped to brew commands only.
        with mock.patch("backend_service.routes.setup.subprocess.run", side_effect=FileNotFoundError):
            resp = self.client.post("/api/setup/install-system-package", json={"package": "longlive"})
        body = resp.json()
        self.assertFalse(body["ok"])
        self.assertNotIn("Homebrew", body["output"])
        self.assertNotIn("brew.sh", body["output"])

    def test_install_system_missing_brew_still_suggests_homebrew(self):
        # The Homebrew hint remains correct for llama.cpp on a fresh Mac
        # with no brew installed — scoping the message shouldn't regress
        # the one place it was right.
        with mock.patch("backend_service.routes.setup.subprocess.run", side_effect=FileNotFoundError):
            resp = self.client.post("/api/setup/install-system-package", json={"package": "llama.cpp"})
        body = resp.json()
        self.assertFalse(body["ok"])
        self.assertIn("Homebrew", body["output"])

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

        ``_extras_site_packages`` is redirected to a per-test temp dir so
        the real user-persistent extras tree (which may contain torch
        installs the user cares about) is never touched during tests.
        """
        extras_dir = Path(self.tempdir.name) / "extras"
        extras_dir.mkdir(parents=True, exist_ok=True)
        return (
            mock.patch("backend_service.routes.setup._site_packages_for", return_value=None),
            mock.patch("backend_service.routes.setup._purge_broken_distributions", return_value=[]),
            mock.patch("backend_service.routes.setup._extras_site_packages", return_value=extras_dir),
        )

    def test_install_cuda_torch_stops_at_first_success(self):
        """First working index wins — we must not keep trying after success."""
        sp_patch, purge_patch, extras_patch = self._patch_cuda_helpers()
        with sp_patch, purge_patch, extras_patch, mock.patch(
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

    def test_install_cuda_torch_targets_extras_directory(self):
        """Install must land in the extras site-packages dir, not the venv.

        Regression guard for the Windows video-on-CPU bug: the endpoint
        used to call pip without ``--target`` so torch landed in the
        read-only bundled venv and was ignored at import time in favour
        of a stale CPU wheel already on PYTHONPATH. Fix: install to
        extras so PYTHONPATH prepends it ahead of the bundled venv.
        """
        sp_patch, purge_patch, extras_patch = self._patch_cuda_helpers()
        with sp_patch, purge_patch, extras_patch, mock.patch(
            "backend_service.routes.setup._read_python_version", return_value="3.12.5"
        ):
            with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
                resp = self.client.post("/api/setup/install-cuda-torch", json={})
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertIn("targetDir", body)
        self.assertTrue(body["targetDir"].endswith("extras"))
        # Every pip invocation must carry --target <extras> so pip writes
        # there instead of the venv's site-packages.
        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0]
            self.assertIn("--target", cmd)
            target_idx = cmd.index("--target")
            self.assertEqual(cmd[target_idx + 1], body["targetDir"])

    def test_install_cuda_torch_purges_stale_torch_from_extras(self):
        """Stale torch + nvidia-* dirs are wiped before the reinstall.

        The user's field report: extras had both ``torch-2.6.0+cu124.dist-info``
        and a ``torch-2.11.0+cpu`` package folder from a prior clobber. Python
        couldn't resolve either cleanly. The endpoint now purges the family
        before reinstalling so the fresh install starts clean.
        """
        extras_dir = Path(self.tempdir.name) / "extras_with_stale"
        extras_dir.mkdir(parents=True, exist_ok=True)
        # Simulate the broken state from the user's real extras dir.
        (extras_dir / "torch-2.6.0+cu124.dist-info").mkdir()
        (extras_dir / "torch-2.6.0+cu124.dist-info" / "METADATA").write_text("Version: 2.6.0+cu124")
        (extras_dir / "torch").mkdir()
        (extras_dir / "torch" / "__init__.py").write_text("# stale")
        (extras_dir / "nvidia_cublas_cu12").mkdir()
        (extras_dir / "torchvision").mkdir()  # SIBLING — must survive the purge
        (extras_dir / "torchvision" / "__init__.py").write_text("# keep me")

        with mock.patch(
            "backend_service.routes.setup._site_packages_for", return_value=None,
        ), mock.patch(
            "backend_service.routes.setup._purge_broken_distributions", return_value=[],
        ), mock.patch(
            "backend_service.routes.setup._extras_site_packages", return_value=extras_dir,
        ), mock.patch(
            "backend_service.routes.setup._read_python_version", return_value="3.12.5",
        ):
            with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
                resp = self.client.post("/api/setup/install-cuda-torch", json={})

        self.assertEqual(resp.status_code, 200)
        # Stale torch + its nvidia runtime dep must be gone.
        self.assertFalse((extras_dir / "torch").exists())
        self.assertFalse((extras_dir / "torch-2.6.0+cu124.dist-info").exists())
        self.assertFalse((extras_dir / "nvidia_cublas_cu12").exists())
        # But torchvision is a separate package — must not be touched.
        self.assertTrue((extras_dir / "torchvision").exists())

    def test_install_cuda_torch_falls_through_to_later_indexes(self):
        """If cu124 has no wheel for the user's Python, move on to cu126."""
        call_results = [
            # cu124 swap fails → deps pass skipped
            mock.Mock(returncode=1, stdout="", stderr="ERROR: No matching distribution found for torch"),
            # cu126 swap succeeds → deps pass runs
            mock.Mock(returncode=0, stdout="Successfully installed torch-2.6.0+cu126", stderr=""),
            mock.Mock(returncode=0, stdout="Requirement already satisfied: sympy", stderr=""),
        ]
        sp_patch, purge_patch, extras_patch = self._patch_cuda_helpers()
        with sp_patch, purge_patch, extras_patch, mock.patch(
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
        sp_patch, purge_patch, extras_patch = self._patch_cuda_helpers()
        with sp_patch, purge_patch, extras_patch, mock.patch(
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
        sp_patch, purge_patch, extras_patch = self._patch_cuda_helpers()
        with sp_patch, purge_patch, extras_patch, mock.patch(
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

            extras_dir = Path(self.tempdir.name) / "extras_for_stub_test"
            extras_dir.mkdir(parents=True, exist_ok=True)

            with mock.patch(
                "backend_service.routes.setup._site_packages_for", return_value=tmp_path,
            ), mock.patch(
                "backend_service.routes.setup._extras_site_packages", return_value=extras_dir,
            ), mock.patch(
                "backend_service.routes.setup._read_python_version", return_value="3.12.5"
            ):
                with mock.patch("backend_service.routes.setup.subprocess.run") as mock_run:
                    mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
                    resp = self.client.post("/api/setup/install-cuda-torch", json={})

            self.assertEqual(resp.status_code, 200)
            self.assertFalse(broken.exists(), "broken ~arkupsafe stub should be removed")

    # ------------------------------------------------------------------
    # Helpers — torch purge + constraint pin
    # ------------------------------------------------------------------

    def test_purge_stale_torch_removes_torch_family_only(self):
        """Purge removes torch + nvidia-* but leaves torchvision / other packages alone."""
        from backend_service.routes.setup import _purge_stale_torch_from_extras

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Targets — must be removed.
            (root / "torch").mkdir()
            (root / "torch-2.5.0+cu124.dist-info").mkdir()
            (root / "nvidia_cublas_cu12").mkdir()
            (root / "nvidia-cudnn-cu12").mkdir()
            # Bystanders — must survive.
            (root / "torchvision").mkdir()
            (root / "torchaudio").mkdir()
            (root / "diffusers").mkdir()

            removed = _purge_stale_torch_from_extras(root)

            self.assertIn("torch", removed)
            self.assertIn("torch-2.5.0+cu124.dist-info", removed)
            self.assertIn("nvidia_cublas_cu12", removed)
            self.assertIn("nvidia-cudnn-cu12", removed)
            self.assertFalse((root / "torch").exists())
            self.assertTrue((root / "torchvision").exists())
            self.assertTrue((root / "torchaudio").exists())
            self.assertTrue((root / "diffusers").exists())

    def test_find_installed_torch_version_parses_metadata(self):
        """Version is read from the dist-info's METADATA file."""
        from backend_service.routes.setup import _find_installed_torch_version

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "torch-2.6.0+cu124.dist-info"
            dist.mkdir()
            (dist / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: torch\nVersion: 2.6.0+cu124\nSummary: Tensors...\n"
            )
            self.assertEqual(_find_installed_torch_version(root), "2.6.0+cu124")

    def test_find_installed_torch_version_returns_none_when_missing(self):
        from backend_service.routes.setup import _find_installed_torch_version

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_find_installed_torch_version(Path(tmp)))

    def test_write_torch_constraint_produces_pip_parseable_pin(self):
        """Constraint file must be a valid pip constraints.txt format."""
        from backend_service.routes.setup import _write_torch_constraint

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_torch_constraint(Path(tmp), "2.6.0+cu124")
            self.assertTrue(path.exists())
            content = path.read_text()
            self.assertEqual(content.strip(), "torch==2.6.0+cu124")

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
             mock.patch("backend_service.routes.setup._read_python_version", return_value="3.13.1"), \
             mock.patch("backend_service.routes.setup.platform.system", return_value="Linux"), \
             mock.patch("backend_service.routes.setup.platform.machine", return_value="x86_64"):
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
             mock.patch("backend_service.routes.setup._read_python_version", return_value="3.14.0"), \
             mock.patch("backend_service.routes.setup.platform.system", return_value="Linux"), \
             mock.patch("backend_service.routes.setup.platform.machine", return_value="x86_64"):
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

    def test_apple_silicon_skips_cuda_walk_and_still_runs_remaining_packages(self):
        """On Apple Silicon, torch CUDA install is skipped; remaining bundle still runs."""
        pip_calls: list[str] = []

        def fake_run(cmd, *args, **kwargs):
            # Track which packages pip was asked to install so we can confirm
            # mlx-video appears (added only on Apple Silicon) and torch does
            # not (skipped on Apple Silicon).
            if isinstance(cmd, (list, tuple)):
                joined = " ".join(str(c) for c in cmd)
                pip_calls.append(joined)
            return mock.Mock(returncode=0, stdout="Successfully installed", stderr="")

        with mock.patch("backend_service.routes.setup.subprocess.run", side_effect=fake_run), \
             mock.patch("backend_service.routes.setup._free_bytes", return_value=100_000_000_000), \
             mock.patch("backend_service.routes.setup._read_python_version", return_value="3.13.1"), \
             mock.patch("backend_service.routes.setup.platform.system", return_value="Darwin"), \
             mock.patch("backend_service.routes.setup.platform.machine", return_value="arm64"):
            self.client.post("/api/setup/install-gpu-bundle", json={})
            import time as _time
            deadline = _time.time() + 5.0
            while _time.time() < deadline:
                status = self.client.get("/api/setup/install-gpu-bundle/status").json()
                if status["done"]:
                    break
                _time.sleep(0.05)

        self.assertTrue(status["done"])
        self.assertEqual(status["phase"], "done")
        self.assertTrue(status["cudaVerified"])
        self.assertIsNone(status["indexUrlUsed"])
        # Torch was never pip-installed (we use the bundled MPS torch).
        self.assertFalse(any("torch>=2.4.0" in call for call in pip_calls))
        # Diffusers still got installed (regression guard for the Apple Silicon branch).
        self.assertTrue(any("diffusers" in call for call in pip_calls))

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


class InstallLongLiveTests(unittest.TestCase):
    """Tests for the async ``/api/setup/install-longlive`` job endpoints.

    Mirrors the GPU-bundle pattern: kick off the job, drive the worker to
    each phase, assert ``InstallLogPanel`` gets the data it needs.
    Mocks ``longlive_installer.install`` so we don't actually clone NVlabs
    or download 8 GB of HF weights from the test runner.
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

        # Reset the module-global job state — pytest runs all tests in a
        # single process so leftover state from a previous case bleeds in.
        import backend_service.routes.setup as setup_module
        self._setup_module = setup_module
        setup_module._LONGLIVE_JOB = setup_module._LongLiveJobState()

        # Force resolve_install to a temp path so we don't probe the real
        # ~/.chaosengine/longlive directory during tests.
        self._longlive_root = Path(self.tempdir.name) / "longlive"
        self._env_patch = mock.patch.dict(
            "os.environ",
            {"CHAOSENGINE_LONGLIVE_ROOT": str(self._longlive_root)},
        )
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self.tempdir.cleanup()

    def _wait_for_job_done(self, deadline_secs: float = 5.0) -> dict:
        import time as _time
        deadline = _time.time() + deadline_secs
        while _time.time() < deadline:
            status = self.client.get("/api/setup/install-longlive/status").json()
            if status["done"]:
                return status
            _time.sleep(0.05)
        return self.client.get("/api/setup/install-longlive/status").json()

    def test_status_before_any_install_reports_idle(self):
        resp = self.client.get("/api/setup/install-longlive/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["phase"], "idle")
        self.assertFalse(body["done"])
        self.assertEqual(body["packageIndex"], 0)

    def test_start_install_happy_path_emits_phase_attempts(self):
        """The fake installer fires every phase's progress callback. The job
        worker should accumulate one attempt per phase and end at done.
        """
        from backend_service import longlive_installer

        def fake_install(*, logger, progress, **_kwargs):
            logger("==> fake install start")
            for phase in longlive_installer.INSTALL_PHASES:
                logger(f"==> fake phase {phase}")
                progress({"phase": phase, "ok": True})

        with mock.patch.object(longlive_installer, "install", side_effect=fake_install):
            start_resp = self.client.post("/api/setup/install-longlive", json={})
            self.assertEqual(start_resp.status_code, 200)
            status = self._wait_for_job_done()

        self.assertTrue(status["done"])
        self.assertEqual(status["phase"], "done")
        self.assertEqual(status["percent"], 100.0)
        # One attempt per phase.
        self.assertEqual(
            len(status["attempts"]),
            len(longlive_installer.INSTALL_PHASES),
        )
        # ``targetDir`` should reflect the resolve_install root we patched in.
        self.assertIsNotNone(status["targetDir"])

    def test_start_install_failure_marks_job_errored(self):
        """An installer exception should land in ``error`` and ``phase==error``."""
        from backend_service import longlive_installer

        def fake_install(*, logger, progress, **_kwargs):
            logger("==> partial install — about to fail")
            progress({"phase": "clone", "ok": True})
            raise longlive_installer.LongLiveInstallError("git fetch failed")

        with mock.patch.object(longlive_installer, "install", side_effect=fake_install):
            self.client.post("/api/setup/install-longlive", json={})
            status = self._wait_for_job_done()

        self.assertTrue(status["done"])
        self.assertEqual(status["phase"], "error")
        self.assertIn("git fetch failed", status["error"])
        # First attempt (clone) succeeded; the failing buffer should still
        # be flushed as an attempt so InstallLogPanel renders the partial
        # log instead of going dark.
        self.assertGreaterEqual(len(status["attempts"]), 1)

    def test_second_start_while_running_returns_existing_job(self):
        """Don't spawn two installers in parallel."""
        import backend_service.routes.setup as setup_module
        setup_module._LONGLIVE_JOB.phase = "downloading"
        setup_module._LONGLIVE_JOB.id = "existing-longlive-id"
        setup_module._LONGLIVE_JOB.done = False
        setup_module._LONGLIVE_JOB.package_current = "pip-requirements"

        resp = self.client.post("/api/setup/install-longlive", json={})
        body = resp.json()
        self.assertEqual(body["id"], "existing-longlive-id")
        self.assertEqual(body["phase"], "downloading")
        self.assertEqual(body["packageCurrent"], "pip-requirements")


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
