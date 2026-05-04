"""Tests for FU-025 Phase 9: mlx-video Wan installer + setup endpoints.

Covers the orchestration helper (download → convert → verify) plus the
``/api/setup/install-mlx-video-wan`` endpoint surface. The actual HF
download + convert subprocess are mocked so the suite runs without
mlx-video installed and without raw Wan weights on disk.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend_service import mlx_video_wan_convert, mlx_video_wan_installer
from backend_service.mlx_video_wan_installer import (
    INSTALL_PHASES,
    SUPPORTED_RAW_REPOS,
    WanInstallError,
    approx_raw_size_gb,
    install,
    raw_dir_for,
)


def _fake_status(repo: str, *, converted: bool = True, has_moe: bool = False):
    return mlx_video_wan_convert.WanConvertStatus(
        repo=repo,
        converted=converted,
        outputDir=str(mlx_video_wan_convert.output_dir_for(repo)),
        hasTransformer=converted,
        hasMoeExperts=has_moe,
        hasVae=converted,
        hasTextEncoder=converted,
        note=None if converted else "Output directory does not exist",
    )


class InstallerHelpersTests(unittest.TestCase):
    def test_install_phases_canonical_order(self):
        # Order is load-bearing — the FastAPI worker walks this list to
        # drive the percent counter.
        self.assertEqual(
            INSTALL_PHASES,
            ("preflight", "download-raw", "convert", "verify"),
        )

    def test_raw_dir_under_raw_root(self):
        path = raw_dir_for("Wan-AI/Wan2.1-T2V-1.3B")
        self.assertEqual(path.name, "Wan-AI__Wan2.1-T2V-1.3B")
        self.assertEqual(path.parent.name, "mlx-video-wan-raw")

    def test_approx_raw_size_known_repos(self):
        self.assertEqual(approx_raw_size_gb("Wan-AI/Wan2.1-T2V-1.3B"), 3.5)
        self.assertGreater(approx_raw_size_gb("Wan-AI/Wan2.2-T2V-A14B"), 50)
        self.assertIsNone(approx_raw_size_gb("Wan-AI/Unknown-Model"))


class InstallPreflightTests(unittest.TestCase):
    def test_preflight_rejects_non_darwin(self):
        with patch(
            "backend_service.mlx_video_wan_installer.platform.system",
            return_value="Linux",
        ):
            with self.assertRaises(WanInstallError) as ctx:
                install("Wan-AI/Wan2.1-T2V-1.3B")
        self.assertIn("Apple Silicon only", str(ctx.exception))

    def test_preflight_rejects_intel_mac(self):
        with patch(
            "backend_service.mlx_video_wan_installer.platform.system",
            return_value="Darwin",
        ), patch(
            "backend_service.mlx_video_wan_installer.platform.machine",
            return_value="x86_64",
        ):
            with self.assertRaises(WanInstallError) as ctx:
                install("Wan-AI/Wan2.1-T2V-1.3B")
        self.assertIn("arm64", str(ctx.exception))

    def test_preflight_rejects_when_mlx_video_missing(self):
        with patch(
            "backend_service.mlx_video_wan_installer.platform.system",
            return_value="Darwin",
        ), patch(
            "backend_service.mlx_video_wan_installer.platform.machine",
            return_value="arm64",
        ), patch(
            "backend_service.mlx_video_wan_installer.is_mlx_video_available",
            return_value=False,
        ):
            with self.assertRaises(WanInstallError) as ctx:
                install("Wan-AI/Wan2.1-T2V-1.3B")
        self.assertIn("mlx-video is not installed", str(ctx.exception))

    def test_preflight_rejects_unsupported_repo(self):
        with patch(
            "backend_service.mlx_video_wan_installer.platform.system",
            return_value="Darwin",
        ), patch(
            "backend_service.mlx_video_wan_installer.platform.machine",
            return_value="arm64",
        ), patch(
            "backend_service.mlx_video_wan_installer.is_mlx_video_available",
            return_value=True,
        ):
            with self.assertRaises(WanInstallError) as ctx:
                install("Lightricks/LTX-Video")
        self.assertIn("Unsupported Wan repo", str(ctx.exception))


class InstallHappyPathTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp(prefix="chaosengine-wan-install-test-")
        self._orig_convert_root = mlx_video_wan_convert.CONVERT_ROOT
        self._orig_raw_root = mlx_video_wan_installer.RAW_ROOT
        mlx_video_wan_convert.CONVERT_ROOT = Path(self.tmpdir) / "converted"
        mlx_video_wan_installer.RAW_ROOT = Path(self.tmpdir) / "raw"

    def tearDown(self):
        mlx_video_wan_convert.CONVERT_ROOT = self._orig_convert_root
        mlx_video_wan_installer.RAW_ROOT = self._orig_raw_root
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _enter_apple_silicon_patches(self, stack):
        stack.enter_context(patch(
            "backend_service.mlx_video_wan_installer.platform.system",
            return_value="Darwin",
        ))
        stack.enter_context(patch(
            "backend_service.mlx_video_wan_installer.platform.machine",
            return_value="arm64",
        ))
        stack.enter_context(patch(
            "backend_service.mlx_video_wan_installer.is_mlx_video_available",
            return_value=True,
        ))

    def test_install_progress_emits_phases_in_order(self):
        from contextlib import ExitStack
        progress_events: list[str] = []
        log_lines: list[str] = []

        def fake_snapshot_download(**kwargs):
            log_lines.append(f"snapshot_download {kwargs}")

        repo = "Wan-AI/Wan2.1-T2V-1.3B"
        out = mlx_video_wan_convert.output_dir_for(repo)

        class _FakeProc:
            stdout = iter(["[INFO] step 1/100\n", "[INFO] done\n"])
            def wait(self, timeout=None):
                out.mkdir(parents=True, exist_ok=True)
                (out / "transformer.safetensors").write_bytes(b"x")
                (out / "Wan2.1_VAE.safetensors").write_bytes(b"x")
                return 0

        fake_hub_module = MagicMock()
        fake_hub_module.snapshot_download = fake_snapshot_download

        with ExitStack() as stack:
            self._enter_apple_silicon_patches(stack)
            stack.enter_context(patch.dict(
                "sys.modules", {"huggingface_hub": fake_hub_module},
            ))
            stack.enter_context(patch(
                "backend_service.mlx_video_wan_installer.subprocess.Popen",
                return_value=_FakeProc(),
            ))
            install(
                repo,
                logger=log_lines.append,
                progress=lambda evt: progress_events.append(str(evt.get("phase"))),
                timeout_seconds=10,
            )

        self.assertEqual(
            progress_events,
            ["preflight", "download-raw", "convert", "verify"],
        )
        self.assertTrue(raw_dir_for(repo).parent.exists())
        self.assertTrue((out / "transformer.safetensors").exists())

    def test_install_raises_when_convert_subprocess_fails(self):
        from contextlib import ExitStack
        repo = "Wan-AI/Wan2.1-T2V-1.3B"

        class _FailProc:
            stdout = iter(["[ERROR] OOM\n"])
            def wait(self, timeout=None):
                return 1

        fake_hub_module = MagicMock()
        fake_hub_module.snapshot_download = lambda **kw: None

        with ExitStack() as stack:
            self._enter_apple_silicon_patches(stack)
            stack.enter_context(patch.dict(
                "sys.modules", {"huggingface_hub": fake_hub_module},
            ))
            stack.enter_context(patch(
                "backend_service.mlx_video_wan_installer.subprocess.Popen",
                return_value=_FailProc(),
            ))
            with self.assertRaises(WanInstallError) as ctx:
                install(repo, timeout_seconds=10, logger=lambda _: None)
        self.assertIn("exited with code 1", str(ctx.exception))

    def test_install_raises_when_verify_finds_partial_output(self):
        from contextlib import ExitStack
        repo = "Wan-AI/Wan2.1-T2V-1.3B"

        class _PartialProc:
            stdout = iter(["[INFO] partial\n"])
            def wait(self, timeout=None):
                return 0

        fake_hub_module = MagicMock()
        fake_hub_module.snapshot_download = lambda **kw: None

        with ExitStack() as stack:
            self._enter_apple_silicon_patches(stack)
            stack.enter_context(patch.dict(
                "sys.modules", {"huggingface_hub": fake_hub_module},
            ))
            stack.enter_context(patch(
                "backend_service.mlx_video_wan_installer.subprocess.Popen",
                return_value=_PartialProc(),
            ))
            with self.assertRaises(WanInstallError) as ctx:
                install(repo, logger=lambda _: None, timeout_seconds=10)
        self.assertIn("incomplete", str(ctx.exception).lower())


_TEST_API_TOKEN = "wan-test-token"


def _wan_test_system_snapshot():
    return {
        "platform": "Darwin",
        "arch": "arm64",
        "hardwareSummary": "Test Machine",
        "backendLabel": "test",
        "appVersion": "0.0.0-test",
        "availableCacheStrategies": [],
        "dflash": {"available": False, "mlxAvailable": False, "vllmAvailable": False, "supportedModels": []},
        "vllmAvailable": False, "mlxAvailable": True, "mlxLmAvailable": True, "mlxUsable": True,
        "ggufAvailable": True, "converterAvailable": False, "nativePython": "/usr/bin/python3",
        "llamaServerPath": "/usr/local/bin/llama-server", "llamaServerTurboPath": None,
        "llamaCliPath": None, "nativeRuntimeMessage": None,
        "totalMemoryGb": 64, "availableMemoryGb": 32, "usedMemoryGb": 32,
        "swapUsedGb": 0, "swapTotalGb": 0, "compressedMemoryGb": 0,
        "memoryPressurePercent": 50.0, "cpuUtilizationPercent": 10.0,
        "gpuUtilizationPercent": None, "spareHeadroomGb": 26.0,
        "battery": None, "runningLlmProcesses": [], "uptimeMinutes": 1.0,
    }


class _WanTestRuntime:
    class _Caps:
        pythonExecutable = "/usr/bin/python3"
        def to_dict(self): return {"pythonExecutable": self.pythonExecutable, "ggufAvailable": True}
    capabilities = _Caps()
    def refresh_capabilities(self, *, force=False): return self.capabilities
    def status(self, **kwargs): return {"engine": "mock", "loadedModel": None, "nativeBackends": {}}


class WanInstallEndpointsTests(unittest.TestCase):
    """Endpoint shape + dispatch checks. The job worker thread is mocked
    so the test doesn't actually spawn the convert subprocess."""

    def setUp(self):
        import tempfile
        from fastapi.testclient import TestClient
        from backend_service.app import create_app
        from backend_service.state import ChaosEngineState

        self._tempdir = tempfile.TemporaryDirectory()
        state = ChaosEngineState(
            system_snapshot_provider=_wan_test_system_snapshot,
            settings_path=Path(self._tempdir.name) / "settings.json",
            benchmarks_path=Path(self._tempdir.name) / "benchmarks.json",
            chat_sessions_path=Path(self._tempdir.name) / "chats.json",
        )
        state.runtime = _WanTestRuntime()
        self.client = TestClient(create_app(state=state, api_token=_TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {_TEST_API_TOKEN}"})

    def tearDown(self):
        self._tempdir.cleanup()

    def test_inventory_lists_all_supported_repos(self):
        resp = self.client.get("/api/setup/mlx-video-wan/inventory")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("items", body)
        self.assertIn("convertRoot", body)
        self.assertIn("rawRoot", body)
        repos = {item["repo"] for item in body["items"]}
        self.assertEqual(repos, set(SUPPORTED_RAW_REPOS))

    def test_inventory_items_carry_size_hint(self):
        resp = self.client.get("/api/setup/mlx-video-wan/inventory")
        for item in resp.json()["items"]:
            self.assertIn("approxRawSizeGb", item)
            self.assertIn("converted", item)
            self.assertIn("status", item)

    def test_install_rejects_unsupported_repo_with_400(self):
        resp = self.client.post(
            "/api/setup/install-mlx-video-wan",
            json={"repo": "Lightricks/LTX-Video"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported Wan repo", resp.json()["detail"])

    def test_install_returns_job_state_immediately(self):
        # Mock the worker so the test doesn't actually start a thread
        # that would fail preflight on a non-Apple-Silicon CI machine.
        with patch(
            "backend_service.routes.setup._wan_install_job_worker",
        ), patch(
            "backend_service.routes.setup.threading.Thread",
        ) as mock_thread:
            mock_thread.return_value = MagicMock()
            resp = self.client.post(
                "/api/setup/install-mlx-video-wan",
                json={"repo": "Wan-AI/Wan2.1-T2V-1.3B"},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("id", body)
        self.assertEqual(body["repo"], "Wan-AI/Wan2.1-T2V-1.3B")
        self.assertIn("phase", body)
        self.assertIn("packageTotal", body)

    def test_status_endpoint_returns_job_snapshot(self):
        resp = self.client.get("/api/setup/install-mlx-video-wan/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        # Shape contract — must always contain these keys for the UI
        # InstallLogPanel.
        for key in ("id", "phase", "message", "packageIndex", "packageTotal", "percent", "attempts", "done"):
            self.assertIn(key, body)


if __name__ == "__main__":
    unittest.main()
