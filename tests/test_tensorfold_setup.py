"""Setup-side tests for the TensorFold lane: install detection, the capability
cache refresh, the install job and its status endpoints, and the system
snapshot payload the launch settings read."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.inference import capabilities as caps_module
from backend_service.inference.base import BackendCapabilities
from backend_service.routes.setup import tensorfold as setup_tf
from backend_service.state import ChaosEngineState
from tests.test_setup_routes import TEST_API_TOKEN, _fake_system_snapshot

def _install(venv: Path, version_file: Path, *, version: str = "0.5.0") -> None:
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "python").write_text("")
    (venv / "bin" / "tensorfold").write_text("")
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text(f"{version}\n2026-09-30T00:00:00Z\nabc123\n")


class DetectTensorFoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.venv = self.tmp / "tensorfold-venv"
        self.version_file = self.tmp / "bin" / "tensorfold.version"
        for name, value in (("_TENSORFOLD_VENV", self.venv), ("_TENSORFOLD_VERSION_FILE", self.version_file)):
            patcher = mock.patch.object(caps_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_absent(self) -> None:
        self.assertEqual(caps_module._detect_tensorfold(), (False, None, None))

    def test_version_file_alone_is_not_an_install(self) -> None:
        self.version_file.parent.mkdir(parents=True)
        self.version_file.write_text("0.5.0\n")
        self.assertEqual(caps_module._detect_tensorfold(), (False, None, None))

    def test_needs_the_cli_script_as_well(self) -> None:
        _install(self.venv, self.version_file)
        (self.venv / "bin" / "tensorfold").unlink()
        self.assertFalse(caps_module._detect_tensorfold()[0])

    def test_installed(self) -> None:
        _install(self.venv, self.version_file, version="0.5.0")
        available, python, version = caps_module._detect_tensorfold()
        self.assertTrue(available)
        self.assertEqual(python, str(self.venv / "bin" / "python"))
        self.assertEqual(version, "0.5.0")

    def test_refresh_updates_the_cached_capabilities_in_place(self) -> None:
        # Importing the app starts the background capability probe, which
        # rebinds the module cache when it finishes; let it land first so it
        # cannot replace the cache this test installs.
        for thread in threading.enumerate():
            if thread.name == "chaosengine-capability-probe":
                thread.join(timeout=120)
        cached = BackendCapabilities(pythonExecutable=sys.executable, mlxAvailable=True, mlxLmAvailable=True, mlxUsable=True)
        self.assertFalse(cached.tensorfoldAvailable)
        with mock.patch.object(caps_module, "_capability_cache", (time.time(), cached)):
            _install(self.venv, self.version_file)
            caps_module.refresh_install_detection()
        self.assertTrue(cached.tensorfoldAvailable)
        self.assertEqual(cached.tensorfoldVersion, "0.5.0")
        self.assertEqual(cached.tensorfoldPythonPath, str(self.venv / "bin" / "python"))

    def test_refresh_without_a_cache_is_a_no_op(self) -> None:
        with mock.patch.object(caps_module, "_capability_cache", None):
            caps_module.refresh_install_detection()


class TensorFoldSetupRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tempdir.name)
        self.addCleanup(self.tempdir.cleanup)
        state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            settings_path=self.tmp / "settings.json",
            benchmarks_path=self.tmp / "benchmarks.json",
            chat_sessions_path=self.tmp / "chats.json",
        )
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})

        self.venv = self.tmp / "tensorfold-venv"
        self.version_file = self.tmp / "bin" / "tensorfold.version"
        for name, value in (("_TENSORFOLD_VENV_DIR", self.venv), ("_TENSORFOLD_VERSION_FILE", self.version_file)):
            patcher = mock.patch.object(setup_tf, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # a fresh job per test
        job_patch = mock.patch.object(setup_tf, "_JOB", setup_tf._TensorfoldJobState())
        job_patch.start()
        self.addCleanup(job_patch.stop)

    def _script(self, body: str) -> Path:
        script = self.tmp / "install-tensorfold.sh"
        script.write_text("#!/usr/bin/env bash\n" + body)
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        patcher = mock.patch.object(setup_tf, "_INSTALL_SCRIPT", script)
        patcher.start()
        self.addCleanup(patcher.stop)
        return script

    def _wait_for_job(self, timeout: float = 20.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.client.get("/api/setup/install-tensorfold/status").json()
            if state["done"]:
                return state
            time.sleep(0.05)
        self.fail("install job did not finish")

    def test_status_when_missing(self) -> None:
        status = self.client.get("/api/setup/tensorfold-status").json()
        self.assertFalse(status["installed"])
        self.assertIsNone(status["version"])
        self.assertIsNone(status["venvPath"])
        self.assertIn("supported", status)

    def test_status_when_installed(self) -> None:
        _install(self.venv, self.version_file, version="0.5.0")
        status = self.client.get("/api/setup/tensorfold-status").json()
        self.assertTrue(status["installed"])
        self.assertEqual(status["version"], "0.5.0")
        self.assertEqual(status["ref"], "abc123")
        self.assertEqual(status["venvPath"], str(self.venv))

    @unittest.skipIf(sys.platform == "win32", "install scripts run under bash")
    def test_install_job_follows_the_script_phases(self) -> None:
        version_file = self.version_file
        self._script(
            f"""
echo PHASE:preflight
echo "checking python"
echo PHASE:creating-venv
echo PHASE:installing
echo "pip output"
echo PHASE:verifying
mkdir -p "{self.venv}/bin" "{version_file.parent}"
touch "{self.venv}/bin/python" "{self.venv}/bin/tensorfold"
printf '0.5.0\\nnow\\nabc123\\n' > "{version_file}"
echo OK
"""
        )
        refreshed = mock.Mock()
        with mock.patch("backend_service.inference.capabilities.refresh_install_detection", refreshed):
            started = self.client.post("/api/setup/install-tensorfold").json()
            self.assertEqual(started["id"], "tensorfold-install")
            final = self._wait_for_job()
        self.assertEqual(final["phase"], "done")
        self.assertIsNone(final["error"])
        self.assertEqual(final["percent"], 100.0)
        self.assertIn("0.5.0", final["message"])
        self.assertEqual([a["phase"] for a in final["attempts"]], ["preflight", "creating-venv", "installing", "verifying"])
        self.assertTrue(all(a["ok"] for a in final["attempts"]))
        self.assertIn("pip output", final["attempts"][2]["output"])
        refreshed.assert_called_once()
        self.assertTrue(self.client.get("/api/setup/tensorfold-status").json()["installed"])

    @unittest.skipIf(sys.platform == "win32", "install scripts run under bash")
    def test_install_job_surfaces_the_failure_reason(self) -> None:
        self._script('echo PHASE:preflight\necho "FAIL:TensorFold needs native arm64 Python"\nexit 1\n')
        refreshed = mock.Mock()
        with mock.patch("backend_service.inference.capabilities.refresh_install_detection", refreshed):
            self.client.post("/api/setup/install-tensorfold")
            final = self._wait_for_job()
        self.assertEqual(final["phase"], "error")
        self.assertEqual(final["error"], "TensorFold needs native arm64 Python")
        self.assertFalse(final["attempts"][-1]["ok"])
        refreshed.assert_not_called()

    @unittest.skipIf(sys.platform == "win32", "install scripts run under bash")
    def test_install_script_runs_without_the_apps_python_environment(self) -> None:
        out = self.tmp / "env.txt"
        self._script(f'echo PHASE:preflight\nenv | grep -E "^(PYTHONHOME|PYTHONPATH|DYLD_LIBRARY_PATH|KEEP_ME)=" > "{out}" || true\necho OK\n')
        with mock.patch.dict(
            os.environ,
            {"PYTHONHOME": "/embedded", "PYTHONPATH": "/extras", "DYLD_LIBRARY_PATH": "/lib", "KEEP_ME": "1"},
        ):
            self.client.post("/api/setup/install-tensorfold")
            self._wait_for_job()
        seen = out.read_text()
        self.assertIn("KEEP_ME=1", seen)
        for name in ("PYTHONHOME", "PYTHONPATH", "DYLD_LIBRARY_PATH"):
            self.assertNotIn(name, seen)

    @unittest.skipIf(sys.platform == "win32", "install scripts run under bash")
    def test_second_post_while_running_does_not_start_another_job(self) -> None:
        self._script("echo PHASE:preflight\nsleep 1\necho OK\n")
        first = self.client.post("/api/setup/install-tensorfold").json()
        second = self.client.post("/api/setup/install-tensorfold").json()
        self.assertEqual(first["startedAt"], second["startedAt"])
        self._wait_for_job()


class TensorFoldSnapshotTests(unittest.TestCase):
    def test_system_snapshot_lists_supported_and_exclusive_models(self) -> None:
        from backend_service.helpers.system import _build_system_snapshot

        snapshot = _build_system_snapshot("0", time.time())
        info = snapshot["tensorfold"]
        self.assertIn("available", info)
        self.assertIn("version", info)
        self.assertIn("Vontra/Qwen3.8-27B-MLX-4bit", info["supportedModels"])
        self.assertIn("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP", info["exclusiveModels"])
        self.assertNotIn("mlx-community/gemma-4-26b-a4b-it-4bit", info["exclusiveModels"])
        self.assertEqual(info["minMemoryGb"]["Vontra/GLM-5.3-Flash-MLX-4bit-MTP"], 256)

    def test_capability_payload_carries_the_install_fields(self) -> None:
        payload = BackendCapabilities(
            pythonExecutable=sys.executable, mlxAvailable=False, mlxLmAvailable=False, mlxUsable=False,
            tensorfoldAvailable=True, tensorfoldPythonPath="/x/python", tensorfoldVersion="0.5.0",
        ).to_dict()
        self.assertTrue(payload["tensorfoldAvailable"])
        self.assertEqual(payload["tensorfoldVersion"], "0.5.0")
        self.assertEqual(payload["tensorfoldPythonPath"], "/x/python")


class InstallScriptTests(unittest.TestCase):
    """The installer refuses everywhere but Apple Silicon, before touching anything."""

    @unittest.skipIf(sys.platform in {"darwin", "win32"}, "asserts the non-Mac refusal")
    def test_refuses_off_macos(self) -> None:
        import subprocess

        script = Path(__file__).parents[1] / "scripts" / "install-tensorfold.sh"
        with tempfile.TemporaryDirectory() as home:
            result = subprocess.run(
                ["bash", str(script)], capture_output=True, text=True, env={**os.environ, "HOME": home}, timeout=60
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("PHASE:preflight", result.stdout)
            self.assertIn("FAIL:", result.stdout)
            self.assertFalse((Path(home) / ".chaosengine" / "tensorfold-venv").exists())

    def test_script_pins_a_commit_and_is_executable(self) -> None:
        script = Path(__file__).parents[1] / "scripts" / "install-tensorfold.sh"
        text = script.read_text()
        self.assertRegex(text, r'TENSORFOLD_REF="\$\{TENSORFOLD_REF:-[0-9a-f]{40}\}"')
        if sys.platform != "win32":
            self.assertTrue(os.access(script, os.X_OK))


if __name__ == "__main__":
    unittest.main()
