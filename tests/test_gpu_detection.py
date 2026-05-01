"""Tests for the Windows / Linux GPU detection helper.

The pre-fix path returned system RAM via ``psutil.virtual_memory().total``
when ``nvidia-smi`` wasn't on PATH — so an RTX 4090 box on Windows showed
12 GB total in the safety estimator instead of 24 GB. The new path probes
``torch.cuda`` via a short-lived subprocess (so we don't lock torch DLLs
in the backend process and break the next ``Install GPU runtime``), then
falls back to ``nvidia-smi``, and only returns ``vram_total_gb=None`` when
neither answers. The frontend treats ``None`` as "unknown" and skips the
spurious crash warning.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from backend_service.helpers import gpu as gpu_module


def _fake_completed_process(returncode: int, stdout: str, stderr: str = ""):
    """Build a CompletedProcess-shaped mock for ``subprocess.run``."""
    return mock.MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


class SnapshotTorchCudaTests(unittest.TestCase):
    def setUp(self) -> None:
        gpu_module.reset_vram_total_cache()
        self.monitor = gpu_module.GPUMonitor()
        # Force the monitor onto the nvidia path even when running these
        # tests on a Mac developer machine.
        self.monitor._system = "Linux"

    def tearDown(self) -> None:
        gpu_module.reset_vram_total_cache()

    def test_torch_cuda_returns_full_vram_for_rtx_4090(self) -> None:
        twenty_four_gb = 24 * 1024 ** 3
        free = 22 * 1024 ** 3
        used = twenty_four_gb - free
        payload = json.dumps({
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "total": twenty_four_gb,
            "used": used,
        })
        with mock.patch.object(self.monitor, "_resolve_python_executable", return_value="/usr/bin/python3"), \
             mock.patch("backend_service.helpers.gpu.subprocess.run", return_value=_fake_completed_process(0, payload)):
            snapshot = self.monitor._snapshot_torch_cuda()
        self.assertIsNotNone(snapshot)
        assert snapshot is not None  # type narrow
        self.assertEqual(snapshot["gpu_name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(snapshot["vram_total_gb"], 24.0)
        # 24 - 22 = 2 GB used.
        self.assertEqual(snapshot["vram_used_gb"], 2.0)

    def test_torch_cuda_unavailable_returns_none(self) -> None:
        # Subprocess exits 0 with empty stdout — the inline script printed
        # nothing because torch.cuda.is_available() was False.
        with mock.patch.object(self.monitor, "_resolve_python_executable", return_value="/usr/bin/python3"), \
             mock.patch("backend_service.helpers.gpu.subprocess.run", return_value=_fake_completed_process(0, "")):
            snapshot = self.monitor._snapshot_torch_cuda()
        self.assertIsNone(snapshot)

    def test_torch_not_installed_returns_none(self) -> None:
        # Subprocess exits 0 with empty stdout — the inline script's
        # ``import torch`` raised, the except branch did sys.exit(0).
        with mock.patch.object(self.monitor, "_resolve_python_executable", return_value="/usr/bin/python3"), \
             mock.patch("backend_service.helpers.gpu.subprocess.run", return_value=_fake_completed_process(0, "")):
            snapshot = self.monitor._snapshot_torch_cuda()
        self.assertIsNone(snapshot)

    def test_subprocess_error_returns_none(self) -> None:
        with mock.patch.object(self.monitor, "_resolve_python_executable", return_value="/usr/bin/python3"), \
             mock.patch(
                "backend_service.helpers.gpu.subprocess.run",
                side_effect=FileNotFoundError("python3 missing"),
             ):
            snapshot = self.monitor._snapshot_torch_cuda()
        self.assertIsNone(snapshot)

    def test_no_python_executable_returns_none(self) -> None:
        with mock.patch.object(self.monitor, "_resolve_python_executable", return_value=None):
            snapshot = self.monitor._snapshot_torch_cuda()
        self.assertIsNone(snapshot)

    def test_does_not_import_torch_in_main_process(self) -> None:
        """Critical: importing torch in-process locks Windows DLLs and
        breaks the next Install GPU runtime click. The probe MUST go via
        a child process so its DLL handles are released on exit."""
        twenty_four_gb = 24 * 1024 ** 3
        payload = json.dumps({"gpu_name": "RTX 4090", "total": twenty_four_gb, "used": 0})
        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(list(cmd))
            return _fake_completed_process(0, payload)

        with mock.patch.object(self.monitor, "_resolve_python_executable", return_value="/usr/bin/python3"), \
             mock.patch("backend_service.helpers.gpu.subprocess.run", side_effect=fake_run):
            self.monitor._snapshot_torch_cuda()
        self.assertEqual(len(captured), 1)
        cmd = captured[0]
        # The probe must spawn a Python with a -c script containing
        # 'import torch'. If the implementation ever switches back to
        # an in-process import this assertion will catch it.
        self.assertEqual(cmd[1], "-c")
        self.assertIn("import torch", cmd[2])

    def test_skipped_on_macos(self) -> None:
        self.monitor._system = "Darwin"
        snapshot = self.monitor._snapshot_torch_cuda()
        self.assertIsNone(snapshot)


class SnapshotNvidiaTests(unittest.TestCase):
    def setUp(self) -> None:
        gpu_module.reset_vram_total_cache()
        self.monitor = gpu_module.GPUMonitor()
        self.monitor._system = "Linux"

    def tearDown(self) -> None:
        gpu_module.reset_vram_total_cache()

    def test_falls_back_to_no_gpu_when_torch_and_nvidia_smi_both_fail(self) -> None:
        with mock.patch.object(self.monitor, "_snapshot_torch_cuda", return_value=None), \
             mock.patch("subprocess.check_output", side_effect=FileNotFoundError):
            snapshot = self.monitor._snapshot_nvidia()
        self.assertEqual(snapshot["gpu_name"], "No GPU detected")
        self.assertIsNone(snapshot["vram_total_gb"])
        self.assertIsNone(snapshot["vram_used_gb"])

    def test_does_not_fall_back_to_system_ram(self) -> None:
        """The whole point of this fix: don't lie that system RAM is VRAM."""
        with mock.patch.object(self.monitor, "_snapshot_torch_cuda", return_value=None), \
             mock.patch("subprocess.check_output", side_effect=FileNotFoundError):
            snapshot = self.monitor._snapshot_nvidia()
        self.assertNotEqual(snapshot["gpu_name"], "System RAM (no GPU detected)")

    def test_torch_cuda_takes_precedence_over_nvidia_smi(self) -> None:
        torch_snapshot = {
            "gpu_name": "RTX 4090",
            "vram_total_gb": 24.0,
            "vram_used_gb": 1.0,
            "utilization_pct": None,
            "temperature_c": None,
            "power_w": None,
        }
        with mock.patch.object(self.monitor, "_snapshot_torch_cuda", return_value=torch_snapshot), \
             mock.patch("subprocess.check_output") as mock_subprocess:
            snapshot = self.monitor._snapshot_nvidia()
        self.assertEqual(snapshot["vram_total_gb"], 24.0)
        mock_subprocess.assert_not_called()


class GetDeviceVramTotalGbTests(unittest.TestCase):
    def setUp(self) -> None:
        gpu_module.reset_vram_total_cache()

    def tearDown(self) -> None:
        gpu_module.reset_vram_total_cache()

    def test_returns_none_when_snapshot_has_no_vram(self) -> None:
        with mock.patch.object(
            gpu_module._monitor,
            "snapshot",
            return_value={"vram_total_gb": None},
        ):
            self.assertIsNone(gpu_module.get_device_vram_total_gb())

    def test_returns_float_when_snapshot_has_vram(self) -> None:
        with mock.patch.object(
            gpu_module._monitor,
            "snapshot",
            return_value={"vram_total_gb": 24.0},
        ):
            self.assertEqual(gpu_module.get_device_vram_total_gb(), 24.0)

    def test_caches_result_for_process_lifetime(self) -> None:
        with mock.patch.object(
            gpu_module._monitor,
            "snapshot",
            return_value={"vram_total_gb": 24.0},
        ) as mock_snapshot:
            gpu_module.get_device_vram_total_gb()
            gpu_module.get_device_vram_total_gb()
            gpu_module.get_device_vram_total_gb()
        self.assertEqual(mock_snapshot.call_count, 1)


if __name__ == "__main__":
    unittest.main()
