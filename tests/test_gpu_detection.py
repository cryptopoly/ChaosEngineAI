"""Tests for the Windows / Linux GPU detection helper.

The pre-fix path returned system RAM via ``psutil.virtual_memory().total``
when ``nvidia-smi`` wasn't on PATH — so an RTX 4090 box on Windows showed
12 GB total in the safety estimator instead of 24 GB. The new path tries
``torch.cuda`` first, falls back to ``nvidia-smi``, and only returns a
``vram_total_gb=None`` when neither answers. The frontend treats ``None``
as "unknown" and skips the spurious crash warning.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from backend_service.helpers import gpu as gpu_module


def _fake_torch_with_cuda(total_bytes: int, free_bytes: int, name: str = "NVIDIA GeForce RTX 4090") -> types.ModuleType:
    cuda = types.SimpleNamespace()
    cuda.is_available = lambda: True
    cuda.current_device = lambda: 0

    class _Props:
        def __init__(self, mem: int, gpu_name: str) -> None:
            self.total_memory = mem
            self.name = gpu_name

    cuda.get_device_properties = lambda device: _Props(total_bytes, name)
    cuda.mem_get_info = lambda device: (free_bytes, total_bytes)

    fake = types.ModuleType("torch")
    fake.cuda = cuda  # type: ignore[attr-defined]
    return fake


def _fake_torch_no_cuda() -> types.ModuleType:
    cuda = types.SimpleNamespace()
    cuda.is_available = lambda: False
    fake = types.ModuleType("torch")
    fake.cuda = cuda  # type: ignore[attr-defined]
    return fake


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
        with mock.patch.dict(sys.modules, {"torch": _fake_torch_with_cuda(twenty_four_gb, free)}):
            snapshot = self.monitor._snapshot_torch_cuda()
        self.assertIsNotNone(snapshot)
        assert snapshot is not None  # type narrow
        self.assertEqual(snapshot["gpu_name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(snapshot["vram_total_gb"], 24.0)
        # 24 - 22 = 2 GB used.
        self.assertEqual(snapshot["vram_used_gb"], 2.0)

    def test_torch_cuda_unavailable_returns_none(self) -> None:
        with mock.patch.dict(sys.modules, {"torch": _fake_torch_no_cuda()}):
            snapshot = self.monitor._snapshot_torch_cuda()
        self.assertIsNone(snapshot)

    def test_torch_not_installed_returns_none(self) -> None:
        # Monkeypatch the import to raise ImportError.
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            # Also remove any previously cached torch entry so the
            # function's ``import torch`` actually invokes the patched
            # ``__import__`` instead of resolving via sys.modules.
            with mock.patch.dict(sys.modules, {}, clear=False):
                sys.modules.pop("torch", None)
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
