import subprocess
import unittest
from unittest.mock import patch, MagicMock

from backend_service.helpers import gpu as gpu_module
from backend_service.helpers.gpu import (
    GPUMonitor,
    get_device_vram_total_gb,
    get_gpu_metrics,
    gpu_status_snapshot,
    nvidia_gpu_present,
    reset_vram_total_cache,
)


_EXPECTED_KEYS = {"gpu_name", "vram_total_gb", "vram_used_gb", "utilization_pct", "temperature_c", "power_w"}


class GPUMonitorSnapshotTests(unittest.TestCase):
    def test_snapshot_returns_expected_keys(self):
        monitor = GPUMonitor()
        snapshot = monitor.snapshot()
        self.assertIsInstance(snapshot, dict)
        for key in _EXPECTED_KEYS:
            self.assertIn(key, snapshot)

    def test_snapshot_gpu_name_is_string(self):
        monitor = GPUMonitor()
        snapshot = monitor.snapshot()
        self.assertIsInstance(snapshot["gpu_name"], str)

    def test_snapshot_vram_values_are_numeric_or_none(self):
        # ``None`` is a valid response when neither torch.cuda nor
        # nvidia-smi can answer — we deliberately don't fall back to
        # system RAM via psutil any more (would mislead the safety
        # estimator). Numeric otherwise.
        monitor = GPUMonitor()
        snapshot = monitor.snapshot()
        self.assertIsInstance(snapshot["vram_total_gb"], (int, float, type(None)))
        self.assertIsInstance(snapshot["vram_used_gb"], (int, float, type(None)))


class GPUMonitorNvidiaTests(unittest.TestCase):
    @patch("backend_service.helpers.gpu.subprocess.check_output")
    def test_nvidia_smi_parsing(self, mock_check_output):
        mock_check_output.return_value = (
            "NVIDIA RTX 4090, 24576, 8192, 45, 62, 280.5"
        )
        monitor = GPUMonitor()
        monitor._system = "Linux"  # Force NVIDIA path
        snapshot = monitor._snapshot_nvidia()

        self.assertEqual(snapshot["gpu_name"], "NVIDIA RTX 4090")
        self.assertAlmostEqual(snapshot["vram_total_gb"], 24.0, places=0)
        self.assertAlmostEqual(snapshot["vram_used_gb"], 8.0, places=0)
        self.assertEqual(snapshot["utilization_pct"], 45.0)
        self.assertEqual(snapshot["temperature_c"], 62.0)
        self.assertAlmostEqual(snapshot["power_w"], 280.5)

    @patch("backend_service.helpers.gpu.subprocess.check_output")
    def test_nvidia_smi_not_found_falls_back(self, mock_check_output):
        mock_check_output.side_effect = FileNotFoundError("nvidia-smi not found")
        monitor = GPUMonitor()
        monitor._system = "Linux"
        snapshot = monitor._snapshot_nvidia()
        # Should still return a valid dict with all keys
        for key in _EXPECTED_KEYS:
            self.assertIn(key, snapshot)

    @patch("backend_service.helpers.gpu.subprocess.check_output")
    def test_nvidia_smi_malformed_output(self, mock_check_output):
        mock_check_output.return_value = "garbage output"
        monitor = GPUMonitor()
        monitor._system = "Linux"
        snapshot = monitor._snapshot_nvidia()
        # Should fall back gracefully
        for key in _EXPECTED_KEYS:
            self.assertIn(key, snapshot)


class GPUMonitorMacOSTests(unittest.TestCase):
    @patch("backend_service.helpers.gpu.subprocess.check_output")
    def test_macos_snapshot_with_sysctl(self, mock_check_output):
        def side_effect(cmd, **kwargs):
            if "machdep.cpu.brand_string" in cmd:
                return "Apple M2 Ultra"
            if "hw.memsize" in cmd:
                return str(48 * 1024**3)  # 48 GB
            return ""

        mock_check_output.side_effect = side_effect
        monitor = GPUMonitor()
        monitor._system = "Darwin"

        # Patch psutil import inside the method
        with patch.dict("sys.modules", {"psutil": MagicMock()}):
            import sys
            mock_psutil = sys.modules["psutil"]
            mock_mem = MagicMock()
            mock_mem.used = 16 * 1024**3
            mock_mem.total = 48 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem

            snapshot = monitor._snapshot_macos()

        self.assertEqual(snapshot["gpu_name"], "Apple M2 Ultra")
        self.assertAlmostEqual(snapshot["vram_total_gb"], 48.0, places=0)

    @patch("backend_service.helpers.gpu.subprocess.check_output")
    def test_macos_sysctl_failure_graceful(self, mock_check_output):
        mock_check_output.side_effect = Exception("sysctl failed")
        monitor = GPUMonitor()
        monitor._system = "Darwin"
        snapshot = monitor._snapshot_macos()
        # Should still return a dict with defaults
        self.assertIn("gpu_name", snapshot)
        self.assertEqual(snapshot["gpu_name"], "Apple Silicon")


class GetGPUMetricsTests(unittest.TestCase):
    def test_returns_dict_with_expected_keys(self):
        metrics = get_gpu_metrics()
        self.assertIsInstance(metrics, dict)
        for key in _EXPECTED_KEYS:
            self.assertIn(key, metrics)


class CachedVramTotalTests(unittest.TestCase):
    """The cache is what keeps the video runtime probe under 15s on Windows."""

    def setUp(self):
        reset_vram_total_cache()

    def tearDown(self):
        reset_vram_total_cache()

    def test_caches_value_after_first_call(self):
        with patch.object(gpu_module._monitor, "snapshot") as mock_snapshot:
            mock_snapshot.return_value = {"vram_total_gb": 24.0}
            first = get_device_vram_total_gb()
            second = get_device_vram_total_gb()
            third = get_device_vram_total_gb()
        self.assertEqual(first, 24.0)
        self.assertEqual(second, 24.0)
        self.assertEqual(third, 24.0)
        # Snapshot must only be called ONCE — that's the whole point of the
        # cache. If this fails the Windows probe regression is back.
        mock_snapshot.assert_called_once()

    def test_caches_none_when_detection_fails(self):
        with patch.object(gpu_module._monitor, "snapshot") as mock_snapshot:
            mock_snapshot.side_effect = RuntimeError("boom")
            first = get_device_vram_total_gb()
            second = get_device_vram_total_gb()
        self.assertIsNone(first)
        self.assertIsNone(second)
        mock_snapshot.assert_called_once()

    def test_caches_none_for_zero_or_missing_value(self):
        with patch.object(gpu_module._monitor, "snapshot") as mock_snapshot:
            mock_snapshot.return_value = {"vram_total_gb": 0}
            self.assertIsNone(get_device_vram_total_gb())
        with patch.object(gpu_module._monitor, "snapshot") as mock_snapshot:
            # Already cached as None — the second snapshot should never be called.
            mock_snapshot.return_value = {"vram_total_gb": 12.0}
            self.assertIsNone(get_device_vram_total_gb())
            mock_snapshot.assert_not_called()


class GpuStatusSnapshotTests(unittest.TestCase):
    """The /api/system/gpu-status endpoint feeds the frontend CPU-fallback banner."""

    def test_snapshot_has_expected_shape(self):
        snapshot = gpu_status_snapshot()
        for key in (
            "platform",
            "nvidiaGpuDetected",
            "torchImported",
            "torchCudaAvailable",
            "torchMpsAvailable",
            "cpuFallbackWarning",
            "recommendation",
        ):
            self.assertIn(key, snapshot)
        self.assertIsInstance(snapshot["platform"], str)
        self.assertIsInstance(snapshot["nvidiaGpuDetected"], bool)
        self.assertIsInstance(snapshot["cpuFallbackWarning"], bool)

    @patch("backend_service.helpers.gpu.platform.system", return_value="Windows")
    @patch("backend_service.helpers.gpu.nvidia_gpu_present", return_value=True)
    def test_recommendation_when_nvidia_present_but_cuda_unavailable(self, _nv, _plat):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": fake_torch}):
            snapshot = gpu_status_snapshot()
        self.assertTrue(snapshot["torchImported"])
        self.assertFalse(snapshot["torchCudaAvailable"])
        self.assertTrue(snapshot["cpuFallbackWarning"])
        self.assertIsNotNone(snapshot["recommendation"])
        # We recommend cu124 now (cu121 has no Python 3.13 wheels and broke
        # fresh Windows installs). Accept either the PyTorch index URL or
        # the in-app button copy.
        self.assertTrue(
            "cu124" in snapshot["recommendation"]
            or "Install CUDA torch" in snapshot["recommendation"]
        )

    @patch("backend_service.helpers.gpu.platform.system", return_value="Windows")
    @patch("backend_service.helpers.gpu.nvidia_gpu_present", return_value=True)
    def test_no_warning_when_cuda_available(self, _nv, _plat):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": fake_torch}):
            snapshot = gpu_status_snapshot()
        self.assertTrue(snapshot["torchCudaAvailable"])
        self.assertFalse(snapshot["cpuFallbackWarning"])
        self.assertIsNone(snapshot["recommendation"])

    @patch("backend_service.helpers.gpu.platform.system", return_value="Darwin")
    @patch("backend_service.helpers.gpu.nvidia_gpu_present", return_value=False)
    def test_no_warning_on_macos_even_with_cpu_torch(self, _nv, _plat):
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": fake_torch}):
            snapshot = gpu_status_snapshot()
        self.assertFalse(snapshot["cpuFallbackWarning"])
        self.assertIsNone(snapshot["recommendation"])

    def test_nvidia_gpu_present_respects_path(self):
        with patch("backend_service.helpers.gpu.shutil.which", return_value="/usr/bin/nvidia-smi"):
            self.assertTrue(nvidia_gpu_present())
        with patch("backend_service.helpers.gpu.shutil.which", return_value=None):
            self.assertFalse(nvidia_gpu_present())


@unittest.skipUnless(hasattr(subprocess, "CREATE_NO_WINDOW"), "Windows-only flag")
class WindowsConsoleSuppressionTests(unittest.TestCase):
    """nvidia-smi must not pop a console window on Windows."""

    def test_subprocess_kwargs_includes_create_no_window(self):
        self.assertIn("creationflags", gpu_module._SUBPROCESS_KWARGS)
        self.assertEqual(
            gpu_module._SUBPROCESS_KWARGS["creationflags"],
            subprocess.CREATE_NO_WINDOW,
        )


if __name__ == "__main__":
    unittest.main()
