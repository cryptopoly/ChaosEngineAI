import unittest
from unittest.mock import patch, MagicMock

from backend_service.helpers.gpu import GPUMonitor, get_gpu_metrics


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

    def test_snapshot_vram_values_are_numeric(self):
        monitor = GPUMonitor()
        snapshot = monitor.snapshot()
        self.assertIsInstance(snapshot["vram_total_gb"], (int, float))
        self.assertIsInstance(snapshot["vram_used_gb"], (int, float))


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


if __name__ == "__main__":
    unittest.main()
