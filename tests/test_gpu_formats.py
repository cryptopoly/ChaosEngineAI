import unittest
from unittest import mock

from backend_service.helpers import gpu_formats


class GpuFormatGateTests(unittest.TestCase):
    def test_nvfp4_is_blackwell_only(self):
        for cap, expected in {
            (7, 5): False,   # Turing
            (8, 0): False,   # A100
            (8, 6): False,   # RTX 30-series
            (8, 9): False,   # RTX 4090: no native NVFP4
            (9, 0): False,   # H100
            (10, 0): True,   # B200
            (12, 0): True,   # RTX 5090
            (12, 1): True,   # DGX Spark
        }.items():
            self.assertEqual(gpu_formats.nvfp4_supported(cap), expected, cap)
        self.assertFalse(gpu_formats.nvfp4_supported(None))

    def test_fp8_starts_at_ada(self):
        self.assertFalse(gpu_formats.fp8_supported((8, 6)))
        for cap in ((8, 9), (9, 0), (12, 0)):
            self.assertTrue(gpu_formats.fp8_supported(cap), cap)
        self.assertFalse(gpu_formats.fp8_supported(None))

    def test_hardware_formats_reads_the_card_when_not_given_one(self):
        with mock.patch.object(gpu_formats, "device_capability", return_value=(8, 9)):
            self.assertEqual(gpu_formats.hardware_formats(), {"fp8": True, "nvfp4": False})
        with mock.patch.object(gpu_formats, "device_capability", return_value=None):
            self.assertEqual(gpu_formats.hardware_formats(), {"fp8": False, "nvfp4": False})
        self.assertEqual(gpu_formats.hardware_formats((12, 0)), {"fp8": True, "nvfp4": True})


if __name__ == "__main__":
    unittest.main()
