"""Tests for Phase 2.0.5-B pre-flight memory gate.

The gate refuses chat generations when the host is already memory-starved
(low available RAM or high pressure). It must produce actionable refusals
without false-positive blocks during normal operation.
"""

import unittest

from backend_service.helpers.memory_gate import (
    gate_chat_generation,
    gate_image_generation,
    gate_video_generation,
)


class GateChatGenerationTests(unittest.TestCase):
    def test_passes_when_memory_is_healthy(self):
        result = gate_chat_generation(available_gb=12.0, pressure_percent=45.0)
        self.assertIsNone(result)

    def test_refuses_when_available_below_floor(self):
        result = gate_chat_generation(available_gb=0.4, pressure_percent=70.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["code"], "memory_gate_low_available")
        self.assertIn("0.4", result["message"])
        self.assertIn("free", result["message"])

    def test_refuses_when_pressure_exceeds_ceiling(self):
        # Ceiling raised to 98% in the post-launch tuning pass — only
        # near-OOM pressure trips the gate now since macOS routinely
        # sits at 90-97% during normal use thanks to compression.
        result = gate_chat_generation(available_gb=2.5, pressure_percent=99.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["code"], "memory_gate_high_pressure")
        self.assertIn("99", result["message"])

    def test_passes_at_high_macos_pressure_with_headroom(self):
        # 95% pressure with several GB free is normal macOS — must not
        # trip the gate. This is the regression fix from the user
        # report ("models that ran fine before now blocked at 97%").
        result = gate_chat_generation(available_gb=4.0, pressure_percent=95.0)
        self.assertIsNone(result)

    def test_low_available_takes_precedence_over_pressure(self):
        # When both signals trip, the low-available message is more
        # actionable (smaller numbers users intuitively grasp), so check it
        # wins the dispatch order.
        result = gate_chat_generation(available_gb=0.2, pressure_percent=99.0)
        self.assertEqual(result["code"], "memory_gate_low_available")

    def test_custom_thresholds_override_defaults(self):
        # A more permissive caller (e.g. a tiny test prompt) can lower the
        # floor without breaking the default policy for normal callers.
        result = gate_chat_generation(
            available_gb=0.5,
            pressure_percent=70.0,
            min_available_gb=0.25,
            max_pressure_percent=92.0,
        )
        self.assertIsNone(result)

    def test_boundary_at_floor_passes(self):
        # `>=` floor passes — only strictly-below trips the gate. Otherwise
        # a system stable at exactly the floor would be perpetually refused.
        result = gate_chat_generation(available_gb=1.0, pressure_percent=70.0)
        self.assertIsNone(result)


class GateImageGenerationTests(unittest.TestCase):
    def test_passes_when_memory_is_healthy(self):
        result = gate_image_generation(available_gb=12.0, pressure_percent=45.0)
        self.assertIsNone(result)

    def test_refuses_below_image_floor(self):
        # Image needs more headroom than chat — 3.5 GB is fine for chat
        # but should trip the image gate (default 4 GB floor).
        result = gate_image_generation(available_gb=3.5, pressure_percent=70.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["code"], "memory_gate_image_low_available")

    def test_refuses_when_image_pressure_high(self):
        result = gate_image_generation(available_gb=10.0, pressure_percent=96.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["code"], "memory_gate_image_high_pressure")


class GateImageModelSizeTests(unittest.TestCase):
    def test_refuses_a_model_larger_than_free_memory(self):
        # Qwen-Image (57 GB) on a 64 GB Mac: the OS kills the backend mid-load.
        result = gate_image_generation(
            available_gb=58.0, pressure_percent=30.0, model_gb=57.0, model_name="Qwen-Image",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["code"], "memory_gate_image_model_too_large")
        self.assertIn("Qwen-Image", result["message"])
        self.assertIn("58.0", result["message"])

    def test_passes_a_model_that_fits(self):
        self.assertIsNone(gate_image_generation(
            available_gb=40.0, pressure_percent=30.0, model_gb=23.8, model_name="FLUX.1 Dev",
        ))

    def test_a_measured_peak_footprint_is_not_charged_headroom_twice(self):
        # Qwen-Image-2.1 peaks at 50 GB: 1.1 x 50 = 55 GB would refuse a 64 GB Mac with 54 GB free.
        refused = gate_image_generation(available_gb=54.0, pressure_percent=30.0, model_gb=50.0, model_name="Qwen-Image-2.1")
        self.assertIsNotNone(refused)
        allowed = gate_image_generation(
            available_gb=54.0, pressure_percent=30.0, model_gb=50.0, model_name="Qwen-Image-2.1", headroom=1.0,
        )
        self.assertIsNone(allowed)
        still_refused = gate_image_generation(
            available_gb=48.0, pressure_percent=30.0, model_gb=50.0, model_name="Qwen-Image-2.1", headroom=1.0,
        )
        self.assertIsNotNone(still_refused)

    def test_on_cuda_the_size_is_judged_against_free_gpu_memory_not_ram(self):
        # RTX 4090 (about 22 GB free) in a 128 GB workstation: FLUX.1 dev at 4096 x 4096 needs ~54 GB more.
        args = dict(
            pressure_percent=30.0, model_gb=23.8, model_name="FLUX.1 Dev",
            width=4096, height=4096, reference_pixels=1024 * 1024, footprint_gb=23.8, headroom=1.0,
        )
        by_ram = gate_image_generation(available_gb=120.0, **args)
        self.assertIsNone(by_ram)
        by_gpu = gate_image_generation(available_gb=120.0, size_available_gb=22.0, **args)
        self.assertIsNotNone(by_gpu)
        self.assertEqual(by_gpu["code"], "memory_gate_image_size_too_large")
        self.assertIn("GPU memory", by_gpu["message"])

    def test_on_cuda_a_size_that_fits_the_card_passes_even_when_ram_could_not_hold_it(self):
        result = gate_image_generation(
            available_gb=30.0, pressure_percent=30.0, model_gb=23.8, model_name="FLUX.1 Dev",
            width=2048, height=2048, reference_pixels=1024 * 1024, footprint_gb=23.8, headroom=1.0,
            size_available_gb=22.0,
        )
        self.assertIsNone(result)

    def test_no_size_means_no_size_check(self):
        self.assertIsNone(gate_image_generation(available_gb=8.0, pressure_percent=30.0))


class GateVideoGenerationTests(unittest.TestCase):
    def test_passes_when_memory_is_healthy(self):
        result = gate_video_generation(available_gb=18.0, pressure_percent=40.0)
        self.assertIsNone(result)

    def test_video_floor_strictest_of_three(self):
        # 5 GB available is fine for chat (1 GB) and image (4 GB) but
        # below the video floor (6 GB).
        result = gate_video_generation(available_gb=5.0, pressure_percent=70.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["code"], "memory_gate_video_low_available")

    def test_refuses_when_video_pressure_high(self):
        result = gate_video_generation(available_gb=20.0, pressure_percent=94.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["code"], "memory_gate_video_high_pressure")


if __name__ == "__main__":
    unittest.main()


class MacosReclaimableMemoryTests(unittest.TestCase):
    """psutil's "available" misses active file cache; the OS's own level does not."""

    def _run(self, *, platform: str, stdout: str = "87\n", returncode: int = 0):
        from unittest import mock

        from backend_service.helpers import memory_gate

        completed = mock.Mock(stdout=stdout, returncode=returncode)
        with mock.patch("sys.platform", platform), \
                mock.patch("subprocess.run", return_value=completed):
            return memory_gate._macos_reclaimable_gb(64 * 1024 ** 3)

    def test_uses_the_os_level_on_macos(self):
        self.assertAlmostEqual(self._run(platform="darwin"), 64 * 0.87, places=3)

    def test_is_none_off_macos(self):
        self.assertIsNone(self._run(platform="linux"))

    def test_ignores_an_unreadable_level(self):
        self.assertIsNone(self._run(platform="darwin", stdout="not a number"))
        self.assertIsNone(self._run(platform="darwin", stdout="0\n"))

    def test_snapshot_never_reports_less_than_psutil(self):
        from unittest import mock

        from backend_service.helpers import memory_gate

        with mock.patch.object(memory_gate, "_macos_reclaimable_gb", return_value=10.0):
            available_gb, _ = memory_gate.snapshot_memory_signals()
        import psutil

        self.assertGreaterEqual(available_gb, round(psutil.virtual_memory().available / 1024 ** 3, 1) - 1.0)


class PressureSignalTests(unittest.TestCase):
    """Pressure comes from available memory; swap must not count as in use."""

    GB = 1024 ** 3

    def _signals(self, *, total_gb, available_gb, swap_used_gb, macos_level_gb=None):
        from types import SimpleNamespace
        from unittest import mock

        from backend_service.helpers import memory_gate

        vm = SimpleNamespace(total=int(total_gb * self.GB), available=int(available_gb * self.GB), used=0)
        swap = SimpleNamespace(used=int(swap_used_gb * self.GB), total=int(64 * self.GB))
        with mock.patch("psutil.virtual_memory", return_value=vm), \
                mock.patch("psutil.swap_memory", return_value=swap), \
                mock.patch.object(memory_gate, "_macos_reclaimable_gb", return_value=macos_level_gb):
            return memory_gate.snapshot_memory_signals()

    def test_lingering_swap_does_not_raise_pressure(self):
        # Live reading from a 64 GB Mac whose OS rated pressure normal: the old
        # formula (used + compressed + swap) gave 100% once a 33 GB model loaded.
        available, pressure = self._signals(total_gb=64, available_gb=34.4, swap_used_gb=11.6)
        self.assertEqual(available, 34.4)
        self.assertAlmostEqual(pressure, 46.25, delta=0.1)

    def test_macos_level_raises_available_and_lowers_pressure(self):
        available, pressure = self._signals(
            total_gb=64, available_gb=34.4, swap_used_gb=11.6, macos_level_gb=64 * 0.85,
        )
        self.assertEqual(available, 54.4)
        self.assertAlmostEqual(pressure, 15.0, places=1)

    def test_a_large_loaded_model_still_lets_chat_through(self):
        # 33 GB model resident, 20 GB still available, 12 GB swap: proceed.
        available, pressure = self._signals(total_gb=64, available_gb=20.0, swap_used_gb=12.0)
        self.assertIsNone(gate_chat_generation(available, pressure))

    def test_nearly_out_of_memory_is_still_refused(self):
        available, pressure = self._signals(total_gb=64, available_gb=0.6, swap_used_gb=0)
        self.assertIsNotNone(gate_chat_generation(available, pressure))
        self.assertIsNotNone(gate_video_generation(available, pressure))

    def test_pressure_helper_bounds(self):
        from backend_service.helpers.memory_gate import pressure_percent_from_available

        self.assertEqual(pressure_percent_from_available(0, 5), 0.0)
        self.assertEqual(pressure_percent_from_available(64, 80), 0.0)
        self.assertEqual(pressure_percent_from_available(64, 0), 100.0)


class ModelMemoryTests(unittest.TestCase):
    def test_apple_silicon_prefers_the_mps_runtime_footprint(self):
        from backend_service.helpers.memory_gate import model_memory_gb

        row = {"sizeGb": 16.4, "runtimeFootprintGb": 14.0, "runtimeFootprintMpsGb": 23.0}
        self.assertEqual(model_memory_gb(row, apple_silicon=True), 23.0)
        self.assertEqual(model_memory_gb({"sizeGb": 6.9, "runtimeFootprintGb": 14.0}, apple_silicon=True), 14.0)

    def test_elsewhere_the_larger_of_download_and_footprint(self):
        from backend_service.helpers.memory_gate import model_memory_gb

        self.assertEqual(model_memory_gb({"sizeGb": 6.9, "runtimeFootprintGb": 14.0}, apple_silicon=False), 14.0)
        self.assertEqual(model_memory_gb({"sizeGb": 45.0, "runtimeFootprintGb": 39.0}, apple_silicon=False), 45.0)
        self.assertIsNone(model_memory_gb({}, apple_silicon=False))

    def test_video_model_too_large_is_refused(self):
        result = gate_video_generation(available_gb=30.0, pressure_percent=40.0, model_gb=48.0, model_name="LTX-2.3")
        self.assertEqual(result["code"], "memory_gate_video_model_too_large")
        self.assertIn("LTX-2.3", result["message"])
        self.assertIsNone(gate_video_generation(available_gb=50.0, pressure_percent=40.0, model_gb=48.0))
