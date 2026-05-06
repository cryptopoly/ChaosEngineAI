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
