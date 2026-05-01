"""Tests for Phase 2.0.5-B pre-flight memory gate.

The gate refuses chat generations when the host is already memory-starved
(low available RAM or high pressure). It must produce actionable refusals
without false-positive blocks during normal operation.
"""

import unittest

from backend_service.helpers.memory_gate import gate_chat_generation


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
        result = gate_chat_generation(available_gb=2.5, pressure_percent=95.0)
        self.assertIsNotNone(result)
        self.assertEqual(result["code"], "memory_gate_high_pressure")
        self.assertIn("95", result["message"])

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


if __name__ == "__main__":
    unittest.main()
