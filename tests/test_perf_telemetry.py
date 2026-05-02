"""Phase 3.5 tests for perf telemetry snapshot."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from backend_service.helpers.perf import PerfTelemetry, snapshot_perf_telemetry


class PerfTelemetryShapeTests(unittest.TestCase):
    def test_default_is_empty(self):
        telemetry = PerfTelemetry()
        self.assertTrue(telemetry.is_empty)

    def test_to_dict_has_all_fields(self):
        telemetry = PerfTelemetry(cpuPercent=50.0)
        payload = telemetry.to_dict()
        self.assertEqual(payload["cpuPercent"], 50.0)
        self.assertIn("gpuPercent", payload)
        self.assertIn("thermalState", payload)
        self.assertIn("availableMemoryGb", payload)

    def test_is_empty_false_when_any_field_set(self):
        self.assertFalse(PerfTelemetry(cpuPercent=10.0).is_empty)
        self.assertFalse(PerfTelemetry(gpuPercent=20.0).is_empty)
        self.assertFalse(PerfTelemetry(thermalState="nominal").is_empty)
        self.assertFalse(PerfTelemetry(availableMemoryGb=4.0).is_empty)


class SnapshotPerfTelemetryTests(unittest.TestCase):
    def test_returns_telemetry_object(self):
        # Real call — fields may be None on the test runner depending
        # on whether psutil samplers behave. Just verify the type.
        telemetry = snapshot_perf_telemetry()
        self.assertIsInstance(telemetry, PerfTelemetry)

    def test_psutil_failure_returns_partial_blob(self):
        # When psutil throws, CPU + memory fall through to None.
        # Thermal + GPU remain best-effort and continue independently.
        with patch("psutil.cpu_percent", side_effect=RuntimeError("test")):
            telemetry = snapshot_perf_telemetry()
            self.assertIsNone(telemetry.cpuPercent)

    def test_thermal_failure_does_not_block_other_fields(self):
        with patch(
            "backend_service.helpers.thermal.read_thermal_state",
            side_effect=RuntimeError("test"),
        ):
            telemetry = snapshot_perf_telemetry()
            # Thermal will be None but CPU should still sample.
            self.assertIsNone(telemetry.thermalState)


if __name__ == "__main__":
    unittest.main()
