"""Tests for the Phase 2.0.5-I thermal classifier.

The classifier is a pure-function helper over `pmset -g therm` output, so
the tests fixture-load representative stdout strings and assert the
mapping into our three-state space (nominal / moderate / critical) plus
the None fallbacks for unparseable input.
"""

import unittest

from backend_service.helpers.thermal import _classify_pmset_output


class ClassifyPmsetOutputTests(unittest.TestCase):
    def test_nominal_when_no_warning_recorded(self):
        output = (
            "Note: No thermal warning level has been recorded\n"
            "Note: No performance warning level has been recorded\n"
        )
        self.assertEqual(_classify_pmset_output(output), "nominal")

    def test_nominal_for_zero_warning_level(self):
        output = "Thermal warning level set to 0.\n"
        self.assertEqual(_classify_pmset_output(output), "nominal")

    def test_moderate_for_low_warning_levels(self):
        for level in (1, 2):
            with self.subTest(level=level):
                output = f"Thermal warning level set to {level}.\n"
                self.assertEqual(_classify_pmset_output(output), "moderate")

    def test_critical_for_high_warning_levels(self):
        for level in (3, 5, 9):
            with self.subTest(level=level):
                output = f"Thermal warning level set to {level}.\n"
                self.assertEqual(_classify_pmset_output(output), "critical")

    def test_moderate_when_cpu_scheduler_limit_below_100(self):
        output = "CPU_Scheduler_Limit  = 80\n"
        self.assertEqual(_classify_pmset_output(output), "moderate")

    def test_nominal_when_cpu_scheduler_limit_at_100(self):
        output = "CPU_Scheduler_Limit  = 100\n"
        self.assertEqual(_classify_pmset_output(output), "nominal")

    def test_returns_none_for_empty_input(self):
        self.assertIsNone(_classify_pmset_output(""))

    def test_returns_none_for_unrelated_output(self):
        # Some other pmset subcommand stdout that doesn't include the
        # thermal-warning sentinel lines should yield None so the watcher
        # treats the data as unknown rather than misclassifying.
        self.assertIsNone(_classify_pmset_output("Battery: AC, charging.\n"))


if __name__ == "__main__":
    unittest.main()
