"""Unit tests for ``scripts/calibrate-cache-speed.py`` (pure functions + the
metrics read; the HTTP layer is exercised by running the script on a Mac)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


def _load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "calibrate-cache-speed.py"
    spec = importlib.util.spec_from_file_location("calibrate_cache_speed", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["calibrate_cache_speed"] = module
    spec.loader.exec_module(module)
    return module


calibrate = _load_script()


class PromptTests(unittest.TestCase):
    def test_longer_target_gives_longer_prompt(self) -> None:
        self.assertLess(len(calibrate.build_prompt(512)), len(calibrate.build_prompt(4096)))

    def test_prompt_is_deterministic_and_never_empty(self) -> None:
        self.assertEqual(calibrate.build_prompt(1024), calibrate.build_prompt(1024))
        self.assertTrue(calibrate.build_prompt(0).strip())


class RatioTests(unittest.TestCase):
    def test_ratio_is_compressed_over_native_per_context(self) -> None:
        ratios = calibrate.speed_ratios(
            {512: 30.0, 4096: 20.0},
            {3: {512: 6.0, 4096: 5.0}},
        )
        self.assertEqual(ratios, {3: {512: 0.2, 4096: 0.25}})

    def test_contexts_without_a_usable_native_speed_are_skipped(self) -> None:
        ratios = calibrate.speed_ratios(
            {512: 30.0, 4096: 0.0},
            {3: {512: 6.0, 4096: 5.0, 16384: 4.0}, 4: {4096: 5.0}},
        )
        self.assertEqual(ratios, {3: {512: 0.2}})

    def test_failed_compressed_run_is_skipped(self) -> None:
        self.assertEqual(calibrate.speed_ratios({512: 30.0}, {3: {512: 0.0}}), {})

    def test_suggested_map_is_the_median_across_contexts(self) -> None:
        suggested = calibrate.suggest_map({3: {512: 0.2, 4096: 0.3, 16384: 0.4}, 2: {512: 0.18}})
        self.assertEqual(suggested, {2: 0.18, 3: 0.3})

    def test_map_line_is_ready_to_paste(self) -> None:
        self.assertEqual(
            calibrate.format_map_line("turboquant", {2: 0.18, 3: 0.2}),
            '"turboquant": {2: 0.18, 3: 0.2},',
        )


class MeasureTests(unittest.TestCase):
    def test_reads_decode_speed_from_the_done_event_metrics(self) -> None:
        helpers = mock.Mock()
        helpers._stream_inference.side_effect = [
            ("text", {"assistant": {"metrics": {"tokS": 25.5}}}),
            ("text", {"assistant": {"metrics": {}}}),
        ]
        with mock.patch("builtins.print"):
            results = calibrate._measure(helpers, 8876, [512, 4096], 128)
        self.assertEqual(results, {512: 25.5, 4096: 0.0})
        body = helpers._stream_inference.call_args_list[0].kwargs["body"]
        self.assertEqual(body["maxTokens"], 128)
        self.assertEqual(body["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
