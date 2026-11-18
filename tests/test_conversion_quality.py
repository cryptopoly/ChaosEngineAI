"""Conversion results are measured: size ratio, bits per weight, perplexity of the converted model against the source.

The page used to show a "Quality" percentage and a "cache reduction" that came
from the KV-cache preview, not from the conversion.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend_service.inference import controller as controller_module
from backend_service.inference import conversion_quality as quality
from backend_service.inference.controller import RuntimeController
from backend_service.models import ConvertModelRequest
from backend_service.progress import CONVERSION_PROGRESS
from tests.test_conversion_helpers import _make_cache_repo

GB = 1024 ** 3


class QualityHelperTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()  # the controller resolves the output path
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "model.safetensors").write_bytes(b"\0" * 4000)
        self.output = self.root / "out-mlx"
        self.output.mkdir()
        (self.output / "model.safetensors").write_bytes(b"\0" * 1000)

    def test_both_models_in_free_memory_can_be_measured(self):
        self.assertIsNone(quality.quality_plan(str(self.source), str(self.output), 10_000))

    def test_a_source_that_is_not_a_folder_is_skipped_with_the_reason(self):
        for source in (None, "Qwen/Qwen3-0.6B", str(self.root / "gone")):
            plan = quality.quality_plan(source, str(self.output), 10 * GB)
            self.assertEqual(plan["status"], "skipped", source)
            self.assertIn("not a model folder", plan["reason"])

    def test_a_source_larger_than_the_free_memory_share_is_skipped(self):
        plan = quality.quality_plan(str(self.source), str(self.output), 5000)  # 4000 > 70 % of 5000
        self.assertEqual(plan["status"], "skipped")
        self.assertIn("memory is free", plan["reason"])
        self.assertIn("Perplexity benchmark", plan["reason"])

    def test_unknown_free_memory_is_skipped_not_guessed(self):
        self.assertEqual(quality.quality_plan(str(self.source), str(self.output), None)["status"], "skipped")

    def test_summary_reports_both_values_and_the_change(self):
        summary = quality.quality_summary(source_perplexity=12.5, converted_perplexity=13.25)
        self.assertEqual(summary["status"], "measured")
        self.assertEqual((summary["sourcePerplexity"], summary["convertedPerplexity"]), (12.5, 13.25))
        self.assertEqual(summary["changePercent"], 6.0)
        self.assertEqual((summary["dataset"], summary["samples"], summary["seqLength"]), ("wikitext-2", 8, 256))
        # A converted model that scores better is a negative change, not clamped to zero.
        self.assertLess(quality.quality_summary(12.5, 12.0)["changePercent"], 0)

    def test_the_child_output_is_read_from_its_last_json_line(self):
        self.assertEqual(quality.parse_measurement('Fetching 9 files\n{"perplexity": 13.29}\n'), 13.29)
        self.assertIsNone(quality.parse_measurement("Traceback (most recent call last):\nValueError: bad"))
        self.assertIsNone(quality.parse_measurement('{"perplexity": "nan"}'))
        self.assertIsNone(quality.parse_measurement(""))

    def test_the_child_refuses_anything_but_one_model_folder(self):
        self.assertEqual(quality.main([]), 2)
        self.assertEqual(quality.main(["a", "b"]), 2)

    def test_measure_quality_is_off_unless_asked_for(self):
        self.assertFalse(ConvertModelRequest().measureQuality)
        self.assertTrue(ConvertModelRequest(measureQuality=True).measureQuality)


class ConvertMeasuredResultTests(unittest.TestCase):
    """``convert_model`` with the converter and the scoring processes replaced."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()  # the controller resolves the output path
        self.controller = RuntimeController.__new__(RuntimeController)
        self.controller.capabilities = SimpleNamespace(
            converterAvailable=True, pythonExecutable="python", mlxLmVersion="0", mlxMessage=None,
        )
        self.controller.refresh_capabilities = lambda force=False: None  # type: ignore[method-assign]
        for name, value in (
            ("_peek_hf_model_type", lambda *a, **k: None),
            ("_free_disk_bytes", lambda _path: 10 ** 12),
        ):
            patcher = mock.patch.object(controller_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.repo = _make_cache_repo(self.root, weight_bytes=16_000)
        self.output = self.root / "Models" / "out-mlx"
        self.probed: list[str] = []

    def _convert(self, *, probe=None, stdout="[INFO] Quantized model with 4.503 bits per weight.", **kwargs):
        def fake_converter(command, _env):
            self.output.mkdir(parents=True)
            (self.output / "model.safetensors").write_bytes(b"\0" * 4_000)
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        def default_probe(path, _env):
            self.probed.append(path)
            value = 13.25 if path == str(self.output) else 12.5
            return subprocess.CompletedProcess([], 0, stdout=f'{{"perplexity": {value}}}\n', stderr="")

        with mock.patch.object(self.controller, "_run_converter", fake_converter), \
                mock.patch.object(self.controller, "_run_quality_probe", probe or default_probe), \
                mock.patch("psutil.virtual_memory", lambda: SimpleNamespace(available=64 * GB)):
            return self.controller.convert_model(
                source_ref="Qwen/Qwen3.5-4B", source_path=str(self.repo), output_path=str(self.output), hf_repo=None,
                quantize=True, q_bits=4, dtype="float16", **kwargs,
            )

    def test_size_ratio_and_bits_per_weight_are_measured_values(self):
        result = self._convert()
        self.assertEqual(result["bitsPerWeight"], 4.503)
        self.assertEqual(result["diskRatio"], round((16_000 + len('{"model_type": "qwen3", "torch_dtype": "bfloat16"}')) / 4_000, 2))
        self.assertIsNone(result["quality"])  # not asked for
        self.assertEqual(self.probed, [])
        self.assertNotIn("qualityPercent", result)
        self.assertNotIn("compressionRatio", result)

    def test_a_run_without_quantization_has_no_bits_per_weight(self):
        self.assertIsNone(self._convert(stdout="[INFO] Loading\n[INFO] Using dtype: float16")["bitsPerWeight"])

    def test_quality_is_the_perplexity_of_both_models_on_the_same_sample(self):
        result = self._convert(measure_quality=True)
        snapshot = str(self.repo / "snapshots" / "abc123")
        self.assertEqual(self.probed, [str(self.output), snapshot])  # the converted model first, then the source
        self.assertEqual(result["quality"]["status"], "measured")
        self.assertEqual(result["quality"]["sourcePerplexity"], 12.5)
        self.assertEqual(result["quality"]["convertedPerplexity"], 13.25)
        self.assertEqual(result["quality"]["changePercent"], 6.0)
        self.assertFalse(CONVERSION_PROGRESS.snapshot()["active"])

    def test_a_model_that_cannot_be_scored_keeps_the_conversion_and_says_why(self):
        def failing(path, _env):
            return subprocess.CompletedProcess(
                [], 1, stdout="", stderr="Traceback (most recent call last):\n  File x\nValueError: Model type xyz not supported.",
            )

        result = self._convert(measure_quality=True, probe=failing)
        self.assertEqual(result["quality"]["status"], "skipped")
        self.assertIn("converted model could not be scored", result["quality"]["reason"])
        self.assertIn("Model type xyz not supported", result["quality"]["reason"])
        self.assertNotIn("Traceback", result["quality"]["reason"])
        self.assertTrue((self.output / "model.safetensors").is_file())

    def test_cancel_during_the_measurement_keeps_the_converted_model(self):
        def cancelled(path, _env):
            self.controller._convert_cancel_requested = True
            return subprocess.CompletedProcess([], -15, stdout="", stderr="")

        result = self._convert(measure_quality=True, probe=cancelled)
        self.assertEqual(result["quality"]["status"], "skipped")
        self.assertIn("Stopped by Cancel", result["quality"]["reason"])
        self.assertIn("was kept", result["quality"]["reason"])
        self.assertTrue((self.output / "model.safetensors").is_file())

    def test_too_little_free_memory_skips_without_starting_a_process(self):
        calls: list[str] = []
        with mock.patch.object(quality, "_MEMORY_SHARE", 1e-12):
            result = self._convert(measure_quality=True, probe=lambda path, _env: calls.append(path))
        self.assertEqual(result["quality"]["status"], "skipped")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
