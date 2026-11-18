"""MLX conversion: source folder, output name, free-space estimate, failure text, command line."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pydantic import ValidationError

from backend_service.inference import controller as controller_module
from backend_service.inference.controller import RuntimeController
from backend_service.inference.conversion import (
    _conversion_source_name,
    _estimate_conversion_output_bytes,
    _free_disk_bytes,
    _resolve_local_model_dir,
    _summarise_convert_failure,
)
from backend_service.models import ConvertModelRequest


def _make_cache_repo(root: Path, name: str = "models--Qwen--Qwen3.5-4B", *, dtype: str = "bfloat16",
                     weight_bytes: int = 16_000) -> Path:
    """A Hugging Face cache folder: files live under snapshots/<revision>/."""
    repo = root / name
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3", "torch_dtype": dtype}))
    (snapshot / "model.safetensors").write_bytes(b"\0" * weight_bytes)
    return repo


class ConversionHelperTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_cache_folder_resolves_to_its_snapshot(self):
        repo = _make_cache_repo(self.root)
        self.assertEqual(_resolve_local_model_dir(str(repo)), str(repo / "snapshots" / "abc123"))

    def test_plain_model_folder_and_unknown_paths_come_back_unchanged(self):
        plain = self.root / "plain"
        plain.mkdir()
        (plain / "config.json").write_text("{}")
        self.assertEqual(_resolve_local_model_dir(str(plain)), str(plain))
        empty = self.root / "empty"
        empty.mkdir()
        self.assertEqual(_resolve_local_model_dir(str(empty)), str(empty))
        self.assertEqual(_resolve_local_model_dir(str(self.root / "missing")), str(self.root / "missing"))

    def test_output_name_keeps_the_model_name(self):
        # Path.stem gave "models--Qwen--Qwen3", which the library then listed as the
        # repo "Qwen/Qwen3-mlx" and could not load.
        self.assertEqual(_conversion_source_name("/x/hub/models--Qwen--Qwen3.5-4B"), "Qwen3.5-4B")
        self.assertEqual(_conversion_source_name("Qwen/Qwen3.5-4B"), "Qwen3.5-4B")
        self.assertEqual(_conversion_source_name("/x/Qwen3-0.6B-Q4_K_M.gguf"), "Qwen3-0.6B-Q4_K_M")
        self.assertEqual(_conversion_source_name(""), "model")

    def test_output_estimate_follows_bits_and_source_width(self):
        repo = _make_cache_repo(self.root, weight_bytes=16_000)
        snapshot = _resolve_local_model_dir(str(repo))
        source = sum(f.stat().st_size for f in Path(snapshot).iterdir())
        four = _estimate_conversion_output_bytes(snapshot, quantize=True, q_bits=4, dtype="float16")
        eight = _estimate_conversion_output_bytes(snapshot, quantize=True, q_bits=8, dtype="float16")
        plain = _estimate_conversion_output_bytes(snapshot, quantize=False, q_bits=4, dtype="float16")
        self.assertEqual(four, int(source * 4.5 / 16))
        self.assertEqual(eight, int(source * 8.5 / 16))
        self.assertEqual(plain, source)
        self.assertIsNone(_estimate_conversion_output_bytes(None, quantize=True, q_bits=4, dtype="float16"))

    def test_source_of_unknown_width_gives_the_smaller_estimate(self):
        repo = _make_cache_repo(self.root, dtype="")
        snapshot = _resolve_local_model_dir(str(repo))
        source = sum(f.stat().st_size for f in Path(snapshot).iterdir())
        self.assertEqual(
            _estimate_conversion_output_bytes(snapshot, quantize=True, q_bits=4, dtype="float16"),
            int(source * 4.5 / 32),
        )

    def test_free_space_is_read_from_the_nearest_existing_parent(self):
        free = _free_disk_bytes(str(self.root / "not" / "made" / "yet"))
        self.assertIsInstance(free, int)
        self.assertGreater(free, 0)

    def test_failure_text_is_the_error_line_not_the_traceback(self):
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "x.py", line 1, in <module>\n'
            "    save()\n"
            "RuntimeError: [write] Unable to write 11796480 bytes to file.\n"
        )
        self.assertEqual(
            _summarise_convert_failure(stderr, "", 1),
            "RuntimeError: [write] Unable to write 11796480 bytes to file.",
        )
        self.assertIn("stopped", _summarise_convert_failure("Fetching 10 files:  80%", "", -15))
        self.assertEqual(_summarise_convert_failure("", "", 1), "mlx_lm.convert failed.")
        self.assertEqual(_summarise_convert_failure("usage: bad flag", "", 2), "usage: bad flag")

    def test_dtype_outside_the_list_is_refused_at_the_boundary(self):
        self.assertEqual(ConvertModelRequest(dtype="bfloat16").dtype, "bfloat16")
        with self.assertRaises(ValidationError):
            ConvertModelRequest(dtype="floaty")


class ConvertModelCommandTests(unittest.TestCase):
    """``RuntimeController.convert_model`` with the converter process replaced."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.controller = RuntimeController.__new__(RuntimeController)
        self.controller.capabilities = SimpleNamespace(
            converterAvailable=True, pythonExecutable="python", mlxLmVersion="0", mlxMessage=None,
        )
        self.controller.refresh_capabilities = lambda force=False: None  # type: ignore[method-assign]
        for name, value in (
            ("_peek_hf_model_type", lambda *a, **k: None),
            ("_default_conversion_output", lambda label: str(self.root / "Models" / f"{label}-mlx")),
        ):
            patcher = mock.patch.object(controller_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run(self, completed=None, *, side_effect=None, free=10**12, **kwargs):
        calls: list[list[str]] = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            if side_effect is not None:
                side_effect(command)
            return completed or subprocess.CompletedProcess(command, 0, stdout="[INFO] Quantizing", stderr="")

        params = dict(source_ref="Qwen/Qwen3.5-4B", source_path=None, output_path=None, hf_repo=None,
                      quantize=True, q_bits=4, dtype="float16")
        params.update(kwargs)
        with mock.patch.object(controller_module.subprocess, "run", fake_run), \
                mock.patch.object(controller_module, "_free_disk_bytes", lambda _path: free):
            try:
                return self.controller.convert_model(**params), calls
            except RuntimeError as exc:
                return exc, calls

    def test_cache_folder_source_is_converted_from_its_snapshot(self):
        repo = _make_cache_repo(self.root)
        result, calls = self._run(source_path=str(repo))
        self.assertIsInstance(result, dict)
        command = calls[0]
        self.assertEqual(command[command.index("--hf-path") + 1], str(repo / "snapshots" / "abc123"))
        self.assertEqual(Path(result["outputPath"]).name, "Qwen3.5-4B-mlx")
        self.assertEqual(result["sourceLabel"], "Qwen3.5-4B")

    def test_missing_source_path_is_named_not_reported_as_gated(self):
        result, calls = self._run(source_path=str(self.root / "gone"))
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("Source path not found", str(result))
        self.assertEqual(calls, [])

    def test_too_little_free_space_stops_before_the_converter_runs(self):
        repo = _make_cache_repo(self.root, weight_bytes=64_000)
        result, calls = self._run(source_path=str(repo), free=1_000)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("Not enough free disk space", str(result))
        self.assertEqual(calls, [])

    def test_failed_run_removes_the_folder_it_created_and_says_disk_full(self):
        target = self.root / "out-mlx"

        def write_partial(_command):
            target.mkdir()
            (target / "model.safetensors").write_bytes(b"\0" * 10)

        failed = subprocess.CompletedProcess(
            [], 1, stdout="",
            stderr="Traceback (most recent call last):\n  x\nRuntimeError: [write] Unable to write 11796480 bytes to file.\n",
        )
        result, _calls = self._run(failed, side_effect=write_partial, output_path=str(target))
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("Ran out of disk space", str(result))
        self.assertFalse(target.exists())

    def test_failed_run_never_removes_a_folder_that_was_already_there(self):
        target = self.root / "existing-mlx"
        target.mkdir()
        (target / "keep.txt").write_text("mine")
        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="ValueError: Cannot save to the path, already exists")
        result, _calls = self._run(failed, output_path=str(target))
        self.assertIn("Output path already exists", str(result))
        self.assertTrue((target / "keep.txt").exists())

    def test_repo_that_does_not_exist_is_not_called_gated(self):
        failed = subprocess.CompletedProcess(
            [], 1, stdout="",
            stderr=(
                "huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error.\n"
                "Repository Not Found for url: https://huggingface.co/api/models/nobody/nothing.\n"
                "If you are trying to access a private or gated repo, make sure you are authenticated.\n"
            ),
        )
        result, _calls = self._run(failed, source_ref="nobody/nothing")
        self.assertIn("repository not found: nobody/nothing", str(result))
        self.assertNotIn("This model is gated", str(result))

    def test_stopped_converter_reports_a_stop_not_progress_bar_text(self):
        failed = subprocess.CompletedProcess([], -15, stdout="", stderr="Fetching 10 files:  80%|####")
        result, _calls = self._run(failed, source_ref="org/model")
        self.assertIn("stopped before it finished", str(result))
        self.assertNotIn("Fetching", str(result))


if __name__ == "__main__":
    unittest.main()
