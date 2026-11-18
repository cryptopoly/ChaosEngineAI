"""MLX conversion: source folder, output name, free-space estimate, failure text, command line."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pydantic import ValidationError

from backend_service.inference import controller as controller_module
from backend_service.inference.controller import RuntimeController
from backend_service.inference.conversion import (
    _conversion_source_name,
    _default_conversion_output,
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
        # "Qwen3-0.6B-GGUF-mlx" could not be loaded: the loader reads "gguf" in a path as a GGUF model.
        self.assertEqual(_conversion_source_name("/x/hub/models--lmstudio-community--Qwen3-0.6B-GGUF"), "Qwen3-0.6B")
        self.assertEqual(_conversion_source_name("org/Model_gguf"), "Model")
        self.assertNotIn("gguf", _conversion_source_name("/x/Qwen3-0.6B-GGUF/qwen3-0.6b-q4_k_m.gguf").lower())

    def test_default_output_folder_keeps_dots_and_stays_inside_models(self):
        with mock.patch.object(Path, "home", return_value=self.root):
            target = Path(_default_conversion_output("Qwen3.5-4B"))
            self.assertEqual(target, self.root / "Models" / "Qwen3.5-4B-mlx")
            target.mkdir()
            self.assertEqual(Path(_default_conversion_output("Qwen3.5-4B")).name, "Qwen3.5-4B-mlx-2")
            for hostile in ("../../etc", "..", "a/b\\c", ""):
                picked = Path(_default_conversion_output(hostile))
                self.assertEqual(picked.parent, self.root / "Models", hostile)
                self.assertNotIn("..", picked.name.replace("...", ""), hostile)

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
        with mock.patch.object(self.controller, "_run_converter", lambda command, _env: fake_run(command)), \
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


class ConversionCancelTests(unittest.TestCase):
    """The converter runs as a process that ``cancel_conversion`` can stop."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.controller = RuntimeController.__new__(RuntimeController)

    def test_nothing_running_means_nothing_to_cancel(self):
        self.assertFalse(self.controller.cancel_conversion())

    def test_converter_output_and_exit_code_are_returned(self):
        done = self.controller._run_converter(
            [sys.executable, "-c", "import sys; print('out'); sys.stderr.write('err'); sys.exit(3)"], dict(os.environ),
        )
        self.assertEqual((done.returncode, done.stdout.strip(), done.stderr), (3, "out", "err"))
        self.assertIsNone(self.controller._convert_process)

    def test_cancel_stops_the_running_converter(self):
        marker = self.root / "started"
        script = f"import pathlib, time; pathlib.Path({str(marker)!r}).write_text('1'); time.sleep(60)"
        result: dict[str, subprocess.CompletedProcess] = {}
        worker = threading.Thread(
            target=lambda: result.update(done=self.controller._run_converter([sys.executable, "-c", script], dict(os.environ))),
        )
        worker.start()
        deadline = time.monotonic() + 20
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(marker.exists(), "the converter process did not start")
        self.controller._convert_active = True  # set by convert_model for the length of a run
        self.assertTrue(self.controller.cancel_conversion())
        worker.join(timeout=20)
        self.assertFalse(worker.is_alive())
        self.assertNotEqual(result["done"].returncode, 0)
        self.assertTrue(self.controller._convert_cancel_requested)
        self.controller._convert_active = False
        self.assertFalse(self.controller.cancel_conversion())

    def test_cancel_during_the_pre_flight_stops_the_run_before_the_converter_starts(self):
        self.controller.capabilities = SimpleNamespace(
            converterAvailable=True, pythonExecutable="python", mlxLmVersion="0", mlxMessage=None,
        )
        self.controller.refresh_capabilities = lambda force=False: None  # type: ignore[method-assign]
        started: list[list[str]] = []

        def cancel_while_checking(*_args, **_kwargs):
            # The model-type peek is one of the pre-flight steps; the user cancels during it.
            self.assertTrue(self.controller.cancel_conversion())
            return None

        with mock.patch.object(controller_module, "_peek_hf_model_type", cancel_while_checking), \
                mock.patch.object(self.controller, "_run_converter", lambda command, _env: started.append(command)):
            with self.assertRaises(RuntimeError) as ctx:
                self.controller.convert_model(
                    source_ref="org/model", source_path=None, output_path=str(self.root / "out-mlx"), hf_repo=None,
                    quantize=True, q_bits=4, dtype="float16",
                )
        self.assertIn("Conversion cancelled", str(ctx.exception))
        self.assertEqual(started, [])
        self.assertFalse(self.controller.cancel_conversion())

    def test_cancelled_run_says_cancelled_and_keeps_nothing(self):
        target = self.root / "out-mlx"
        self.controller.capabilities = SimpleNamespace(
            converterAvailable=True, pythonExecutable="python", mlxLmVersion="0", mlxMessage=None,
        )
        self.controller.refresh_capabilities = lambda force=False: None  # type: ignore[method-assign]

        def cancelled_run(command, _env):
            target.mkdir()
            (target / "model.safetensors").write_bytes(b"\0" * 10)
            self.assertTrue(self.controller.cancel_conversion())
            return subprocess.CompletedProcess(command, -15, "", "Fetching 10 files:  80%")

        with mock.patch.object(controller_module, "_peek_hf_model_type", lambda *a, **k: None), \
                mock.patch.object(self.controller, "_run_converter", cancelled_run):
            with self.assertRaises(RuntimeError) as ctx:
                self.controller.convert_model(
                    source_ref="org/model", source_path=None, output_path=str(target), hf_repo=None,
                    quantize=True, q_bits=4, dtype="float16",
                )
        self.assertIn("Conversion cancelled", str(ctx.exception))
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
