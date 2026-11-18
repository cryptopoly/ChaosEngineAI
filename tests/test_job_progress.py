"""Real progress for benchmarks and conversion: what the tracker and the converter watcher report."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from backend_service.inference.conversion_progress import ConverterWatch, parse_bits_per_weight, run_watched
from backend_service.progress import BENCHMARK_PROGRESS, CONVERSION_PROGRESS, JobProgressTracker
from backend_service.state import benchmarks as bench


class JobProgressTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = JobProgressTracker(kind="benchmark")

    def test_idle_snapshot_has_no_progress(self):
        snapshot = self.tracker.snapshot()
        self.assertEqual((snapshot["active"], snapshot["phase"], snapshot["totalSteps"], snapshot["detail"]),
                         (False, "idle", 0, {}))

    def test_a_new_phase_starts_uncounted(self):
        # A phase that cannot count must not inherit the total of the one before it:
        # the dialog would show a percentage for it.
        self.tracker.begin(run_label="task_accuracy", phase="dataset")
        self.tracker.set_phase("evaluating")
        self.tracker.set_step(40, 100)
        self.tracker.set_detail(correct=17)
        snapshot = self.tracker.snapshot()
        self.assertEqual((snapshot["step"], snapshot["totalSteps"], snapshot["detail"]), (40, 100, {"correct": 17}))
        self.tracker.set_phase("saving")
        snapshot = self.tracker.snapshot()
        self.assertEqual((snapshot["phase"], snapshot["step"], snapshot["totalSteps"], snapshot["detail"]),
                         ("saving", 0, 0, {}))

    def test_finish_clears_everything(self):
        self.tracker.begin(phase="evaluating")
        self.tracker.set_step(3, 9)
        self.tracker.set_detail(correct=1)
        self.tracker.finish()
        snapshot = self.tracker.snapshot()
        self.assertEqual((snapshot["active"], snapshot["totalSteps"], snapshot["detail"]), (False, 0, {}))

    def test_detail_is_ignored_when_nothing_runs(self):
        self.tracker.set_detail(bytesWritten=5)
        self.assertEqual(self.tracker.snapshot()["detail"], {})


class WorkerProgressTests(unittest.TestCase):
    def setUp(self):
        BENCHMARK_PROGRESS.begin(run_label="task_accuracy", phase="dataset")
        self.addCleanup(BENCHMARK_PROGRESS.finish)

    def test_worker_counts_land_on_the_tracker(self):
        bench._worker_progress({"phase": "evaluating", "message": "Question 0/50", "step": 0, "total": 50, "correct": 0})
        bench._worker_progress({"phase": "evaluating", "message": "Question 12/50, 5 correct", "step": 12, "total": 50, "correct": 5})
        snapshot = BENCHMARK_PROGRESS.snapshot()
        self.assertEqual(snapshot["phase"], "evaluating")
        self.assertEqual((snapshot["step"], snapshot["totalSteps"]), (12, 50))
        self.assertEqual(snapshot["detail"], {"correct": 5})
        self.assertEqual(snapshot["message"], "Question 12/50, 5 correct")

    def test_a_line_without_counts_changes_no_numbers(self):
        bench._worker_progress({"phase": "dataset", "message": "Loading the questions"})
        snapshot = BENCHMARK_PROGRESS.snapshot()
        self.assertEqual((snapshot["phase"], snapshot["step"], snapshot["totalSteps"]), ("dataset", 0, 0))


class ConverterWatchTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.output = Path(self.tempdir.name) / "out-mlx"
        self.tracker = JobProgressTracker(kind="conversion")
        self.tracker.begin(phase="preparing")
        self.watch = ConverterWatch(self.tracker, output_dir=str(self.output), expected_bytes=1000)

    def _phase(self):
        return self.tracker.snapshot()["phase"]

    def test_the_converters_own_lines_set_the_phase(self):
        self.assertEqual(self._phase(), "preparing")
        self.watch.feed_stdout("[INFO] Loading\n")
        self.assertEqual(self._phase(), "loading")
        self.watch.feed_stdout("[INFO] Using dtype: float16\n[INFO] Quan")
        self.assertEqual(self._phase(), "loading")
        self.watch.feed_stdout("tizing\n")  # a line split across two reads
        self.assertEqual(self._phase(), "quantizing")
        self.watch.feed_stdout("[INFO] Quantized model with 4.503 bits per weight.\n")
        snapshot = self.tracker.snapshot()
        self.assertEqual(snapshot["phase"], "saving")
        self.assertEqual(snapshot["detail"]["bitsPerWeight"], 4.503)
        self.assertEqual(snapshot["totalSteps"], 0)  # nothing to count: no percentage

    def test_download_counts_files_and_hands_over_to_loading(self):
        self.watch.feed_stdout("[INFO] Loading\n")
        self.watch.feed_stderr("Fetching 11 files:   0%|          | 0/11 [00:00<?, ?it/s]\r")
        self.watch.feed_stderr("Fetching 11 files:  45%|####5     | 5/11 [00:10<00:12,  2.00s/it]\r")
        snapshot = self.tracker.snapshot()
        self.assertEqual((snapshot["phase"], snapshot["step"], snapshot["totalSteps"]), ("downloading", 5, 11))
        self.watch.feed_stderr("Fetching 11 files: 100%|##########| 11/11 [00:20<00:00,  1.9s/it]\n")
        snapshot = self.tracker.snapshot()
        self.assertEqual((snapshot["phase"], snapshot["totalSteps"]), ("loading", 0))

    def test_a_local_source_that_only_lists_its_files_is_not_a_download(self):
        # Files already on disk go from nothing to N/N in one line.
        self.watch.feed_stdout("[INFO] Loading\n")
        self.watch.feed_stderr("Fetching 9 files: 100%|##########| 9/9 [00:00<00:00, 9000it/s]\n")
        self.assertEqual(self._phase(), "loading")

    def test_bytes_in_the_output_folder_are_measured(self):
        self.watch.tick()  # no folder yet
        self.assertEqual(self._phase(), "preparing")
        self.output.mkdir()
        self.watch.tick()  # empty folder
        self.assertEqual(self._phase(), "preparing")
        (self.output / "model.safetensors").write_bytes(b"x" * 300)
        self.watch.tick()
        snapshot = self.tracker.snapshot()
        self.assertEqual(snapshot["phase"], "saving")
        self.assertEqual(snapshot["detail"], {"bytesWritten": 300, "expectedBytes": 1000})
        (self.output / "model.safetensors").write_bytes(b"x" * 700)
        self.watch.tick()
        self.assertEqual(self.tracker.snapshot()["detail"]["bytesWritten"], 700)

    def test_no_output_folder_to_watch_measures_nothing(self):
        watch = ConverterWatch(self.tracker, output_dir=None, expected_bytes=None)
        watch.tick()
        self.assertEqual(self.tracker.snapshot()["detail"], {})

    def test_bits_per_weight_is_read_from_the_converter_output(self):
        self.assertEqual(parse_bits_per_weight("[INFO] Quantized model with 8.502 bits per weight."), 8.502)
        self.assertIsNone(parse_bits_per_weight("[INFO] Loading\n[INFO] Using dtype: float16"))
        self.assertIsNone(parse_bits_per_weight(""))

    def test_run_watched_reports_phases_from_a_real_process_and_returns_its_output(self):
        script = (
            "import sys, time, pathlib\n"
            "print('[INFO] Loading'); time.sleep(0.4)\n"
            "print('[INFO] Quantizing'); time.sleep(0.4)\n"
            f"out = pathlib.Path({str(self.output)!r}); out.mkdir()\n"
            "(out / 'model.safetensors').write_bytes(b'x' * 2048)\n"
            "time.sleep(0.8)\n"
            "sys.stderr.write('warn'); sys.exit(0)\n"
        )
        seen: list[str] = []
        original = self.tracker.set_phase
        self.tracker.set_phase = lambda phase, message="": (seen.append(phase), original(phase, message))[1]
        process = subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        stdout, stderr = run_watched(process, self.watch, timeout=30, poll_seconds=0.1)
        self.assertEqual(process.returncode, 0)
        self.assertIn("[INFO] Quantizing", stdout)
        self.assertEqual(stderr, "warn")
        self.assertEqual(seen, ["loading", "quantizing", "saving"])
        self.assertEqual(self.tracker.snapshot()["detail"]["bytesWritten"], 2048)

    def test_run_watched_ends_a_process_that_runs_past_the_limit(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with self.assertRaises(subprocess.TimeoutExpired):
            run_watched(process, self.watch, timeout=0.3, poll_seconds=0.1)
        self.assertIsNotNone(process.poll())


class ConversionTrackerSingletonTests(unittest.TestCase):
    def test_the_conversion_tracker_is_idle_between_runs(self):
        self.assertFalse(CONVERSION_PROGRESS.snapshot()["active"])


if __name__ == "__main__":
    unittest.main()
