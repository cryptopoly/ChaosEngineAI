"""Benchmark cancel: the run stops, nothing is saved, the model the run loaded is unloaded.

Before this, Cancel only stopped the client from waiting: the backend finished
the run, saved it and kept the model loaded while the UI said "cancelled".
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.mlx_worker_eval import CANCELLED_MESSAGE, raise_if_cancelled
from backend_service.models import BenchmarkRunRequest
from backend_service.progress import BENCHMARK_PROGRESS
from backend_service.state import ChaosEngineState
from backend_service.state import benchmarks as bench
from tests.test_backend_service import FakeRuntime
from tests.test_setup_routes import TEST_API_TOKEN, _fake_system_snapshot


class _Engine:
    """An MLX-like engine whose eval loops stop when the cancel file appears."""

    engine_name = "mlx"
    engine_label = "MLX"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.progress: list[dict] = []
        # A worker stuck inside one item (a dataset download, a very long batch) never looks at the cancel file.
        self.ignore_cancel_file = False
        self.process_ended = threading.Event()

    def _loop(self, progress_callback, cancel_file, total):
        self.started.set()
        for step in range(1, total + 1):
            if self.process_ended.is_set():
                raise RuntimeError("MLX worker exited unexpectedly: worker exited with code -9")
            if not self.ignore_cancel_file and cancel_file and os.path.exists(cancel_file):
                raise RuntimeError(CANCELLED_MESSAGE)
            if progress_callback:
                progress_callback({"phase": "evaluating", "step": step, "total": total, "correct": step // 2})
            time.sleep(0.01)

    def eval_task_accuracy(self, *, task_name, limit, num_shots, progress_callback=None, cancel_file=None):
        self._loop(progress_callback, cancel_file, limit)
        return {
            "taskName": task_name, "accuracy": 0.5, "correct": limit // 2, "total": limit,
            "numShots": num_shots, "evalSeconds": 0.1, "scoringMethod": "next_token",
        }

    def eval_perplexity(self, *, dataset, num_samples, seq_length, batch_size, progress_callback=None, cancel_file=None):
        self._loop(progress_callback, cancel_file, num_samples)
        return {
            "perplexity": 12.0, "standardError": 0.2, "evalSeconds": 0.1, "evalTokensPerSecond": 100.0,
            "numSamples": num_samples, "seqLength": seq_length, "dataset": dataset,
        }


class _Runtime(FakeRuntime):
    """Fake runtime whose load and generate block until the model is unloaded, like a real engine process."""

    def __init__(self) -> None:
        super().__init__()
        self.engine = _Engine()
        self.block_load = False
        self.block_generate = False
        self.in_load = threading.Event()
        self.in_generate = threading.Event()
        self.stopped = threading.Event()
        self.unload_calls = 0

    def load_model(self, **kwargs):
        if self.block_load:
            self.in_load.set()
            self.stopped.wait(10)
            raise RuntimeError("The model process ended while loading.")
        return super().load_model(**kwargs)

    def generate(self, **kwargs):
        if self.block_generate:
            self.in_generate.set()
            self.stopped.wait(10)
            raise RuntimeError("The model process ended while generating.")
        return super().generate(**kwargs)

    def unload_model(self) -> None:
        self.unload_calls += 1
        self.stopped.set()
        self.engine.process_ended.set()
        super().unload_model()


class BenchmarkCancelTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        root = Path(self.tempdir.name)
        self.state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            library_provider=lambda: [],
            settings_path=root / "settings.json",
            benchmarks_path=root / "benchmarks.json",
            chat_sessions_path=root / "chats.json",
            library_cache_path=root / "library_cache.json",
        )
        self.runtime = _Runtime()
        self.state.runtime = self.runtime
        self.result: dict = {}
        # A benchmark refuses a model path that is not on disk, so the fake model has a real folder.
        self.model_path = str(root / "model")
        Path(self.model_path).mkdir()

    def _start(self, **fields):
        request = BenchmarkRunRequest(modelRef="org/model", modelName="Model", path=self.model_path, **fields)

        def target():
            try:
                self.result["value"] = self.state.run_benchmark(request)
            except Exception as exc:  # noqa: BLE001 - asserted by the test
                self.result["error"] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread

    def _preload(self):
        self.runtime.load_model(model_ref="org/model", model_name="Model", path=self.model_path)
        self.runtime.unload_calls = 0

    def test_cancel_with_nothing_running_answers_false(self):
        self.assertFalse(bench.cancel_benchmark(self.state))
        self.assertEqual(self.runtime.unload_calls, 0)

    def test_run_that_finishes_is_saved_and_reports_no_cancel(self):
        self._preload()
        response = self.state.run_benchmark(
            BenchmarkRunRequest(modelRef="org/model", path=self.model_path, mode="task_accuracy", taskLimit=10)
        )
        self.assertNotIn("cancelled", response)
        self.assertEqual(len(self.state.benchmark_runs), 1)
        self.assertFalse(BENCHMARK_PROGRESS.snapshot()["active"])
        # The worker's counts reached the tracker while the run was going.
        self.assertEqual(self.runtime.engine.started.is_set(), True)

    def test_cancel_between_questions_keeps_a_model_that_was_already_loaded(self):
        self._preload()
        thread = self._start(mode="task_accuracy", taskLimit=5000)
        self.assertTrue(self.runtime.engine.started.wait(5))
        self.assertTrue(bench.cancel_benchmark(self.state))
        thread.join(10)
        self.assertFalse(thread.is_alive())
        response = self.result["value"]
        self.assertTrue(response["cancelled"])
        self.assertFalse(response["modelUnloaded"])
        self.assertIsNone(response["result"])
        self.assertIn("Nothing was saved", response["message"])
        self.assertIn("stays loaded", response["message"])
        self.assertEqual(self.state.benchmark_runs, [])
        self.assertEqual(self.runtime.unload_calls, 0)
        self.assertIsNotNone(self.runtime.loaded_model)
        self.assertFalse(BENCHMARK_PROGRESS.snapshot()["active"])

    def test_a_worker_that_does_not_stop_between_items_has_its_process_ended(self):
        self._preload()
        self.runtime.engine.ignore_cancel_file = True
        thread = self._start(mode="task_accuracy", taskLimit=5000)
        self.assertTrue(self.runtime.engine.started.wait(5))
        with mock.patch.object(bench, "_COOPERATIVE_CANCEL_SECONDS", 0.2):
            self.assertTrue(bench.cancel_benchmark(self.state))
        thread.join(10)
        self.assertFalse(thread.is_alive())
        response = self.result["value"]
        self.assertTrue(response["cancelled"])
        self.assertTrue(response["modelUnloaded"])  # the price of a stop the worker did not take
        self.assertGreaterEqual(self.runtime.unload_calls, 1)
        self.assertEqual(self.state.benchmark_runs, [])

    def test_cancel_mid_eval_unloads_the_model_the_run_loaded(self):
        thread = self._start(mode="perplexity", perplexityNumSamples=1024)
        self.assertTrue(self.runtime.engine.started.wait(5))
        self.assertTrue(bench.cancel_benchmark(self.state))
        thread.join(10)
        response = self.result["value"]
        self.assertTrue(response["cancelled"])
        self.assertTrue(response["modelUnloaded"])
        self.assertIn("was unloaded", response["message"])
        self.assertEqual(self.state.benchmark_runs, [])
        self.assertIsNone(self.runtime.loaded_model)
        self.assertIsNone(response["runtime"]["loadedModel"])

    def test_cancel_during_the_model_load_ends_the_load(self):
        self.runtime.block_load = True
        thread = self._start(mode="throughput")
        self.assertTrue(self.runtime.in_load.wait(5))
        self.assertEqual(BENCHMARK_PROGRESS.snapshot()["phase"], "loading")
        self.assertTrue(bench.cancel_benchmark(self.state))
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertTrue(self.result["value"]["cancelled"])
        self.assertTrue(self.result["value"]["modelUnloaded"])
        self.assertGreaterEqual(self.runtime.unload_calls, 1)
        self.assertEqual(self.state.benchmark_runs, [])

    def test_cancel_during_generation_ends_the_model_process_even_if_it_was_loaded_before(self):
        # A generation in flight cannot be interrupted from outside, so the engine is stopped.
        self._preload()
        self.runtime.block_generate = True
        thread = self._start(mode="throughput")
        self.assertTrue(self.runtime.in_generate.wait(5))
        self.assertEqual(BENCHMARK_PROGRESS.snapshot()["phase"], "generating")
        self.assertTrue(bench.cancel_benchmark(self.state))
        thread.join(10)
        response = self.result["value"]
        self.assertTrue(response["cancelled"])
        self.assertTrue(response["modelUnloaded"])
        self.assertIsNone(self.runtime.loaded_model)
        self.assertEqual(self.state.benchmark_runs, [])

    def test_a_model_path_that_is_not_on_disk_is_named_before_anything_loads(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.state.run_benchmark(BenchmarkRunRequest(modelRef="org/gone", path="/nonexistent/e2e-model"))
        self.assertIn("Model files not found at /nonexistent/e2e-model", str(ctx.exception))
        self.assertNotIn("Repo id", str(ctx.exception))
        self.assertEqual(self.runtime.load_requests, [])
        self.assertFalse(BENCHMARK_PROGRESS.snapshot()["active"])

    def test_a_real_failure_is_still_an_error_not_a_cancel(self):
        self._preload()
        self.runtime.engine.eval_perplexity = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("dataset missing"))
        with self.assertRaises(RuntimeError) as ctx:
            self.state.run_benchmark(BenchmarkRunRequest(modelRef="org/model", path=self.model_path, mode="perplexity"))
        self.assertIn("dataset missing", str(ctx.exception))
        self.assertFalse(BENCHMARK_PROGRESS.snapshot()["active"])

    def test_a_second_run_is_refused_while_one_is_running(self):
        self._preload()
        thread = self._start(mode="task_accuracy", taskLimit=5000)
        self.assertTrue(self.runtime.engine.started.wait(5))
        try:
            with self.assertRaises(bench.BenchmarkBusy):
                self.state.run_benchmark(BenchmarkRunRequest(modelRef="org/model", path=self.model_path))
        finally:
            bench.cancel_benchmark(self.state)
            thread.join(10)

    def test_the_cancel_file_is_removed_after_the_run(self):
        self._preload()
        seen: list[str] = []
        original = self.runtime.engine.eval_task_accuracy

        def spy(**kwargs):
            seen.append(kwargs["cancel_file"])
            return original(**kwargs)

        self.runtime.engine.eval_task_accuracy = spy
        self.state.run_benchmark(
            BenchmarkRunRequest(modelRef="org/model", path=self.model_path, mode="task_accuracy", taskLimit=10)
        )
        self.assertEqual(len(seen), 1)
        self.assertFalse(os.path.exists(seen[0]))

    def test_routes_cancel_is_false_when_idle_and_a_busy_run_is_409(self):
        client = TestClient(create_app(state=self.state, api_token=TEST_API_TOKEN))
        client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})
        idle = client.post("/api/benchmarks/cancel")
        self.assertEqual(idle.status_code, 200)
        self.assertEqual(idle.json(), {"cancelled": False})

        self._preload()
        thread = self._start(mode="task_accuracy", taskLimit=5000)
        self.assertTrue(self.runtime.engine.started.wait(5))
        try:
            busy = client.post("/api/benchmarks/run", json={"modelRef": "org/model", "path": self.model_path})
            self.assertEqual(busy.status_code, 409)
            cancelled = client.post("/api/benchmarks/cancel")
            self.assertEqual(cancelled.json(), {"cancelled": True})
        finally:
            thread.join(10)
        self.assertTrue(self.result["value"]["cancelled"])


class WorkerCancelFileTests(unittest.TestCase):
    def test_no_file_named_or_no_file_on_disk_goes_on(self):
        raise_if_cancelled({})
        raise_if_cancelled({"cancelFile": None})
        with tempfile.TemporaryDirectory() as folder:
            raise_if_cancelled({"cancelFile": str(Path(folder) / "cancel")})

    def test_the_file_on_disk_stops_the_loop(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "cancel"
            marker.touch()
            with self.assertRaises(RuntimeError) as ctx:
                raise_if_cancelled({"cancelFile": str(marker)})
            self.assertEqual(str(ctx.exception), CANCELLED_MESSAGE)


if __name__ == "__main__":
    unittest.main()
