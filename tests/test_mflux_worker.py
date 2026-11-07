"""Warm mflux worker (I1) and the shared keep-warm client (C2), with a stub engine instead of mflux.

The real ``backend_service/mflux_worker.py`` runs under this interpreter with
``CHAOSENGINE_WARM_ENGINE_MODULE`` pointing at a fake engine, so the protocol, cancel, crash fallback
and policy are tested end to end without loading a model.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_service import warm_worker_client as wwc
from backend_service.image_runtime import mflux_engine
from backend_service.image_runtime.mflux_engine import MfluxImageEngine, WarmFallback
from backend_service.image_runtime.types import ImageGenerationConfig
from backend_service.progress import IMAGE_PROGRESS, GenerationCancelled

ROOT = Path(__file__).resolve().parents[1]

FAKE_ENGINE = textwrap.dedent('''
    import os, time

    class Engine:
        def __init__(self):
            self.loads = 0
            self.model = None

        def loaded(self):
            return {"model": self.model}

        def unload(self):
            self.model = None

        def generate(self, request, hooks):
            argv = request["argv"]
            prompt = argv[argv.index("--prompt") + 1]
            output = argv[argv.index("--output") + 1]
            model = argv[argv.index("--model") + 1]
            warm = self.model == model
            if not warm:
                hooks.stage("load")
                self.loads += 1
                self.model = model
            if prompt == "crash":
                os._exit(3)
            if prompt == "fail":
                raise RuntimeError("engine said no")
            hooks.stage("encode")
            steps = 40 if prompt in ("slow", "stubborn") else 2
            for step in range(1, steps + 1):
                if prompt != "stubborn":
                    hooks.check_cancel()
                hooks.step(step, steps)
                time.sleep(0.05)
            hooks.stage("decode")
            with open(output, "wb") as handle:
                handle.write(b"PNG:" + prompt.encode() + b":" + argv[argv.index("--seed") + 1].encode())
            return {"output": output, "warm": warm, "peak_mlx_gb": 1.5,
                    "timings": {"loadSeconds": 0.0, "encodeSeconds": 0.0, "denoiseSeconds": 0.1,
                                "decodeSeconds": 0.0, "totalSeconds": 0.1, "warm": warm}, "loads": self.loads}

    def create_engine():
        return Engine()
''')


def _config(prompt: str = "a cat", **overrides) -> ImageGenerationConfig:
    values = dict(modelId="iris", modelName="Iris", repo="speridlabs/iris-3b", prompt=prompt, negativePrompt="",
                  width=512, height=512, steps=4, guidance=1.0, batchSize=1, seed=7, runtime="mflux")
    values.update(overrides)
    return ImageGenerationConfig(**values)


class _WorkerCase(unittest.TestCase):
    script = ROOT / "backend_service" / "mflux_worker.py"
    kind = "image"
    fake_source = FAKE_ENGINE

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="warm-test-"))
        self.fake = self.tmp / "fake_engine.py"
        self.fake.write_text(self.fake_source, encoding="utf-8")
        env = {**os.environ, "CHAOSENGINE_WARM_ENGINE_MODULE": str(self.fake), "PYTHONPATH": ""}
        self.worker = wwc.WarmWorker(
            name="test",
            kind=self.kind,
            command=lambda: [sys.executable, str(self.script)],
            env=lambda: env,
            log_path=self.tmp / "worker.log",
            hard_cancel_after_s=1.0,
        )
        IMAGE_PROGRESS.finish()

    def tearDown(self) -> None:
        self.worker.stop("test done")
        with wwc._REGISTRY_LOCK:
            if self.worker in wwc._WORKERS:
                wwc._WORKERS.remove(self.worker)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _generate(self, prompt: str, seed: int = 7, **kwargs):
        out = self.tmp / f"o-{seed}.png"
        argv = ["--model", "iris-3b", "--prompt", prompt, "--seed", str(seed), "--output", str(out)]
        return self.worker.request("generate", timeout=30, job=f"j{seed}", command="mflux-generate-iris", argv=argv, **kwargs), out


class ProtocolTests(_WorkerCase):
    def test_load_once_then_run_many(self) -> None:
        events: list[dict] = []
        first, out = self._generate("a cat", 1, on_event=events.append)
        second, _ = self._generate("a dog", 2)
        self.assertTrue(out.read_bytes().startswith(b"PNG:a cat"))
        self.assertFalse(first["warm"])
        self.assertTrue(second["warm"])
        self.assertEqual(second["loads"], 1)
        self.assertIn("load", [e.get("stage") for e in events if e.get("event") == "stage"])
        self.assertTrue(any(e.get("event") == "step" for e in events))
        self.assertIn("footprint_gb", second)

    def test_status_and_unload(self) -> None:
        self._generate("a cat")
        self.assertEqual(self.worker.status()["loaded"], {"model": "iris-3b"})
        self.worker.request("unload", timeout=10)
        self.assertEqual(self.worker.status()["loaded"], {"model": None})

    def test_an_engine_error_comes_back_and_the_worker_lives(self) -> None:
        with self.assertRaises(wwc.WarmWorkerError) as caught:
            self._generate("fail")
        self.assertIn("engine said no", str(caught.exception))
        self.assertTrue(self.worker.running())
        reply, _ = self._generate("a cat", 3)
        self.assertTrue(reply["ok"])

    def test_cooperative_cancel(self) -> None:
        flag = threading.Event()
        threading.Timer(0.3, flag.set).start()
        with self.assertRaises(wwc.WarmWorkerCancelled):
            self._generate("slow", cancel_check=flag.is_set)
        self.assertTrue(self.worker.running())  # the model stays warm after a cancel

    def test_a_job_that_ignores_cancel_is_killed(self) -> None:
        flag = threading.Event()
        threading.Timer(0.3, flag.set).start()
        with self.assertRaises(wwc.WarmWorkerCancelled):
            self._generate("stubborn", cancel_check=flag.is_set)
        self.assertFalse(self.worker.running())

    def test_a_crash_is_reported_and_the_next_request_restarts(self) -> None:
        with self.assertRaises(wwc.WarmWorkerError):
            self._generate("crash")
        reply, _ = self._generate("a cat", 4)
        self.assertFalse(reply["warm"])


class PolicyTests(_WorkerCase):
    def test_another_heavy_kind_stops_an_idle_worker(self) -> None:
        self._generate("a cat")
        self.assertEqual(wwc.release_for_kind("image"), [])
        self.assertEqual(wwc.release_for_kind("voice"), [])
        self.assertIn("test", wwc.release_for_kind("video"))
        self.assertFalse(self.worker.running())
        self.assertEqual(self.worker.last_stop_reason, "a video job started")

    def test_the_heavy_job_queue_calls_the_policy(self) -> None:
        from backend_service.helpers.job_queue import HeavyJobQueue

        seen: list[str] = []
        queue = HeavyJobQueue()
        queue.add_start_listener(seen.append)
        with queue.slot("music"):
            pass
        self.assertEqual(seen, ["music"])

    def test_a_busy_worker_is_never_stopped_by_the_policy(self) -> None:
        done = threading.Event()
        threading.Thread(target=lambda: (self._generate("slow", 9), done.set()), daemon=True).start()
        time.sleep(0.5)
        self.assertFalse(self.worker.stop_if_idle())
        self.assertEqual(wwc.release_for_kind("music"), [])
        self.assertTrue(done.wait(10))

    def test_a_model_over_half_the_memory_is_not_kept(self) -> None:
        reply, _ = self._generate("a cat")
        with mock.patch.object(wwc, "total_memory_gb", return_value=1.0):
            self.worker.after_job({**reply, "footprint_gb": 0.9})
        self.assertFalse(self.worker.running())

    def test_idle_expiry(self) -> None:
        self._generate("a cat")
        self.assertFalse(self.worker.idle_expired())
        self.assertTrue(self.worker.idle_expired(now=time.monotonic() + wwc.IDLE_MINUTES * 60 + 1))

    def test_the_switch_turns_warm_workers_off(self) -> None:
        with mock.patch.dict(os.environ, {"CHAOSENGINE_WARM_WORKERS": "0"}):
            self.assertFalse(wwc.warm_workers_enabled())
            self.assertFalse(mflux_engine.warm_worker_wanted())
        with mock.patch.dict(os.environ, {"CHAOSENGINE_WARM_WORKERS": "1"}):
            self.assertTrue(wwc.warm_workers_enabled())


class EngineTests(_WorkerCase):
    """``MfluxImageEngine.generate`` through the warm worker, with the CLI path as the fallback."""

    def setUp(self) -> None:
        super().setUp()
        self._patches = [
            mock.patch.object(mflux_engine, "_WORKER", self.worker),
            mock.patch.object(mflux_engine, "warm_worker_wanted", return_value=True),
            mock.patch.object(MfluxImageEngine, "probe", return_value={"available": True, "version": "0.24.0"}),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()
        super().tearDown()

    def test_a_batch_runs_warm_with_timings(self) -> None:
        with mock.patch.object(MfluxImageEngine, "_run", side_effect=AssertionError("the CLI must not run")):
            images = MfluxImageEngine().generate(_config(batchSize=2))
        self.assertEqual([i.seed for i in images], [7, 8])
        self.assertEqual(images[0].bytes, b"PNG:a cat:7")
        self.assertFalse(images[0].timings["warm"])
        self.assertTrue(images[1].timings["warm"])
        self.assertIn("model kept loaded", images[1].runtimeNote)
        self.assertIn("peak 1.5 GB", images[0].runtimeNote)
        self.assertFalse(IMAGE_PROGRESS.snapshot()["active"])

    def test_a_crashed_worker_falls_back_to_the_cli(self) -> None:
        def cli(args, out):
            out.write_bytes(b"cli")
            return 3.0

        with mock.patch.object(MfluxImageEngine, "_run", side_effect=cli):
            images = MfluxImageEngine().generate(_config("crash", batchSize=2))
        self.assertEqual([i.bytes for i in images], [b"cli", b"cli"])
        self.assertIn("warm worker failed", images[0].runtimeNote)
        self.assertNotIn("warm worker failed", images[1].runtimeNote)  # the rest of the batch goes straight to the CLI

    def test_cancel_reaches_the_worker(self) -> None:
        threading.Timer(0.4, IMAGE_PROGRESS.request_cancel).start()
        with mock.patch.object(MfluxImageEngine, "_run", side_effect=AssertionError("no fallback after a cancel")):
            with self.assertRaises(GenerationCancelled):
                MfluxImageEngine().generate(_config("slow"))

    def test_the_worker_gets_the_cli_arguments(self) -> None:
        seen: dict = {}
        real = self.worker.request

        def spy(cmd, **kwargs):
            seen.update(kwargs)
            return real(cmd, **kwargs)

        refs = str(self.tmp / "ref.png")
        with mock.patch.object(self.worker, "request", side_effect=spy):
            MfluxImageEngine().generate(_config(repo="black-forest-labs/flux-3-image-klein", referenceImagePaths=[refs], mfluxQuantize=8))
        self.assertEqual(seen["command"], "mflux-generate-flux3-edit")
        self.assertIn("--image-paths", seen["argv"])
        self.assertIn(refs, seen["argv"])
        self.assertEqual(seen["argv"][seen["argv"].index("--quantize") + 1], "8")

    def test_without_the_worker_the_cli_runs(self) -> None:
        with mock.patch.object(mflux_engine, "warm_worker_wanted", return_value=False), \
                mock.patch.object(MfluxImageEngine, "_run", side_effect=lambda args, out: (out.write_bytes(b"cli"), None)[1]):
            [image] = MfluxImageEngine().generate(_config())
        self.assertEqual(image.bytes, b"cli")
        self.assertFalse(self.worker.running())

    def test_run_warm_raises_fallback_when_no_image_was_written(self) -> None:
        with mock.patch.object(self.worker, "request", return_value={"ok": True, "warm": True}):
            with self.assertRaises(WarmFallback):
                MfluxImageEngine()._run_warm(["/x/mflux-generate-iris", "--output", str(self.tmp / "none.png")], self.tmp / "none.png")


class CliTimingTests(unittest.TestCase):
    def test_the_cli_split_is_read_off_the_bar(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        try:
            out = tmp / "o.png"
            script = tmp / "fake.py"
            script.write_text(textwrap.dedent(f"""
                import time
                time.sleep(0.3)
                print("  0%|  | 0/2 [00:00<?, ?it/s]", flush=True)
                time.sleep(0.3)
                print("100%|##| 2/2 [00:00<00:00, 1.0s/it]", flush=True)
                time.sleep(0.2)
                open({str(out)!r}, "wb").write(b"png")
            """), encoding="utf-8")
            engine = MfluxImageEngine()
            engine._run([sys.executable, str(script)], out)
            timings = engine._cli_timings
            self.assertGreaterEqual(timings["loadSeconds"], 0.25)
            self.assertGreaterEqual(timings["denoiseSeconds"], 0.25)
            self.assertGreaterEqual(timings["decodeSeconds"], 0.15)
            self.assertFalse(timings["warm"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
