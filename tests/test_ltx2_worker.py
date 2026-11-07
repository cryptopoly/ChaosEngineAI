"""Warm ltx-2-mlx worker (V1) with a stub engine, and the shared output parser (C1).

The real ``backend_service/ltx2_worker.py`` runs under this interpreter with
``CHAOSENGINE_WARM_ENGINE_MODULE`` pointing at a fake engine that prints what ltx-2-mlx prints (phase
lines and tqdm bars on stderr); the worker forwards them as ``line`` events.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from unittest import mock

from backend_service import ltx2_mlx_runtime as ltx2
from backend_service import warm_worker_client as wwc
from backend_service.progress import VIDEO_PROGRESS, GenerationCancelled
from backend_service.video_runtime.types import VideoGenerationConfig

ROOT = Path(__file__).resolve().parents[1]

FAKE_ENGINE = textwrap.dedent('''
    import os, sys, time

    class Engine:
        def __init__(self):
            self.model = None

        def loaded(self):
            return {"model": self.model}

        def unload(self):
            self.model = None

        def generate(self, request, hooks):
            argv = request["argv"]
            assert argv[:2] == ["generate", "--distilled"], argv
            prompt = argv[argv.index("--prompt") + 1]
            output = argv[argv.index("--output") + 1]
            model = argv[argv.index("--model") + 1]
            warm = self.model == model
            if prompt == "crash":
                os._exit(4)
            if not warm:
                print("[Loading transformer (transformer-distilled.safetensors)] ...", file=sys.stderr, flush=True)
                self.model = model
                print("[Loading transformer (transformer-distilled.safetensors)] done in 2.5s", file=sys.stderr, flush=True)
            print("[Encoding prompt] ...", file=sys.stderr, flush=True)
            print("[Encoding prompt] done in 1.5s", file=sys.stderr, flush=True)
            steps = 60 if prompt == "slow" else 8
            for n in range(steps + 1):
                # tqdm redraws with a carriage return
                sys.stderr.write(f"\\rDenoising (stage 1): |##| {n}/{steps} [00:01<00:01, 1.0s/it]")
                sys.stderr.flush()
                time.sleep(0.05)
            sys.stderr.write("\\n")
            print("[Decoding video + audio + muxing] ...", file=sys.stderr, flush=True)
            print("[Decoding video + audio + muxing] done in 0.5s", file=sys.stderr, flush=True)
            with open(output, "wb") as handle:
                handle.write(b"MP4:" + prompt.encode())
            return {"output": output, "warm": warm}

    def create_engine():
        return Engine()
''')


def _config(prompt: str = "waves", **overrides) -> VideoGenerationConfig:
    values = dict(modelId="ltx", modelName="LTX-2.5", repo="dgrauet/ltx-2.5-mlx-q4", prompt=prompt, negativePrompt="",
                  width=512, height=320, numFrames=33, fps=24, guidance=1.0, seed=11)
    values.update(overrides)
    return VideoGenerationConfig(**values)


class Ltx2WorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ltx2-warm-test-"))
        self.pack = self.tmp / "pack"
        self.pack.mkdir()
        for name in ltx2.REQUIRED_FILES_25:
            (self.pack / name).write_bytes(b"")
        fake = self.tmp / "fake_engine.py"
        fake.write_text(FAKE_ENGINE, encoding="utf-8")
        env = {**os.environ, "CHAOSENGINE_WARM_ENGINE_MODULE": str(fake), "PYTHONPATH": ""}
        self.worker = wwc.WarmWorker(
            name="ltx-test", kind="video", command=lambda: [sys.executable, str(ROOT / "backend_service" / "ltx2_worker.py")],
            env=lambda: env, log_path=self.tmp / "worker.log", hard_cancel_after_s=2.0,
        )
        self._patches = [
            mock.patch.object(ltx2, "_WORKER", self.worker),
            mock.patch.object(ltx2, "warm_worker_wanted", return_value=True),
            mock.patch.object(ltx2, "_local_snapshot", return_value=self.pack),
            mock.patch.object(ltx2.Ltx2MlxVideoEngine, "probe",
                              return_value=mock.Mock(realGenerationAvailable=True, message="ok")),
        ]
        for patch in self._patches:
            patch.start()
        VIDEO_PROGRESS.finish()

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()
        self.worker.stop("test done")
        with wwc._REGISTRY_LOCK:
            if self.worker in wwc._WORKERS:
                wwc._WORKERS.remove(self.worker)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clips_run_warm_with_the_timing_split(self) -> None:
        engine = ltx2.Ltx2MlxVideoEngine()
        with mock.patch.object(ltx2.Ltx2MlxVideoEngine, "_run", side_effect=AssertionError("the CLI must not run")):
            first = engine.generate(_config())
            second = engine.generate(_config("gulls"))
        self.assertEqual(first.bytes, b"MP4:waves")
        self.assertFalse(first.timings["warm"])
        self.assertTrue(second.timings["warm"])
        self.assertEqual(first.timings["loadSeconds"], 2.5)
        self.assertEqual(second.timings["loadSeconds"], 0.0)
        self.assertEqual(second.timings["encodeSeconds"], 1.5)
        self.assertEqual(second.timings["decodeSeconds"], 0.5)
        self.assertIn("pipeline kept loaded", second.runtimeNote)
        self.assertFalse(VIDEO_PROGRESS.snapshot()["active"])

    def test_a_crash_falls_back_to_the_cli(self) -> None:
        def cli(self_, args, output):
            output.write_bytes(b"cli")
            self_._cli_timings = {"totalSeconds": 1.0, "warm": False}

        with mock.patch.object(ltx2.Ltx2MlxVideoEngine, "_run", autospec=True, side_effect=cli):
            video = ltx2.Ltx2MlxVideoEngine().generate(_config("crash"))
        self.assertEqual(video.bytes, b"cli")
        self.assertIn("warm worker failed", video.runtimeNote)
        self.assertEqual(video.timings, {"totalSeconds": 1.0, "warm": False})

    def test_cancel_stops_an_engine_without_a_step_hook(self) -> None:
        threading.Timer(0.6, VIDEO_PROGRESS.request_cancel).start()
        with mock.patch.object(ltx2.Ltx2MlxVideoEngine, "_run", side_effect=AssertionError("no fallback after a cancel")):
            with self.assertRaises(GenerationCancelled):
                ltx2.Ltx2MlxVideoEngine().generate(_config("slow"))
        self.assertTrue(self.worker.running())  # the output write raised Cancelled; no kill needed

    def test_unload_stops_an_idle_worker(self) -> None:
        ltx2.Ltx2MlxVideoEngine().generate(_config())
        self.assertTrue(self.worker.running())
        ltx2.Ltx2MlxVideoEngine().unload()
        self.assertFalse(self.worker.running())

    def test_the_worker_rejects_anything_but_the_distilled_pipeline(self) -> None:
        # The real engine refuses other modes so the CLI handles them; the fake checks the argv shape.
        args = ltx2.Ltx2MlxVideoEngine._build_command(_config(), self.pack, None, 512, 320, 1, self.tmp / "o.mp4")
        self.assertEqual(args[1:3], ["generate", "--distilled"])


class OutputParserTests(unittest.TestCase):
    def test_the_split_from_phase_lines(self) -> None:
        parser = ltx2.Ltx2OutputParser()
        for line in [
            "[Loading text encoder (Gemma)] ...",
            "[Loading text encoder (Gemma)] done in 1.6s",
            "[Encoding prompt] done in 3.7s",
            "[Loading transformer (transformer-distilled.safetensors)] done in 2.9s",
            "Denoising (stage 1):  50%|##5 | 4/8 [00:11<00:11, 2.8s/it]",
            "[Loading decoders (VAE + audio + vocoder)] done in 0.1s",
            "[Decoding video + audio + muxing] done in 2.1s",
        ]:
            parser.feed(line)
        timings = parser.timings(30.0, startup_s=1.0)
        self.assertEqual(timings["loadSeconds"], 5.6)
        self.assertEqual(timings["encodeSeconds"], 3.7)
        self.assertEqual(timings["decodeSeconds"], 2.1)
        self.assertEqual(timings["denoiseSeconds"], 18.6)
        self.assertEqual(timings["totalSeconds"], 30.0)


if __name__ == "__main__":
    unittest.main()
