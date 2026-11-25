"""Video upscale jobs and routes: the warm worker protocol with a fake engine, the heavy-job slot, cancel, resume from
the manifest, path checks and the add-on and fit refusals. A pixel-doubling fake stands in for the model."""

from __future__ import annotations

import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_service.helpers.job_queue import HeavyJobQueue
from backend_service.routes import video_upscale as route
from backend_service.video_upscale import addon, jobs, pipeline
from backend_service.video_upscale.engine import UpscaleError
from tests.video_upscale_fakes import DoublePixels, count_frames, find_ffmpeg, make_clip

FFMPEG = find_ffmpeg()

FAKE_ENGINE = textwrap.dedent("""
    from backend_service.video_upscale.worker import UpscaleEngine

    class Double:
        scale = 2

        def __call__(self, frames):
            return frames.repeat(2, axis=1).repeat(2, axis=2)

    def create_engine():
        return UpscaleEngine(loader=lambda weights, tile: Double())
""")


def _wait(job, seconds=60.0):
    job.thread.join(timeout=seconds)
    return job


@unittest.skipUnless(FFMPEG, "ffmpeg is not available")
class _Clip(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.src = make_clip(FFMPEG, self.root / "in.mp4", seconds=2.0, fps=10)

    def spec(self, name="up.mp4", **kw):
        args = dict(model_id="realesr-general-x4v3", weights="w.pth", size=(192, 128), segment_seconds=0.5)
        args.update(kw)
        return pipeline.make_spec(FFMPEG, self.src, self.root / name, **args)


class WorkerProtocolTests(_Clip):
    def test_segments_run_in_the_worker_process(self):
        engine_file = self.root / "fake_engine.py"
        engine_file.write_text(FAKE_ENGINE, encoding="utf-8")
        with mock.patch.dict("os.environ", {"CHAOSENGINE_WARM_ENGINE_MODULE": str(engine_file)}), \
                mock.patch.object(jobs, "_WORKER_LOG", self.root / "worker.log"), mock.patch.object(jobs, "_WORKER", None):
            try:
                steps = []
                runner = jobs.worker_runner()

                def spy(req, on_frames, is_cancelled):
                    return runner(req, lambda n: (steps.append(n), on_frames(n)), is_cancelled)

                result = pipeline.run_job(self.spec(), spy, exe=FFMPEG, work=self.root / "work")
                self.assertEqual(jobs.worker().kind, "upscale")
                self.assertTrue(jobs.worker().running())
            finally:
                if jobs._WORKER is not None:
                    jobs._WORKER.stop("test done")
        self.assertEqual(result["frames"], 20)
        self.assertEqual(count_frames(FFMPEG, self.root / "up.mp4"), (20, 192, 128))
        self.assertTrue(steps)

    def test_worker_env_puts_the_addon_last(self):
        env = jobs._worker_env()
        parts = env["PYTHONPATH"].split(__import__("os").pathsep)
        self.assertEqual(parts[-1], str(addon.site_dir()))
        self.assertTrue((Path(parts[0]) / "backend_service").is_dir())


class JobTests(_Clip):
    def manager(self, factory=None, queue=None):
        return jobs.UpscaleJobs(runner_factory=factory or (lambda: pipeline.local_runner(DoublePixels())),
                                root=lambda: self.root / "cache", queue=queue or HeavyJobQueue())

    def test_a_job_runs_on_the_slot_and_cleans_up(self):
        queue = HeavyJobQueue()
        kinds = []
        queue.add_start_listener(kinds.append)
        manager = self.manager(queue=queue)
        job = _wait(manager.start(self.spec()))
        self.assertEqual(job.state, "done", job.error)
        self.assertEqual(kinds, ["upscale"])
        self.assertEqual(job.to_dict()["framesDone"], 20)
        self.assertFalse((self.root / "cache" / job.id).exists())
        self.assertTrue((self.root / "up.mp4").is_file())
        self.assertIs(manager.get(job.id), job)

    def test_frames_per_second_is_a_number_while_running_and_when_done(self):
        gate = threading.Event()
        base = pipeline.local_runner(DoublePixels())

        def factory():
            def run(req, on_frames, is_cancelled):
                gate.wait(10)
                return base(req, on_frames, is_cancelled)
            return run

        job = self.manager(factory).start(self.spec())
        try:
            for _ in range(200):
                if job.state == "running" and job.to_dict()["framesPerSecond"] is not None:
                    break
                time.sleep(0.02)
            running = job.to_dict()
            self.assertEqual(running["state"], "running")
            self.assertIsInstance(running["framesPerSecond"], float)
        finally:
            gate.set()
        done = _wait(job).to_dict()
        self.assertEqual(done["state"], "done", done["error"])
        self.assertIsInstance(done["framesPerSecond"], float)
        self.assertGreater(done["framesPerSecond"], 0)
        self.assertEqual((done["framesDone"], done["framesTotal"]), (20, 20))

    def test_same_request_while_running_returns_the_running_job(self):
        gate = threading.Event()
        base = pipeline.local_runner(DoublePixels())

        def factory():
            def run(req, on_frames, is_cancelled):
                gate.wait(10)
                return base(req, on_frames, is_cancelled)
            return run

        manager = self.manager(factory)
        first = manager.start(self.spec())
        self.assertIs(manager.start(self.spec()), first)
        gate.set()
        self.assertEqual(_wait(first).state, "done")

    def test_waits_behind_another_heavy_job_and_cancels_while_waiting(self):
        queue = HeavyJobQueue()
        other = queue.enqueue("video", "ui", "a render")
        queue.wait_turn(other)
        try:
            manager = self.manager(queue=queue)
            job = manager.start(self.spec())
            for _ in range(100):
                if job.ticket_id is not None and queue.position(_ticket(queue, job.ticket_id)) > 0:
                    break
                time.sleep(0.02)
            self.assertEqual(job.state, "queued")
            self.assertTrue(manager.cancel(job.id))
            self.assertEqual(_wait(job, 10).state, "cancelled")
        finally:
            queue.release(other)

    def test_cancel_then_resume_skips_finished_segments(self):
        counter = {"segments": 0, "frames": 0}
        holder = {}

        def factory():
            fake = DoublePixels()
            base = pipeline.local_runner(fake)

            def run(req, on_frames, is_cancelled):
                result = base(req, on_frames, is_cancelled)
                counter["segments"] += 1
                counter["frames"] += result["frames"]
                if counter["segments"] == 2 and "job" in holder:
                    holder["job"].cancel_event.set()
                return result
            return run

        manager = self.manager(factory)
        job = holder["job"] = manager.start(self.spec())
        self.assertEqual(_wait(job).state, "cancelled")
        self.assertTrue(job.to_dict()["resumable"])
        # A restarted backend: a new manager reads the manifest back.
        fresh = self.manager(factory)
        seen = fresh.get(job.id)
        self.assertEqual((seen.state, seen.frames_done), ("cancelled", 10))
        with mock.patch.object(jobs, "check_ready", return_value={}):
            resumed = _wait(fresh.resume(job.id))
        self.assertEqual(resumed.state, "done", resumed.error)
        self.assertEqual(resumed.resumed_segments, 2)
        self.assertEqual(counter["frames"], 20)
        self.assertEqual(count_frames(FFMPEG, self.root / "up.mp4"), (20, 192, 128))

    def test_a_job_that_was_running_reads_back_as_interrupted(self):
        spec = self.spec()
        pipeline.Manifest.open(self.root / "cache" / spec.key(), spec)  # state "running", no live thread
        self.assertEqual(self.manager().get(spec.key()).state, "interrupted")
        self.assertIsNone(self.manager().get("../../etc"))

    def test_resume_refuses_a_changed_input(self):
        spec = self.spec()
        pipeline.Manifest.open(self.root / "cache" / spec.key(), spec)
        self.src.write_bytes(self.src.read_bytes() + b"\0")
        with self.assertRaises(UpscaleError) as caught:
            self.manager().resume(spec.key())
        self.assertEqual(caught.exception.status, 409)

    def test_a_failure_is_reported(self):
        def factory():
            def run(req, on_frames, is_cancelled):
                raise RuntimeError("the GPU fell over")
            return run

        job = _wait(self.manager(factory).start(self.spec()))
        self.assertEqual((job.state, job.error), ("failed", "the GPU fell over"))


def _ticket(queue, ticket_id):
    with queue._cond:
        for ticket in [queue._running, *queue._waiting]:
            if ticket is not None and ticket.id == ticket_id:
                return ticket
    return None


class BuildSpecTests(_Clip):
    def test_refuses_until_the_addon_is_installed(self):
        with mock.patch.object(addon, "site_installed", return_value=False):
            with self.assertRaises(UpscaleError) as caught:
                jobs.build_spec(self.src, self.root / "o.mp4", "realesr-general-x4v3", (192, 128))
        self.assertEqual(caught.exception.status, 409)
        self.assertIn("Install video upscale", str(caught.exception))
        with self.assertRaises(UpscaleError):
            jobs.check_ready("nope")

    def test_spec_carries_the_row_and_the_encoder(self):
        with mock.patch.object(addon, "site_installed", return_value=True), \
                mock.patch.object(addon, "model_installed", return_value=True):
            spec = jobs.build_spec(self.src, self.root / "o.mp4", "realesrgan-x4plus", (192, 128))
        self.assertEqual((spec.tile, spec.batch), (512, 1))
        self.assertIn("-c:v", spec.codec_args)
        self.assertIn("yuv420p", spec.codec_args)
        self.assertTrue(spec.weights.endswith("RealESRGAN_x4plus.pth"))


class RouteTests(_Clip):
    def setUp(self):
        super().setUp()
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        manager = jobs.UpscaleJobs(runner_factory=lambda: pipeline.local_runner(DoublePixels()),
                                   root=lambda: self.root / "cache", queue=HeavyJobQueue())
        for patcher in (mock.patch.object(jobs, "JOBS", manager),
                        mock.patch.object(addon, "site_installed", return_value=True),
                        mock.patch.object(addon, "model_installed", return_value=True),
                        mock.patch.object(route, "output_roots", return_value=[self.outputs.resolve()]),
                        mock.patch.object(route, "default_output_dir", return_value=self.outputs / "upscaled")):
            patcher.start()
            self.addCleanup(patcher.stop)
        app = FastAPI()
        app.include_router(route.router)
        self.client = TestClient(app)

    def test_start_status_and_finish(self):
        res = self.client.post("/api/video/upscale", json={"inputPath": str(self.src), "scale": 2})
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual((data["width"], data["height"]), (192, 128))
        self.assertTrue(data["output"].startswith(str(self.outputs.resolve())))
        _wait(jobs.JOBS.get(data["id"]))
        status = self.client.get(f"/api/video/upscale/jobs/{data['id']}").json()
        self.assertEqual(status["state"], "done", status["error"])
        self.assertEqual(count_frames(FFMPEG, Path(status["output"])), (20, 192, 128))

    def test_output_outside_the_output_folders_is_refused(self):
        res = self.client.post("/api/video/upscale", json={"inputPath": str(self.src), "scale": 2,
                                                           "outputPath": str(self.root / "elsewhere.mp4")})
        self.assertEqual(res.status_code, 400)
        self.assertIn("output folders", res.json()["detail"])
        bad = self.client.post("/api/video/upscale", json={"inputPath": str(self.src), "scale": 2,
                                                           "outputPath": str(self.outputs / "x.exe")})
        self.assertEqual(bad.status_code, 400)
        relative = self.client.post("/api/video/upscale", json={"inputPath": str(self.src), "scale": 2,
                                                                "outputPath": "x.mp4"})
        self.assertEqual(relative.status_code, 400)

    def test_input_checks(self):
        missing = self.client.post("/api/video/upscale", json={"inputPath": str(self.root / "nope.mp4")})
        self.assertEqual(missing.status_code, 400)
        text = self.root / "notes.txt"
        text.write_text("x")
        self.assertEqual(self.client.post("/api/video/upscale", json={"inputPath": str(text)}).status_code, 400)
        both = self.client.post("/api/video/upscale", json={"inputPath": str(self.src), "scale": 2, "width": 100,
                                                            "height": 100})
        self.assertEqual(both.status_code, 422)

    def test_too_big_is_refused_with_the_size(self):
        res = self.client.post("/api/video/upscale", json={"inputPath": str(self.src), "width": 4800, "height": 3200})
        self.assertEqual(res.status_code, 400)
        self.assertIn("4800x3200", res.json()["detail"])

    def test_not_installed_names_the_install_step(self):
        with mock.patch.object(addon, "model_installed", return_value=False):
            res = self.client.post("/api/video/upscale", json={"inputPath": str(self.src)})
        self.assertEqual(res.status_code, 409)
        self.assertIn("Setup", res.json()["detail"])

    def test_a_model_that_will_not_fit_is_refused(self):
        row = {"id": "realesr-general-x4v3", "name": "Real-ESRGAN general x4v3",
               "fit": {"status": "wont-fit", "reason": "Peak memory about 2.0 GB (catalog estimate)"}}
        with mock.patch.object(route, "catalog_with_fit", return_value=[row]):
            res = self.client.post("/api/video/upscale", json={"inputPath": str(self.src)})
        self.assertEqual(res.status_code, 409)
        self.assertIn("will not fit", res.json()["detail"])

    def test_models_lists_fit_and_install_state(self):
        data = self.client.get("/api/video/upscale/models").json()
        self.assertTrue(all("fit" in row and row["installed"] for row in data["models"]))

    def test_unknown_job(self):
        self.assertEqual(self.client.get("/api/video/upscale/jobs/0123456789abcdef").status_code, 404)
        self.assertEqual(self.client.post("/api/video/upscale/jobs/0123456789abcdef/resume").status_code, 404)
        self.assertEqual(self.client.post("/api/video/upscale/jobs/0123456789abcdef/cancel").json(), {"cancelled": False})


if __name__ == "__main__":
    unittest.main()
