"""Video upscale routes the Upscale tab needs: the job list, the finished film and first-frame routes, the probe and
the deinterlace flag in the models route. Real ffmpeg on tiny generated clips (CPU), no model, no server."""

from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_service.helpers.job_queue import HeavyJobQueue
from backend_service.routes import video_upscale as route
from backend_service.storyboard import ffmpeg_bin
from backend_service.video_upscale import addon, jobs, pipeline
from tests.test_video_upscale_jobs import _Clip
from tests.video_upscale_fakes import DoublePixels

PNG = b"\x89PNG\r\n\x1a\n"


class _Routes(_Clip):
    def setUp(self):
        super().setUp()
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.manager = jobs.UpscaleJobs(runner_factory=lambda: pipeline.local_runner(DoublePixels()),
                                        root=lambda: self.root / "cache", queue=HeavyJobQueue())
        for patcher in (mock.patch.object(jobs, "JOBS", self.manager),
                        mock.patch.object(addon, "site_installed", return_value=True),
                        mock.patch.object(addon, "model_installed", return_value=True),
                        mock.patch.object(route, "output_roots", return_value=[self.outputs.resolve()]),
                        mock.patch.object(route, "default_output_dir", return_value=self.outputs / "upscaled")):
            patcher.start()
            self.addCleanup(patcher.stop)
        app = FastAPI()
        app.include_router(route.router)
        self.client = TestClient(app)

    def finished_job(self, output: Path | None = None, state: str = "done") -> jobs.UpscaleJob:
        """A job that never ran: registered in the manager with a state and an output path."""
        out = output or self.outputs / "film.mp4"
        job = jobs.UpscaleJob(dataclasses.replace(self.spec(), output=str(out)))
        job.state = state
        self.manager._jobs[job.id] = job
        return job


class ListTests(_Routes):
    def test_newest_first_and_capped_at_twenty(self):
        base = self.spec()
        for i in range(22):
            job = jobs.UpscaleJob(dataclasses.replace(base, width=200 + i))
            job.started_at = 1000.0 + i
            self.manager._jobs[job.id] = job
        listed = self.client.get("/api/video/upscale/jobs").json()["jobs"]
        self.assertEqual(len(listed), 20)
        self.assertEqual([j["width"] for j in listed], list(range(221, 201, -1)))
        single = self.client.get(f"/api/video/upscale/jobs/{listed[0]['id']}").json()
        self.assertEqual(listed[0], single)

    def test_includes_an_unfinished_job_left_on_disk(self):
        spec = self.spec()
        pipeline.Manifest.open(self.root / "cache" / spec.key(), spec)
        listed = self.client.get("/api/video/upscale/jobs").json()["jobs"]
        self.assertEqual([(j["id"], j["state"]) for j in listed], [(spec.key(), "interrupted")])

    def test_empty(self):
        self.assertEqual(self.client.get("/api/video/upscale/jobs").json(), {"jobs": []})


class FileTests(_Routes):
    def test_streams_a_finished_film_with_ranges(self):
        job = self.finished_job()
        Path(job.spec.output).write_bytes(bytes(range(256)) * 4)
        res = self.client.get(f"/api/video/upscale/jobs/{job.id}/file")
        self.assertEqual((res.status_code, res.headers["content-type"], len(res.content)), (200, "video/mp4", 1024))
        part = self.client.get(f"/api/video/upscale/jobs/{job.id}/file", headers={"Range": "bytes=10-19"})
        self.assertEqual((part.status_code, part.content), (206, bytes(range(10, 20))))

    def test_unknown_unfinished_missing_and_outside_are_404(self):
        self.assertEqual(self.client.get("/api/video/upscale/jobs/0123456789abcdef/file").status_code, 404)
        job = self.finished_job(state="running")
        Path(job.spec.output).write_bytes(b"x")
        self.assertEqual(self.client.get(f"/api/video/upscale/jobs/{job.id}/file").status_code, 404)
        job.state = "done"
        Path(job.spec.output).unlink()
        self.assertEqual(self.client.get(f"/api/video/upscale/jobs/{job.id}/file").status_code, 404)
        elsewhere = self.root / "elsewhere.mp4"
        elsewhere.write_bytes(b"x")
        outside = self.finished_job(output=elsewhere)
        self.assertEqual(self.client.get(f"/api/video/upscale/jobs/{outside.id}/file").status_code, 404)


class FrameTests(_Routes):
    def test_input_and_output_first_frames_are_pngs_and_cached(self):
        job = self.finished_job()
        Path(job.spec.output).write_bytes(self.src.read_bytes())  # any readable film stands in for the result
        for which in ("input", "output"):
            res = self.client.get(f"/api/video/upscale/jobs/{job.id}/frame", params={"which": which})
            self.assertEqual((res.status_code, res.headers["content-type"]), (200, "image/png"), res.text)
            self.assertTrue(res.content.startswith(PNG))
        cached = self.root / "cache" / "frames" / f"{job.id}-input.png"
        self.assertTrue(cached.is_file())
        stamp = cached.stat().st_mtime_ns
        with mock.patch.object(route.subprocess, "run", side_effect=AssertionError("ffmpeg ran again")):
            again = self.client.get(f"/api/video/upscale/jobs/{job.id}/frame", params={"which": "input"})
        self.assertEqual(again.status_code, 200)
        self.assertEqual(cached.stat().st_mtime_ns, stamp)

    def test_unfinished_unknown_and_bad_values(self):
        job = self.finished_job(state="running")
        self.assertEqual(self.client.get(f"/api/video/upscale/jobs/{job.id}/frame").status_code, 404)
        self.assertEqual(self.client.get(f"/api/video/upscale/jobs/{job.id}/frame", params={"which": "x"}).status_code, 422)
        self.assertEqual(self.client.get("/api/video/upscale/jobs/0123456789abcdef/frame",
                                         params={"which": "input"}).status_code, 404)

    def test_an_ffmpeg_failure_is_a_422_and_leaves_no_cache(self):
        job = self.finished_job()
        failed = mock.Mock(returncode=1, stderr="boom")
        with mock.patch.object(route.subprocess, "run", return_value=failed):
            res = self.client.get(f"/api/video/upscale/jobs/{job.id}/frame", params={"which": "input"})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(list((self.root / "cache" / "frames").glob("*.png")), [])

    def test_runs_ffmpeg_in_list_form(self):
        job = self.finished_job()
        seen = {}

        def fake(args, **kw):
            seen["args"], seen["shell"] = args, kw.get("shell")
            Path(args[-1]).write_bytes(PNG)
            return mock.Mock(returncode=0, stderr="")

        with mock.patch.object(route.subprocess, "run", side_effect=fake):
            res = self.client.get(f"/api/video/upscale/jobs/{job.id}/frame", params={"which": "input"})
        self.assertEqual(res.status_code, 200)
        self.assertIsInstance(seen["args"], list)
        self.assertFalse(seen["shell"])
        self.assertIn(str(self.src.resolve()), seen["args"])


class ProbeTests(_Routes):
    def test_probe_reports_the_source(self):
        res = self.client.post("/api/video/upscale/probe", json={"inputPath": str(self.src)})
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual((data["width"], data["height"], data["fps"], data["hasAudio"]), (96, 64, 10.0, True))
        self.assertAlmostEqual(data["durationSeconds"], 2.0, delta=0.2)
        self.assertEqual(data["frames"], 20)
        self.assertEqual(data["inputPath"], str(self.src.resolve()))
        self.assertIn(data["interlaced"], (False, None))
        self.assertEqual(set(data), {"inputPath", "width", "height", "fps", "durationSeconds", "frames", "hasAudio",
                                     "interlaced"})

    def test_probe_uses_the_start_rules(self):
        def post(path):
            return self.client.post("/api/video/upscale/probe", json={"inputPath": str(path)})

        self.assertEqual(post(self.root / "nope.mp4").status_code, 400)
        text = self.root / "notes.txt"
        text.write_text("x")
        self.assertEqual(post(text).status_code, 400)
        junk = self.root / "junk.mp4"
        junk.write_bytes(b"not a video")
        self.assertEqual(post(junk).status_code, 400)
        self.assertEqual(self.client.post("/api/video/upscale/probe", json={}).status_code, 422)

    def test_probe_without_ffmpeg_offers_the_install(self):
        with mock.patch.object(ffmpeg_bin, "ffmpeg_exe", side_effect=ffmpeg_bin.FfmpegMissing()):
            res = self.client.post("/api/video/upscale/probe", json={"inputPath": str(self.src)})
        self.assertEqual(res.status_code, 409)


class FieldOrderTests(unittest.TestCase):
    def test_reads_the_stream_line(self):
        line = "  Stream #0:0: Video: h264 (High), yuv420p(%s), 720x576 [SAR 16:11 DAR 20:11], 25 fps, 25 tbr, 1k tbn"
        cases = {"tv, bt709, progressive": False, "tv, top first": True, "top coded first (swapped)": True,
                 "tv, bt709, bottom coded first": True, "tv, bt470bg, tb": True, "tv, bt709": None}
        for text, expected in cases.items():
            with self.subTest(text):
                self.assertIs(pipeline.interlaced_from_probe(line % text), expected)
        self.assertIsNone(pipeline.interlaced_from_probe("Duration: 00:00:01.00"))


class DeinterlaceFlagTests(_Routes):
    def models(self):
        return self.client.get("/api/video/upscale/models").json()

    def test_follows_the_filter_list(self):
        with mock.patch.object(ffmpeg_bin, "ffmpeg_filters", return_value=frozenset({"scale", "bwdif"})):
            self.assertIs(self.models()["deinterlaceAvailable"], True)
        with mock.patch.object(ffmpeg_bin, "ffmpeg_filters", return_value=frozenset({"scale"})):
            self.assertIs(self.models()["deinterlaceAvailable"], False)

    def test_false_without_ffmpeg(self):
        with mock.patch.object(ffmpeg_bin, "ffmpeg_filters", side_effect=ffmpeg_bin.FfmpegMissing()):
            self.assertIs(self.models()["deinterlaceAvailable"], False)


if __name__ == "__main__":
    unittest.main()
