"""Video upscale fast tier: tiles, size cap, the ffmpeg frame pipeline, the resume manifest and the final mux.

Everything runs with a pixel-doubling fake upscaler on tiny lavfi clips; no model, no GPU, no download.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend_service.video_upscale import engine, pipeline
from tests.video_upscale_fakes import DoublePixels, count_frames, find_ffmpeg, make_clip

FFMPEG = find_ffmpeg()


class EngineTests(unittest.TestCase):
    def test_tiles_give_the_same_picture_as_one_pass(self):
        rng = np.random.default_rng(1)
        frames = rng.integers(0, 255, size=(2, 37, 53, 3), dtype=np.uint8)
        whole = DoublePixels()(frames)
        calls = []

        def run(part):
            calls.append(part.shape)
            return DoublePixels()(part)

        tiled = engine.upscale_tiled(run, frames, 2, tile=16, overlap=4)
        np.testing.assert_array_equal(tiled, whole)
        self.assertGreater(len(calls), 4)
        self.assertTrue(all(shape[1] == 16 and shape[2] == 16 for shape in calls))

    def test_tile_zero_or_larger_than_the_frame_is_one_call(self):
        calls = []
        frames = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        engine.upscale_tiled(lambda f: calls.append(1) or DoublePixels()(f), frames, 2, tile=0)
        engine.upscale_tiled(lambda f: calls.append(1) or DoublePixels()(f), frames, 2, tile=64)
        self.assertEqual(len(calls), 2)

    def test_spans_cover_the_axis_once(self):
        for length, tile, overlap in ((100, 32, 8), (33, 32, 8), (64, 32, 0), (1000, 512, 32)):
            spans = engine._spans(length, tile, overlap)
            kept = [k for _s, _e, a, b in spans for k in range(a, b)]
            self.assertEqual(kept, list(range(length)))
            for start, end, a, b in spans:
                self.assertTrue(start <= a < b <= end)

    def test_wrong_model_output_is_refused(self):
        with self.assertRaises(engine.UpscaleError):
            engine.upscale_tiled(lambda f: f, np.zeros((1, 4, 4, 3), np.uint8), 2, tile=0)

    def test_size_cap(self):
        self.assertEqual(engine.target_size(1280, 720, 3), (3840, 2160))
        engine.check_target(3840, 2160)
        engine.check_target(4096, 2304)
        with self.assertRaises(engine.UpscaleError) as caught:
            engine.check_target(*engine.target_size(1280, 720, 4))
        self.assertIn("5120x2880", str(caught.exception))
        self.assertIn("4096", str(caught.exception))

    def test_engine_module_does_not_import_torch(self):
        code = ("import sys; import backend_service.video_upscale.engine, backend_service.video_upscale.pipeline; "
                "print('torch' in sys.modules, 'spandrel' in sys.modules)")
        out = subprocess.run([__import__('sys').executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.split(), ["False", "False"])


class FpsAndPlanTests(unittest.TestCase):
    def test_ntsc_rates_are_fractions(self):
        self.assertEqual(pipeline.fps_text(29.97), "30000/1001")
        self.assertEqual(pipeline.fps_text(23.98), "24000/1001")
        self.assertEqual(pipeline.fps_text(25.0), "25")
        self.assertAlmostEqual(pipeline.fps_value("30000/1001"), 29.97, places=2)

    def test_base_rate_is_read_from_tbr(self):
        line = ("  Stream #0:0[0x1]: Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 160x90 "
                "[SAR 1:1 DAR 16:9], 52 kb/s, 25.04 fps, 24 tbr, 1000k tbn (default)")
        self.assertEqual(pipeline._TBR.search(line).group(1), "24")
        self.assertEqual(pipeline._SAR.search(line).groups(), ("1", "1"))

    def test_segments(self):
        segs = pipeline.plan_segments(23, 10)
        self.assertEqual([(s["start"], s["last"]) for s in segs], [(0, False), (10, False), (20, True)])
        self.assertEqual(len(pipeline.plan_segments(0, 10)), 1)

    def test_decode_seeks_half_a_frame_early_and_reads_whole_frames(self):
        req = {"ffmpeg": "ff", "input": "in.mp4", "first_frame": 20, "frames": 10, "last": False, "fps": "10",
               "deinterlace": True}
        cmd = pipeline.decode_command(req)
        self.assertEqual(cmd[cmd.index("-ss") + 1], "1.950000")
        self.assertEqual(cmd[cmd.index("-frames:v") + 1], "10")
        self.assertIn("bwdif=mode=send_frame,setpts=PTS-STARTPTS,fps=10", cmd)
        self.assertEqual(cmd[cmd.index("-protocol_whitelist") + 1], "file,pipe")
        last = pipeline.decode_command({**req, "first_frame": 0, "last": True, "deinterlace": False})
        self.assertNotIn("-ss", last)
        self.assertNotIn("-frames:v", last)

    def test_encode_scales_to_the_exact_target_and_keeps_the_rate(self):
        req = {"ffmpeg": "ff", "width": 300, "height": 200, "fps": "30000/1001", "sar": "16/15",
               "codec_args": ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]}
        cmd = pipeline.encode_command(req, (400, 256), Path("o.mp4"))
        self.assertIn("scale=300:200:flags=lanczos,setsar=16/15,format=yuv420p", cmd)
        storyboard = pipeline.encode_command({**req, "avtb": True}, (400, 256), Path("o.mp4"))
        self.assertIn("scale=300:200:flags=lanczos,setsar=16/15,format=yuv420p,settb=AVTB,setpts=N/(FRAME_RATE*TB)",
                      storyboard)
        self.assertEqual(cmd[cmd.index("-framerate") + 1], "30000/1001")
        same = pipeline.encode_command(req, (300, 200), Path("o.mp4"))
        self.assertFalse(any(part.startswith("scale=") for part in same))

    def test_segment_request_keys_do_not_collide_with_the_worker_client(self):
        import inspect

        from backend_service.warm_worker_client import WarmWorker

        own = set(inspect.signature(WarmWorker.request).parameters) | {"cmd", "id"}
        spec = pipeline.JobSpec(input="i", output="o", model_id="m", weights="w", tile=0, batch=1, width=4, height=4,
                                source={"width": 2, "height": 2, "fps": "10", "frames": 1}, input_bytes=1, input_mtime=0)
        req = pipeline.segment_request(spec, "ff", pipeline.plan_segments(1, 1)[0], Path("o.mp4"))
        self.assertEqual(own & set(req), set())

    def test_concat_list_takes_only_segment_names(self):
        self.assertEqual(pipeline.concat_list(["seg-00000.mp4"]), "file 'seg-00000.mp4'\n")
        for bad in ("../seg-00000.mp4", "seg-0.mp4", "x'.mp4"):
            with self.assertRaises(ValueError):
                pipeline.concat_list([bad])


@unittest.skipUnless(FFMPEG, "ffmpeg is not available")
class PipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.src = make_clip(FFMPEG, self.root / "in.mp4", seconds=2.0, fps=10, size="96x64")

    def tearDown(self):
        self._tmp.cleanup()

    def _spec(self, out: Path, **kw):
        defaults = dict(model_id="fake", weights="none", size=(192, 128), segment_seconds=0.5)
        defaults.update(kw)
        return pipeline.make_spec(FFMPEG, self.src, out, **defaults)

    def test_probe(self):
        info = pipeline.probe_source(FFMPEG, self.src)
        self.assertEqual((info.width, info.height, info.fps, info.frames, info.has_audio), (96, 64, "10", 20, True))

    def test_whole_job_doubles_every_frame_and_keeps_the_sound(self):
        out = self.root / "out" / "up.mp4"
        spec = self._spec(out)
        fake = DoublePixels()
        seen = []
        result = pipeline.run_job(spec, pipeline.local_runner(fake), exe=FFMPEG, work=self.root / "work",
                                  progress=lambda done, total, i, n: seen.append((done, total, i, n)))
        self.assertEqual(result["frames"], 20)
        self.assertEqual(result["segments"], 4)
        self.assertEqual(fake.frames, 20)
        self.assertEqual(count_frames(FFMPEG, out), (20, 192, 128))
        probe = pipeline.probe_source(FFMPEG, out)
        self.assertTrue(probe.has_audio)
        self.assertEqual(probe.fps, "10")
        self.assertAlmostEqual(probe.duration, 2.0, delta=0.15)
        self.assertEqual(seen[-1][0], 20)
        manifest = json.loads((self.root / "work" / "manifest.json").read_text())
        self.assertEqual(manifest["state"], "done")
        self.assertTrue(all(s["done"] for s in manifest["segments"]))

    def test_upscaled_picture_matches_the_source_doubled(self):
        out = self.root / "up.mp4"
        pipeline.run_job(self._spec(out, segment_seconds=10), pipeline.local_runner(DoublePixels()), exe=FFMPEG,
                         work=self.root / "w")
        first_src = subprocess.run([FFMPEG, "-v", "error", "-i", str(self.src), "-frames:v", "1", "-f", "rawvideo",
                                    "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        first_out = subprocess.run([FFMPEG, "-v", "error", "-i", str(out), "-frames:v", "1", "-vf",
                                    "scale=96:64:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                   capture_output=True, check=True).stdout
        a = np.frombuffer(first_src, np.uint8).astype(int)
        b = np.frombuffer(first_out, np.uint8).astype(int)
        self.assertLess(np.abs(a - b).mean(), 12)  # two lossy encodes apart, same picture

    def test_target_other_than_the_model_scale_is_scaled_with_lanczos(self):
        out = self.root / "up.mp4"
        pipeline.run_job(self._spec(out, size=(144, 96)), pipeline.local_runner(DoublePixels()), exe=FFMPEG,
                         work=self.root / "w")
        self.assertEqual(count_frames(FFMPEG, out), (20, 144, 96))

    def test_cancel_then_resume_skips_finished_segments(self):
        out = self.root / "up.mp4"
        spec = self._spec(out)
        work = self.root / "work"
        fake = DoublePixels()
        state = {"segments": 0}
        base = pipeline.local_runner(fake)

        def runner(req, on_frames, is_cancelled):
            result = base(req, on_frames, is_cancelled)
            state["segments"] += 1
            return result

        stop = lambda: state["segments"] >= 2  # noqa: E731 - cancel after two segments
        with self.assertRaises(pipeline.Cancelled):
            pipeline.run_job(spec, runner, exe=FFMPEG, work=work, is_cancelled=stop)
        manifest = json.loads((work / "manifest.json").read_text())
        self.assertEqual(manifest["state"], "cancelled")
        self.assertEqual([s["done"] for s in manifest["segments"]], [True, True, False, False])
        self.assertFalse(out.exists())

        again = DoublePixels()
        result = pipeline.run_job(pipeline.JobSpec.from_dict(manifest["spec"]), pipeline.local_runner(again),
                                  exe=FFMPEG, work=work)
        self.assertEqual(result["resumedSegments"], 2)
        self.assertEqual(again.frames, 10)  # only the two unfinished segments
        self.assertEqual(count_frames(FFMPEG, out), (20, 192, 128))

    def test_a_changed_request_starts_a_fresh_manifest(self):
        work = self.root / "work"
        spec = self._spec(self.root / "a.mp4")
        pipeline.Manifest.open(work, spec).mark(0, 5)
        other = self._spec(self.root / "a.mp4", size=(144, 96))
        self.assertNotEqual(spec.key(), other.key())
        self.assertFalse(pipeline.Manifest.open(work, other).segments[0]["done"])

    def test_cancel_inside_a_segment_leaves_no_partial_file(self):
        out = self.root / "up.mp4"
        spec = self._spec(out, segment_seconds=10)
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] > 3

        with self.assertRaises(pipeline.Cancelled):
            pipeline.run_job(spec, pipeline.local_runner(DoublePixels()), exe=FFMPEG, work=self.root / "w",
                             is_cancelled=cancel)
        self.assertEqual([p.name for p in (self.root / "w").glob("seg-*")], [])

    def _clip_with_long_sound(self, frames: int, sound_seconds: float) -> Path:
        """``frames`` of picture at 10 fps with a sound track that runs past the picture (B1316: the container's
        duration, not the frame count, then says the file is longer than it is)."""
        video = make_clip(FFMPEG, self.root / "v.mp4", seconds=frames / 10, fps=10, audio=False)
        sound = self.root / "a.m4a"
        subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        f"sine=frequency=440:sample_rate=48000:duration={sound_seconds}", "-c:a", "aac", str(sound)],
                       check=True)
        out = self.root / "long-sound.mp4"
        subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", str(video), "-i", str(sound),
                        "-c", "copy", str(out)], check=True)
        return out

    def test_probe_counts_the_real_frames_not_duration_times_rate(self):
        src = self._clip_with_long_sound(21, 3.0)
        self.assertEqual(count_frames(FFMPEG, src)[0], 21)
        self.assertEqual(pipeline.probe_source(FFMPEG, src).frames, 21)  # duration x rate says 30

    def test_output_has_exactly_the_source_frames_when_the_last_segment_is_short(self):
        # 21 frames in 2 s segments: the last segment holds one frame and used to come out padded to the container's end.
        src = self._clip_with_long_sound(21, 3.0)
        out = self.root / "up.mp4"
        spec = pipeline.make_spec(FFMPEG, src, out, model_id="fake", weights="none", size=(192, 128), segment_seconds=2.0)
        seen = []
        result = pipeline.run_job(spec, pipeline.local_runner(DoublePixels()), exe=FFMPEG, work=self.root / "w",
                                  progress=lambda done, total, i, n: seen.append((done, total)))
        self.assertEqual(result["frames"], 21)
        self.assertEqual(count_frames(FFMPEG, out), (21, 192, 128))
        self.assertTrue(all(total == 21 for _done, total in seen), seen)

    def test_a_clip_without_sound(self):
        src = make_clip(FFMPEG, self.root / "silent.mp4", seconds=1.0, fps=10, audio=False)
        out = self.root / "up.mkv"
        spec = pipeline.make_spec(FFMPEG, src, out, model_id="fake", weights="none", size=(192, 128))
        pipeline.run_job(spec, pipeline.local_runner(DoublePixels()), exe=FFMPEG, work=self.root / "w")
        self.assertEqual(count_frames(FFMPEG, out), (10, 192, 128))
        self.assertFalse(pipeline.probe_source(FFMPEG, out).has_audio)

    def test_deinterlace_runs_when_the_build_has_bwdif(self):
        from backend_service.storyboard import ffmpeg_bin

        if pipeline.DEINTERLACE_FILTER not in ffmpeg_bin.ffmpeg_filters():
            self.skipTest("this ffmpeg has no bwdif")
        out = self.root / "up.mp4"
        pipeline.run_job(self._spec(out, deinterlace=True, segment_seconds=10), pipeline.local_runner(DoublePixels()),
                         exe=FFMPEG, work=self.root / "w")
        self.assertEqual(count_frames(FFMPEG, out)[0], 20)

    def test_oversized_target_is_refused_before_any_work(self):
        with self.assertRaises(engine.UpscaleError):
            self._spec(self.root / "x.mp4", size=(5000, 3000))


if __name__ == "__main__":
    unittest.main()
