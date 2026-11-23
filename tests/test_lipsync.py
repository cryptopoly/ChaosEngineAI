"""Lip sync (D-103, FU-330): shot rules, memory gate, audio preparation and mux, the Mac command line, the worker
protocol with a fake engine, the video route's ``audioPath`` branch with the pipeline stubbed, the agent tool and the
Storyboard vocal field. No test loads a model; the real run is the E2E phase and the dated progress entry."""

from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from backend_service import lipsync_runtime as lip
from backend_service.catalog import VIDEO_MODEL_FAMILIES
from tests.test_storyboard_routes import StoryboardRouteCase

CUDA_REPO = "OzzyGT/LTX-2.3-Distilled-bnb-nf4"
MLX_REPO = "dgrauet/ltx-2.3-mlx-q4"


def _variant(repo: str) -> dict:
    return next(v for f in VIDEO_MODEL_FAMILIES for v in f["variants"] if v["repo"] == repo)


def _ffmpeg() -> str | None:
    from backend_service.storyboard import ffmpeg_bin

    found = ffmpeg_bin.find_ffmpeg(allow_path=True)
    return found[0] if found else None


def _write_wav(path: Path, seconds: float, rate: int = 24000, channels: int = 1) -> Path:
    import numpy as np
    import soundfile as sf

    t = np.arange(int(seconds * rate)) / rate
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    sf.write(str(path), np.stack([tone] * channels, axis=1) if channels > 1 else tone, rate)
    return path


def _write_mp4(exe: str, path: Path, frames: int, fps: int = 24, size: str = "64x48") -> Path:
    subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=s={size}:r={fps}", "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True)
    return path


class ShotRuleTests(unittest.TestCase):
    def test_frames_follow_8k_plus_1_within_3_to_8_seconds(self) -> None:
        self.assertEqual(lip.shot_frames(121, 24), 121)
        self.assertEqual(lip.shot_frames(124, 24), 121)  # rounded down to 8k+1
        self.assertEqual(lip.shot_frames(73, 24), 73)  # 3.0 s
        self.assertEqual(lip.shot_frames(193, 24), 193)  # 8.0 s
        for bad in (65, 72, 201, 257):
            with self.assertRaises(lip.LipSyncError) as caught:
                lip.shot_frames(bad, 24)
            self.assertIn("3 to 8 s", str(caught.exception))
            self.assertEqual(caught.exception.status, 400)

    def test_only_ltx23_rows_lip_sync(self) -> None:
        self.assertEqual(lip.engine_for(_variant(CUDA_REPO)), "cuda")
        self.assertEqual(lip.engine_for(_variant(MLX_REPO)), "mlx")
        self.assertIsNone(lip.engine_for(_variant("dgrauet/ltx-2.5-mlx-q4")))
        self.assertIsNone(lip.engine_for(_variant("diffusers/LTX-2.3-Distilled-Diffusers")))

    def test_catalog_rows_carry_the_measured_and_untested_figures(self) -> None:
        block = _variant(CUDA_REPO)["lipSync"]
        self.assertEqual(block["basis"], "measured")
        self.assertEqual(block["secondsRange"], [110.0, 126.0])
        self.assertEqual((block["cardGb"], block["hostRamGb"]), (24.0, 37.0))
        self.assertEqual(_variant(MLX_REPO)["lipSync"]["basis"], "untested")
        self.assertIn("licence", _variant(CUDA_REPO)["note"])

    def test_estimate_is_the_measured_shot_and_scales_with_the_shot(self) -> None:
        cuda = _variant(CUDA_REPO)
        self.assertAlmostEqual(lip.estimate_seconds(cuda, 768, 512, 121), 118.0)
        self.assertGreater(lip.estimate_seconds(cuda, 768, 512, 193), 118.0)
        self.assertIsNone(lip.estimate_seconds(_variant(MLX_REPO), 768, 512, 121))  # never measured

    def test_audio_must_be_wav_or_flac(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mp3 = Path(tmp) / "v.mp3"
            mp3.write_bytes(b"ID3" + b"x" * 100)
            with self.assertRaises(lip.LipSyncError):
                lip.resolve_audio(str(mp3))
            with self.assertRaises(lip.LipSyncError):
                lip.resolve_audio(str(Path(tmp) / "missing.wav"))
            wav = _write_wav(Path(tmp) / "v.wav", 0.5)
            self.assertEqual(lip.resolve_audio(str(wav)), wav.resolve())


class SingingPromptTests(unittest.TestCase):
    """``lipSyncSinging`` (experiment 4 of docs/research/lip-sync-singing-experiments-2026-11.md)."""

    def test_the_cue_is_appended_once(self) -> None:
        self.assertEqual(lip.singing_prompt("A woman sings in a studio, static camera."),
                         "A woman sings in a studio, static camera, singing, mouth opening and closing with each syllable.")
        cued = lip.singing_prompt("a man on a stage")
        self.assertEqual(lip.singing_prompt(cued), cued)

    def test_a_prompt_that_already_asks_for_syllables_is_kept(self) -> None:
        prompt = "Close-up of a man singing, mouth opening and closing with each syllable, warm light."
        self.assertEqual(lip.singing_prompt(prompt), prompt)

    def test_the_request_field_is_off_by_default(self) -> None:
        from backend_service.models import VideoGenerationRequest

        body = VideoGenerationRequest(modelId=CUDA_REPO, prompt="x", audioPath="a.wav")
        self.assertFalse(body.lipSyncSinging)


class MemoryGateTests(unittest.TestCase):
    need = lip.DEFAULT_NEEDS

    def refusal(self, **kw):
        values = {"card_gb": 23.99, "free_vram_gb": 22.4, "available_ram_gb": 45.0, "worker_loaded": False}
        values.update(kw)
        return lip.memory_refusal(need=self.need, **values)

    def test_a_4090_with_free_ram_fits(self) -> None:
        self.assertIsNone(self.refusal())

    def test_a_smaller_card_is_refused_plainly(self) -> None:
        message = self.refusal(card_gb=16.0)
        self.assertIn("24 GB NVIDIA card", message)
        self.assertIn("16.0 GB", message)

    def test_what_is_short_is_named(self) -> None:
        message = self.refusal(free_vram_gb=18.0, available_ram_gb=30.0)
        self.assertIn("4.0 GB short", message)  # GPU
        self.assertIn("7.0 GB short", message)  # RAM
        self.assertIn("system memory", message)

    def test_a_loaded_worker_already_holds_its_ram(self) -> None:
        self.assertIsNone(self.refusal(available_ram_gb=10.0, free_vram_gb=21.0, worker_loaded=True))

    def test_unreadable_figures_never_block(self) -> None:
        self.assertIsNone(self.refusal(card_gb=None, free_vram_gb=None, available_ram_gb=None))


class CommandTests(unittest.TestCase):
    def test_prepare_audio_upmixes_resamples_and_pads(self) -> None:
        args = lip.prepare_audio_command("ffmpeg", Path("in.wav"), Path("out.wav"), 1.5, 121 / 24)
        self.assertEqual(args[args.index("-ss") + 1], "1.500")
        self.assertEqual(args[args.index("-ac") + 1], "2")
        self.assertEqual(args[args.index("-ar") + 1], "48000")
        self.assertEqual(args[args.index("-af") + 1], "apad")
        self.assertEqual(args[args.index("-t") + 1], "5.041667")

    def test_mux_copies_the_video_and_takes_the_prepared_audio(self) -> None:
        args = lip.mux_command("ffmpeg", Path("v.mp4"), Path("a.wav"), Path("o.mp4"))
        self.assertEqual(args[args.index("-c:v") + 1], "copy")
        self.assertIn("1:a:0", args)
        self.assertIn("-shortest", args)

    def test_ffmpeg_reads_local_files_only(self) -> None:
        # Security review 2026-11-20: ffmpeg picks the demuxer from the content, so a ".wav" that is really a playlist
        # could make it fetch URLs. Every lip-sync ffmpeg call holds its inputs to local files, like the Storyboard's.
        for args in (lip.prepare_audio_command("ffmpeg", Path("in.wav"), Path("out.wav"), 0.0, 5.0),
                     lip.mux_command("ffmpeg", Path("v.mp4"), Path("a.wav"), Path("o.mp4"))):
            self.assertEqual(args[args.index("-protocol_whitelist") + 1], "file,pipe")
            self.assertLess(args.index("-protocol_whitelist"), args.index("-i"))

    def test_mac_a2v_command_line(self) -> None:
        args = lip.mlx_a2v_command(
            pack=Path("/p"), gemma=Path("/g"), prompt="a singer", width=768, height=512, frames=121, fps=24, seed=7,
            audio=Path("/a.wav"), image=Path("/s.png"), output=Path("/o.mp4"),
        )
        self.assertTrue(args[0].endswith("ltx-2-mlx"))
        self.assertEqual(args[1:3], ["a2v", "--distilled"])
        pairs = {args[i]: args[i + 1] for i in range(3, len(args) - 1) if args[i].startswith("-")}
        self.assertEqual(pairs["--model"], str(Path("/p")))
        self.assertEqual(pairs["--gemma"], str(Path("/g")))
        self.assertEqual((pairs["-H"], pairs["-W"], pairs["--frames"], pairs["--frame-rate"]), ("512", "768", "121", "24"))
        self.assertEqual(pairs["--seed"], "7")
        self.assertEqual(pairs["--audio"], str(Path("/a.wav")))
        i = args.index("--image")
        self.assertEqual(args[i + 1:i + 4], [str(Path("/s.png")), "0", "1.0"])
        self.assertEqual(args[-2:], ["--output", str(Path("/o.mp4"))])
        # The distilled flow rejects CFG, STG and a negative prompt: none may be sent.
        for flag in ("--cfg-scale", "--stg-scale", "--negative-prompt", "--enable-teacache"):
            self.assertNotIn(flag, args)


@unittest.skipUnless(_ffmpeg(), "no ffmpeg on this machine")
class FfmpegTests(unittest.TestCase):
    def test_a_short_mono_clip_becomes_the_shot_length_in_stereo(self) -> None:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            src = _write_wav(Path(tmp) / "kokoro.wav", 2.0)
            out = Path(tmp) / "shot.wav"
            lip.prepare_audio(src, out, 0.0, 121 / 24)
            data, rate = sf.read(str(out), always_2d=True)
            self.assertEqual(rate, 48000)
            self.assertEqual(data.shape[1], 2)
            self.assertAlmostEqual(data.shape[0] / rate, 121 / 24, delta=0.002)
            self.assertGreater(abs(data[: rate]).max(), 0.1)  # the speech part
            self.assertEqual(abs(data[int(2.1 * rate):]).max(), 0.0)  # padded with silence

    def test_a_long_clip_is_trimmed_from_the_start_offset(self) -> None:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            src = _write_wav(Path(tmp) / "song.flac".replace(".flac", ".wav"), 12.0, rate=44100, channels=2)
            out = Path(tmp) / "shot.wav"
            lip.prepare_audio(src, out, 4.0, 3.0)
            data, rate = sf.read(str(out), always_2d=True)
            self.assertAlmostEqual(data.shape[0] / rate, 3.0, delta=0.002)
            self.assertGreater(abs(data).min(axis=1).max(), 0.0)

    def test_an_unreadable_clip_is_a_plain_400(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.wav"
            bad.write_bytes(b"RIFF" + b"\x00" * 200)
            with self.assertRaises(lip.LipSyncError) as caught:
                lip.prepare_audio(bad, Path(tmp) / "o.wav", 0.0, 3.0)
            self.assertEqual(caught.exception.status, 400)

    def test_mux_puts_the_vocal_in_the_mp4(self) -> None:
        from backend_service.helpers.video_audio import mp4_has_audio_file

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            video = _write_mp4(_ffmpeg(), folder / "v.mp4", 73)
            wav = folder / "a.wav"
            lip.prepare_audio(_write_wav(folder / "s.wav", 1.0), wav, 0.0, 73 / 24)
            lip.mux(video, wav, folder / "o.mp4")
            self.assertTrue(mp4_has_audio_file(folder / "o.mp4"))


FAKE_ENGINE = textwrap.dedent('''
    import time
    from pathlib import Path


    class Engine:
        def __init__(self):
            self.loaded_snapshot = None

        def loaded(self):
            return {"snapshot": self.loaded_snapshot}

        def unload(self):
            self.loaded_snapshot = None

        def generate(self, request, hooks):
            warm = self.loaded_snapshot == request["snapshot"]
            self.loaded_snapshot = request["snapshot"]
            hooks.stage("load")
            hooks.stage("encode")
            for step in range(int(request["steps"])):
                hooks.step(step + 1, int(request["steps"]))
                hooks.check_cancel()
                time.sleep(float(request.get("prompt") == "slow") * 0.5)
            Path(request["output"]).write_bytes(b"mp4")
            return {"warm": warm, "load_s": 1.0, "encode_s": 0.1, "denoise_s": 2.0, "decode_s": 0.5,
                    "vram_peak_gb": 17.2, "host_peak_gb": 36.5}


    def create_engine():
        return Engine()
''')


class WorkerProtocolTests(unittest.TestCase):
    """The real worker process (``python -m backend_service.lipsync_worker``) with a fake engine: start-up from the
    backend's own Python, steps into VIDEO_PROGRESS, warm reuse, and Cancel."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        engine = Path(self.tmp.name) / "fake_engine.py"
        engine.write_text(FAKE_ENGINE, encoding="utf-8")
        env = mock.patch.dict(os.environ, {"CHAOSENGINE_WARM_ENGINE_MODULE": str(engine)})
        env.start()
        self.addCleanup(env.stop)
        log = mock.patch.object(lip, "_WORKER_LOG", Path(self.tmp.name) / "worker.log")
        log.start()
        self.addCleanup(log.stop)
        lip._WORKER, lip._LOADED_REPO = None, None
        self.addCleanup(self._stop)
        snap = mock.patch("backend_service.helpers.hf_local.local_snapshot_path", return_value=self.tmp.name)
        snap.start()
        self.addCleanup(snap.stop)

    def _stop(self) -> None:
        if lip._WORKER is not None:
            lip._WORKER.stop("test over")
        lip._WORKER, lip._LOADED_REPO = None, None

    def run_shot(self, prompt: str = "a singer") -> dict:
        from backend_service.progress import VIDEO_PROGRESS

        out = Path(self.tmp.name) / f"out-{prompt}.mp4"
        VIDEO_PROGRESS.begin(run_label="t", total_steps=8, phase="loading", message="")
        try:
            result = lip._run_cuda(repo=CUDA_REPO, prompt=prompt, width=768, height=512, frames=121, fps=24, seed=1,
                                   audio=Path("a.wav"), image=Path("s.png"), output=out)
            self.assertEqual(VIDEO_PROGRESS.snapshot().get("step"), 8)
        finally:
            VIDEO_PROGRESS.finish()
        return result

    def test_two_shots_reuse_the_worker_and_report_the_split(self) -> None:
        first = self.run_shot()
        self.assertFalse(first["timings"]["warm"])
        self.assertEqual(first["timings"]["gpuPeakGb"], 17.2)
        pid = lip._WORKER.pid
        second = self.run_shot()
        self.assertTrue(second["timings"]["warm"])
        self.assertEqual(lip._WORKER.pid, pid)
        self.assertEqual(lip.loaded_repo(), CUDA_REPO)
        self.assertTrue(lip.release_worker("test"))
        self.assertIsNone(lip.loaded_repo())

    def test_cancel_reaches_the_worker(self) -> None:
        import threading

        from backend_service.progress import VIDEO_PROGRESS, GenerationCancelled

        threading.Timer(1.2, VIDEO_PROGRESS.request_cancel).start()
        with self.assertRaises(GenerationCancelled):
            self.run_shot("slow")


class RouteTests(unittest.TestCase):
    """POST /api/video/generate with ``audioPath``: the stubbed run, the refusals and their HTTP codes."""

    def setUp(self) -> None:
        from tests.test_video_routes import make_client, restore_env

        self.client, self.tempdir, snapshot = make_client()
        self.addCleanup(self.tempdir.cleanup)
        self.addCleanup(restore_env, snapshot)
        self.folder = Path(self.tempdir.name)
        from tests.storyboard_fixtures import png

        self.still = png(self.folder / "still.png")
        self.wav = _write_wav(self.folder / "line.wav", 3.7)
        local = mock.patch("backend_service.routes.video._video_variant_available_locally", return_value=True)
        local.start()
        self.addCleanup(local.stop)
        gpu = mock.patch.multiple("backend_service.helpers.gpu", get_device_vram_total_gb=lambda: 23.99,
                                  get_device_vram_free_gb=lambda: 22.4)
        gpu.start()
        self.addCleanup(gpu.stop)
        ram = mock.patch("backend_service.helpers.memory_gate.snapshot_memory_signals", return_value=(45.0, 30.0))
        ram.start()
        self.addCleanup(ram.stop)
        # An earlier test file can leave the shared image progress marked active; the video route then answers 409
        # "An image generation is still running" before lip sync is reached (seen in combined Windows runs).
        from backend_service.progress import IMAGE_PROGRESS

        IMAGE_PROGRESS.finish()

    def body(self, **overrides) -> dict:
        body = {"modelId": CUDA_REPO, "prompt": "a woman sings to the camera", "numFrames": 121, "fps": 24,
                "width": 768, "height": 512, "seed": 3, "startImagePath": str(self.still), "audioPath": str(self.wav)}
        body.update(overrides)
        return body

    def post(self, **overrides):
        return self.client.post("/api/video/generate", json=self.body(**overrides))

    def test_a_row_that_cannot_lip_sync_is_400(self) -> None:
        response = self.post(modelId="diffusers/LTX-2.3-Distilled-Diffusers")
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("cannot lip-sync", response.json()["detail"])

    def test_missing_still_mp3_and_length_are_400(self) -> None:
        self.assertIn("start", self.post(startImagePath=None).json()["detail"])
        mp3 = self.folder / "v.mp3"
        mp3.write_bytes(b"ID3" + b"x" * 64)
        self.assertEqual(self.post(audioPath=str(mp3)).status_code, 400)
        response = self.post(numFrames=257)
        self.assertEqual(response.status_code, 400)
        self.assertIn("3 to 8 s", response.json()["detail"])

    def test_not_downloaded_is_409(self) -> None:
        with mock.patch("backend_service.routes.video._video_variant_available_locally", return_value=False):
            self.assertEqual(self.post().status_code, 409)

    def test_short_memory_is_503_with_the_reason(self) -> None:
        with mock.patch("backend_service.helpers.memory_gate.snapshot_memory_signals", return_value=(20.0, 70.0)):
            response = self.post()
        self.assertEqual(response.status_code, 503)
        self.assertIn("17.0 GB short", response.json()["detail"])

    @unittest.skipUnless(_ffmpeg(), "no ffmpeg on this machine")
    def test_a_shot_is_saved_with_its_vocal(self) -> None:
        seen: dict = {}

        def fake_run(**kw):
            seen.update(kw)
            _write_mp4(_ffmpeg(), kw["output"], kw["frames"], kw["fps"])
            return {"width": kw["width"], "height": kw["height"], "timings": {"totalSeconds": 1.0},
                    "runtimeLabel": "LTX-2.3 audio to video (diffusers, 4-bit)"}

        with mock.patch.object(lip, "_run_cuda", side_effect=fake_run):
            response = self.post(numFrames=124)
        self.assertEqual(response.status_code, 200, response.text)
        art = response.json()["artifact"]
        self.assertEqual((art["numFrames"], art["fps"], art["width"], art["height"]), (121, 24, 768, 512))
        self.assertTrue(art["hasAudio"])
        self.assertEqual(art["audioPath"], str(self.wav.resolve()))
        self.assertAlmostEqual(art["clipDurationSeconds"], 5.042, places=3)
        self.assertIn("frames 124 changed to 121", art["runtimeNote"])
        self.assertEqual(seen["seed"], 3)
        self.assertTrue(Path(art["videoPath"]).is_file())

    @unittest.skipUnless(_ffmpeg(), "no ffmpeg on this machine")
    def test_a_sung_vocal_sends_the_singing_cue_and_says_so(self) -> None:
        sent: list[str] = []

        def fake_run(**kw):
            sent.append(kw["prompt"])
            _write_mp4(_ffmpeg(), kw["output"], kw["frames"], kw["fps"])
            return {"width": kw["width"], "height": kw["height"], "timings": {}, "runtimeLabel": "x"}

        with mock.patch.object(lip, "_run_cuda", side_effect=fake_run):
            sung = self.post(lipSyncSinging=True, lipSyncTakes=1)
            spoken = self.post()
        self.assertEqual(sung.status_code, 200, sung.text)
        self.assertEqual(sent[0], lip.singing_prompt("a woman sings to the camera"))
        self.assertEqual(sent[1], "a woman sings to the camera")
        art = sung.json()["artifact"]
        self.assertEqual(art["prompt"], "a woman sings to the camera")  # the gallery keeps the user's prompt
        self.assertIn("sung vocal", art["runtimeNote"])
        self.assertNotIn("sung vocal", spoken.json()["artifact"]["runtimeNote"])

    @unittest.skipUnless(_ffmpeg(), "no ffmpeg on this machine")
    def test_a_sung_shot_makes_three_takes_and_keeps_the_best_scored(self) -> None:
        from backend_service import lipsync_takes
        from backend_service.lip_sync_score import SyncScore

        seeds: list[int] = []
        labels: list[str] = []

        def fake_run(**kw):
            seeds.append(kw["seed"])
            labels.append(kw["label"])
            _write_mp4(_ffmpeg(), kw["output"], kw["frames"], kw["fps"])
            return {"width": kw["width"], "height": kw["height"], "timings": {}, "runtimeLabel": "x"}

        def sync(score: float, usable: bool) -> SyncScore:
            return SyncScore(frames=121, face_frames=121, voiced_frames=80, r=0.5 if usable else 0.1, r_lag0=0.4, lag=1,
                             chance_p95=0.3, chance_mean=0.1, beats_chance=usable, closed_while_voiced=0.1,
                             open_while_silent=0.5, balanced=0.7, motion=1.0, turned_away=0.0, score=score,
                             mouth_mean=0.03)

        results = iter([(sync(0.60, False), None), (sync(0.45, True), None), (sync(0.40, True), None)])
        with mock.patch.object(lip, "_run_cuda", side_effect=fake_run),                 mock.patch.object(lipsync_takes, "scorer", return_value=("python", Path("face.task"))),                 mock.patch.object(lipsync_takes, "score", side_effect=lambda *a: next(results)):
            response = self.post(lipSyncSinging=True)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(seeds, [3, 4, 5])
        self.assertEqual(labels, ["Take 1 of 3: ", "Take 2 of 3: ", "Take 3 of 3: "])
        art = response.json()["artifact"]
        self.assertEqual(art["seed"], 4)  # the usable take beats the higher-scored unusable one
        kept = [t for t in art["lipSyncTakes"] if t["kept"]]
        self.assertEqual([t["take"] for t in kept], [2])
        self.assertIn("kept take 2 (seed 4, score 0.45, usable)", art["runtimeNote"])
        gallery = {o["artifactId"]: o for o in response.json()["outputs"]}
        others = [t["artifactId"] for t in art["lipSyncTakes"] if not t["kept"]]
        self.assertEqual(len(others), 2)
        for other in others:  # the other takes are one click away in the gallery
            self.assertIn(other, gallery)
            self.assertIn("not kept", gallery[other]["runtimeNote"])
            self.assertTrue(Path(gallery[other]["videoPath"]).is_file())

    @unittest.skipUnless(_ffmpeg(), "no ffmpeg on this machine")
    def test_takes_without_the_score_keep_take_one_and_say_why(self) -> None:
        from backend_service import lipsync_takes

        def fake_run(**kw):
            _write_mp4(_ffmpeg(), kw["output"], kw["frames"], kw["fps"])
            return {"width": kw["width"], "height": kw["height"], "timings": {}, "runtimeLabel": "x"}

        with mock.patch.object(lip, "_run_cuda", side_effect=fake_run) as run,                 mock.patch.object(lipsync_takes, "scorer", return_value="the sync score needs MediaPipe"):
            response = self.post(lipSyncTakes=2)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(run.call_count, 2)
        art = response.json()["artifact"]
        self.assertEqual(art["seed"], 3)
        self.assertIn("were not scored (the sync score needs MediaPipe); kept take 1", art["runtimeNote"])

    @unittest.skipUnless(_ffmpeg(), "no ffmpeg on this machine")
    def test_cancel_between_takes_stops_the_run(self) -> None:
        from backend_service import lipsync_takes
        from backend_service.progress import VIDEO_PROGRESS

        def fake_run(**kw):
            _write_mp4(_ffmpeg(), kw["output"], kw["frames"], kw["fps"])
            VIDEO_PROGRESS.request_cancel()  # Cancel pressed while take 1 finishes
            return {"width": kw["width"], "height": kw["height"], "timings": {}, "runtimeLabel": "x"}

        with mock.patch.object(lip, "_run_cuda", side_effect=fake_run) as run,                 mock.patch.object(lipsync_takes, "scorer", return_value="no score"):
            response = self.post(lipSyncTakes=3)
        self.assertEqual((response.status_code, response.json()["detail"]), (409, "cancelled"))
        self.assertEqual(run.call_count, 1)

    def test_more_than_four_takes_is_refused(self) -> None:
        self.assertEqual(self.post(lipSyncTakes=5).status_code, 422)

    def test_cancel_is_409(self) -> None:
        from backend_service.progress import GenerationCancelled

        with mock.patch.object(lip, "prepare_audio"), \
                mock.patch.object(lip, "_run_cuda", side_effect=GenerationCancelled("x")):
            response = self.post()
        self.assertEqual((response.status_code, response.json()["detail"]), (409, "cancelled"))


class AgentToolTests(unittest.TestCase):
    def test_the_tool_is_registered_as_a_job_tool(self) -> None:
        from backend_service.mcp_bridge import scopes, tools

        names = [t["name"] for t in tools.TOOLS]
        self.assertIn("make_lipsync_shot", names)
        self.assertIn("make_lipsync_shot", tools.HANDLERS)
        self.assertTrue(scopes.starts_job("make_lipsync_shot"))
        self.assertEqual(scopes.required_scope("make_lipsync_shot"), "generate")
        self.assertIn("make_lipsync_shot", scopes.ANNOTATIONS)
        schema = next(t for t in tools.TOOLS if t["name"] == "make_lipsync_shot")["inputSchema"]
        self.assertEqual(sorted(schema["required"]), ["audio", "prompt", "start_image"])

    def test_the_tool_builds_a_lip_sync_request(self) -> None:
        from types import SimpleNamespace

        from backend_service.mcp_bridge import lipsync_tool
        from tests.storyboard_fixtures import png

        captured: dict = {}

        def fake_submit(kind, label, run, estimate_seconds=None):
            captured.update(kind=kind, label=label, run=run, estimate=estimate_seconds)
            return SimpleNamespace(describe=lambda: {"jobId": "j1"})

        with tempfile.TemporaryDirectory() as tmp:
            still = png(Path(tmp) / "face.png")
            wav = _write_wav(Path(tmp) / "line.wav", 5.0)
            with mock.patch.object(lipsync_tool, "_lipsync_variant", return_value=_variant(CUDA_REPO)), \
                    mock.patch("backend_service.mcp_bridge.files.output_roots", return_value=[Path(tmp).resolve()]), \
                    mock.patch.object(lipsync_tool.STORE, "submit", side_effect=fake_submit), \
                    mock.patch("backend_service.routes.video._generate_video_locked") as locked:
                locked.return_value = {"artifact": {"videoPath": "x.mp4", "hasAudio": True, "numFrames": 121}, "runtime": {}}
                answer = lipsync_tool.make_lipsync_shot(SimpleNamespace(request=None, app=None),
                                                        {"prompt": "sings", "start_image": str(still), "audio": str(wav),
                                                         "singing": True})
                self.assertGreater(captured["estimate"], 250)  # three takes by default for a sung vocal
                self.assertEqual(answer["jobId"], "j1")
                self.assertEqual(captured["kind"], "video")
                self.assertGreater(captured["estimate"], 100)
                output, _notes = captured["run"]()
            body = locked.call_args[0][1]
        self.assertEqual(body.audioPath, str(wav.resolve()))
        self.assertEqual(body.numFrames, 121)  # 5 s at 24 fps, 8k+1 covering the clip
        self.assertTrue(body.lipSyncSinging)
        self.assertTrue(output["hasAudio"])

    def test_the_tool_refuses_audio_outside_the_output_folders(self) -> None:
        # Security review 2026-11-20: the clip carries the vocal as its sound, so an agent naming any WAV on the disk
        # could have it copied into a downloadable output. Like a Storyboard vocal, only uploads and outputs go in.
        from types import SimpleNamespace

        from backend_service.mcp_bridge import lipsync_tool
        from backend_service.mcp_bridge.tools import ToolError
        from tests.storyboard_fixtures import png

        with tempfile.TemporaryDirectory() as outputs, tempfile.TemporaryDirectory() as private:
            still = png(Path(outputs) / "face.png")
            wav = _write_wav(Path(private) / "private.wav", 5.0)
            with mock.patch.object(lipsync_tool, "_lipsync_variant", return_value=_variant(CUDA_REPO)), \
                    mock.patch("backend_service.mcp_bridge.files.output_roots", return_value=[Path(outputs).resolve()]), \
                    mock.patch.object(lipsync_tool.STORE, "submit") as submit:
                with self.assertRaises(ToolError) as caught:
                    lipsync_tool.make_lipsync_shot(SimpleNamespace(request=None, app=None),
                                                   {"prompt": "sings", "start_image": str(still), "audio": str(wav)})
            submit.assert_not_called()
        self.assertIn("upload", str(caught.exception))


class StoryboardVocalTests(StoryboardRouteCase):
    def test_a_shot_takes_a_vocal_and_loses_it_again(self) -> None:
        c = self
        pid, data = c.create(shots=1)
        sid = data["project"]["shots"][0]["id"]
        wav = _write_wav(c.src / "vocal.wav", 1.0)
        data = c.op(pid, data["project"]["revision"], "set_vocal_clip", {"shotId": sid, "audio": str(wav)})
        vocal = data["project"]["shots"][0]["vocalClip"]
        self.assertTrue(vocal.startswith(f"media/{sid}/vocal-") and vocal.endswith(".wav"), vocal)
        mp3 = c.src / "v.mp3"
        mp3.write_bytes(b"ID3" + b"x" * 64)
        c.op(pid, data["project"]["revision"], "set_vocal_clip", {"shotId": sid, "audio": str(mp3)}, code=400)
        data = c.op(pid, data["project"]["revision"], "set_vocal_clip", {"shotId": sid})
        self.assertIsNone(data["project"]["shots"][0]["vocalClip"])

    def test_a_vocal_changes_the_shot_fingerprint_only_when_set(self) -> None:
        from backend_service.storyboard.deps import fingerprint
        from backend_service.storyboard.models import Project, Shot

        project = Project(id="0123456789abcdef", name="p", shots=[Shot(id="s1")])
        before = fingerprint(project, project.shots[0])
        project.shots[0].vocalClip = "media/s1/vocal-1.wav"
        self.assertNotEqual(fingerprint(project, project.shots[0]), before)

    def test_a_lip_sync_shot_is_priced_from_the_lip_sync_figures(self) -> None:
        from backend_service.storyboard.estimate import _clip_seconds
        from backend_service.storyboard.models import Shot

        shot = Shot(id="s1", seconds=5.0, vocalClip="media/s1/vocal-1.wav")
        self.assertAlmostEqual(_clip_seconds(_variant(CUDA_REPO), shot, 768, 512, 24), 118.0)


if __name__ == "__main__":
    unittest.main()
