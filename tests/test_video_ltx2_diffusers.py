"""LTX-2.3 on diffusers (NVIDIA): pipeline choice, arguments, pictures, and the sound that goes into the mp4."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend_service.video_runtime import DiffusersVideoEngine
from backend_service.video_runtime import audio_mux
from backend_service.video_runtime.pipeline_helpers import (
    FrameList,
    build_pipeline_kwargs,
    invoke_pipeline,
    pipeline_class_for_repo,
    recompress_still,
)
from backend_service.video_runtime.repos import (
    IMAGE_TO_VIDEO_REGISTRY,
    LTX2_DIFFUSERS_REPOS,
    image_input_support,
    is_ltx2_distilled_repo,
)
from backend_service.video_runtime.types import VideoGenerationConfig

DISTILLED_NF4 = "OzzyGT/LTX-2.3-Distilled-bnb-nf4"
DEV_BF16 = "diffusers/LTX-2.3-Diffusers"


def _config(repo: str = DISTILLED_NF4, start=None, end=None, **extra) -> VideoGenerationConfig:
    return VideoGenerationConfig(
        modelId=repo, modelName="m", repo=repo, prompt="waves", negativePrompt="", width=768, height=512,
        numFrames=97, fps=24, steps=50, guidance=3.0, seed=1, startImagePath=start, endImagePath=end, **extra,
    )


def _pipeline(name: str):
    return type(name, (), {})()


def _png(path: Path, colour=(200, 30, 30), size=(640, 360)) -> Path:
    from PIL import Image

    Image.new("RGB", size, colour).save(path)
    return path


@dataclass
class _Condition:
    frames: object
    index: int = 0
    strength: float = 1.0
    crf: int | None = None


def _fake_diffusers_modules():
    """Stand-ins for the two diffusers modules the argument builder imports (they need transformers to import)."""
    utils = types.ModuleType("diffusers.pipelines.ltx2.utils")
    utils.DEFAULT_NEGATIVE_PROMPT = "blurry, distorted"
    utils.DISTILLED_SIGMA_VALUES = [1.0, 0.99, 0.98, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
    condition = types.ModuleType("diffusers.pipelines.ltx2.pipeline_ltx2_condition")
    condition.LTX2VideoCondition = _Condition
    return {"diffusers.pipelines.ltx2.utils": utils, "diffusers.pipelines.ltx2.pipeline_ltx2_condition": condition}


class RegistryTests(unittest.TestCase):
    def test_the_rows_and_their_capabilities(self) -> None:
        self.assertEqual(set(LTX2_DIFFUSERS_REPOS), {DISTILLED_NF4, "diffusers/LTX-2.3-Distilled-Diffusers", DEV_BF16, "Lightricks/LTX-2.5-Diffusers"})
        self.assertTrue(is_ltx2_distilled_repo(DISTILLED_NF4))
        self.assertFalse(is_ltx2_distilled_repo(DEV_BF16))
        for repo in LTX2_DIFFUSERS_REPOS:
            self.assertEqual(image_input_support(repo), (True, True))  # start and end pictures
            self.assertIn(repo, IMAGE_TO_VIDEO_REGISTRY)

    def test_the_catalog_rows_match_the_registry(self) -> None:
        from backend_service.catalog.video_models import VIDEO_MODEL_FAMILIES
        from backend_service.video_runtime.repos import _VIDEO_PIPELINE_DEFAULTS

        rows = {v["repo"]: v for f in VIDEO_MODEL_FAMILIES for v in f["variants"] if f["id"] in {"ltx-2-diffusers", "ltx-2-5-diffusers"}}
        self.assertEqual(set(rows), set(LTX2_DIFFUSERS_REPOS))
        for repo, row in rows.items():
            self.assertTrue(row["cudaOnly"], repo)  # the Mac has its own LTX-2 engine
            self.assertEqual(row["taskSupport"], ["txt2video", "img2video"])
            self.assertEqual(_VIDEO_PIPELINE_DEFAULTS[repo]["scheduler"], None)
        self.assertEqual(_VIDEO_PIPELINE_DEFAULTS[DISTILLED_NF4], {"steps": 8, "guidance": 1.0, "scheduler": None})
        self.assertEqual(_VIDEO_PIPELINE_DEFAULTS[DEV_BF16]["steps"], 30)
        # the big rows say what loading needs (the repo's fp32 text encoder doubles its size on disk)
        self.assertLess(rows[DEV_BF16]["loadMemoryGb"], rows[DEV_BF16]["sizeGb"])
        self.assertLess(rows[DISTILLED_NF4]["runtimeFootprintGb"], 32.0)  # a 4-bit pack for 24 GB cards

    def test_the_catalog_payload_flags_the_distilled_rows_as_fixed_schedule(self) -> None:
        from backend_service.helpers.video import _video_model_payloads

        with mock.patch("backend_service.helpers.video.is_apple_silicon", return_value=False):
            families = _video_model_payloads([])
        rows = {v["repo"]: v for f in families for v in f["variants"]}
        self.assertTrue(rows[DISTILLED_NF4]["fixedSampler"])
        self.assertTrue(rows["diffusers/LTX-2.3-Distilled-Diffusers"]["fixedSampler"])
        self.assertFalse(rows[DEV_BF16]["fixedSampler"])
        self.assertFalse(rows["Wan-AI/Wan2.2-TI2V-5B-Diffusers"]["fixedSampler"])
        for repo in (DISTILLED_NF4, DEV_BF16):
            self.assertTrue(rows[repo]["acceptsImageInput"])
            self.assertTrue(rows[repo]["acceptsEndImage"])

    def test_the_picture_mode_is_part_of_the_loaded_pipeline_key(self) -> None:
        engine = DiffusersVideoEngine()
        keys = {
            "text": engine._compute_variant_key(_config()),
            "start": engine._compute_variant_key(_config(start="s.png")),
            "end": engine._compute_variant_key(_config(start="s.png", end="e.png")),
        }
        self.assertEqual(len(set(keys.values())), 3, keys)
        self.assertIn("i2v", keys["start"])
        self.assertIn("cond", keys["end"])
        self.assertTrue(engine._uses_end_image(_config(end="e.png")))
        self.assertFalse(engine._uses_end_image(_config(start="s.png")))
        # Wan 2.2 TI2V-5B has no end-picture class: an end picture does not change what is loaded
        ti2v = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        self.assertFalse(engine._uses_end_image(_config(ti2v, start="s.png", end="e.png")))


class ClassChoiceTests(unittest.TestCase):
    def test_text_start_and_end_modes(self) -> None:
        self.assertEqual(pipeline_class_for_repo(DISTILLED_NF4).__name__, "LTX2Pipeline")
        self.assertEqual(pipeline_class_for_repo(DISTILLED_NF4, image_input=True).__name__, "LTX2ImageToVideoPipeline")
        self.assertEqual(pipeline_class_for_repo(DISTILLED_NF4, image_input=True, end_image=True).__name__, "LTX2ConditionPipeline")
        self.assertEqual(pipeline_class_for_repo(DISTILLED_NF4, end_image=True).__name__, "LTX2ConditionPipeline")  # an end alone
        # a model with no condition class keeps its image class when an end picture is asked for
        self.assertEqual(
            pipeline_class_for_repo("Wan-AI/Wan2.2-TI2V-5B-Diffusers", image_input=True, end_image=True).__name__,
            "WanImageToVideoPipeline",
        )


class ArgumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        patcher = mock.patch.dict(sys.modules, _fake_diffusers_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        # no libx264 here: the pictures stay as they are, which is what these tests want
        recompress = mock.patch("backend_service.video_runtime.pipeline_helpers.recompress_still", side_effect=lambda p, crf: p)
        self.recompress = recompress.start()
        self.addCleanup(recompress.stop)

    def test_the_distilled_model_runs_its_fixed_schedule_without_guidance(self) -> None:
        kwargs = build_pipeline_kwargs(_config(DISTILLED_NF4), None, _pipeline("LTX2Pipeline"))
        self.assertEqual(kwargs["num_inference_steps"], 8)
        self.assertEqual(kwargs["guidance_scale"], 1.0)
        self.assertEqual(len(kwargs["sigmas"]), 9)
        self.assertNotIn("negative_prompt", kwargs)
        self.assertEqual(kwargs["frame_rate"], 24.0)
        self.assertEqual(kwargs["image_crf"], 0)  # the pictures are compressed here, not by the pipeline (PyAV)
        self.assertFalse(kwargs["__cfg_decay"])

    def test_the_dev_model_keeps_the_requested_steps_and_gets_the_default_negative_prompt(self) -> None:
        kwargs = build_pipeline_kwargs(_config(DEV_BF16), None, _pipeline("LTX2Pipeline"))
        self.assertEqual(kwargs["num_inference_steps"], 50)
        self.assertNotIn("sigmas", kwargs)
        self.assertEqual(kwargs["negative_prompt"], "blurry, distorted")
        own = build_pipeline_kwargs(VideoGenerationConfig(**{**_config(DEV_BF16).__dict__, "negativePrompt": "no cats"}), None,
                                    _pipeline("LTX2Pipeline"))
        self.assertEqual(own["negative_prompt"], "no cats")

    def test_a_start_picture_goes_to_the_image_pipeline(self) -> None:
        still = _png(self.dir / "s.png")
        kwargs = build_pipeline_kwargs(_config(start=str(still)), None, _pipeline("LTX2ImageToVideoPipeline"))
        self.assertEqual(kwargs["image"].size, (768, 512))
        with self.assertRaises(RuntimeError):
            build_pipeline_kwargs(_config(), None, _pipeline("LTX2ImageToVideoPipeline"))

    def test_start_and_end_pictures_become_conditions_at_the_first_and_last_latent_frame(self) -> None:
        start, end = _png(self.dir / "s.png"), _png(self.dir / "e.png", (30, 30, 200))
        kwargs = build_pipeline_kwargs(_config(start=str(start), end=str(end)), None, _pipeline("LTX2ConditionPipeline"))
        first, last = kwargs["conditions"]
        self.assertEqual((first.index, last.index), (0, -1))  # a latent index: -1 is the last, as in diffusers' example
        self.assertEqual((first.strength, last.strength), (1.0, 1.0))
        self.assertEqual((first.crf, last.crf), (0, 0))
        only_end = build_pipeline_kwargs(_config(end=str(end)), None, _pipeline("LTX2ConditionPipeline"))
        self.assertEqual([c.index for c in only_end["conditions"]], [-1])
        with self.assertRaises(RuntimeError):
            build_pipeline_kwargs(_config(), None, _pipeline("LTX2ConditionPipeline"))

    def test_the_pictures_are_compressed_before_they_go_in(self) -> None:
        still = _png(self.dir / "s.png")
        build_pipeline_kwargs(_config(start=str(still)), None, _pipeline("LTX2ImageToVideoPipeline"))
        self.assertEqual(self.recompress.call_args.args[1], 33)


class RecompressTests(unittest.TestCase):
    def test_no_compression_asked_for_means_the_same_picture(self) -> None:
        from PIL import Image

        picture = Image.new("RGB", (64, 64), (10, 20, 30))
        self.assertIs(recompress_still(picture, 0), picture)

    def test_without_ffmpeg_the_picture_is_left_alone(self) -> None:
        from PIL import Image

        picture = Image.new("RGB", (64, 64), (10, 20, 30))
        with mock.patch("backend_service.storyboard.ffmpeg_bin.find_ffmpeg", return_value=None):
            self.assertIs(recompress_still(picture, 33), picture)

    def test_with_libx264_the_picture_comes_back_the_same_size(self) -> None:
        from PIL import Image

        from backend_service.storyboard import ffmpeg_bin

        found = ffmpeg_bin.find_ffmpeg(allow_path=True)
        try:
            usable = found is not None and ffmpeg_bin.h264_encoder() == "libx264"
        except RuntimeError:
            usable = False
        if not usable:
            self.skipTest("no libx264 ffmpeg here")
        picture = Image.new("RGB", (128, 96), (200, 40, 40))
        out = recompress_still(picture, 33)
        self.assertEqual(out.size, (128, 96))
        self.assertEqual(out.mode, "RGB")
        red, green, blue = out.getpixel((64, 48))
        self.assertGreater(red, 150)
        self.assertLess(green, 100)


class SoundTests(unittest.TestCase):
    def test_waveform_shapes_become_interleaved_16_bit_samples(self) -> None:
        import numpy as np

        stereo = np.stack([np.full(4, 0.5, dtype=np.float32), np.full(4, -0.5, dtype=np.float32)])  # (channels, samples)
        pcm, channels = audio_mux.waveform_to_pcm16(stereo[None])  # (batch, channels, samples)
        self.assertEqual(channels, 2)
        samples = np.frombuffer(pcm, dtype="<i2")
        self.assertEqual(len(samples), 8)
        self.assertEqual(list(samples[:2]), [16383, -16383])  # L R L R ...
        pcm_t, channels_t = audio_mux.waveform_to_pcm16(stereo.T)  # (samples, channels)
        self.assertEqual((pcm_t, channels_t), (pcm, 2))
        mono, mono_channels = audio_mux.waveform_to_pcm16(np.zeros(5, dtype=np.float32))
        self.assertEqual((len(mono), mono_channels), (10, 1))
        loud, _ = audio_mux.waveform_to_pcm16(np.array([[5.0, -5.0]], dtype=np.float32))
        self.assertEqual(list(np.frombuffer(loud, dtype="<i2")), [32767, -32767])  # clipped, not wrapped

    def test_a_torch_tensor_is_accepted(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        pcm, channels = audio_mux.waveform_to_pcm16(torch.zeros(1, 2, 3))
        self.assertEqual((len(pcm), channels), (12, 2))

    def test_the_sound_lands_in_the_mp4_as_an_audio_track(self) -> None:
        import numpy as np

        from backend_service.helpers.video_audio import mp4_has_audio_bytes
        from backend_service.storyboard import ffmpeg_bin

        found = ffmpeg_bin.find_ffmpeg(allow_path=True)
        if found is None:
            self.skipTest("no ffmpeg here")
        with tempfile.TemporaryDirectory() as tmp:
            silent = Path(tmp) / "silent.mp4"
            made = subprocess.run([found[0], "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                                   "color=c=red:s=64x64:d=1:r=24", "-pix_fmt", "yuv420p", str(silent)], capture_output=True)
            if made.returncode != 0:
                self.skipTest("this ffmpeg cannot make a test clip")
            silent_bytes = silent.read_bytes()
            tone = np.sin(np.linspace(0, 440 * 2 * np.pi, 48000)).astype(np.float32)
            result = audio_mux.mux_audio_into_mp4(silent_bytes, np.stack([tone, tone]), 48000)
        self.assertFalse(mp4_has_audio_bytes(silent_bytes))
        self.assertTrue(mp4_has_audio_bytes(result))

    def test_a_missing_ffmpeg_or_empty_sound_is_a_readable_error(self) -> None:
        import numpy as np

        with mock.patch("backend_service.storyboard.ffmpeg_bin.find_ffmpeg", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg was not found"):
                audio_mux.mux_audio_into_mp4(b"x", np.zeros((2, 10), dtype=np.float32), 48000)
        with mock.patch("backend_service.storyboard.ffmpeg_bin.find_ffmpeg", return_value=("ffmpeg", "path")):
            with self.assertRaisesRegex(RuntimeError, "no usable sound"):
                audio_mux.mux_audio_into_mp4(b"x", np.zeros((2, 0), dtype=np.float32), 48000)

    def test_invoke_pipeline_carries_the_sound_on_the_frame_list(self) -> None:
        audio = object()
        result = SimpleNamespace(frames=[["f1", "f2"]], audio=audio)
        pipeline = mock.Mock(return_value=result)
        pipeline.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48000))
        frames = invoke_pipeline(pipeline, {"num_inference_steps": 8, "guidance_scale": 1.0})
        self.assertIsInstance(frames, FrameList)
        self.assertEqual(list(frames), ["f1", "f2"])
        self.assertIs(frames.audio, audio)
        self.assertEqual(frames.audio_sample_rate, 48000)
        silent = invoke_pipeline(mock.Mock(return_value=SimpleNamespace(frames=[["f"]])), {"num_inference_steps": 4})
        self.assertIsNone(silent.audio)


class GenerateWithSoundTests(unittest.TestCase):
    def _engine(self):
        engine = DiffusersVideoEngine()
        shim = SimpleNamespace(
            Generator=mock.MagicMock(return_value=mock.MagicMock(manual_seed=lambda s: None)),
            cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        )
        engine._torch = shim  # type: ignore[assignment]
        engine._device = "cpu"
        return engine

    def _run(self, engine, frames, mux):
        from backend_service import video_runtime

        with mock.patch.object(video_runtime, "_find_missing", return_value=[]), \
                mock.patch.object(engine, "_ensure_pipeline", return_value=mock.MagicMock()), \
                mock.patch.object(engine, "_invoke_pipeline", return_value=frames), \
                mock.patch.object(engine, "_encode_frames_to_mp4", return_value=b"silent-mp4"), \
                mock.patch("backend_service.video_runtime.audio_mux.mux_audio_into_mp4", side_effect=mux):
            return engine.generate(_config())

    def test_a_clip_with_sound_is_muxed_and_a_silent_one_is_left_alone(self) -> None:
        frames = FrameList([mock.MagicMock() for _ in range(9)])
        frames.audio, frames.audio_sample_rate = object(), 48000
        calls = []
        result = self._run(self._engine(), frames, lambda mp4, audio, rate: calls.append((mp4, rate)) or b"muxed-mp4")
        self.assertEqual(result.bytes, b"muxed-mp4")
        self.assertEqual(calls, [(b"silent-mp4", 48000)])
        calls.clear()
        silent = self._run(self._engine(), FrameList([mock.MagicMock() for _ in range(9)]), lambda *a: calls.append(a) or b"x")
        self.assertEqual(silent.bytes, b"silent-mp4")
        self.assertEqual(calls, [])

    def test_a_failed_mux_keeps_the_silent_clip_and_says_so(self) -> None:
        frames = FrameList([mock.MagicMock() for _ in range(9)])
        frames.audio, frames.audio_sample_rate = object(), 48000

        def broken(*args):
            raise RuntimeError("ffmpeg was not found, so the sound could not be added to the clip.")

        result = self._run(self._engine(), frames, broken)
        self.assertEqual(result.bytes, b"silent-mp4")
        self.assertIn("The clip is silent", result.runtimeNote or "")


if __name__ == "__main__":
    unittest.main()
