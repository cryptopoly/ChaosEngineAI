"""Unit tests for ``backend_service.longlive_engine``.

LongLive uses a subprocess engine (torchrun) against an isolated venv, so
we can't exercise the real pipeline without CUDA + a 10+GB weight
download. Tests pin the surface logic: latent-frame math, YAML render,
install detection, and VideoRuntimeManager dispatch routing.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from backend_service import longlive_engine
from backend_service.longlive_engine import (
    DEFAULT_FRAMES_PER_BLOCK,
    LATENT_TO_PIXEL_FRAMES,
    LONGLIVE_OUTPUT_FPS,
    LongLiveEngine,
    LongLiveInstallInfo,
    _render_longlive_yaml,
    compute_latent_frames,
    pixel_frames_for_latents,
    resolve_install,
)
from backend_service.video_runtime import (
    VideoGenerationConfig,
    VideoRuntimeManager,
    _is_longlive_repo,
)


class FrameMathTests(unittest.TestCase):
    def test_compute_latent_frames_rounds_to_block(self):
        # 4 seconds @ 16 fps = 64 pixel frames → 16 latent frames → already
        # a multiple of 3? 16 % 3 == 1 → rounds up to 18.
        self.assertEqual(compute_latent_frames(4.0), 18)

    def test_compute_latent_frames_never_below_one_block(self):
        # Sub-second request still produces a valid block-aligned count.
        result = compute_latent_frames(0.1)
        self.assertGreaterEqual(result, DEFAULT_FRAMES_PER_BLOCK)
        self.assertEqual(result % DEFAULT_FRAMES_PER_BLOCK, 0)

    def test_pixel_frames_matches_vae_factor(self):
        self.assertEqual(pixel_frames_for_latents(30), 30 * LATENT_TO_PIXEL_FRAMES)

    def test_longlive_output_fps_is_16(self):
        # Guards against accidental drift in the VAE / fps mapping. If the
        # upstream LongLive model changes output FPS we'd need to re-derive
        # compute_latent_frames and the GeneratedVideo.fps constant.
        self.assertEqual(LONGLIVE_OUTPUT_FPS, 16)


class InstallResolutionTests(unittest.TestCase):
    def test_resolve_install_points_under_root(self):
        with TemporaryDirectory() as tmp:
            info = resolve_install(Path(tmp))
            self.assertEqual(info.root, Path(tmp))
            self.assertEqual(info.repo_dir, Path(tmp) / "repo")
            self.assertEqual(info.marker, Path(tmp) / "ready.marker")
            self.assertFalse(info.ready)

    def test_install_ready_requires_marker_repo_and_venv(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = resolve_install(root)
            # Still not ready — nothing exists yet.
            self.assertFalse(info.ready)
            info.repo_dir.mkdir(parents=True)
            info.venv_python.parent.mkdir(parents=True, exist_ok=True)
            info.venv_python.touch()
            info.marker.write_text("ok", encoding="utf-8")
            # Re-resolve so the filesystem check runs on populated paths.
            self.assertTrue(resolve_install(root).ready)


class YamlRenderTests(unittest.TestCase):
    def _make_install(self, root: Path) -> LongLiveInstallInfo:
        return resolve_install(root)

    def test_yaml_includes_block_count_and_user_overrides(self):
        with TemporaryDirectory() as tmp:
            install = self._make_install(Path(tmp))
            install.weights_dir.mkdir(parents=True, exist_ok=True)
            prompt = Path(tmp) / "prompt.txt"
            prompt.write_text("a dog on the beach", encoding="utf-8")
            yaml = _render_longlive_yaml(
                install=install,
                prompt_file=prompt,
                output_dir=Path(tmp) / "videos",
                num_output_frames=24,
                seed=42,
                infinite=False,
            )
            self.assertIn("num_output_frames: 24", yaml)
            self.assertIn("seed: 42", yaml)
            self.assertIn(str(prompt), yaml)
            self.assertIn("use_infinite_attention: false", yaml)
            self.assertIn("lora.pt", yaml)

    def test_yaml_enables_infinite_attention_when_requested(self):
        with TemporaryDirectory() as tmp:
            install = self._make_install(Path(tmp))
            install.weights_dir.mkdir(parents=True, exist_ok=True)
            prompt = Path(tmp) / "prompt.txt"
            prompt.write_text("x", encoding="utf-8")
            yaml = _render_longlive_yaml(
                install=install,
                prompt_file=prompt,
                output_dir=Path(tmp) / "videos",
                num_output_frames=300,
                seed=0,
                infinite=True,
            )
            self.assertIn("use_infinite_attention: true", yaml)


class ProbeTests(unittest.TestCase):
    def test_probe_reports_unsupported_on_darwin(self):
        engine = LongLiveEngine(install_root=Path("/tmp/nonexistent-longlive"))
        with mock.patch("platform.system", return_value="Darwin"):
            status = engine.probe()
        self.assertFalse(status.realGenerationAvailable)
        self.assertIn("macOS", status.message)

    def test_probe_reports_install_missing_on_linux(self):
        with TemporaryDirectory() as tmp:
            engine = LongLiveEngine(install_root=Path(tmp))
            with mock.patch("platform.system", return_value="Linux"), \
                 mock.patch.object(longlive_engine, "_cuda_available", return_value=True):
                status = engine.probe()
        self.assertFalse(status.realGenerationAvailable)
        self.assertIn("not installed", status.message.lower())

    def test_probe_reports_ready_when_install_and_cuda_present(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = resolve_install(root)
            info.repo_dir.mkdir(parents=True)
            info.venv_python.parent.mkdir(parents=True, exist_ok=True)
            info.venv_python.touch()
            info.marker.write_text("ok", encoding="utf-8")
            engine = LongLiveEngine(install_root=root)
            with mock.patch("platform.system", return_value="Linux"), \
                 mock.patch.object(longlive_engine, "_cuda_available", return_value=True):
                status = engine.probe()
        self.assertTrue(status.realGenerationAvailable)
        self.assertEqual(status.device, "cuda")


class DispatchRoutingTests(unittest.TestCase):
    def test_is_longlive_repo_matches_longlive_prefixes_only(self):
        self.assertTrue(_is_longlive_repo("NVlabs/LongLive-1.3B"))
        self.assertTrue(_is_longlive_repo("NVlabs/LongLive-Variant"))
        self.assertFalse(_is_longlive_repo("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))
        self.assertFalse(_is_longlive_repo(None))
        self.assertFalse(_is_longlive_repo(""))

    def test_manager_routes_longlive_repo_to_longlive_engine(self):
        manager = VideoRuntimeManager()

        fake_video = mock.Mock()
        fake_video.seed = 1
        fake_video.bytes = b"fake-mp4"
        fake_video.extension = "mp4"
        fake_video.mimeType = "video/mp4"
        fake_video.durationSeconds = 2.0
        fake_video.frameCount = 32
        fake_video.fps = 16
        fake_video.width = 832
        fake_video.height = 480
        fake_video.runtimeLabel = "LongLive"
        fake_video.runtimeNote = None

        fake_longlive = mock.Mock()
        fake_longlive.probe.return_value = mock.Mock(
            realGenerationAvailable=True,
            to_dict=lambda: {"activeEngine": "longlive"},
        )
        fake_longlive.generate.return_value = fake_video

        manager._longlive = fake_longlive  # type: ignore[attr-defined]

        config = VideoGenerationConfig(
            modelId="NVlabs/LongLive-1.3B",
            modelName="LongLive 1.3B",
            repo="NVlabs/LongLive-1.3B",
            prompt="hello",
            negativePrompt="",
            width=832,
            height=480,
            numFrames=32,
            fps=16,
            guidance=5.0,
        )
        video, runtime = manager.generate(config)
        self.assertIs(video, fake_video)
        self.assertEqual(runtime, {"activeEngine": "longlive"})
        fake_longlive.generate.assert_called_once()

    def test_manager_does_not_route_non_longlive_to_longlive_engine(self):
        manager = VideoRuntimeManager()
        fake_longlive = mock.Mock()
        manager._longlive = fake_longlive  # type: ignore[attr-defined]
        manager._engine = mock.Mock()  # type: ignore[attr-defined]
        manager._engine.probe.return_value = mock.Mock(
            realGenerationAvailable=False,
            message="not ready",
        )
        config = VideoGenerationConfig(
            modelId="Lightricks/LTX-Video",
            modelName="LTX-Video",
            repo="Lightricks/LTX-Video",
            prompt="hello",
            negativePrompt="",
            width=768,
            height=512,
            numFrames=17,
            fps=24,
            guidance=3.0,
        )
        with self.assertRaises(RuntimeError):
            manager.generate(config)
        fake_longlive.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
