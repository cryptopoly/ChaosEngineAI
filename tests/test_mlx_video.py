"""Tests for mlx-video Apple Silicon runtime (FU-009 LTX-2 path).

Covers:
- Platform gating (Darwin arm64 only).
- Install-state probe (``missingDependencies`` when ``mlx_video`` not importable).
- Repo routing helper + supported-repo set (LTX-2 prince-canuma variants).
- Preload/unload bookkeeping.
- ``generate()`` builds the right ``python -m mlx_video.ltx_2.generate`` CLI
  and surfaces the rendered mp4 as ``GeneratedVideo``.
- Manager dispatches LTX-2 repos to mlx-video before falling through to
  diffusers.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend_service.mlx_video_runtime import (
    MlxVideoEngine,
    _SUPPORTED_REPOS,
    _is_mlx_video_repo,
    _ltx2_generation_needs_spatial_upscaler,
    _parse_step_fraction,
    _resolve_entry_point,
    supported_repos,
)
from backend_service.video_runtime import (
    VideoGenerationConfig,
    VideoRuntimeManager,
)


def _make_config(repo: str = "prince-canuma/LTX-2-distilled") -> VideoGenerationConfig:
    return VideoGenerationConfig(
        modelId="mlx-video-test",
        modelName="test",
        repo=repo,
        prompt="a cat surfing",
        negativePrompt="",
        width=512,
        height=512,
        numFrames=24,
        fps=16,
        guidance=5.0,
        steps=20,
        seed=42,
    )


class MlxVideoSupportedReposTests(unittest.TestCase):
    def test_supported_repos_snapshot(self):
        repos = supported_repos()
        self.assertIn("prince-canuma/LTX-2-distilled", repos)
        self.assertIn("prince-canuma/LTX-2-dev", repos)
        self.assertIn("prince-canuma/LTX-2.3-distilled", repos)
        self.assertIn("prince-canuma/LTX-2.3-dev", repos)

    def test_supported_repos_excludes_wan(self):
        # Wan paths still need a custom convert step — must not advertise
        # them as mlx-video supported until that's bundled.
        for repo in (
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ):
            self.assertNotIn(repo, supported_repos())

    def test_supported_repos_is_frozen(self):
        self.assertIsInstance(supported_repos(), frozenset)
        self.assertEqual(supported_repos(), _SUPPORTED_REPOS)

    def test_is_mlx_video_repo_matches_set(self):
        for repo in _SUPPORTED_REPOS:
            self.assertTrue(_is_mlx_video_repo(repo))

    def test_is_mlx_video_repo_rejects_unrelated(self):
        self.assertFalse(_is_mlx_video_repo("genmo/mochi-1-preview"))
        self.assertFalse(_is_mlx_video_repo("THUDM/CogVideoX-5b"))
        self.assertFalse(_is_mlx_video_repo("NVlabs/LongLive-1.3B"))
        self.assertFalse(_is_mlx_video_repo("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))
        self.assertFalse(_is_mlx_video_repo(None))
        self.assertFalse(_is_mlx_video_repo(""))


class MlxVideoEntryPointTests(unittest.TestCase):
    def test_resolve_entry_point_for_ltx2(self):
        # Real module path under ``mlx_video.models.ltx_2.generate`` —
        # the ``mlx_video.ltx_2.generate`` name is a console-script
        # alias in mlx-video's pyproject, not an importable module.
        self.assertEqual(
            _resolve_entry_point("prince-canuma/LTX-2-distilled"),
            "mlx_video.models.ltx_2.generate",
        )
        self.assertEqual(
            _resolve_entry_point("prince-canuma/LTX-2.3-dev"),
            "mlx_video.models.ltx_2.generate",
        )

    def test_resolve_entry_point_unknown_raises(self):
        with self.assertRaises(RuntimeError):
            _resolve_entry_point("genmo/mochi-1-preview")


class MlxVideoProbeTests(unittest.TestCase):
    def test_probe_unavailable_on_non_darwin(self):
        engine = MlxVideoEngine()
        with patch("backend_service.mlx_video_runtime.platform.system", return_value="Linux"):
            status = engine.probe()
        self.assertFalse(status.realGenerationAvailable)
        self.assertEqual(status.activeEngine, "mlx-video")
        self.assertIn("Apple Silicon only", status.message)
        self.assertIsNone(status.device)

    def test_probe_unavailable_on_intel_mac(self):
        engine = MlxVideoEngine()
        with patch("backend_service.mlx_video_runtime.platform.system", return_value="Darwin"), \
             patch("backend_service.mlx_video_runtime.platform.machine", return_value="x86_64"):
            status = engine.probe()
        self.assertFalse(status.realGenerationAvailable)
        self.assertIn("Apple Silicon only", status.message)

    def test_probe_reports_missing_package_on_apple_silicon(self):
        engine = MlxVideoEngine()
        with patch("backend_service.mlx_video_runtime.platform.system", return_value="Darwin"), \
             patch("backend_service.mlx_video_runtime.platform.machine", return_value="arm64"), \
             patch("backend_service.mlx_video_runtime.importlib.util.find_spec", return_value=None):
            status = engine.probe()
        self.assertFalse(status.realGenerationAvailable)
        self.assertIn("mlx-video", status.missingDependencies)
        self.assertEqual(status.expectedDevice, "mps")
        self.assertIn("not installed", status.message)

    def test_probe_reports_ready_when_installed(self):
        engine = MlxVideoEngine()
        fake_spec = object()
        with patch("backend_service.mlx_video_runtime.platform.system", return_value="Darwin"), \
             patch("backend_service.mlx_video_runtime.platform.machine", return_value="arm64"), \
             patch("backend_service.mlx_video_runtime.importlib.util.find_spec", return_value=fake_spec):
            status = engine.probe()
        self.assertTrue(status.realGenerationAvailable)
        self.assertEqual(status.device, "mps")
        self.assertEqual(status.expectedDevice, "mps")
        self.assertIn("LTX-2", status.message)


def _patch_apple_silicon_with_mlx_video():
    return [
        patch("backend_service.mlx_video_runtime.platform.system", return_value="Darwin"),
        patch("backend_service.mlx_video_runtime.platform.machine", return_value="arm64"),
        patch("backend_service.mlx_video_runtime.importlib.util.find_spec", return_value=object()),
    ]


class MlxVideoLifecycleTests(unittest.TestCase):
    def test_preload_rejects_unsupported_repo(self):
        engine = MlxVideoEngine()
        with self.assertRaises(RuntimeError) as ctx:
            engine.preload("genmo/mochi-1-preview")
        self.assertIn("does not support", str(ctx.exception))

    def test_preload_remembers_supported_repo(self):
        engine = MlxVideoEngine()
        for p in _patch_apple_silicon_with_mlx_video():
            p.start()
        self.addCleanup(patch.stopall)
        status = engine.preload("prince-canuma/LTX-2-distilled")
        self.assertEqual(status.loadedModelRepo, "prince-canuma/LTX-2-distilled")

    def test_unload_clears_matching_repo(self):
        engine = MlxVideoEngine()
        for p in _patch_apple_silicon_with_mlx_video():
            p.start()
        self.addCleanup(patch.stopall)
        engine.preload("prince-canuma/LTX-2-distilled")
        status = engine.unload("prince-canuma/LTX-2-distilled")
        self.assertIsNone(status.loadedModelRepo)

    def test_unload_noop_when_repo_mismatch(self):
        engine = MlxVideoEngine()
        for p in _patch_apple_silicon_with_mlx_video():
            p.start()
        self.addCleanup(patch.stopall)
        engine.preload("prince-canuma/LTX-2-distilled")
        status = engine.unload("prince-canuma/LTX-2-dev")
        self.assertEqual(status.loadedModelRepo, "prince-canuma/LTX-2-distilled")


class MlxVideoGenerateCmdTests(unittest.TestCase):
    def test_build_cmd_shape_for_ltx2_distilled(self):
        engine = MlxVideoEngine()
        config = _make_config("prince-canuma/LTX-2-distilled")
        cmd = engine._build_cmd(config, Path("/tmp/out.mp4"))
        self.assertIn("-m", cmd)
        # Real module path — ``mlx_video.ltx_2.generate`` is a console-
        # script alias, not an importable module.
        self.assertIn("mlx_video.models.ltx_2.generate", cmd)
        # mlx-video CLI uses ``--model-repo`` (not ``--model``).
        self.assertIn("--model-repo", cmd)
        self.assertIn("prince-canuma/LTX-2-distilled", cmd)
        # Distilled repo → ``--pipeline distilled`` (fastest path).
        self.assertIn("--pipeline", cmd)
        self.assertIn("distilled", cmd)
        self.assertIn("--prompt", cmd)
        self.assertIn("a cat surfing", cmd)
        self.assertIn("--num-frames", cmd)
        self.assertIn("24", cmd)
        self.assertIn("--fps", cmd)
        self.assertIn("16", cmd)
        self.assertIn("--height", cmd)
        self.assertIn("512", cmd)
        self.assertIn("--width", cmd)
        self.assertIn("--steps", cmd)
        self.assertIn("20", cmd)
        # ``--cfg-scale`` (not ``--guidance``).
        self.assertIn("--cfg-scale", cmd)
        self.assertIn("5.0", cmd)
        # ``--output-path`` (not ``--output``).
        self.assertIn("--output-path", cmd)
        self.assertIn("/tmp/out.mp4", cmd)
        self.assertIn("--seed", cmd)
        self.assertIn("42", cmd)
        # STG (Spatial-Temporal Guidance) — quality lever default-on.
        self.assertIn("--stg-scale", cmd)
        self.assertIn("1.0", cmd)

    def test_build_cmd_picks_dev_pipeline_for_dev_repo(self):
        engine = MlxVideoEngine()
        config = _make_config("prince-canuma/LTX-2-dev")
        cmd = engine._build_cmd(config, Path("/tmp/out.mp4"))
        # Dev repo → ``--pipeline dev`` (higher quality, single-stage).
        self.assertIn("--pipeline", cmd)
        # Find the value after --pipeline.
        idx = cmd.index("--pipeline")
        self.assertEqual(cmd[idx + 1], "dev")

    def test_build_cmd_picks_dev_pipeline_for_ltx2_3_dev(self):
        engine = MlxVideoEngine()
        config = _make_config("prince-canuma/LTX-2.3-dev")
        cmd = engine._build_cmd(config, Path("/tmp/out.mp4"))
        idx = cmd.index("--pipeline")
        self.assertEqual(cmd[idx + 1], "dev")

    def test_build_cmd_omits_seed_when_none(self):
        engine = MlxVideoEngine()
        config = _make_config()
        config = VideoGenerationConfig(
            modelId=config.modelId,
            modelName=config.modelName,
            repo=config.repo,
            prompt=config.prompt,
            negativePrompt=config.negativePrompt,
            width=config.width,
            height=config.height,
            numFrames=config.numFrames,
            fps=config.fps,
            guidance=config.guidance,
            steps=config.steps,
            seed=None,
        )
        cmd = engine._build_cmd(config, Path("/tmp/out.mp4"))
        self.assertNotIn("--seed", cmd)

    def test_distilled_pipeline_needs_spatial_upscaler(self):
        self.assertTrue(_ltx2_generation_needs_spatial_upscaler("prince-canuma/LTX-2-distilled"))
        self.assertTrue(_ltx2_generation_needs_spatial_upscaler("prince-canuma/LTX-2.3-distilled"))
        self.assertFalse(_ltx2_generation_needs_spatial_upscaler("prince-canuma/LTX-2-dev"))

    def test_build_cmd_can_pin_resolved_spatial_upscaler(self):
        engine = MlxVideoEngine()
        config = _make_config("prince-canuma/LTX-2-distilled")
        with patch(
            "backend_service.mlx_video_runtime._resolve_ltx2_spatial_upscaler",
            return_value=Path("/tmp/ltx-2-spatial-upscaler-x2-1.0.safetensors"),
        ) as resolve_upscaler:
            cmd = engine._build_cmd(
                config,
                Path("/tmp/out.mp4"),
                resolve_aux_files=True,
            )
        resolve_upscaler.assert_called_once_with(
            "prince-canuma/LTX-2-distilled",
            allow_download=True,
        )
        self.assertIn("--spatial-upscaler", cmd)
        idx = cmd.index("--spatial-upscaler")
        self.assertEqual(
            cmd[idx + 1],
            "/tmp/ltx-2-spatial-upscaler-x2-1.0.safetensors",
        )


class MlxVideoGenerateE2ETests(unittest.TestCase):
    def test_generate_writes_output_and_returns_video(self):
        engine = MlxVideoEngine()

        def fake_launch(cmd, workspace, on_progress):
            (workspace / "out.mp4").write_bytes(b"\x00\x01fakeMP4")

        with patch.object(engine, "_launch", side_effect=fake_launch), \
             patch.object(engine, "probe") as mock_probe, \
             patch("backend_service.mlx_video_runtime.time.monotonic", side_effect=[100.0, 102.5]), \
             patch(
                 "backend_service.mlx_video_runtime._resolve_ltx2_spatial_upscaler",
                 return_value=Path("/tmp/ltx-2-spatial-upscaler-x2-1.0.safetensors"),
             ):
            mock_probe.return_value = MagicMock(
                realGenerationAvailable=True,
                message="ready",
                to_dict=lambda: {"activeEngine": "mlx-video"},
            )
            video = engine.generate(_make_config())

        self.assertEqual(video.bytes, b"\x00\x01fakeMP4")
        self.assertEqual(video.seed, 42)
        self.assertEqual(video.extension, "mp4")
        self.assertEqual(video.durationSeconds, 2.5)
        self.assertEqual(video.fps, 16)
        self.assertEqual(video.frameCount, 24)
        self.assertEqual(video.runtimeLabel, "mlx-video (MLX native)")
        self.assertIn("fixed 8+3", video.runtimeNote or "")
        self.assertEqual(video.effectiveSteps, 11)
        self.assertEqual(video.effectiveGuidance, 1.0)

    def test_generate_resolves_random_seed_before_launch(self):
        engine = MlxVideoEngine()
        config = _make_config()
        config = VideoGenerationConfig(
            modelId=config.modelId,
            modelName=config.modelName,
            repo=config.repo,
            prompt=config.prompt,
            negativePrompt=config.negativePrompt,
            width=config.width,
            height=config.height,
            numFrames=config.numFrames,
            fps=config.fps,
            guidance=config.guidance,
            steps=config.steps,
            seed=None,
        )
        launched_cmd: list[str] = []

        def fake_launch(cmd, workspace, on_progress):
            launched_cmd[:] = cmd
            (workspace / "out.mp4").write_bytes(b"\x00\x01fakeMP4")

        with patch.object(engine, "_launch", side_effect=fake_launch), \
             patch.object(engine, "probe") as mock_probe, \
             patch("backend_service.mlx_video_runtime._resolve_video_seed", return_value=1234), \
             patch("backend_service.mlx_video_runtime.time.monotonic", side_effect=[10.0, 11.0]), \
             patch(
                 "backend_service.mlx_video_runtime._resolve_ltx2_spatial_upscaler",
                 return_value=Path("/tmp/ltx-2-spatial-upscaler-x2-1.0.safetensors"),
             ):
            mock_probe.return_value = MagicMock(
                realGenerationAvailable=True,
                message="ready",
                to_dict=lambda: {"activeEngine": "mlx-video"},
            )
            video = engine.generate(config)

        self.assertEqual(video.seed, 1234)
        self.assertIn("--seed", launched_cmd)
        self.assertEqual(launched_cmd[launched_cmd.index("--seed") + 1], "1234")

    def test_generate_raises_when_no_output_produced(self):
        engine = MlxVideoEngine()

        def fake_launch(cmd, workspace, on_progress):
            return None  # no mp4 produced

        with patch.object(engine, "_launch", side_effect=fake_launch), \
             patch.object(engine, "probe") as mock_probe, \
             patch(
                 "backend_service.mlx_video_runtime._resolve_ltx2_spatial_upscaler",
                 return_value=Path("/tmp/ltx-2-spatial-upscaler-x2-1.0.safetensors"),
             ):
            mock_probe.return_value = MagicMock(
                realGenerationAvailable=True,
                message="ready",
                to_dict=lambda: {"activeEngine": "mlx-video"},
            )
            with self.assertRaises(RuntimeError) as ctx:
                engine.generate(_make_config())
            self.assertIn("no mp4", str(ctx.exception))

    def test_generate_raises_when_not_ready(self):
        engine = MlxVideoEngine()
        with patch.object(engine, "probe") as mock_probe:
            mock_probe.return_value = MagicMock(
                realGenerationAvailable=False,
                message="install mlx-video",
            )
            with self.assertRaises(RuntimeError) as ctx:
                engine.generate(_make_config())
            self.assertIn("install mlx-video", str(ctx.exception))


class MlxVideoStepFractionTests(unittest.TestCase):
    def test_parses_step_progress_line(self):
        self.assertAlmostEqual(_parse_step_fraction("step 12/30"), 0.4)
        self.assertAlmostEqual(_parse_step_fraction("  step  5 / 10  "), 0.5)
        self.assertEqual(_parse_step_fraction("step 30/30"), 1.0)

    def test_returns_none_for_non_step_lines(self):
        self.assertIsNone(_parse_step_fraction("loading weights"))
        self.assertIsNone(_parse_step_fraction(""))

    def test_clamps_invalid_total_to_none(self):
        self.assertIsNone(_parse_step_fraction("step 5/0"))


class VideoManagerExposesMlxVideoTests(unittest.TestCase):
    def test_manager_exposes_mlx_video_capabilities(self):
        manager = VideoRuntimeManager()
        caps = manager.mlx_video_capabilities()
        self.assertIsInstance(caps, dict)
        self.assertEqual(caps["activeEngine"], "mlx-video")

    def test_manager_lazy_constructs_mlx_video(self):
        manager = VideoRuntimeManager()
        self.assertIsNone(manager._mlx_video)
        manager.mlx_video_capabilities()
        self.assertIsNotNone(manager._mlx_video)

    def test_manager_routes_ltx2_to_mlx_video_when_ready(self):
        manager = VideoRuntimeManager()
        config = _make_config("prince-canuma/LTX-2-distilled")

        fake_video = MagicMock()
        fake_engine = MagicMock()
        fake_engine.probe.return_value = MagicMock(
            realGenerationAvailable=True,
            to_dict=lambda: {"activeEngine": "mlx-video"},
        )
        fake_engine.generate.return_value = fake_video
        manager._mlx_video = fake_engine

        # Ensure diffusers engine would NOT be hit.
        manager._engine = MagicMock()

        video, runtime = manager.generate(config)

        self.assertIs(video, fake_video)
        fake_engine.generate.assert_called_once_with(config)
        manager._engine.generate.assert_not_called()
        self.assertEqual(runtime["activeEngine"], "mlx-video")

    def test_manager_falls_back_to_diffusers_when_mlx_video_unavailable(self):
        manager = VideoRuntimeManager()
        config = _make_config("prince-canuma/LTX-2-distilled")

        unavailable_mlx = MagicMock()
        unavailable_mlx.probe.return_value = MagicMock(
            realGenerationAvailable=False,
            message="mlx-video not installed",
        )
        manager._mlx_video = unavailable_mlx

        diffusers_video = MagicMock()
        diffusers_engine = MagicMock()
        diffusers_engine.probe.return_value = MagicMock(
            realGenerationAvailable=True,
            to_dict=lambda: {"activeEngine": "diffusers"},
        )
        diffusers_engine.generate.return_value = diffusers_video
        manager._engine = diffusers_engine

        video, runtime = manager.generate(config)
        self.assertIs(video, diffusers_video)
        self.assertEqual(runtime["activeEngine"], "diffusers")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
