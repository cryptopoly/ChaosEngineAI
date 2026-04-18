"""Unit tests for ``backend_service.video_runtime``.

We never actually load a video model in tests (those weights are 10-25GB).
The tests exercise the surface logic: probe dependency detection, pipeline
class registry routing, preload/unload lifecycle via monkey-patched seams.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from backend_service import video_runtime
from backend_service.video_runtime import (
    DiffusersVideoEngine,
    PIPELINE_REGISTRY,
    VideoGenerationConfig,
    VideoRuntimeManager,
    VideoRuntimeStatus,
)


class ProbeTests(unittest.TestCase):
    def test_probe_flags_missing_core_deps_as_unavailable(self):
        engine = DiffusersVideoEngine()
        # Simulate a machine with no diffusers/torch installed.
        with mock.patch.object(
            video_runtime,
            "_find_missing",
            side_effect=[["diffusers", "torch"], ["imageio", "imageio-ffmpeg"]],
        ):
            status = engine.probe()
        self.assertIsInstance(status, VideoRuntimeStatus)
        self.assertFalse(status.realGenerationAvailable)
        self.assertEqual(status.activeEngine, "placeholder")
        # All missing deps (core + output) surface in the list for a single
        # clear install hint.
        self.assertIn("diffusers", status.missingDependencies)
        self.assertIn("torch", status.missingDependencies)
        self.assertIn("imageio", status.missingDependencies)

    def test_probe_reports_ready_when_all_deps_and_torch_import_cleanly(self):
        engine = DiffusersVideoEngine()

        # Core deps present, output deps present.
        with mock.patch.object(video_runtime, "_find_missing", return_value=[]):
            status = engine.probe()

        self.assertTrue(status.realGenerationAvailable)
        self.assertEqual(status.activeEngine, "diffusers")
        self.assertIn(status.device, {"cuda", "mps", "cpu"})
        self.assertEqual(status.missingDependencies, [])

    def test_probe_reports_ready_but_warns_when_only_output_deps_missing(self):
        engine = DiffusersVideoEngine()

        with mock.patch.object(
            video_runtime,
            "_find_missing",
            # First call: core deps — all present.
            # Second call: output deps — imageio missing.
            side_effect=[[], ["imageio", "imageio-ffmpeg"]],
        ):
            status = engine.probe()

        self.assertTrue(status.realGenerationAvailable)
        self.assertEqual(status.activeEngine, "diffusers")
        self.assertIn("imageio", status.missingDependencies)
        self.assertIn("mp4", status.message.lower())


class PipelineRegistryTests(unittest.TestCase):
    def test_registry_covers_all_first_wave_engines(self):
        expected = {
            "Lightricks/LTX-Video",
            "genmo/mochi-1-preview",
            "Wan-AI/Wan2.1-T2V-1.3B",
            "Wan-AI/Wan2.1-T2V-14B",
            "Wan-AI/Wan2.2-T2V-A14B",
            "tencent/HunyuanVideo",
        }
        self.assertEqual(set(PIPELINE_REGISTRY.keys()), expected)
        for entry in PIPELINE_REGISTRY.values():
            self.assertIn("class_name", entry)
            self.assertEqual(entry["task"], "txt2video")

    def test_wan_variants_all_route_to_wanpipeline(self):
        """Wan 2.1 and 2.2 use the same pipeline class — version difference is in the weights."""
        wan_repos = [repo for repo in PIPELINE_REGISTRY if repo.startswith("Wan-AI/")]
        self.assertGreaterEqual(len(wan_repos), 3, "expected 1.3B, 14B, and A14B Wan entries")
        for repo in wan_repos:
            self.assertEqual(PIPELINE_REGISTRY[repo]["class_name"], "WanPipeline")

    def test_pipeline_class_raises_for_unknown_repo(self):
        engine = DiffusersVideoEngine()
        with self.assertRaises(RuntimeError) as ctx:
            engine._pipeline_class("not-a-real/repo")
        self.assertIn("not-a-real/repo", str(ctx.exception))

    def test_pipeline_class_resolves_known_repo(self):
        """Sanity-check: diffusers 0.37+ actually exposes the classes we registered."""
        engine = DiffusersVideoEngine()
        for repo, entry in PIPELINE_REGISTRY.items():
            try:
                pipeline_cls = engine._pipeline_class(repo)
            except RuntimeError:
                # Older diffusers — the runtime itself handles this case with a
                # helpful error. Skip individual pipeline class assertions on
                # older builds rather than failing the test suite.
                continue
            self.assertEqual(pipeline_cls.__name__, entry["class_name"])


class PreloadLifecycleTests(unittest.TestCase):
    def test_preload_reports_ready_status_after_ensure_pipeline(self):
        engine = DiffusersVideoEngine()
        stub_pipeline = mock.MagicMock()
        with mock.patch.object(engine, "_ensure_pipeline", return_value=stub_pipeline) as ensure:
            with mock.patch.object(video_runtime, "_find_missing", return_value=[]):
                status = engine.preload("Lightricks/LTX-Video")
        ensure.assert_called_once_with("Lightricks/LTX-Video")
        self.assertTrue(status.realGenerationAvailable)

    def test_unload_without_active_pipeline_is_noop(self):
        engine = DiffusersVideoEngine()
        with mock.patch.object(video_runtime, "_find_missing", return_value=[]):
            status = engine.unload()
        self.assertIsNone(status.loadedModelRepo)

    def test_unload_with_mismatched_repo_is_noop(self):
        engine = DiffusersVideoEngine()
        engine._pipeline = mock.MagicMock()
        engine._loaded_repo = "Lightricks/LTX-Video"
        engine._torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        engine._device = "cpu"
        with mock.patch.object(video_runtime, "_find_missing", return_value=[]):
            status = engine.unload("tencent/HunyuanVideo")
        # Engine should NOT have cleared state since the repo didn't match.
        self.assertEqual(status.loadedModelRepo, "Lightricks/LTX-Video")
        self.assertIsNotNone(engine._pipeline)

    def test_unload_clears_state_when_repo_matches(self):
        engine = DiffusersVideoEngine()
        engine._pipeline = mock.MagicMock()
        engine._loaded_repo = "Lightricks/LTX-Video"
        engine._torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            mps=None,
        )
        engine._device = "cpu"
        with mock.patch.object(video_runtime, "_find_missing", return_value=[]):
            status = engine.unload("Lightricks/LTX-Video")
        self.assertIsNone(status.loadedModelRepo)
        self.assertIsNone(engine._pipeline)
        self.assertIsNone(engine._loaded_repo)


class VideoRuntimeManagerTests(unittest.TestCase):
    def test_capabilities_delegates_to_engine_probe(self):
        manager = VideoRuntimeManager()
        with mock.patch.object(video_runtime, "_find_missing", return_value=[]):
            capabilities = manager.capabilities()
        self.assertIn("realGenerationAvailable", capabilities)
        self.assertIn("activeEngine", capabilities)

    def test_preload_raises_when_runtime_is_not_ready(self):
        manager = VideoRuntimeManager()
        with mock.patch.object(
            video_runtime,
            "_find_missing",
            side_effect=[["diffusers"], []],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                manager.preload("Lightricks/LTX-Video")
        self.assertIn("diffusers", str(ctx.exception).lower())


class GenerationConfigTests(unittest.TestCase):
    def test_config_is_frozen_dataclass_with_expected_fields(self):
        cfg = VideoGenerationConfig(
            modelId="Lightricks/LTX-Video",
            modelName="LTX-Video",
            repo="Lightricks/LTX-Video",
            prompt="a cat riding a bike",
            negativePrompt="",
            width=768,
            height=512,
            numFrames=97,
            fps=24,
            guidance=3.0,
            seed=42,
        )
        self.assertEqual(cfg.prompt, "a cat riding a bike")
        self.assertEqual(cfg.seed, 42)
        # Frozen dataclass: mutation should raise.
        with self.assertRaises(Exception):
            cfg.seed = 7  # type: ignore[misc]

    def test_engine_generate_raises_not_implemented(self):
        engine = DiffusersVideoEngine()
        cfg = VideoGenerationConfig(
            modelId="Lightricks/LTX-Video",
            modelName="LTX-Video",
            repo="Lightricks/LTX-Video",
            prompt="test",
            negativePrompt="",
            width=768,
            height=512,
            numFrames=97,
            fps=24,
            guidance=3.0,
            seed=None,
        )
        with self.assertRaises(NotImplementedError):
            engine.generate(cfg)


if __name__ == "__main__":
    unittest.main()
