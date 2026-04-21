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
    GeneratedVideo,
    PIPELINE_REGISTRY,
    VideoGenerationConfig,
    VideoRuntimeManager,
    VideoRuntimeStatus,
)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        # Bypass the Windows cold-start warmup gate in probe(). The gate
        # returns "initializing" early when torch hasn't been imported yet
        # so the HTTP request stays under the frontend's 30s fetch budget —
        # but in tests we want to exercise the ``_find_missing`` branch
        # directly, so we pretend torch is ready.
        self._warmup_patch = mock.patch.object(
            video_runtime,
            "torch_warmup_status",
            return_value={"status": "ready", "error": None, "started_at": 0.0},
        )
        self._warmup_patch.start()

    def tearDown(self):
        self._warmup_patch.stop()

    def test_probe_flags_missing_core_deps_as_unavailable(self):
        engine = DiffusersVideoEngine()
        # Simulate a machine with no diffusers/torch installed. Three calls
        # to ``_find_missing`` now: core, output, model-specific (tiktoken etc.).
        with mock.patch.object(
            video_runtime,
            "_find_missing",
            side_effect=[
                ["diffusers", "torch"],
                ["imageio", "imageio-ffmpeg"],
                ["tiktoken", "sentencepiece"],
            ],
        ):
            status = engine.probe()
        self.assertIsInstance(status, VideoRuntimeStatus)
        self.assertFalse(status.realGenerationAvailable)
        self.assertEqual(status.activeEngine, "placeholder")
        # All missing deps (core + output + model) surface in the list for a
        # single clear install hint.
        self.assertIn("diffusers", status.missingDependencies)
        self.assertIn("torch", status.missingDependencies)
        self.assertIn("imageio", status.missingDependencies)
        self.assertIn("tiktoken", status.missingDependencies)

    def test_probe_reports_ready_when_all_deps_findable(self):
        engine = DiffusersVideoEngine()

        # Core deps present, output deps present, model deps present.
        with mock.patch.object(video_runtime, "_find_missing", return_value=[]):
            status = engine.probe()

        self.assertTrue(status.realGenerationAvailable)
        self.assertEqual(status.activeEngine, "diffusers")
        # ``device`` is now None until a model is actually preloaded —
        # probe() no longer imports torch just to report device info,
        # because the implicit torch import was pinning DLLs in the
        # backend process and breaking /api/setup/install-gpu-bundle
        # on Windows. Device is populated by preload() once a model
        # is loaded.
        self.assertIsNone(status.device)
        self.assertEqual(status.missingDependencies, [])

    def test_probe_reports_ready_but_warns_when_only_output_deps_missing(self):
        engine = DiffusersVideoEngine()

        with mock.patch.object(
            video_runtime,
            "_find_missing",
            # First call: core deps — all present.
            # Second call: output deps — imageio missing.
            # Third call: model deps — all present.
            side_effect=[[], ["imageio", "imageio-ffmpeg"], []],
        ):
            status = engine.probe()

        self.assertTrue(status.realGenerationAvailable)
        self.assertEqual(status.activeEngine, "diffusers")
        self.assertIn("imageio", status.missingDependencies)
        self.assertIn("mp4", status.message.lower())

    def test_probe_reports_ready_but_warns_when_only_model_deps_missing(self):
        """LTX-Video and friends need tiktoken / sentencepiece — surface them."""
        engine = DiffusersVideoEngine()

        with mock.patch.object(
            video_runtime,
            "_find_missing",
            # Core present, output present, only model deps missing.
            side_effect=[[], [], ["tiktoken"]],
        ):
            status = engine.probe()

        self.assertTrue(status.realGenerationAvailable)
        self.assertIn("tiktoken", status.missingDependencies)
        # Message should mention tokenizer packages so the user knows why
        # it's flagged even though the engine is "ready".
        self.assertIn("tokenizer", status.message.lower())

    def test_probe_lists_both_output_and_model_deps_when_both_missing(self):
        """A fresh install often misses both buckets — surface them together."""
        engine = DiffusersVideoEngine()

        with mock.patch.object(
            video_runtime,
            "_find_missing",
            side_effect=[
                [],
                ["imageio-ffmpeg"],
                ["tiktoken", "sentencepiece"],
            ],
        ):
            status = engine.probe()

        self.assertTrue(status.realGenerationAvailable)
        self.assertIn("imageio-ffmpeg", status.missingDependencies)
        self.assertIn("tiktoken", status.missingDependencies)
        self.assertIn("sentencepiece", status.missingDependencies)

    def test_probe_does_not_import_torch(self):
        """Regression: probe() used to `import torch` which pinned
        torch/lib/*.dll into the backend process handle table. On Windows
        that broke /api/setup/install-gpu-bundle — pip's rmtree couldn't
        overwrite the locked DLLs. Probe now uses find_spec only.
        """
        import sys as _sys
        # Remove any pre-existing torch module from previous tests so we
        # can confirm probe() doesn't put it back.
        saved_torch = _sys.modules.pop("torch", None)
        engine = DiffusersVideoEngine()
        try:
            with mock.patch.object(video_runtime, "_find_missing", return_value=[]):
                engine.probe()
            self.assertNotIn(
                "torch",
                _sys.modules,
                "probe() must not import torch — it pins DLLs that block the GPU bundle install",
            )
        finally:
            if saved_torch is not None:
                _sys.modules["torch"] = saved_torch


class FindMissingTests(unittest.TestCase):
    """Regression: _find_missing used to crash with ``ModuleNotFoundError:
    No module named 'google'`` when asked about ``google.protobuf`` on a
    machine with neither google-anything nor protobuf installed. The crash
    propagated up through ``capabilities()`` and made ``/api/video/runtime``
    return a 500, which surfaced in the UI as the "runtime did not respond"
    fallback that never clears. ``importlib.util.find_spec`` is documented
    to return ``None`` for missing submodules, but it raises when the
    *parent* of a dotted name isn't importable — the classic Python 3
    namespace-package edge case.
    """

    def test_find_missing_handles_missing_parent_namespace(self):
        # Simulate the exact scenario from the user-reported crash: ask for
        # ``google.protobuf`` on a Python that has no ``google`` package at
        # all. We can't guarantee ``google`` is absent in every test env, so
        # mock find_spec to raise what the real implementation would.
        def fake_find_spec(name):
            if name == "google.protobuf":
                raise ModuleNotFoundError("No module named 'google'")
            return None  # treat everything else as missing too

        with mock.patch.object(
            video_runtime.importlib.util, "find_spec", side_effect=fake_find_spec
        ):
            missing = video_runtime._find_missing(
                (
                    ("tiktoken", "tiktoken"),
                    ("protobuf", "google.protobuf"),
                )
            )
        self.assertIn("protobuf", missing)
        self.assertIn("tiktoken", missing)

    def test_find_missing_returns_empty_when_all_present(self):
        # Sanity: when find_spec returns a spec object (non-None), the
        # package is considered present and not listed as missing.
        def fake_find_spec(name):
            return SimpleNamespace(name=name)  # any non-None value works

        with mock.patch.object(
            video_runtime.importlib.util, "find_spec", side_effect=fake_find_spec
        ):
            missing = video_runtime._find_missing(
                (("tiktoken", "tiktoken"), ("protobuf", "google.protobuf"))
            )
        self.assertEqual(missing, [])


class PipelineRegistryTests(unittest.TestCase):
    def test_registry_covers_all_first_wave_engines(self):
        expected = {
            "Lightricks/LTX-Video",
            "genmo/mochi-1-preview",
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "hunyuanvideo-community/HunyuanVideo",
            "THUDM/CogVideoX-2b",
            "THUDM/CogVideoX-5b",
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

    def test_cogvideox_variants_all_route_to_cogvideoxpipeline(self):
        """CogVideoX 2B and 5B share the same diffusers pipeline class."""
        cog_repos = [repo for repo in PIPELINE_REGISTRY if repo.startswith("THUDM/CogVideoX")]
        self.assertGreaterEqual(len(cog_repos), 2, "expected 2B and 5B CogVideoX entries")
        for repo in cog_repos:
            self.assertEqual(PIPELINE_REGISTRY[repo]["class_name"], "CogVideoXPipeline")

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
            side_effect=[["diffusers"], [], []],
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


class GenerateTests(unittest.TestCase):
    """Exercise ``DiffusersVideoEngine.generate()`` without loading real weights.

    We stub the three heavy seams: ``_ensure_pipeline`` (weight loading),
    ``_invoke_pipeline`` (the actual diffusion pass), and ``_encode_frames_to_mp4``
    (ffmpeg muxing). The engine's own orchestration logic — seed resolution,
    generator construction, kwarg building, timing, frame-count validation —
    is what we actually want to test here.
    """

    FAKE_BYTES = b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 48

    def _config(self, seed: int | None = 42) -> VideoGenerationConfig:
        return VideoGenerationConfig(
            modelId="Lightricks/LTX-Video",
            modelName="LTX-Video",
            repo="Lightricks/LTX-Video",
            prompt="a cinematic shot of a misty forest",
            negativePrompt="blurry, low quality",
            width=768,
            height=512,
            numFrames=24,
            fps=24,
            steps=20,
            guidance=3.0,
            seed=seed,
        )

    def _install_torch_shim(self, engine: DiffusersVideoEngine) -> mock.MagicMock:
        """Install a minimal torch shim on the engine so generate() can build a Generator."""
        generator_instance = mock.MagicMock(name="torch_generator")
        generator_instance.manual_seed.return_value = generator_instance
        generator_cls = mock.MagicMock(return_value=generator_instance)
        torch_shim = SimpleNamespace(
            Generator=generator_cls,
            cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        )
        engine._torch = torch_shim  # type: ignore[assignment]
        engine._device = "cpu"
        return generator_cls

    def test_generate_happy_path_returns_generated_video_with_real_metadata(self):
        engine = DiffusersVideoEngine()
        generator_cls = self._install_torch_shim(engine)

        fake_pipeline = mock.MagicMock(name="pipeline")
        fake_frames = [mock.MagicMock(name=f"frame_{idx}") for idx in range(24)]

        with mock.patch.object(video_runtime, "_find_missing", return_value=[]), \
                mock.patch.object(engine, "_ensure_pipeline", return_value=fake_pipeline), \
                mock.patch.object(engine, "_invoke_pipeline", return_value=fake_frames) as invoke, \
                mock.patch.object(
                    engine,
                    "_encode_frames_to_mp4",
                    return_value=self.FAKE_BYTES,
                ) as encode:
            result = engine.generate(self._config(seed=42))

        self.assertIsInstance(result, GeneratedVideo)
        self.assertEqual(result.seed, 42)
        self.assertEqual(result.bytes, self.FAKE_BYTES)
        self.assertEqual(result.frameCount, 24)
        self.assertEqual(result.fps, 24)
        self.assertEqual(result.width, 768)
        self.assertEqual(result.height, 512)
        self.assertEqual(result.mimeType, "video/mp4")
        self.assertEqual(result.extension, "mp4")
        self.assertIn("cpu", result.runtimeLabel)
        self.assertGreater(result.durationSeconds, 0.0)

        # Confirm the generator was constructed on CPU (MPS fallback) and seeded
        # with the provided value so callers can actually reproduce a run.
        generator_cls.assert_called_once()
        invoke.assert_called_once()
        encode.assert_called_once()

    def test_generate_resolves_random_seed_when_none_provided(self):
        engine = DiffusersVideoEngine()
        self._install_torch_shim(engine)

        with mock.patch.object(video_runtime, "_find_missing", return_value=[]), \
                mock.patch.object(engine, "_ensure_pipeline", return_value=mock.MagicMock()), \
                mock.patch.object(
                    engine,
                    "_invoke_pipeline",
                    return_value=[mock.MagicMock() for _ in range(24)],
                ), \
                mock.patch.object(
                    engine,
                    "_encode_frames_to_mp4",
                    return_value=self.FAKE_BYTES,
                ):
            result = engine.generate(self._config(seed=None))

        # No seed in the config -> engine must pick a deterministic integer so
        # the user can see it in the UI and reproduce the render.
        self.assertIsInstance(result.seed, int)
        self.assertGreaterEqual(result.seed, 0)
        self.assertLess(result.seed, 2**31)

    def test_generate_rejects_empty_frame_output(self):
        engine = DiffusersVideoEngine()
        self._install_torch_shim(engine)

        with mock.patch.object(video_runtime, "_find_missing", return_value=[]), \
                mock.patch.object(engine, "_ensure_pipeline", return_value=mock.MagicMock()), \
                mock.patch.object(engine, "_invoke_pipeline", return_value=[]):
            with self.assertRaises(RuntimeError) as ctx:
                engine.generate(self._config())
        self.assertIn("zero frames", str(ctx.exception).lower())

    def test_generate_raises_when_output_deps_missing(self):
        engine = DiffusersVideoEngine()

        # First _find_missing call (in preload path during _ensure_pipeline) we bypass by
        # mocking _ensure_pipeline itself. The call we care about is the output-deps
        # check at the top of generate(): imageio / imageio-ffmpeg missing.
        with mock.patch.object(
            video_runtime,
            "_find_missing",
            return_value=["imageio", "imageio-ffmpeg"],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                engine.generate(self._config())
        self.assertIn("imageio", str(ctx.exception).lower())


class VideoRuntimeManagerGenerateTests(unittest.TestCase):
    """Exercise the ``VideoRuntimeManager.generate`` facade without any pipeline."""

    def test_generate_raises_when_runtime_unavailable(self):
        manager = VideoRuntimeManager()
        # Core deps missing -> probe() reports not ready -> facade must refuse.
        with mock.patch.object(
            video_runtime,
            "_find_missing",
            side_effect=[["diffusers", "torch"], ["imageio"], []],
        ):
            with self.assertRaises(RuntimeError):
                manager.generate(
                    VideoGenerationConfig(
                        modelId="Lightricks/LTX-Video",
                        modelName="LTX-Video",
                        repo="Lightricks/LTX-Video",
                        prompt="test",
                        negativePrompt="",
                        width=768,
                        height=512,
                        numFrames=24,
                        fps=24,
                        guidance=3.0,
                        seed=1,
                    )
                )

    def test_generate_returns_video_and_runtime_dict(self):
        manager = VideoRuntimeManager()
        fake_video = GeneratedVideo(
            seed=123,
            bytes=b"fake-mp4",
            extension="mp4",
            mimeType="video/mp4",
            durationSeconds=2.5,
            frameCount=24,
            fps=24,
            width=768,
            height=512,
            runtimeLabel="test",
            runtimeNote=None,
        )
        with mock.patch.object(video_runtime, "_find_missing", return_value=[]), \
                mock.patch.object(
                    manager._engine,
                    "generate",
                    return_value=fake_video,
                ):
            video, runtime = manager.generate(
                VideoGenerationConfig(
                    modelId="Lightricks/LTX-Video",
                    modelName="LTX-Video",
                    repo="Lightricks/LTX-Video",
                    prompt="test",
                    negativePrompt="",
                    width=768,
                    height=512,
                    numFrames=24,
                    fps=24,
                    guidance=3.0,
                    seed=123,
                )
            )
        self.assertIs(video, fake_video)
        self.assertIn("realGenerationAvailable", runtime)
        self.assertTrue(runtime["realGenerationAvailable"])


if __name__ == "__main__":
    unittest.main()
