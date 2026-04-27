"""Unit tests for ``backend_service.video_runtime``.

We never actually load a video model in tests (those weights are 10-25GB).
The tests exercise the surface logic: probe dependency detection, pipeline
class registry routing, preload/unload lifecycle via monkey-patched seams.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
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
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
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


class InterpolateFramesTests(unittest.TestCase):
    """``_interpolate_frames`` inserts blended intermediates so the
    encoder sees ``(len-1) * factor + 1`` frames. Factor 1 must be a
    true no-op to keep existing generation paths byte-identical."""

    def _solid_frames(self, n: int, size: int = 4):
        import numpy as np
        return [
            np.full((size, size, 3), index * 64, dtype=np.uint8)
            for index in range(n)
        ]

    def test_factor_1_is_noop(self):
        from backend_service.video_runtime import _interpolate_frames
        frames = self._solid_frames(3)
        out = _interpolate_frames(frames, 1)
        self.assertEqual(len(out), 3)

    def test_factor_2_doubles_the_stride(self):
        from backend_service.video_runtime import _interpolate_frames
        frames = self._solid_frames(3)
        out = _interpolate_frames(frames, 2)
        self.assertEqual(len(out), 5)

    def test_factor_4_quadruples_the_stride(self):
        from backend_service.video_runtime import _interpolate_frames
        frames = self._solid_frames(3)
        out = _interpolate_frames(frames, 4)
        self.assertEqual(len(out), 9)

    def test_blend_is_linear_midpoint(self):
        from backend_service.video_runtime import _interpolate_frames
        import numpy as np
        a = np.full((2, 2, 3), 0, dtype=np.uint8)
        b = np.full((2, 2, 3), 100, dtype=np.uint8)
        out = _interpolate_frames([a, b], 2)
        self.assertEqual(int(out[1][0, 0, 0]), 50)

    def test_empty_or_single_frame_returns_unchanged(self):
        from backend_service.video_runtime import _interpolate_frames
        self.assertEqual(_interpolate_frames([], 2), [])
        frames = self._solid_frames(1)
        out = _interpolate_frames(frames, 4)
        self.assertEqual(len(out), 1)


class AlignWanNumFramesTests(unittest.TestCase):
    """Wan models need num_frames in the (4k+1) form."""

    def test_non_wan_repo_passes_through_unchanged(self):
        from backend_service.video_runtime import _align_wan_num_frames
        out, note = _align_wan_num_frames("Lightricks/LTX-Video", 24)
        self.assertEqual(out, 24)
        self.assertIsNone(note)

    def test_wan_24_rounds_down_to_21(self):
        from backend_service.video_runtime import _align_wan_num_frames
        out, note = _align_wan_num_frames("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 24)
        self.assertEqual(out, 21)
        self.assertIsNotNone(note)
        self.assertIn("21", note or "")

    def test_wan_already_aligned_emits_no_note(self):
        from backend_service.video_runtime import _align_wan_num_frames
        for valid in (5, 9, 25, 49, 81, 97):
            out, note = _align_wan_num_frames("Wan-AI/Wan2.1-T2V-14B-Diffusers", valid)
            self.assertEqual(out, valid)
            self.assertIsNone(note)

    def test_wan_below_minimum_clamps_to_5(self):
        from backend_service.video_runtime import _align_wan_num_frames
        out, note = _align_wan_num_frames("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 4)
        self.assertEqual(out, 5)
        self.assertIsNotNone(note)


class ResolveVideoDefaultsTests(unittest.TestCase):
    """``_resolve_video_defaults`` only substitutes when the request kept
    schema defaults (50 steps / CFG 3.0). Explicit values must survive."""

    def test_schema_defaults_substituted_for_wan(self):
        from backend_service.video_runtime import _resolve_video_defaults
        resolved = _resolve_video_defaults("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 50, 3.0)
        self.assertEqual(resolved["steps"], 30)
        self.assertEqual(resolved["guidance"], 6.0)
        self.assertEqual(resolved["scheduler"], "unipc")
        self.assertTrue(resolved["substituted"])

    def test_user_overrides_preserved(self):
        from backend_service.video_runtime import _resolve_video_defaults
        resolved = _resolve_video_defaults("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 60, 8.5)
        self.assertEqual(resolved["steps"], 60)
        self.assertEqual(resolved["guidance"], 8.5)
        self.assertFalse(resolved["substituted"])

    def test_partial_override_preserves_user_field(self):
        # User explicitly tuned guidance to 8.0 but kept default 50 steps.
        # Steps gets the model-tuned value; guidance stays 8.0.
        from backend_service.video_runtime import _resolve_video_defaults
        resolved = _resolve_video_defaults("Lightricks/LTX-Video", 50, 8.0)
        self.assertEqual(resolved["steps"], 30)
        self.assertEqual(resolved["guidance"], 8.0)
        self.assertTrue(resolved["substituted"])

    def test_unknown_repo_returns_request_values(self):
        from backend_service.video_runtime import _resolve_video_defaults
        resolved = _resolve_video_defaults("not/a-real-repo", 50, 3.0)
        self.assertEqual(resolved["steps"], 50)
        self.assertEqual(resolved["guidance"], 3.0)
        self.assertIsNone(resolved["scheduler"])
        self.assertFalse(resolved["substituted"])


class ShouldApplyMemorySaversTests(unittest.TestCase):
    """Slicing + tiling cut quality. Only enable under real pressure."""

    def test_force_env_overrides_everything(self):
        from backend_service.video_runtime import _should_apply_memory_savers
        with mock.patch.dict("os.environ", {"CHAOSENGINE_VIDEO_FORCE_SLICING": "1"}):
            self.assertTrue(_should_apply_memory_savers("cuda", 24.0, 1.0))

    def test_cpu_always_enables_savers(self):
        from backend_service.video_runtime import _should_apply_memory_savers
        self.assertTrue(_should_apply_memory_savers("cpu", 64.0, 5.0))

    def test_unknown_memory_stays_safe(self):
        from backend_service.video_runtime import _should_apply_memory_savers
        self.assertTrue(_should_apply_memory_savers("mps", None, 5.0))
        self.assertTrue(_should_apply_memory_savers("mps", 64.0, None))

    def test_64gb_mac_with_small_wan_does_not_slice(self):
        from backend_service.video_runtime import _should_apply_memory_savers
        # Wan 2.1 1.3B on a 64 GB Mac → ~9 / 64 = 14% → no slicing.
        self.assertFalse(_should_apply_memory_savers("mps", 64.0, 9.0))

    def test_4090_with_wan_14b_bf16_engages_slicing(self):
        from backend_service.video_runtime import _should_apply_memory_savers
        # 28 / 24 = 117% → slicing required.
        self.assertTrue(_should_apply_memory_savers("cuda", 24.0, 28.0))

    def test_just_above_threshold_engages_slicing(self):
        from backend_service.video_runtime import _should_apply_memory_savers
        # 17.5 / 24 = 72.9% → past the 70% threshold.
        self.assertTrue(_should_apply_memory_savers("cuda", 24.0, 17.5))


class EstimateModelFootprintTests(unittest.TestCase):
    def test_known_repo_returns_table_value(self):
        from backend_service.video_runtime import _estimate_model_footprint_gb
        self.assertAlmostEqual(
            _estimate_model_footprint_gb(
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "torch.bfloat16"
            ),
            9.0,
        )

    def test_gguf_q4_shrinks_footprint(self):
        from backend_service.video_runtime import _estimate_model_footprint_gb
        full = _estimate_model_footprint_gb(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers", "torch.bfloat16"
        )
        gguf_q4 = _estimate_model_footprint_gb(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "torch.bfloat16",
            gguf_file="wan2.1-t2v-14B-Q4_K_M.gguf",
        )
        self.assertLess(gguf_q4, full)

    def test_unknown_repo_returns_none(self):
        from backend_service.video_runtime import _estimate_model_footprint_gb
        self.assertIsNone(_estimate_model_footprint_gb("not/known", "torch.bfloat16"))


class PreferredTorchDtypeTests(unittest.TestCase):
    """``_preferred_torch_dtype`` must return bf16 on CUDA, probe MPS for
    bf16 capability with a clean fp16 fallback, and stay fp32 on CPU."""

    def _engine(self) -> DiffusersVideoEngine:
        return DiffusersVideoEngine()

    def test_cuda_returns_bfloat16(self):
        engine = self._engine()
        torch_shim = SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32")
        self.assertEqual(engine._preferred_torch_dtype(torch_shim, "cuda"), "bf16")

    def test_cpu_returns_float32(self):
        engine = self._engine()
        torch_shim = SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32")
        self.assertEqual(engine._preferred_torch_dtype(torch_shim, "cpu"), "fp32")

    def test_mps_probe_succeeds_returns_bfloat16(self):
        engine = self._engine()
        zeros = mock.MagicMock(return_value=mock.MagicMock())
        torch_shim = SimpleNamespace(
            bfloat16="bf16", float16="fp16", float32="fp32", zeros=zeros
        )
        self.assertEqual(engine._preferred_torch_dtype(torch_shim, "mps"), "bf16")
        zeros.assert_called_once()

    def test_mps_probe_falls_back_to_fp16_on_runtime_error(self):
        engine = self._engine()
        zeros = mock.MagicMock(side_effect=RuntimeError("MPS bf16 unsupported"))
        torch_shim = SimpleNamespace(
            bfloat16="bf16", float16="fp16", float32="fp32", zeros=zeros
        )
        self.assertEqual(engine._preferred_torch_dtype(torch_shim, "mps"), "fp16")

    def test_mps_env_opt_out_forces_fp16(self):
        engine = self._engine()
        zeros = mock.MagicMock()
        torch_shim = SimpleNamespace(
            bfloat16="bf16", float16="fp16", float32="fp32", zeros=zeros
        )
        with mock.patch.dict("os.environ", {"CHAOSENGINE_VIDEO_MPS_BF16": "0"}):
            self.assertEqual(engine._preferred_torch_dtype(torch_shim, "mps"), "fp16")
        zeros.assert_not_called()


class SwapSchedulerTests(unittest.TestCase):
    def _pipeline_with_scheduler(self, scheduler_cls_name: str) -> mock.MagicMock:
        # Build a real class with the desired name so ``type(instance).__name__``
        # matches what ``_swap_scheduler`` checks for.
        scheduler_cls = type(scheduler_cls_name, (), {})
        scheduler_instance = scheduler_cls()
        scheduler_instance.config = {"baked": "config"}
        pipeline = mock.MagicMock(name="pipeline")
        pipeline.scheduler = scheduler_instance
        return pipeline

    def test_none_scheduler_id_short_circuits(self):
        engine = DiffusersVideoEngine()
        pipeline = self._pipeline_with_scheduler("FlowMatchEulerDiscreteScheduler")
        self.assertIsNone(engine._swap_scheduler(pipeline, None))

    def test_unknown_scheduler_id_short_circuits(self):
        engine = DiffusersVideoEngine()
        pipeline = self._pipeline_with_scheduler("FlowMatchEulerDiscreteScheduler")
        self.assertIsNone(engine._swap_scheduler(pipeline, "not-a-scheduler"))

    def test_swap_replaces_scheduler_via_from_config(self):
        engine = DiffusersVideoEngine()
        pipeline = self._pipeline_with_scheduler("FlowMatchEulerDiscreteScheduler")

        new_scheduler_instance = mock.MagicMock(name="UniPCInstance")
        scheduler_cls = mock.MagicMock(from_config=mock.MagicMock(
            return_value=new_scheduler_instance
        ))
        fake_diffusers = SimpleNamespace(UniPCMultistepScheduler=scheduler_cls)
        with mock.patch.object(
            video_runtime.importlib, "import_module", return_value=fake_diffusers
        ):
            note = engine._swap_scheduler(pipeline, "unipc")

        self.assertIsNotNone(note)
        self.assertIn("unipc", note or "")
        self.assertIs(pipeline.scheduler, new_scheduler_instance)
        scheduler_cls.from_config.assert_called_once_with({"baked": "config"})

    def test_swap_skipped_when_already_on_target(self):
        engine = DiffusersVideoEngine()
        pipeline = self._pipeline_with_scheduler("UniPCMultistepScheduler")
        self.assertIsNone(engine._swap_scheduler(pipeline, "unipc"))

    def test_missing_class_in_diffusers_returns_warning_note(self):
        engine = DiffusersVideoEngine()
        pipeline = self._pipeline_with_scheduler("FlowMatchEulerDiscreteScheduler")
        empty_diffusers = SimpleNamespace()
        with mock.patch.object(
            video_runtime.importlib, "import_module", return_value=empty_diffusers
        ):
            note = engine._swap_scheduler(pipeline, "unipc")
        self.assertIsNotNone(note)
        self.assertIn("not available", note or "")


class FinalizeConfigTests(unittest.TestCase):
    def _config(self, **overrides) -> VideoGenerationConfig:
        defaults = dict(
            modelId="x",
            modelName="LTX-Video",
            repo="Lightricks/LTX-Video",
            prompt="p",
            negativePrompt="",
            width=768,
            height=512,
            numFrames=24,
            fps=24,
            steps=50,
            guidance=3.0,
        )
        defaults.update(overrides)
        return VideoGenerationConfig(**defaults)

    def test_finalize_substitutes_defaults_for_ltx(self):
        engine = DiffusersVideoEngine()
        cfg, notes = engine._finalize_config(self._config())
        self.assertEqual(cfg.steps, 30)
        self.assertEqual(cfg.guidance, 3.0)
        # LTX force-swaps to FlowMatchEulerDiscreteScheduler — only
        # scheduler that accepts ``set_timesteps(mu=...)``. Older cached
        # snapshots have plain EulerDiscreteScheduler baked in.
        self.assertEqual(cfg.scheduler, "flow-euler")
        self.assertTrue(any("Substituting" in n for n in notes))

    def test_finalize_aligns_wan_frames_and_keeps_user_steps(self):
        engine = DiffusersVideoEngine()
        cfg, notes = engine._finalize_config(self._config(
            repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            numFrames=24,
            steps=40,
            guidance=6.5,
        ))
        self.assertEqual(cfg.numFrames, 21)
        self.assertEqual(cfg.steps, 40)
        self.assertEqual(cfg.guidance, 6.5)
        self.assertTrue(any("Aligned" in n for n in notes))

    def test_explicit_scheduler_overrides_model_default(self):
        engine = DiffusersVideoEngine()
        cfg, _ = engine._finalize_config(self._config(scheduler="ddim"))
        self.assertEqual(cfg.scheduler, "ddim")

    def test_unknown_scheduler_drops_to_pipeline_default(self):
        engine = DiffusersVideoEngine()
        cfg, notes = engine._finalize_config(self._config(scheduler="not-real"))
        self.assertIsNone(cfg.scheduler)
        self.assertTrue(any("Unknown scheduler" in n for n in notes))

    def test_auto_scheduler_uses_model_table(self):
        engine = DiffusersVideoEngine()
        cfg, _ = engine._finalize_config(self._config(
            repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            numFrames=25,
            scheduler="auto",
        ))
        self.assertEqual(cfg.scheduler, "unipc")

    def test_finalize_emits_ltx_auto_tune_note(self):
        # Phase D: surface the LTX kwarg auto-tune so the user knows why
        # output matches the Lightricks reference even without new sliders.
        engine = DiffusersVideoEngine()
        _, notes = engine._finalize_config(self._config(fps=24))
        self.assertTrue(any("LTX-Video auto-tuned" in n for n in notes))
        self.assertTrue(any("frame_rate=24" in n for n in notes))
        self.assertTrue(any("decode_timestep=0.05" in n for n in notes))

    def test_finalize_no_ltx_note_for_wan(self):
        engine = DiffusersVideoEngine()
        _, notes = engine._finalize_config(self._config(
            repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            numFrames=25,
        ))
        self.assertFalse(any("LTX-Video auto-tuned" in n for n in notes))


class EnhancePromptTests(unittest.TestCase):
    """Phase E1: template-based prompt enhancer.

    Short prompts (< 25 words) get a per-model structural suffix appended.
    Long prompts pass through untouched. ``enhancePrompt=False`` skips
    the enhancer entirely. The enhancer is idempotent — a second call on
    an already-enhanced prompt is a no-op.
    """

    def test_short_ltx_prompt_gets_enhanced(self):
        from backend_service.video_runtime import _enhance_prompt
        out, note = _enhance_prompt("Lightricks/LTX-Video", "cartoon llama eating straw")
        self.assertNotEqual(out, "cartoon llama eating straw")
        self.assertIn("cartoon llama eating straw", out)
        self.assertIn("cinematic", out.lower())
        self.assertIsNotNone(note)
        self.assertIn("Auto-enhanced", note)

    def test_short_wan_prompt_gets_enhanced(self):
        from backend_service.video_runtime import _enhance_prompt
        out, _ = _enhance_prompt("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "a cat in snow")
        self.assertIn("a cat in snow", out)
        self.assertIn("35mm", out)

    def test_long_prompt_skips_enhancement(self):
        from backend_service.video_runtime import _enhance_prompt
        long_prompt = " ".join(["word"] * 30)
        out, note = _enhance_prompt("Lightricks/LTX-Video", long_prompt)
        self.assertEqual(out, long_prompt)
        self.assertIsNone(note)

    def test_unknown_repo_returns_unchanged(self):
        from backend_service.video_runtime import _enhance_prompt
        out, note = _enhance_prompt("not/a-real-repo", "short prompt")
        self.assertEqual(out, "short prompt")
        self.assertIsNone(note)

    def test_empty_prompt_returns_unchanged(self):
        from backend_service.video_runtime import _enhance_prompt
        out, note = _enhance_prompt("Lightricks/LTX-Video", "")
        self.assertEqual(out, "")
        self.assertIsNone(note)

    def test_idempotent_when_suffix_already_present(self):
        # Second call must not double-append the suffix.
        from backend_service.video_runtime import _enhance_prompt
        first, _ = _enhance_prompt("Lightricks/LTX-Video", "cartoon llama")
        second, note = _enhance_prompt("Lightricks/LTX-Video", first)
        self.assertEqual(first, second)
        self.assertIsNone(note)

    def test_finalize_config_appends_when_enhance_enabled(self):
        # End-to-end: short LTX prompt with enhancePrompt=True should
        # land on the resolved config with the suffix attached and a
        # run-log note generated.
        engine = DiffusersVideoEngine()
        cfg = VideoGenerationConfig(
            modelId="x",
            modelName="LTX-Video",
            repo="Lightricks/LTX-Video",
            prompt="cartoon llama",
            negativePrompt="",
            width=768,
            height=512,
            numFrames=25,
            fps=24,
            steps=30,
            guidance=3.0,
            enhancePrompt=True,
        )
        resolved, notes = engine._finalize_config(cfg)
        self.assertNotEqual(resolved.prompt, "cartoon llama")
        self.assertIn("cartoon llama", resolved.prompt)
        self.assertTrue(any("Auto-enhanced" in n for n in notes))

    def test_finalize_config_skips_when_enhance_disabled(self):
        engine = DiffusersVideoEngine()
        cfg = VideoGenerationConfig(
            modelId="x",
            modelName="LTX-Video",
            repo="Lightricks/LTX-Video",
            prompt="cartoon llama",
            negativePrompt="",
            width=768,
            height=512,
            numFrames=25,
            fps=24,
            steps=30,
            guidance=3.0,
            enhancePrompt=False,
        )
        resolved, notes = engine._finalize_config(cfg)
        self.assertEqual(resolved.prompt, "cartoon llama")
        self.assertFalse(any("Auto-enhanced" in n for n in notes))

    def test_short_ltx2_prompt_gets_enhanced(self):
        # Phase E2.1: prince-canuma/LTX-2-* now in enhancer dict.
        from backend_service.video_runtime import _enhance_prompt
        out, note = _enhance_prompt("prince-canuma/LTX-2-distilled", "drone shot")
        self.assertNotEqual(out, "drone shot")
        self.assertIn("drone shot", out)
        self.assertIn("cinematic", out.lower())
        self.assertIsNotNone(note)

    def test_short_ltx2_3_prompt_gets_enhanced(self):
        from backend_service.video_runtime import _enhance_prompt
        out, _ = _enhance_prompt("prince-canuma/LTX-2.3-distilled", "skater in tokyo")
        self.assertIn("skater in tokyo", out)
        self.assertIn("cinematic", out.lower())


class CfgDecayTests(unittest.TestCase):
    """Phase E2.2: linear CFG decay across the sampling schedule.

    Flow-match video models oversaturate when CFG stays high throughout;
    decay lets early steps lock semantics (high CFG) while late steps
    preserve fine detail (low CFG → 1.0 by the final step).
    """

    def test_decay_sets_pipeline_guidance_scale_at_each_step(self):
        from backend_service.video_runtime import DiffusersVideoEngine
        engine = DiffusersVideoEngine()
        callback = engine._make_step_callback(
            total_steps=4, initial_guidance=4.0, cfg_decay=True,
        )
        # Stub pipeline carrying a mutable guidance_scale.
        class StubPipeline:
            guidance_scale = 4.0
        pipeline = StubPipeline()
        # Step 0 finished — set up scale for step 1. Linear ramp from
        # 4.0 (i=0) to 1.0 (i=total-1=3): scale at i=1 is 4 - 1*(3/3) = 3.
        callback(pipeline, 0, None, {})
        self.assertAlmostEqual(pipeline.guidance_scale, 3.0, places=5)
        callback(pipeline, 1, None, {})
        self.assertAlmostEqual(pipeline.guidance_scale, 2.0, places=5)
        callback(pipeline, 2, None, {})
        self.assertAlmostEqual(pipeline.guidance_scale, 1.0, places=5)

    def test_decay_disabled_leaves_guidance_scale_alone(self):
        from backend_service.video_runtime import DiffusersVideoEngine
        engine = DiffusersVideoEngine()
        callback = engine._make_step_callback(
            total_steps=4, initial_guidance=4.0, cfg_decay=False,
        )
        class StubPipeline:
            guidance_scale = 4.0
        pipeline = StubPipeline()
        callback(pipeline, 0, None, {})
        callback(pipeline, 1, None, {})
        self.assertEqual(pipeline.guidance_scale, 4.0)

    def test_decay_skipped_when_initial_guidance_is_one(self):
        # No floor to ramp toward — decay would flatline.
        from backend_service.video_runtime import DiffusersVideoEngine
        engine = DiffusersVideoEngine()
        callback = engine._make_step_callback(
            total_steps=4, initial_guidance=1.0, cfg_decay=True,
        )
        class StubPipeline:
            guidance_scale = 1.0
        pipeline = StubPipeline()
        callback(pipeline, 0, None, {})
        self.assertEqual(pipeline.guidance_scale, 1.0)

    def test_finalize_emits_cfg_decay_note(self):
        from backend_service.video_runtime import DiffusersVideoEngine
        engine = DiffusersVideoEngine()
        cfg = VideoGenerationConfig(
            modelId="x",
            modelName="LTX-Video",
            repo="Lightricks/LTX-Video",
            prompt="a long detailed cinematic prompt about a cat in a kitchen with lots of sun",
            negativePrompt="",
            width=768,
            height=512,
            numFrames=25,
            fps=24,
            steps=30,
            guidance=3.0,
            cfgDecay=True,
        )
        _, notes = engine._finalize_config(cfg)
        self.assertTrue(any("CFG decay" in n for n in notes))

    def test_finalize_no_cfg_decay_note_when_disabled(self):
        from backend_service.video_runtime import DiffusersVideoEngine
        engine = DiffusersVideoEngine()
        cfg = VideoGenerationConfig(
            modelId="x",
            modelName="LTX-Video",
            repo="Lightricks/LTX-Video",
            prompt="a long detailed cinematic prompt about a cat in a kitchen with lots of sun",
            negativePrompt="",
            width=768,
            height=512,
            numFrames=25,
            fps=24,
            steps=30,
            guidance=3.0,
            cfgDecay=False,
        )
        _, notes = engine._finalize_config(cfg)
        self.assertFalse(any("CFG decay" in n for n in notes))


class BuildPipelineKwargsLtxTests(unittest.TestCase):
    """Phase D: LTX-Video kwarg parity with Lightricks reference defaults.

    Without these kwargs the diffusers LTXPipeline produces rainbow / blurry
    output because the model conditions on default frame_rate=25 (mismatch
    with our 24 fps export) and the VAE decodes from final latent without
    the small denoise pass that cleans compression artifacts.
    """

    def _config(self, **overrides):
        defaults = dict(
            modelId="x",
            modelName="LTX-Video",
            repo="Lightricks/LTX-Video",
            prompt="a cat in a kitchen",
            negativePrompt="",
            width=768,
            height=512,
            numFrames=25,
            fps=24,
            steps=30,
            guidance=3.0,
        )
        defaults.update(overrides)
        return VideoGenerationConfig(**defaults)

    def _engine_with_pipeline_class(self, class_name: str) -> DiffusersVideoEngine:
        engine = DiffusersVideoEngine()
        # Inject a stand-in pipeline whose ``type(...).__name__`` matches
        # the dispatch the LTX branch keys on. Real diffusers pipelines
        # carry a lot of state; we only need the class name probe.
        Pipeline = type(class_name, (), {})
        engine._pipeline = Pipeline()
        return engine

    def test_ltx_pipeline_receives_decode_timestep(self):
        engine = self._engine_with_pipeline_class("LTXPipeline")
        kwargs = engine._build_pipeline_kwargs(self._config(), generator=None)
        self.assertEqual(kwargs["decode_timestep"], 0.05)
        self.assertEqual(kwargs["decode_noise_scale"], 0.025)
        self.assertEqual(kwargs["guidance_rescale"], 0.7)

    def test_ltx_pipeline_receives_frame_rate_from_config_fps(self):
        engine = self._engine_with_pipeline_class("LTXPipeline")
        kwargs = engine._build_pipeline_kwargs(self._config(fps=30), generator=None)
        self.assertEqual(kwargs["frame_rate"], 30)

    def test_ltx_pipeline_default_negative_when_empty(self):
        engine = self._engine_with_pipeline_class("LTXPipeline")
        kwargs = engine._build_pipeline_kwargs(self._config(negativePrompt=""), generator=None)
        self.assertIn("worst quality", kwargs["negative_prompt"])
        self.assertIn("inconsistent motion", kwargs["negative_prompt"])

    def test_ltx_pipeline_user_negative_preserved(self):
        engine = self._engine_with_pipeline_class("LTXPipeline")
        kwargs = engine._build_pipeline_kwargs(
            self._config(negativePrompt="my custom negative"),
            generator=None,
        )
        self.assertEqual(kwargs["negative_prompt"], "my custom negative")

    def test_non_ltx_pipeline_does_not_get_ltx_kwargs(self):
        engine = self._engine_with_pipeline_class("WanPipeline")
        kwargs = engine._build_pipeline_kwargs(
            self._config(repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
            generator=None,
        )
        self.assertNotIn("decode_timestep", kwargs)
        self.assertNotIn("decode_noise_scale", kwargs)
        self.assertNotIn("guidance_rescale", kwargs)
        self.assertNotIn("frame_rate", kwargs)


class TryLoadBnbNf4TransformerTests(unittest.TestCase):
    """Cover the NF4 loader's failure modes + happy path.

    Mirrors the GGUF loader's contract: every failure returns
    ``(None, note)`` so the caller falls back to the standard transformer.
    """

    def setUp(self) -> None:
        self.engine = DiffusersVideoEngine()
        self.torch = SimpleNamespace(bfloat16="bfloat16-sentinel")

    def test_non_cuda_device_returns_note(self):
        result, note = self.engine._try_load_bnb_nf4_transformer(
            repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            local_path="/tmp/snap",
            torch=self.torch,
            device="mps",
        )
        self.assertIsNone(result)
        self.assertIn("CUDA", note)

    def test_missing_bitsandbytes_returns_note(self):
        with mock.patch(
            "importlib.util.find_spec",
            side_effect=lambda name: None if name == "bitsandbytes" else mock.DEFAULT,
        ):
            result, note = self.engine._try_load_bnb_nf4_transformer(
                repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                local_path="/tmp/snap",
                torch=self.torch,
                device="cuda",
            )
        self.assertIsNone(result)
        self.assertIn("bitsandbytes", note)

    def test_unmapped_repo_returns_note(self):
        # bitsandbytes + diffusers BitsAndBytesConfig present, but the repo
        # has no class registered → we surface a clear note.
        fake_diffusers = SimpleNamespace(BitsAndBytesConfig=lambda **kw: None)
        with mock.patch("importlib.util.find_spec", return_value=object()), \
             mock.patch.dict(
                 "sys.modules",
                 {"diffusers": fake_diffusers},
                 clear=False,
             ):
            result, note = self.engine._try_load_bnb_nf4_transformer(
                repo="some/unknown-repo",
                local_path="/tmp/snap",
                torch=self.torch,
                device="cuda",
            )
        self.assertIsNone(result)
        self.assertIn("No NF4 transformer class", note)

    def test_happy_path_returns_transformer(self):
        captured: dict[str, Any] = {}

        class _FakeTransformer:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                captured["path"] = path
                captured.update(kwargs)
                return SimpleNamespace(name="fake-transformer")

        class _FakeBnbConfig:
            def __init__(self, **kwargs):
                captured["bnb_kwargs"] = kwargs

        fake_diffusers = SimpleNamespace(
            BitsAndBytesConfig=_FakeBnbConfig,
            WanTransformer3DModel=_FakeTransformer,
        )
        with mock.patch("importlib.util.find_spec", return_value=object()), \
             mock.patch.dict(
                 "sys.modules",
                 {"diffusers": fake_diffusers},
                 clear=False,
             ):
            result, note = self.engine._try_load_bnb_nf4_transformer(
                repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                local_path="/tmp/snap",
                torch=self.torch,
                device="cuda",
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "fake-transformer")
        self.assertIn("NF4", note)
        # Confirm we passed subfolder="transformer" and the bf16 compute dtype.
        self.assertEqual(captured["path"], "/tmp/snap")
        self.assertEqual(captured["subfolder"], "transformer")
        self.assertEqual(captured["torch_dtype"], "bfloat16-sentinel")
        self.assertEqual(captured["bnb_kwargs"]["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(captured["bnb_kwargs"]["load_in_4bit"])

    def test_load_failure_returns_fallback_note(self):
        class _FakeTransformer:
            @classmethod
            def from_pretrained(cls, *_a, **_kw):
                raise RuntimeError("snapshot subfolder missing")

        fake_diffusers = SimpleNamespace(
            BitsAndBytesConfig=lambda **kw: None,
            WanTransformer3DModel=_FakeTransformer,
        )
        with mock.patch("importlib.util.find_spec", return_value=object()), \
             mock.patch.dict(
                 "sys.modules",
                 {"diffusers": fake_diffusers},
                 clear=False,
             ):
            result, note = self.engine._try_load_bnb_nf4_transformer(
                repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                local_path="/tmp/snap",
                torch=self.torch,
                device="cuda",
            )
        self.assertIsNone(result)
        self.assertIn("falling back", note)


class InvokeLtxRefinerTests(unittest.TestCase):
    """Two-stage spatial upscale via LTXLatentUpsamplePipeline."""

    def _make_engine(self) -> DiffusersVideoEngine:
        engine = DiffusersVideoEngine()
        engine._device = "cpu"
        return engine

    def test_missing_upscaler_class_raises(self):
        engine = self._make_engine()
        fake_diffusers = SimpleNamespace()  # no LTXLatentUpsamplePipeline
        torch = SimpleNamespace(float32="fp32")
        with mock.patch.dict(
            "sys.modules",
            {
                "diffusers": fake_diffusers,
                "huggingface_hub": SimpleNamespace(
                    snapshot_download=lambda **_kw: "/tmp/upscaler"
                ),
            },
            clear=False,
        ):
            with self.assertRaises(RuntimeError):
                engine._invoke_pipeline_with_ltx_refiner(
                    pipeline=lambda **_kw: SimpleNamespace(frames=[1, 2, 3]),
                    kwargs={"prompt": "p"},
                    torch=torch,
                )

    def test_latents_none_raises(self):
        engine = self._make_engine()

        class _FakeUpscaler:
            @classmethod
            def from_pretrained(cls, *_a, **_kw):
                return cls()

        fake_diffusers = SimpleNamespace(LTXLatentUpsamplePipeline=_FakeUpscaler)
        torch = SimpleNamespace(float32="fp32")
        # Base pipeline returns no frames attribute → wrapper raises before
        # touching the upscaler.
        def _bad_base(**_kw):
            return SimpleNamespace(frames=None)

        with mock.patch.dict(
            "sys.modules",
            {
                "diffusers": fake_diffusers,
                "huggingface_hub": SimpleNamespace(
                    snapshot_download=lambda **_kw: "/tmp/upscaler"
                ),
            },
            clear=False,
        ):
            with self.assertRaises(RuntimeError):
                engine._invoke_pipeline_with_ltx_refiner(
                    pipeline=_bad_base,
                    kwargs={"prompt": "p"},
                    torch=torch,
                )

    def test_happy_path_returns_refined_frames(self):
        engine = self._make_engine()

        # Stub upscaler that records the latents it was handed and returns
        # a list of fake frames.
        captured: dict[str, Any] = {}

        class _FakeUpscaler:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                captured["upscaler_path"] = path
                captured["upscaler_kwargs"] = kwargs
                return cls()

            def __call__(self, *, latents):
                captured["latents"] = latents
                return SimpleNamespace(frames=[["frame-a", "frame-b"]])

            def to(self, _device):
                return self

        fake_diffusers = SimpleNamespace(LTXLatentUpsamplePipeline=_FakeUpscaler)
        torch = SimpleNamespace(float32="fp32")

        def _base_pipeline(**kwargs):
            captured["base_kwargs"] = kwargs
            return SimpleNamespace(frames="latents-tensor-stub")

        with mock.patch.dict(
            "sys.modules",
            {
                "diffusers": fake_diffusers,
                "huggingface_hub": SimpleNamespace(
                    snapshot_download=lambda **_kw: "/tmp/upscaler"
                ),
            },
            clear=False,
        ):
            frames = engine._invoke_pipeline_with_ltx_refiner(
                pipeline=_base_pipeline,
                kwargs={"prompt": "p", "num_inference_steps": 30},
                torch=torch,
            )

        self.assertEqual(frames, ["frame-a", "frame-b"])
        # Base run requested latents; upscaler got the latents tensor.
        self.assertEqual(captured["base_kwargs"]["output_type"], "latent")
        self.assertEqual(captured["latents"], "latents-tensor-stub")
        self.assertEqual(captured["upscaler_path"], "/tmp/upscaler")


class TryLoadGgufTransformerTests(unittest.TestCase):
    """Backfill for the GGUF transformer loader's failure paths."""

    def setUp(self) -> None:
        self.engine = DiffusersVideoEngine()
        self.torch = SimpleNamespace(bfloat16="bfloat16-sentinel")

    def test_missing_gguf_package(self):
        with mock.patch(
            "importlib.util.find_spec",
            side_effect=lambda name: None if name == "gguf" else mock.DEFAULT,
        ):
            result, note = self.engine._try_load_gguf_transformer(
                repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                gguf_repo="city96/Wan2.1-T2V-1.3B-gguf",
                gguf_file="wan2.1-t2v-1.3B-Q6_K.gguf",
                torch=self.torch,
            )
        self.assertIsNone(result)
        self.assertIn("gguf", note)

    def test_unmapped_repo(self):
        fake_diffusers = SimpleNamespace(GGUFQuantizationConfig=lambda **kw: None)
        with mock.patch("importlib.util.find_spec", return_value=object()), \
             mock.patch.dict(
                 "sys.modules",
                 {"diffusers": fake_diffusers},
                 clear=False,
             ):
            result, note = self.engine._try_load_gguf_transformer(
                repo="some/unknown-repo",
                gguf_repo="some/unknown-repo-gguf",
                gguf_file="x.gguf",
                torch=self.torch,
            )
        self.assertIsNone(result)
        self.assertIn("No GGUF transformer class", note)

    def test_happy_path_returns_transformer(self):
        captured: dict[str, Any] = {}

        class _FakeTransformer:
            @classmethod
            def from_single_file(cls, path, **kwargs):
                captured["path"] = path
                captured.update(kwargs)
                return SimpleNamespace(name="gguf-transformer")

        class _FakeQuantConfig:
            def __init__(self, **kwargs):
                captured["quant_kwargs"] = kwargs

        fake_diffusers = SimpleNamespace(
            GGUFQuantizationConfig=_FakeQuantConfig,
            WanTransformer3DModel=_FakeTransformer,
        )
        fake_hub = SimpleNamespace(
            hf_hub_download=lambda **kwargs: f"/tmp/{kwargs['filename']}"
        )
        with mock.patch("importlib.util.find_spec", return_value=object()), \
             mock.patch.dict(
                 "sys.modules",
                 {"diffusers": fake_diffusers, "huggingface_hub": fake_hub},
                 clear=False,
             ):
            result, note = self.engine._try_load_gguf_transformer(
                repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                gguf_repo="city96/Wan2.1-T2V-1.3B-gguf",
                gguf_file="wan2.1-t2v-1.3B-Q6_K.gguf",
                torch=self.torch,
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "gguf-transformer")
        self.assertIn("GGUF", note)
        self.assertEqual(captured["path"], "/tmp/wan2.1-t2v-1.3B-Q6_K.gguf")


if __name__ == "__main__":
    unittest.main()
