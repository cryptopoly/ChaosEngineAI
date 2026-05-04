"""Tests for stable-diffusion.cpp image runtime (FU-008 image subset).

Mirrors ``test_sdcpp_video.py``. Covers:
- Probe reports availability based on staged binary.
- Repo routing helper + supported-repo set (FLUX/SD3/SDXL/Qwen-Image/Z-Image).
- Preload/unload bookkeeping.
- Generate path: missing binary, unsupported repo, missing GGUF, CLI args,
  subprocess streaming, cancellation, output-missing, happy-path bytes.
- Manager dispatch routes ``config.runtime == "sdcpp"`` to the engine
  with diffusers fallback on failure.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from backend_service.image_runtime import (
    GeneratedImage,
    ImageGenerationConfig,
)
from backend_service.sdcpp_image_runtime import (
    SdCppImageEngine,
    _SUPPORTED_REPOS,
    _is_sdcpp_image_repo,
    _resolve_sd_binary,
    supported_repos,
)


def _make_config(
    repo: str = "black-forest-labs/FLUX.1-schnell",
    *,
    gguf_repo: str | None = "city96/FLUX.1-schnell-gguf",
    gguf_file: str | None = "flux1-schnell-Q4_K_M.gguf",
    runtime: str | None = "sdcpp",
    batch: int = 1,
) -> ImageGenerationConfig:
    return ImageGenerationConfig(
        modelId="sdcpp-img-test",
        modelName="test",
        repo=repo,
        prompt="a corgi astronaut on the moon",
        negativePrompt="",
        width=1024,
        height=1024,
        steps=4,
        guidance=3.5,
        batchSize=batch,
        seed=7,
        ggufRepo=gguf_repo,
        ggufFile=gguf_file,
        runtime=runtime,
    )


class SdCppImageSupportedReposTests(unittest.TestCase):
    def test_supported_repos_includes_flux1(self):
        repos = supported_repos()
        self.assertIn("black-forest-labs/FLUX.1-schnell", repos)
        self.assertIn("black-forest-labs/FLUX.1-dev", repos)

    def test_supported_repos_includes_sd3_sdxl(self):
        repos = supported_repos()
        self.assertIn("stabilityai/stable-diffusion-3.5-large", repos)
        self.assertIn("stabilityai/stable-diffusion-xl-base-1.0", repos)

    def test_supported_repos_includes_qwen_image(self):
        self.assertIn("Qwen/Qwen-Image", supported_repos())
        self.assertIn("Qwen/Qwen-Image-2512", supported_repos())

    def test_is_sdcpp_image_repo(self):
        self.assertTrue(_is_sdcpp_image_repo("black-forest-labs/FLUX.1-dev"))
        self.assertFalse(_is_sdcpp_image_repo("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))
        self.assertFalse(_is_sdcpp_image_repo(None))
        self.assertFalse(_is_sdcpp_image_repo(""))


class SdCppImageResolveBinaryTests(unittest.TestCase):
    def test_returns_none_when_no_env_no_managed(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHAOSENGINE_SDCPP_BIN", None)
            os.environ.pop("HOME", None)
            self.assertIsNone(_resolve_sd_binary())

    def test_returns_env_path_when_set(self):
        with patch.dict(os.environ, {}, clear=False):
            tmp = Path("/tmp/sdcpp-img-test-binary")
            tmp.write_text("")
            try:
                os.environ["CHAOSENGINE_SDCPP_BIN"] = str(tmp)
                self.assertEqual(_resolve_sd_binary(), tmp)
            finally:
                tmp.unlink(missing_ok=True)


class SdCppImageEngineProbeTests(unittest.TestCase):
    def test_probe_missing_binary(self):
        engine = SdCppImageEngine()
        with patch(
            "backend_service.sdcpp_image_runtime._resolve_sd_binary",
            return_value=None,
        ):
            probe = engine.probe()
        self.assertFalse(probe["available"])
        self.assertIn("not staged", probe["reason"])

    def test_probe_with_binary_reports_ready(self):
        engine = SdCppImageEngine()
        with patch(
            "backend_service.sdcpp_image_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ):
            probe = engine.probe()
        self.assertTrue(probe["available"])
        self.assertEqual(probe["binary"], "/tmp/sd")


class SdCppImageEnginePreloadTests(unittest.TestCase):
    def test_preload_supported_repo(self):
        engine = SdCppImageEngine()
        with patch(
            "backend_service.sdcpp_image_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ):
            engine.preload("black-forest-labs/FLUX.1-dev")
        self.assertEqual(engine._loaded_repo, "black-forest-labs/FLUX.1-dev")

    def test_preload_unsupported_repo_raises(self):
        engine = SdCppImageEngine()
        with self.assertRaises(RuntimeError) as ctx:
            engine.preload("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.assertIn("does not support", str(ctx.exception))

    def test_unload_clears_loaded(self):
        engine = SdCppImageEngine()
        engine._loaded_repo = "black-forest-labs/FLUX.1-dev"
        engine.unload()
        self.assertIsNone(engine._loaded_repo)


class SdCppImageEngineGenerateTests(unittest.TestCase):
    """Phase 4 / FU-008 image subset: generate() mirrors the video lane
    but emits a PNG via sd.cpp subprocess."""

    def test_generate_raises_when_binary_missing(self):
        engine = SdCppImageEngine()
        config = _make_config()
        with patch(
            "backend_service.sdcpp_image_runtime._resolve_sd_binary",
            return_value=None,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                engine.generate(config)
        self.assertIn("not staged", str(ctx.exception).lower())

    def test_generate_raises_for_unsupported_repo(self):
        engine = SdCppImageEngine()
        config = _make_config(repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        with patch(
            "backend_service.sdcpp_image_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                engine.generate(config)
        self.assertIn("does not support", str(ctx.exception))

    def test_generate_raises_when_gguf_file_missing(self):
        engine = SdCppImageEngine()
        config = _make_config(gguf_repo=None, gguf_file=None)
        with patch(
            "backend_service.sdcpp_image_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                engine.generate(config)
        self.assertIn("GGUF variant", str(ctx.exception))

    def test_build_cli_args_carries_image_flags_and_no_video_flags(self):
        engine = SdCppImageEngine()
        config = _make_config()
        args = engine._build_cli_args(
            binary=Path("/tmp/sd"),
            config=config,
            model_path="/tmp/flux.gguf",
            output_path=Path("/tmp/out.png"),
            seed=42,
        )
        self.assertEqual(args[0], "/tmp/sd")
        self.assertIn("--diffusion-model", args)
        self.assertIn("/tmp/flux.gguf", args)
        self.assertIn("-p", args)
        self.assertIn("a corgi astronaut on the moon", args)
        self.assertIn("-W", args)
        self.assertIn("1024", args)
        self.assertIn("--steps", args)
        self.assertIn("4", args)
        self.assertIn("--cfg-scale", args)
        self.assertIn("3.5", args)
        self.assertIn("--seed", args)
        self.assertIn("42", args)
        self.assertIn("-o", args)
        self.assertIn("/tmp/out.png", args)
        # Video-only flags must NOT leak into the image path.
        self.assertNotIn("--video-frames", args)
        self.assertNotIn("--fps", args)

    def test_build_cli_args_includes_negative_prompt_when_set(self):
        engine = SdCppImageEngine()
        config = ImageGenerationConfig(
            modelId="x", modelName="x",
            repo="black-forest-labs/FLUX.1-schnell",
            prompt="cat", negativePrompt="blurry, low quality",
            width=512, height=512, steps=4, guidance=4.0, batchSize=1, seed=1,
        )
        args = engine._build_cli_args(
            binary=Path("/tmp/sd"),
            config=config,
            model_path="/tmp/m.gguf",
            output_path=Path("/tmp/x.png"),
            seed=1,
        )
        self.assertIn("--negative-prompt", args)
        self.assertIn("blurry, low quality", args)

    def test_run_subprocess_streams_progress_and_returns_bytes(self):
        import tempfile
        engine = SdCppImageEngine()
        config = _make_config()
        tmpdir = tempfile.mkdtemp(prefix="sdcpp-img-test-")
        out_path = Path(tmpdir) / "fake.png"
        out_path.write_bytes(b"fake-png-bytes")

        class _FakeStdout:
            def __iter__(self):
                return iter([
                    "[INFO] step 1/4\n",
                    "[INFO] step 2/4\n",
                    "[INFO] done\n",
                ])

        mock_proc = MagicMock()
        mock_proc.stdout = _FakeStdout()
        mock_proc.wait.return_value = 0

        with patch(
            "backend_service.sdcpp_image_runtime.subprocess.Popen",
            return_value=mock_proc,
        ), patch("backend_service.progress.IMAGE_PROGRESS.set_step") as mock_set_step, \
             patch("backend_service.progress.IMAGE_PROGRESS.is_cancelled", return_value=False):
            data = engine._run_subprocess(
                args=["/tmp/sd", "--steps", "4"],
                config=config,
                output_path=out_path,
            )
        self.assertEqual(data, b"fake-png-bytes")
        self.assertEqual(mock_set_step.call_count, 2)

    def test_run_subprocess_raises_when_exit_code_nonzero(self):
        engine = SdCppImageEngine()
        config = _make_config()

        class _FakeStdout:
            def __iter__(self):
                return iter(["[ERROR] CUDA out of memory\n"])

        mock_proc = MagicMock()
        mock_proc.stdout = _FakeStdout()
        mock_proc.wait.return_value = 137

        with patch(
            "backend_service.sdcpp_image_runtime.subprocess.Popen",
            return_value=mock_proc,
        ), patch("backend_service.progress.IMAGE_PROGRESS.set_step"), \
             patch("backend_service.progress.IMAGE_PROGRESS.is_cancelled", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                engine._run_subprocess(
                    args=["/tmp/sd"],
                    config=config,
                    output_path=Path("/tmp/missing.png"),
                )
        msg = str(ctx.exception)
        self.assertIn("exited with code 137", msg)
        self.assertIn("CUDA out of memory", msg)

    def test_run_subprocess_raises_when_output_missing(self):
        engine = SdCppImageEngine()
        config = _make_config()

        class _FakeStdout:
            def __iter__(self):
                return iter(["[INFO] step 1/1 done\n"])

        mock_proc = MagicMock()
        mock_proc.stdout = _FakeStdout()
        mock_proc.wait.return_value = 0
        with patch(
            "backend_service.sdcpp_image_runtime.subprocess.Popen",
            return_value=mock_proc,
        ), patch("backend_service.progress.IMAGE_PROGRESS.set_step"), \
             patch("backend_service.progress.IMAGE_PROGRESS.is_cancelled", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                engine._run_subprocess(
                    args=["/tmp/sd"],
                    config=config,
                    output_path=Path("/tmp/never-written.png"),
                )
        self.assertIn("output file", str(ctx.exception).lower())

    def test_run_subprocess_terminates_on_cancel(self):
        engine = SdCppImageEngine()
        config = _make_config()

        class _FakeStdout:
            def __iter__(self):
                return iter(["[INFO] step 1/4\n"])

        mock_proc = MagicMock()
        mock_proc.stdout = _FakeStdout()
        mock_proc.wait.return_value = 0
        with patch(
            "backend_service.sdcpp_image_runtime.subprocess.Popen",
            return_value=mock_proc,
        ), patch("backend_service.progress.IMAGE_PROGRESS.set_step"), \
             patch(
                 "backend_service.progress.IMAGE_PROGRESS.is_cancelled",
                 return_value=True,
             ):
            with self.assertRaises(RuntimeError) as ctx:
                engine._run_subprocess(
                    args=["/tmp/sd"],
                    config=config,
                    output_path=Path("/tmp/cancelled.png"),
                )
        self.assertIn("cancelled", str(ctx.exception).lower())
        mock_proc.terminate.assert_called()

    def test_generate_happy_path_returns_generated_image(self):
        engine = SdCppImageEngine()
        config = _make_config()

        class _FakeStdout:
            def __iter__(self):
                return iter(["[INFO] step 1/4\n", "[INFO] step 4/4\n"])

        captured: dict[str, Any] = {}

        def _popen_factory(args, **kwargs):
            captured["args"] = args
            output = Path(args[args.index("-o") + 1])
            output.write_bytes(b"deadbeef-png-bytes")
            mock_proc = MagicMock()
            mock_proc.stdout = _FakeStdout()
            mock_proc.wait.return_value = 0
            return mock_proc

        with patch(
            "backend_service.sdcpp_image_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ), patch(
            "backend_service.sdcpp_image_runtime.SdCppImageEngine._resolve_gguf_path",
            return_value="/tmp/flux.gguf",
        ), patch(
            "backend_service.sdcpp_image_runtime.subprocess.Popen",
            side_effect=_popen_factory,
        ), patch("backend_service.progress.IMAGE_PROGRESS.set_step"), \
             patch("backend_service.progress.IMAGE_PROGRESS.is_cancelled", return_value=False):
            results = engine.generate(config)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIsInstance(result, GeneratedImage)
        self.assertEqual(result.bytes, b"deadbeef-png-bytes")
        self.assertEqual(result.extension, "png")
        self.assertEqual(result.mimeType, "image/png")
        self.assertEqual(result.runtimeLabel, "stable-diffusion.cpp")
        self.assertIsNotNone(result.runtimeNote)
        self.assertIn("/tmp/flux.gguf", captured["args"])
        self.assertIn("a corgi astronaut on the moon", captured["args"])

    def test_generate_batch_produces_one_image_per_seed(self):
        engine = SdCppImageEngine()
        config = _make_config(batch=3)

        seen_seeds: list[int] = []

        class _FakeStdout:
            def __iter__(self):
                return iter(["[INFO] step 1/4\n"])

        def _popen_factory(args, **kwargs):
            seen_seeds.append(int(args[args.index("--seed") + 1]))
            output = Path(args[args.index("-o") + 1])
            output.write_bytes(b"img")
            mock_proc = MagicMock()
            mock_proc.stdout = _FakeStdout()
            mock_proc.wait.return_value = 0
            return mock_proc

        with patch(
            "backend_service.sdcpp_image_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ), patch(
            "backend_service.sdcpp_image_runtime.SdCppImageEngine._resolve_gguf_path",
            return_value="/tmp/flux.gguf",
        ), patch(
            "backend_service.sdcpp_image_runtime.subprocess.Popen",
            side_effect=_popen_factory,
        ), patch("backend_service.progress.IMAGE_PROGRESS.set_step"), \
             patch("backend_service.progress.IMAGE_PROGRESS.is_cancelled", return_value=False):
            results = engine.generate(config)

        self.assertEqual(len(results), 3)
        # Each batch index should advance the seed by 1.
        self.assertEqual(seen_seeds, [7, 8, 9])
        # Outputs carry the matching seeds.
        self.assertEqual([r.seed for r in results], [7, 8, 9])


class ImageRuntimeManagerSdCppDispatchTests(unittest.TestCase):
    """Manager routes ``runtime="sdcpp"`` to the engine and falls back
    to diffusers on probe failure or runtime error."""

    def test_manager_has_sdcpp_engine_field(self):
        from backend_service.image_runtime import ImageRuntimeManager
        manager = ImageRuntimeManager()
        self.assertIsNotNone(manager._sdcpp)
        self.assertEqual(manager._sdcpp.runtime_label, "stable-diffusion.cpp")

    def test_manager_falls_back_to_diffusers_when_sdcpp_unavailable(self):
        from backend_service.image_runtime import ImageRuntimeManager
        manager = ImageRuntimeManager()
        config = _make_config()

        # sd.cpp binary missing → probe returns available=False → manager
        # should fall through to diffusers (which we stub to also fail
        # cleanly so we can assert the dispatch path).
        sdcpp_probe = MagicMock(return_value={
            "available": False,
            "reason": "stable-diffusion.cpp binary not staged.",
        })
        manager._sdcpp.probe = sdcpp_probe  # type: ignore[method-assign]
        sdcpp_generate = MagicMock(side_effect=AssertionError("must not be called"))
        manager._sdcpp.generate = sdcpp_generate  # type: ignore[method-assign]

        # Stub diffusers.probe to look ready, then have generate raise
        # so the manager falls into the placeholder path. We're not
        # exercising the placeholder; we just want to confirm the sd.cpp
        # branch hands off cleanly without invoking ``generate``.
        from backend_service.image_runtime import ImageRuntimeStatus
        diffusers_status = ImageRuntimeStatus(
            activeEngine="diffusers",
            realGenerationAvailable=True,
            device="mps",
            pythonExecutable=None,
            missingDependencies=[],
            loadedModelRepo=None,
            message="diffusers ready",
        )
        manager._diffusers.probe = MagicMock(return_value=diffusers_status)  # type: ignore[method-assign]
        manager._diffusers.generate = MagicMock(side_effect=RuntimeError("stubbed"))  # type: ignore[method-assign]
        manager._placeholder.generate = MagicMock(return_value=[
            GeneratedImage(
                seed=1, bytes=b"x", extension="png", mimeType="image/png",
                durationSeconds=0.1, runtimeLabel="placeholder",
            )
        ])  # type: ignore[method-assign]

        images, status = manager.generate(config)
        sdcpp_probe.assert_called()
        sdcpp_generate.assert_not_called()
        self.assertEqual(len(images), 1)
        self.assertEqual(status["activeEngine"], "placeholder")

    def test_manager_uses_sdcpp_when_probe_ready(self):
        from backend_service.image_runtime import ImageRuntimeManager
        manager = ImageRuntimeManager()
        config = _make_config()

        manager._sdcpp.probe = MagicMock(return_value={  # type: ignore[method-assign]
            "available": True,
            "reason": None,
            "binary": "/tmp/sd",
            "device": "mps",
        })
        sample_image = GeneratedImage(
            seed=42, bytes=b"sd-png-bytes", extension="png",
            mimeType="image/png", durationSeconds=4.5,
            runtimeLabel="stable-diffusion.cpp",
        )
        manager._sdcpp.generate = MagicMock(return_value=[sample_image])  # type: ignore[method-assign]

        # Stub diffusers probe so the manager can build the status dict.
        from backend_service.image_runtime import ImageRuntimeStatus
        manager._diffusers.probe = MagicMock(return_value=ImageRuntimeStatus(  # type: ignore[method-assign]
            activeEngine="diffusers",
            realGenerationAvailable=True,
            device="mps",
            pythonExecutable=None,
            missingDependencies=[],
            loadedModelRepo=None,
            message="diffusers ready",
        ))

        images, status = manager.generate(config)
        self.assertEqual(images, [sample_image])
        self.assertEqual(status["activeEngine"], "sd.cpp")


class SdCppImageCatalogTests(unittest.TestCase):
    """Catalog must carry ``engine="sdcpp"`` + ``ggufRepo`` + ``ggufFile``
    on the variants that route to this engine."""

    def test_catalog_has_sdcpp_variants(self):
        from backend_service.catalog.image_models import IMAGE_MODEL_FAMILIES
        sdcpp_variants = [
            v for f in IMAGE_MODEL_FAMILIES for v in f.get("variants", [])
            if v.get("engine") == "sdcpp"
        ]
        self.assertGreaterEqual(len(sdcpp_variants), 2)
        for variant in sdcpp_variants:
            self.assertIn(variant.get("repo"), supported_repos())
            self.assertTrue(variant.get("ggufRepo"))
            self.assertTrue(variant.get("ggufFile"))


if __name__ == "__main__":
    unittest.main()
