"""Tests for stable-diffusion.cpp video runtime (FU-008).

Covers:
- Probe reports ``missingDependencies=["sd"]`` when binary not staged.
- Probe reports ``realGenerationAvailable=True`` once the binary is staged.
- Repo routing helper + supported-repo set (Wan 2.1 / 2.2 diffusers ids).
- Preload/unload bookkeeping.
- ``generate()`` builds CLI args, spawns the subprocess, streams stdout
  into ``VIDEO_PROGRESS``, and returns a populated ``GeneratedVideo``.
- Manager exposes ``sdcpp_video_capabilities()``.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from backend_service.sdcpp_video_runtime import (
    SdCppVideoEngine,
    _SUPPORTED_REPOS,
    _is_sdcpp_video_repo,
    _resolve_sd_binary,
    supported_repos,
)
from backend_service.video_runtime import (
    GeneratedVideo,
    VideoGenerationConfig,
    VideoRuntimeManager,
)


def _make_config(
    repo: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    *,
    gguf_repo: str | None = "city96/Wan2.1-T2V-1.3B-gguf",
    gguf_file: str | None = "wan2.1-t2v-1.3B-Q4_K_M.gguf",
) -> VideoGenerationConfig:
    return VideoGenerationConfig(
        modelId="sdcpp-test",
        modelName="test",
        repo=repo,
        prompt="a corgi running",
        negativePrompt="",
        width=832,
        height=480,
        numFrames=25,
        fps=24,
        guidance=6.0,
        steps=30,
        seed=7,
        ggufRepo=gguf_repo,
        ggufFile=gguf_file,
    )


class SdCppSupportedReposTests(unittest.TestCase):
    def test_supported_repos_includes_wan_2_1(self):
        repos = supported_repos()
        self.assertIn("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", repos)
        self.assertIn("Wan-AI/Wan2.1-T2V-14B-Diffusers", repos)

    def test_supported_repos_includes_wan_2_2(self):
        repos = supported_repos()
        self.assertIn("Wan-AI/Wan2.2-TI2V-5B-Diffusers", repos)
        self.assertIn("Wan-AI/Wan2.2-T2V-A14B-Diffusers", repos)

    def test_is_sdcpp_video_repo(self):
        self.assertTrue(_is_sdcpp_video_repo("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))
        self.assertFalse(_is_sdcpp_video_repo("prince-canuma/LTX-2-distilled"))
        self.assertFalse(_is_sdcpp_video_repo(None))
        self.assertFalse(_is_sdcpp_video_repo(""))


class SdCppResolveBinaryTests(unittest.TestCase):
    def test_returns_none_when_no_env_no_managed(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHAOSENGINE_SDCPP_BIN", None)
            os.environ.pop("HOME", None)
            self.assertIsNone(_resolve_sd_binary())

    def test_returns_env_path_when_set(self):
        with patch.dict(os.environ, {}, clear=False):
            tmp = Path("/tmp/sdcpp-test-binary")
            tmp.write_text("")
            try:
                os.environ["CHAOSENGINE_SDCPP_BIN"] = str(tmp)
                self.assertEqual(_resolve_sd_binary(), tmp)
            finally:
                tmp.unlink(missing_ok=True)

    def test_skips_env_path_when_missing(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ["CHAOSENGINE_SDCPP_BIN"] = "/nonexistent/sd-binary"
            os.environ["HOME"] = "/tmp/no-such-home"
            self.assertIsNone(_resolve_sd_binary())


class SdCppEngineProbeTests(unittest.TestCase):
    def test_probe_missing_binary(self):
        engine = SdCppVideoEngine()
        with patch(
            "backend_service.sdcpp_video_runtime._resolve_sd_binary",
            return_value=None,
        ):
            status = engine.probe()
        self.assertFalse(status.realGenerationAvailable)
        self.assertEqual(status.missingDependencies, ["sd"])
        self.assertEqual(status.activeEngine, "sd.cpp")

    def test_probe_with_binary_reports_ready(self):
        engine = SdCppVideoEngine()
        with patch(
            "backend_service.sdcpp_video_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ):
            status = engine.probe()
        # Phase 3: generate() now wired, so binary-present means ready.
        self.assertTrue(status.realGenerationAvailable)
        self.assertIn("generate path active", status.message.lower())


class SdCppEnginePreloadTests(unittest.TestCase):
    def test_preload_supported_repo(self):
        engine = SdCppVideoEngine()
        with patch(
            "backend_service.sdcpp_video_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ):
            status = engine.preload("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.assertEqual(engine._loaded_repo, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.assertEqual(status.loadedModelRepo, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    def test_preload_unsupported_repo_raises(self):
        engine = SdCppVideoEngine()
        with self.assertRaises(RuntimeError) as ctx:
            engine.preload("Lightricks/LTX-Video")
        self.assertIn("does not support", str(ctx.exception))

    def test_unload_clears_loaded(self):
        engine = SdCppVideoEngine()
        engine._loaded_repo = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        with patch(
            "backend_service.sdcpp_video_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ):
            engine.unload()
        self.assertIsNone(engine._loaded_repo)


class SdCppEngineGenerateTests(unittest.TestCase):
    """Phase 3 / FU-008: generate() now spawns sd.cpp subprocess."""

    def test_generate_raises_when_binary_missing(self):
        engine = SdCppVideoEngine()
        config = _make_config()
        with patch(
            "backend_service.sdcpp_video_runtime._resolve_sd_binary",
            return_value=None,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                engine.generate(config)
        self.assertIn("not staged", str(ctx.exception).lower())

    def test_generate_raises_for_unsupported_repo(self):
        engine = SdCppVideoEngine()
        config = _make_config(repo="Lightricks/LTX-Video")
        with patch(
            "backend_service.sdcpp_video_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                engine.generate(config)
        self.assertIn("does not support", str(ctx.exception))

    def test_generate_raises_when_gguf_file_missing(self):
        engine = SdCppVideoEngine()
        config = _make_config(gguf_repo=None, gguf_file=None)
        with patch(
            "backend_service.sdcpp_video_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                engine.generate(config)
        self.assertIn("GGUF variant", str(ctx.exception))

    def test_build_cli_args_carries_all_required_flags(self):
        engine = SdCppVideoEngine()
        config = _make_config()
        args = engine._build_cli_args(
            binary=Path("/tmp/sd"),
            config=config,
            model_path="/tmp/wan.gguf",
            output_path=Path("/tmp/out.mp4"),
            seed=42,
        )
        self.assertEqual(args[0], "/tmp/sd")
        self.assertIn("--diffusion-model", args)
        self.assertIn("/tmp/wan.gguf", args)
        self.assertIn("-p", args)
        self.assertIn("a corgi running", args)
        self.assertIn("-W", args)
        self.assertIn("832", args)
        self.assertIn("-H", args)
        self.assertIn("480", args)
        self.assertIn("--steps", args)
        self.assertIn("30", args)
        self.assertIn("--cfg-scale", args)
        self.assertIn("6", args)
        self.assertIn("--seed", args)
        self.assertIn("42", args)
        self.assertIn("-o", args)
        self.assertIn("/tmp/out.mp4", args)
        self.assertIn("--video-frames", args)
        self.assertIn("25", args)
        self.assertIn("--fps", args)
        self.assertIn("24", args)

    def test_build_cli_args_includes_negative_prompt_when_set(self):
        engine = SdCppVideoEngine()
        config = VideoGenerationConfig(
            modelId="x",
            modelName="x",
            repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            prompt="cat",
            negativePrompt="blurry",
            width=512,
            height=512,
            numFrames=8,
            fps=8,
            guidance=4.0,
            steps=4,
            seed=1,
        )
        args = engine._build_cli_args(
            binary=Path("/tmp/sd"),
            config=config,
            model_path="/tmp/m.gguf",
            output_path=Path("/tmp/x.mp4"),
            seed=1,
        )
        self.assertIn("--negative-prompt", args)
        self.assertIn("blurry", args)

    def test_run_subprocess_streams_progress_and_returns_bytes(self):
        engine = SdCppVideoEngine()
        config = _make_config()

        # Output path: write a small payload before the subprocess returns
        # so the post-run read picks something up.
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="sdcpp-test-")
        out_path = Path(tmpdir) / "fake.webm"
        out_path.write_bytes(b"fake-webm-bytes")

        # Mock subprocess.Popen with a stdout iterator that emits two
        # progress-style lines plus a benign info line.
        class _FakeStdout:
            def __init__(self, lines: list[str]) -> None:
                self._iter = iter(lines)

            def __iter__(self):
                return self._iter

        mock_proc = MagicMock()
        mock_proc.stdout = _FakeStdout(
            ["[INFO] step 1/4 processing\n", "[INFO] step 2/4 processing\n", "[INFO] done\n"]
        )
        mock_proc.wait.return_value = 0

        with patch(
            "backend_service.sdcpp_video_runtime.subprocess.Popen",
            return_value=mock_proc,
        ) as mock_popen, \
             patch("backend_service.progress.VIDEO_PROGRESS.set_step") as mock_set_step, \
             patch("backend_service.progress.VIDEO_PROGRESS.is_cancelled", return_value=False):
            data = engine._run_subprocess(
                args=["/tmp/sd", "--steps", "4"],
                config=config,
                output_path=out_path,
            )

        self.assertEqual(data, b"fake-webm-bytes")
        mock_popen.assert_called_once()
        # Two step lines should produce two set_step calls with totals.
        self.assertEqual(mock_set_step.call_count, 2)
        first = mock_set_step.call_args_list[0]
        self.assertEqual(first.args, (1,))
        self.assertEqual(first.kwargs.get("total"), 4)

    def test_run_subprocess_raises_when_exit_code_nonzero(self):
        engine = SdCppVideoEngine()
        config = _make_config()

        class _FakeStdout:
            def __iter__(self):
                return iter(["[ERROR] CUDA out of memory\n"])

        mock_proc = MagicMock()
        mock_proc.stdout = _FakeStdout()
        mock_proc.wait.return_value = 137  # OOM kill code
        with patch(
            "backend_service.sdcpp_video_runtime.subprocess.Popen",
            return_value=mock_proc,
        ), \
             patch("backend_service.progress.VIDEO_PROGRESS.set_step"), \
             patch("backend_service.progress.VIDEO_PROGRESS.is_cancelled", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                engine._run_subprocess(
                    args=["/tmp/sd"],
                    config=config,
                    output_path=Path("/tmp/missing.mp4"),
                )
        msg = str(ctx.exception)
        self.assertIn("exited with code 137", msg)
        self.assertIn("CUDA out of memory", msg)

    def test_run_subprocess_raises_when_output_missing(self):
        engine = SdCppVideoEngine()
        config = _make_config()

        class _FakeStdout:
            def __iter__(self):
                return iter(["[INFO] step 1/1 done\n"])

        mock_proc = MagicMock()
        mock_proc.stdout = _FakeStdout()
        mock_proc.wait.return_value = 0
        with patch(
            "backend_service.sdcpp_video_runtime.subprocess.Popen",
            return_value=mock_proc,
        ), \
             patch("backend_service.progress.VIDEO_PROGRESS.set_step"), \
             patch("backend_service.progress.VIDEO_PROGRESS.is_cancelled", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                engine._run_subprocess(
                    args=["/tmp/sd"],
                    config=config,
                    output_path=Path("/tmp/never-written.mp4"),
                )
        self.assertIn("output file", str(ctx.exception).lower())

    def test_run_subprocess_terminates_on_cancel(self):
        engine = SdCppVideoEngine()
        config = _make_config()

        class _FakeStdout:
            def __iter__(self):
                return iter(["[INFO] step 1/4\n", "[INFO] step 2/4\n"])

        mock_proc = MagicMock()
        mock_proc.stdout = _FakeStdout()
        mock_proc.wait.return_value = 0
        with patch(
            "backend_service.sdcpp_video_runtime.subprocess.Popen",
            return_value=mock_proc,
        ), \
             patch("backend_service.progress.VIDEO_PROGRESS.set_step"), \
             patch(
                 "backend_service.progress.VIDEO_PROGRESS.is_cancelled",
                 return_value=True,
             ):
            with self.assertRaises(RuntimeError) as ctx:
                engine._run_subprocess(
                    args=["/tmp/sd"],
                    config=config,
                    output_path=Path("/tmp/cancelled.mp4"),
                )
        self.assertIn("cancelled", str(ctx.exception).lower())
        mock_proc.terminate.assert_called()

    def test_generate_happy_path_returns_generated_video(self):
        engine = SdCppVideoEngine()
        config = _make_config()

        class _FakeStdout:
            def __iter__(self):
                return iter(["[INFO] step 1/4\n", "[INFO] step 4/4\n"])

        # generate() spawns the subprocess inside a TemporaryDirectory.
        # Pre-write the expected output by stubbing subprocess.Popen
        # with a side effect that creates the file.
        captured: dict[str, Any] = {}

        def _popen_factory(args, **kwargs):
            captured["args"] = args
            # Path is the value passed via -o; create it now so
            # output_path.exists() is True after the loop.
            output = Path(args[args.index("-o") + 1])
            output.write_bytes(b"deadbeef-webm-bytes")
            mock_proc = MagicMock()
            mock_proc.stdout = _FakeStdout()
            mock_proc.wait.return_value = 0
            return mock_proc

        with patch(
            "backend_service.sdcpp_video_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ), patch(
            "backend_service.sdcpp_video_runtime.SdCppVideoEngine._resolve_gguf_path",
            return_value="/tmp/wan.gguf",
        ), patch(
            "backend_service.sdcpp_video_runtime.subprocess.Popen",
            side_effect=_popen_factory,
        ), patch("backend_service.progress.VIDEO_PROGRESS.set_step"), \
             patch("backend_service.progress.VIDEO_PROGRESS.is_cancelled", return_value=False):
            result = engine.generate(config)

        self.assertIsInstance(result, GeneratedVideo)
        self.assertEqual(result.bytes, b"deadbeef-webm-bytes")
        self.assertEqual(result.frameCount, 25)
        self.assertEqual(result.fps, 24)
        self.assertEqual(result.width, 832)
        self.assertEqual(result.height, 480)
        self.assertEqual(result.extension, "webm")
        self.assertEqual(result.mimeType, "video/webm")
        self.assertEqual(result.runtimeLabel, "stable-diffusion.cpp")
        self.assertIsNotNone(result.runtimeNote)
        self.assertIn("/tmp/wan.gguf", captured["args"])
        self.assertIn("a corgi running", captured["args"])


class SdCppManagerCapabilitiesTests(unittest.TestCase):
    def test_capabilities_exposed(self):
        manager = VideoRuntimeManager()
        with patch(
            "backend_service.sdcpp_video_runtime._resolve_sd_binary",
            return_value=None,
        ):
            caps = manager.sdcpp_video_capabilities()
        self.assertEqual(caps["activeEngine"], "sd.cpp")
        self.assertFalse(caps["realGenerationAvailable"])
        self.assertEqual(caps.get("missingDependencies"), ["sd"])

    def test_is_sdcpp_video_repo_helper(self):
        manager = VideoRuntimeManager()
        self.assertTrue(
            manager._is_sdcpp_video_repo("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        )
        self.assertFalse(manager._is_sdcpp_video_repo("Lightricks/LTX-Video"))
        self.assertFalse(manager._is_sdcpp_video_repo(None))


if __name__ == "__main__":
    unittest.main()
