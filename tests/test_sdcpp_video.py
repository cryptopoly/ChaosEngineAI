"""Tests for stable-diffusion.cpp video runtime (FU-008 scaffold).

Covers:
- Probe reports ``missingDependencies=["sd"]`` when binary not staged.
- Probe reports the staged binary path when ``CHAOSENGINE_SDCPP_BIN`` set.
- Repo routing helper + supported-repo set (Wan 2.1 / 2.2 diffusers ids).
- Preload/unload bookkeeping.
- ``generate()`` raises ``NotImplementedError`` (scaffold gate).
- Manager exposes ``sdcpp_video_capabilities()``.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from backend_service.sdcpp_video_runtime import (
    SdCppVideoEngine,
    _SUPPORTED_REPOS,
    _is_sdcpp_video_repo,
    _resolve_sd_binary,
    supported_repos,
)
from backend_service.video_runtime import (
    VideoGenerationConfig,
    VideoRuntimeManager,
)


def _make_config(repo: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers") -> VideoGenerationConfig:
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

    def test_probe_with_binary_still_scaffold(self):
        engine = SdCppVideoEngine()
        with patch(
            "backend_service.sdcpp_video_runtime._resolve_sd_binary",
            return_value=Path("/tmp/sd"),
        ):
            status = engine.probe()
        # Binary present but generate() not wired yet → False
        self.assertFalse(status.realGenerationAvailable)
        self.assertIn("scaffold", status.message.lower())


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
    def test_generate_raises_not_implemented(self):
        engine = SdCppVideoEngine()
        config = _make_config()
        with self.assertRaises(NotImplementedError) as ctx:
            engine.generate(config)
        self.assertIn("scaffold", str(ctx.exception).lower())


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
