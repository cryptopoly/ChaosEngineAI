"""Tests for mlx-video Apple Silicon runtime scaffold (FU-009).

Probe-only scope — generation raises ``NotImplementedError`` until the
follow-up promotes from scaffold. These tests pin:

- Platform gating (Darwin arm64 only).
- Install-state probe (``missingDependencies`` when ``mlx_video`` is
  not importable).
- Repo routing helper + supported-repo set.
- Preload/unload bookkeeping.
- ``generate()`` raises with a clear pointer to FU-009.
- Manager exposes ``mlx_video_capabilities()``.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from backend_service.mlx_video_runtime import (
    MlxVideoEngine,
    _SUPPORTED_REPOS,
    _is_mlx_video_repo,
    supported_repos,
)
from backend_service.video_runtime import (
    VideoGenerationConfig,
    VideoRuntimeManager,
)


def _make_config(repo: str = "Lightricks/LTX-2-19B") -> VideoGenerationConfig:
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
    )


class MlxVideoSupportedReposTests(unittest.TestCase):
    def test_supported_repos_snapshot(self):
        repos = supported_repos()
        # LTX-2 + Wan2.1 1.3B/14B + Wan2.2 T2V/TI2V/I2V
        self.assertIn("Lightricks/LTX-2-19B", repos)
        self.assertIn("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", repos)
        self.assertIn("Wan-AI/Wan2.1-T2V-14B-Diffusers", repos)
        self.assertIn("Wan-AI/Wan2.2-T2V-A14B-Diffusers", repos)
        self.assertIn("Wan-AI/Wan2.2-TI2V-5B-Diffusers", repos)
        self.assertIn("Wan-AI/Wan2.2-I2V-A14B-Diffusers", repos)

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
        self.assertFalse(_is_mlx_video_repo(None))
        self.assertFalse(_is_mlx_video_repo(""))


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

    def test_probe_reports_scaffold_when_installed(self):
        engine = MlxVideoEngine()
        fake_spec = object()  # non-None sentinel
        with patch("backend_service.mlx_video_runtime.platform.system", return_value="Darwin"), \
             patch("backend_service.mlx_video_runtime.platform.machine", return_value="arm64"), \
             patch("backend_service.mlx_video_runtime.importlib.util.find_spec", return_value=fake_spec):
            status = engine.probe()
        # Installed but gen path is scaffold-only — must not advertise ready.
        self.assertFalse(status.realGenerationAvailable)
        self.assertEqual(status.device, "mps")
        self.assertEqual(status.expectedDevice, "mps")
        self.assertIn("scaffold-only", status.message)


class MlxVideoLifecycleTests(unittest.TestCase):
    def test_preload_rejects_unsupported_repo(self):
        engine = MlxVideoEngine()
        with self.assertRaises(RuntimeError) as ctx:
            engine.preload("genmo/mochi-1-preview")
        self.assertIn("does not support", str(ctx.exception))

    def test_preload_remembers_supported_repo(self):
        engine = MlxVideoEngine()
        with patch("backend_service.mlx_video_runtime.platform.system", return_value="Darwin"), \
             patch("backend_service.mlx_video_runtime.platform.machine", return_value="arm64"), \
             patch("backend_service.mlx_video_runtime.importlib.util.find_spec", return_value=object()):
            status = engine.preload("Lightricks/LTX-2-19B")
        self.assertEqual(status.loadedModelRepo, "Lightricks/LTX-2-19B")

    def test_unload_clears_matching_repo(self):
        engine = MlxVideoEngine()
        with patch("backend_service.mlx_video_runtime.platform.system", return_value="Darwin"), \
             patch("backend_service.mlx_video_runtime.platform.machine", return_value="arm64"), \
             patch("backend_service.mlx_video_runtime.importlib.util.find_spec", return_value=object()):
            engine.preload("Lightricks/LTX-2-19B")
            status = engine.unload("Lightricks/LTX-2-19B")
        self.assertIsNone(status.loadedModelRepo)

    def test_unload_noop_when_repo_mismatch(self):
        engine = MlxVideoEngine()
        with patch("backend_service.mlx_video_runtime.platform.system", return_value="Darwin"), \
             patch("backend_service.mlx_video_runtime.platform.machine", return_value="arm64"), \
             patch("backend_service.mlx_video_runtime.importlib.util.find_spec", return_value=object()):
            engine.preload("Lightricks/LTX-2-19B")
            status = engine.unload("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        # Different repo → keep the loaded one.
        self.assertEqual(status.loadedModelRepo, "Lightricks/LTX-2-19B")


class MlxVideoGenerateRaisesTests(unittest.TestCase):
    def test_generate_raises_not_implemented(self):
        engine = MlxVideoEngine()
        with self.assertRaises(NotImplementedError) as ctx:
            engine.generate(_make_config())
        msg = str(ctx.exception)
        self.assertIn("FU-009", msg)
        self.assertIn("scaffold", msg)


class VideoManagerExposesMlxVideoTests(unittest.TestCase):
    def test_manager_exposes_mlx_video_capabilities(self):
        manager = VideoRuntimeManager()
        caps = manager.mlx_video_capabilities()
        self.assertIsInstance(caps, dict)
        self.assertEqual(caps["activeEngine"], "mlx-video")
        # realGenerationAvailable is always False in this phase.
        self.assertFalse(caps["realGenerationAvailable"])

    def test_manager_lazy_constructs_mlx_video(self):
        manager = VideoRuntimeManager()
        self.assertIsNone(manager._mlx_video)
        manager.mlx_video_capabilities()
        self.assertIsNotNone(manager._mlx_video)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
