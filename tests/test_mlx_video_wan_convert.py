"""Tests for FU-025: mlx-video Wan2.1/2.2 convert wrapper.

Covers the helper plumbing — ``slug_for`` / ``output_dir_for`` /
``is_supported_raw_repo`` / ``status_for`` / ``list_converted`` /
``run_convert``. The actual upstream
``mlx_video.models.wan_2.convert.convert_wan_checkpoint`` is mocked
via ``subprocess.run`` so the suite runs without mlx-video installed
and without raw Wan weights on disk (Wan2.1 1.3B is ~3 GB; A14B is
~67 GB — not test fixtures).
"""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend_service import mlx_video_wan_convert as wan_convert
from backend_service.mlx_video_wan_convert import (
    SUPPORTED_RAW_REPOS,
    WanConvertStatus,
    is_mlx_video_available,
    is_supported_raw_repo,
    list_converted,
    output_dir_for,
    run_convert,
    slug_for,
    status_for,
)


class SlugTests(unittest.TestCase):
    def test_slug_replaces_slash_with_double_underscore(self):
        self.assertEqual(slug_for("Wan-AI/Wan2.1-T2V-1.3B"), "Wan-AI__Wan2.1-T2V-1.3B")

    def test_slug_round_trips_via_name_to_repo(self):
        for repo in SUPPORTED_RAW_REPOS:
            slug = slug_for(repo)
            self.assertNotIn("/", slug)
            # Reverse: split on first __ recovers the repo.
            self.assertEqual(slug.replace("__", "/", 1), repo)

    def test_output_dir_under_convert_root(self):
        path = output_dir_for("Wan-AI/Wan2.2-TI2V-5B")
        self.assertEqual(path.name, "Wan-AI__Wan2.2-TI2V-5B")
        self.assertEqual(path.parent.name, "mlx-video-wan")


class IsSupportedRawRepoTests(unittest.TestCase):
    def test_recognises_known_wan_repos(self):
        self.assertTrue(is_supported_raw_repo("Wan-AI/Wan2.1-T2V-1.3B"))
        self.assertTrue(is_supported_raw_repo("Wan-AI/Wan2.2-T2V-A14B"))
        self.assertTrue(is_supported_raw_repo("Wan-AI/Wan2.2-I2V-A14B"))

    def test_rejects_diffusers_mirrors(self):
        # The -Diffusers mirrors go through the diffusers path; the
        # upstream convert script cannot handle their layout.
        self.assertFalse(is_supported_raw_repo("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"))
        self.assertFalse(is_supported_raw_repo("Wan-AI/Wan2.2-TI2V-5B-Diffusers"))

    def test_rejects_other_video_models(self):
        self.assertFalse(is_supported_raw_repo("Lightricks/LTX-Video"))
        self.assertFalse(is_supported_raw_repo("genmo/mochi-1-preview"))
        self.assertFalse(is_supported_raw_repo("THUDM/CogVideoX-2b"))
        self.assertFalse(is_supported_raw_repo(None))
        self.assertFalse(is_supported_raw_repo(""))


class StatusForTests(unittest.TestCase):
    def setUp(self):
        # Redirect CONVERT_ROOT to a tempdir for each test.
        import tempfile
        self.tmpdir = tempfile.mkdtemp(prefix="chaosengine-wan-test-")
        self._orig_root = wan_convert.CONVERT_ROOT
        wan_convert.CONVERT_ROOT = Path(self.tmpdir)

    def tearDown(self):
        wan_convert.CONVERT_ROOT = self._orig_root
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_status_when_output_dir_missing(self):
        status = status_for("Wan-AI/Wan2.1-T2V-1.3B")
        self.assertFalse(status.converted)
        self.assertFalse(status.hasTransformer)
        self.assertFalse(status.hasVae)
        self.assertIn("does not exist", status.note)

    def test_status_when_only_dir_exists(self):
        out = output_dir_for("Wan-AI/Wan2.1-T2V-1.3B")
        out.mkdir(parents=True)
        status = status_for("Wan-AI/Wan2.1-T2V-1.3B")
        self.assertFalse(status.converted)
        self.assertIn("conversion incomplete", status.note)

    def test_status_when_wan21_single_transformer_present(self):
        out = output_dir_for("Wan-AI/Wan2.1-T2V-1.3B")
        out.mkdir(parents=True)
        (out / "transformer-00001-of-00001.safetensors").write_bytes(b"fake")
        (out / "Wan2.1_VAE.safetensors").write_bytes(b"fake")
        (out / "models_t5_umt5-xxl-enc-bf16.safetensors").write_bytes(b"fake")
        status = status_for("Wan-AI/Wan2.1-T2V-1.3B")
        self.assertTrue(status.converted)
        self.assertTrue(status.hasTransformer)
        self.assertFalse(status.hasMoeExperts)
        self.assertTrue(status.hasVae)
        self.assertTrue(status.hasTextEncoder)

    def test_status_when_wan22_moe_experts_present(self):
        out = output_dir_for("Wan-AI/Wan2.2-T2V-A14B")
        out.mkdir(parents=True)
        (out / "high_noise_model").mkdir()
        (out / "low_noise_model").mkdir()
        (out / "vae.safetensors").write_bytes(b"fake")
        status = status_for("Wan-AI/Wan2.2-T2V-A14B")
        self.assertTrue(status.converted)
        self.assertTrue(status.hasMoeExperts)
        self.assertTrue(status.hasTransformer)  # MoE counts as transformer present
        self.assertTrue(status.hasVae)

    def test_status_returns_dict_via_to_dict(self):
        status = status_for("Wan-AI/Wan2.1-T2V-1.3B")
        d = status.to_dict()
        self.assertEqual(d["repo"], "Wan-AI/Wan2.1-T2V-1.3B")
        self.assertIn("converted", d)
        self.assertIn("outputDir", d)


class ListConvertedTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp(prefix="chaosengine-wan-list-test-")
        self._orig_root = wan_convert.CONVERT_ROOT
        wan_convert.CONVERT_ROOT = Path(self.tmpdir)

    def tearDown(self):
        wan_convert.CONVERT_ROOT = self._orig_root
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_empty_when_root_missing(self):
        wan_convert.CONVERT_ROOT = Path(self.tmpdir) / "nonexistent"
        self.assertEqual(list_converted(), [])

    def test_returns_only_converted_supported_repos(self):
        # Set up two slugs: one fully converted (Wan2.1), one partial.
        full = output_dir_for("Wan-AI/Wan2.1-T2V-1.3B")
        full.mkdir(parents=True)
        (full / "transformer.safetensors").write_bytes(b"x")
        (full / "Wan2.1_VAE.safetensors").write_bytes(b"x")

        partial = output_dir_for("Wan-AI/Wan2.2-TI2V-5B")
        partial.mkdir(parents=True)
        # Missing VAE → not converted

        # Also a stray dir that isn't a known repo slug.
        (Path(wan_convert.CONVERT_ROOT) / "Some-Other__Repo").mkdir()

        results = list_converted()
        repos = [s.repo for s in results]
        self.assertIn("Wan-AI/Wan2.1-T2V-1.3B", repos)
        self.assertNotIn("Wan-AI/Wan2.2-TI2V-5B", repos)
        # Stray dir filtered out (not in SUPPORTED_RAW_REPOS).
        self.assertEqual(len(results), 1)


class RunConvertTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp(prefix="chaosengine-wan-run-test-")
        self._orig_root = wan_convert.CONVERT_ROOT
        wan_convert.CONVERT_ROOT = Path(self.tmpdir)
        # Pretend a raw checkpoint exists.
        self.checkpoint = Path(self.tmpdir) / "raw-wan-21"
        self.checkpoint.mkdir()
        (self.checkpoint / "Wan2.1_VAE.pth").write_bytes(b"fake")

    def tearDown(self):
        wan_convert.CONVERT_ROOT = self._orig_root
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rejects_unsupported_repo(self):
        with self.assertRaises(ValueError) as ctx:
            run_convert(self.checkpoint, "Lightricks/LTX-Video")
        self.assertIn("Unsupported Wan repo", str(ctx.exception))

    def test_raises_when_mlx_video_missing(self):
        with patch(
            "backend_service.mlx_video_wan_convert.is_mlx_video_available",
            return_value=False,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                run_convert(self.checkpoint, "Wan-AI/Wan2.1-T2V-1.3B")
        self.assertIn("mlx-video is not installed", str(ctx.exception))

    def test_raises_when_checkpoint_dir_missing(self):
        with patch(
            "backend_service.mlx_video_wan_convert.is_mlx_video_available",
            return_value=True,
        ):
            with self.assertRaises(FileNotFoundError) as ctx:
                run_convert("/tmp/nope-does-not-exist", "Wan-AI/Wan2.1-T2V-1.3B")
        self.assertIn("Checkpoint dir not found", str(ctx.exception))

    def test_raises_when_subprocess_exits_nonzero(self):
        fake_proc = subprocess.CompletedProcess(
            args=["python"], returncode=1, stdout="", stderr="OOM during conversion",
        )
        with patch(
            "backend_service.mlx_video_wan_convert.is_mlx_video_available",
            return_value=True,
        ), patch(
            "backend_service.mlx_video_wan_convert.subprocess.run",
            return_value=fake_proc,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                run_convert(self.checkpoint, "Wan-AI/Wan2.1-T2V-1.3B")
        self.assertIn("exited with code 1", str(ctx.exception))
        self.assertIn("OOM during conversion", str(ctx.exception))

    def test_raises_when_subprocess_times_out(self):
        timeout_exc = subprocess.TimeoutExpired(cmd=["python"], timeout=10)
        timeout_exc.stderr = "stalled"
        with patch(
            "backend_service.mlx_video_wan_convert.is_mlx_video_available",
            return_value=True,
        ), patch(
            "backend_service.mlx_video_wan_convert.subprocess.run",
            side_effect=timeout_exc,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                run_convert(self.checkpoint, "Wan-AI/Wan2.1-T2V-1.3B", timeout_seconds=10)
        self.assertIn("timed out after 10s", str(ctx.exception))

    def test_happy_path_returns_post_convert_status(self):
        out = output_dir_for("Wan-AI/Wan2.1-T2V-1.3B")
        captured: dict[str, object] = {}

        def _fake_run(args, **kwargs):
            captured["args"] = args
            # Simulate the convert script writing output files.
            out.mkdir(parents=True, exist_ok=True)
            (out / "transformer.safetensors").write_bytes(b"x")
            (out / "Wan2.1_VAE.safetensors").write_bytes(b"x")
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="ok", stderr="",
            )

        with patch(
            "backend_service.mlx_video_wan_convert.is_mlx_video_available",
            return_value=True,
        ), patch(
            "backend_service.mlx_video_wan_convert.subprocess.run",
            side_effect=_fake_run,
        ):
            status = run_convert(self.checkpoint, "Wan-AI/Wan2.1-T2V-1.3B")

        self.assertTrue(status.converted)
        self.assertTrue(status.hasTransformer)
        self.assertTrue(status.hasVae)
        # Verify CLI args we forwarded to the convert module.
        self.assertEqual(captured["args"][1], "-m")
        self.assertEqual(captured["args"][2], "mlx_video.models.wan_2.convert")
        self.assertIn("--checkpoint-dir", captured["args"])
        self.assertIn("--output-dir", captured["args"])
        self.assertIn("--dtype", captured["args"])
        self.assertIn("bfloat16", captured["args"])

    def test_quantize_flags_threaded_through(self):
        out = output_dir_for("Wan-AI/Wan2.1-T2V-1.3B")
        captured: dict[str, object] = {}

        def _fake_run(args, **kwargs):
            captured["args"] = args
            out.mkdir(parents=True, exist_ok=True)
            (out / "transformer.safetensors").write_bytes(b"x")
            (out / "vae.safetensors").write_bytes(b"x")
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="", stderr="",
            )

        with patch(
            "backend_service.mlx_video_wan_convert.is_mlx_video_available",
            return_value=True,
        ), patch(
            "backend_service.mlx_video_wan_convert.subprocess.run",
            side_effect=_fake_run,
        ):
            run_convert(
                self.checkpoint, "Wan-AI/Wan2.1-T2V-1.3B",
                quantize=True, bits=4, group_size=64,
            )
        self.assertIn("--quantize", captured["args"])
        self.assertIn("--bits", captured["args"])
        self.assertIn("4", captured["args"])
        self.assertIn("--group-size", captured["args"])


class ConvertRootEnvOverrideTests(unittest.TestCase):
    def test_env_var_overrides_default_root(self):
        # Force a re-import so the module-level CONVERT_ROOT picks up the
        # env override at module-load time (per the implementation).
        import importlib
        import os as _os

        original = _os.environ.get("CHAOSENGINE_MLX_VIDEO_WAN_DIR")
        _os.environ["CHAOSENGINE_MLX_VIDEO_WAN_DIR"] = "/tmp/chaosengine-wan-override-test"
        try:
            from backend_service import mlx_video_wan_convert as mod
            importlib.reload(mod)
            self.assertEqual(
                str(mod.CONVERT_ROOT),
                "/tmp/chaosengine-wan-override-test",
            )
        finally:
            if original is None:
                _os.environ.pop("CHAOSENGINE_MLX_VIDEO_WAN_DIR", None)
            else:
                _os.environ["CHAOSENGINE_MLX_VIDEO_WAN_DIR"] = original
            from backend_service import mlx_video_wan_convert as mod_reset
            importlib.reload(mod_reset)


if __name__ == "__main__":
    unittest.main()
