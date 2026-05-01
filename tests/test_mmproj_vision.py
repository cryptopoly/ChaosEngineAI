"""Tests for the mmproj sibling resolver and visionEnabled flag flip.

The hotfix that closed the silent-image-drop bug stays in force when
no mmproj is present (vision flag stays False). When llama-server
starts with `--mmproj`, the runtime sets `visionEnabled=True` and the
capability resolver promotes `supportsVision` so the composer's image
attach button shows up again.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend_service.inference import _resolve_mmproj_path
from backend_service.catalog.capabilities import resolve_capabilities


class ResolveMmprojPathTests(unittest.TestCase):
    def test_returns_none_when_path_is_none(self):
        self.assertIsNone(_resolve_mmproj_path(None))

    def test_returns_none_when_path_does_not_exist(self):
        self.assertIsNone(_resolve_mmproj_path("/nonexistent/model.gguf"))

    def test_returns_none_when_no_mmproj_sibling(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            main = tmp_path / "model.gguf"
            main.write_bytes(b"\x00")
            self.assertIsNone(_resolve_mmproj_path(str(main)))

    def test_finds_mmproj_in_same_directory(self):
        # The standard HF cache layout puts the projector next to the
        # main weights — most common case.
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            main = tmp_path / "model.gguf"
            main.write_bytes(b"\x00")
            mmproj = tmp_path / "mmproj.gguf"
            mmproj.write_bytes(b"\x00")
            self.assertEqual(_resolve_mmproj_path(str(main)), str(mmproj))

    def test_finds_mmproj_with_descriptive_filename(self):
        # Some repos publish projectors with descriptive prefixes
        # (e.g. `gemma-3-27b-mmproj-Q4_K_M.gguf`). Substring match
        # picks them up regardless of the exact name.
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            main = tmp_path / "gemma-3-27b-it-qat-4bit.gguf"
            main.write_bytes(b"\x00")
            mmproj = tmp_path / "gemma-3-27b-mmproj-Q4_K_M.gguf"
            mmproj.write_bytes(b"\x00")
            self.assertEqual(_resolve_mmproj_path(str(main)), str(mmproj))

    def test_picks_largest_when_multiple_projectors_present(self):
        # Some downloads contain both a quantised and a full-precision
        # projector. The full-precision one (larger file) is the
        # better quality choice.
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            main = tmp_path / "model.gguf"
            main.write_bytes(b"\x00" * 100)
            small = tmp_path / "mmproj-Q4.gguf"
            small.write_bytes(b"\x00" * 10)
            big = tmp_path / "mmproj-f16.gguf"
            big.write_bytes(b"\x00" * 50)
            self.assertEqual(_resolve_mmproj_path(str(main)), str(big))

    def test_finds_mmproj_in_sibling_directory(self):
        # Some HF caches keep projectors one level up (in the snapshot
        # root rather than the file's immediate folder). The walker
        # checks the parent's parent too.
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            weights_dir = tmp_path / "weights"
            weights_dir.mkdir()
            main = weights_dir / "model.gguf"
            main.write_bytes(b"\x00")
            sibling = tmp_path / "projectors"
            sibling.mkdir()
            mmproj = sibling / "mmproj.gguf"
            mmproj.write_bytes(b"\x00")
            self.assertEqual(_resolve_mmproj_path(str(main)), str(mmproj))


class VisionCapabilityFlipTests(unittest.TestCase):
    def test_supports_vision_false_when_runtime_disabled(self):
        caps = resolve_capabilities(
            "google/gemma-3-27b-it-qat-4bit",
            None,
            engine="llama.cpp",
            vision_enabled=False,
        )
        self.assertFalse(caps.supportsVision)

    def test_supports_vision_true_when_runtime_loads_mmproj(self):
        # Once the loader confirms `--mmproj` was passed,
        # `LoadedModelInfo.visionEnabled` becomes True and the
        # capability resolver promotes the typed flag — composer
        # image-attach button comes back. Use a catalog entry whose
        # capabilities list includes "vision" so the typed flag has
        # something to promote.
        caps = resolve_capabilities(
            "google/gemma-4-E4B-it",
            None,
            engine="llama.cpp",
            vision_enabled=True,
        )
        self.assertTrue(caps.supportsVision)

    def test_mlx_engine_still_demotes_even_when_mmproj_loaded(self):
        # Belt-and-braces: any future MLX-equivalent that claims
        # `vision_enabled=True` should still demote because the MLX
        # worker has no image-carrying code path. Re-enable when
        # mlx-vlm wiring lands.
        caps = resolve_capabilities(
            "mlx-community/llava-v1.6-mistral-7b",
            None,
            engine="mlx",
            vision_enabled=True,
        )
        self.assertFalse(caps.supportsVision)


if __name__ == "__main__":
    unittest.main()
