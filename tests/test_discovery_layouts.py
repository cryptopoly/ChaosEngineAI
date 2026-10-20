"""Library discovery over the folder layouts users actually keep models in."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from backend_service.helpers.discovery import _discover_local_models, _directory_model_name


def _write_gguf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # GGUF magic, version 3, no tensors, no metadata: enough for the walker.
    path.write_bytes(b"GGUF" + (3).to_bytes(4, "little") + bytes(16) + bytes(4096))
    return path


def _discover(root: Path) -> list[dict]:
    return _discover_local_models([{"id": "t", "label": "t", "path": str(root), "enabled": True}])


class DirectoryModelNameTests(unittest.TestCase):
    def test_modelscope_underscores_become_dots(self) -> None:
        self.assertEqual(
            _directory_model_name("Qwen3___5-9B-Claude-4___6-Opus-Reasoning-Distilled-GGUF"),
            "Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
        )

    def test_other_names_are_left_alone(self) -> None:
        for name in ("Qwen3.5-9B-GGUF", "my_model_v2", "a__b", "___lead", "trail___"):
            self.assertEqual(_directory_model_name(name), name)


class DiscoveryLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_modelscope_cache_layout_is_listed_under_its_real_name(self) -> None:
        repo = self.root / "hub" / "models" / "Jackrong" / "Qwen3___5-9B-Distilled-GGUF"
        _write_gguf(repo / "Qwen3.5-9B-Distilled.Q4_K_M.gguf")

        items = _discover(self.root)

        self.assertEqual([item["name"] for item in items], ["Qwen3.5-9B-Distilled-GGUF"])
        self.assertEqual(items[0]["format"], "GGUF")
        self.assertEqual(items[0]["backend"], "llama.cpp")

    def test_a_loose_gguf_in_the_directory_is_listed(self) -> None:
        _write_gguf(self.root / "Some-Model.Q8_0.gguf")

        items = _discover(self.root)

        self.assertEqual([(item["name"], item["format"], item["sourceKind"]) for item in items],
                         [("Some-Model.Q8_0", "GGUF", "File")])

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_a_symlinked_model_folder_is_followed(self) -> None:
        library = self.root / "library"
        library.mkdir()
        real = self.root / "other-drive" / "My-Model-GGUF"
        _write_gguf(real / "My-Model.Q8_0.gguf")
        (library / "My-Model-GGUF").symlink_to(real, target_is_directory=True)

        items = _discover(library)

        self.assertEqual([item["name"] for item in items], ["My-Model-GGUF"])
        self.assertEqual(Path(items[0]["path"]), real.resolve())

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_a_link_back_up_the_tree_does_not_loop(self) -> None:
        _write_gguf(self.root / "a" / "Loop-GGUF" / "Loop.Q4_K_M.gguf")
        (self.root / "a" / "up").symlink_to(self.root, target_is_directory=True)

        items = _discover(self.root)

        self.assertEqual([item["name"] for item in items], ["Loop-GGUF"])

    @unittest.skipIf(os.name == "nt", "symlinks need privileges on Windows")
    def test_the_same_folder_through_two_paths_is_listed_once(self) -> None:
        real = self.root / "store" / "Twice-GGUF"
        _write_gguf(real / "Twice.Q4_K_M.gguf")
        (self.root / "alias").symlink_to(self.root / "store", target_is_directory=True)

        items = _discover(self.root)

        self.assertEqual([item["name"] for item in items], ["Twice-GGUF"])


if __name__ == "__main__":
    unittest.main()
