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


class PartialBlobTests(unittest.TestCase):
    """Partial blobs from another file must not hide a complete snapshot."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "models--org--Model-GGUF"
        self.snap = self.repo / "snapshots" / "abc"
        self.blobs = self.repo / "blobs"
        self.snap.mkdir(parents=True)
        self.blobs.mkdir()
        (self.blobs / "deadbeef.1234.incomplete").write_bytes(b"x")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _link(self, name: str) -> None:
        blob = _write_gguf(self.blobs / f"{name}.blob")
        (self.snap / name).symlink_to(blob)

    def _broken(self) -> tuple[bool, str | None]:
        from backend_service.helpers.discovery import _detect_broken_library_item

        return _detect_broken_library_item(self.repo, "GGUF", "HF cache")

    def test_complete_quant_beside_a_cancelled_one_loads(self) -> None:
        self._link("Model-Q4_K_M.gguf")
        self.assertEqual(self._broken(), (False, None))

    def test_only_a_projector_still_counts_as_incomplete(self) -> None:
        self._link("mmproj-Model-f16.gguf")
        self.assertTrue(self._broken()[0])

    def test_split_gguf_needs_every_part(self) -> None:
        self._link("Model-Q4_K_M-00001-of-00002.gguf")
        self.assertTrue(self._broken()[0])
        self._link("Model-Q4_K_M-00002-of-00002.gguf")
        self.assertEqual(self._broken(), (False, None))


class HfCacheNameTests(unittest.TestCase):
    def test_org_ending_in_models_keeps_its_name(self) -> None:
        from backend_service.helpers.discovery import _detect_directory_model

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "models--meta-models--Muse-Glimmer-30B-GGUF"
            _write_gguf(repo / "snapshots" / "abc" / "Muse-Q4_K_M.gguf")
            detected = _detect_directory_model(repo)
        self.assertIsNotNone(detected)
        self.assertEqual(detected[0], "meta-models/Muse-Glimmer-30B-GGUF")


class MovedHfCacheTests(unittest.TestCase):
    """FU-610: the built-in Hugging Face cache entry follows HF_HUB_CACHE."""

    ENTRY = {"id": "hf-cache", "label": "Hugging Face cache", "path": "~/.cache/huggingface/hub", "enabled": True}

    def test_builtin_entry_scans_the_moved_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as home:
            _write_gguf(Path(tmp) / "models--org--Model-GGUF" / "snapshots" / "abc" / "Model-Q4_K_M.gguf")
            env = {"HF_HUB_CACHE": tmp, "HOME": home, "USERPROFILE": home}
            old = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                items = _discover_local_models([dict(self.ENTRY)])
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        self.assertEqual([i["name"] for i in items], ["org/Model-GGUF"])

    def test_entry_the_user_pointed_elsewhere_is_left_alone(self) -> None:
        from backend_service.helpers.hf_cache_paths import model_directory_root

        entry = {**self.ENTRY, "path": "~/Other"}
        self.assertEqual(model_directory_root(entry), Path(os.path.expanduser("~/Other")).expanduser())


class DrafterOnlyRepoTests(unittest.TestCase):
    """FU-611: a repo holding only a drafter is not a runnable text model."""

    def _types(self, *files: str) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            for name in files:
                _write_gguf(Path(tmp) / "models--org--Model-GGUF" / "snapshots" / "abc" / name)
            return [i["modelType"] for i in _discover(Path(tmp))]

    def test_drafter_only_repo_is_a_draft(self) -> None:
        self.assertEqual(self._types("mtp-gemma-4-12b-it.gguf"), ["draft"])

    def test_projector_only_repo_is_other(self) -> None:
        self.assertEqual(self._types("mmproj-F16.gguf"), ["other"])

    def test_repo_with_a_model_and_a_drafter_stays_text(self) -> None:
        self.assertEqual(self._types("Model-Q4_K_M.gguf", "mtp-model.gguf"), ["text"])
