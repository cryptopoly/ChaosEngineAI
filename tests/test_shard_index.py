"""Downloads leave out weight shards no index file uses (``helpers.shard_index``)."""
from __future__ import annotations

import fnmatch
import unittest

from backend_service.helpers.shard_index import pruned_allow_patterns, unused_shard_files

GB = 10**9

# Trimmed from prince-canuma/LTX-2.3-distilled: two shard sets, one indexed.
LTX_FILES = {
    "transformer/config.json": 1_000,
    "transformer/model.safetensors.index.json": 50_000,
    **{f"transformer/model-{i:05d}-of-00003.safetensors": 4 * GB for i in range(1, 4)},
    **{f"transformer/model-{i:05d}-of-00002.safetensors": 5 * GB for i in range(1, 3)},
    "vae/model.safetensors": GB,
    "README.md": 100,
}
LTX_INDEX = {"weight_map": {f"w{i}": f"model-{i:05d}-of-00003.safetensors" for i in range(1, 4)}}


class UnusedShardTests(unittest.TestCase):
    def test_stale_shard_set_is_unused(self):
        unused = unused_shard_files(LTX_FILES, {"transformer/model.safetensors.index.json": LTX_INDEX})
        self.assertEqual(unused, {f"transformer/model-{i:05d}-of-00002.safetensors" for i in (1, 2)})

    def test_non_shard_files_are_never_dropped(self):
        files = {"unet/diffusion_pytorch_model.fp16.safetensors": 1, **dict.fromkeys(LTX_FILES, 1)}
        index = {"unet/diffusion_pytorch_model.safetensors.index.json": {"weight_map": {"a": "x-00001-of-00001.safetensors"}}}
        self.assertNotIn("unet/diffusion_pytorch_model.fp16.safetensors", unused_shard_files(files, index))

    def test_folders_without_an_index_are_left_alone(self):
        self.assertEqual(unused_shard_files(["other/model-00001-of-00002.safetensors"], {}), set())


class PrunedPatternsTests(unittest.TestCase):
    def _prune(self, patterns, fetch=lambda repo, path: LTX_INDEX):
        return pruned_allow_patterns("x/y", patterns, list_files=lambda repo: LTX_FILES, fetch_json=fetch)

    def test_skips_the_unused_set_and_keeps_everything_else(self):
        patterns, kept, skipped = self._prune(["transformer/**", "vae/**", "*.md"])
        fetched = {p for p in LTX_FILES if any(fnmatch.fnmatch(p, pat) for pat in patterns)}
        self.assertEqual(skipped, 10 * GB)
        self.assertNotIn("transformer/model-00001-of-00002.safetensors", fetched)
        self.assertIn("transformer/model-00001-of-00003.safetensors", fetched)
        self.assertIn("transformer/model.safetensors.index.json", fetched)
        self.assertIn("vae/model.safetensors", fetched)
        self.assertEqual(kept, sum(LTX_FILES[p] for p in fetched))

    def test_respects_the_original_allow_list(self):
        patterns, _, _ = self._prune(["transformer/**"])
        fetched = {p for p in LTX_FILES if any(fnmatch.fnmatch(p, pat) for pat in patterns)}
        self.assertNotIn("vae/model.safetensors", fetched)

    def test_unreadable_index_keeps_the_original_patterns(self):
        self.assertIsNone(self._prune(None, fetch=lambda repo, path: None))

    def test_nothing_to_skip_returns_none(self):
        clean = {k: v for k, v in LTX_FILES.items() if "-of-00002" not in k}
        self.assertIsNone(pruned_allow_patterns(
            "x/y", None, list_files=lambda repo: clean, fetch_json=lambda repo, path: LTX_INDEX,
        ))

    def test_special_characters_match_literally(self):
        from backend_service.helpers.shard_index import _literal_pattern

        self.assertTrue(fnmatch.fnmatch("a[1]*.json", _literal_pattern("a[1]*.json")))
        self.assertFalse(fnmatch.fnmatch("a1x.json", _literal_pattern("a[1]*.json")))


if __name__ == "__main__":
    unittest.main()
