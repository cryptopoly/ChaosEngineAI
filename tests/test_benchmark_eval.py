"""Benchmark evaluation helpers: perplexity dataset and windows, answer scoring."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend_service.mlx_worker_eval import (
    PERPLEXITY_DATASETS,
    perplexity_dataset_spec,
    perplexity_windows,
)
from backend_service.state.benchmarks import _local_kv_architecture
from backend_service.task_datasets import score_answer


class PerplexityDatasetTests(unittest.TestCase):
    def test_wikitext_2_points_at_a_real_hub_repo_with_its_config(self):
        # "wikitext/wikitext-2-raw-v1" is not a Hub repo: every run failed with
        # "doesn't exist on the Hub or cannot be accessed".
        repo, config, split = perplexity_dataset_spec("wikitext-2")
        self.assertEqual(repo, "Salesforce/wikitext")
        self.assertEqual(config, "wikitext-2-raw-v1")
        self.assertEqual(split, "test")
        for name, (hub_repo, _config, _split) in PERPLEXITY_DATASETS.items():
            self.assertEqual(hub_repo.count("/"), 1, name)

    def test_unknown_dataset_is_refused_with_the_supported_list(self):
        with self.assertRaises(RuntimeError) as ctx:
            perplexity_dataset_spec("made-up")
        self.assertIn("wikitext-2", str(ctx.exception))

    def test_windows_have_the_asked_shape_and_are_repeatable(self):
        tokens = list(range(1000))
        first = perplexity_windows(tokens, 4, 128)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(len(window) == 128 for window in first))
        self.assertEqual(first, perplexity_windows(tokens, 4, 128))
        for window in first:
            self.assertEqual(window[0] % 128, 0)
            self.assertEqual(window, list(range(window[0], window[0] + 128)))

    def test_fewer_windows_than_asked_returns_what_exists(self):
        self.assertEqual(len(perplexity_windows(list(range(300)), 64, 128)), 2)

    def test_text_shorter_than_one_window_is_an_error(self):
        with self.assertRaises(RuntimeError):
            perplexity_windows(list(range(50)), 8, 128)


class ScoreAnswerTests(unittest.TestCase):
    def test_standalone_letters_are_scored(self):
        self.assertTrue(score_answer("mmlu", " B", "B"))
        self.assertTrue(score_answer("mmlu", "\nA.", "A"))
        self.assertTrue(score_answer("mmlu", "(C)", "C"))
        self.assertTrue(score_answer("mmlu", "The answer is D", "D"))
        self.assertTrue(score_answer("mmlu", "b.", "B"))
        self.assertFalse(score_answer("mmlu", " B", "C"))

    def test_a_word_that_starts_with_a_choice_letter_is_not_an_answer(self):
        # Real output from a 0.6B model on HellaSwag, scored as correct for B before.
        self.assertFalse(score_answer("hellaswag", "\n\nBased on", "B"))
        self.assertFalse(score_answer("mmlu", "Choices:\n", "C"))
        self.assertFalse(score_answer("mmlu", "Definitely", "D"))

    def test_no_letter_is_wrong(self):
        self.assertFalse(score_answer("mmlu", "Let's analyze", "A"))
        self.assertFalse(score_answer("mmlu", "", "A"))


class BenchmarkCacheSizingTests(unittest.TestCase):
    """The reported cache size comes from the model's own layout, not from a 7B guess."""

    CONFIG = {
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 1024,
        "head_dim": 128,
    }

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _preview(self, arch):
        from backend_service.app import compute_cache_preview

        return compute_cache_preview(
            bits=4, fp16_layers=0, context_tokens=8192, params_b=7.0,
            system_stats={"totalMemoryGb": 64, "availableMemoryGb": 32}, arch=arch,
        )

    def test_plain_folder_and_cache_folder_are_both_read(self):
        plain = self.root / "plain"
        plain.mkdir()
        (plain / "config.json").write_text(json.dumps(self.CONFIG))
        snapshot = self.root / "models--org--small" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(json.dumps(self.CONFIG))
        for path in (plain, self.root / "models--org--small"):
            arch = _local_kv_architecture(str(path))
            self.assertIsNotNone(arch, path)
            self.assertLess(self._preview(arch)["baselineCacheGb"], self._preview(None)["baselineCacheGb"])

    def test_missing_path_falls_back_to_the_guess(self):
        self.assertIsNone(_local_kv_architecture(None))
        self.assertIsNone(_local_kv_architecture(str(self.root / "gone")))


if __name__ == "__main__":
    unittest.main()
