"""Benchmark evaluation helpers: perplexity dataset and windows, next-token answer scoring."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service import mlx_worker_eval

from backend_service.mlx_worker_eval import (
    PERPLEXITY_DATASETS,
    choice_token_ids,
    perplexity_dataset_spec,
    perplexity_windows,
    pick_choice,
)
from backend_service.state.benchmarks import _local_kv_architecture
from backend_service.task_datasets import CHOICE_LETTERS


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


class _Tokenizer:
    """Maps a few strings to ids the way a real tokenizer would: " A" and "A" are different tokens."""

    VOCAB = {" A": 10, "A": 11, " B": 20, "B": 21, " C": 30, "C": 31, " D": 40, "D": 41}

    def encode(self, text):
        if text in self.VOCAB:
            return [self.VOCAB[text]]
        # A prompt: its last id says which question it is.
        return [1, 2, int(text.rsplit("#", 1)[-1])]


class ChoiceScoringTests(unittest.TestCase):
    def test_both_spellings_of_a_letter_count_for_it(self):
        ids = choice_token_ids(_Tokenizer(), CHOICE_LETTERS)
        self.assertEqual(ids, [[10, 11], [20, 21], [30, 31], [40, 41]])

    def test_a_tokenizer_with_one_token_per_letter_lists_it_once(self):
        class OneToken:
            def encode(self, text):
                return [ord(text.strip())]

        self.assertEqual(choice_token_ids(OneToken(), ("A", "B")), [[65], [66]])

    def test_the_highest_score_wins_and_a_tie_goes_to_the_first(self):
        self.assertEqual(pick_choice([0.1, 2.5, -1.0, 2.4], CHOICE_LETTERS), "B")
        self.assertEqual(pick_choice([1.0, 1.0, 1.0, 1.0], CHOICE_LETTERS), "A")

    def test_tasks_carry_no_generation_budget_any_more(self):
        # The scorer reads the next-token scores; nothing is generated.
        import inspect

        from backend_service import task_datasets

        self.assertNotIn("max_tokens", inspect.getsource(task_datasets))
        self.assertFalse(hasattr(task_datasets, "score_answer"))


@unittest.skipUnless(importlib.util.find_spec("mlx"), "MLX is not installed on this machine")
class TaskAccuracyEvalTests(unittest.TestCase):
    """``eval_task_accuracy`` against a model whose next-token scores are known."""

    def _model(self, favourite_by_question):
        import mlx.core as mx

        class Model:
            """Scores a sentence opener highest of all, then one answer letter."""

            def __init__(self):
                self.scored_positions: list[int] = []

            def make_cache(self):
                return []

            def __call__(self, tokens, cache=None):
                # Only the last token of a prompt may reach the output layer.
                self.scored_positions.append(int(tokens.shape[1]))
                question = int(tokens[0, -1].item())
                row = [0.0] * 64
                # "Let" (id 50) wins overall: a generating scorer would read that as no answer;
                # the next-token scorer only compares the four letters.
                row[50] = 9.0
                row[favourite_by_question.get(question, 0)] = 5.0
                return mx.array([[row]])

        return Model()

    def _run(self, tasks, favourites, request=None):
        progress: list[dict] = []
        with mock.patch("backend_service.task_datasets.load_task_data", lambda *_args: tasks), \
                mock.patch.object(mlx_worker_eval, "emit_progress",
                                  lambda phase, percent, message=None, **counts: progress.append({"phase": phase, **counts})):
            self.model = self._model(favourites)
            result = mlx_worker_eval.eval_task_accuracy(
                model=self.model, tokenizer=_Tokenizer(),
                request={"taskName": "mmlu", "limit": len(tasks), "numShots": 0, **(request or {})},
            )
        return result, progress

    def test_the_letter_with_the_highest_next_token_score_is_the_answer(self):
        tasks = [
            {"prompt": "Q #0", "correct_answer": "B"},   # model prefers " B" (id 20): right
            {"prompt": "Q #1", "correct_answer": "C"},   # model prefers "C" without the space (id 31): right
            {"prompt": "Q #2", "correct_answer": "A"},   # model prefers " D": wrong
        ]
        result, progress = self._run(tasks, {0: 20, 1: 31, 2: 40})
        self.assertEqual((result["correct"], result["total"], result["accuracy"]), (2, 3, 0.6667))
        self.assertEqual(result["scoringMethod"], "next_token")
        self.assertEqual(result["numShots"], 0)
        steps = [(p.get("step"), p.get("total"), p.get("correct")) for p in progress if p["phase"] == "evaluating"]
        self.assertEqual(steps, [(0, 3, 0), (1, 3, 1), (2, 3, 2), (3, 3, 2)])
        # Per question: the prompt without its last token (2 tokens) into the cache, then the last token alone.
        self.assertEqual(self.model.scored_positions, [2, 1] * 3)

    def test_a_cancel_file_stops_the_run_between_two_questions(self):
        marker = Path(tempfile.mkdtemp()) / "cancel"
        marker.touch()
        self.addCleanup(lambda: marker.unlink(missing_ok=True))
        with self.assertRaises(RuntimeError) as ctx:
            self._run([{"prompt": "Q #0", "correct_answer": "A"}], {0: 10}, {"cancelFile": str(marker)})
        self.assertEqual(str(ctx.exception), mlx_worker_eval.CANCELLED_MESSAGE)


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
