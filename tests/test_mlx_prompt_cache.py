"""Tests for the MLX per-session prompt-cache reuse logic (tier 4).

Exercises backend_service/mlx_worker_prompt_cache.py with a fake worker
state and patched mlx-lm cache primitives — no real model load. The
correctness contract under test: the persisted token list always equals
the cache's positional contents, and any uncertainty falls back to a fresh
full prefill.
"""

import unittest
from unittest import mock

from backend_service import mlx_worker_prompt_cache as pc

CACHE_MOD = "mlx_lm.models.cache"


class FakeCache:
    """Sentinel standing in for an mlx-lm prompt cache."""

    def __init__(self, label):
        self.label = label


class FakeState:
    def __init__(self, *, base_cache=None, base_note=None, tokens=None, model_ref="m"):
        self._base = (base_cache, base_note)
        self._tokens = list(tokens or [])
        self.model = object()
        self._loaded_model_ref = model_ref
        self.tokenizer = self
        self._persist_cache = None
        self._persist_tokens = []
        self._persist_cache_model_ref = None

    def _make_cache(self):
        return self._base

    def encode(self, _text):  # stands in for tokenizer.encode
        return list(self._tokens)


class CommonPrefixTests(unittest.TestCase):
    def test_common_prefix_len(self):
        self.assertEqual(pc._common_prefix_len([1, 2, 3], [1, 2, 9]), 2)
        self.assertEqual(pc._common_prefix_len([1, 2], [9]), 0)
        self.assertEqual(pc._common_prefix_len([1, 2, 3], [1, 2, 3, 4]), 3)


class AcquireCompressionTests(unittest.TestCase):
    def test_compression_strategy_passthrough(self):
        comp = FakeCache("compression")
        state = FakeState(base_cache=comp, base_note="cn")
        acq = pc.acquire(state, "p-text")
        self.assertIs(acq.cache, comp)
        self.assertEqual(acq.prompt_feed, "p-text")  # string, unchanged
        self.assertFalse(acq.managed)
        self.assertIs(acq.fields_cache, comp)
        self.assertIsNone(acq.commit_tokens)


class AcquireNativeTests(unittest.TestCase):
    def _patches(self, *, can_trim=True, trim=lambda c, n: n, fresh_label="fresh"):
        return (
            mock.patch(f"{CACHE_MOD}.make_prompt_cache", return_value=FakeCache(fresh_label)),
            mock.patch(f"{CACHE_MOD}.can_trim_prompt_cache", return_value=can_trim),
            mock.patch(f"{CACHE_MOD}.trim_prompt_cache", side_effect=trim),
        )

    def test_fresh_native_cache_full_prefill(self):
        state = FakeState(base_cache=None, tokens=[1, 2, 3])
        with self._patches()[0], self._patches()[1], self._patches()[2]:
            acq = pc.acquire(state, "ignored")
        self.assertTrue(acq.managed)
        self.assertIsInstance(acq.cache, FakeCache)
        self.assertEqual(acq.prompt_feed, [1, 2, 3])  # full token list
        self.assertEqual(acq.commit_tokens, [1, 2, 3])
        self.assertIsNone(acq.fields_cache)

    def test_reuse_hit_feeds_only_suffix_no_trim(self):
        persist = FakeCache("persist")
        state = FakeState(base_cache=None, tokens=[1, 2, 3, 4, 5], model_ref="m")
        state._persist_cache = persist
        state._persist_tokens = [1, 2, 3]
        state._persist_cache_model_ref = "m"
        m1, m2, m3 = self._patches()
        with m1, m2, m3 as trim:
            acq = pc.acquire(state, "ignored")
        self.assertIs(acq.cache, persist)            # reused, not fresh
        self.assertEqual(acq.prompt_feed, [4, 5])    # suffix only
        self.assertEqual(acq.commit_tokens, [1, 2, 3, 4, 5])
        trim.assert_not_called()                     # num_to_trim == 0

    def test_reuse_with_divergence_trims_tail(self):
        persist = FakeCache("persist")
        state = FakeState(base_cache=None, tokens=[1, 2, 3, 4], model_ref="m")
        state._persist_cache = persist
        state._persist_tokens = [1, 2, 3, 9, 9]   # diverges after index 3
        state._persist_cache_model_ref = "m"
        m1, m2, m3 = self._patches()
        with m1, m2, m3 as trim:
            acq = pc.acquire(state, "ignored")
        self.assertIs(acq.cache, persist)
        trim.assert_called_once_with(persist, 2)  # 5 cached - 3 common
        self.assertEqual(acq.prompt_feed, [4])    # full[3:]

    def test_reset_on_model_change(self):
        state = FakeState(base_cache=None, tokens=[1, 2, 3], model_ref="new")
        state._persist_cache = FakeCache("stale")
        state._persist_tokens = [1, 2, 3]
        state._persist_cache_model_ref = "old"
        m1, m2, m3 = self._patches()
        with m1, m2, m3:
            acq = pc.acquire(state, "ignored")
        self.assertEqual(acq.prompt_feed, [1, 2, 3])  # fresh → full prefill
        self.assertEqual(acq.cache.label, "fresh")

    def test_reset_when_cache_not_trimmable(self):
        state = FakeState(base_cache=None, tokens=[1, 2, 3, 4], model_ref="m")
        state._persist_cache = FakeCache("persist")
        state._persist_tokens = [1, 2, 3]
        state._persist_cache_model_ref = "m"
        m1, m2, m3 = self._patches(can_trim=False)
        with m1, m2, m3:
            acq = pc.acquire(state, "ignored")
        self.assertEqual(acq.cache.label, "fresh")
        self.assertEqual(acq.prompt_feed, [1, 2, 3, 4])

    def test_reset_when_no_common_prefix(self):
        state = FakeState(base_cache=None, tokens=[7, 8, 9], model_ref="m")
        state._persist_cache = FakeCache("persist")
        state._persist_tokens = [1, 2, 3]
        state._persist_cache_model_ref = "m"
        m1, m2, m3 = self._patches()
        with m1, m2, m3:
            acq = pc.acquire(state, "ignored")
        self.assertEqual(acq.cache.label, "fresh")
        self.assertEqual(acq.prompt_feed, [7, 8, 9])

    def test_partial_trim_falls_back_to_fresh(self):
        state = FakeState(base_cache=None, tokens=[1, 2, 3, 4], model_ref="m")
        state._persist_cache = FakeCache("persist")
        state._persist_tokens = [1, 2, 3, 9, 9]
        state._persist_cache_model_ref = "m"
        # trim returns fewer than requested → unsafe → fresh
        m1, m2, m3 = self._patches(trim=lambda c, n: n - 1)
        with m1, m2, m3:
            acq = pc.acquire(state, "ignored")
        self.assertEqual(acq.cache.label, "fresh")
        self.assertEqual(acq.prompt_feed, [1, 2, 3, 4])


class CommitInvalidateTests(unittest.TestCase):
    def test_commit_accounting_is_prompt_plus_generated(self):
        state = FakeState()
        cache = FakeCache("c")
        pc.commit(state, cache=cache, commit_tokens=[1, 2, 3], generated_ids=[4, 5], model_ref="m")
        self.assertIs(state._persist_cache, cache)
        self.assertEqual(state._persist_tokens, [1, 2, 3, 4, 5])
        self.assertEqual(state._persist_cache_model_ref, "m")

    def test_commit_noop_when_not_managed(self):
        state = FakeState()
        pc.commit(state, cache=None, commit_tokens=None, generated_ids=[4], model_ref="m")
        self.assertIsNone(state._persist_cache)
        self.assertEqual(state._persist_tokens, [])

    def test_invalidate_clears(self):
        state = FakeState()
        state._persist_cache = FakeCache("c")
        state._persist_tokens = [1, 2]
        state._persist_cache_model_ref = "m"
        pc.invalidate(state)
        self.assertIsNone(state._persist_cache)
        self.assertEqual(state._persist_tokens, [])
        self.assertIsNone(state._persist_cache_model_ref)


if __name__ == "__main__":
    unittest.main()


try:  # Real-MLX checks run wherever mlx + mlx-lm import (CPU builds too).
    import mlx.core as _mx  # noqa: F401
    from mlx_lm.models import qwen3_next as _qwen3_next
    _HAVE_MLX = True
except Exception:  # noqa: BLE001
    _HAVE_MLX = False


class _RealState:
    """Worker-state stand-in around a real (tiny, random) mlx-lm model."""

    def __init__(self, model, tokens_by_text):
        self.model = model
        self.tokenizer = self
        self._tokens_by_text = tokens_by_text
        self._loaded_model_ref = "tiny-hybrid"
        self._persist_cache = None
        self._persist_tokens = []
        self._persist_cache_model_ref = None

    def _make_cache(self):
        return None, None

    def encode(self, text):
        return list(self._tokens_by_text[text])


@unittest.skipUnless(_HAVE_MLX, "mlx / mlx-lm not installed")
class HybridCheckpointTests(unittest.TestCase):
    """FU-090: hybrid (GDN + attention) caches reuse history via a
    recurrent-state checkpoint, with output identical to a full prefill."""

    def _model(self):
        import mlx.core as mx

        mx.random.seed(0)
        args = _qwen3_next.ModelArgs(
            model_type="qwen3_next", hidden_size=64, num_hidden_layers=4,
            intermediate_size=128, num_attention_heads=4, linear_num_value_heads=4,
            linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
            linear_conv_kernel_dim=4, num_experts=4, num_experts_per_tok=2,
            decoder_sparse_step=1, shared_expert_intermediate_size=64,
            mlp_only_layers=[], moe_intermediate_size=32, rms_norm_eps=1e-6,
            vocab_size=97, num_key_value_heads=2, rope_theta=10000.0,
            partial_rotary_factor=0.25, max_position_embeddings=512, head_dim=16,
            full_attention_interval=2,
        )
        model = _qwen3_next.Model(args)
        mx.eval(model.parameters())
        return model

    def _last_logits(self, model, cache, feed):
        import mlx.core as mx

        out = model(mx.array(feed)[None], cache=cache)
        mx.eval(out)
        return out[0, -1]

    def test_second_turn_restores_checkpoint_and_matches_full_prefill(self):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        model = self._model()
        turn1 = [5, 9, 13, 2, 40, 41, 7, 8, 3]
        generated = [60, 61, 62]
        turn2 = turn1 + [60, 61, 70, 71, 72, 11]  # re-rendered history diverges after 2 gen tokens
        state = _RealState(model, {"t1": turn1, "t2": turn2})

        acq1 = pc.acquire(state, "t1")
        self.assertTrue(acq1.managed)
        self.assertEqual(acq1.prompt_feed, turn1[-1:])  # prefilled all but the last
        # Simulate generation: last prompt token + generated tokens enter the cache.
        self._last_logits(model, acq1.cache, acq1.prompt_feed + generated)
        pc.commit(state, cache=acq1.cache, commit_tokens=acq1.commit_tokens,
                  generated_ids=generated, model_ref="tiny-hybrid")
        self.assertEqual(state._persist_snapshot[0], len(turn1) - 1)

        acq2 = pc.acquire(state, "t2")
        self.assertIs(acq2.cache, acq1.cache)  # reused, not rebuilt
        self.assertEqual(acq2.prompt_feed, turn2[-1:])
        reused = self._last_logits(model, acq2.cache, acq2.prompt_feed)

        reference = self._last_logits(model, make_prompt_cache(model), turn2)
        self.assertTrue(mx.allclose(reused, reference, atol=1e-4).item())

    def test_no_checkpoint_reuse_when_prefix_short_of_boundary(self):
        model = self._model()
        turn1 = [5, 9, 13, 2, 40, 41, 7, 8, 3]
        turn2 = [5, 9, 99, 98, 97]  # diverges before the checkpoint
        state = _RealState(model, {"t1": turn1, "t2": turn2})
        acq1 = pc.acquire(state, "t1")
        self._last_logits(model, acq1.cache, acq1.prompt_feed)
        pc.commit(state, cache=acq1.cache, commit_tokens=acq1.commit_tokens,
                  generated_ids=[], model_ref="tiny-hybrid")
        acq2 = pc.acquire(state, "t2")
        self.assertIsNot(acq2.cache, acq1.cache)  # fresh cache, full prefill
        self.assertEqual(acq2.prompt_feed, turn2[-1:])
