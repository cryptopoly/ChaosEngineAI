"""DDTree tree verification on real MLX with a tiny random Qwen3.

The verify pass runs a whole draft tree through the target at once. It is
exact only if every node is rotated at ``prefix + depth`` and sees nothing
but the committed prefix and its own ancestors; then its logits equal an
unbatched forward over its root-to-node path. These tests compare against
that reference, check the cache compaction that keeps the accepted path,
and run the generation loop end to end against plain greedy decoding. No
weights are needed; they skip without mlx-lm. The last class covers the
app-side bookkeeping when the worker drops a tree budget.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from unittest import mock

import numpy as np

_HAS_MLX = importlib.util.find_spec("mlx_lm") is not None

VOCAB = 48
PREFIX = [3, 17, 5, 29, 11, 40, 2]
# A tree whose flat order differs from depth order and whose rows have
# earlier non-ancestors: root 0; depth 1: 1, 2; depth 2: 3 (of 1), 4 (of 2),
# 6 (of 1); depth 3: 5 (of 3), 8 (of 4); depth 4: 7 (of 5).
PARENTS = [-1, 0, 0, 1, 2, 3, 1, 5, 4]
TREE_TOKENS = [9, 14, 21, 7, 33, 12, 26, 4, 18]


def _depths(parents):
    depths = [0] * len(parents)
    for index in range(1, len(parents)):
        depths[index] = depths[parents[index]] + 1
    return depths


def _visibility(parents):
    size = len(parents)
    vis = np.zeros((size, size), dtype=np.bool_)
    vis[0, 0] = True
    for index in range(1, size):
        vis[index, :index] = vis[parents[index], :index]
        vis[index, index] = True
    return vis


def _path(parents, node):
    path = []
    while node != -1:
        path.append(node)
        node = parents[node]
    return path[::-1]


@unittest.skipUnless(_HAS_MLX, "mlx-lm is not installed")
class TreeVerifyTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        from mlx_lm.models import qwen3

        mx.random.seed(7)
        args = qwen3.ModelArgs(
            model_type="qwen3", hidden_size=64, num_hidden_layers=3, intermediate_size=128,
            num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=VOCAB, num_key_value_heads=2,
            max_position_embeddings=512, head_dim=16, tie_word_embeddings=True, rope_theta=10000.0,
        )
        self.model = qwen3.Model(args)
        mx.eval(self.model.parameters())
        self.text = self.model.model
        self.capture = {0, 1, 2, 3}

    def _logits(self, hidden):
        return self.text.embed_tokens.as_linear(hidden)

    def _reference(self, tokens):
        """Plain causal forward over ``tokens``: last-position logits and
        per-layer hidden states."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)
        hidden = self.text.embed_tokens(mx.array(tokens)[None])
        captured = {0: hidden[:, -1]}
        for index, (layer, layer_cache) in enumerate(zip(self.text.layers, cache)):
            hidden = layer(hidden, "causal", layer_cache)
            captured[index + 1] = hidden[:, -1]
        logits = self._logits(self.text.norm(hidden))[0, -1]
        mx.eval(logits, *captured.values())
        return logits, captured

    def _prefilled_cache(self):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)
        mx.eval(self.model(mx.array(PREFIX)[None], cache=cache))
        return cache

    def _verify(self, cache):
        import mlx.core as mx
        from backend_service.ddtree_attention import tree_attention_mask, tree_forward, tree_positions

        logits, captured = tree_forward(
            text_model=self.text,
            logits_fn=self._logits,
            cache=cache,
            input_ids=mx.array(TREE_TOKENS)[None],
            positions=tree_positions(len(PREFIX), _depths(PARENTS)[1:]),
            mask=tree_attention_mask(mx, _visibility(PARENTS), len(PREFIX)),
            capture_layer_ids=self.capture,
        )
        mx.eval(logits, *captured.values())
        return logits, captured

    def test_tree_is_not_trivially_flat(self):
        # Guards the other tests: with depth == flat index or with every row
        # seeing every earlier row, a broken position or mask would pass.
        depths = _depths(PARENTS)
        self.assertTrue(any(depth != index for index, depth in enumerate(depths)))
        vis = _visibility(PARENTS)
        self.assertTrue(any(not vis[row, col] for row in range(len(PARENTS)) for col in range(row)))

    def test_every_node_matches_an_unbatched_forward_over_its_path(self):
        import mlx.core as mx

        logits, captured = self._verify(self._prefilled_cache())
        for node in range(len(PARENTS)):
            path_tokens = [TREE_TOKENS[i] for i in _path(PARENTS, node)]
            ref_logits, ref_hidden = self._reference(PREFIX + path_tokens)
            self.assertTrue(
                mx.allclose(logits[0, node], ref_logits, atol=1e-4).item(),
                f"node {node} (depth {_depths(PARENTS)[node]}) logits differ",
            )
            for layer_id in self.capture:
                self.assertTrue(
                    mx.allclose(captured[layer_id][0, node], ref_hidden[layer_id][0], atol=1e-4).item(),
                    f"node {node} layer {layer_id} hidden differs",
                )

    def test_cache_compaction_keeps_the_accepted_path(self):
        import mlx.core as mx
        from backend_service.ddtree_attention import compact_kv_cache

        for accepted in ([0, 2, 4, 8], [0, 1, 3, 5, 7], [0, 1, 6], [0]):
            cache = self._prefilled_cache()
            self._verify(cache)
            compact_kv_cache(cache, len(PREFIX), accepted)
            self.assertEqual(cache[0].offset, len(PREFIX) + len(accepted))
            next_token = 23
            out = self.model(mx.array([[next_token]]), cache=cache)[0, -1]
            path_tokens = [TREE_TOKENS[i] for i in accepted]
            ref_logits, _ = self._reference(PREFIX + path_tokens + [next_token])
            self.assertTrue(mx.allclose(out, ref_logits, atol=1e-4).item(), f"path {accepted}")

    def test_chunked_prefill_matches_a_single_forward(self):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from backend_service.ddtree_attention import prefill_with_capture

        cache = make_prompt_cache(self.model)
        logits, captured = prefill_with_capture(
            text_model=self.text, logits_fn=self._logits, cache=cache,
            input_ids=mx.array(PREFIX)[None], capture_layer_ids=self.capture, step=3,
        )
        ref_logits, ref_hidden = self._reference(PREFIX)
        self.assertTrue(mx.allclose(logits[0], ref_logits, atol=1e-4).item())
        self.assertEqual(captured[2].shape[1], len(PREFIX))
        self.assertTrue(mx.allclose(captured[2][0, -1], ref_hidden[2][0], atol=1e-4).item())
        self.assertEqual(cache[0].offset, len(PREFIX))

    def test_rejects_recurrent_layers_and_other_caches(self):
        from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache, make_prompt_cache
        from backend_service.ddtree_attention import tree_verify_unsupported_reason

        self.assertIsNone(tree_verify_unsupported_reason(self.text, make_prompt_cache(self.model)))
        rotating = [RotatingKVCache(max_size=64) for _ in self.text.layers]
        self.assertIn("RotatingKVCache", tree_verify_unsupported_reason(self.text, rotating))
        quantized = [QuantizedKVCache() for _ in self.text.layers]
        self.assertIn("QuantizedKVCache", tree_verify_unsupported_reason(self.text, quantized))
        hybrid = types.SimpleNamespace(layers=[types.SimpleNamespace(is_linear=True, linear_attn=object())])
        self.assertIn("recurrent", tree_verify_unsupported_reason(hybrid, [object()]))


class _TinyTargetOps:
    """The slice of dflash-mlx's per-family adapter the DDTree loop uses."""

    def text_model(self, model):
        return model.model

    def embed_tokens(self, model):
        return model.model.embed_tokens

    def logits_from_hidden(self, model, hidden):
        return model.lm_head(hidden)

    def extract_context_feature(self, captured, target_layer_ids):
        import mlx.core as mx

        return mx.concatenate([captured[i + 1] for i in target_layer_ids], axis=-1)

    def make_cache(self, model, *, enable_speculative_linear_cache):
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(model)


class _OracleDraft:
    """Stand-in drafter that knows the true greedy continuation and ranks it
    first, second or not at all depending on depth, so trees accept paths
    off the top-1 chain."""

    block_size = 8
    target_layer_ids = [0, 2]
    layers = [None]

    def __init__(self, embedding, reference, prompt_len):
        self.mask_token_id = VOCAB - 1
        self.embedding = embedding
        self.reference = reference
        self.prompt_len = prompt_len
        self.context = 0
        self.calls = 0

    def __call__(self, *, noise_embedding, target_hidden, cache):
        import mlx.core as mx

        self.context += int(target_hidden.shape[1])
        self.calls += 1
        root_index = self.context - self.prompt_len
        rows = [mx.zeros((self.embedding.shape[1],))]
        for depth in range(1, int(noise_embedding.shape[1])):
            truth_index = root_index + depth
            truth = self.reference[truth_index] if truth_index < len(self.reference) else 0
            other = (truth + 5 + depth) % (VOCAB - 1)
            pattern = (self.calls + depth) % 4
            if pattern == 0:
                first, second = truth, other
            elif pattern == 1:
                first, second = other, truth
            elif pattern == 2:
                first, second = truth, (other + 1) % (VOCAB - 1)
            else:
                first, second = other, (other + 2) % (VOCAB - 1)
            rows.append(8.0 * self.embedding[first] + 6.0 * self.embedding[second])
        return mx.stack(rows)[None]


@unittest.skipUnless(_HAS_MLX, "mlx-lm is not installed")
class GenerateLosslessTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        from mlx_lm.models import qwen3

        mx.random.seed(11)
        args = qwen3.ModelArgs(
            model_type="qwen3", hidden_size=64, num_hidden_layers=3, intermediate_size=128,
            num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=VOCAB, num_key_value_heads=2,
            max_position_embeddings=512, head_dim=16, tie_word_embeddings=False, rope_theta=10000.0,
        )
        self.model = qwen3.Model(args)
        # A freshly initialised model with tied embeddings repeats one token
        # whatever the context. An untied head and stronger attention make
        # greedy output depend on context and positions, so a wrong position,
        # mask or cache entry changes the tokens.
        for layer in self.model.model.layers:
            attn = layer.self_attn
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
                proj.weight = proj.weight * 2.0
        mx.eval(self.model.parameters())

    def _greedy(self, prompt, count):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)
        logits = self.model(mx.array(prompt)[None], cache=cache)[0, -1]
        tokens = []
        for _ in range(count):
            token = int(mx.argmax(logits).item())
            tokens.append(token)
            logits = self.model(mx.array([[token]]), cache=cache)[0, -1]
        return tokens

    def _run(self, budget, prompt, count, *, stop=None):
        from backend_service import ddtree

        reference = self._greedy(prompt, count + 16)
        draft = _OracleDraft(self.model.lm_head.weight, reference, len(prompt))
        runtime = types.ModuleType("dflash_mlx.runtime")
        runtime.resolve_target_ops = lambda model: _TinyTargetOps()
        package = types.ModuleType("dflash_mlx")
        package.runtime = runtime
        compacted = []
        real_compact = ddtree.compact_kv_cache

        def spy(cache, past, accepted):
            compacted.append(list(accepted))
            return real_compact(cache, past, accepted)

        with mock.patch.dict(sys.modules, {"dflash_mlx": package, "dflash_mlx.runtime": runtime}), \
                mock.patch.object(ddtree, "_make_draft_cache", return_value=[]), \
                mock.patch.object(ddtree, "compact_kv_cache", side_effect=spy):
            result = ddtree.generate_ddtree_mlx(
                target_model=self.model,
                tokenizer=types.SimpleNamespace(decode=lambda ids: f"<{ids[0]}>"),
                draft_model=draft,
                prompt_tokens=prompt,
                max_new_tokens=count,
                tree_budget=budget,
                stop_token_ids=stop,
            )
        return result, reference, compacted

    def test_output_equals_plain_greedy_for_every_budget(self):
        prompt = [5, 9, 13, 2, 40, 41, 7, 8, 3]
        off_chain = 0
        for budget in (0, 2, 8, 24, 64):
            with self.subTest(budget=budget):
                result, reference, compacted = self._run(budget, prompt, 60)
                self.assertGreater(len(set(reference[:60])), 3)
                self.assertEqual(result["generated_tokens"], reference[:60])
                self.assertEqual(result["output_tokens"], 60)
                self.assertGreater(result["accepted_from_draft"], 0)
                off_chain += sum(1 for path in compacted if path != list(range(len(path))))
                spans = result["accepted_spans"]
                self.assertEqual(sum(span["length"] for span in spans), len(result["accepted_token_text"]))
        # The oracle ranks the true token second at some depths, so wider
        # trees must accept paths that leave the flat top-1 order.
        self.assertGreater(off_chain, 0)

    def test_stops_at_a_stop_token(self):
        prompt = [5, 9, 13, 2, 40, 41, 7, 8, 3]
        reference = self._greedy(prompt, 40)
        stop = reference[17]
        first = reference.index(stop)
        result, _, _ = self._run(16, prompt, 40, stop=[stop])
        self.assertEqual(result["generated_tokens"], reference[: first + 1])


@unittest.skipUnless(_HAS_MLX, "mlx-lm is not installed")
class SupportProbeTests(unittest.TestCase):
    def test_pure_attention_qwen3_is_supported_and_hybrid_qwen3_next_is_not(self):
        import mlx.core as mx
        from mlx_lm.models import qwen3, qwen3_next
        from backend_service.ddtree import ddtree_unsupported_reason

        mx.random.seed(0)
        dense = qwen3.Model(qwen3.ModelArgs(
            model_type="qwen3", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
            num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=VOCAB, num_key_value_heads=2,
            max_position_embeddings=128, head_dim=8, tie_word_embeddings=True, rope_theta=10000.0,
        ))
        hybrid = qwen3_next.Model(qwen3_next.ModelArgs(
            model_type="qwen3_next", hidden_size=64, num_hidden_layers=4,
            intermediate_size=128, num_attention_heads=4, linear_num_value_heads=4,
            linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
            linear_conv_kernel_dim=4, num_experts=4, num_experts_per_tok=2,
            decoder_sparse_step=1, shared_expert_intermediate_size=64,
            mlp_only_layers=[], moe_intermediate_size=32, rms_norm_eps=1e-6,
            vocab_size=97, num_key_value_heads=2, rope_theta=10000.0,
            partial_rotary_factor=0.25, max_position_embeddings=512, head_dim=16,
            full_attention_interval=2,
        ))
        self.assertIsNone(ddtree_unsupported_reason(dense, _TinyTargetOps()))
        self.assertIn("recurrent", ddtree_unsupported_reason(hybrid, _TinyTargetOps()))
        broken = types.SimpleNamespace()
        self.assertIn("could not be prepared", ddtree_unsupported_reason(broken, _TinyTargetOps()))


@unittest.skipUnless(_HAS_MLX, "mlx-lm is not installed")
class TreeBuildTests(unittest.TestCase):
    def test_top_k_builder_matches_the_logits_builder(self):
        from backend_service.ddtree import build_ddtree_tree, build_ddtree_tree_from_topk, _draft_top_k
        import mlx.core as mx

        rng = np.random.default_rng(3)
        logits = rng.normal(size=(6, 40)).astype(np.float32)
        for budget in (1, 5, 17, 40):
            expected = build_ddtree_tree(logits, budget)
            ids, log_probs = _draft_top_k(mx, mx.array(logits), min(budget, 40), None)
            got = build_ddtree_tree_from_topk(ids, log_probs, budget)
            np.testing.assert_array_equal(got[0], expected[0])
            np.testing.assert_array_equal(got[1], expected[1])
            self.assertEqual(got[2], expected[2])
            np.testing.assert_array_equal(got[4], expected[4])

    def test_budget_one_per_depth_is_the_greedy_chain(self):
        from backend_service.ddtree import build_ddtree_tree_from_topk

        ids = np.array([[4], [9], [2]])
        log_probs = np.array([[-0.1], [-0.2], [-0.3]])
        tokens, depths, parents, child_maps, vis = build_ddtree_tree_from_topk(ids, log_probs, 3)
        self.assertEqual(tokens.tolist(), [4, 9, 2])
        self.assertEqual(depths.tolist(), [1, 2, 3])
        self.assertEqual(parents, [-1, 0, 1, 2])
        self.assertTrue(np.array_equal(vis, np.tril(np.ones((4, 4), dtype=bool))))


class DroppedTreeBudgetTests(unittest.TestCase):
    """A target DDTree cannot verify loads with tree budget 0; asking again
    for the same budget must not count as a profile change (reload)."""

    def _reasons(self, loaded, tree_budget):
        from backend_service.state.metrics import runtime_profile_change_reasons

        return runtime_profile_change_reasons(
            loaded, cache_bits=0, fp16_layers=0, fused_attention=False, cache_strategy="native",
            fit_model_in_memory=True, context_tokens=8192, speculative_decoding=True,
            tree_budget=tree_budget,
        )

    def _loaded(self, **fields):
        from backend_service.inference.base import LoadedModelInfo

        base = dict(
            ref="m", name="m", backend="mlx", source="library", engine="mlx", cacheStrategy="native",
            cacheBits=0, fp16Layers=0, fusedAttention=False, fitModelInMemory=True,
            contextTokens=8192, loadedAt="now", speculativeDecoding=True,
        )
        base.update(fields)
        return LoadedModelInfo(**base)

    def test_dropped_budget_matches_the_request(self):
        loaded = self._loaded(treeBudget=0, requestedTreeBudget=8)
        self.assertEqual(self._reasons(loaded, 8), [])
        self.assertEqual(self._reasons(loaded, 12), ["tree budget 8 -> 12"])
        self.assertEqual(loaded.to_dict()["requestedTreeBudget"], 8)

    def test_without_a_recorded_request_the_running_budget_is_compared(self):
        loaded = self._loaded(treeBudget=16)
        self.assertEqual(self._reasons(loaded, 16), [])
        self.assertEqual(self._reasons(loaded, 8), ["tree budget 16 -> 8"])


if __name__ == "__main__":
    unittest.main()
