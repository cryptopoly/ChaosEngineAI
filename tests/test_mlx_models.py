"""Kolibri 1 on MLX: the vendored architecture, its loader hook and the effort switch.

mlx-lm 0.31 / 0.32 do not ship ``kolibri1`` (ml-explore/mlx-lm#1945 is open), so
``backend_service/mlx_models`` carries the architecture and hands it to
``mlx_lm.load`` through the ``model_file`` hook. The model tests build a tiny
random Kolibri on real MLX and skip without it.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend_service import mlx_models
from backend_service.helpers.quantization import _unsupported_mlx_quantization_reason
from backend_service.mlx_worker_prompt import _build_prompt_text, chat_template_switches

_HAS_MLX = importlib.util.find_spec("mlx_lm") is not None

TINY = {
    "model_type": "kolibri1",
    "hidden_size": 32,
    "num_hidden_layers": 5,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "vocab_size": 64,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 16,
    "sliding_window": 8,
    "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
    "rope_theta": 10000.0,
    "norm_topk_prob": False,
    "tie_word_embeddings": False,
    "max_position_embeddings": 4096,
}


def _snapshot(config: dict | str | None) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="ce-kolibri-"))
    if config is not None:
        text = config if isinstance(config, str) else json.dumps(config)
        (directory / "config.json").write_text(text, encoding="utf-8")
    return directory


class VendoredModelConfigTests(unittest.TestCase):
    def test_a_kolibri_checkpoint_gets_the_vendored_file(self) -> None:
        # Pinned to "mlx-lm has no kolibri1" so the test keeps passing once it does.
        with mock.patch.object(mlx_models, "_mlx_lm_ships", return_value=False):
            config = mlx_models.vendored_model_config(_snapshot({"model_type": "kolibri1"}))
        self.assertIsNotNone(config)
        self.assertEqual(Path(config["model_file"]), Path(mlx_models.__file__).with_name("kolibri1.py"))
        self.assertTrue(Path(config["model_file"]).is_file())

    def test_other_architectures_load_the_normal_way(self) -> None:
        self.assertIsNone(mlx_models.vendored_model_config(_snapshot({"model_type": "qwen3_moe"})))

    def test_a_config_that_names_its_own_model_file_is_left_alone(self) -> None:
        config = {"model_type": "kolibri1", "model_file": "custom.py"}
        self.assertIsNone(mlx_models.vendored_model_config(_snapshot(config)))

    def test_a_native_mlx_lm_module_wins(self) -> None:
        with mock.patch.object(mlx_models, "_mlx_lm_ships", return_value=True):
            self.assertIsNone(mlx_models.vendored_model_config(_snapshot({"model_type": "kolibri1"})))

    def test_the_mlx_lm_probe_reads_the_module_search(self) -> None:
        with mock.patch.object(importlib.util, "find_spec", return_value=object()):
            self.assertTrue(mlx_models._mlx_lm_ships("kolibri1"))
        with mock.patch.object(importlib.util, "find_spec", return_value=None):
            self.assertFalse(mlx_models._mlx_lm_ships("kolibri1"))
        with mock.patch.object(importlib.util, "find_spec", side_effect=ModuleNotFoundError("mlx_lm")):
            self.assertFalse(mlx_models._mlx_lm_ships("kolibri1"))

    def test_a_missing_or_broken_config_is_ignored(self) -> None:
        self.assertIsNone(mlx_models.vendored_model_config(_snapshot(None)))
        self.assertIsNone(mlx_models.vendored_model_config(_snapshot("{not json")))
        self.assertIsNone(mlx_models.vendored_model_config(_snapshot("[1, 2]")))
        self.assertIsNone(mlx_models.vendored_model_config("owner/not-a-local-path"))

    def test_the_mlx_quantisation_the_conversions_use_is_accepted(self) -> None:
        config = {
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
                "model.embed_tokens": {"group_size": 64, "bits": 8},
                "lm_head": {"group_size": 64, "bits": 8},
            }
        }
        self.assertIsNone(_unsupported_mlx_quantization_reason(config))


@unittest.skipUnless(_HAS_MLX, "needs mlx-lm")
class KolibriTinyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mlx.core as mx

        cls.mx = mx
        path = Path(mlx_models.__file__).with_name("kolibri1.py")
        spec = importlib.util.spec_from_file_location("kolibri1_under_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def _model(self):
        mx = self.mx
        mx.random.seed(0)
        model = self.module.Model(self.module.ModelArgs.from_dict(TINY))
        mx.eval(model.parameters())
        return model

    def test_cache_types_follow_the_layer_pattern(self) -> None:
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        caches = self._model().make_cache()
        self.assertEqual([type(c) for c in caches], [RotatingKVCache] * 4 + [KVCache])
        self.assertEqual(caches[0].max_size, TINY["sliding_window"])

    def test_cached_decoding_matches_a_full_forward_pass_across_the_window(self) -> None:
        # 12 tokens against a window of 8: the sliding layers drop old keys, and
        # the rotating cache has to drop exactly the ones the window mask hides.
        import numpy as np

        mx = self.mx
        model = self._model()
        tokens = mx.array([[3, 17, 5, 9, 41, 2, 28, 11, 7, 33, 19, 4]])
        full = np.array(model(tokens))

        caches = model.make_cache()
        steps = [np.array(model(tokens[:, :8], cache=caches))]
        for i in range(8, 12):
            steps.append(np.array(model(tokens[:, i : i + 1], cache=caches)))
        decoded = np.concatenate(steps, axis=1)
        np.testing.assert_allclose(decoded, full, rtol=1e-4, atol=1e-4)

    def test_the_router_picks_on_the_biased_scores_but_weighs_with_the_unbiased_ones(self) -> None:
        import numpy as np

        mx = self.mx
        block = self._model().model.layers[0].mlp
        experts, top_k, hidden = TINY["num_experts"], TINY["num_experts_per_tok"], TINY["hidden_size"]
        block.expert_bias = mx.array(np.linspace(-3.0, 3.0, experts).astype("float32"))
        x = mx.random.normal((1, 5, hidden))
        got = np.array(block(x)).reshape(-1, hidden)

        def silu(a):
            return a / (1.0 + np.exp(-a))

        flat = np.array(x).reshape(-1, hidden)
        logits = flat @ np.array(block.gate.weight).T
        picked = np.argsort(-(logits + np.array(block.expert_bias)), axis=-1)[:, :top_k]
        weights = 1.0 / (1.0 + np.exp(-np.take_along_axis(logits, picked, axis=-1)))
        gate_w, up_w, down_w = (np.array(getattr(block.switch_mlp, n).weight) for n in ("gate_proj", "up_proj", "down_proj"))
        want = np.zeros_like(flat)
        for t in range(flat.shape[0]):
            for slot, e in enumerate(picked[t]):
                want[t] += weights[t, slot] * (down_w[e] @ (silu(gate_w[e] @ flat[t]) * (up_w[e] @ flat[t])))
        shared = block.shared_experts
        sg, su, sd = (np.array(getattr(shared, n).weight) for n in ("gate_proj", "up_proj", "down_proj"))
        want += (silu(flat @ sg.T) * (flat @ su.T)) @ sd.T

        # The bias must change which experts win, or the check proves nothing.
        unbiased = np.argsort(-logits, axis=-1)[:, :top_k]
        self.assertFalse(np.array_equal(np.sort(picked, axis=-1), np.sort(unbiased, axis=-1)))
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)

    def test_mlx_lm_loads_the_checkpoint_through_the_model_file_hook(self) -> None:
        import numpy as np
        from mlx.utils import tree_flatten
        from mlx_lm.utils import load_model

        mx = self.mx
        model = self._model()
        directory = _snapshot(TINY)
        mx.save_safetensors(str(directory / "model.safetensors"), dict(tree_flatten(model.parameters())))

        with mock.patch.object(mlx_models, "_mlx_lm_ships", return_value=False):
            hook = mlx_models.vendored_model_config(directory)
        self.assertIsNotNone(hook)
        loaded, config = load_model(directory, model_config=hook)
        self.assertEqual(config["model_type"], "kolibri1")

        tokens = mx.array([[1, 2, 3, 4, 5, 6]])
        np.testing.assert_allclose(np.array(loaded(tokens)), np.array(model(tokens)), rtol=1e-5, atol=1e-5)

    def test_sanitize_stacks_the_experts_and_renames_the_router_bias(self) -> None:
        mx = self.mx
        model = self._model()
        experts, inter, hidden = TINY["num_experts"], TINY["moe_intermediate_size"], TINY["hidden_size"]
        weights = {"model.layers.0.moe.router.expert_bias": mx.zeros((experts,))}
        for e in range(experts):
            for name, shape in (("gate_proj", (inter, hidden)), ("up_proj", (inter, hidden)), ("down_proj", (hidden, inter))):
                weights[f"model.layers.0.mlp.experts.{e}.{name}.weight"] = mx.full(shape, float(e))

        out = model.sanitize(weights)

        self.assertIn("model.layers.0.mlp.expert_bias", out)
        self.assertNotIn("model.layers.0.moe.router.expert_bias", out)
        self.assertEqual(out["model.layers.0.mlp.switch_mlp.gate_proj.weight"].shape, (experts, inter, hidden))
        self.assertEqual(out["model.layers.0.mlp.switch_mlp.down_proj.weight"].shape, (experts, hidden, inter))
        self.assertEqual(float(out["model.layers.0.mlp.switch_mlp.up_proj.weight"][3, 0, 0]), 3.0)
        self.assertFalse([k for k in out if ".experts." in k])


KOLIBRI_TEMPLATE = (
    "{%- set _sv_reasoning_effort = reasoning_effort | default(none) -%}"
    "{%- if _sv_reasoning_effort is not none -%}"
    "{%- set _sv_thinking_disabled = _sv_reasoning_effort == 'none' -%}{%- endif -%}"
)
LEVELS_ONLY_TEMPLATE = "Reasoning: {{ reasoning_effort | default('medium') }}"


def _tok(template):
    return SimpleNamespace(chat_template=template)


class ChatTemplateSwitchesTests(unittest.TestCase):
    def test_the_chosen_effort_reaches_a_template_that_defines_levels(self) -> None:
        for template in (KOLIBRI_TEMPLATE, LEVELS_ONLY_TEMPLATE):
            for effort in ("low", "medium", "high"):
                request = {"thinkingMode": "auto", "reasoningEffort": effort}
                self.assertEqual(chat_template_switches(_tok(template), request), {"reasoning_effort": effort})

    def test_thinking_off_selects_the_none_level_where_the_template_has_one(self) -> None:
        for request in ({"thinkingMode": "off"}, {}):
            self.assertEqual(chat_template_switches(_tok(KOLIBRI_TEMPLATE), request), {"reasoning_effort": "none"})

    def test_a_template_without_a_none_level_is_left_to_its_default(self) -> None:
        self.assertEqual(chat_template_switches(_tok(LEVELS_ONLY_TEMPLATE), {"thinkingMode": "off"}), {})

    def test_thinking_on_without_an_effort_keeps_the_template_default(self) -> None:
        self.assertEqual(chat_template_switches(_tok(KOLIBRI_TEMPLATE), {"thinkingMode": "auto"}), {})

    def test_other_templates_get_nothing(self) -> None:
        request = {"thinkingMode": "auto", "reasoningEffort": "high"}
        self.assertEqual(chat_template_switches(_tok("{{ messages }}"), request), {})
        self.assertEqual(chat_template_switches(_tok(None), request), {})
        self.assertEqual(chat_template_switches(SimpleNamespace(), request), {})


class PromptBuilderForwardsSwitchesTests(unittest.TestCase):
    class _Tokenizer:
        chat_template = KOLIBRI_TEMPLATE

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(kwargs)
            return "rendered"

    def test_the_switches_are_passed_to_the_template(self) -> None:
        tokenizer = self._Tokenizer()
        text, _ = _build_prompt_text(
            tokenizer, history=[], prompt="Hallo", system_prompt=None,
            template_kwargs={"reasoning_effort": "low"},
        )
        self.assertEqual(text, "rendered")
        self.assertEqual(
            tokenizer.calls, [{"tokenize": False, "add_generation_prompt": True, "reasoning_effort": "low"}]
        )

    def test_without_switches_the_call_is_unchanged(self) -> None:
        tokenizer = self._Tokenizer()
        _build_prompt_text(tokenizer, history=[], prompt="Hallo", system_prompt=None)
        self.assertEqual(tokenizer.calls, [{"tokenize": False, "add_generation_prompt": True}])


if __name__ == "__main__":
    unittest.main()
