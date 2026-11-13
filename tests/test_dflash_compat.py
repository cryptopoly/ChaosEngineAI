"""FU-135 / FU-057: preparing a loaded target for dflash-mlx, and the load-time self-check."""
import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock

from backend_service import dflash_compat

HAVE_MLX = importlib.util.find_spec("mlx") is not None
HAVE_DFLASH = importlib.util.find_spec("dflash_mlx") is not None


@unittest.skipUnless(HAVE_DFLASH, "dflash-mlx not installed")
class PinnedAdapterTests(unittest.TestCase):
    def _ops(self):
        from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps

        return QwenGdnTargetOps()

    def test_the_language_model_wins_over_a_top_level_model_alias(self):
        # FU-135: mlx-lm 0.32 gave the top-level Qwen3.5 / 3.6 Model a ``model`` attribute; the
        # untied lm_head lives on ``language_model``. dflash-mlx 0.1.10 picks it without a patch.
        language_model = SimpleNamespace(model=object(), args=SimpleNamespace(tie_word_embeddings=False))
        top = SimpleNamespace(model=object(), language_model=language_model)
        self.assertIs(self._ops().text_wrapper(top), language_model)

    def test_a_plain_text_model_is_still_its_own_wrapper(self):
        plain = SimpleNamespace(model=object())
        self.assertIs(self._ops().text_wrapper(plain), plain)

    def test_the_default_draft_quant_follows_the_upstream_registry(self):
        self.assertEqual(dflash_compat.default_draft_quant("z-lab/Qwen3.6-27B-DFlash"), "w4")
        self.assertIsNone(dflash_compat.default_draft_quant("z-lab/Qwen3-4B-DFlash-b16"))
        self.assertIsNone(dflash_compat.default_draft_quant("someone/unknown-drafter"))
        self.assertIsNone(dflash_compat.default_draft_quant(None))


class PrepareTargetTests(unittest.TestCase):
    def test_hooks_only_when_the_family_has_no_verify_linears(self):
        ops = SimpleNamespace(
            install_speculative_hooks=Mock(),
            capabilities_for=Mock(return_value=SimpleNamespace(supports_verify_linear=False)),
        )
        self.assertEqual(dflash_compat.prepare_target("model", ops), 0)
        ops.install_speculative_hooks.assert_called_once_with("model")

    @unittest.skipUnless(HAVE_DFLASH, "dflash-mlx not installed")
    def test_verify_linears_are_installed_when_supported(self):
        ops = SimpleNamespace(
            install_speculative_hooks=Mock(),
            capabilities_for=Mock(return_value=SimpleNamespace(supports_verify_linear=True)),
        )
        with mock.patch("dflash_mlx.verify_linear.install_verify_linears", return_value=7) as install:
            self.assertEqual(dflash_compat.prepare_target("model", ops), 7)
        install.assert_called_once_with("model", enable_qmm=True)


@unittest.skipUnless(HAVE_MLX, "mlx not installed")
class SelfCheckTests(unittest.TestCase):
    @staticmethod
    def _model(top: int, vocab: int = 8):
        import mlx.core as mx

        logits = mx.zeros((1, 16, vocab))
        logits[0, -1, top] = 10.0  # type: ignore[index]

        class Model:
            def __call__(self, ids, cache=None):
                return logits

        return Model()

    @staticmethod
    def _ops(top: int, shift: float = 0.0, vocab: int = 8):
        import mlx.core as mx

        logits = mx.zeros((1, 16, vocab))
        logits[0, -1, top] = 10.0 + shift  # type: ignore[index]
        return SimpleNamespace(
            make_cache=lambda model, enable_speculative_linear_cache: [],
            forward_with_hidden_capture=lambda model, **kwargs: (logits, {}),
        )

    def _check(self, model, ops):
        # make_prompt_cache needs a real model; the fake needs none.
        from unittest import mock

        with mock.patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]):
            return dflash_compat.self_check(model, ops)

    def test_agreeing_logits_pass(self):
        self.assertIsNone(self._check(self._model(3), self._ops(3)))

    def test_a_different_next_token_is_reported(self):
        self.assertIn("different next token", self._check(self._model(3), self._ops(5)))

    def test_logits_that_drift_far_are_reported(self):
        self.assertIn("differ from the model's", self._check(self._model(3), self._ops(3, shift=4.0)))

    def test_a_check_that_cannot_run_lets_the_lane_through(self):
        ops = SimpleNamespace(make_cache=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no cache")))
        self.assertIsNone(self._check(self._model(3), ops))


if __name__ == "__main__":
    unittest.main()
