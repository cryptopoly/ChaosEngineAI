"""FU-135: dflash-mlx 0.1.5.1 read the wrong output head under mlx-lm 0.32."""
import importlib.util
import unittest
from types import SimpleNamespace

from backend_service import dflash_compat

HAVE_MLX = importlib.util.find_spec("mlx") is not None
HAVE_DFLASH = importlib.util.find_spec("dflash_mlx") is not None


@unittest.skipUnless(HAVE_DFLASH, "dflash-mlx not installed")
class TextWrapperPatchTests(unittest.TestCase):
    def _ops(self):
        import dflash_mlx.engine.target_qwen_gdn as gdn

        cls = next(v for v in vars(gdn).values()
                   if isinstance(v, type) and v.__module__ == gdn.__name__ and hasattr(v, "text_wrapper"))
        return cls()

    def test_the_language_model_wins_over_a_top_level_model_alias(self):
        # mlx-lm 0.32: the top-level Model gained a ``model`` attribute, the untied lm_head and
        # tie_word_embeddings live one level down, on ``language_model``.
        self.assertTrue(dflash_compat.patch_text_wrapper())
        language_model = SimpleNamespace(model=object(), args=SimpleNamespace(tie_word_embeddings=False))
        top = SimpleNamespace(model=object(), language_model=language_model)
        self.assertIs(self._ops().text_wrapper(top), language_model)

    def test_a_plain_text_model_is_still_its_own_wrapper(self):
        dflash_compat.patch_text_wrapper()
        plain = SimpleNamespace(model=object())
        self.assertIs(self._ops().text_wrapper(plain), plain)

    def test_the_patch_is_applied_once(self):
        dflash_compat.patch_text_wrapper()
        first = type(self._ops()).text_wrapper
        dflash_compat.patch_text_wrapper()
        self.assertIs(type(self._ops()).text_wrapper, first)


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
