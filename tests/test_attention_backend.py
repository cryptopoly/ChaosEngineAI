"""SageAttention must not break a model that passes an attention mask."""
import unittest
from types import SimpleNamespace
from unittest import mock

from backend_service.helpers import attention_backend


class _Transformer:
    def __init__(self):
        self.backends: list[str] = []

    def set_attention_backend(self, name: str) -> None:
        self.backends.append(name)


class _Pipeline:
    error: Exception | None = None

    def __init__(self):
        self.transformer = _Transformer()
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        if self.error is not None and self.transformer.backends[-1] == "sage":
            raise self.error
        return f"clip:{prompt}"


def _applied(pipeline):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    with mock.patch.dict("sys.modules", {"torch": fake_torch}), \
            mock.patch("importlib.util.find_spec", return_value=object()):
        return attention_backend.maybe_apply_sage_attention(pipeline)


class SageMaskFallbackTests(unittest.TestCase):
    def test_a_model_without_a_mask_keeps_sage(self):
        pipeline = _Pipeline()
        self.assertEqual(_applied(pipeline), "Attention: SageAttention")
        self.assertEqual(pipeline("a"), "clip:a")
        self.assertEqual(pipeline.transformer.backends, ["sage"])

    def test_a_mask_error_goes_back_to_pytorch_attention_and_repeats_the_call(self):
        pipeline = _Pipeline()
        pipeline.error = ValueError("`attn_mask` is not supported for sage attention")
        _applied(pipeline)
        self.assertEqual(pipeline("a"), "clip:a")
        self.assertEqual(pipeline.transformer.backends, ["sage", "native"])
        self.assertEqual(pipeline.calls, 2)
        self.assertEqual(pipeline("b"), "clip:b")  # the failed attempt is not repeated
        self.assertEqual(pipeline.calls, 3)

    def test_the_pipeline_keeps_its_class_name_and_module(self):
        pipeline = _Pipeline()
        _applied(pipeline)
        self.assertEqual(type(pipeline).__name__, "_Pipeline")
        self.assertEqual(type(pipeline).__module__, _Pipeline.__module__)
        self.assertIsInstance(pipeline, _Pipeline)

    def test_the_call_signature_survives_for_the_runtime_kwarg_filter(self):
        import inspect

        pipeline = _Pipeline()
        _applied(pipeline)
        self.assertEqual(list(inspect.signature(pipeline.__call__).parameters), ["prompt"])

    def test_any_other_error_is_raised(self):
        pipeline = _Pipeline()
        pipeline.error = ValueError("something else")
        _applied(pipeline)
        with self.assertRaises(ValueError):
            pipeline("a")
        self.assertEqual(pipeline.transformer.backends, ["sage"])


if __name__ == "__main__":
    unittest.main()
