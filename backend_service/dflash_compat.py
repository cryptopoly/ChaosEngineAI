"""Keep the pinned dflash-mlx working with newer mlx-lm releases, and notice when it does not.

dflash-mlx 0.1.5.1 picks the module that owns a Qwen target's output head with
``text_wrapper``: the model itself when it has a ``model`` attribute, otherwise its
``language_model``. mlx-lm 0.32 gave the top-level Qwen3.5 / 3.6 / Next ``Model`` a ``model``
attribute, so the adapter started reading ``tie_word_embeddings`` from the wrong object, took the
default (tied) and projected the hidden state through the embedding matrix. Qwen3.6-27B has an
untied head, so every draft step produced garbage (wrong first token, nothing accepted, 8 tok/s)
while the same pin works on mlx-lm 0.31.3 (FU-135).

``patch_text_wrapper`` makes the adapter prefer ``language_model`` when it owns a ``model``.
``self_check`` then runs a short prompt through the adapter's own forward pass and through the
model, so any other mismatch between the two libraries turns DFlash off instead of producing text
(a check that cannot run at all lets the lane through, so it never disables a working one).
"""

from __future__ import annotations

from typing import Any

_PATCHED_FLAG = "_chaosengine_text_wrapper_patched"


def patch_text_wrapper() -> bool:
    """Prefer ``language_model`` over a top-level ``model`` alias. Idempotent; True when applied."""
    try:
        import dflash_mlx.engine.target_qwen_gdn as gdn
    except ImportError:
        return False
    for value in vars(gdn).values():
        if isinstance(value, type) and value.__module__ == gdn.__name__ and hasattr(value, "text_wrapper"):
            if getattr(value, _PATCHED_FLAG, False):
                return True
            original = value.text_wrapper

            def text_wrapper(self: Any, target_model: Any, _original: Any = original) -> Any:
                language_model = getattr(target_model, "language_model", None)
                if language_model is not None and hasattr(language_model, "model"):
                    return language_model
                return _original(self, target_model)

            value.text_wrapper = text_wrapper
            setattr(value, _PATCHED_FLAG, True)
            return True
    return False


def self_check(target_model: Any, target_ops: Any, *, tolerance: float = 1.0) -> str | None:
    """None when the adapter's forward pass agrees with the model's own, else a reason.

    Sixteen fixed token ids go through ``model(...)`` and through
    ``target_ops.forward_with_hidden_capture`` with a plain cache; the last position's logits must
    pick the same token and stay within ``tolerance`` (they agree to 0.000 when the adapter is
    right; garbage differs by 20 or more). An error while checking returns None.
    """
    try:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        ids = mx.array([list(range(1000, 1016))])
        reference = target_model(ids, cache=make_prompt_cache(target_model))[0, -1].astype(mx.float32)
        cache = target_ops.make_cache(target_model, enable_speculative_linear_cache=False)
        logits, _ = target_ops.forward_with_hidden_capture(
            target_model, input_ids=ids, cache=cache, capture_layer_ids=set()
        )
        candidate = logits[0, -1].astype(mx.float32)
        mx.eval(reference, candidate)
        if int(mx.argmax(candidate).item()) != int(mx.argmax(reference).item()):
            return "the draft runtime picks a different next token than the model does"
        worst = float(mx.abs(candidate - reference).max().item())
        if worst > tolerance:
            return f"the draft runtime's logits differ from the model's by {worst:.1f}"
    except Exception:  # noqa: BLE001 - a check that cannot run must not disable a working lane
        return None
    return None
