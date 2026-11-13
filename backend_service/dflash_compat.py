"""Prepare an already loaded MLX target for the pinned dflash-mlx, and notice when they disagree.

Upstream's own loader (``dflash_mlx.runtime.bundle.load_runtime_bundle``) loads a second copy of
the target. The worker reuses the model it already holds, so it repeats the preparation steps that
loader takes: the per-family speculative hooks, the verify-length quantized linears and the
per-drafter quantization default (``prepare_target`` and ``default_draft_quant``).

``self_check`` then runs a short prompt through the adapter's own forward pass and through the
model, so any mismatch between the two libraries turns DFlash off instead of producing text (a
check that cannot run at all lets the lane through, so it never disables a working one). It was
added for FU-135: dflash-mlx 0.1.5.1 read the wrong output head of hybrid Qwen3.5 / 3.6 / Next
models under mlx-lm 0.32 and replied with garbage. dflash-mlx 0.1.10 picks the right head itself
(``text_wrapper`` prefers ``language_model``), so the old patch is gone and only the check stays.
"""

from __future__ import annotations

from typing import Any


def prepare_target(target_model: Any, target_ops: Any) -> int:
    """Install the speculative hooks and, where the family supports it, the verify linears.

    Mirrors ``dflash_mlx.runtime.loading.load_target_bundle`` with the default verify mode.
    Returns the number of linears swapped (0 when the family does not use them).
    """
    target_ops.install_speculative_hooks(target_model)
    if not getattr(target_ops.capabilities_for(target_model), "supports_verify_linear", False):
        return 0
    from dflash_mlx.verify_linear import install_verify_linears

    return int(install_verify_linears(target_model, enable_qmm=True))


def default_draft_quant(draft_ref: str | None) -> str | None:
    """The quantization dflash-mlx applies by default to this drafter (``"w4"`` for the larger
    Qwen3.5 / 3.6 drafters), or None. Matched on the drafter repo, so a local target path works."""
    if not draft_ref:
        return None
    try:
        from dflash_mlx.runtime.registry import MODEL_SUPPORT_SPECS
    except ImportError:
        return None
    for spec in MODEL_SUPPORT_SPECS:
        if spec.draft_ref == draft_ref:
            return spec.defaults.draft_quant or None
    return None


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
