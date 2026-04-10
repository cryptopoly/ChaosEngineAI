"""TurboQuant MLX adapter for ChaosEngineAI.

Provides PolarQuant-style adaptive KV cache compression for MLX on Apple
Silicon.  Each transformer layer gets either a full-precision (fp16) KV cache
or a quantised one, depending on the ``fp16_layers`` parameter which keeps the
first and last N layers at higher fidelity.

Required hooks
--------------
ChaosEngineAI's ``TurboQuantStrategy`` probes for two top-level callables:

* ``make_adaptive_cache(num_layers, bits, fp16_layers, fused, model)``
* ``apply_patch()``

Both are exported from this module.
"""

from __future__ import annotations

__version__ = "0.2.0"

_patched = False


def apply_patch() -> None:
    """Prepare the MLX/Metal runtime for quantised KV caches.

    This is intentionally a no-op in the current build because ``mlx_lm``'s
    ``QuantizedKVCache`` already handles kernel dispatch internally.  The hook
    exists so that future TurboQuant Metal kernel bundles can be injected here.
    """
    global _patched
    _patched = True


def make_adaptive_cache(
    num_layers: int,
    *,
    bits: int = 3,
    fp16_layers: int = 0,
    fused: bool = False,
    model: object | None = None,
) -> list:
    """Build a per-layer KV cache list for ``mlx-lm`` generate.

    Parameters
    ----------
    num_layers:
        Total number of transformer layers in the model.
    bits:
        Quantisation bit-width for compressed layers (1-8, default 3).
    fp16_layers:
        Number of layers to keep at full precision at the *start* and *end*
        of the stack.  For example ``fp16_layers=4`` means layers 0-3 and
        (N-4)-(N-1) use an unquantised cache.
    fused:
        Reserved for future fused-kernel path.  Currently unused.
    model:
        The ``mlx.nn.Module`` model instance.  Passed through to
        ``mlx_lm``'s ``make_prompt_cache`` when available, otherwise
        ignored.

    Returns
    -------
    list
        A prompt-cache list compatible with ``mlx_lm.utils.generate_step``.
    """
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    clamped_bits = max(2, min(8, bits))
    fp16_count = max(0, min(fp16_layers, num_layers // 2))

    cache: list = []
    for i in range(num_layers):
        is_fp16 = i < fp16_count or i >= (num_layers - fp16_count)
        if is_fp16:
            cache.append(KVCache())
        else:
            cache.append(QuantizedKVCache(group_size=64, bits=clamped_bits))
    return cache
