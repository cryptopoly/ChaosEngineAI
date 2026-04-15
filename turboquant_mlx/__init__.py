"""TurboQuant MLX adapter for ChaosEngineAI.

Provides TurboQuant KV cache compression for MLX on Apple Silicon using
the ``turboquant-mlx-full`` package's ``TurboQuantKVCache`` — Hadamard
rotation + Lloyd-Max codebook compression achieving ~4.6x at 3-bit.

The adapter detects two integration paths:

1. **Full TurboQuant** (``turboquant-mlx-full`` installed):
   Uses ``TurboQuantKVCache`` which stores keys/values in compressed
   TurboQuant format and dequantizes to float16 for attention.  Works
   with all model architectures including hybrid attention (Qwen3.5,
   GPT-OSS) and variable head dimensions (GLM-4).

2. **Fallback** (basic ``QuantizedKVCache`` from mlx-lm):
   Used when ``turboquant-mlx-full`` is not installed.  May have
   compatibility issues with mlx-lm 0.31.x.

Required hooks
--------------
ChaosEngineAI's ``TurboQuantStrategy`` probes for two top-level callables:

* ``make_adaptive_cache(num_layers, bits, fp16_layers, fused, model)``
* ``apply_patch()``

Both are exported from this module.

Install: ``./.venv/bin/python3 -m pip install turboquant-mlx-full``
"""

from __future__ import annotations

__version__ = "0.3.0"

_patched = False


def _find_pip_turboquant_path() -> str | None:
    """Find the pip-installed turboquant-mlx-full package in site-packages.

    Our local ``turboquant_mlx/`` directory shadows the pip package, so
    we locate it directly in site-packages.
    """
    import sysconfig
    from pathlib import Path

    site_packages = sysconfig.get_path("purelib")
    if site_packages:
        candidate = Path(site_packages) / "turboquant_mlx" / "layers" / "polar_kv_cache.py"
        if candidate.exists():
            return str(Path(site_packages) / "turboquant_mlx")
    return None


def _has_full_turboquant() -> bool:
    """Check if the full TurboQuantKVCache implementation is available."""
    return _find_pip_turboquant_path() is not None


_tq_kv_cache_class = None


def _import_turboquant_kv_cache():
    """Import TurboQuantKVCache from the pip-installed package.

    The pip package installs as ``turboquant_mlx`` which is shadowed by
    our local adapter directory.  We work around this by temporarily
    making the pip package's modules importable under an alias, executing
    the module, then cleaning up.
    """
    global _tq_kv_cache_class
    if _tq_kv_cache_class is not None:
        return _tq_kv_cache_class

    pip_path = _find_pip_turboquant_path()
    if not pip_path:
        raise ImportError("turboquant-mlx-full is not installed")

    import sys
    from pathlib import Path

    site_packages = str(Path(pip_path).parent)

    # Temporarily swap our local turboquant_mlx out of sys.modules
    # so the pip package's internal imports resolve to site-packages.
    saved_module = sys.modules.pop("turboquant_mlx", None)
    saved_sub = {k: sys.modules.pop(k, None) for k in list(sys.modules) if k.startswith("turboquant_mlx.")}

    # Add site-packages to front of path so the pip package wins
    sys.path.insert(0, site_packages)
    try:
        import importlib
        pip_mod = importlib.import_module("turboquant_mlx.layers.polar_kv_cache")
        _tq_kv_cache_class = pip_mod.TurboQuantKVCache
    finally:
        sys.path.remove(site_packages)
        # Restore our local module
        if saved_module is not None:
            sys.modules["turboquant_mlx"] = saved_module
        for k, v in saved_sub.items():
            if v is not None:
                sys.modules[k] = v

    return _tq_kv_cache_class


def apply_patch() -> None:
    """Prepare the MLX/Metal runtime for TurboQuant KV caches.

    With the full ``turboquant-mlx-full`` package, this verifies the
    ``TurboQuantKVCache`` class is importable.  Otherwise a no-op.
    """
    global _patched
    if _has_full_turboquant():
        _import_turboquant_kv_cache()  # verify import works
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
    from mlx_lm.models.cache import KVCache

    clamped_bits = max(2, min(8, bits))
    fp16_count = max(0, min(fp16_layers, num_layers // 2))

    # Prefer the full TurboQuantKVCache which handles its own
    # rotation, codebook, and mask construction — compatible with
    # all mlx-lm versions and model architectures.
    if _has_full_turboquant():
        TurboQuantKVCache = _import_turboquant_kv_cache()

        cache: list = []
        for i in range(num_layers):
            is_fp16 = i < fp16_count or i >= (num_layers - fp16_count)
            if is_fp16:
                cache.append(KVCache())
            else:
                cache.append(TurboQuantKVCache(
                    tq_bits=clamped_bits,
                    group_size=64,
                ))
        return cache

    # Fallback: use mlx-lm's built-in QuantizedKVCache.
    # This may fail with mlx-lm 0.31.x on some models due to
    # create_attention_mask signature changes.
    from mlx_lm.models.cache import QuantizedKVCache

    cache = []
    for i in range(num_layers):
        is_fp16 = i < fp16_count or i >= (num_layers - fp16_count)
        if is_fp16:
            cache.append(KVCache())
        else:
            cache.append(QuantizedKVCache(group_size=64, bits=clamped_bits))
    return cache
