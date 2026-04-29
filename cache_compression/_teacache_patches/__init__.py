"""Vendored TeaCache forward patches.

Each module in this package exports a ``teacache_forward`` callable that
``cache_compression.teacache.apply_diffusers_hook`` installs onto a diffusers
transformer class at runtime. We do **not** monkey-patch at import time —
patches are applied only when the cache strategy is requested.

Upstream: https://github.com/ali-vilab/TeaCache (Apache 2.0). Copyright and
attribution headers are preserved on each vendored module.
"""
