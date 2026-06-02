"""Cheap diffusers availability probe — version metadata only, no import.

The cache-strategy registry builds ``availableCacheStrategies`` for the
system snapshot, which runs at backend startup (state init → snapshot).
The diffusion strategies (fbcache / taylorseer / magcache / pab /
fastercache) used to answer ``is_available()`` by importing
``diffusers.hooks`` — which transitively pulls ``torch`` + ``torch._dynamo``
+ ``sympy`` and cost ~1.6 s on every cold start (FU-080).

``importlib.metadata.version`` reads the installed package's metadata from
disk without executing its ``__init__`` — so we can answer "is diffusers
new enough for this strategy?" without dragging the whole torch stack into
the startup path. The *real* import stays lazy inside each strategy's
``apply_*`` method, which raises a clean NotImplementedError if the install
is somehow broken despite a satisfactory version.
"""

from __future__ import annotations

import importlib.metadata
from functools import lru_cache


@lru_cache(maxsize=1)
def diffusers_version() -> tuple[int, ...] | None:
    """Installed ``diffusers`` version as an int tuple, or None if absent.

    Reads package metadata only — never imports ``diffusers``.
    """
    try:
        raw = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        return None
    parts: list[int] = []
    for chunk in raw.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def diffusers_at_least(major: int, minor: int) -> bool:
    """True when installed diffusers >= ``major.minor`` (no import)."""
    version = diffusers_version()
    return version is not None and version >= (major, minor)
