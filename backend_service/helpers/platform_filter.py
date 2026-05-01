"""Platform-aware filtering for the image + video model catalogs.

Some catalog variants only run on Apple Silicon: ``mflux`` (image) routes
through ``mflux``/``mlx-lm`` and ``prince-canuma/LTX-2-*`` (video) routes
through ``mlx-video``. Both of those Python packages depend on ``mlx``,
which has no Linux or Windows wheels. Surfacing those variants in the
Image Studio / Video Studio dropdowns on the wrong OS lets users pick
something that cannot run, so this module strips them server-side
before the payload reaches the frontend.

The detection is conservative: a variant is treated as MLX-only iff it
declares so explicitly via ``mlxOnly`` or it carries one of the runtime
labels we know is Apple-only. New runtime labels need to be added here
when they ship — falsely keeping an entry visible is a regression we'd
catch at smoke test, falsely hiding one isn't.
"""

from __future__ import annotations

import platform
from typing import Any


_MLX_ONLY_RUNTIME_MARKERS: tuple[str, ...] = (
    "mflux (MLX native)",
    "mlx-video (MLX native)",
)

_MLX_ONLY_ENGINES: frozenset[str] = frozenset({"mflux", "mlx-video"})


def is_apple_silicon(system: str | None = None, machine: str | None = None) -> bool:
    """True iff the host is Darwin running on arm64.

    Both arguments are exposed for tests so the platform check can be
    pinned without monkeypatching ``platform`` itself. They default to
    the live host values.
    """
    sys_name = system if system is not None else platform.system()
    arch = machine if machine is not None else platform.machine()
    return sys_name == "Darwin" and arch == "arm64"


def is_mlx_only_variant(variant: dict[str, Any]) -> bool:
    """True iff the variant cannot run outside Apple Silicon."""
    if variant.get("mlxOnly") is True:
        return True
    engine = str(variant.get("engine") or "").strip().lower()
    if engine in _MLX_ONLY_ENGINES:
        return True
    runtime = str(variant.get("runtime") or "")
    return any(marker in runtime for marker in _MLX_ONLY_RUNTIME_MARKERS)


def filter_mlx_only_families(
    families: list[dict[str, Any]],
    *,
    on_apple_silicon: bool,
) -> list[dict[str, Any]]:
    """Strip MLX-only variants from a catalog payload on non-Apple hosts.

    On Apple Silicon every variant is preserved untouched. On every other
    OS the MLX-only variants are dropped from each family's ``variants``
    list, and any family whose entire variant set is MLX-only is dropped
    from the result so the UI doesn't render an empty card.

    Returns a new list — the input is not mutated.
    """
    if on_apple_silicon:
        return families

    filtered: list[dict[str, Any]] = []
    for family in families:
        variants = [
            variant
            for variant in family.get("variants", [])
            if not is_mlx_only_variant(variant)
        ]
        if not variants:
            continue
        new_family = dict(family)
        new_family["variants"] = variants
        filtered.append(new_family)
    return filtered
