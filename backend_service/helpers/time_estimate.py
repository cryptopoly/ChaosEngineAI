"""Studio time estimates scaled from a row's catalog figure to the size, frames and steps asked for (FU-165).

``seconds = catalog x (fixed + (1 - fixed) x steps/ref_steps x w(T)/w(T_ref))`` with ``w(T) = T + T^2/quad`` and
``T = W/16 x H/16 x F/4`` (images: F = 1). ``fixed`` is the share of the catalog time that does not grow with the
work (load, text encoding, decode); ``quad`` sets how fast attention cost overtakes the linear part. The catalog row
says what its figure was measured at in ``estimateRef`` (``width``, ``height``, ``frames``, ``steps``, optional
``fixed``, ``quad`` and ``ignoresSteps`` for engines that run a set schedule). Without a reference the flat catalog
figure is returned. Pure and import-light: ``src/utils/timeEstimate.ts`` mirrors it and both are checked against
``tests/fixtures/time_estimate_cases.json``.
"""

from __future__ import annotations

import math
from typing import Any

# Fitted on four Wan 2.2 TI2V runs on an RTX 4090 (within 5 %), FU-165.
VIDEO_FIXED = 0.04
DEFAULT_QUAD = 53300.0
# Image rows that were not timed: diffusers on CUDA fits about 0.6 (FLUX.2 Klein, 1024 vs 2048 px). mflux on an
# M4 Max measured 0.01 (FLUX 3 Klein), 0.1 and 0.35 (Qwen-Image-2.1 base and Turbo) and 0.25 (Iris-3B); 0.1 is the
# middle (docs/research/time-estimate-fit-2026-11.md).
IMAGE_FIXED = 0.6
MFLUX_FIXED = 0.1


def default_fixed(kind: str, engine: str | None = None) -> float:
    """The engine default share for a row with no measured ``fixed`` of its own."""
    if kind == "video":
        return VIDEO_FIXED
    if engine == "mflux":
        return MFLUX_FIXED
    return IMAGE_FIXED


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def _round1(value: float) -> float:
    """Half-up to one decimal, the same as ``Math.round(x * 10) / 10`` in the TypeScript mirror."""
    return math.floor(value * 10 + 0.5) / 10


def _work(width: float, height: float, frames: float, quad: float) -> float:
    tokens = width / 16 * height / 16 * frames / 4
    return tokens + tokens * tokens / quad


def scaled_seconds(
    catalog_seconds: Any,
    ref: Any,
    width: Any,
    height: Any,
    frames: Any = 1,
    steps: Any = None,
    fixed: float | None = None,
    quad: float | None = None,
) -> float | None:
    """Seconds for this run, or ``None`` when the row has no catalog time. Never raises.

    A missing or broken reference, or a missing size, gives the flat catalog figure. ``frames`` / ``steps`` left
    out (or not positive) count as the reference's. The row's own ``fixed`` / ``quad`` win over the arguments.
    """
    catalog = _positive(catalog_seconds)
    if catalog is None:
        return None
    try:
        if not isinstance(ref, dict):
            return _round1(catalog)
        ref_w, ref_h = _positive(ref.get("width")), _positive(ref.get("height"))
        ref_f, ref_s = _positive(ref.get("frames")), _positive(ref.get("steps"))
        w, h = _positive(width), _positive(height)
        if None in (ref_w, ref_h, ref_f, ref_s, w, h):
            return _round1(catalog)
        f = _positive(frames) or ref_f
        s = ref_s if ref.get("ignoresSteps") else (_positive(steps) or ref_s)
        share = _positive(ref.get("fixed")) or _positive(fixed)
        share = VIDEO_FIXED if share is None else min(share, 1.0)
        q = _positive(ref.get("quad")) or _positive(quad) or DEFAULT_QUAD
        ratio = (s / ref_s) * (_work(w, h, f, q) / _work(ref_w, ref_h, ref_f, q))
        seconds = catalog * (share + (1.0 - share) * ratio)
        if not math.isfinite(seconds):
            return _round1(catalog)
        return _round1(seconds)
    except Exception:  # noqa: BLE001 - an estimate never fails the caller
        return _round1(catalog)


def variant_seconds(
    variant: dict[str, Any] | None,
    kind: str,
    width: Any,
    height: Any,
    frames: Any = 1,
    steps: Any = None,
) -> float | None:
    """A catalog row's time for these settings, with the engine default share when the row has none."""
    if not variant:
        return None
    return scaled_seconds(
        variant.get("estimatedGenerationSeconds"),
        variant.get("estimateRef"),
        width,
        height,
        1 if kind == "image" else frames,
        steps,
        fixed=default_fixed(kind, variant.get("engine")),
    )
