"""The chat rows to recommend on this machine, best first: the Dashboard's pick and Chat Models' short list.

Both read the same ``fit`` the Models page shows (``helpers.fit``), so the Dashboard, Chat Models and the fit
badge cannot disagree (FU-456, D-105 B). The candidates are each family's best-fit row (``bestFitVariantId``,
which only ever names a row that ``fits``); they are ordered with the same rule ``best_fit_variant_id`` uses
inside a family: newest release, then the larger model, then the faster, then the smaller. Speed is this
machine's own measured median when there is one (chat and benchmark history), else the bandwidth estimate.
The headroom shown beside the pick is that row's own ``fit.headroomPct``, not a constant per size class.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

from backend_service.helpers.fit import _positive, _release_key, estimate_chat_tok_s
from backend_service.helpers.hardware_class import HardwareClass

logger = logging.getLogger(__name__)

PICK_LIMIT = 5


def _speed(
    variant: dict[str, Any],
    hardware: HardwareClass,
    measured_speed_for: Callable[[dict[str, Any]], float | None] | None,
    bandwidth: Callable[[], float],
) -> tuple[float | None, str | None]:
    """``(tok/s, "measured" | "estimated")`` or ``(None, None)``."""
    if measured_speed_for is not None:
        try:
            measured = measured_speed_for(variant)
        except Exception:
            measured = None
        if measured:
            return round(float(measured), 1), "measured"
    try:
        estimate = estimate_chat_tok_s(
            variant, arch=None, bandwidth_gbps=bandwidth(), cuda=hardware.kind == "cuda" and not hardware.unified,
        )
    except Exception:
        estimate = None
    return (round(float(estimate), 1), "estimated") if estimate else (None, None)


def chat_picks(
    families: list[dict[str, Any]],
    hardware: HardwareClass,
    *,
    measured_speed_for: Callable[[dict[str, Any]], float | None] | None = None,
    bandwidth_gbps: Callable[[], float] | None = None,
    limit: int = PICK_LIMIT,
) -> list[dict[str, Any]]:
    """Up to ``limit`` recommended chat rows, best first (empty when no row fits comfortably)."""
    cached: list[float] = []

    def bandwidth() -> float:
        if not cached:
            if bandwidth_gbps is not None:
                cached.append(bandwidth_gbps())
            else:
                from backend_service.helpers.bandwidth import memory_bandwidth

                cached.append(memory_bandwidth()[0])
        return cached[0]

    rows: list[dict[str, Any]] = []
    for family in families:
        best_id = family.get("bestFitVariantId")
        if not best_id:
            continue
        variant = next((v for v in family.get("variants") or [] if v.get("id") == best_id), None)
        fit = (variant or {}).get("fit") or {}
        if variant is None or fit.get("status") != "fits":
            continue
        tok_s, tok_s_basis = _speed(variant, hardware, measured_speed_for, bandwidth)
        rows.append({
            "variantId": str(variant["id"]),
            "familyId": str(family.get("id") or variant.get("familyId") or ""),
            "familyName": family.get("name"),
            "name": variant.get("name"),
            "repo": variant.get("repo"),
            "sizeGb": _positive(variant.get("sizeGb")),
            "paramsB": _positive(variant.get("paramsB")),
            "format": variant.get("format"),
            "quantization": variant.get("quantization"),
            "releaseDate": variant.get("releaseDate"),
            "availableLocally": bool(variant.get("availableLocally")),
            "fit": dict(fit),
            "tokS": tok_s,
            "tokSBasis": tok_s_basis,
            "hardwareLabel": hardware.label,
            "_release": _release_key(variant),
        })

    def order(row: dict[str, Any]) -> tuple[Any, ...]:
        year, month = row["_release"]
        return (
            -year, -month,
            -(row["paramsB"] or 0.0),
            -(row["tokS"] or 0.0),
            row["fit"].get("needGb") or math.inf,
            row["sizeGb"] or math.inf,
            row["variantId"],
        )

    rows.sort(key=order)
    for row in rows:
        row.pop("_release", None)
    return rows[:limit]


def apply_chat_picks(
    recommendation: dict[str, Any],
    families: list[dict[str, Any]],
    system_stats: dict[str, Any],
    *,
    measured_speed_for: Callable[[dict[str, Any]], float | None] | None = None,
) -> dict[str, Any]:
    """Attach ``chatPicks`` and ``bestFit`` to the Dashboard recommendation; the headroom becomes the pick's own.

    Never raises: without a pick (nothing fits comfortably, or the hardware could not be read) the
    recommendation keeps its size-class title and ``headroomPercent`` stays ``None``.
    """
    recommendation.setdefault("chatPicks", [])
    recommendation.setdefault("bestFit", None)
    try:
        from backend_service.helpers.hardware_class import hardware_class

        picks = chat_picks(families, hardware_class(system_stats), measured_speed_for=measured_speed_for)
    except Exception:
        logger.warning("Could not pick the recommended chat models", exc_info=True)
        return recommendation
    recommendation["chatPicks"] = picks
    if picks:
        best = picks[0]
        recommendation["bestFit"] = best
        recommendation["headroomPercent"] = best["fit"].get("headroomPct")
        recommendation["targetModel"] = best["name"]
    return recommendation
