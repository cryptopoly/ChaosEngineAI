"""The chat rows to recommend on this machine, best first: the Dashboard's pick and Chat Models' short list.

Both read the same ``fit`` the Models page shows (``helpers.fit``), so the Dashboard, Chat Models and the fit
badge cannot disagree (FU-456, D-105 B). The candidates are each family's best-fit row (``bestFitVariantId``,
which only ever names a row that ``fits``); they are ordered with the same rule ``best_fit_variant_id`` uses
inside a family: newest release, then the larger model, then the faster, then the smaller. Speed is this
machine's own measured median when there is one (chat and benchmark history), else the row's RTX 4090 figure
on a comparable card (``measuredGpuTokS``), else the bandwidth estimate.
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
# A catalogue row's RTX 4090 speed (``measuredGpuTokS``) is shown only on a discrete NVIDIA card this large.
REFERENCE_MIN_GB = 20.0


def _speed(
    variant: dict[str, Any],
    hardware: HardwareClass,
    measured_speed_for: Callable[[dict[str, Any]], float | None] | None,
    bandwidth: Callable[[], float],
) -> tuple[float | None, str | None]:
    """``(tok/s, basis)`` or ``(None, None)``. Basis: ``measured`` (this computer's own runs), ``rtx4090`` (the
    catalogue row's plain-decode figure from the RTX 4090, used only on a discrete NVIDIA card of 20 GB or more,
    where it is a fair guide), else ``estimated`` (memory bandwidth)."""
    if measured_speed_for is not None:
        try:
            measured = measured_speed_for(variant)
        except Exception:
            measured = None
        if measured:
            return round(float(measured), 1), "measured"
    reference = _positive(variant.get("measuredGpuTokS"))
    if reference is not None and hardware.kind == "cuda" and not hardware.unified and hardware.usable_gb >= REFERENCE_MIN_GB:
        return round(reference, 1), "rtx4090"
    try:
        estimate = estimate_chat_tok_s(
            variant, arch=None, bandwidth_gbps=bandwidth(), cuda=hardware.kind == "cuda" and not hardware.unified,
        )
    except Exception:
        estimate = None
    return (round(float(estimate), 1), "estimated") if estimate else (None, None)


def _count(value: Any) -> int | None:
    number = _positive(value)
    return int(number) if number is not None else None


def chat_picks(
    families: list[dict[str, Any]],
    hardware: HardwareClass,
    *,
    measured_speed_for: Callable[[dict[str, Any]], float | None] | None = None,
    bandwidth_gbps: Callable[[], float] | None = None,
    limit: int = PICK_LIMIT,
    hf_stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Up to ``limit`` recommended chat rows, best first (empty when no row fits comfortably).

    ``hf_stats`` is ``helpers.hf_stats.load_stats()``; each row then carries the Hub's ``likes`` and
    ``downloads`` (last 30 days) with ``statsAsOf``, or ``None`` for each when there is no number for its repo."""
    stats_repos = (hf_stats or {}).get("repos") or {}
    stats_as_of = (hf_stats or {}).get("fetchedAt")
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
        counters = stats_repos.get(str(variant.get("repo") or "")) or {}
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
            "likes": _count(counters.get("likes")),
            "downloads": _count(counters.get("downloads")),
            "statsAsOf": stats_as_of if counters else None,
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
    use_hf_stats: bool = False,
) -> dict[str, Any]:
    """Attach ``chatPicks`` and ``bestFit`` to the Dashboard recommendation; the headroom becomes the pick's own.

    Never raises: without a pick (nothing fits comfortably, or the hardware could not be read) the
    recommendation keeps its size-class title and ``headroomPercent`` stays ``None``.
    """
    recommendation.setdefault("chatPicks", [])
    recommendation.setdefault("bestFit", None)
    try:
        from backend_service.helpers.hardware_class import hardware_class

        hf_stats = None
        if use_hf_stats:
            from backend_service.helpers import hf_stats as hf_stats_module

            hf_stats = hf_stats_module.load_stats()
        picks = chat_picks(
            families, hardware_class(system_stats), measured_speed_for=measured_speed_for, hf_stats=hf_stats,
        )
        if use_hf_stats and picks:
            # Never waited on: the numbers appear on a later load once the background fetch has written them.
            hf_stats_module.refresh_in_background([str(pick["repo"]) for pick in picks if pick.get("repo")])
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
