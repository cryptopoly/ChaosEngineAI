"""The ``fit`` answer for a speech model row (Voice > Models, D-73 M3b).

Same shape and bands as every other area (``helpers.fit``): the speech worker's measured peak (``estimates.peakGb``)
against this machine's usable memory. A row that does not run here, or has not been measured, is ``unknown`` with
the reason; nothing is guessed.
"""

from __future__ import annotations

from typing import Any

from backend_service.helpers.fit import _positive, _unknown, classify_share
from backend_service.helpers.hardware_class import HardwareClass


def voice_fit(peak_gb: Any, measured_on: str | None, supported: bool, hardware: HardwareClass) -> dict[str, Any]:
    if not supported:
        return _unknown(hardware, "This model does not run on this machine.")
    peak = _positive(peak_gb)
    if peak is None:
        return _unknown(hardware, "Not measured yet.")
    usable = hardware.usable_gb
    if usable <= 0:
        return _unknown(hardware, "This machine's usable memory could not be read.")
    status = classify_share(peak, usable)
    if status == "wont-fit" and peak <= usable:
        status = "tight"  # a measured peak did run on this kind of machine
    where = f", measured on {measured_on}" if measured_on else ""
    return {
        "status": status,
        "needGb": round(peak, 1),
        "usableGb": round(usable, 1),
        "headroomPct": round((usable - peak) / usable * 100),
        "reason": f"Peak memory {peak:.1f} GB while speaking{where}",
        "basis": "measured",
    }


def annotate_voice_models(models: list[dict[str, Any]], hardware: HardwareClass) -> dict[str, Any]:
    """Add ``fit`` to each row and return Kokoro's own row data (part of the runtime, no catalog row)."""
    from backend_service.catalog.voice_models import CHAT_MEASURED_ON, KOKORO_CHAT_ESTIMATES  # noqa: PLC0415

    for row in models:
        estimates = row.get("estimates") or {}
        row["fit"] = voice_fit(estimates.get("peakGb"), estimates.get("measuredOn"), bool(row.get("supported")), hardware)
    peak = KOKORO_CHAT_ESTIMATES.get("chatPeakGb")
    return {"peakGb": peak, "fit": voice_fit(peak, CHAT_MEASURED_ON, True, hardware)}
