"""The ``fit`` answer for a music catalog variant (``helpers.fit.variant_fit`` with kind ``music``).

Music is judged on the catalog's generation peak, not on a size heuristic: ``estimates.peakGb`` (the
measured process peak on Apple Silicon) or ``estimates.cuda.peakGb`` (a model-card figure on an NVIDIA
GPU), against the machine's usable memory with the same 70 % / 92 % bands as image and video. The row's
``platforms`` list is the platform gate, and ``minMemoryGb`` refuses smaller Macs outright (the XL model).
"""

from __future__ import annotations

from typing import Any

from backend_service.helpers.fit import _positive, _unknown, classify_share
from backend_service.helpers.hardware_class import HardwareClass

# A 64 GB Mac reports a total a little under 64 (GiB rounding); allow for it.
_TOTAL_MEMORY_TOLERANCE_GB = 1.0


def music_need(variant: dict[str, Any], hardware: HardwareClass) -> tuple[float, str, str] | str:
    """``(need GB, reason, basis)`` for a music variant, or the reason it is ``unknown`` here."""
    platforms = {str(p) for p in variant.get("platforms") or []}
    estimates = variant.get("estimates") or {}
    if hardware.kind == "apple":
        if "apple-silicon" not in platforms:
            return "This model does not run on Apple Silicon."
        peak = _positive(estimates.get("peakGb"))
        if peak is None:
            return "No memory figure for this model."
        idle = _positive(estimates.get("idleGb"))
        idle_text = f" ({idle:.1f} GB idle)" if idle else ""
        if estimates.get("basis") == "measured":
            where = estimates.get("measuredOn") or "an Apple Silicon Mac"
            return peak, f"Peak memory {peak:.1f} GB while generating{idle_text}, measured on {where}", "measured"
        return peak, f"Peak memory about {peak:.1f} GB while generating{idle_text} (catalog estimate)", "estimated"
    if hardware.kind == "cuda":
        if "cuda" not in platforms:
            return "This model does not run on an NVIDIA GPU."
        cuda = estimates.get("cuda") or {}
        peak = _positive(cuda.get("peakGb"))
        if peak is None:
            return "Not measured on an NVIDIA GPU, and the CUDA music lane is not built yet."
        return (
            peak,
            f"Peak memory about {peak:.0f} GB on an NVIDIA GPU ({cuda.get('source') or 'model card'}; not measured). "
            "The CUDA music lane is not built yet",
            "estimated",
        )
    return "Needs Apple Silicon (MLX) or an NVIDIA GPU."


def music_fit(variant: dict[str, Any], hardware: HardwareClass) -> dict[str, Any]:
    """The ``fit`` dict (same keys as every other kind)."""
    need = music_need(variant, hardware)
    if isinstance(need, str):
        return _unknown(hardware, need)
    need_gb, reason, basis = need
    usable = hardware.usable_gb
    if usable <= 0:
        return _unknown(hardware, "This machine's usable memory could not be read.")
    minimum = _positive(variant.get("minMemoryGb"))
    if hardware.kind == "apple" and minimum and hardware.total_gb + _TOTAL_MEMORY_TOLERANCE_GB < minimum:
        status = "wont-fit"
        reason = f"Needs a Mac with {minimum:.0f} GB or more. {reason}"
    else:
        status = classify_share(need_gb, usable)
        if status == "wont-fit" and basis == "measured" and need_gb <= usable:
            status = "tight"  # a measured peak did run on this kind of machine
    if variant.get("unloadsOthers"):
        reason += ". Everything else is unloaded first"
    return {
        "status": status,
        "needGb": round(need_gb, 1),
        "usableGb": round(usable, 1),
        "headroomPct": round((usable - need_gb) / usable * 100),
        "reason": reason,
        "basis": basis,
    }
