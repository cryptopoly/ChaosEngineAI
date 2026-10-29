"""Model family payload helpers — catalog enrichment for the dashboard.

Stitches the curated ``MODEL_FAMILIES`` catalog rows with on-disk library
state to produce the variant payloads the chat picker / dashboard
renders. Plus a small ``Reveal in Finder`` shell-out used by the model
list "Reveal" action.

Extracted from ``backend_service/helpers/discovery.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.discovery`` so existing
``from backend_service.helpers.discovery import _model_family_payloads``
imports keep working.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any, Callable


def _reveal_path_in_file_manager(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{resolved} does not exist.")

    system_name = platform.system()
    if system_name == "Darwin":
        command = ["open", "-R", str(resolved)]
    elif system_name == "Windows":
        if resolved.is_file():
            command = ["explorer", f"/select,{resolved}"]
        else:
            command = ["explorer", str(resolved)]
    else:
        command = ["xdg-open", str(resolved.parent if resolved.is_file() else resolved)]

    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _estimate_runtime_memory_gb(params_b: float, quantization: str) -> float:
    lowered = quantization.lower()
    # 1-bit (GGUF Q1_0) and 2-bit / ternary builds (e.g. Bonsai) must not
    # fall through to the full-precision factor — a 27B at 1-bit is ~4 GB
    # of weights, not ~28.
    if "1-bit" in lowered or "q1_" in lowered:
        quant_factor = 0.15
    elif "2-bit" in lowered or "q2_" in lowered or "ternary" in lowered:
        quant_factor = 0.32
    elif "q4" in lowered or "4-bit" in lowered:
        quant_factor = 0.72
    elif "fp8" in lowered or "8" in lowered:
        quant_factor = 0.82
    else:
        quant_factor = 1.0
    return round(max(1.2, params_b * quant_factor + 1.6), 1)


def _variant_available_locally(variant: dict[str, Any], library: list[dict[str, Any]]) -> bool:
    candidates = {
        str(variant.get("repo") or "").lower(),
        str(variant.get("name") or "").lower(),
        str(variant.get("id") or "").lower(),
    }
    compact_candidates = {candidate.split("/")[-1] for candidate in candidates if candidate}
    for item in library:
        name = str(item.get("name") or "").lower()
        if name in candidates or any(candidate and candidate in name for candidate in candidates):
            return True
        if any(candidate and candidate in name for candidate in compact_candidates):
            return True
    return False


def measured_speed_lookup(state: Any) -> Callable[[dict[str, Any]], float | None]:
    """``measured_speed_for`` for ``_model_family_payloads``: this machine's median plain-decode tok/s."""
    from backend_service.helpers.speed_history import measured_decode_tok_s

    def lookup(variant: dict[str, Any]) -> float | None:
        return measured_decode_tok_s(
            state.chat_sessions,
            state.benchmark_runs,
            model_path=None,
            model_repo=str(variant.get("repo") or "") or None,
            strategy="native",
        )

    return lookup


def _model_family_payloads(
    system_stats: dict[str, Any],
    library: list[dict[str, Any]],
    *,
    measured_speed_for: Callable[[dict[str, Any]], float | None] | None = None,
) -> list[dict[str, Any]]:
    """Catalog families with per-variant availability, memory estimates and hardware fit.

    ``measured_speed_for`` returns this machine's own median tok/s for a variant (from chat and
    benchmark history); the best-fit pick uses it, ahead of the bandwidth estimate, to choose
    between variants released in the same month.
    """
    from backend_service.catalog import MODEL_FAMILIES
    from backend_service.helpers.fit import annotate_chat_families
    from backend_service.helpers.formatting import _parse_context_label
    from backend_service.helpers.hardware_class import hardware_class

    payloads: list[dict[str, Any]] = []
    explicit_memory: dict[str, float] = {}
    for family in MODEL_FAMILIES:
        variants: list[dict[str, Any]] = []
        for variant in family["variants"]:
            if variant.get("estimatedMemoryGb"):
                explicit_memory[str(variant["id"])] = float(variant["estimatedMemoryGb"])
            # A row may state its own figure: the parameter-count formula is
            # meaningless when the total is unknown (``paramsB`` 0) or the
            # weights are mapped rather than resident.
            runtime_memory = float(variant.get("estimatedMemoryGb") or 0) or _estimate_runtime_memory_gb(
                variant["paramsB"], variant["quantization"]
            )
            variants.append(
                {
                    **variant,
                    "familyId": family["id"],
                    "estimatedMemoryGb": runtime_memory,
                    "estimatedCompressedMemoryGb": round(max(1.0, runtime_memory * 0.68), 1),
                    "availableLocally": _variant_available_locally(variant, library),
                    "maxContext": _parse_context_label(variant.get("contextWindow")),
                }
            )

        payloads.append(
            {
                **family,
                "variants": variants,
            }
        )

    try:
        hardware = hardware_class(system_stats)
    except Exception:
        return payloads
    return annotate_chat_families(
        payloads, hardware, explicit_memory=explicit_memory, measured_speed_for=measured_speed_for,
    )
