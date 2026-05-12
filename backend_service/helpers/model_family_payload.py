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
from typing import Any


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
    if "q4" in lowered or "4-bit" in lowered:
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


def _model_family_payloads(system_stats: dict[str, Any], library: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from backend_service.catalog import MODEL_FAMILIES
    from backend_service.helpers.formatting import _parse_context_label

    payloads: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        variants: list[dict[str, Any]] = []
        for variant in family["variants"]:
            runtime_memory = _estimate_runtime_memory_gb(variant["paramsB"], variant["quantization"])
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

    return payloads
