"""Access gate for gated video rows (LTX-2.5 on NVIDIA): no broken offer.

A row flagged ``gated`` is usable when its weights are complete on disk, or when a Hugging Face token is saved (the
download then settles whether that account accepted the licence; a refusal comes back as the usual gated-repo
message). With neither it stays listed but blocked, with the reason, and is never recommended or downloadable.
"""

from __future__ import annotations

from typing import Any


def gated_access_reason(variant: dict[str, Any], *, available_locally: bool, token: str) -> str | None:
    """Why a gated row cannot be used or downloaded now, or ``None`` when it can."""
    if variant.get("gated") is not True or available_locally or str(token or "").strip():
        return None
    repo = str(variant.get("repo") or "this repo")
    return (
        f"{repo} is gated on Hugging Face. Accept its licence on the model page, then add a Hugging Face token "
        "with read access in Settings. The weights are not on this computer yet."
    )
