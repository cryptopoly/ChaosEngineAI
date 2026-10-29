"""Files from other repos that a download brings along with its model.

A model sometimes cannot run without a file its author publishes in a repo
of its own: NVIDIA ships the Nemotron 3.5 Bolt DFlash drafter apart from the
model, and Iris-3B reads its text encoder from a Qwen repo. Fetching them
with the model means the first launch works offline and the progress the
user sees covers the whole download, instead of a silent second download on
first use.
"""

from __future__ import annotations


def companion_downloads(repo_id: str) -> list[tuple[str, list[str]]]:
    """``(repo, allow patterns)`` pairs to fetch after ``repo_id`` itself."""
    from backend_service.catalog import IMAGE_MODEL_FAMILIES  # noqa: PLC0415
    from backend_service.helpers.text_gguf import pinned_drafter  # noqa: PLC0415

    found: list[tuple[str, list[str]]] = []
    drafter = pinned_drafter(repo_id)
    if drafter is not None:
        found.append((drafter[0], [drafter[1]]))
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if str(variant.get("repo") or "") != repo_id:
                continue
            for companion in variant.get("downloadCompanions") or []:
                entry = (str(companion["repo"]), [str(p) for p in companion["patterns"]])
                if entry not in found:
                    found.append(entry)
    return found
