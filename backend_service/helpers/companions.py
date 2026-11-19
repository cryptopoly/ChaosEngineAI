"""Files from other repos that a download brings along with its model.

A model sometimes cannot run without a file its author publishes in a repo
of its own: NVIDIA ships the Nemotron 3.5 Bolt DFlash drafter apart from the
model, and Iris-3B reads its text encoder from a Qwen repo. Fetching them
with the model means the first launch works offline and the progress the
user sees covers the whole download, instead of a silent second download on
first use.

A Nunchaku variant is downloaded under its pack repo (the 4-bit transformer)
and brings from the base FLUX.1 repo only the parts the pipeline reads around
that transformer: text encoders, tokenizers, VAE, scheduler and the
transformer's config, never the 24 GB of bf16 transformer shards.
"""

from __future__ import annotations

# The base-repo files a Nunchaku variant reads (about 9.9 GB of the 57.9 GB FLUX.1 repos).
NUNCHAKU_BASE_ALLOW_PATTERNS: list[str] = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "vae/*",
    "transformer/config.json",
    "*.md",
    "LICENSE*",
]


def companion_downloads(repo_id: str) -> list[tuple[str, list[str]]]:
    """``(repo, allow patterns)`` pairs to fetch after ``repo_id`` itself."""
    from backend_service.catalog import (  # noqa: PLC0415
        IMAGE_MODEL_FAMILIES,
        MUSIC_MODEL_FAMILIES,
        VIDEO_MODEL_FAMILIES,
    )
    from backend_service.helpers.text_gguf import pinned_drafter  # noqa: PLC0415

    found: list[tuple[str, list[str]]] = []
    drafter = pinned_drafter(repo_id)
    if drafter is not None:
        found.append((drafter[0], [drafter[1]]))
    # Music rows name the packs they need in ``requires`` (separate catalog rows the user downloads
    # with their own progress), so they carry no companions today; the branch keeps that explicit.
    for family in (*IMAGE_MODEL_FAMILIES, *VIDEO_MODEL_FAMILIES, *MUSIC_MODEL_FAMILIES):
        for variant in family["variants"]:
            if variant.get("nunchakuRepo") and str(variant["nunchakuRepo"]) == repo_id:
                entry = (str(variant["repo"]), list(NUNCHAKU_BASE_ALLOW_PATTERNS))
                if entry not in found:
                    found.append(entry)
                continue
            if str(variant.get("repo") or "") != repo_id:
                continue
            for companion in variant.get("downloadCompanions") or []:
                entry = (str(companion["repo"]), [str(p) for p in companion["patterns"]])
                if entry not in found:
                    found.append(entry)
    return found
