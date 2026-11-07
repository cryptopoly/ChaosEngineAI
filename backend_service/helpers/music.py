"""Music catalog payloads and the facts about a music repo the shared download code asks for.

``music_model_payloads`` returns the catalog families with ``availableLocally``, ``weightsSource``,
``packInstalled``, ``missingRequires`` and ``fit`` per variant (the same shape idea as the image and
video payloads). The rest answer the questions ``state/downloads.py``, ``helpers/huggingface.py`` and
``helpers/companions.py`` ask about a repo: is it a music repo, which revision to fetch, did the
download produce what the engine needs. A variant counts as available when its weights are already in
the checkpoints layout or in the Hugging Face cache at the pinned revision.
"""

from __future__ import annotations

import copy
from typing import Any

from backend_service.catalog.music_models import MUSIC_MODEL_FAMILIES, music_variants


def find_music_variant(model_id: str | None) -> dict[str, Any] | None:
    """The catalog row for an id or repo; ``None`` for anything else (never a path)."""
    wanted = str(model_id or "").strip()
    return next((v for v in music_variants() if wanted in (v["id"], v["repo"])), None)


def default_music_variant() -> dict[str, Any]:
    family = MUSIC_MODEL_FAMILIES[0]
    return next(v for v in family["variants"] if v["id"] == family["defaultVariantId"])


def is_music_repo(repo: str) -> bool:
    return any(v["repo"] == repo for v in music_variants())


def music_repo_revision(repo: str) -> str | None:
    """The pinned Hub revision for a music repo, else ``None`` (other repos are not pinned)."""
    return next((str(v["revision"]) for v in music_variants() if v["repo"] == repo), None)


def _snapshot_folder(variant: dict[str, Any], folder: str) -> str:
    subdir = str(variant.get("checkpointSubdir") or "")
    if subdir and (folder == subdir or folder.startswith(subdir + "/")):
        return folder[len(subdir):].lstrip("/")
    return folder


def weights_in_checkpoints(variant: dict[str, Any]) -> bool:
    from backend_service import music_env

    root = music_env.checkpoints_dir()
    if not music_env.mirrored_revision_ok(variant):
        return False  # another revision is mirrored: the pinned one must come from the cache
    return all(music_env.has_weights(root / folder) for folder in variant.get("weightFolders") or [])


def weights_in_cache(variant: dict[str, Any]) -> bool:
    from backend_service import music_env

    snapshot = music_env.hf_snapshot(str(variant["repo"]), str(variant["revision"]))
    if snapshot is None:
        return False
    return all(
        music_env.has_weights(snapshot / _snapshot_folder(variant, folder))
        for folder in variant.get("weightFolders") or []
    )


def weights_source(variant: dict[str, Any]) -> str | None:
    """``checkpoints`` (ready for the engine), ``hf-cache`` (mirrored on first use) or ``None``."""
    if weights_in_checkpoints(variant):
        return "checkpoints"
    if weights_in_cache(variant):
        return "hf-cache"
    return None


def music_repo_allow_patterns(repo: str) -> list[str] | None:
    """The files a catalog row needs from its repo (``downloadPatterns``), else ``None`` (the whole repo)."""
    variant = next((v for v in music_variants() if v["repo"] == repo), None)
    patterns = (variant or {}).get("downloadPatterns")
    return [str(p) for p in patterns] if patterns else None


def gated_music_message(repo: str, error: str) -> str | None:
    """The plain licence message for a refused download of a gated music repo, else ``None``.

    Only a refusal (HTTP 401 / 403, "gated", "restricted") is rewritten; any other failure keeps its own text.
    """
    variant = next((v for v in music_variants() if v["repo"] == repo), None)
    if variant is None or not (variant.get("licence") or {}).get("gated"):
        return None
    lowered = str(error).lower()
    if not any(k in lowered for k in ("401", "403", "gated", "restricted", "refused access", "authentication", "not authorized")):
        return None
    return (
        f"{variant['name']} is gated. Open huggingface.co/{repo} while signed in, read and accept the licence "
        "there, save your Hugging Face token in Settings, then download again."
    )


def music_download_validation_error(repo: str) -> str | None:
    """A readable reason a finished music download is unusable, or ``None`` (also for other repos)."""
    variant = next((v for v in music_variants() if v["repo"] == repo), None)
    if variant is None:
        return None
    if weights_in_cache(variant):
        return None
    return (
        f"The download of {repo} did not produce the pinned revision {str(variant['revision'])[:10]} with "
        "all of its weight folders. Retry the download."
    )


def _annotate(variant: dict[str, Any], installed: dict[str, bool]) -> dict[str, Any]:
    from backend_service import music_env

    row = copy.deepcopy(variant)
    source = weights_source(variant)
    row["weightsSource"] = source
    row["availableLocally"] = source is not None
    row["missingRequires"] = [repo for repo in variant.get("requires") or [] if not installed.get(repo)]
    if variant.get("kind") == "sound":
        from backend_service import music_sound_env

        row["packInstalled"] = music_sound_env.is_installed()
        row["textEncoderReady"] = music_sound_env.text_encoder_ready(variant)
        row["availableLocally"] = bool(row["availableLocally"] and row["textEncoderReady"])
    else:
        row["packInstalled"] = music_env.is_installed()
    row["ready"] = bool(row["availableLocally"] and not row["missingRequires"] and row["packInstalled"])
    return row


def music_model_payloads(hardware: Any = None) -> list[dict[str, Any]]:
    """Catalog families with availability and ``fit`` per variant and ``bestFitVariantId`` per family."""
    from backend_service.helpers.fit import annotate_fit

    installed = {v["repo"]: weights_source(v) is not None for v in music_variants()}
    families = []
    for family in MUSIC_MODEL_FAMILIES:
        payload = {k: copy.deepcopy(v) for k, v in family.items() if k != "variants"}
        payload["variants"] = [_annotate(v, installed) for v in family["variants"]]
        families.append(payload)
    return annotate_fit(families, "music", hardware)


def installed_music_variants(hardware: Any = None) -> list[dict[str, Any]]:
    """Variants the engine can run now (weights and required packs on disk), with ``fit``."""
    return [
        v for f in music_model_payloads(hardware) for v in f["variants"]
        if v["availableLocally"] and not v["missingRequires"]
    ]
