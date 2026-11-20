"""Image repo validation + friendly download-error helpers.

Pure helpers used at the boundary of the image download flow:
- "is this a known image repo?" predicates
- "is the local snapshot loadable?" check
- friendly translation of HF gated-repo errors so the UI can tell users to
  request access instead of just surfacing a 401.

Extracted from ``backend_service/helpers/images.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.images`` so existing
``from backend_service.helpers.images import _image_download_validation_error``
imports keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend_service.catalog import IMAGE_MODEL_FAMILIES
from backend_service.helpers.discovery import _candidate_model_dirs
from backend_service.helpers.huggingface import _hf_repo_snapshot_dir
from backend_service.image_runtime import validate_local_diffusers_snapshot


def _is_image_repo(repo_id: str) -> bool:
    return any(
        str(variant.get("repo") or "") == repo_id
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
    )


def _sdcpp_gguf_files(repo_id: str) -> list[str]:
    """GGUF files of the stable-diffusion.cpp variants that live in ``repo_id``.

    Those repos hold bare GGUF files, not a diffusers pipeline, so "installed"
    means one of the files is on disk.
    """
    return [
        str(variant["ggufFile"])
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
        if variant.get("engine") == "sdcpp"
        and variant.get("ggufFile")
        and str(variant.get("repo") or "") == repo_id
        and str(variant.get("ggufRepo") or "") == repo_id
    ]


def _pinned_download_files(repo_id: str) -> list[str]:
    """Files a variant pins with ``downloadFiles`` when its repo holds several models.

    Iris-3B keeps the Turbo, base, depth and upscaler checkpoints in one 42 GB
    repo; the Turbo variant needs only its own folder.
    """
    files: list[str] = []
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if str(variant.get("repo") or "") == repo_id:
                files.extend(str(name) for name in variant.get("downloadFiles") or [])
    return list(dict.fromkeys(files))


# The folders mflux loads (its own allow-list in each model's weight
# definition). FLUX 3 ships ``modular_model_index.json`` instead of the
# ``model_index.json`` a diffusers snapshot has, so the diffusers check
# would never call it installed.
_MFLUX_COMPONENT_DIRS = ("transformer", "text_encoder", "vae")


def _mflux_only_repo(repo_id: str) -> bool:
    """True when every catalog variant of ``repo_id`` runs on mflux."""
    engines = [
        variant.get("engine")
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("repo") or "") == repo_id
    ]
    return bool(engines) and all(engine == "mflux" for engine in engines)


def _has_mflux_row(repo_id: str) -> bool:
    return any(
        variant.get("engine") == "mflux"
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("repo") or "") == repo_id
    )


def _mflux_snapshot_error(snapshot_dir: Path, repo_id: str) -> str | None:
    missing = [
        name for name in _MFLUX_COMPONENT_DIRS
        if not any((snapshot_dir / name).glob("*.safetensors"))
    ]
    if not missing:
        return None
    return f"The snapshot of {repo_id} has no weights in: {', '.join(missing)}. Retry the download."


def _nunchaku_variant_for_pack(pack_repo: str) -> dict[str, Any] | None:
    """The catalog variant whose Nunchaku pack is ``pack_repo``."""
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant.get("nunchakuRepo") and str(variant["nunchakuRepo"]) == pack_repo:
                return variant
    return None


def _nunchaku_base_repos() -> set[str]:
    """Base repos a Nunchaku variant fetches only in part (no bf16 transformer shards)."""
    return {
        str(variant.get("repo") or "")
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
        if variant.get("nunchakuRepo")
    }


def _transformer_weights_present(snapshot_dir: Path) -> bool:
    """True when the snapshot holds transformer weights, not just ``transformer/config.json``."""
    return any((snapshot_dir / "transformer").glob("*.safetensors"))


def _nunchaku_variant_ready(variant: dict[str, Any]) -> bool:
    """A Nunchaku variant runs when the base repo's pipeline parts and its pack are both on disk."""
    base_dir = _hf_repo_snapshot_dir(str(variant.get("repo") or ""))
    pack_dir = _hf_repo_snapshot_dir(str(variant.get("nunchakuRepo") or ""))
    if base_dir is None or pack_dir is None:
        return False
    if validate_local_diffusers_snapshot(Path(base_dir), str(variant["repo"])) is not None:
        return False
    # A dangling link or an empty file (an unfinished copy) is not a pack.
    return any(path.is_file() and path.stat().st_size > 0 for path in Path(pack_dir).glob("*.safetensors"))


def _nunchaku_download_validation_error(pack_repo: str) -> str | None:
    """After a Nunchaku download: both the pack and the base repo's parts must be on disk."""
    variant = _nunchaku_variant_for_pack(pack_repo)
    if variant is None or _nunchaku_variant_ready(variant):
        return None
    return (
        f"The download of {variant['name']} is incomplete: it needs {pack_repo} and the text encoders "
        f"and VAE of {variant['repo']}. Retry the download."
    )


def _nunchaku_generation_block(variant: dict[str, Any], download_state: str | None) -> str | None:
    """Why a Nunchaku variant cannot generate yet, or None when its pack and base parts are on disk.

    Without this check the run went to diffusers, failed on the missing files, and answered HTTP 200 with a
    placeholder picture and a raw Hugging Face error (RTX 4090, 2026-11-19); with the Hub reachable it could
    instead start fetching the whole 24 GB base repo.
    """
    if not variant.get("nunchakuRepo"):
        return None
    if _nunchaku_variant_ready(variant):
        # FU-478: the files are there but the Nunchaku package is not. The download has no standard transformer to
        # fall back to, so say so before any load starts (the load used to fail after it, with a note that said
        # "Falling back to the standard transformer" and then that there was nothing to fall back to).
        from backend_service.inference.accelerators import nunchaku_available

        if not nunchaku_available():
            return (
                f"{variant['name']} needs the Nunchaku package, which is not installed. Install Nunchaku from "
                "Performance boosters in Image Studio, or pick another FLUX.1 row."
            )
        return None
    pack_repo = str(variant["nunchakuRepo"])
    if download_state == "downloading":
        return f"{variant['name']} is still downloading. Generate again when the download has finished."
    if _hf_repo_snapshot_dir(pack_repo) is None and _hf_repo_snapshot_dir(str(variant.get("repo") or "")) is None:
        return (
            f"{variant['name']} is not downloaded. Download it on Image Models first: it needs the 4-bit pack "
            f"{pack_repo} and the text encoders and VAE of {variant['repo']}."
        )
    return _nunchaku_download_validation_error(pack_repo)


def _image_repo_runtime_ready(repo_id: str) -> bool:
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return False
    gguf_files = _sdcpp_gguf_files(repo_id)
    if gguf_files:
        return any((Path(snapshot_dir) / name).is_file() for name in gguf_files)
    pinned = _pinned_download_files(repo_id)
    if pinned:
        return all((Path(snapshot_dir) / name).is_file() for name in pinned)
    if _mflux_only_repo(repo_id):
        return _mflux_snapshot_error(Path(snapshot_dir), repo_id) is None
    if _has_mflux_row(repo_id) and _mflux_snapshot_error(Path(snapshot_dir), repo_id) is None:
        # FLUX 3 Klein has an mflux row and a diffusers row: an mflux download made before the
        # modular index was fetched still counts.
        return True
    if validate_local_diffusers_snapshot(snapshot_dir, repo_id) is not None:
        return False
    # A Nunchaku download leaves the base repo without its own transformer weights.
    return repo_id not in _nunchaku_base_repos() or _transformer_weights_present(Path(snapshot_dir))


def _image_variant_available_locally(variant: dict[str, Any], library: list[dict[str, Any]]) -> bool:
    if variant.get("nunchakuRepo"):
        return _nunchaku_variant_ready(variant)
    repo = str(variant.get("repo") or "")
    if variant.get("engine") == "sdcpp" and variant.get("ggufFile") and str(variant.get("ggufRepo") or "") == repo:
        snapshot_dir = _hf_repo_snapshot_dir(repo)
        return snapshot_dir is not None and (Path(snapshot_dir) / str(variant["ggufFile"])).is_file()
    if repo and _image_repo_runtime_ready(repo):
        return True

    candidates = {
        str(variant.get("repo") or "").lower(),
        str(variant.get("name") or "").lower(),
        str(variant.get("id") or "").lower(),
    }
    compact_candidates = {candidate.split("/")[-1] for candidate in candidates if candidate}
    for item in library:
        name = str(item.get("name") or "").lower()
        # Whole names only: a substring test made a Qwen/Qwen-Image-2.1
        # snapshot count as Qwen/Qwen-Image (and its Lightning variant)
        # installed after the user had deleted it.
        if not (name in candidates or name.split("/")[-1] in compact_candidates):
            continue
        item_path = Path(str(item.get("path") or "")).expanduser()
        needs_transformer = repo in _nunchaku_base_repos()
        for directory in _candidate_model_dirs(item_path):
            if validate_local_diffusers_snapshot(directory) is None and (
                not needs_transformer or _transformer_weights_present(directory)
            ):
                return True
    return False


def _image_download_validation_error(repo_id: str) -> str | None:
    if not _is_image_repo(repo_id):
        return None
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return (
            f"Download did not produce a local snapshot for {repo_id}. "
            "Retry the download and make sure the backend can access Hugging Face."
        )
    gguf_files = _sdcpp_gguf_files(repo_id)
    if gguf_files:
        if any((Path(snapshot_dir) / name).is_file() for name in gguf_files):
            return None
        return f"Download finished but none of {', '.join(gguf_files)} is in the snapshot of {repo_id}."
    pinned = _pinned_download_files(repo_id)
    if pinned:
        missing = [name for name in pinned if not (Path(snapshot_dir) / name).is_file()]
        if missing:
            return f"The snapshot of {repo_id} is missing {', '.join(missing)}. Retry the download."
        return None
    if _mflux_only_repo(repo_id):
        return _mflux_snapshot_error(Path(snapshot_dir), repo_id)
    return validate_local_diffusers_snapshot(snapshot_dir, repo_id)


def _friendly_image_download_error(repo_id: str, error: str) -> str:
    if not _is_image_repo(repo_id):
        return error
    lowered = error.lower()
    if (
        "cannot access gated repo" in lowered
        or "gated repo" in lowered
        or "authorized list" in lowered
        or ("access to model" in lowered and "restricted" in lowered)
    ):
        return (
            f"{repo_id} is gated on Hugging Face. Your account or token is not approved for this model yet. "
            f"Open https://huggingface.co/{repo_id}, request or accept access, add a read-enabled HF_TOKEN in Settings, then retry."
        )
    return error
