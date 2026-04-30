"""Video model helpers: variant lookup, install detection, payload shaping,
output CRUD.

Mirrors ``helpers/images.py`` so the routes for ``/api/video/*`` can drop in
alongside the image routes without a new mental model.
"""

from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from backend_service.catalog import VIDEO_MODEL_FAMILIES
from backend_service.helpers.formatting import _bytes_to_gb
from backend_service.helpers.huggingface import _format_release_label, _hf_repo_snapshot_dir
from backend_service.helpers.images import _image_repo_live_metadata, _snapshot_on_disk_bytes
from backend_service.image_runtime import validate_local_diffusers_snapshot


def _video_model_payloads(library: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the catalog families enriched with per-variant availability
    plus live Hugging Face metadata (downloads, likes, lastModified).

    Live fetches run through ``_image_repo_live_metadata`` — despite the
    name it's a generic HF-repo fetcher, same one the image catalog uses,
    with a 6-hour in-process cache. Called in parallel across repos with
    an 8-second wall-clock budget; repos whose fetch times out fall back
    to the curated defaults on the variant dict.
    """
    repo_metadata: dict[str, dict[str, Any]] = {}
    repos = sorted({
        str(variant.get("repo") or "")
        for family in VIDEO_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("repo") or "")
    })
    if repos:
        with ThreadPoolExecutor(max_workers=min(4, len(repos))) as executor:
            future_map = {
                executor.submit(_image_repo_live_metadata, repo): repo
                for repo in repos
            }
            try:
                for future in as_completed(future_map, timeout=8):
                    repo = future_map[future]
                    try:
                        repo_metadata[repo] = future.result(timeout=2)
                    except Exception:
                        # Individual fetch failures fall back to curated
                        # defaults — the catalog row still renders, just
                        # without live downloads/likes.
                        repo_metadata[repo] = {}
            except FuturesTimeout:
                pass

    families: list[dict[str, Any]] = []
    for family in VIDEO_MODEL_FAMILIES:
        variants: list[dict[str, Any]] = []
        for variant in family["variants"]:
            enriched = dict(variant)
            repo = str(enriched.get("repo") or "")
            live_metadata = repo_metadata.get(repo, {}) if repo else {}
            # Merge live metadata first so curated fields (releaseDate,
            # familyName) still win when both exist.
            enriched = {**enriched, **live_metadata}
            validation_error = _video_variant_validation_error(enriched)
            local_data_repos = _video_variant_local_data_repos(enriched)
            enriched["availableLocally"] = validation_error is None
            enriched["hasLocalData"] = (
                enriched["availableLocally"]
                or bool(local_data_repos)
            )
            primary_local_repo = (
                repo
                if repo and repo in local_data_repos
                else local_data_repos[0]
                if local_data_repos
                else None
            )
            enriched["localDataRepos"] = local_data_repos
            enriched["primaryLocalRepo"] = primary_local_repo
            enriched["localStatusReason"] = (
                _video_variant_local_status_reason(enriched, validation_error)
                if enriched["hasLocalData"] and validation_error
                else None
            )
            enriched["familyName"] = family["name"]
            release_date = str(variant.get("releaseDate") or "").strip() or None
            enriched["releaseDate"] = release_date
            # Prefer the curated releaseLabel when the catalog specifies a
            # releaseDate; fall back to HF's createdAt-derived label.
            enriched["releaseLabel"] = (
                _format_release_label(release_date)
                or live_metadata.get("releaseLabel")
            )
            # Absolute path to the HF snapshot, used by the Reveal File button.
            # Only populated when there is actually something on disk so the
            # UI can reliably hide the button otherwise.
            snapshot_dir = (
                _hf_repo_snapshot_dir(primary_local_repo)
                if (enriched["hasLocalData"] and primary_local_repo)
                else None
            )
            enriched["localPath"] = str(snapshot_dir) if snapshot_dir else None
            on_disk_bytes = _snapshot_on_disk_bytes(snapshot_dir)
            enriched["onDiskBytes"] = on_disk_bytes
            enriched["onDiskGb"] = _bytes_to_gb(on_disk_bytes) if on_disk_bytes else None
            variants.append(enriched)
        payload = dict(family)
        payload["variants"] = variants
        families.append(payload)
    return families


def _find_video_variant(model_id: str) -> dict[str, Any] | None:
    for family in VIDEO_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["id"] == model_id:
                return variant
    return None


def _find_video_variant_by_repo(repo: str) -> dict[str, Any] | None:
    for family in VIDEO_MODEL_FAMILIES:
        for variant in family["variants"]:
            if repo in _video_variant_download_repos(variant):
                return variant
    return None


def _video_variant_download_repos(variant: dict[str, Any]) -> list[str]:
    repos: list[str] = []
    for key in ("repo", "ggufRepo", "textEncoderRepo"):
        repo = str(variant.get(key) or "").strip()
        if repo and repo not in repos:
            repos.append(repo)
    return repos


def _is_video_repo(repo_id: str) -> bool:
    return any(
        str(variant.get("repo") or "") == repo_id
        for family in VIDEO_MODEL_FAMILIES
        for variant in family["variants"]
    )


def _is_video_download_repo(repo_id: str) -> bool:
    return repo_id in _video_download_repo_ids()


def _video_repo_runtime_ready(repo_id: str) -> bool:
    """True if the local snapshot is complete enough to load.

    Routes the validator by engine: mlx-video repos ship component folders
    without ``model_index.json``, so the diffusers-shape check would always
    falsely fail for them. Diffusers repos still go through
    ``validate_local_diffusers_snapshot``.
    """
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return False
    if _is_mlx_video_routed_repo(repo_id):
        return _validate_mlx_video_snapshot(snapshot_dir, repo_id) is None
    return validate_local_diffusers_snapshot(snapshot_dir, repo_id) is None


def _video_repo_has_any_local_data(repo_id: str) -> bool:
    """True if we have a partial or complete snapshot on disk.

    Distinct from ``_video_repo_runtime_ready`` — this is the softer signal used
    to tell the UI "something downloaded for this repo" even if it's incomplete.
    """
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return False
    root = Path(snapshot_dir)
    if not root.exists():
        return False
    try:
        return any(
            candidate.is_file() or candidate.is_symlink()
            for candidate in root.iterdir()
            if not candidate.name.startswith(".")
        )
    except OSError:
        return False


def _video_variant_local_data_repos(variant: dict[str, Any]) -> list[str]:
    return [
        repo
        for repo in _video_variant_download_repos(variant)
        if _video_repo_has_any_local_data(repo)
    ]


def _video_variant_available_locally(variant: dict[str, Any]) -> bool:
    return _video_variant_validation_error(variant) is None


def _video_variant_has_any_local_data(variant: dict[str, Any]) -> bool:
    return bool(_video_variant_local_data_repos(variant))


def _video_variant_validation_error(variant: dict[str, Any]) -> str | None:
    repo = str(variant.get("repo") or "")
    if not repo:
        return "Video model variant is missing its base repo id."
    repo_error = _video_download_validation_error(repo)
    if repo_error:
        return repo_error
    text_error = _video_variant_mlx_text_components_validation_error(variant)
    if text_error:
        return text_error
    return _video_variant_gguf_validation_error(variant)


def _video_variant_mlx_text_components_validation_error(variant: dict[str, Any]) -> str | None:
    repo = str(variant.get("repo") or "")
    if not _is_mlx_video_routed_repo(repo):
        return None
    snapshot_dir = _hf_repo_snapshot_dir(repo)
    if snapshot_dir is None:
        return None
    missing = _missing_mlx_text_components(Path(snapshot_dir))
    if not missing:
        return None

    text_encoder_repo = str(variant.get("textEncoderRepo") or "").strip()
    if text_encoder_repo and text_encoder_repo != repo:
        text_snapshot = _hf_repo_snapshot_dir(text_encoder_repo)
        if text_snapshot is not None and not _missing_mlx_text_components(Path(text_snapshot)):
            return None
        return (
            "The local snapshot is missing shared mlx-video text components: "
            f"{', '.join(missing)}. Download the shared text encoder "
            f"({text_encoder_repo}) and retry."
        )

    return (
        "The local snapshot is incomplete. Missing mlx-video components: "
        f"{', '.join(missing)}. Re-download the model and keep ChaosEngineAI "
        "open until the download completes."
    )


def _video_variant_missing_text_encoder_repo(variant: dict[str, Any]) -> str | None:
    error = _video_variant_mlx_text_components_validation_error(variant)
    text_encoder_repo = str(variant.get("textEncoderRepo") or "").strip()
    if error and text_encoder_repo:
        return text_encoder_repo
    return None


def _video_variant_local_status_reason(
    variant: dict[str, Any],
    validation_error: str | None,
) -> str | None:
    if not validation_error:
        return None
    gguf_file = str(variant.get("ggufFile") or "").strip()
    gguf_repo = str(variant.get("ggufRepo") or "").strip()
    if gguf_file and gguf_repo and "GGUF transformer file is missing" in validation_error:
        return f"Base model installed; missing GGUF transformer: {gguf_repo}/{gguf_file}."

    prefix = "The local snapshot is incomplete. Missing mlx-video components: "
    if validation_error.startswith(prefix):
        missing = validation_error[len(prefix):].split(". Re-download", 1)[0]
        return f"Missing MLX components: {missing}."

    shared_prefix = "The local snapshot is missing shared mlx-video text components: "
    if validation_error.startswith(shared_prefix):
        missing = validation_error[len(shared_prefix):].split(". Download", 1)[0]
        text_encoder_repo = str(variant.get("textEncoderRepo") or "").strip()
        source = f" from {text_encoder_repo}" if text_encoder_repo else ""
        return f"Missing shared MLX text components{source}: {missing}."

    if validation_error.startswith("The selected GGUF transformer resolved to a cache path"):
        return f"GGUF transformer cache path is invalid: {gguf_repo}/{gguf_file}."

    return validation_error


def _video_variant_gguf_validation_error(variant: dict[str, Any]) -> str | None:
    gguf_file = str(variant.get("ggufFile") or "").strip()
    if not gguf_file:
        return None
    gguf_repo = str(variant.get("ggufRepo") or "").strip()
    if not gguf_repo:
        return (
            f"{variant.get('name') or 'This GGUF video variant'} is missing "
            "its GGUF repository metadata."
        )
    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        local_path = hf_hub_download(
            repo_id=gguf_repo,
            filename=gguf_file,
            local_files_only=True,
        )
    except Exception:
        return (
            "The base diffusers snapshot is installed, but the selected GGUF "
            f"transformer file is missing: {gguf_repo}/{gguf_file}. Download "
            "the GGUF variant before generating so the app does not fall back "
            "to the full BF16 transformer."
        )
    if not Path(local_path).exists():
        return (
            "The selected GGUF transformer resolved to a cache path that does "
            f"not exist: {gguf_repo}/{gguf_file}. Retry the GGUF download."
        )
    return None


def _video_download_repo_ids() -> set[str]:
    repos = {
        str(variant.get("repo") or "")
        for family in VIDEO_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("repo") or "")
    }
    repos.update(
        str(variant.get("ggufRepo") or "")
        for family in VIDEO_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("ggufRepo") or "")
    )
    repos.update(
        str(variant.get("textEncoderRepo") or "")
        for family in VIDEO_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("textEncoderRepo") or "")
    )
    return repos


# Diffusers pipelines only need the standard per-component folders
# (scheduler/, text_encoder/, tokenizer/, transformer/ or unet/, vae/)
# plus ``model_index.json`` at the root. Video repos frequently ship
# historical checkpoints (``ltx-video-0.9.safetensors`` and friends) as
# siblings — without an allowlist ``snapshot_download`` pulls every one
# of them, which can inflate a 2 GB diffusers pipeline into a 200 GB
# download. Keep this list conservative so future component folders still
# come through, but block the legacy standalone safetensors.
_VIDEO_DIFFUSERS_ALLOW_PATTERNS: list[str] = [
    "model_index.json",
    "scheduler/**",
    "text_encoder/**",
    "text_encoder_2/**",
    "text_encoder_3/**",
    "tokenizer/**",
    "tokenizer_2/**",
    "tokenizer_3/**",
    "transformer/**",
    "transformer_2/**",
    "unet/**",
    "vae/**",
    "feature_extractor/**",
    "image_encoder/**",
    "safety_checker/**",
    "*.md",
    "LICENSE*",
]

_VIDEO_MLX_ALLOW_PATTERNS: list[str] = [
    "text_encoder/**",
    "tokenizer/**",
    "text_projections/**",
    "audio_vae/**",
    "transformer/**",
    "vae/**",
    "vocoder/**",
    "*spatial-upscaler*.safetensors",
    "*.md",
    "LICENSE*",
]

_VIDEO_MLX_TEXT_ENCODER_ALLOW_PATTERNS: list[str] = [
    "text_encoder/**",
    "tokenizer/**",
    "*.md",
    "LICENSE*",
]


def _video_repo_allow_patterns(repo_id: str) -> list[str] | None:
    """Patterns to pass to ``snapshot_download`` for a video repo.

    Returns ``None`` for non-video repos so the caller can pass the value
    through unconditionally without special-casing. For video repos the
    allowlist keeps the download scoped to the diffusers pipeline layout
    — see the comment on ``_VIDEO_DIFFUSERS_ALLOW_PATTERNS`` for why this
    matters.
    """
    if not _is_video_repo(repo_id):
        return None
    if _is_mlx_video_routed_repo(repo_id):
        return list(_VIDEO_MLX_ALLOW_PATTERNS)
    return list(_VIDEO_DIFFUSERS_ALLOW_PATTERNS)


def _video_download_validation_error(repo_id: str) -> str | None:
    if not _is_video_repo(repo_id):
        return None
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return (
            f"Download did not produce a local snapshot for {repo_id}. "
            "Retry the download and make sure the backend can access Hugging Face."
        )
    # mlx-video routed repos (e.g. ``prince-canuma/LTX-2-*``) ship MLX
    # layouts without ``model_index.json``. Don't apply the diffusers-shape
    # validator to them; check for the expected MLX component folders instead.
    if _is_mlx_video_routed_repo(repo_id):
        return _validate_mlx_video_snapshot(snapshot_dir, repo_id)
    return validate_local_diffusers_snapshot(snapshot_dir, repo_id)


def _is_mlx_video_routed_repo(repo_id: str) -> bool:
    """True iff this repo is meant to load through mlx-video on Apple Silicon.

    Imports ``mlx_video_runtime`` lazily so the validator path doesn't drag
    that module's torch warmup costs into every video catalog refresh.
    """
    try:
        from backend_service.mlx_video_runtime import _is_mlx_video_repo
    except Exception:
        return False
    return _is_mlx_video_repo(repo_id)


# Component folders any mlx-video LTX-2 snapshot must carry. Subset of the
# diffusers layout — no model_index.json. Lifted from the ``prince-canuma/
# LTX-2-distilled`` repo tree as the canonical shape; bump as new mlx-video
# families with different layouts come online.
_MLX_VIDEO_LTX2_REQUIRED_COMPONENTS: tuple[str, ...] = (
    "text_encoder",
    "tokenizer",
    "text_projections",
    "transformer",
    "vae",
)

_MLX_VIDEO_LTX23_REQUIRED_COMPONENTS: tuple[str, ...] = (
    "audio_vae",
    "text_projections",
    "transformer",
    "vae",
    "vocoder",
)


def _mlx_video_required_components(repo_id: str | None = None) -> tuple[str, ...]:
    repo_key = str(repo_id or "").lower()
    if "ltx-2.3" in repo_key:
        return _MLX_VIDEO_LTX23_REQUIRED_COMPONENTS
    return _MLX_VIDEO_LTX2_REQUIRED_COMPONENTS


def _missing_mlx_text_components(root: Path) -> list[str]:
    missing: list[str] = []
    checks = {
        "text_encoder": (
            root / "text_encoder" / "config.json",
            root / "text_encoder" / "model.safetensors.index.json",
        ),
        "tokenizer": (
            root / "tokenizer" / "tokenizer.json",
            root / "tokenizer" / "tokenizer.model",
        ),
    }
    for component, required_paths in checks.items():
        component_dir = root / component
        if not component_dir.is_dir():
            missing.append(component)
            continue
        if not all(path.exists() for path in required_paths):
            missing.append(component)
    return missing


def _validate_mlx_video_snapshot(snapshot_dir: str, repo_id: str | None = None) -> str | None:
    """Return ``None`` if the snapshot has the required MLX component folders.

    Mirrors the contract of ``validate_local_diffusers_snapshot`` so the
    callers can swap one for the other without restructuring the result
    handling. Each missing folder is named explicitly so the user sees
    which file an interrupted download stopped on.
    """
    root = Path(snapshot_dir)
    if not root.exists():
        return (
            f"Local snapshot directory does not exist at {root}. "
            "Re-download the model."
        )
    missing: list[str] = []
    for component in _mlx_video_required_components(repo_id):
        component_dir = root / component
        if not component_dir.is_dir():
            missing.append(component)
            continue
        # Empty component dirs indicate a half-completed download — count
        # them as missing so the retry CTA fires.
        try:
            if not any(component_dir.iterdir()):
                missing.append(f"{component} (empty)")
        except OSError:
            missing.append(component)
    if missing:
        return (
            "The local snapshot is incomplete. Missing mlx-video components: "
            f"{', '.join(missing)}. Re-download the model and keep ChaosEngineAI "
            "open until the download completes."
        )
    return None


# ---- Video output CRUD ----
#
# Video artifacts differ from image artifacts in one important way: an mp4 is
# the real deliverable and there's no cheap "preview" we can embed inline. The
# frontend loads the file directly via a dedicated ``/file`` endpoint rather
# than getting a base64 data URL in the list payload.


def _video_output_directory(video_outputs_dir: Path, created_at: str | None = None) -> Path:
    day_label = (created_at or datetime.utcnow().isoformat())[:10]
    output_dir = video_outputs_dir / day_label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _hydrate_video_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or "")
    model_name = str(payload.get("modelName") or payload.get("modelId") or "Video model")
    return {
        "artifactId": str(payload.get("artifactId") or ""),
        "modelId": str(payload.get("modelId") or ""),
        "modelName": model_name,
        "prompt": prompt,
        "negativePrompt": str(payload.get("negativePrompt") or ""),
        "width": int(payload.get("width") or 768),
        "height": int(payload.get("height") or 512),
        "numFrames": int(payload.get("numFrames") or 0),
        "fps": int(payload.get("fps") or 24),
        "steps": int(payload.get("steps") or 0),
        "guidance": float(payload.get("guidance") or 0.0),
        "seed": int(payload.get("seed") or 0),
        "createdAt": str(
            payload.get("createdAt") or datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        ),
        "durationSeconds": float(payload.get("durationSeconds") or 0.0),
        "clipDurationSeconds": float(payload.get("clipDurationSeconds") or 0.0),
        "videoPath": str(payload.get("videoPath") or "") or None,
        "metadataPath": str(payload.get("metadataPath") or "") or None,
        "videoMimeType": str(payload.get("videoMimeType") or "video/mp4"),
        "videoExtension": str(payload.get("videoExtension") or "mp4"),
        "runtimeLabel": str(payload.get("runtimeLabel") or ""),
        "runtimeNote": str(payload.get("runtimeNote") or "") or None,
    }


def _save_video_artifact(artifact: dict[str, Any], video_outputs_dir: Path) -> dict[str, Any]:
    created_at = str(
        artifact.get("createdAt") or datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    )
    output_dir = _video_output_directory(video_outputs_dir, created_at)
    artifact_id = str(artifact["artifactId"])
    extension = str(artifact.get("videoExtension") or "mp4").lstrip(".")
    video_path = output_dir / f"{artifact_id}.{extension}"
    metadata_path = output_dir / f"{artifact_id}.json"

    video_bytes = artifact.get("videoBytes")
    if isinstance(video_bytes, str):
        video_bytes = base64.b64decode(video_bytes.encode("ascii"))
    if isinstance(video_bytes, (bytes, bytearray)):
        video_path.write_bytes(bytes(video_bytes))
    else:
        raise ValueError(
            "Cannot persist video artifact: no raw bytes supplied. "
            "Pass `videoBytes` as bytes from the generation pipeline."
        )

    persisted = {
        **artifact,
        "videoPath": str(video_path),
        "metadataPath": str(metadata_path),
    }
    metadata_payload = {
        key: value
        for key, value in persisted.items()
        if key not in {"videoBytes", "videoMimeType", "videoExtension"}
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return _hydrate_video_artifact(persisted)


def _load_video_outputs(video_outputs_dir: Path) -> list[dict[str, Any]]:
    if not video_outputs_dir.exists():
        return []
    outputs: list[dict[str, Any]] = []
    for metadata_path in video_outputs_dir.rglob("*.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        outputs.append(_hydrate_video_artifact({**payload, "metadataPath": str(metadata_path)}))
    outputs.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    return outputs


def _find_video_output(artifact_id: str, video_outputs_dir: Path) -> dict[str, Any] | None:
    for output in _load_video_outputs(video_outputs_dir):
        if output.get("artifactId") == artifact_id:
            return output
    return None


def _delete_video_output(artifact_id: str, video_outputs_dir: Path) -> bool:
    if not video_outputs_dir.exists():
        return False
    found = False
    for metadata_path in video_outputs_dir.rglob(f"{artifact_id}.json"):
        found = True
        video_path = metadata_path.with_suffix(".mp4")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("videoPath"):
                video_path = Path(str(payload["videoPath"]))
        except (OSError, json.JSONDecodeError):
            pass
        try:
            metadata_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            pass
    return found
