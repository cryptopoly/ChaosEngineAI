"""Local diffusers snapshot validator.

Before any pipeline ``from_pretrained`` we walk the snapshot tree on
disk to catch incomplete downloads up-front — saves users a confusing
"no file named config.json found in directory <snapshot_root>" error
that points at the snapshot root instead of the missing component.

Validates four classes of incompleteness:

1. Missing top-level ``model_index.json``.
2. Missing component sub-directories (every non-private key in
   ``model_index.json`` should map to a folder containing one of
   ``config.json`` / ``scheduler_config.json`` / ``tokenizer_config.json`` /
   ``preprocessor_config.json``).
3. Broken symlinks anywhere under the snapshot — these usually mean
   ``snapshot_download`` half-finished and the user retried with a
   stale ``.locks/`` lying around.
4. Missing weight shards — every ``*.index.json`` lists a ``weight_map``
   pointing at concrete shard files; if any shard isn't on disk the
   pipeline will partial-load and crash mid-forward with a cryptic
   layer-not-found error.

Returns a user-actionable error string when any check fails, or
``None`` when the snapshot looks complete enough to attempt loading.

Re-exported from ``backend_service.image_runtime`` so the existing
imports across ``sdcpp_image_runtime``, ``video_runtime``,
``helpers/images``, ``helpers/video``, and ``tests/test_sdcpp_image``
keep working.

Extracted from ``image_runtime.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import json
from pathlib import Path


def _snapshot_retry_guidance(repo: str | None = None) -> str:
    guidance = "Re-download the model and keep ChaosEngineAI open until the download completes."
    if repo:
        guidance += (
            f" If this model is gated, accept access on https://huggingface.co/{repo} if prompted, "
            "add HF_TOKEN in Settings, then retry."
        )
    return guidance


def _snapshot_visible_label(local_root: Path) -> str:
    try:
        visible_files = sorted(
            candidate.name
            for candidate in local_root.iterdir()
            if not candidate.name.startswith(".")
        )
    except OSError:
        visible_files = []
    return ", ".join(visible_files[:6]) if visible_files else "no files"


def validate_local_diffusers_snapshot(
    local_root: Path,
    repo: str | None = None,
    ignored_weight_index_dirs: set[str] | None = None,
) -> str | None:
    model_index_path = local_root / "model_index.json"
    if not model_index_path.exists():
        visible_label = _snapshot_visible_label(local_root)
        return (
            "The local snapshot is incomplete and cannot be opened as a diffusers pipeline "
            f"(missing model_index.json; found {visible_label}). {_snapshot_retry_guidance(repo)}"
        )

    # Verify each component listed in model_index.json actually has its folder
    # on disk with a recognisable config file. Diffusers will otherwise raise a
    # cryptic "no file named config.json found in directory <snapshot_root>"
    # error from inside ``from_pretrained`` that points at the snapshot root,
    # which is hard to action without knowing which subfolder is missing.
    # This typically happens when a download started before allow_patterns was
    # applied — HF queues the legacy root-level safetensors first and the user
    # tries to load before the per-component folders finish landing.
    try:
        pipeline_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            "The local snapshot's model_index.json could not be read "
            f"({exc}). {_snapshot_retry_guidance(repo)}"
        )

    missing_components: list[str] = []
    if isinstance(pipeline_index, dict):
        # Any of these names being present in a subfolder is enough to call it
        # a real component directory — diffusers picks the right one based on
        # the class type at load time.
        component_config_names = (
            "config.json",
            "scheduler_config.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
        )
        for component_name, descriptor in pipeline_index.items():
            if component_name.startswith("_"):
                continue  # ``_class_name`` / ``_diffusers_version`` metadata
            if not isinstance(descriptor, (list, tuple)) or len(descriptor) < 2:
                continue
            # Pipelines list ``[null, null]`` for optional components that the
            # checkpoint deliberately omits (e.g. safety_checker on community
            # models). Skip those — they aren't expected on disk.
            if descriptor[0] is None or descriptor[1] is None:
                continue
            component_dir = local_root / component_name
            if not component_dir.is_dir():
                missing_components.append(component_name)
                continue
            if not any((component_dir / name).exists() for name in component_config_names):
                missing_components.append(component_name)

    if missing_components:
        label = ", ".join(missing_components[:4])
        if len(missing_components) > 4:
            label += f" (+{len(missing_components) - 4} more)"
        return (
            "The local snapshot is incomplete and cannot be opened as a diffusers pipeline "
            f"(missing components: {label}). {_snapshot_retry_guidance(repo)}"
        )

    broken_links: list[str] = []
    weight_index_paths: list[Path] = []
    try:
        for candidate in local_root.rglob("*"):
            if candidate.is_dir():
                continue
            if candidate.is_symlink() and not candidate.exists():
                broken_links.append(candidate.relative_to(local_root).as_posix())
            if candidate.name.endswith(".index.json"):
                rel_parts = candidate.relative_to(local_root).parts
                if rel_parts and ignored_weight_index_dirs and rel_parts[0] in ignored_weight_index_dirs:
                    continue
                weight_index_paths.append(candidate)
    except OSError as exc:
        return (
            "The local snapshot could not be inspected before loading "
            f"({exc}). {_snapshot_retry_guidance(repo)}"
        )

    if broken_links:
        label = ", ".join(broken_links[:3])
        if len(broken_links) > 3:
            label += f" (+{len(broken_links) - 3} more)"
        return (
            "The local snapshot is incomplete and cannot be opened as a diffusers pipeline "
            f"(missing local files: {label}). {_snapshot_retry_guidance(repo)}"
        )

    missing_shards: list[str] = []
    for index_path in weight_index_paths:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rel_path = index_path.relative_to(local_root).as_posix()
            return (
                "The local snapshot is incomplete and cannot be opened as a diffusers pipeline "
                f"(could not read {rel_path}: {exc}). {_snapshot_retry_guidance(repo)}"
            )
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            continue
        shard_names = sorted({value for value in weight_map.values() if isinstance(value, str) and value})
        for shard_name in shard_names:
            shard_path = index_path.parent / shard_name
            if shard_path.exists():
                continue
            missing_shards.append(shard_path.relative_to(local_root).as_posix())

    if missing_shards:
        label = ", ".join(missing_shards[:3])
        if len(missing_shards) > 3:
            label += f" (+{len(missing_shards) - 3} more)"
        return (
            "The local snapshot is incomplete and cannot be opened as a diffusers pipeline "
            f"(missing weight shards: {label}). {_snapshot_retry_guidance(repo)}"
        )

    return None
