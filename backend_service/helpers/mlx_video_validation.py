"""mlx-video snapshot validation — required-folder probe + routing predicate.

mlx-video LTX-2 / LTX-2.3 repos ship a component-folder layout without
``model_index.json``, so ``validate_local_diffusers_snapshot`` always
falsely flags them as incomplete. These helpers know what each mlx-video
family requires on disk and produce the same shape of error string as
the diffusers validator.

Extracted from ``backend_service/helpers/video.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.video`` so existing imports
keep working.
"""

from __future__ import annotations

from pathlib import Path


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
