"""Model discovery: scanning directories, detecting formats and quantization."""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from backend_service.helpers.formatting import (
    _bytes_to_gb,
    _detect_model_max_context,
    _main_gguf_file,
)
from backend_service.helpers.model_classifier import (
    _DRAFT_MODEL_KEYWORDS,
    _IMAGE_MODEL_KEYWORDS,
    _VIDEO_MODEL_KEYWORDS,
    _looks_like_draft_model,
    _looks_like_image_model,
    _looks_like_video_model,
)
from backend_service.helpers.quantization import (
    _UNSUPPORTED_MLX_QUANT_ALGOS,
    _dtype_quantization_label,
    _mlx_quantization_bits,
    _quantization_algo_label,
    _quantization_label_from_text,
    _unsupported_mlx_quantization_reason,
)
from backend_service.helpers.settings import _normalize_slug
from backend_service.helpers.snapshot_integrity import (
    _SHARDED_WEIGHT_RE,
    _incomplete_gguf_directory_reason,
    _incomplete_sharded_weight_reason,
    _list_weight_files,
)


def _path_size_bytes(path: Path, *, seen: set[tuple[int, int]] | None = None) -> int:
    visited = seen if seen is not None else set()
    zero_inode_dirs: set[str] = set()
    zero_inode_files_by_size: dict[int, list[str]] = {}

    def _mark_seen(candidate: str | Path, stat_result: os.stat_result, *, is_dir: bool) -> bool:
        identity = (stat_result.st_dev, stat_result.st_ino)
        if identity[1]:
            if identity in visited:
                return False
            visited.add(identity)
            return True

        candidate_str = str(candidate)
        if is_dir:
            dir_key = os.path.normcase(os.path.realpath(candidate_str))
            if dir_key in zero_inode_dirs:
                return False
            zero_inode_dirs.add(dir_key)
            return True

        same_size_files = zero_inode_files_by_size.setdefault(int(stat_result.st_size), [])
        for existing in same_size_files:
            try:
                if os.path.samefile(candidate_str, existing):
                    return False
            except OSError:
                continue
        same_size_files.append(candidate_str)
        return True

    try:
        root_stat = path.stat()
    except OSError:
        return 0

    try:
        root_is_dir = path.is_dir()
    except OSError:
        root_is_dir = False
    if not _mark_seen(path, root_stat, is_dir=root_is_dir):
        return 0

    if not root_is_dir:
        return int(root_stat.st_size)

    total = 0
    stack: list[str] = [str(path)]
    while stack:
        current = stack.pop()
        try:
            iterator = os.scandir(current)
        except OSError:
            continue
        with iterator as entries:
            for entry in entries:
                try:
                    entry_stat = entry.stat(follow_symlinks=True)
                except OSError:
                    continue
                try:
                    is_dir = entry.is_dir(follow_symlinks=True)
                except OSError:
                    is_dir = False
                if not _mark_seen(entry.path, entry_stat, is_dir=is_dir):
                    continue
                if is_dir:
                    stack.append(entry.path)
                else:
                    total += int(entry_stat.st_size)
    return total


def _du_size_gb(path: Path) -> float:
    return _bytes_to_gb(_path_size_bytes(path))


def _relative_depth(path: Path, root: Path) -> int:
    try:
        return len(path.relative_to(root).parts)
    except ValueError:
        return 0


def _candidate_model_dirs(path: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(candidate: Path) -> None:
        try:
            if not candidate.is_dir():
                return
        except OSError:
            return
        key = str(candidate)
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    if path.is_dir():
        _add(path)
        snapshots = path / "snapshots"
        try:
            if snapshots.is_dir():
                for snap in sorted(snapshots.iterdir()):
                    _add(snap)
        except OSError:
            pass
    else:
        _add(path.parent)
    return candidates


def _read_model_config(path: Path) -> dict[str, Any] | None:
    for directory in _candidate_model_dirs(path):
        candidate = directory / "config.json"
        try:
            if candidate.exists():
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            continue
    return None


def _model_has_files(path: Path, pattern: str) -> bool:
    try:
        return any(path.rglob(pattern))
    except OSError:
        return False




def _detect_storage_format(path: Path, *, name_hint: str = "") -> str:
    lowered_hint = f"{name_hint} {path}".lower()
    if path.is_file() and path.suffix.lower() == ".gguf":
        return "GGUF"
    if _model_has_files(path, "*.gguf"):
        return "GGUF"

    config = _read_model_config(path)
    has_safetensors = _model_has_files(path, "*.safetensors")
    has_pytorch_bin = _model_has_files(path, "pytorch_model*.bin")
    looks_like_mlx = "mlx-community" in lowered_hint or bool(re.search(r"(^|[^a-z])mlx([^a-z]|$)", lowered_hint))
    unsupported_reason = _unsupported_mlx_quantization_reason(config)

    if unsupported_reason and (config is not None or has_safetensors or has_pytorch_bin):
        return "Transformers"
    if _mlx_quantization_bits(config) is not None and (config is not None or has_safetensors or has_pytorch_bin):
        return "MLX"
    if looks_like_mlx and (config is not None or has_safetensors or has_pytorch_bin):
        return "MLX"
    if has_safetensors or has_pytorch_bin:
        return "Transformers"
    if config is not None:
        return "MLX" if looks_like_mlx else "Transformers"
    return "unknown"


def _detect_model_quantization(path: Path, fmt: str, *, name_hint: str = "") -> str | None:
    text_hint = f"{name_hint} {path}"
    fmt_upper = (fmt or "").upper()
    if fmt_upper == "GGUF":
        main_file = _main_gguf_file(path if path.is_dir() else path.parent)
        if main_file is not None:
            label = _quantization_label_from_text(main_file.name)
            if label:
                return label
        return _quantization_label_from_text(text_hint)

    config = _read_model_config(path)
    quant_algo = _quantization_algo_label(config)
    if quant_algo:
        return quant_algo
    bits = _mlx_quantization_bits(config)
    if bits is not None:
        return f"{bits}-bit"
    dtype_label = _dtype_quantization_label(config)
    if dtype_label:
        return dtype_label
    return _quantization_label_from_text(text_hint)


def _detect_directory_model(path: Path) -> tuple[str, str, str] | None:
    source_kind = "HF cache" if path.name.startswith("models--") else "Directory"
    name = path.name.replace("models--", "").replace("--", "/") if source_kind == "HF cache" else path.name
    if source_kind == "HF cache":
        detected_format = _detect_storage_format(path, name_hint=name)
        return (name, detected_format, source_kind) if detected_format != "unknown" else (name, "Transformers", source_kind)
    if any(path.glob("*.gguf")) or any(path.glob("*.gguf.part")):
        return (name, "GGUF", source_kind)
    if (path / "config.json").exists() or (path / "tokenizer.json").exists():
        return (name, _detect_storage_format(path, name_hint=name), source_kind)
    return None


def _detect_broken_library_item(child: Path, file_format: str, source_kind: str | None = None) -> tuple[bool, str | None]:
    """Return (broken, reason) for a discovered library item.

    Only directory-style entries can be broken; individual .gguf/.safetensors
    files are assumed healthy if they exist on disk.
    """
    try:
        if not child.is_dir():
            return (False, None)
    except OSError:
        return (False, None)

    fmt = (file_format or "").lower()
    source = (source_kind or "").lower()
    try:
        config = _read_model_config(child)
        unsupported_reason = _unsupported_mlx_quantization_reason(config)
        if unsupported_reason:
            return (True, unsupported_reason)
        # HF cache entries are polymorphic: the same layout
        # (models--owner--name/snapshots/<rev>/...) can hold GGUF-only
        # mirrors, Transformers safetensors, MLX, or any combination.
        # Only flag broken if NONE of the expected weight formats are
        # present anywhere inside. Looking at file extensions instead of
        # the format label avoids the false-positive where an HF-cache
        # Transformers repo gets mislabelled as "GGUF broken" just
        # because the format slot says "HF cache".
        if source == "hf cache":
            try:
                if any((child / "blobs").glob("*.incomplete")):
                    return (True, "Hugging Face download is incomplete: partial blob files are still present.")
            except OSError:
                pass
            for candidate in _candidate_model_dirs(child):
                shard_reason = _incomplete_sharded_weight_reason(candidate)
                if shard_reason:
                    return (True, shard_reason)
            has_gguf = any(child.rglob("*.gguf"))
            has_safetensors = any(child.rglob("*.safetensors"))
            has_pytorch_bin = any(child.rglob("pytorch_model*.bin"))
            if not (has_gguf or has_safetensors or has_pytorch_bin):
                return (True, "No .gguf, .safetensors, or pytorch weights found in HF cache entry")
            return (False, None)
        if fmt == "gguf" or "gguf" in fmt:
            gguf_reason = _incomplete_gguf_directory_reason(child)
            if gguf_reason:
                return (True, gguf_reason)
            if not any(child.rglob("*.gguf")):
                return (True, "No .gguf weights present")
            return (False, None)
        if fmt == "mlx":
            shard_reason = _incomplete_sharded_weight_reason(child)
            if shard_reason:
                return (True, shard_reason)
            if not any(child.glob("*.safetensors")) and not (child / "model.safetensors").exists():
                return (True, "MLX directory missing model.safetensors")
            return (False, None)
        if fmt == "transformers":
            shard_reason = _incomplete_sharded_weight_reason(child)
            if shard_reason:
                return (True, shard_reason)
            has_safetensors = any(child.glob("*.safetensors"))
            has_pytorch_bin = any(child.glob("pytorch_model*.bin"))
            if not has_safetensors and not has_pytorch_bin:
                return (True, "Transformers directory has no safetensors or pytorch weights")
            return (False, None)
    except OSError:
        return (False, None)
    return (False, None)


def _iter_discovered_models(root: Path, *, max_depth: int = 8) -> list[tuple[Path, str, str, str]]:
    discovered: list[tuple[Path, str, str, str]] = []
    # `.locks` is the Hugging Face hub lockfile directory. It mirrors the
    # `models--owner--name/` naming scheme, which would otherwise cause
    # the detector to produce phantom "broken" HF cache duplicates (lock
    # dirs contain no weights).
    skip_names = {"blobs", "refs", ".locks", ".cache", ".git", "__pycache__", ".venv", "node_modules"}

    for current_root, dirnames, filenames in os.walk(root):
        current = Path(current_root)
        depth = _relative_depth(current, root)
        if depth > max_depth:
            dirnames[:] = []
            continue

        # Prune by explicit skip list AND any dotfile/dot-directory so we
        # never wander into HF's `.locks`, `.cache`, etc.
        dirnames[:] = [
            name for name in dirnames
            if name not in skip_names and not name.startswith(".")
        ]

        if current != root:
            detected = _detect_directory_model(current)
            if detected is not None:
                discovered.append((current, detected[0], detected[1], detected[2]))
                dirnames[:] = []
                continue

        for filename in filenames:
            child = current / filename
            suffix = child.suffix.lower()
            if suffix not in {".gguf", ".safetensors"}:
                continue
            if suffix == ".safetensors" and (current / "config.json").exists():
                continue
            discovered.append((child, child.stem, suffix.replace(".", "").upper(), "File"))

    return discovered


def _discover_local_models(model_directories: list[dict[str, Any]], limit: int = 500) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for directory in model_directories:
        if not directory.get("enabled", True):
            continue
        raw_path = str(directory.get("path") or "").strip()
        if not raw_path:
            continue

        root = Path(os.path.expanduser(raw_path)).expanduser()
        if not root.exists():
            continue

        directory_label = str(directory.get("label") or root.name or "Model directory")
        directory_id = str(directory.get("id") or _normalize_slug(directory_label, "directory"))
        try:
            discovered = _iter_discovered_models(root)
        except OSError:
            continue

        for child, name, file_format, source_kind in discovered:
            if len(items) >= limit:
                return items
            try:
                if not child.exists():
                    continue
                path_key = str(child.resolve())
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                max_context = _detect_model_max_context(child, file_format)
                broken, broken_reason = _detect_broken_library_item(child, file_format, source_kind)
                quantization = _detect_model_quantization(child, file_format, name_hint=name)
                backend = "llama.cpp" if file_format == "GGUF" else "mlx"
                if _looks_like_video_model(name):
                    model_type = "video"
                elif _looks_like_image_model(child, name):
                    model_type = "image"
                elif _looks_like_draft_model(name):
                    model_type = "draft"
                else:
                    model_type = "text"
                items.append(
                    {
                        "name": name,
                        "path": path_key,
                        "format": file_format,
                        "sourceKind": source_kind,
                        "quantization": quantization,
                        "backend": backend,
                        "modelType": model_type,
                        "sizeGb": _du_size_gb(child),
                        "lastModified": time.strftime("%Y-%m-%d %H:%M", time.localtime(child.stat().st_mtime)),
                        "actions": ["Run Chat", "Run Server", "Cache Preview", "Delete"],
                        "directoryId": directory_id,
                        "directoryLabel": directory_label,
                        "directoryPath": str(root),
                        "maxContext": max_context,
                        "broken": broken,
                        "brokenReason": broken_reason,
                    }
                    )
            except OSError:
                continue

    return items


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
