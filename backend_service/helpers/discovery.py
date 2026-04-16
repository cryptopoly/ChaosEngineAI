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
from backend_service.helpers.settings import _normalize_slug
from backend_service.helpers.system import _safe_run

_UNSUPPORTED_MLX_QUANT_ALGOS = {"NVFP4", "NVINT4"}


def _path_size_bytes(path: Path, *, seen: set[tuple[int, int]] | None = None) -> int:
    visited = seen if seen is not None else set()
    try:
        stat_result = path.stat()
    except OSError:
        return 0

    file_id = (stat_result.st_dev, stat_result.st_ino)
    if file_id in visited:
        return 0
    visited.add(file_id)

    if path.is_file():
        return int(stat_result.st_size)

    total = 0
    try:
        children = list(path.iterdir())
    except OSError:
        return 0

    for child in children:
        total += _path_size_bytes(child, seen=visited)
    return total


def _du_size_gb(path: Path) -> float:
    if path.is_file():
        return _bytes_to_gb(_path_size_bytes(path))

    payload = _safe_run(["du", "-sk", str(path)], timeout=4.0)
    if payload:
        try:
            size_kb = int(payload.split()[0])
            size_gb = round(size_kb / (1024 ** 2), 1)
            if size_gb > 0:
                return size_gb
        except (ValueError, IndexError):
            pass

    fallback_bytes = _path_size_bytes(path)
    return _bytes_to_gb(fallback_bytes) if fallback_bytes > 0 else 0.0


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


_SHARDED_WEIGHT_RE = re.compile(
    r"(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.(?P<suffix>safetensors|bin)$",
    re.IGNORECASE,
)


def _incomplete_sharded_weight_reason(path: Path) -> str | None:
    try:
        files = [entry.name for entry in path.iterdir() if entry.is_file()]
    except OSError:
        return None

    shard_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for filename in files:
        match = _SHARDED_WEIGHT_RE.match(filename)
        if not match:
            continue
        key = (match.group("prefix"), match.group("suffix").lower())
        expected_total = int(match.group("total"))
        shard_index = int(match.group("index"))
        group = shard_groups.setdefault(key, {"expected_total": expected_total, "present": set()})
        group["expected_total"] = max(int(group["expected_total"]), expected_total)
        group["present"].add(shard_index)

    for (_prefix, suffix), group in shard_groups.items():
        expected_total = int(group["expected_total"])
        present = set(group["present"])
        if expected_total <= 1:
            continue
        missing = [index for index in range(1, expected_total + 1) if index not in present]
        if missing:
            sample = ", ".join(f"{index:05d}" for index in missing[:3])
            more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
            return (
                f"Incomplete sharded {suffix} checkpoint: found {len(present)} of {expected_total} shard files. "
                f"Missing shards include {sample}{more}."
            )
    return None


def _incomplete_gguf_directory_reason(path: Path) -> str | None:
    try:
        gguf_files = [entry for entry in path.rglob("*.gguf") if entry.is_file()]
        part_files = [entry for entry in path.rglob("*.gguf.part") if entry.is_file()]
    except OSError:
        return None

    main_files = [entry for entry in gguf_files if "mmproj" not in entry.name.lower()]
    if main_files:
        return None
    if part_files:
        sample = ", ".join(entry.name for entry in part_files[:2])
        more = f" (+{len(part_files) - 2} more)" if len(part_files) > 2 else ""
        return (
            f"GGUF download is incomplete: main model weights are still downloading "
            f"({sample}{more})."
        )
    if gguf_files:
        return "GGUF directory only contains a vision projector (mmproj) and no main model weights."
    return None


def _quantization_label_from_text(text: str) -> str | None:
    lowered = text.lower()
    match = re.search(r"\b(q\d(?:_[a-z0-9]+)*)\b", lowered)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(\d+)[-_ ]?bit\b", lowered)
    if match:
        return f"{int(match.group(1))}-bit"
    if "bf16" in lowered or "bfloat16" in lowered:
        return "BF16"
    if "fp16" in lowered or "float16" in lowered:
        return "FP16"
    if "fp8" in lowered or "float8" in lowered:
        return "FP8"
    if "fp32" in lowered or "float32" in lowered:
        return "FP32"
    return None


def _mlx_quantization_bits(config: dict[str, Any] | None) -> int | None:
    if not isinstance(config, dict):
        return None
    if _unsupported_mlx_quantization_reason(config):
        return None
    for key in ("quantization", "quantization_config"):
        payload = config.get(key)
        if isinstance(payload, dict):
            bits = payload.get("bits")
            if isinstance(bits, (int, float)) and bits > 0:
                try:
                    return int(bits)
                except (TypeError, ValueError):
                    return None
    return None


def _quantization_algo_label(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    payload = config.get("quantization_config")
    if not isinstance(payload, dict):
        return None
    algo = payload.get("quant_algo")
    if isinstance(algo, str) and algo.strip():
        return algo.strip().upper()
    return None


def _unsupported_mlx_quantization_reason(config: dict[str, Any] | None) -> str | None:
    algo = _quantization_algo_label(config)
    if not algo or algo not in _UNSUPPORTED_MLX_QUANT_ALGOS:
        return None
    method = ""
    if isinstance(config, dict):
        payload = config.get("quantization_config")
        if isinstance(payload, dict):
            raw_method = payload.get("quant_method")
            if isinstance(raw_method, str) and raw_method.strip():
                method = raw_method.strip()
    method_label = f" (via {method})" if method else ""
    return (
        f"This model uses {algo} quantisation{method_label}, which is not supported by the MLX runtime. "
        f"It needs a CUDA/NVIDIA runtime such as vLLM with modelopt support, or a different build such as GGUF or MLX."
    )


def _dtype_quantization_label(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    candidates: list[Any] = [config.get("torch_dtype"), config.get("dtype")]
    for nested_key in ("text_config", "llm_config"):
        nested = config.get(nested_key)
        if isinstance(nested, dict):
            candidates.extend([nested.get("torch_dtype"), nested.get("dtype")])
    for value in candidates:
        if not value:
            continue
        label = _quantization_label_from_text(str(value))
        if label:
            return label
    return None


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


_IMAGE_MODEL_KEYWORDS = (
    "stable-diffusion", "sdxl", "flux.", "flux1", "flux-",
    "dall-e", "imagen", "kandinsky", "wuerstchen",
    "diffusion-pipe", "qwen-image", "qwen/qwen-image",
)


_DRAFT_MODEL_KEYWORDS = (
    "-dflash", "/dflash", "-draft", "-eagle",
)


def _looks_like_draft_model(name: str) -> bool:
    """Return True if this looks like a speculative decoding draft model.

    Draft models (DFlash, EAGLE, etc.) are companion checkpoints, not
    standalone LLMs.  They should not appear in the model picker.
    """
    lower = name.lower()
    return any(kw in lower for kw in _DRAFT_MODEL_KEYWORDS)


def _looks_like_image_model(path: Path, name: str) -> bool:
    """Return True if this looks like a diffusion / image generation model."""
    lower_name = name.lower()
    if any(kw in lower_name for kw in _IMAGE_MODEL_KEYWORDS):
        return True
    # Diffusers models have model_index.json
    if (path / "model_index.json").exists():
        return True
    return False


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


def _list_weight_files(raw_path: str) -> dict[str, Any]:
    """Inspect a model path and list its weight files.

    Used by the frontend picker to let users choose a specific .gguf when a
    directory contains multiple weights. Mirrors ``_resolve_gguf_path`` logic
    for GGUF directories.
    """
    target = Path(os.path.expanduser(raw_path or "")).expanduser()
    if not target.exists():
        return {
            "path": str(target),
            "format": "unknown",
            "files": [],
            "broken": True,
            "brokenReason": "Path does not exist",
        }

    def _gb(p: Path) -> float:
        try:
            return round(p.stat().st_size / (1024 ** 3), 2)
        except OSError:
            return 0.0

    # Single file
    if target.is_file():
        suffix = target.suffix.lower()
        if suffix == ".gguf":
            fmt = "GGUF"
        elif suffix == ".safetensors":
            fmt = "Transformers"
        else:
            fmt = "unknown"
        return {
            "path": str(target),
            "format": fmt,
            "files": [
                {
                    "name": target.name,
                    "path": str(target),
                    "sizeGb": _gb(target),
                    "role": "main",
                }
            ],
            "broken": False,
            "brokenReason": None,
        }

    # Directory
    ggufs = sorted(target.rglob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
    gguf_partials = sorted(target.rglob("*.gguf.part"))
    if ggufs or gguf_partials:
        broken_reason = _incomplete_gguf_directory_reason(target)
        files = []
        for f in ggufs:
            is_mmproj = "mmproj" in f.name.lower()
            files.append(
                {
                    "name": f.name,
                    "path": str(f),
                    "sizeGb": _gb(f),
                    "role": "mmproj" if is_mmproj else "main",
                }
            )
        for f in gguf_partials:
            files.append(
                {
                    "name": f.name,
                    "path": str(f),
                    "sizeGb": _gb(f),
                    "role": "partial",
                }
            )
        return {
            "path": str(target),
            "format": "GGUF",
            "files": files,
            "broken": broken_reason is not None,
            "brokenReason": broken_reason,
        }

    safetensors = sorted(target.glob("*.safetensors"))
    if safetensors:
        shard_reason = _incomplete_sharded_weight_reason(target)
        files = [
            {
                "name": f.name,
                "path": str(f),
                "sizeGb": _gb(f),
                "role": "main",
            }
            for f in safetensors
        ]
        has_mlx = any(f.name == "model.safetensors" for f in safetensors) or (target / "model.safetensors").exists()
        fmt = "MLX" if has_mlx and not (target / "model.safetensors.index.json").exists() else "Transformers"
        return {
            "path": str(target),
            "format": fmt,
            "files": files,
            "broken": shard_reason is not None,
            "brokenReason": shard_reason,
        }

    # No weights found
    return {
        "path": str(target),
        "format": "unknown",
        "files": [],
        "broken": True,
        "brokenReason": "No .gguf or .safetensors weights found",
    }


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
                if _looks_like_image_model(child, name):
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
