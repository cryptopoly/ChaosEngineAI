"""Pinned GGUF files for text-catalog variants.

Community GGUF repos usually ship several quants side by side — Q4_K_M,
Q6_K, Q8_0 and often a BF16 — so a whole-repo snapshot of a 27B model is
~70 GB and of a frontier MoE runs to terabytes. A text variant that sets
``ggufFile`` names the one file it means (the first shard, for split
files): downloads fetch only that quant plus any vision projector, and the
loader prefers it over the "largest .gguf in the directory" fallback,
which would otherwise pick the BF16 / Q8_0 sitting next to it.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Iterable

_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def pinned_gguf_file(repo_id: str | None) -> str | None:
    """The ``ggufFile`` a text-catalog variant pins for ``repo_id``, if any."""
    if not repo_id:
        return None
    from backend_service.catalog.text_models import MODEL_FAMILIES  # noqa: PLC0415

    for family in MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant.get("repo") == repo_id and variant.get("ggufFile"):
                return str(variant["ggufFile"])
    return None


def gguf_file_pattern(gguf_file: str) -> str:
    """Glob matching every shard of a (possibly split) GGUF file."""
    match = _SHARD_RE.search(gguf_file)
    if match is None:
        return gguf_file
    return f"{gguf_file[: match.start()]}-*-of-{match.group(2)}.gguf"


# Speculative-decoding sidecars ggml-org packs ship beside the main GGUF
# (llama.cpp ``common/download.cpp``: ``find_best_{mtp,dspark,dflash,eagle3}``).
# Order = llama.cpp's own preference when it infers the type from a draft
# repo (``common/arg.cpp``): an MTP head first, DSpark over DFlash (its
# sidecar carries the extra Markov head), EAGLE-3 last. DFlash 2 drafters
# are ``dflash-`` files — llama.cpp tells them apart by GGUF metadata.
SPEC_SIDECAR_TYPES: tuple[tuple[str, str], ...] = (
    ("mtp-", "draft-mtp"),
    ("dspark-", "draft-dspark"),
    ("dflash-", "draft-dflash"),
    ("eagle3-", "draft-eagle3"),
)

_SIDECAR_PREFIXES = tuple(prefix for prefix, _ in SPEC_SIDECAR_TYPES)
_QUANT_TAG_RE = re.compile(r"[-.]([A-Z0-9_]+)$", re.IGNORECASE)


def is_aux_gguf(name: str) -> bool:
    """True for projector / imatrix / spec-dec sidecar files — never the
    model itself (mirrors llama.cpp's ``gguf_filename_is_model``).

    Sidecar prefixes match case-sensitively, exactly like llama.cpp: a
    main model with baked-in heads such as ``Qwen3.6-27B-MTP-Q8_0.gguf``
    is a model, not an ``mtp-`` sidecar.
    """
    base = Path(name).name
    lower = base.lower()
    if "mmproj" in lower or "imatrix" in lower:
        return True
    return any(prefix in base for prefix in _SIDECAR_PREFIXES)


def _quant_bits(name: str) -> int:
    """llama.cpp's ``extract_quant_bits``: first number in the quant tag."""
    stem = Path(name).name
    if stem.lower().endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    match = _SHARD_RE.search(Path(name).name)
    if match is not None:
        stem = Path(name).name[: match.start()]
    tag = _QUANT_TAG_RE.search(stem)
    if tag is None:
        return 0
    digits = re.search(r"\d+", tag.group(1))
    return int(digits.group(0)) if digits else 0


def _best_sidecar(names: list[str], prefix: str, model_name: str) -> str | None:
    """Pick the sidecar whose quant is closest to the model's (first shard)."""
    model_bits = _quant_bits(model_name)
    candidates = [
        n for n in names
        if prefix in Path(n).name and n.lower().endswith(".gguf")
    ]
    if not candidates:
        return None
    first_shards = [
        n for n in candidates
        if (m := _SHARD_RE.search(Path(n).name)) is None or m.group(1) == "00001"
    ]
    return min(first_shards or candidates, key=lambda n: (abs(_quant_bits(n) - model_bits), n))


def resolve_spec_sidecar(model_gguf_path: str | None) -> tuple[str, str] | None:
    """``(spec_type, sidecar_path)`` for a draft sidecar next to the model.

    Scans only the model file's own directory (same reasoning as the
    mmproj resolver: a neighbouring repo's drafter would mismatch the
    target and crash llama-server).
    """
    if not model_gguf_path:
        return None
    main = Path(model_gguf_path)
    if not main.is_file():
        return None
    try:
        names = [entry.name for entry in main.parent.iterdir() if entry.is_file()]
    except OSError:
        return None
    for prefix, spec_type in SPEC_SIDECAR_TYPES:
        best = _best_sidecar(names, prefix, main.name)
        if best is not None:
            return spec_type, str(main.parent / best)
    return None


def text_repo_allow_patterns(repo_id: str, files: Iterable[dict] | None = None) -> list[str] | None:
    """``snapshot_download`` allowlist for a pinned text GGUF repo.

    ``None`` for unpinned repos so the caller downloads them in full.
    ``*mmproj*.gguf`` keeps the vision projector the llama.cpp engine
    wires via ``--mmproj`` (FU-072). When the Hub file list is supplied,
    the best-matching spec-dec sidecar of each kind (FU-089) rides along
    so speculative decoding works offline — one quant each, not every
    precision the repo ships.
    """
    gguf_file = pinned_gguf_file(repo_id)
    if gguf_file is None:
        return None
    patterns = [gguf_file_pattern(gguf_file), "*mmproj*.gguf", "*.md", "LICENSE*"]
    if files is not None:
        names = [str(entry.get("path") or "") for entry in files]
        for prefix, _spec_type in SPEC_SIDECAR_TYPES:
            best = _best_sidecar(names, prefix, gguf_file)
            if best is not None:
                patterns.append(gguf_file_pattern(best))
    return patterns


def matched_size_bytes(files: Iterable[dict], patterns: list[str]) -> int:
    """Sum ``sizeBytes`` of the Hub file entries an allowlist would fetch."""
    total = 0
    for entry in files:
        path = str(entry.get("path") or "")
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            total += int(entry.get("sizeBytes") or 0)
    return total


def repo_id_from_path(path: Path) -> str | None:
    """Recover ``owner/name`` from an HF-cache dir (``models--owner--name``)
    or an ``owner/name`` model-directory layout."""
    for part in reversed(path.parts):
        if part.startswith("models--"):
            owner, _, name = part[len("models--"):].partition("--")
            if owner and name:
                return f"{owner}/{name}"
    if len(path.parts) >= 2:
        return f"{path.parts[-2]}/{path.parts[-1]}"
    return None


def pick_model_gguf(directory: Path, model_files: list[Path]) -> Path:
    """Choose the GGUF to load from ``model_files`` (largest first).

    The catalog pin wins when it is on disk. Otherwise keep the largest
    file, but step back to its first shard — llama.cpp opens split GGUFs
    from ``-00001-of-``, and the largest shard is rarely that one.
    """
    pinned = pinned_gguf_file(repo_id_from_path(directory))
    if pinned:
        pattern = gguf_file_pattern(Path(pinned).name)
        matches = sorted(f for f in model_files if fnmatch.fnmatch(f.name, pattern))
        if matches:
            return matches[0]
    chosen = model_files[0]
    match = _SHARD_RE.search(chosen.name)
    if match and match.group(1) != "00001":
        first = chosen.with_name(f"{chosen.name[: match.start()]}-00001-of-{match.group(2)}.gguf")
        if first.exists():
            return first
    return chosen
