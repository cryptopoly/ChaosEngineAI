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


# Some repos publish a second GGUF layout under another extension so tools
# that list the repo do not mix the two (Unsloth's Inkling 2 Mini keeps the
# upstream-llama.cpp files in ``Mainline_Support/`` as ``*.gguf_file``). The
# pin names the Hub file; on disk the app adds a ``.gguf`` link beside it.
RENAMED_GGUF_SUFFIX = ".gguf_file"


def local_gguf_name(name: str) -> str:
    """The ``.gguf`` name a pinned Hub file has on disk (``x.gguf_file`` -> ``x.gguf``)."""
    if name.lower().endswith(RENAMED_GGUF_SUFFIX):
        return name[: -len("_file")]
    return name


def ensure_gguf_aliases(directory: Path) -> list[Path]:
    """Give every ``*.gguf_file`` under ``directory`` a ``.gguf`` link beside it.

    Every GGUF code path (loader, fit, discovery) looks for ``*.gguf``. In a
    Hugging Face snapshot the file is itself a symlink to a blob, so the alias
    is a symlink to the same blob and costs no disk; a plain file (a cache
    without symlinks) gets a hard link. Returns the aliases created.
    """
    import os  # noqa: PLC0415

    created: list[Path] = []
    try:
        sources = [p for p in directory.rglob(f"*{RENAMED_GGUF_SUFFIX}") if p.is_file()]
    except OSError:
        return created
    for source in sources:
        alias = source.with_name(local_gguf_name(source.name))
        if os.path.lexists(alias):
            continue
        try:
            if source.is_symlink():
                try:
                    os.symlink(os.readlink(source), alias)
                except OSError:  # no symlink right (Windows without developer mode)
                    os.link(source.resolve(), alias)
            else:
                os.link(source, alias)
            created.append(alias)
        except OSError:
            continue
    return created


def gguf_file_pattern(gguf_file: str) -> str:
    """Glob matching every shard of a (possibly split) GGUF file."""
    match = _SHARD_RE.search(gguf_file)
    if match is None:
        return gguf_file
    return f"{gguf_file[: match.start()]}-*-of-{match.group(2)}.gguf"


# Speculative-decoding sidecars ggml-org packs ship beside the main GGUF
# (llama.cpp ``common/download.cpp``: ``find_best_{mtp,dspark,dflash,eagle3}``).
# Order = which one the app runs when several are on disk. llama.cpp's own
# order puts an MTP head first; the app puts the block drafters first
# (FU-178): on Qwen3.8-27B Q4 on an M4 Max the DFlash 2 drafter decoded
# 55-59 tok/s against 32-34 for its MTP head at N=3 and 15 without drafting.
# DSpark stays over DFlash (its sidecar carries the extra Markov head),
# EAGLE-3 last. DFlash 2 drafters are ``dflash-`` files: ``draft-dflash``
# runs every DFlash-family drafter and llama.cpp picks the DFlash 2 code
# from the file's metadata (``llama_model_dflash_variant``), while
# ``draft-dflash2`` only refuses a drafter that is not DFlash 2.
SPEC_SIDECAR_TYPES: tuple[tuple[str, str], ...] = (
    ("dspark-", "draft-dspark"),
    ("dflash-", "draft-dflash"),
    ("mtp-", "draft-mtp"),
    ("eagle3-", "draft-eagle3"),
)
# Spec types that draft a whole block per pass (the engine asks for a full
# block; llama.cpp clamps it to the drafter's trained block size).
BLOCK_DRAFT_TYPES: frozenset[str] = frozenset({"draft-dflash", "draft-dflash2", "draft-dspark"})

_SIDECAR_PREFIXES = tuple(prefix for prefix, _ in SPEC_SIDECAR_TYPES)
_QUANT_TAG_RE = re.compile(r"[-.]([A-Z0-9_]+)$", re.IGNORECASE)


# Drafters their authors publish in a repo of their own (NVIDIA ships the
# Nemotron 3.5 Bolt DFlash drafter apart from the model, under a name that
# carries no ``dflash-`` prefix). Paired with the target by exact repo id,
# never by file name, so a stranger's drafter is never loaded against the
# wrong model. ``(drafter repo, drafter file, llama.cpp spec type)``.
# The catalog's Qwen3.8-27B GGUF that ships no drafter of its own gets z-lab's
# DFlash 2 GGUF (Apache-2.0, a mirror of incoai/Qwen3.8-27B-DFlash2-GGUF);
# its Q8_0 decoded about 10 % faster than its Q4_K_M for 0.9 GB more (FU-178).
_QWEN38_DFLASH2 = ("z-lab/Qwen3.8-27B-DFlash2-GGUF", "Qwen3.8-27B-DFlash2-Q8_0.gguf", "draft-dflash")

PINNED_DRAFTERS: dict[str, tuple[str, str, str]] = {
    "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB": (
        "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash",
        "Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf",
        "draft-dflash",
    ),
    "lmstudio-community/Qwen3.8-27B-GGUF": _QWEN38_DFLASH2,
    # Google's QAT GGUFs ship no drafter, and its ``-qat-q4_0-unquantized-assistant`` repos are
    # safetensors that llama.cpp cannot load. Unsloth converted the Gemma 4 assistants to GGUF
    # (``mtp-`` files); llama.cpp b12441 runs them as ``draft-mtp`` against the QAT target. Measured
    # 2026-11-19 on an RTX 4090 (12B QAT, 500-token prompt, 300 tokens): 92.8 tok/s plain, 123 tok/s
    # at ``--spec-draft-n-max 2`` (the engine's default), 104 at 5.
    "google/gemma-4-12B-it-qat-q4_0-gguf": ("unsloth/gemma-4-12B-it-GGUF", "mtp-gemma-4-12b-it.gguf", "draft-mtp"),
    # 26B-A4B QAT: 160 tok/s plain, 181 at n=2, 169 at n=3 (same prompt).
    "google/gemma-4-26B-A4B-it-qat-q4_0-gguf": (
        "unsloth/gemma-4-26B-A4B-it-GGUF", "mtp-gemma-4-26B-A4B-it.gguf", "draft-mtp",
    ),
    # 31B QAT (FU-264, 2026-11-20, RTX 4090, same prompt): 45.0 tok/s plain, 58.9 and 61.1 at n=2 (two runs
    # of three), 57.0 at n=1, 57.1 at n=3; 19.3 GB of VRAM plain, 19.8 GB with the drafter.
    "google/gemma-4-31B-it-qat-q4_0-gguf": ("unsloth/gemma-4-31B-it-GGUF", "mtp-gemma-4-31B-it.gguf", "draft-mtp"),
}


def pinned_drafter(repo_id: str | None) -> tuple[str, str, str] | None:
    """``(drafter repo, file, spec type)`` for a target with a separately published drafter."""
    return PINNED_DRAFTERS.get(repo_id) if repo_id else None


def resolve_pinned_drafter(
    repo_id: str | None, model_path: str | None = None
) -> tuple[str, str] | None:
    """``(spec_type, path)`` of the pinned drafter when it is in the Hub cache.

    ``model_path`` covers a model launched from its library folder, where the
    repo id is only recoverable from the Hub-cache layout.
    """
    pin = pinned_drafter(repo_id)
    if pin is None and model_path:
        pin = pinned_drafter(repo_id_from_path(Path(model_path)))
    if pin is None:
        return None
    from backend_service.helpers.hf_cache_paths import _hf_repo_snapshot_dir  # noqa: PLC0415

    drafter_repo, drafter_file, spec_type = pin
    snapshot = _hf_repo_snapshot_dir(drafter_repo)
    if snapshot is None:
        return None
    candidate = snapshot / drafter_file
    return (spec_type, str(candidate)) if candidate.is_file() else None


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


# Sidecar kinds a repo ships that measured slower than plain decode, so they are neither downloaded nor run
# (FU-263). ggml-org's Qwen3.6-35B-A3B Q4_K_M on an RTX 4090 (500-token prompt, 300 tokens): plain 175.9 and
# 177.7 tok/s, its DFlash drafter 119.0 and 122.1 (6.5 % of the drafted tokens accepted), its mtp- sidecar at
# two per round 204.3. The DFlash drafter pays on the dense 27B (FU-178), not on this 3B-active MoE.
SIDECAR_KINDS_SKIPPED: dict[str, frozenset[str]] = {
    "ggml-org/Qwen3.6-35B-A3B-GGUF": frozenset({"dflash-"}),
    # Inkling 2 Mini's root ``dspark-`` drafter is in an older format that upstream llama.cpp rejects, and the
    # upstream-format drafter needs an unmerged llama.cpp pull request (FU-540).
    "unsloth/Inkling-2-Mini-GGUF": frozenset({"dspark-"}),
}
_SIDECAR_SUBDIR = "MTP"
_MMPROJ_PREFERENCE = ("f16", "q8_0", "bf16")


def _best_mmproj(names: list[str]) -> str | None:
    """One projector out of the precisions a repo ships (FU-259): F16, then Q8_0, then BF16."""
    candidates = sorted(n for n in names if "mmproj" in Path(n).name.lower() and n.lower().endswith(".gguf"))
    if not candidates:
        return None
    for tag in _MMPROJ_PREFERENCE:
        for name in candidates:
            if tag in re.split(r"[-.]", Path(name).name.lower()):
                return name
    return candidates[0]


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
        # Unsloth keeps its sidecars in an ``MTP/`` subfolder (FU-259). Only that
        # one folder is read, so a neighbouring repo's drafter is still never used.
        for sub in main.parent.iterdir():
            if sub.is_dir() and sub.name.lower() == _SIDECAR_SUBDIR.lower():
                names.extend(f"{sub.name}/{entry.name}" for entry in sub.iterdir() if entry.is_file())
    except OSError:
        return None
    skipped = SIDECAR_KINDS_SKIPPED.get(repo_id_from_path(main) or "", frozenset())
    for prefix, spec_type in SPEC_SIDECAR_TYPES:
        if prefix in skipped:
            continue
        best = _best_sidecar(names, prefix, main.name)
        if best is not None:
            return spec_type, str(main.parent / best)
    return None


def text_repo_allow_patterns(repo_id: str, files: Iterable[dict] | None = None) -> list[str] | None:
    """``snapshot_download`` allowlist for a pinned text GGUF repo.

    ``None`` for unpinned repos so the caller downloads them in full.
    ``*mmproj*.gguf`` keeps the vision projector the llama.cpp engine
    wires via ``--mmproj`` (FU-072); with the Hub file list, one projector
    only (FU-259). When the Hub file list is supplied,
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
        mmproj = _best_mmproj(names)
        if mmproj is not None:
            patterns[1] = mmproj
        for prefix, _spec_type in SPEC_SIDECAR_TYPES:
            if prefix in SIDECAR_KINDS_SKIPPED.get(repo_id, frozenset()):
                continue
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
        pattern = gguf_file_pattern(local_gguf_name(Path(pinned).name))
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


def gguf_load_bytes(directory: Path) -> int | None:
    """Bytes of the GGUF llama.cpp would load from ``directory`` (all shards), or None.

    A repo folder can hold many quants (a 24B model's BF16, F16, Q8_0 ... is 150 GB
    on disk), but one file loads, so memory estimates must use that file.
    """
    try:
        files = [f for f in directory.rglob("*.gguf") if not is_aux_gguf(f.name)]
        if not files:
            return None
        files.sort(key=lambda f: f.stat().st_size, reverse=True)
        chosen = pick_model_gguf(directory, files)
        total = chosen.stat().st_size
        match = _SHARD_RE.search(chosen.name)
        if match:
            stem = chosen.name[: match.start()]
            for shard in chosen.parent.glob(f"{stem}-*-of-{match.group(2)}.gguf"):
                if shard != chosen:
                    total += shard.stat().st_size
        return total
    except OSError:
        return None
