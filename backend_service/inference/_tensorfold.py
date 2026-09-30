"""TensorFold model registry and routing helpers.

TensorFold (https://github.com/ashhart/TensorFold, MIT) is a local
OpenAI-compatible server for Apple Silicon whose per-family kernels verify
speculative drafts *exactly*: a drafted reply is byte-identical to the
serial reply, so drafting costs no quality. It serves only the model
families it has a recipe for, and only checkpoints whose storage format
matches what each family's kernels read.

This module is pure data + tiny filesystem helpers (no heavy imports) so the
controller, the engine and the system-stats payload can all consult it:

* ``TENSORFOLD_FAMILIES`` — the families TensorFold serves on MLX, their
  ``config.json`` ``model_type`` values, the checkpoints its docs name as
  tested, and each family's optional external draft model. A family needs a
  ``load()`` in its TensorFold package to be listed: Qwen3.6-35B-A3B
  (``qwen3_5_moe``) ships only a CUDA engine, so it is deliberately absent.
* ``match_model`` — which family (if any) a model belongs to, and how it
  should be routed (see :class:`TensorFoldMatch`).
* ``resolve_model_dir`` / ``native_context_window`` — locate the local
  snapshot the server is started on, without ever touching the network.

Every fact below is taken from TensorFold's own source at v0.5.0
(``src/tensorfold/families/*/__init__.py`` ``MODEL_TYPES`` / ``MODELS`` /
``DRAFTER``) and its recipe book; re-check it when the pin in
``scripts/install-tensorfold.sh`` moves.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

@dataclass(frozen=True)
class TensorFoldFamily:
    key: str
    title: str
    # ``config.json`` ``model_type`` values the family serves.
    model_types: tuple[str, ...]
    # Checkpoints TensorFold's docs name as tested on MLX.
    tested_repos: tuple[str, ...]
    # Optional external draft model the Mac engine can draft with.
    drafter: str | None = None
    # Extra ``tensorfold serve`` flags that go with that drafter.
    drafter_args: tuple[str, ...] = ()
    # True when no other engine we ship can load the family on Apple Silicon
    # (stock mlx-lm has no such architecture), so a model of this family is
    # always sent to TensorFold — or fails with an install hint.
    exclusive: bool = False
    # Minimum Mac unified memory TensorFold's docs state, for the UI.
    min_memory_gb: int | None = None


TENSORFOLD_FAMILIES: tuple[TensorFoldFamily, ...] = (
    TensorFoldFamily(
        key="qwen3_8_dense",
        title="Qwen3.8 dense",
        model_types=("qwen3_5",),
        tested_repos=("Vontra/Qwen3.8-27B-MLX-4bit",),
        drafter="z-lab/Qwen3.8-27B-DFlash2",
    ),
    TensorFoldFamily(
        key="qwen3_8_flash_next",
        title="Qwen3.8 Flash Next",
        model_types=("qwen4_exp",),
        tested_repos=("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP",),
        exclusive=True,
        min_memory_gb=128,
    ),
    TensorFoldFamily(
        key="nemotron_h",
        title="Nemotron 3.5 Lightning",
        model_types=("nemotron_h",),
        tested_repos=("Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit",),
    ),
    TensorFoldFamily(
        key="gemma4",
        title="Gemma 4",
        model_types=("gemma4", "gemma4_text"),
        tested_repos=("mlx-community/gemma-4-26b-a4b-it-4bit",),
        drafter="z-lab/gemma-4-26B-A4B-it-DFlash",
        drafter_args=("--drafter-bits", "8"),
    ),
    TensorFoldFamily(
        key="glm5_flash",
        title="GLM-5.3-Flash",
        model_types=("glm5_next",),
        tested_repos=("Vontra/GLM-5.3-Flash-MLX-4bit-MTP",),
        # The optional incoai DFlash2 drafter is CUDA-only and carries
        # non-commercial terms; the Mac engine drafts with the MTP layer.
        exclusive=True,
        min_memory_gb=256,
    ),
    TensorFoldFamily(
        key="deepseek_v4_flash",
        title="DeepSeek-V4-Flash",
        model_types=("deepseek_v4",),
        tested_repos=("mlx-community/DeepSeek-V4-Flash-4bit",),
        drafter="Vontra/DeepSeek-V4-Flash-DSpark-MLX",
        exclusive=True,
        min_memory_gb=256,
    ),
    TensorFoldFamily(
        key="ternary_bonsai_2",
        title="Ternary Bonsai 2",
        model_types=("prism_hadamard_qwen35",),
        tested_repos=("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",),
        drafter="z-lab/Qwen3.8-27B-DFlash2",
        exclusive=True,
        min_memory_gb=32,
    ),
)

_REPO_INDEX: dict[str, TensorFoldFamily] = {
    repo.lower(): family
    for family in TENSORFOLD_FAMILIES
    for repo in family.tested_repos
}

_EXCLUSIVE_MODEL_TYPES: dict[str, TensorFoldFamily] = {
    model_type: family
    for family in TENSORFOLD_FAMILIES
    if family.exclusive
    for model_type in family.model_types
}


def supported_repos() -> list[str]:
    """Every checkpoint TensorFold's docs name as tested, for the UI."""
    return [repo for family in TENSORFOLD_FAMILIES for repo in family.tested_repos]


def exclusive_repos() -> list[str]:
    """Tested checkpoints of families no other shipped engine can load."""
    return [
        repo
        for family in TENSORFOLD_FAMILIES
        if family.exclusive
        for repo in family.tested_repos
    ]


def min_memory_gb_by_repo() -> dict[str, int]:
    """Unified memory (GB) TensorFold's docs state for the checkpoints that name one."""
    return {
        repo: family.min_memory_gb
        for family in TENSORFOLD_FAMILIES
        if family.min_memory_gb
        for repo in family.tested_repos
    }


@dataclass(frozen=True)
class TensorFoldMatch:
    """How a model relates to TensorFold.

    ``tier`` decides routing when the request did not name the engine:

    * ``"exclusive"`` — only TensorFold can run it; always routed (an
      uninstalled TensorFold becomes an install prompt, not a stack trace).
    * ``"tested"`` — a checkpoint TensorFold is tested with that stock MLX
      also loads. It goes to TensorFold when the request asks for speculative
      decoding and TensorFold is installed, the same way the MTPLX lane is
      chosen; otherwise it loads on the standard MLX engine.
    """

    family: TensorFoldFamily
    repo: str | None
    matched_by: str  # "repo" | "config"

    @property
    def tier(self) -> str:
        return "exclusive" if self.family.exclusive else "tested"


def _repo_from_path(path: str | None) -> str | None:
    if not path:
        return None
    from backend_service.helpers.text_gguf import repo_id_from_path

    try:
        return repo_id_from_path(Path(path))
    except (OSError, ValueError):
        return None


def read_model_type(model_dir: Path | None) -> str | None:
    """``model_type`` from ``config.json``, as TensorFold's own detector reads it."""
    if model_dir is None:
        return None
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    model_type = config.get("model_type")
    if not model_type:
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            model_type = text_config.get("model_type")
    return str(model_type) if model_type else None


def match_model(
    *,
    canonical_repo: str | None = None,
    model_ref: str | None = None,
    runtime_target: str | None = None,
    path: str | None = None,
    model_dir: Path | None = None,
) -> TensorFoldMatch | None:
    """Find the TensorFold family a model belongs to, if any.

    Repo ids are checked first (they identify the tested checkpoints). A
    local ``config.json`` is consulted only for *exclusive* families, where
    ``model_type`` uniquely identifies a family stock mlx-lm cannot load —
    ``model_type`` alone is not evidence of support for shared families
    (every Qwen3.5/3.6/3.8 dense model says ``qwen3_5``, and TensorFold's
    per-family check refuses most of them).
    """
    repos: list[str] = [r for r in (canonical_repo, model_ref, runtime_target) if r]
    derived = _repo_from_path(path)
    if derived:
        repos.append(derived)
    for repo in repos:
        family = _REPO_INDEX.get(repo.lower())
        if family is not None:
            return TensorFoldMatch(family=family, repo=repo, matched_by="repo")

    directory = model_dir or _directory_candidate(path, runtime_target)
    family = _EXCLUSIVE_MODEL_TYPES.get(read_model_type(directory) or "")
    if family is not None:
        return TensorFoldMatch(family=family, repo=None, matched_by="config")
    return None


def _directory_candidate(*candidates: str | None) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        expanded = Path(candidate).expanduser()
        # A repo id (``owner/name``) is a relative path; it must never be
        # resolved against whatever directory the backend happens to run in.
        if not expanded.is_absolute():
            continue
        try:
            resolved = _config_dir(expanded)
        except OSError:
            continue
        if resolved is not None:
            return resolved
    return None


def _config_dir(directory: Path) -> Path | None:
    """``directory`` itself when it holds ``config.json``, else the newest
    Hugging Face snapshot beneath it (a ``models--owner--name`` cache entry)."""
    if not directory.is_dir():
        return None
    if (directory / "config.json").is_file():
        return directory
    snapshots = directory / "snapshots"
    if snapshots.is_dir():
        found = [s for s in snapshots.iterdir() if (s / "config.json").is_file()]
        if found:
            return max(found, key=lambda s: s.stat().st_mtime)
    return None


def resolve_model_dir(
    *,
    path: str | None,
    runtime_target: str | None,
    repos: Iterable[str | None],
) -> Path | None:
    """The local directory to start ``tensorfold serve`` on, or ``None``.

    Never downloads: a local path wins, then the Hugging Face cache (which
    honours the relocated-cache setting). ``tensorfold serve`` given a bare
    repo id would fetch weights itself, so the engine always hands it a
    directory instead.
    """
    found = _directory_candidate(path, runtime_target)
    if found is not None:
        return found
    from backend_service.helpers.hf_cache_paths import _hf_repo_snapshot_dir

    for repo in repos:
        if not repo or "/" not in repo or Path(repo).expanduser().exists():
            continue
        snapshot = _hf_repo_snapshot_dir(repo)
        if snapshot is not None and (snapshot / "config.json").is_file():
            return snapshot
    return None


_SHARD_RE = re.compile(r"model-(\d+)-of-(\d+)\.safetensors")


def weights_complete(snapshot: Path) -> bool:
    """Whether ``snapshot`` holds every weight file its checkpoint names.

    Mirrors TensorFold's own check (``hub._cached_weights_complete``): the
    Hugging Face cache also returns partial snapshots (an interrupted pull),
    and TensorFold refuses to serve one. Checking first turns that into an
    immediate, readable error instead of a failed start.
    """
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            names = set(weight_map.values())
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False
        if not weight_map:
            return False
        return all(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and (snapshot / name).is_file()
            for name in names
        )

    shards = list(snapshot.glob("model-*-of-*.safetensors"))
    if shards:
        matches = [_SHARD_RE.fullmatch(shard.name) for shard in shards]
        if not all(matches):
            return False
        totals = {int(match.group(2)) for match in matches if match}
        if len(totals) != 1:
            return False
        total = totals.pop()
        present = {int(match.group(1)) for match in matches if match}
        return len(shards) == total and present == set(range(1, total + 1))

    return (snapshot / "model.safetensors").is_file()


def native_context_window(model_dir: Path) -> int | None:
    """The checkpoint's own context window (``max_position_embeddings``).

    ``tensorfold serve --context N`` refuses N beyond it, so the engine
    clamps to it first.
    """
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    text_config = config.get("text_config")
    for source in (text_config if isinstance(text_config, dict) else None, config):
        if not source:
            continue
        limit = source.get("max_position_embeddings")
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            return limit
    return None
