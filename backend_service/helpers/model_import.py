"""Import existing Ollama / LM Studio models by reference (#4).

The #1 switching cost for a local-AI app is re-downloading models you
already have. This module discovers models in the Ollama blob store and
LM Studio cache and registers them into ChaosEngineAI *by reference* — a
symlink into a managed ``<dataDir>/imported-models/`` directory, never a
copy — so the existing library scan picks them up and they load like any
other model.

Ollama stores weights as digest-named blobs (``blobs/sha256-<hex>``, no
extension) with an OCI-style manifest per model under
``manifests/<registry>/<ns>/<model>/<tag>``. We parse the manifest to
find the ``application/vnd.ollama.image.model`` layer, resolve its blob,
and symlink it with a ``.gguf`` extension so the GGUF-aware scanner sees
it.

LM Studio stores real ``.gguf`` files in a nested
``<publisher>/<repo>/<file>.gguf`` tree, so those are discovered directly.

All discovery is read-only; ``import_candidate`` is the only mutating
call and only ever creates a symlink.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Env overrides (mirror Ollama's own OLLAMA_MODELS) so power users with
# relocated stores can still import.
_ENV_OLLAMA_DIR = "CHAOSENGINE_OLLAMA_DIR"
_ENV_OLLAMA_MODELS = "OLLAMA_MODELS"
_ENV_LMSTUDIO_DIR = "CHAOSENGINE_LMSTUDIO_DIR"

_OLLAMA_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass
class ImportCandidate:
    name: str  # e.g. "llama3.2:latest" or "bartowski/Qwen3-8B-GGUF/file"
    repo: str  # name without tag — used as canonicalRepo on load
    source: str  # "ollama" | "lmstudio"
    path: str  # absolute path to the on-disk weights (blob or .gguf)
    size_bytes: int
    fmt: str = "GGUF"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repo": self.repo,
            "source": self.source,
            "path": self.path,
            "sizeBytes": self.size_bytes,
            "sizeGb": round(self.size_bytes / 1e9, 2),
            "format": self.fmt,
        }


# --------------------------------------------------------------------------
# Path discovery
# --------------------------------------------------------------------------


def default_ollama_models_dir() -> Path | None:
    """Resolve the Ollama *models* dir (the one containing blobs/ + manifests/)."""
    override = os.environ.get(_ENV_OLLAMA_MODELS)
    if override:
        return Path(override).expanduser()
    root_override = os.environ.get(_ENV_OLLAMA_DIR)
    root = Path(root_override).expanduser() if root_override else Path.home() / ".ollama"
    # The blobs/manifests live under ``<root>/models`` for a standard install;
    # accept ``<root>`` directly if it already contains them.
    if (root / "models" / "blobs").is_dir():
        return root / "models"
    if (root / "blobs").is_dir():
        return root
    return root / "models"


def default_lmstudio_dirs() -> list[Path]:
    override = os.environ.get(_ENV_LMSTUDIO_DIR)
    if override:
        return [Path(override).expanduser()]
    home = Path.home()
    return [
        home / ".lmstudio" / "models",
        home / ".cache" / "lm-studio" / "models",
    ]


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------


def parse_ollama_manifest(raw: dict[str, Any]) -> tuple[str | None, int]:
    """Return ``(blob_hex, size)`` for the model layer. Pure — for tests.

    ``blob_hex`` is the 64-char sha256 hex of the weights blob, or None
    if the manifest has no model layer / a malformed digest.
    """
    layers = raw.get("layers") or []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        if layer.get("mediaType") != _OLLAMA_MODEL_MEDIA_TYPE:
            continue
        digest = str(layer.get("digest") or "")
        # Ollama digests are ``sha256:<hex>``.
        hex_part = digest.split(":", 1)[1] if ":" in digest else digest
        if _SHA256_RE.match(hex_part):
            size = int(layer.get("size") or 0)
            return hex_part, size
    return None, 0


def _ollama_name_from_manifest_path(manifest_path: Path, manifests_root: Path) -> tuple[str, str]:
    """Derive ``(name, repo)`` from the manifest path.

    Layout: ``manifests/<registry>/<ns...>/<model>/<tag>``. The tag is the
    filename; the ``<registry>`` segment and a ``library`` namespace are
    dropped for the friendly name (``llama3.2:latest``).
    """
    rel = manifest_path.relative_to(manifests_root).parts
    # rel = (registry, ns..., model, tag)
    tag = rel[-1] if rel else "latest"
    middle = list(rel[1:-1])  # drop registry, drop tag
    if middle and middle[0] == "library":
        middle = middle[1:]
    repo = "/".join(middle) if middle else manifest_path.parent.name
    return f"{repo}:{tag}", repo


def scan_ollama(models_dir: Path | None) -> list[ImportCandidate]:
    if models_dir is None:
        return []
    manifests_root = models_dir / "manifests"
    blobs_dir = models_dir / "blobs"
    if not manifests_root.is_dir() or not blobs_dir.is_dir():
        return []

    candidates: list[ImportCandidate] = []
    for manifest_path in manifests_root.rglob("*"):
        if not manifest_path.is_file():
            continue
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        blob_hex, size = parse_ollama_manifest(raw)
        if blob_hex is None:
            continue
        blob_path = blobs_dir / f"sha256-{blob_hex}"
        if not blob_path.is_file():
            continue
        name, repo = _ollama_name_from_manifest_path(manifest_path, manifests_root)
        actual_size = size or blob_path.stat().st_size
        candidates.append(
            ImportCandidate(name=name, repo=repo, source="ollama", path=str(blob_path), size_bytes=actual_size)
        )
    candidates.sort(key=lambda c: c.name)
    return candidates


# --------------------------------------------------------------------------
# LM Studio
# --------------------------------------------------------------------------


def scan_lmstudio(dirs: list[Path]) -> list[ImportCandidate]:
    candidates: list[ImportCandidate] = []
    seen: set[str] = set()
    for root in dirs:
        if not root.is_dir():
            continue
        for gguf in root.rglob("*.gguf"):
            if not gguf.is_file():
                continue
            real = str(gguf.resolve())
            if real in seen:
                continue
            seen.add(real)
            rel = gguf.relative_to(root)
            # publisher/repo from the directory layout when present.
            repo = "/".join(rel.parts[:-1]) if len(rel.parts) > 1 else gguf.stem
            try:
                size = gguf.stat().st_size
            except OSError:
                size = 0
            candidates.append(
                ImportCandidate(name=str(rel), repo=repo, source="lmstudio", path=str(gguf), size_bytes=size)
            )
    candidates.sort(key=lambda c: c.name)
    return candidates


# --------------------------------------------------------------------------
# Scan + import
# --------------------------------------------------------------------------


def scan_importable() -> dict[str, Any]:
    ollama_dir = default_ollama_models_dir()
    lmstudio_dirs = default_lmstudio_dirs()
    ollama = scan_ollama(ollama_dir)
    lmstudio = scan_lmstudio(lmstudio_dirs)
    return {
        "ollama": {
            "available": ollama_dir is not None and (ollama_dir / "blobs").is_dir(),
            "dir": str(ollama_dir) if ollama_dir else None,
            "models": [c.to_dict() for c in ollama],
        },
        "lmstudio": {
            "available": any(d.is_dir() for d in lmstudio_dirs),
            "dirs": [str(d) for d in lmstudio_dirs if d.is_dir()],
            "models": [c.to_dict() for c in lmstudio],
        },
    }


def imported_dir(data_dir: Path) -> Path:
    return data_dir / "imported-models"


def _slug(value: str) -> str:
    cleaned = _SLUG_RE.sub("-", value).strip("-")
    return cleaned or "model"


def import_by_reference(*, source: str, path: str, name: str, data_dir: Path) -> dict[str, Any]:
    """Symlink an existing model file into the managed imported dir.

    Returns ``{importedPath, alreadyImported, importedDir}``. Raises
    ``FileNotFoundError`` if the source weights are missing and
    ``OSError`` if the symlink can't be created (e.g. Windows without
    privilege) — callers translate those into user-facing messages.
    """
    src = Path(path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"Source weights not found: {src}")

    dest_dir = imported_dir(data_dir) / source
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{_slug(name)}.gguf"

    if dest.exists() or dest.is_symlink():
        return {"importedPath": str(dest), "alreadyImported": True, "importedDir": str(imported_dir(data_dir))}

    os.symlink(src, dest)
    return {"importedPath": str(dest), "alreadyImported": False, "importedDir": str(imported_dir(data_dir))}
