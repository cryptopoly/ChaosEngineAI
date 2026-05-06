"""Settings: data location, user preferences, model directories."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIRECTORIES: list[dict[str, Any]] = [
    {
        "id": "hf-cache",
        "label": "Hugging Face cache",
        "path": "~/.cache/huggingface/hub",
        "enabled": True,
        "source": "default",
    },
    {
        "id": "mlx-cache",
        "label": "MLX cache",
        "path": "~/.cache/mlx",
        "enabled": True,
        "source": "default",
    },
    {
        "id": "home-models",
        "label": "Models",
        "path": "~/Models",
        "enabled": True,
        "source": "default",
    },
]
DEFAULT_LAUNCH_PREFERENCES = {
    "contextTokens": 8192,
    "maxTokens": 4096,
    "temperature": 0.7,
    "cacheStrategy": "native",
    "cacheBits": 0,
    "fp16Layers": 0,
    "fusedAttention": False,
    "fitModelInMemory": True,
}


def _load_data_location(bootstrap_path: Path, bootstrap_dir: Path) -> Path:
    """Read the bootstrap pointer file. Falls back to ``bootstrap_dir``."""
    if not bootstrap_path.exists():
        return bootstrap_dir
    try:
        payload = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return bootstrap_dir
    raw = payload.get("dataDirectory") if isinstance(payload, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return bootstrap_dir
    try:
        return Path(os.path.expanduser(raw)).resolve()
    except (OSError, RuntimeError):
        return bootstrap_dir


def _save_data_location(target: Path) -> None:
    """Write the bootstrap pointer atomically. ``target`` must be resolved."""
    bootstrap_dir = Path.home() / ".chaosengine"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_path = bootstrap_dir / ".location.json"
    tmp = bootstrap_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"dataDirectory": str(target)}, indent=2),
        encoding="utf-8",
    )
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(str(tmp), str(bootstrap_path))


def _migrate_data_directory(old: Path, new: Path) -> dict[str, Any]:
    """Copy known data files/dirs from ``old`` to ``new``.

    Idempotent: skips entries that already exist at the destination. Never
    deletes anything from ``old``. Validates writability via a probe file.
    Raises ``RuntimeError`` if ``new`` is not writable.
    """
    import shutil

    old = Path(os.path.expanduser(str(old))).resolve()
    new = Path(os.path.expanduser(str(new))).resolve()
    try:
        new.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot create data directory {new}: {exc}") from exc

    probe = new / ".chaosengine-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"Data directory {new} is not writable: {exc}") from exc

    summary: dict[str, Any] = {
        "copied": [],
        "skipped": [],
        "from": str(old),
        "to": str(new),
    }
    if old == new:
        return summary

    for name in ("settings.json", "benchmark-history.json", "chat-sessions.json"):
        src = old / name
        dst = new / name
        if not src.exists():
            continue
        if dst.exists():
            summary["skipped"].append(name)
            continue
        try:
            shutil.copy2(src, dst)
            summary["copied"].append(name)
        except OSError as exc:
            raise RuntimeError(f"Failed to copy {name}: {exc}") from exc

    docs_src = old / "documents"
    docs_dst = new / "documents"
    if docs_src.exists() and docs_src.is_dir():
        if docs_dst.exists():
            summary["skipped"].append("documents/")
        else:
            try:
                shutil.copytree(str(docs_src), str(docs_dst))
                summary["copied"].append("documents/")
            except OSError as exc:
                raise RuntimeError(f"Failed to copy documents/: {exc}") from exc

    return summary


class DataLocation:
    """Resolves where ChaosEngineAI persists user data.

    The bootstrap pointer at ``~/.chaosengine/.location.json`` may redirect
    the actual data directory to a user-chosen path (e.g. a Dropbox folder).
    Missing or unreadable pointer means data lives at the bootstrap dir for
    backwards compatibility with older installs.
    """

    def __init__(self) -> None:
        self.bootstrap_dir: Path = Path.home() / ".chaosengine"
        self.bootstrap_path: Path = self.bootstrap_dir / ".location.json"
        self.bootstrap_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir: Path = _load_data_location(self.bootstrap_path, self.bootstrap_dir)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.data_dir = self.bootstrap_dir
            self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def settings_path(self) -> Path:
        return self.data_dir / "settings.json"

    @property
    def benchmarks_path(self) -> Path:
        return self.data_dir / "benchmark-history.json"

    @property
    def chat_sessions_path(self) -> Path:
        return self.data_dir / "chat-sessions.json"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def image_outputs_dir(self) -> Path:
        return self.images_dir / "outputs"

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def video_outputs_dir(self) -> Path:
        return self.videos_dir / "outputs"


def _normalize_slug(value: str, fallback: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or fallback


def _default_settings(default_port: int, data_dir: Path) -> dict[str, Any]:
    return {
        "modelDirectories": [dict(entry) for entry in DEFAULT_MODEL_DIRECTORIES],
        "preferredServerPort": default_port,
        "allowRemoteConnections": False,
        # Default on — the API token is auto-generated and passed to the
        # frontend via /api/auth/session, so the built-in UI works out of
        # the box. Users who connect external clients (OpenWebUI, scripts,
        # another desktop app) can flip this off from the Server tab.
        "requireApiAuth": True,
        "autoStartServer": False,
        "launchPreferences": dict(DEFAULT_LAUNCH_PREFERENCES),
        "remoteProviders": [],
        "huggingFaceToken": "",
        "dataDirectory": str(data_dir),
        # Empty string means "use the default under dataDirectory". Anything
        # else is treated as an absolute (or ~-relative) override path.
        "imageOutputsDirectory": "",
        "videoOutputsDirectory": "",
        # Override for the Hugging Face cache root (HF_HOME). Empty = use
        # the platform default (``~/.cache/huggingface`` on Linux/Mac,
        # ``%USERPROFILE%\.cache\huggingface`` on Windows). When set, the
        # Tauri shell injects this as HF_HOME before spawning the backend
        # so every downstream ``snapshot_download`` lands on the chosen
        # drive. Moving existing models between locations is handled by
        # the ``/api/settings/storage/move`` endpoint.
        "hfCachePath": "",
    }


def _normalize_model_directory_entry(entry: dict[str, Any], index: int) -> dict[str, Any]:
    raw_path = str(entry.get("path") or "").strip()
    label = str(entry.get("label") or "").strip()
    if not label:
        label = Path(os.path.expanduser(raw_path or f"directory-{index + 1}")).name or f"Directory {index + 1}"
    directory_id = _normalize_slug(str(entry.get("id") or label), f"directory-{index + 1}")
    return {
        "id": directory_id,
        "label": label,
        "path": raw_path,
        "enabled": bool(entry.get("enabled", True)),
        "source": "default" if str(entry.get("source") or "user") == "default" else "user",
    }


def _normalize_model_directories(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not str(entry.get("path") or "").strip():
            continue
        item = _normalize_model_directory_entry(entry, index)
        base_id = item["id"]
        suffix = 2
        while item["id"] in seen_ids:
            item["id"] = f"{base_id}-{suffix}"
            suffix += 1
        seen_ids.add(item["id"])
        normalized.append(item)
    return normalized


def _normalize_launch_preferences(payload: dict[str, Any] | None) -> dict[str, Any]:
    defaults = dict(DEFAULT_LAUNCH_PREFERENCES)
    if not isinstance(payload, dict):
        return defaults

    result = dict(defaults)

    # Migrate legacy TurboQuant fields to the new cache strategy model.
    if "useTurboQuant" in payload and "cacheStrategy" not in payload:
        result["cacheStrategy"] = "native"
    if "turboQuantBits" in payload and "cacheBits" not in payload:
        payload["cacheBits"] = payload["turboQuantBits"]

    integer_fields = {
        "contextTokens": (256, 2097152),
        "maxTokens": (1, 32768),
        "cacheBits": (0, 8),
        "fp16Layers": (0, 16),
    }
    for key, (minimum, maximum) in integer_fields.items():
        if key in payload:
            try:
                result[key] = max(minimum, min(maximum, int(payload[key])))
            except (TypeError, ValueError):
                pass

    if result.get("maxTokens", 0) < defaults["maxTokens"]:
        result["maxTokens"] = defaults["maxTokens"]

    if "temperature" in payload:
        try:
            result["temperature"] = max(0.0, min(2.0, float(payload["temperature"])))
        except (TypeError, ValueError):
            pass

    for key in ("fusedAttention", "fitModelInMemory"):
        if key in payload:
            result[key] = bool(payload[key])

    if "cacheStrategy" in payload:
        val = str(payload["cacheStrategy"]).strip()
        result["cacheStrategy"] = val if val else "native"

    return result


def _load_settings(path: Path, default_port: int, data_dir: Path) -> dict[str, Any]:
    settings = _default_settings(default_port, data_dir)
    if not path.exists():
        return settings

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings

    model_directories = payload.get("modelDirectories")
    if isinstance(model_directories, list):
        normalized = _normalize_model_directories(model_directories)
        settings["modelDirectories"] = normalized

    try:
        preferred_port = int(payload.get("preferredServerPort", default_port))
        settings["preferredServerPort"] = max(1024, min(65535, preferred_port))
    except (TypeError, ValueError):
        settings["preferredServerPort"] = default_port

    settings["allowRemoteConnections"] = bool(payload.get("allowRemoteConnections", False))
    # Default True: if the key is missing from an older settings.json we
    # preserve the secure default rather than silently opening the API.
    settings["requireApiAuth"] = bool(payload.get("requireApiAuth", True))
    settings["autoStartServer"] = bool(payload.get("autoStartServer", False))

    settings["launchPreferences"] = _normalize_launch_preferences(payload.get("launchPreferences"))

    hf_token = payload.get("huggingFaceToken")
    if isinstance(hf_token, str):
        settings["huggingFaceToken"] = hf_token
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    for key in ("imageOutputsDirectory", "videoOutputsDirectory", "hfCachePath"):
        raw = payload.get(key)
        if isinstance(raw, str):
            settings[key] = raw.strip()

    return settings


def _save_settings(settings: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(str(tmp), str(path))
