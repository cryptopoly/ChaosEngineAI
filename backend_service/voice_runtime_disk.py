"""What the voice runtime holds on disk, removing an add-on, and removing the runtime (D-73 M4, M5).

The runtime is the voice venv (``voice_env``) plus its own small downloads: the Kokoro quick voices (an add-on since
D-75), the turn detector and the speaker-label models. ``pack_parts`` lists those four with their size (Kokoro 0 when
left out); ``remove_extra`` unticks an add-on (pip uninstall of the add-on's own packages, or Kokoro's files through
the checked path helpers, and a marker edit); ``remove_runtime`` deletes the venv, and the
runtime's own downloads only when asked. Downloaded dictation and speech models, cloned voices and the Library stay.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from backend_service import voice_env

_PARTS_CACHE: dict[str, Any] = {}


def _tree_bytes(folder: Path) -> int:
    """Bytes of the files under ``folder`` (symlinks not followed)."""
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def _repo_bytes(repo: str) -> int:
    from backend_service.helpers.discovery import _path_size_bytes  # noqa: PLC0415
    from backend_service.helpers.hf_cache_paths import _hf_repo_cache_dir  # noqa: PLC0415

    folder = _hf_repo_cache_dir(repo)
    try:
        return _path_size_bytes(folder) if folder.exists() else 0
    except OSError:
        return 0


def pack_parts() -> list[dict[str, Any]]:
    """The runtime's four parts with their size on disk: the speech packages, the Kokoro quick voices, turn detection
    and speaker labels. Measured once per install (the packages folder holds tens of thousands of files)."""
    if not voice_env.is_installed():
        return []
    try:
        stamp = f"{voice_env.lane()}:{voice_env._MARKER.stat().st_mtime_ns}"
    except OSError:
        stamp = voice_env.lane()
    if _PARTS_CACHE.get("stamp") == stamp:
        return [dict(part) for part in _PARTS_CACHE["parts"]]
    if voice_env.lane() == "mlx":
        kokoro = _repo_bytes(voice_env.KOKORO_MLX_REPO)
    else:
        kokoro = _tree_bytes(voice_env.kokoro_onnx_dir()) if voice_env.kokoro_onnx_dir().exists() else 0
    sizes = {
        "packages": _tree_bytes(voice_env.VOICE_VENV_DIR),
        "kokoro": kokoro,
        "turns": _repo_bytes(voice_env.VAD_REPO),
        "speakers": sum(_repo_bytes(repo) for repo, _rev, _file in voice_env.DIARIZE_FILES.values()),
    }
    parts = [{"id": name, "sizeGb": round(size / 1e9, 3)} for name, size in sizes.items()]
    _PARTS_CACHE.update(stamp=stamp, parts=parts)
    return [dict(part) for part in parts]


# What unticking an add-on uninstalls. The Mandarin add-on is misaki's ``zh`` extra: its own requirements are read from
# misaki's metadata in the venv, so misaki itself (needed for English) stays. The speech GPU add-on: only the packages
# nothing else in the runtime uses (NOT VERIFIED on Windows or Linux, like the add-on itself).
_SPEECH_GPU_REMOVE = [
    "torch", "torchaudio", "torchcodec", "chatterbox-tts", "voxcpm", "diffusers", "s3tokenizer", "resemble-perth",
    "conformer",
]
_REQUIREMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _extra_requirements(package: str, extra: str) -> list[str]:
    """Requirement names a package's ``extra`` adds, from its METADATA in the venv."""
    names: set[str] = set()
    patterns = (f"lib/python3*/site-packages/{package}-*.dist-info/METADATA", f"Lib/site-packages/{package}-*.dist-info/METADATA")
    for pattern in patterns:
        for metadata in voice_env.VOICE_VENV_DIR.glob(pattern):
            try:
                lines = metadata.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.startswith("Requires-Dist:") or f'extra == "{extra}"' not in line.replace("'", '"'):
                    continue
                match = _REQUIREMENT_NAME.match(line.split(":", 1)[1].split(";", 1)[0].strip())
                if match and match.group(0).lower() != package:
                    names.add(match.group(0))
    return sorted(names)


def extra_packages_to_remove(name: str) -> list[str]:
    if name == "chinese":
        return _extra_requirements("misaki", "zh")
    if name == "speech-gpu":
        return list(_SPEECH_GPU_REMOVE)
    return []


def _kokoro_onnx_folder() -> Path:
    """``voice_env.kokoro_onnx_dir()``, checked before it is deleted: resolved, named ``kokoro`` and directly in the
    voice data folder."""
    data = voice_env.VOICE_DATA_DIR.resolve()
    folder = voice_env.kokoro_onnx_dir().resolve()
    if folder.parent != data or folder.name != "kokoro":
        raise RuntimeError(f"Refusing to delete {folder}.")
    return folder


def remove_kokoro_files() -> tuple[list[str], int]:
    """Delete the Kokoro quick voices of this lane: the MLX repo in the Hugging Face cache, or the kokoro-onnx folder.
    Returns what was removed and the bytes freed. The caller has checked that voice is idle and stopped the worker."""
    from backend_service.voice_model_files import repo_cache_dir  # noqa: PLC0415

    removed: list[str] = []
    freed = 0
    if voice_env.lane() == "mlx":
        target = repo_cache_dir(voice_env.KOKORO_MLX_REPO)
        if target.exists():
            freed += _repo_bytes(voice_env.KOKORO_MLX_REPO)
            shutil.rmtree(target)
            removed.append(voice_env.KOKORO_MLX_REPO)
    else:
        folder = _kokoro_onnx_folder()
        if folder.exists():
            freed += _tree_bytes(folder)
            shutil.rmtree(folder)
            removed.append("kokoro")
    _PARTS_CACHE.clear()
    return removed, freed


def remove_extra(name: str) -> dict[str, Any]:
    """Untick an add-on: uninstall its packages from the venv (Kokoro: delete its files) and drop it from the marker.
    Raises ValueError for an unknown or not-installed add-on, RuntimeError while an install or a voice job runs or
    when the uninstall fails."""
    from backend_service import voice_client  # noqa: PLC0415
    from backend_service.voice_model_files import voice_busy_reason  # noqa: PLC0415

    if name not in voice_env.EXTRAS:
        raise ValueError(f"Unknown add-on: {name}.")
    if name not in voice_env.installed_extras():
        raise ValueError(f"{voice_env.EXTRAS[name][2]} is not installed.")
    if voice_env.job_status()["running"]:
        raise RuntimeError("The voice runtime is installing. Wait for it to finish.")
    reason = voice_busy_reason()
    if reason:
        raise RuntimeError(reason)
    voice_client.stop_speech_worker()  # it may have the add-on's packages imported (or Kokoro loaded)
    if name == "kokoro":
        removed, freed = remove_kokoro_files()
        _drop_from_marker(name)
        return {"removed": name, "packages": [], "files": removed, "freedGb": round(freed / 1e9, 2)}
    packages = extra_packages_to_remove(name)
    if packages:
        result = subprocess.run(
            [str(voice_env.venv_python()), "-m", "pip", "uninstall", "-y", *packages],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=voice_env.worker_env(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Removing {voice_env.EXTRAS[name][2]} failed: {result.stdout[-400:]}{result.stderr[-400:]}")
    if name == "speech-gpu":
        remove_qwen_site()
    _drop_from_marker(name)
    return {"removed": name, "packages": packages}


def remove_qwen_site() -> None:
    """Delete the folder with Qwen3-TTS's own transformers, checked first: directly inside the voice venv and named
    ``qwen-site`` (the add-on's packages are uninstalled by pip, this folder is not one of them)."""
    folder = voice_env.qwen_site_dir()
    if folder.parent != voice_env.VOICE_VENV_DIR or folder.name != "qwen-site":
        raise RuntimeError(f"Refusing to delete {folder}.")
    if folder.is_dir():
        shutil.rmtree(folder)


def _drop_from_marker(name: str) -> None:
    marker = voice_env.read_marker()
    if marker is None:
        return
    marker["extras"] = [extra for extra in marker.get("extras", []) if extra != name]
    voice_env._MARKER.write_text(json.dumps(marker), encoding="utf-8")


def _safe_venv_dir() -> Path:
    """The venv folder, checked before it is deleted: it must look like the voice venv (a ``pyvenv.cfg`` or the
    marker in it) and must not be the home folder, one of its parents or a filesystem root."""
    folder = voice_env.VOICE_VENV_DIR.resolve()
    home = Path.home().resolve()
    if folder == Path(folder.anchor) or folder == home or home.is_relative_to(folder):
        raise RuntimeError(f"Refusing to delete {folder}.")
    if not ((folder / "pyvenv.cfg").is_file() or (folder / voice_env._MARKER.name).is_file()):
        raise RuntimeError(f"{folder} does not look like the voice runtime; nothing was deleted.")
    return folder


def remove_runtime(also_runtime_models: bool = False) -> dict[str, Any]:
    """Remove the voice runtime: stop the workers, delete the venv and its marker. Downloaded dictation and speech
    models stay; ``also_runtime_models`` also deletes the runtime's own downloads (Kokoro, turn detection, speaker
    labels). Cloned voices and the Library are never touched. Raises ValueError when it is not installed and
    RuntimeError while an install or a voice job runs."""
    from backend_service import voice_client  # noqa: PLC0415
    from backend_service.voice_model_files import repo_cache_dir, voice_busy_reason  # noqa: PLC0415

    if voice_env.job_status()["running"]:
        raise RuntimeError("The voice runtime is installing. Cancel the install first.")
    reason = voice_busy_reason()
    if reason:
        raise RuntimeError(reason)
    if not voice_env.VOICE_VENV_DIR.exists():
        raise ValueError("The voice runtime is not installed.")
    folder = _safe_venv_dir()
    voice_client.stop_all()
    freed = _tree_bytes(folder)
    shutil.rmtree(folder)
    removed_models: list[str] = []
    if also_runtime_models:
        kokoro_removed, kokoro_freed = remove_kokoro_files()
        removed_models += kokoro_removed
        freed += kokoro_freed
        repos = [voice_env.VAD_REPO, *(repo for repo, _rev, _file in voice_env.DIARIZE_FILES.values())]
        for repo in repos:
            target = repo_cache_dir(repo)
            if target.exists():
                freed += _repo_bytes(repo)
                shutil.rmtree(target)
                removed_models.append(repo)
    _PARTS_CACHE.clear()
    return {"removed": True, "freedGb": round(freed / 1e9, 2), "removedModels": removed_models}


def _source_is(source: str, repo: str) -> bool:
    return source == repo or f"models--{repo.replace('/', '--')}" in source


def in_memory() -> list[dict[str, Any]]:
    """The models the speech worker holds now: ``[{"lane": "tts" | "stt", "name", "peakGb"}]`` (``peakGb`` from the
    catalog's measured numbers, None when not measured). Empty when the worker is not running."""
    from backend_service import voice_client  # noqa: PLC0415
    from backend_service.catalog import voice_models as catalog  # noqa: PLC0415
    from backend_service.voice_runtime import list_stt_models  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    for lane_name, source in sorted(voice_client.loaded_sources().items()):
        name, peak = source.rstrip("/").rsplit("/", 1)[-1], None
        if source in (voice_env.KOKORO_MLX_REPO, str(voice_env.kokoro_onnx_dir())) or _source_is(source, voice_env.KOKORO_MLX_REPO):
            name, peak = "Kokoro", catalog.KOKORO_CHAT_ESTIMATES.get("chatPeakGb")
        elif lane_name == "tts":
            row = next((r for r in catalog.VOICE_MODELS if _source_is(source, r["repo"])), None)
            if row is not None:
                name, peak = row["name"], (row.get("estimates") or {}).get("peakGb")
        else:
            model = next((m for m in list_stt_models() if _source_is(source, str(m["id"]))), None)
            if model is not None:
                name = model["name"]
        rows.append({"lane": lane_name, "name": name, "peakGb": peak})
    return rows
