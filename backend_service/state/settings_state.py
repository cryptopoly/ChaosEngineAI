"""Settings payload + update for ``ChaosEngineState``.

Two helpers lifted out of ``state/__init__.py``:

* ``settings_payload`` — render the user-visible settings shape with
  per-directory model counts, masked API keys / HF token, and the
  resolved data + image + video output directory paths.
* ``update_settings`` — apply a settings patch: normalise model
  directories, validate output-path overrides (must be absolute or
  ``~``-relative — never bare relative), migrate the data directory
  when changed, persist remote-provider entries while preserving
  existing API keys when only metadata changed, refresh the library
  cache, and return ``{"settings": ..., "restartRequired"?, "migrationSummary"?}``.

Both take the ``ChaosEngineState`` instance as their first argument.
The class methods become thin wrappers.

Extracted as part of the v0.8.0 Phase 1a-8 refactor.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from backend_service.helpers.settings import (
    _migrate_data_directory,
    _normalize_launch_preferences,
    _normalize_model_directories,
    _save_data_location,
)
from backend_service.models import UpdateSettingsRequest
from backend_service.state._helpers import _normalize_remote_provider_api_base


if TYPE_CHECKING:
    from backend_service.state import ChaosEngineState


def settings_payload(
    state: ChaosEngineState, library: list[dict[str, Any]]
) -> dict[str, Any]:
    from backend_service.app import DATA_LOCATION

    model_counts: dict[str, int] = {}
    for item in library:
        directory_id = item.get("directoryId")
        if not directory_id:
            continue
        model_counts[directory_id] = model_counts.get(directory_id, 0) + 1

    directories: list[dict[str, Any]] = []
    for directory in state.settings["modelDirectories"]:
        expanded = Path(os.path.expanduser(str(directory.get("path") or ""))).expanduser()
        directories.append(
            {
                **directory,
                "exists": expanded.exists(),
                "modelCount": model_counts.get(directory["id"], 0),
            }
        )

    # Mask API keys when returning to the frontend
    remote_providers = state.settings.get("remoteProviders") or []
    masked_providers = []
    for p in remote_providers:
        api_key = p.get("apiKey", "")
        masked_providers.append({
            "id": p.get("id"),
            "label": p.get("label"),
            "apiBase": p.get("apiBase"),
            "model": p.get("model"),
            "hasApiKey": bool(api_key),
            "apiKeyMasked": ("•" * 8 + api_key[-4:]) if len(api_key) > 4 else "",
        })

    hf_token_value = str(state.settings.get("huggingFaceToken") or "")
    if len(hf_token_value) > 4:
        hf_token_masked = "•" * 8 + hf_token_value[-4:]
    else:
        hf_token_masked = ""

    return {
        "modelDirectories": directories,
        "preferredServerPort": state.settings["preferredServerPort"],
        "allowRemoteConnections": bool(state.settings.get("allowRemoteConnections", False)),
        "requireApiAuth": bool(state.settings.get("requireApiAuth", True)),
        "autoStartServer": bool(state.settings.get("autoStartServer", False)),
        "launchPreferences": state._launch_preferences(),
        "remoteProviders": masked_providers,
        "huggingFaceToken": hf_token_masked,
        "hasHuggingFaceToken": bool(hf_token_value),
        "dataDirectory": str(DATA_LOCATION.data_dir),
        # Per-modality output overrides (empty == use default under
        # dataDirectory). The frontend uses these to render the picker
        # value and the resolved path used for new artifacts.
        "imageOutputsDirectory": str(state.settings.get("imageOutputsDirectory") or ""),
        "videoOutputsDirectory": str(state.settings.get("videoOutputsDirectory") or ""),
        # Hugging Face cache root override. Empty string means "use the
        # platform default" — the frontend renders the resolved path
        # alongside the override input so users always know where
        # models are actually landing.
        "hfCachePath": str(state.settings.get("hfCachePath") or ""),
        "favoriteModelRefs": list(state.settings.get("favoriteModelRefs") or []),
    }


def update_settings(
    state: ChaosEngineState, request: UpdateSettingsRequest
) -> dict[str, Any]:
    """Returns ``{"settings": ..., "restartRequired"?: bool, "migrationSummary"?: dict}``."""
    from backend_service.app import (
        DATA_LOCATION,
        DEFAULT_HOST,
        _default_settings,
        _save_settings,
    )

    with state._lock:
        next_settings = _default_settings()
        next_settings["modelDirectories"] = [
            dict(entry) for entry in state.settings["modelDirectories"]
        ]
        next_settings["preferredServerPort"] = state.settings["preferredServerPort"]
        next_settings["allowRemoteConnections"] = bool(
            state.settings.get("allowRemoteConnections", False)
        )
        next_settings["requireApiAuth"] = bool(state.settings.get("requireApiAuth", True))
        next_settings["launchPreferences"] = state._launch_preferences()
        next_settings["remoteProviders"] = list(state.settings.get("remoteProviders") or [])
        next_settings["huggingFaceToken"] = str(state.settings.get("huggingFaceToken") or "")
        next_settings["imageOutputsDirectory"] = str(
            state.settings.get("imageOutputsDirectory") or ""
        )
        next_settings["videoOutputsDirectory"] = str(
            state.settings.get("videoOutputsDirectory") or ""
        )
        next_settings["hfCachePath"] = str(state.settings.get("hfCachePath") or "")
        next_settings["favoriteModelRefs"] = list(
            state.settings.get("favoriteModelRefs") or []
        )

        if request.modelDirectories is not None:
            next_settings["modelDirectories"] = _normalize_model_directories(
                [entry.model_dump() for entry in request.modelDirectories]
            )
        if request.preferredServerPort is not None:
            next_settings["preferredServerPort"] = request.preferredServerPort
        if request.allowRemoteConnections is not None:
            next_settings["allowRemoteConnections"] = request.allowRemoteConnections
        if request.requireApiAuth is not None:
            next_settings["requireApiAuth"] = request.requireApiAuth
        if request.autoStartServer is not None:
            next_settings["autoStartServer"] = request.autoStartServer
        if request.launchPreferences is not None:
            next_settings["launchPreferences"] = _normalize_launch_preferences(
                request.launchPreferences.model_dump()
            )
        if request.remoteProviders is not None:
            existing_by_id = {
                p.get("id"): p for p in (state.settings.get("remoteProviders") or [])
            }
            normalized = []
            for provider in request.remoteProviders:
                api_base = _normalize_remote_provider_api_base(provider.apiBase)
                api_key = provider.apiKey.strip()
                existing_provider = existing_by_id.get(provider.id) or {}
                existing_api_base = str(existing_provider.get("apiBase") or "").strip().rstrip("/")
                existing_api_key = str(existing_provider.get("apiKey") or "")
                if not api_key and provider.id in existing_by_id:
                    if not existing_api_key:
                        api_key = ""
                    elif existing_api_base == api_base:
                        api_key = existing_api_key
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Provider {provider.id} changed its API base. "
                                "Re-enter the API key before saving this change."
                            ),
                        )
                normalized.append({
                    "id": provider.id,
                    "label": provider.label,
                    "apiBase": api_base,
                    "apiKey": api_key,
                    "model": provider.model,
                })
            next_settings["remoteProviders"] = normalized

        if request.huggingFaceToken is not None:
            previous_token_value = str(next_settings.get("huggingFaceToken") or "")
            token_value = request.huggingFaceToken.strip()
            next_settings["huggingFaceToken"] = token_value
            if token_value:
                os.environ["HF_TOKEN"] = token_value
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token_value
            else:
                os.environ.pop("HF_TOKEN", None)
                os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            if token_value != previous_token_value:
                from backend_service.helpers.huggingface import _clear_huggingface_caches
                from backend_service.helpers.images import _clear_image_discover_caches

                _clear_huggingface_caches()
                _clear_image_discover_caches()

        # Output directory overrides. Empty string clears the override.
        # Anything non-empty must be absolute or ~-relative — same rule as
        # dataDirectory — so we don't silently end up writing artifacts to
        # the working directory of whoever launched the backend.
        for field_name, label in (
            ("imageOutputsDirectory", "imageOutputsDirectory"),
            ("videoOutputsDirectory", "videoOutputsDirectory"),
            ("hfCachePath", "hfCachePath"),
        ):
            raw_value = getattr(request, field_name, None)
            if raw_value is None:
                continue
            cleaned = raw_value.strip()
            # Accept Windows-style absolute paths (``D:\...``) alongside
            # POSIX and ``~``-relative paths. Bare relative paths are
            # rejected — silently writing artifacts to the backend's cwd
            # is exactly the "where did my models go?" class of bug we
            # want to avoid.
            if cleaned and not (
                cleaned.startswith("/")
                or cleaned.startswith("~")
                or (len(cleaned) >= 2 and cleaned[1] == ":")
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"{label} must be an absolute path or start with ~.",
                )
            next_settings[field_name] = cleaned

        if request.favoriteModelRefs is not None:
            seen: set[str] = set()
            cleaned_favs: list[str] = []
            for raw_ref in request.favoriteModelRefs:
                ref = str(raw_ref or "").strip()
                if not ref or ref in seen:
                    continue
                seen.add(ref)
                cleaned_favs.append(ref)
            next_settings["favoriteModelRefs"] = cleaned_favs

        data_migration: dict[str, Any] | None = None
        restart_required_for_data_dir = False
        if request.dataDirectory is not None:
            raw_dir = request.dataDirectory.strip()
            if raw_dir:
                if not (raw_dir.startswith("/") or raw_dir.startswith("~")):
                    raise HTTPException(
                        status_code=400,
                        detail="dataDirectory must be an absolute path or start with ~.",
                    )
                new_dir = Path(os.path.expanduser(raw_dir)).resolve()
                if new_dir != DATA_LOCATION.data_dir:
                    try:
                        data_migration = _migrate_data_directory(
                            DATA_LOCATION.data_dir, new_dir
                        )
                        _save_data_location(new_dir)
                        restart_required_for_data_dir = True
                    except RuntimeError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc

        _save_settings(next_settings, state._settings_path)
        state.settings = next_settings
        state._library_cache = None
        library = state._library(force=True)

        state.add_log(
            "settings",
            "info",
            (
                f"Saved settings with {len(state.settings['modelDirectories'])} model "
                f"directories, preferred API port {state.settings['preferredServerPort']}, "
                f"and remote access {'enabled' if state.settings['allowRemoteConnections'] else 'disabled'}."
            ),
        )
        if state.settings["preferredServerPort"] != state.server_port:
            state.add_log(
                "server",
                "info",
                (
                    f"Preferred API port changed to {state.settings['preferredServerPort']}. "
                    "Restart the API service to apply it."
                ),
            )
        if bool(state.settings.get("allowRemoteConnections", False)) != (DEFAULT_HOST != "127.0.0.1"):
            state.add_log(
                "server",
                "info",
                "Remote connection setting changed. Restart the API service to apply the new bind mode.",
            )
        state.add_activity(
            "Settings updated",
            (
                f"{len(library)} models discovered across "
                f"{len(state.settings['modelDirectories'])} configured directories."
            ),
        )
        payload = state._settings_payload(library)
        response: dict[str, Any] = {"settings": payload}
        if restart_required_for_data_dir:
            response["restartRequired"] = True
        if data_migration is not None:
            response["migrationSummary"] = data_migration
        return response
