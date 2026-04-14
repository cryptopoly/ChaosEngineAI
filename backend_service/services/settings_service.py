"""Settings management service."""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any


class SettingsService:
    """Thin wrapper providing organized access to settings on ChaosEngineState.

    This does NOT duplicate the settings dict -- it references the one owned
    by the underlying state object so that existing code keeps working.
    """

    def __init__(self, state: Any) -> None:
        self._state = state
        self._lock = RLock()

    # -- Read helpers --------------------------------------------------------

    @property
    def settings(self) -> dict[str, Any]:
        """Return the live settings dict (same reference as state.settings)."""
        return self._state.settings

    def get_settings(self) -> dict[str, Any]:
        """Return a shallow copy of the current settings."""
        with self._lock:
            return dict(self._state.settings)

    @property
    def model_directories(self) -> list[dict[str, Any]]:
        return self._state.settings.get("modelDirectories", [])

    @property
    def launch_preferences(self) -> dict[str, Any]:
        return self._state.settings.get("launchPreferences", {})

    @property
    def remote_providers(self) -> list[dict[str, Any]]:
        return self._state.settings.get("remoteProviders", [])

    @property
    def hf_token(self) -> str:
        return self._state.settings.get("huggingFaceToken", "")

    # -- Write helpers -------------------------------------------------------

    def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Merge *patch* into current settings and persist.

        Returns the updated settings dict.
        """
        with self._lock:
            self._state.settings.update(patch)
            self._persist()
            return dict(self._state.settings)

    def load_settings(self) -> dict[str, Any]:
        """Reload settings from disk via the state's path."""
        from backend_service.app import _load_settings

        with self._lock:
            self._state.settings = _load_settings(self._state._settings_path)
            return dict(self._state.settings)

    def save_settings(self) -> None:
        """Persist current settings to disk."""
        with self._lock:
            self._persist()

    # -- Internal ------------------------------------------------------------

    def _persist(self) -> None:
        path: Path = self._state._settings_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._state.settings, indent=2, default=str),
            encoding="utf-8",
        )
