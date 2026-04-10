"""Service layer: organized domain access over ChaosEngineState.

``ServiceCoordinator`` is a backward-compatible facade -- it holds the
original state object and exposes domain-specific sub-services for new
features, while delegating any attribute not found on itself to the
underlying state.
"""
from __future__ import annotations

from typing import Any

from backend_service.services.log_service import LogService
from backend_service.services.settings_service import SettingsService
from backend_service.services.download_service import DownloadService


class ServiceCoordinator:
    """Facade over the existing ChaosEngineState that provides
    domain-organized access for new features while maintaining
    backward compatibility."""

    def __init__(self, state: Any) -> None:
        # Keep a reference to the original monolithic state
        self._state = state

        # Domain sub-services
        self.log = LogService()
        self.settings_svc = SettingsService(state)
        self.downloads = DownloadService(state)

    # -- Backward-compatible property delegation ----------------------------

    @property
    def runtime(self) -> Any:
        return self._state.runtime

    @property
    def settings(self) -> dict[str, Any]:
        return self._state.settings

    @settings.setter
    def settings(self, value: dict[str, Any]) -> None:
        self._state.settings = value

    @property
    def chat_sessions(self) -> list[dict[str, Any]]:
        return self._state.chat_sessions

    @chat_sessions.setter
    def chat_sessions(self, value: list[dict[str, Any]]) -> None:
        self._state.chat_sessions = value

    @property
    def benchmark_runs(self) -> list[dict[str, Any]]:
        return self._state.benchmark_runs

    @benchmark_runs.setter
    def benchmark_runs(self, value: list[dict[str, Any]]) -> None:
        self._state.benchmark_runs = value

    @property
    def server_port(self) -> int:
        return self._state.server_port

    @server_port.setter
    def server_port(self, value: int) -> None:
        self._state.server_port = value

    @property
    def active_requests(self) -> int:
        return self._state.active_requests

    @active_requests.setter
    def active_requests(self, value: int) -> None:
        self._state.active_requests = value

    @property
    def requests_served(self) -> int:
        return self._state.requests_served

    @requests_served.setter
    def requests_served(self, value: int) -> None:
        self._state.requests_served = value

    # -- Catch-all delegation -----------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Forward any attribute not explicitly defined to the underlying state."""
        return getattr(self._state, name)
