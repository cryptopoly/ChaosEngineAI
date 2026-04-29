"""Tests for the async library scan + lazy runtime properties."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.state import ChaosEngineState


def _fake_system_snapshot():
    return {
        "platform": "Darwin",
        "arch": "arm64",
        "hardwareSummary": "Apple Silicon / 48 GB unified memory",
        "backendLabel": "Python sidecar",
        "appVersion": "0.5.1",
        "mlxAvailable": False,
        "mlxLmAvailable": False,
        "mlxUsable": False,
        "ggufAvailable": False,
        "converterAvailable": False,
        "nativePython": "/tmp/python",
        "llamaServerPath": None,
        "llamaCliPath": None,
        "nativeRuntimeMessage": None,
        "totalMemoryGb": 48.0,
        "availableMemoryGb": 30.0,
        "usedMemoryGb": 18.0,
        "swapUsedGb": 0.0,
        "cpuUtilizationPercent": 12.0,
        "gpuUtilizationPercent": None,
        "spareHeadroomGb": 24.0,
        "runningLlmProcesses": [],
        "uptimeMinutes": 1.0,
    }


class AsyncLibraryScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(self.tmp.name)
        self.kwargs = dict(
            system_snapshot_provider=_fake_system_snapshot,
            settings_path=tmpdir / "settings.json",
            benchmarks_path=tmpdir / "benchmarks.json",
            chat_sessions_path=tmpdir / "chat-sessions.json",
            library_cache_path=tmpdir / "library_cache.json",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_library_provider_short_circuits_async_scan(self):
        provider_calls = []

        def provider():
            provider_calls.append(True)
            return [{"name": "fake/model", "path": "/tmp/fake"}]

        state = ChaosEngineState(library_provider=provider, **self.kwargs)
        self.assertTrue(state._library_scan_done.is_set())
        self.assertFalse(state._library_scan_started)
        result = state._library()
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(len(provider_calls), 1)

    def test_workspace_returns_scanning_status_until_scan_finishes(self):
        slow_event = mock.MagicMock()

        def slow_scan(directories):
            slow_event.calls += 1
            return [{"name": "slow/entry", "path": "/tmp/slow"}]

        slow_event.calls = 0
        with mock.patch(
            "backend_service.state._discover_local_models",
            side_effect=slow_scan,
        ):
            state = ChaosEngineState(**self.kwargs)
            try:
                self.assertTrue(state._library_scan_done.wait(2.0))
                workspace = state.workspace()
                self.assertEqual(workspace["libraryStatus"], "ready")
                self.assertEqual(len(workspace["library"]), 1)
            finally:
                state._library_scan_done.set()

    def test_kick_scan_is_idempotent(self):
        scan_calls = []

        def fake_scan(directories):
            scan_calls.append(directories)
            return []

        with mock.patch(
            "backend_service.state._discover_local_models",
            side_effect=fake_scan,
        ):
            state = ChaosEngineState(**self.kwargs)
            self.assertTrue(state._library_scan_done.wait(2.0))
            initial_calls = len(scan_calls)
            state._kick_library_scan()
            state._kick_library_scan()
            state._library_scan_done.wait(2.0)
            self.assertGreaterEqual(len(scan_calls), initial_calls)


class LazyRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(self.tmp.name)
        self.kwargs = dict(
            system_snapshot_provider=_fake_system_snapshot,
            library_provider=lambda: [],
            settings_path=tmpdir / "settings.json",
            benchmarks_path=tmpdir / "benchmarks.json",
            chat_sessions_path=tmpdir / "chat-sessions.json",
            library_cache_path=tmpdir / "library_cache.json",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_runtimes_unconstructed_after_init(self):
        state = ChaosEngineState(**self.kwargs)
        self.assertIsNone(state._image_runtime)
        self.assertIsNone(state._video_runtime)

    def test_runtime_setter_keeps_test_compat(self):
        state = ChaosEngineState(**self.kwargs)
        marker = object()
        state.image_runtime = marker
        self.assertIs(state.image_runtime, marker)
        state.video_runtime = marker
        self.assertIs(state.video_runtime, marker)


if __name__ == "__main__":
    unittest.main()
