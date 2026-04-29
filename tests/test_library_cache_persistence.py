"""Tests for the disk-backed library cache."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers.persistence import (
    LIBRARY_CACHE_VERSION,
    _library_cache_fingerprint,
    _load_library_cache,
    _save_library_cache,
)
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


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_skips_disabled_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir()
            (root / "b").mkdir()
            fingerprint = _library_cache_fingerprint([
                {"path": str(root / "a"), "enabled": True},
                {"path": str(root / "b"), "enabled": False},
            ])
            self.assertIn(str(root / "a"), fingerprint)
            self.assertNotIn(str(root / "b"), fingerprint)

    def test_fingerprint_handles_missing_directory(self):
        fingerprint = _library_cache_fingerprint([
            {"path": "/does/not/exist", "enabled": True},
        ])
        self.assertEqual(fingerprint["/does/not/exist"], 0.0)

    def test_fingerprint_changes_when_child_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            entries = [{"path": str(root), "enabled": True}]
            before = _library_cache_fingerprint(entries)
            time.sleep(0.05)
            (root / "new-model.gguf").write_bytes(b"x")
            after = _library_cache_fingerprint(entries)
            self.assertNotEqual(before, after)


class SaveLoadTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            items = [{"name": "foo", "path": "/tmp/foo"}]
            fingerprint = {"/tmp/foo": 1700000000.0}
            _save_library_cache(items, fingerprint, path)
            payload = _load_library_cache(path)
            self.assertIsNotNone(payload)
            self.assertEqual(payload["version"], LIBRARY_CACHE_VERSION)
            self.assertEqual(payload["items"], items)
            self.assertEqual(payload["fingerprint"], fingerprint)

    def test_load_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_load_library_cache(Path(tmp) / "absent.json"))

    def test_load_rejects_version_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            path.write_text(
                json.dumps({"version": 999, "fingerprint": {}, "items": []}),
                encoding="utf-8",
            )
            self.assertIsNone(_load_library_cache(path))

    def test_load_rejects_corrupt_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(_load_library_cache(path))


class StateIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(self.tmp.name)
        self.cache_path = tmpdir / "library_cache.json"
        self.kwargs = dict(
            system_snapshot_provider=_fake_system_snapshot,
            settings_path=tmpdir / "settings.json",
            benchmarks_path=tmpdir / "benchmarks.json",
            chat_sessions_path=tmpdir / "chat-sessions.json",
            library_cache_path=self.cache_path,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_scan_writes_cache_to_disk(self):
        with mock.patch(
            "backend_service.state._discover_local_models",
            return_value=[{"name": "alpha/beta", "path": "/tmp/alpha"}],
        ):
            state = ChaosEngineState(**self.kwargs)
            self.assertTrue(state._library_scan_done.wait(2.0))
        self.assertTrue(self.cache_path.exists())
        payload = _load_library_cache(self.cache_path)
        self.assertIsNotNone(payload)
        self.assertEqual(len(payload["items"]), 1)

    def test_warm_start_uses_disk_cache(self):
        with mock.patch(
            "backend_service.state._discover_local_models",
            return_value=[{"name": "from-disk", "path": "/tmp/from-disk"}],
        ):
            first = ChaosEngineState(**self.kwargs)
            self.assertTrue(first._library_scan_done.wait(2.0))

        scan_calls = {"count": 0}

        def counting_scan(directories):
            scan_calls["count"] += 1
            return [{"name": "from-disk", "path": "/tmp/from-disk"}]

        with mock.patch(
            "backend_service.state._discover_local_models",
            side_effect=counting_scan,
        ):
            second = ChaosEngineState(**self.kwargs)
            self.assertTrue(second._library_scan_done.is_set())
            workspace = second.workspace()
            self.assertEqual(workspace["libraryStatus"], "ready")
            self.assertEqual(len(workspace["library"]), 1)


if __name__ == "__main__":
    unittest.main()
