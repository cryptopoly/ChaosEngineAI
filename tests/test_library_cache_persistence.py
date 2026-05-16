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
        # The vanished-entries filter (bug 1) prunes cached items whose
        # path is gone, so use a path that actually exists on disk.
        on_disk = Path(self.tmp.name) / "from-disk"
        on_disk.mkdir()
        entry = {"name": "from-disk", "path": str(on_disk)}

        with mock.patch(
            "backend_service.state._discover_local_models",
            return_value=[entry],
        ):
            first = ChaosEngineState(**self.kwargs)
            self.assertTrue(first._library_scan_done.wait(2.0))

        scan_calls = {"count": 0}

        def counting_scan(directories):
            scan_calls["count"] += 1
            return [entry]

        with mock.patch(
            "backend_service.state._discover_local_models",
            side_effect=counting_scan,
        ):
            second = ChaosEngineState(**self.kwargs)
            self.assertTrue(second._library_scan_done.is_set())
            workspace = second.workspace()
            self.assertEqual(workspace["libraryStatus"], "ready")
            self.assertEqual(len(workspace["library"]), 1)

    def test_library_filters_vanished_entries_on_read(self):
        # Bug 1: an entry whose path was removed on disk after the last
        # scan must disappear from ``_library()`` immediately without
        # requiring a fresh full rescan. A background rescan is kicked
        # so the persisted cache catches up, but the response returned
        # right now already excludes the stale entry.
        on_disk = Path(self.tmp.name) / "stage"
        on_disk.mkdir()
        real_path = on_disk / "owner" / "alive"
        real_path.mkdir(parents=True)
        gone_path = on_disk / "owner" / "gone"
        gone_path.mkdir(parents=True)

        entries = [
            {"name": "owner/alive", "path": str(real_path)},
            {"name": "owner/gone", "path": str(gone_path)},
        ]

        with mock.patch(
            "backend_service.state._discover_local_models",
            return_value=entries,
        ):
            state = ChaosEngineState(**self.kwargs)
            self.assertTrue(state._library_scan_done.wait(2.0))

        # Simulate the gone path being removed after the cache scan.
        import shutil

        shutil.rmtree(gone_path)

        # Patch the rescan so the kick doesn't immediately re-add the
        # stale entry (we want to confirm the per-request filter).
        with mock.patch(
            "backend_service.state._discover_local_models",
            return_value=[{"name": "owner/alive", "path": str(real_path)}],
        ):
            names = [item["name"] for item in state._library()]

        self.assertIn("owner/alive", names)
        self.assertNotIn("owner/gone", names)

    def test_workspace_excludes_vanished_library_entries(self):
        # End-to-end variant of the bug 1 fix: ``/api/workspace`` consumers
        # see the filtered view because ``workspace()`` calls ``_library()``.
        on_disk = Path(self.tmp.name) / "stage"
        on_disk.mkdir()
        real_path = on_disk / "owner" / "alive"
        real_path.mkdir(parents=True)
        gone_path = on_disk / "owner" / "gone"
        gone_path.mkdir(parents=True)

        with mock.patch(
            "backend_service.state._discover_local_models",
            return_value=[
                {"name": "owner/alive", "path": str(real_path)},
                {"name": "owner/gone", "path": str(gone_path)},
            ],
        ):
            state = ChaosEngineState(**self.kwargs)
            self.assertTrue(state._library_scan_done.wait(2.0))

        import shutil

        shutil.rmtree(gone_path)

        with mock.patch(
            "backend_service.state._discover_local_models",
            return_value=[{"name": "owner/alive", "path": str(real_path)}],
        ):
            workspace = state.workspace()

        names = [item["name"] for item in workspace["library"]]
        self.assertIn("owner/alive", names)
        self.assertNotIn("owner/gone", names)

    def test_find_library_entry_prefers_healthy_over_broken(self):
        # Bug 2 helper: two entries with the same ``name`` (different
        # paths) — one marked broken, one healthy. The lookup must
        # return the healthy one, regardless of insertion order.
        on_disk = Path(self.tmp.name) / "stage"
        on_disk.mkdir()
        healthy = on_disk / "real" / "owner" / "name"
        healthy.mkdir(parents=True)
        broken = on_disk / "stub" / "owner" / "name"
        broken.mkdir(parents=True)

        entries = [
            {
                "name": "owner/name",
                "path": str(broken),
                "broken": True,
                "brokenReason": "stub",
            },
            {
                "name": "owner/name",
                "path": str(healthy),
                "broken": False,
                "brokenReason": None,
            },
        ]

        with mock.patch(
            "backend_service.state._discover_local_models",
            return_value=entries,
        ):
            state = ChaosEngineState(**self.kwargs)
            self.assertTrue(state._library_scan_done.wait(2.0))

        found = state._find_library_entry(None, "owner/name")
        self.assertIsNotNone(found)
        self.assertEqual(found["path"], str(healthy))
        self.assertFalse(found.get("broken"))


if __name__ == "__main__":
    unittest.main()
