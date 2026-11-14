"""Round trip of the `strongControlBorders` accessibility setting (default true, false persists, missing key reads true)."""

import json
import tempfile
import unittest
from pathlib import Path

from backend_service.state import ChaosEngineState
from tests.test_backend_service import FakeRuntime, fake_library, fake_system_snapshot, make_test_client


class StrongControlBordersTests(unittest.TestCase):
    def _client(self, settings_path: Path):
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=settings_path,
            benchmarks_path=settings_path.parent / "benchmark-history.json",
            chat_sessions_path=settings_path.parent / "chat-sessions.json",
        )
        state.runtime = FakeRuntime()
        return make_test_client(state)

    def test_default_true_false_persists_and_unrelated_update_keeps_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            client = self._client(path)
            self.assertIs(client.get("/api/settings").json()["settings"]["strongControlBorders"], True)
            off = client.patch("/api/settings", json={"strongControlBorders": False}).json()["settings"]
            self.assertIs(off["strongControlBorders"], False)
            self.assertIs(json.loads(path.read_text())["strongControlBorders"], False)
            kept = client.patch("/api/settings", json={"preferredServerPort": 8898}).json()["settings"]
            self.assertIs(kept["strongControlBorders"], False)
            # A fresh backend on the same file still reads false; then back on.
            again = self._client(path)
            self.assertIs(again.get("/api/settings").json()["settings"]["strongControlBorders"], False)
            on = again.patch("/api/settings", json={"strongControlBorders": True}).json()["settings"]
            self.assertIs(on["strongControlBorders"], True)

    def test_missing_key_reads_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({"preferredServerPort": 8899}))
            self.assertIs(self._client(path).get("/api/settings").json()["settings"]["strongControlBorders"], True)


if __name__ == "__main__":
    unittest.main()
