"""Phase 3.7 tests for workspace registry."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend_service.helpers.workspaces import WorkspaceRegistry


class WorkspaceRegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self.registry = WorkspaceRegistry(
            tmp_path / "workspaces.json",
            tmp_path / "workspaces",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_starts_empty(self):
        self.assertEqual(self.registry.list_all(), [])

    def test_create_assigns_id_and_timestamps(self):
        ws = self.registry.create("Research", "Climate notes")
        self.assertIn("id", ws)
        self.assertEqual(ws["title"], "Research")
        self.assertEqual(ws["description"], "Climate notes")
        self.assertEqual(ws["documents"], [])
        self.assertIn("createdAt", ws)
        self.assertIn("updatedAt", ws)

    def test_create_makes_workspace_subdir(self):
        ws = self.registry.create("Research")
        self.assertTrue(self.registry.workspace_dir(ws["id"]).exists())

    def test_persists_across_instances(self):
        ws = self.registry.create("Research")
        # New instance reads the same file.
        registry2 = WorkspaceRegistry(self.registry._path, self.registry._dir)
        loaded = registry2.get(ws["id"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["title"], "Research")

    def test_update_changes_fields(self):
        ws = self.registry.create("Research")
        updated = self.registry.update(
            ws["id"], title="Climate research", description="Notes",
        )
        self.assertEqual(updated["title"], "Climate research")
        self.assertEqual(updated["description"], "Notes")

    def test_update_returns_none_for_missing(self):
        self.assertIsNone(self.registry.update("missing", title="X"))

    def test_delete_removes_entry_and_dir(self):
        ws = self.registry.create("Research")
        # Drop a file in the workspace dir to confirm cleanup.
        target_dir = self.registry.workspace_dir(ws["id"])
        (target_dir / "doc.txt").write_text("hi", encoding="utf-8")
        self.assertTrue(self.registry.delete(ws["id"]))
        self.assertIsNone(self.registry.get(ws["id"]))
        self.assertFalse(target_dir.exists())

    def test_delete_returns_false_for_missing(self):
        self.assertFalse(self.registry.delete("missing"))

    def test_load_handles_corrupt_file(self):
        self.registry._path.write_text("not json", encoding="utf-8")
        registry2 = WorkspaceRegistry(self.registry._path, self.registry._dir)
        # Corrupt file → empty registry rather than crash.
        self.assertEqual(registry2.list_all(), [])

    def test_save_writes_valid_json_list(self):
        self.registry.create("A")
        self.registry.create("B")
        data = json.loads(self.registry._path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)


if __name__ == "__main__":
    unittest.main()
