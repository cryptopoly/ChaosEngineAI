import json
import tempfile
import unittest
from pathlib import Path

from backend_service.helpers.prompts import PromptLibrary


class PromptLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmpdir.name)
        self.library = PromptLibrary(self.data_dir)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_creation_seeds_five_templates(self):
        templates = self.library.list_all()
        self.assertEqual(len(templates), 5)

    def test_seed_template_ids(self):
        templates = self.library.list_all()
        ids = {t["id"] for t in templates}
        self.assertIn("builtin-coding-assistant", ids)
        self.assertIn("builtin-creative-writer", ids)
        self.assertIn("builtin-data-analyst", ids)
        self.assertIn("builtin-translator", ids)
        self.assertIn("builtin-summarizer", ids)

    def test_get_existing_template(self):
        tmpl = self.library.get("builtin-coding-assistant")
        self.assertIsNotNone(tmpl)
        self.assertEqual(tmpl["name"], "Coding Assistant")
        self.assertIn("coding", tmpl["tags"])

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.library.get("nonexistent-id"))

    def test_create_template(self):
        new = self.library.create({
            "name": "Custom Bot",
            "systemPrompt": "You are a custom bot.",
            "tags": ["custom", "bot"],
            "category": "Custom",
        })
        self.assertIn("id", new)
        self.assertEqual(new["name"], "Custom Bot")
        self.assertIn("createdAt", new)
        self.assertIn("updatedAt", new)

        # Should now be retrievable
        fetched = self.library.get(new["id"])
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["name"], "Custom Bot")

    def test_create_with_explicit_id(self):
        new = self.library.create({
            "id": "my-custom-id",
            "name": "Explicit ID",
            "systemPrompt": "Test.",
        })
        self.assertEqual(new["id"], "my-custom-id")

    def test_update_template(self):
        updated = self.library.update("builtin-coding-assistant", {
            "name": "Senior Coding Assistant",
            "tags": ["coding", "senior"],
        })
        self.assertIsNotNone(updated)
        self.assertEqual(updated["name"], "Senior Coding Assistant")
        self.assertIn("senior", updated["tags"])

    def test_update_nonexistent_returns_none(self):
        result = self.library.update("nonexistent", {"name": "Nope"})
        self.assertIsNone(result)

    def test_delete_template(self):
        self.assertTrue(self.library.delete("builtin-summarizer"))
        self.assertIsNone(self.library.get("builtin-summarizer"))
        templates = self.library.list_all()
        self.assertEqual(len(templates), 4)

    def test_delete_nonexistent_returns_false(self):
        self.assertFalse(self.library.delete("nonexistent"))

    def test_search_by_name(self):
        results = self.library.search(query="coding")
        self.assertGreater(len(results), 0)
        self.assertTrue(any("Coding" in r["name"] for r in results))

    def test_search_by_tag(self):
        results = self.library.search(tags=["writing"])
        self.assertGreater(len(results), 0)
        self.assertTrue(any("Creative" in r["name"] for r in results))

    def test_search_by_category(self):
        results = self.library.search(category="Data")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Data Analyst")

    def test_search_combined_filters(self):
        results = self.library.search(query="creative", tags=["writing"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Creative Writer")

    def test_search_no_match(self):
        results = self.library.search(query="zzz_no_match_zzz")
        self.assertEqual(results, [])

    def test_persistence_save_and_reload(self):
        self.library.create({
            "name": "Persisted Bot",
            "systemPrompt": "I persist.",
            "tags": ["persist"],
        })
        # Create a new library instance from the same directory
        library2 = PromptLibrary(self.data_dir)
        templates = library2.list_all()
        names = {t["name"] for t in templates}
        self.assertIn("Persisted Bot", names)
        self.assertEqual(len(templates), 6)  # 5 seeds + 1 new

    def test_persistence_file_is_valid_json(self):
        file_path = self.data_dir / "prompt_templates.json"
        self.assertTrue(file_path.exists())
        data = json.loads(file_path.read_text())
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 5)

    def test_template_has_timestamps(self):
        templates = self.library.list_all()
        for tmpl in templates:
            self.assertIn("createdAt", tmpl)
            self.assertIn("updatedAt", tmpl)
            self.assertIsInstance(tmpl["createdAt"], float)


if __name__ == "__main__":
    unittest.main()
