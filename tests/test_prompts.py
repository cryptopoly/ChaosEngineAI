import json
import tempfile
import unittest
from pathlib import Path

from backend_service.helpers.prompts import (
    PromptLibrary,
    apply_variables,
    extract_placeholders,
)


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


class VariableSubstitutionTests(unittest.TestCase):
    def test_extract_placeholders_returns_unique_in_order(self):
        text = "Hi {{name}}, you owe {{amount}}. Thanks {{name}}."
        self.assertEqual(extract_placeholders(text), ["name", "amount"])

    def test_extract_placeholders_tolerates_inner_whitespace(self):
        text = "Topic: {{ topic }} | Audience: {{audience}}"
        self.assertEqual(extract_placeholders(text), ["topic", "audience"])

    def test_apply_variables_substitutes_known_names(self):
        text = "Hello {{name}}, welcome to {{place}}."
        out = apply_variables(text, {"name": "Ada", "place": "Earth"})
        self.assertEqual(out, "Hello Ada, welcome to Earth.")

    def test_apply_variables_keeps_unknown_placeholders(self):
        text = "Hi {{name}}, your token is {{secret}}."
        out = apply_variables(text, {"name": "Ada"})
        self.assertEqual(out, "Hi Ada, your token is {{secret}}.")

    def test_apply_variables_coerces_booleans_and_numbers(self):
        text = "Active: {{active}}, count: {{count}}"
        out = apply_variables(text, {"active": True, "count": 42})
        self.assertEqual(out, "Active: true, count: 42")

    def test_apply_variables_treats_none_as_empty(self):
        text = "Note: {{note}}"
        out = apply_variables(text, {"note": None})
        self.assertEqual(out, "Note: ")


class TemplatePresetTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.library = PromptLibrary(Path(self.tmpdir.name))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_create_persists_variables_and_presets(self):
        new = self.library.create({
            "name": "Pirate translator",
            "systemPrompt": "Translate {{text}} into {{tone}} pirate.",
            "variables": [
                {"name": "text", "type": "string"},
                {"name": "tone", "type": "string", "default": "swashbuckling"},
            ],
            "presetSamplers": {"topP": 0.85, "topK": 40},
            "presetModelRef": "Qwen3-7B",
        })
        self.assertEqual(len(new["variables"]), 2)
        self.assertEqual(new["variables"][0]["name"], "text")
        self.assertEqual(new["presetSamplers"], {"topP": 0.85, "topK": 40})
        self.assertEqual(new["presetModelRef"], "Qwen3-7B")

    def test_update_preserves_unspecified_preset_fields(self):
        created = self.library.create({
            "name": "Pirate translator",
            "systemPrompt": "Translate {{text}}",
            "variables": [{"name": "text", "type": "string"}],
            "presetSamplers": {"topP": 0.9},
            "presetModelRef": "Qwen3-7B",
        })
        # Only update the name; presets should stick.
        updated = self.library.update(created["id"], {"name": "Renamed"})
        self.assertEqual(updated["name"], "Renamed")
        self.assertEqual(updated["presetSamplers"], {"topP": 0.9})
        self.assertEqual(updated["presetModelRef"], "Qwen3-7B")
        self.assertEqual(len(updated["variables"]), 1)

    def test_create_drops_invalid_variable_entries(self):
        new = self.library.create({
            "name": "Mixed bag",
            "systemPrompt": "Hi {{name}}",
            "variables": [
                {"name": "name", "type": "string"},
                {"type": "string"},  # missing name
                "not-an-object",  # wrong shape
                {"name": "name", "type": "string"},  # duplicate
                {"name": "count", "type": "weird"},  # invalid type → coerces to string
            ],
        })
        names = [v["name"] for v in new["variables"]]
        self.assertEqual(names, ["name", "count"])
        self.assertEqual(new["variables"][1]["type"], "string")


if __name__ == "__main__":
    unittest.main()
