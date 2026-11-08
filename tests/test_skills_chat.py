"""D-62: skills routes, the prompt block, load_skill / read_skill_file and the Chat stream wiring."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.skills import chat as skills_chat
from backend_service.skills import importer, store
from backend_service.state import ChaosEngineState
from backend_service.tools import ToolRegistry
from tests.test_backend_service import fake_library, fake_system_snapshot, make_test_client
from tests.test_chat_studio_tools import ScriptedRuntime, call, result

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples" / "skills"


class SkillsEnvCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        env = mock.patch.dict(os.environ, {"CHAOSENGINE_SKILLS_DIR": str(self.base / "skills")})
        env.start()
        self.addCleanup(env.stop)

    def import_examples(self, enable=True):
        for name in ("plain-summaries", "storyboard-house-style"):
            store.save(importer.load(str(EXAMPLES / name)))
            if enable:
                store.set_enabled(name, True)


class SkillRoutesTests(SkillsEnvCase):
    def setUp(self):
        super().setUp()
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot, library_provider=fake_library,
            settings_path=self.base / "settings.json", benchmarks_path=self.base / "b.json",
            chat_sessions_path=self.base / "c.json",
        )
        self.client = make_test_client(state)

    def test_import_list_enable_delete(self):
        self.assertEqual(self.client.get("/api/skills").json(), {"skills": [], "count": 0})
        response = self.client.post("/api/skills/import", json={"path": str(EXAMPLES / "storyboard-house-style")})
        self.assertEqual(response.status_code, 200, response.text)
        skill = response.json()["skill"]
        self.assertEqual(skill["id"], "storyboard-house-style")
        self.assertFalse(skill["enabled"])
        self.assertEqual([f["path"] for f in skill["files"]], ["SKILL.md", "palette.md"])
        self.assertEqual(skill["source"], "storyboard-house-style")  # base name only, never the full path
        listed = self.client.get("/api/skills").json()
        self.assertEqual(listed["count"], 1)
        enabled = self.client.post("/api/skills/storyboard-house-style/enabled", json={"enabled": True}).json()
        self.assertTrue(enabled["skill"]["enabled"])
        self.assertEqual(self.client.delete("/api/skills/storyboard-house-style").json(), {"deleted": "storyboard-house-style"})
        self.assertEqual(self.client.get("/api/skills").json()["count"], 0)

    def test_errors(self):
        self.client.post("/api/skills/import", json={"path": str(EXAMPLES / "plain-summaries")})
        dup = self.client.post("/api/skills/import", json={"path": str(EXAMPLES / "plain-summaries")})
        self.assertEqual(dup.status_code, 409)
        self.assertIn("already imported", dup.json()["detail"])
        missing = self.client.post("/api/skills/import", json={"path": str(self.base / "nope")})
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(self.client.post("/api/skills/import", json={"path": ""}).status_code, 422)
        self.assertEqual(self.client.post("/api/skills/BAD..ID/enabled", json={"enabled": True}).status_code, 400)
        self.assertEqual(self.client.post("/api/skills/not-here/enabled", json={"enabled": True}).status_code, 404)
        self.assertEqual(self.client.delete("/api/skills/not-here").status_code, 404)
        self.assertEqual(self.client.post("/api/skills/x/enabled", json={}).status_code, 422)


class PromptAndToolTests(SkillsEnvCase):
    def test_prompt_lists_only_enabled_skills_with_capped_descriptions(self):
        self.import_examples(enable=False)
        self.assertEqual(skills_chat.with_chat_skills("base", None, []), ("base", None))
        store.set_enabled("plain-summaries", True)
        long = dict(store.get("plain-summaries"), description="word " * 200)
        text = skills_chat.skills_prompt([long])
        line = text.splitlines()[-1]
        self.assertTrue(line.startswith("- plain-summaries: "))
        self.assertLessEqual(len(line), len("- plain-summaries: ") + skills_chat.PROMPT_DESCRIPTION_CHARS)
        prompt, registry = skills_chat.with_chat_skills("base", None, [])
        self.assertTrue(prompt.startswith("base\n\nSkills are on."))
        self.assertIn("plain-summaries", prompt)
        self.assertNotIn("storyboard-house-style", prompt)
        self.assertNotIn("Read the whole text once", prompt)  # full text only on request
        self.assertEqual(sorted(registry.available_names()), ["load_skill", "read_skill_file"])

    def test_chat_tools_join_an_existing_registry(self):
        self.import_examples()
        base = ToolRegistry()
        _, registry = skills_chat.with_chat_skills("", base, None)
        self.assertIs(registry, base)
        _, with_calc = skills_chat.with_chat_skills("", None, ["calculator"])
        self.assertIn("calculator", with_calc.available_names())

    def test_load_skill_and_read_file(self):
        self.import_examples()
        skills = skills_chat.enabled_skills()
        load, read = skills_chat.skill_tools(skills)
        out = load.execute_structured(name="storyboard-house-style")
        self.assertIn("Shot list format", out.text)
        self.assertNotIn("description: House style", out.text)  # front matter stripped
        self.assertIn("read_skill_file", out.text)
        self.assertIn("palette.md", out.text)
        self.assertTrue(out.text.startswith("[Imported skill"))
        self.assertIn("#C8553D", read.execute(name="Storyboard-House-Style", path="palette.md"))
        bad_file = read.execute(name="storyboard-house-style", path="../skill.json")
        self.assertTrue(bad_file.startswith("Error:"))
        self.assertIn("palette.md", bad_file)

    def test_unknown_name_lists_valid_names(self):
        self.import_examples()
        load, read = skills_chat.skill_tools(skills_chat.enabled_skills())
        for tool in (load, read):
            text = tool.execute(name="nope", path="x")
            self.assertEqual(text, "Error: Unknown skill 'nope'. Valid names: plain-summaries, storyboard-house-style.")

    def test_returned_text_is_capped(self):
        skill = importer.ImportedSkill(name="big", description="d",
                                       files={"SKILL.md": "---\nname: big\ndescription: d\n---\n" + "x" * 20000})
        store.save(skill)
        store.set_enabled("big", True)
        load, _ = skills_chat.skill_tools(skills_chat.enabled_skills())
        self.assertLess(len(load.execute(name="big")), skills_chat.MAX_RETURN_CHARS + 300)


class ChatStreamSkillsTests(SkillsEnvCase):
    def setUp(self):
        super().setUp()
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot, library_provider=fake_library,
            settings_path=self.base / "settings.json", benchmarks_path=self.base / "b.json",
            chat_sessions_path=self.base / "c.json",
        )
        self.runtime = ScriptedRuntime()
        state.runtime = self.runtime
        self.client = make_test_client(state)
        self.client.post("/api/models/load", json={"modelRef": "google/gemma-4-E4B-it", "modelName": "Gemma",
                                                   "source": "catalog", "backend": "mock"})
        self.session = self.client.post("/api/chat/sessions", json={"title": "t"}).json()["session"]

    def stream(self, **body):
        response = self.client.post("/api/chat/generate/stream", json={"sessionId": self.session["id"], "prompt": "go", **body})
        return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]

    def test_switch_off_or_no_enabled_skill_adds_nothing(self):
        self.import_examples(enable=False)
        self.stream(skills=True)
        self.stream()
        for tools, prompt in zip(self.runtime.tools_seen, self.runtime.system_prompts):
            self.assertNotIn("load_skill", tools)
            self.assertNotIn("Skills are on", prompt)

    def test_switch_on_offers_the_list_and_load_skill_runs(self):
        self.import_examples()
        self.runtime.replies = [result("", [call("load_skill", {"name": "plain-summaries"}, "c1")]), result("Done.")]
        events = self.stream(skills=True)
        self.assertEqual(sorted(self.runtime.tools_seen[0]), ["load_skill", "read_skill_file"])
        self.assertIn("- storyboard-house-style:", self.runtime.system_prompts[0])
        done = [e for e in events if e.get("done")][-1]
        calls = done["assistant"].get("toolCalls") or []
        self.assertEqual(calls[0]["name"], "load_skill")
        self.assertIn("Read the whole text once", calls[0]["result"])

    def test_skills_with_studio_tools_share_one_registry(self):
        self.import_examples()
        self.stream(skills=True, studioTools=True)
        self.assertIn("load_skill", self.runtime.tools_seen[0])
        self.assertIn("generate_image", self.runtime.tools_seen[0])
        self.assertIn("Studio tools are on", self.runtime.system_prompts[0])
        self.assertIn("Skills are on", self.runtime.system_prompts[0])


if __name__ == "__main__":
    unittest.main()
