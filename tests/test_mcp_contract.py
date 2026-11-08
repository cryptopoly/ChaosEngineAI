"""Contract tests for the agent tool list: annotations, short descriptions, valid schemas, a golden snapshot.

To accept a deliberate tool change: ``UPDATE_MCP_GOLDEN=1 pytest tests/test_mcp_contract.py``.
"""

import json
import os
import unittest
from pathlib import Path

from backend_service.mcp_bridge import scopes
from backend_service.mcp_bridge.protocol import INSTRUCTIONS
from backend_service.mcp_bridge.tools import HANDLERS, TOOLS, validate

GOLDEN = Path(__file__).parent / "golden" / "mcp_tools_list.json"
MAX_DESCRIPTION = 260
MAX_PARAM_DESCRIPTION = 300
ALLOWED_KEYS = {"type", "description", "enum", "minimum", "maximum", "minLength", "maxLength", "items", "maxItems"}
TYPES = {"string", "integer", "number", "boolean", "array", "object"}
VERBS = ("Read:", "Make:", "Edit:", "Send:", "Stop:")
HINTS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")
UNCAPPED_OK = {"path", "file", "model", "voice", "language", "project_id", "shot_id", "take_id", "clip_id",
               "job_id", "start_image", "end_image"}

# One valid call per tool; a new tool must add one here.
EXAMPLES = {
    "system_status": {},
    "recommend_models": {"kind": "image", "limit": 3},
    "list_models": {"kind": "video"},
    "list_voices": {},
    "list_jobs": {"state": "done", "limit": 5},
    "generate_image": {"prompt": "a red cube", "width": 512, "height": 512, "seed": 1, "project_id": "p", "project_name": "Harbour"},
    "generate_video": {"prompt": "waves", "seconds": 3},
    "speak": {"text": "Hello", "speed": 1.0},
    "upload_file": {"data_base64": "iVBORw0KGgo="},
    "transcribe": {"path": "/tmp/a.wav"},
    "make_music": {"caption": "calm piano", "instrumental": True, "seconds": 30},
    "make_sound": {"caption": "door creaking", "seconds": 4, "loop": False},
    "job_status": {"job_id": "job-1"},
    "wait_for_job": {"job_id": "job-1", "timeout_seconds": 5},
    "cancel_job": {"job_id": "job-1"},
    "storyboard_project": {"action": "list"},
    "storyboard_shot": {"action": "lock", "project_id": "p", "revision": 1, "shot_id": "s"},
    "storyboard_audio": {"action": "remove", "project_id": "p", "revision": 1, "clip_id": "c"},
    "storyboard_generate": {"project_id": "p", "revision": 1},
    "storyboard_render": {"project_id": "p", "revision": 1, "preview": True, "notify": False},
    "storyboard_captions": {"action": "add", "project_id": "p", "revision": 1, "start": 0, "end": 1, "text": "Hi"},
    "storyboard_export_otio": {"project_id": "p"},
    "music_library": {"action": "rename", "track_id": "t", "title": "New"},
    "list_cloned_voices": {},
    "speak_cloned": {"voice_id": "v", "text": "Hello", "permission_confirmed": True},
}


class ToolContractTests(unittest.TestCase):
    def test_every_tool_has_a_handler_example_and_annotations(self):
        names = [t["name"] for t in TOOLS]
        self.assertEqual(len(names), len(set(names)))
        for tool in TOOLS:
            name = tool["name"]
            self.assertIn(name, HANDLERS)
            ann = tool.get("annotations")
            self.assertIsNotNone(ann, name)
            for hint in HINTS:
                self.assertIsInstance(ann.get(hint), bool, f"{name}.{hint}")
            self.assertTrue(ann.get("title"), name)
            if ann["readOnlyHint"]:
                self.assertFalse(ann["destructiveHint"], name)
        self.assertEqual(set(EXAMPLES), set(names), "every tool needs an example in this test")

    def test_old_upload_name_is_an_alias(self):
        self.assertNotIn("upload_image", [t["name"] for t in TOOLS])
        self.assertIs(HANDLERS["upload_image"], HANDLERS["upload_file"])
        self.assertEqual(scopes.required_scope("upload_image"), "files")

    def test_descriptions_are_short_and_start_with_a_verb(self):
        for tool in TOOLS:
            text = tool["description"]
            self.assertLessEqual(len(text), MAX_DESCRIPTION, tool["name"])
            self.assertTrue(text.startswith(VERBS), f"{tool['name']}: {text[:20]}")
            for pname, spec in tool["inputSchema"]["properties"].items():
                self.assertLessEqual(len(spec.get("description", "")), MAX_PARAM_DESCRIPTION, f"{tool['name']}.{pname}")

    def test_schemas_use_only_the_supported_subset(self):
        for tool in TOOLS:
            schema = tool["inputSchema"]
            self.assertEqual(schema["type"], "object")
            self.assertIs(schema["additionalProperties"], False)
            for required in schema["required"]:
                self.assertIn(required, schema["properties"], tool["name"])
            for pname, spec in schema["properties"].items():
                where = f"{tool['name']}.{pname}"
                self.assertLessEqual(set(spec), ALLOWED_KEYS, where)
                self.assertIn(spec.get("type"), TYPES, where)
                if spec["type"] == "array":
                    self.assertIn(spec.get("items", {}).get("type"), TYPES, where)
                if "enum" in spec:
                    self.assertEqual(spec["type"], "string", where)
                    self.assertTrue(all(isinstance(v, str) for v in spec["enum"]), where)
                if "minimum" in spec and "maximum" in spec:
                    self.assertLessEqual(spec["minimum"], spec["maximum"], where)
                if spec["type"] == "string" and "enum" not in spec and "maxLength" not in spec:
                    self.assertIn(pname, UNCAPPED_OK, f"{where} has no maxLength")

    def test_examples_are_valid_against_their_own_schema(self):
        for tool in TOOLS:
            self.assertIsNone(validate(tool["inputSchema"], EXAMPLES[tool["name"]]), tool["name"])

    def test_required_arguments_are_enforced(self):
        for tool in TOOLS:
            for required in tool["inputSchema"]["required"]:
                args = {k: v for k, v in EXAMPLES[tool["name"]].items() if k != required}
                self.assertIn("missing required", validate(tool["inputSchema"], args) or "", f"{tool['name']}.{required}")

    def test_every_tool_has_a_scope(self):
        for tool in TOOLS:
            self.assertIn(scopes.required_scope(tool["name"], EXAMPLES[tool["name"]]), ("read", "generate", "files", "clone"))

    def test_instructions_name_music_and_storyboard(self):
        for word in ("music", "storyboard", "list_jobs", "upload_file"):
            self.assertIn(word, INSTRUCTIONS)

    def test_golden_tools_list(self):
        current = json.dumps(TOOLS, indent=2, sort_keys=True) + "\n"
        if os.environ.get("UPDATE_MCP_GOLDEN"):
            GOLDEN.parent.mkdir(exist_ok=True)
            GOLDEN.write_text(current, encoding="utf-8")
        self.assertTrue(GOLDEN.is_file(), "run with UPDATE_MCP_GOLDEN=1 to create the snapshot")
        self.assertEqual(
            json.loads(GOLDEN.read_text(encoding="utf-8")), json.loads(current),
            "A tool's name, description, annotations or schema changed. If intended, rerun with UPDATE_MCP_GOLDEN=1.",
        )


if __name__ == "__main__":
    unittest.main()
