"""FU-231: the text tool block on the MLX path prints the values a call may take."""

from __future__ import annotations

import unittest

from backend_service.mlx_worker_request import _format_tools_for_prompt


def _tool(name: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": "d", "parameters": {
        "type": "object", "properties": properties, "required": required or []}}}


class ToolPromptLimitTests(unittest.TestCase):
    def test_enum_range_default_and_nested_keys_are_printed(self) -> None:
        block = _format_tools_for_prompt([
            _tool("list_models", {"kind": {"type": "string", "enum": ["text", "image", "video"],
                                           "description": "Which kind."}}, ["kind"]),
            _tool("speak", {"speed": {"type": "number", "minimum": 0.5, "maximum": 2, "default": 1,
                                      "description": "Pace."}}),
            _tool("storyboard_shot", {"fields": {"type": "object", "description": "Shot fields.", "properties": {
                "seconds": {"type": "integer", "minimum": 1, "maximum": 10},
                "stillPrompt": {"type": "string"}}}}),
        ]) or ""
        self.assertIn("kind: string (required) [one of: text, image, video] — Which kind.", block)
        self.assertIn("speed: number [0.5 to 2, default 1] — Pace.", block)
        self.assertIn("keys: seconds: integer [1 to 10]; stillPrompt: string", block)

    def test_a_plain_parameter_reads_as_before(self) -> None:
        block = _format_tools_for_prompt([_tool("t", {"q": {"type": "string", "description": "Ask."}}, ["q"])]) or ""
        self.assertIn("- q: string (required) — Ask.", block)

    def test_no_tools_gives_no_block(self) -> None:
        self.assertIsNone(_format_tools_for_prompt(None))


if __name__ == "__main__":
    unittest.main()
