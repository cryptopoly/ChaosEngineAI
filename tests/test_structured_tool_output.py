"""Tests for the Phase 2.8 structured-tool-output protocol.

Each built-in tool that opted in to `execute_structured` is exercised
end-to-end: it returns a `StructuredToolOutput` with the expected
`render_as` and `data` shape, and the legacy `execute(...)` path
still works for callers that bypass the structured route.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest import mock

from backend_service.tools import StructuredToolOutput
from backend_service.tools.calculator import CalculatorTool
from backend_service.tools.file_reader import FileReaderTool


class StructuredCalculatorTests(unittest.TestCase):
    def test_returns_code_render_for_valid_expression(self):
        tool = CalculatorTool()
        out = tool.execute_structured(expression="2 + 2")
        self.assertIsInstance(out, StructuredToolOutput)
        self.assertEqual(out.render_as, "code")
        self.assertEqual(out.data["code"], "2 + 2 = 4")
        self.assertEqual(out.text, "2 + 2 = 4")

    def test_returns_markdown_render_for_error(self):
        tool = CalculatorTool()
        out = tool.execute_structured(expression="2 + ")
        self.assertEqual(out.render_as, "markdown")
        self.assertTrue(out.text.startswith("Error"))

    def test_legacy_execute_unchanged(self):
        tool = CalculatorTool()
        # Plain text path must still produce the human-readable form
        # so callers that don't use `execute_structured` keep working.
        self.assertEqual(tool.execute(expression="2 + 2"), "2 + 2 = 4")


class StructuredFileReaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._roots_patch = mock.patch(
            "backend_service.tools.file_reader._configured_allowed_roots",
            return_value=[Path(self._tmp.name).resolve()],
        )
        self._roots_patch.start()

    def tearDown(self):
        self._roots_patch.stop()
        self._tmp.cleanup()

    def _write(self, name: str, body: str) -> str:
        path = Path(self._tmp.name) / name
        path.write_text(body, encoding="utf-8")
        return str(path)

    def test_python_file_renders_as_code_with_language(self):
        path = self._write("hello.py", "print('hi')\n")
        tool = FileReaderTool()
        out = tool.execute_structured(path=path)
        self.assertEqual(out.render_as, "code")
        self.assertEqual(out.data["language"], "py")
        self.assertIn("print('hi')", out.data["code"])

    def test_markdown_file_renders_as_markdown(self):
        path = self._write("notes.md", "# Title\n\nBody")
        tool = FileReaderTool()
        out = tool.execute_structured(path=path)
        self.assertEqual(out.render_as, "markdown")
        self.assertIn("# Title", out.data["markdown"])

    def test_unknown_extension_falls_back_to_text_language(self):
        path = self._write("data.txt", "line one\nline two\n")
        tool = FileReaderTool()
        out = tool.execute_structured(path=path)
        self.assertEqual(out.render_as, "code")
        self.assertEqual(out.data["language"], "txt")

    def test_error_path_renders_markdown(self):
        tool = FileReaderTool()
        out = tool.execute_structured(path="/nonexistent/file.py")
        self.assertEqual(out.render_as, "markdown")
        self.assertTrue(out.text.startswith("Error"))


class StructuredWebSearchTests(unittest.TestCase):
    def test_returns_table_with_columns_and_rows(self):
        from backend_service.tools.web_search import WebSearchTool

        tool = WebSearchTool()
        with mock.patch.object(
            tool,
            "_search_results",
            return_value=[
                {"title": "Result A", "url": "https://example.com/a", "snippet": "first hit"},
                {"title": "Result B", "url": "https://example.com/b", "snippet": "second hit"},
            ],
        ):
            out = tool.execute_structured(query="test query")
        self.assertEqual(out.render_as, "table")
        self.assertEqual(out.data["columns"], ["#", "Title", "URL", "Snippet"])
        self.assertEqual(len(out.data["rows"]), 2)
        self.assertEqual(out.data["rows"][0][1], "Result A")

    def test_empty_query_renders_markdown_error(self):
        from backend_service.tools.web_search import WebSearchTool

        tool = WebSearchTool()
        out = tool.execute_structured(query="")
        self.assertEqual(out.render_as, "markdown")
        self.assertIn("no search query", out.text.lower())

    def test_no_results_renders_markdown_message(self):
        from backend_service.tools.web_search import WebSearchTool

        tool = WebSearchTool()
        with mock.patch.object(tool, "_search_results", return_value=[]):
            out = tool.execute_structured(query="ghost")
        self.assertEqual(out.render_as, "markdown")
        self.assertIn("No results found", out.text)


class BaseToolDefaultsTests(unittest.TestCase):
    def test_default_execute_structured_returns_none(self):
        # Tools that don't override `execute_structured` must keep the
        # legacy text path active. Use the calculator's parent class
        # contract directly via a minimal subclass.
        from backend_service.tools import BaseTool

        class _Plain(BaseTool):
            name = "plain"
            description = ""

            def parameters_schema(self):
                return {"type": "object", "properties": {}}

            def execute(self, **kwargs):
                return "ok"

        tool = _Plain()
        self.assertIsNone(tool.execute_structured())


if __name__ == "__main__":
    unittest.main()
