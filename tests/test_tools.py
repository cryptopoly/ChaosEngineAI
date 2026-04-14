import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend_service.tools import BaseTool, ToolRegistry
from backend_service.tools.calculator import CalculatorTool
from backend_service.tools.code_executor import CodeExecutorTool
from backend_service.tools.file_reader import FileReaderTool
from backend_service.tools.web_search import WebSearchTool


class CalculatorToolTests(unittest.TestCase):
    def setUp(self):
        self.calc = CalculatorTool()

    def test_basic_arithmetic(self):
        result = self.calc.execute(expression="2 + 3")
        self.assertIn("5", result)

    def test_multiplication(self):
        result = self.calc.execute(expression="7 * 8")
        self.assertIn("56", result)

    def test_division(self):
        result = self.calc.execute(expression="10 / 4")
        self.assertIn("2.5", result)

    def test_sqrt(self):
        result = self.calc.execute(expression="sqrt(144)")
        self.assertIn("12", result)

    def test_pi_constant(self):
        result = self.calc.execute(expression="pi")
        self.assertIn("3.14159", result)

    def test_complex_expression(self):
        result = self.calc.execute(expression="sqrt(16) + 2**3")
        self.assertIn("12", result)

    def test_division_by_zero(self):
        result = self.calc.execute(expression="1 / 0")
        self.assertIn("Error", result)

    def test_empty_expression(self):
        result = self.calc.execute(expression="")
        self.assertIn("Error", result)

    def test_no_expression_key(self):
        result = self.calc.execute()
        self.assertIn("Error", result)

    def test_unsupported_function(self):
        result = self.calc.execute(expression="__import__('os')")
        self.assertIn("Error", result)

    def test_name(self):
        self.assertEqual(self.calc.name, "calculator")

    def test_openai_schema(self):
        schema = self.calc.openai_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "calculator")
        self.assertIn("parameters", schema["function"])


class CodeExecutorToolTests(unittest.TestCase):
    def setUp(self):
        self.executor = CodeExecutorTool()

    def test_simple_code(self):
        result = self.executor.execute(code="print('hello world')")
        self.assertIn("hello world", result)

    def test_empty_code(self):
        result = self.executor.execute(code="")
        self.assertIn("Error", result)

    def test_no_code_key(self):
        result = self.executor.execute()
        self.assertIn("Error", result)

    def test_code_with_math(self):
        result = self.executor.execute(code="print(2 + 3)")
        self.assertIn("5", result)

    def test_code_exceeding_length_limit(self):
        result = self.executor.execute(code="x = 1\n" * 10001)
        self.assertIn("Error", result)

    def test_code_with_syntax_error(self):
        result = self.executor.execute(code="def foo(")
        # Should show errors in output
        self.assertTrue("Error" in result or "SyntaxError" in result or "Exit code" in result)

    def test_code_no_output(self):
        result = self.executor.execute(code="x = 42")
        self.assertIn("no output", result.lower())

    def test_name(self):
        self.assertEqual(self.executor.name, "code_executor")

    def test_parameters_schema(self):
        schema = self.executor.parameters_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("code", schema["properties"])


class FileReaderToolTests(unittest.TestCase):
    def setUp(self):
        self.reader = FileReaderTool()
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_read_valid_txt_file(self):
        path = Path(self.tmpdir.name) / "test.txt"
        path.write_text("Hello from file.", encoding="utf-8")
        result = self.reader.execute(path=str(path))
        self.assertIn("Hello from file.", result)

    def test_read_valid_py_file(self):
        path = Path(self.tmpdir.name) / "script.py"
        path.write_text("print('hi')", encoding="utf-8")
        result = self.reader.execute(path=str(path))
        self.assertIn("print('hi')", result)

    def test_missing_file(self):
        result = self.reader.execute(path="/tmp/nonexistent_file_12345.txt")
        self.assertIn("Error", result)

    def test_empty_path(self):
        result = self.reader.execute(path="")
        self.assertIn("Error", result)

    def test_disallowed_extension(self):
        path = Path(self.tmpdir.name) / "data.exe"
        path.write_text("binary stuff", encoding="utf-8")
        result = self.reader.execute(path=str(path))
        self.assertIn("Error", result)
        self.assertIn("not supported", result)

    def test_max_lines(self):
        path = Path(self.tmpdir.name) / "long.txt"
        lines = [f"Line {i}" for i in range(100)]
        path.write_text("\n".join(lines), encoding="utf-8")
        result = self.reader.execute(path=str(path), max_lines=5)
        self.assertIn("Line 0", result)
        self.assertIn("truncated", result)

    def test_directory_not_file(self):
        result = self.reader.execute(path=self.tmpdir.name)
        self.assertIn("Error", result)

    def test_name(self):
        self.assertEqual(self.reader.name, "file_reader")


class WebSearchToolTests(unittest.TestCase):
    def setUp(self):
        self.search = WebSearchTool()

    def test_empty_query(self):
        result = self.search.execute(query="")
        self.assertIn("Error", result)

    @patch("backend_service.tools.web_search.urllib.request.urlopen")
    def test_successful_search(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "AbstractText": "Python is a programming language.",
            "AbstractURL": "https://python.org",
            "Heading": "Python",
            "RelatedTopics": [],
        }).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = self.search.execute(query="python programming")
        self.assertIn("Python", result)
        self.assertIn("python.org", result)

    @patch("backend_service.tools.web_search.urllib.request.urlopen")
    def test_no_results(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "AbstractText": "",
            "AbstractURL": "",
            "Heading": "",
            "RelatedTopics": [],
        }).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = self.search.execute(query="xyznonexistent123")
        self.assertIn("No results", result)

    @patch("backend_service.tools.web_search.urllib.request.urlopen")
    def test_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection timeout")
        result = self.search.execute(query="test query")
        self.assertIn("failed", result.lower())

    def test_name(self):
        self.assertEqual(self.search.name, "web_search")

    def test_parameters_schema(self):
        schema = self.search.parameters_schema()
        self.assertIn("query", schema["properties"])
        self.assertIn("query", schema["required"])


class ToolRegistryTests(unittest.TestCase):
    def test_register_and_get(self):
        reg = ToolRegistry()
        calc = CalculatorTool()
        reg.register(calc)
        self.assertIs(reg.get("calculator"), calc)

    def test_get_unknown_returns_none(self):
        reg = ToolRegistry()
        self.assertIsNone(reg.get("nonexistent"))

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.register(FileReaderTool())
        tools = reg.list_tools()
        self.assertEqual(len(tools), 2)
        names = {t.name for t in tools}
        self.assertEqual(names, {"calculator", "file_reader"})

    def test_openai_schemas(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        schemas = reg.openai_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["type"], "function")
        self.assertEqual(schemas[0]["function"]["name"], "calculator")

    def test_available_names(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.register(WebSearchTool())
        names = reg.available_names()
        self.assertIn("calculator", names)
        self.assertIn("web_search", names)

    def test_discover_registers_all_builtins(self):
        reg = ToolRegistry()
        reg.discover()
        names = reg.available_names()
        self.assertIn("calculator", names)
        self.assertIn("code_executor", names)
        self.assertIn("file_reader", names)
        self.assertIn("web_search", names)


if __name__ == "__main__":
    unittest.main()
