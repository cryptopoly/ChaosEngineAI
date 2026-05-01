"""Tests for the Phase 2.10 MCP client + tool adapter.

The full subprocess round-trip is covered by an in-test MCP server
that mimics the JSON-RPC protocol — `subprocess.Popen` is invoked
against a Python `-c` snippet so the test runs anywhere without an
external dependency. Pure helpers (`_parse_json_rpc_line`,
`_flatten_tool_result`, `_safe_name`) get direct unit tests.
"""

from __future__ import annotations

import json
import sys
import unittest
from unittest.mock import MagicMock

from backend_service.mcp.client import (
    DEFAULT_INITIALIZE_TIMEOUT_S,
    McpClient,
    McpClientError,
    McpServerConfig,
    McpToolDescriptor,
    _flatten_tool_result,
    _parse_json_rpc_line,
)
from backend_service.mcp.loader import close_all, load_mcp_tools
from backend_service.mcp.tool_adapter import McpTool, _safe_name


class JsonRpcLineParserTests(unittest.TestCase):
    def test_parses_valid_response(self):
        line = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
        parsed = _parse_json_rpc_line(line)
        self.assertEqual(parsed["id"], 1)
        self.assertTrue(parsed["result"]["ok"])

    def test_returns_none_on_empty_line(self):
        self.assertIsNone(_parse_json_rpc_line(""))
        self.assertIsNone(_parse_json_rpc_line("   "))

    def test_returns_none_on_log_lines(self):
        # MCP servers sometimes emit human-readable log output between
        # JSON-RPC frames; the client must skip them rather than crash.
        self.assertIsNone(_parse_json_rpc_line("Server starting up..."))

    def test_returns_none_on_invalid_json(self):
        self.assertIsNone(_parse_json_rpc_line("{ invalid json"))

    def test_returns_none_on_non_jsonrpc_object(self):
        self.assertIsNone(_parse_json_rpc_line(json.dumps({"version": "1.0", "ok": True})))

    def test_returns_none_on_array_payload(self):
        self.assertIsNone(_parse_json_rpc_line(json.dumps([1, 2, 3])))


class FlattenToolResultTests(unittest.TestCase):
    def test_concatenates_text_parts(self):
        result = {"content": [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]}
        self.assertEqual(_flatten_tool_result(result), "Hello\nWorld")

    def test_marks_error_results(self):
        result = {"isError": True, "content": [{"type": "text", "text": "boom"}]}
        self.assertEqual(_flatten_tool_result(result), "[MCP error] boom")

    def test_serialises_non_text_parts(self):
        result = {"content": [
            {"type": "text", "text": "label"},
            {"type": "image", "data": "<base64>"},
        ]}
        flattened = _flatten_tool_result(result)
        self.assertIn("label", flattened)
        self.assertIn("<base64>", flattened)

    def test_empty_content_list_returns_empty_string(self):
        self.assertEqual(_flatten_tool_result({"content": []}), "")

    def test_non_dict_result_falls_back_to_str(self):
        self.assertEqual(_flatten_tool_result("plain"), "plain")
        self.assertEqual(_flatten_tool_result(None), "")


class McpServerConfigTests(unittest.TestCase):
    def test_round_trips_through_dict(self):
        config = McpServerConfig(
            id="filesystem",
            command="npx",
            args=("-y", "@mcp/filesystem"),
            env={"ROOT": "/tmp"},
            enabled=True,
        )
        rebuilt = McpServerConfig.from_dict(config.to_dict())
        self.assertEqual(rebuilt, config)

    def test_rejects_missing_id(self):
        with self.assertRaises(McpClientError):
            McpServerConfig.from_dict({"command": "echo"})

    def test_rejects_missing_command(self):
        with self.assertRaises(McpClientError):
            McpServerConfig.from_dict({"id": "x"})

    def test_rejects_non_dict_payload(self):
        with self.assertRaises(McpClientError):
            McpServerConfig.from_dict("not a dict")  # type: ignore[arg-type]

    def test_rejects_non_list_args(self):
        with self.assertRaises(McpClientError):
            McpServerConfig.from_dict({"id": "x", "command": "echo", "args": "not a list"})

    def test_rejects_non_dict_env(self):
        with self.assertRaises(McpClientError):
            McpServerConfig.from_dict({"id": "x", "command": "echo", "env": ["not", "a", "dict"]})


class SafeNameTests(unittest.TestCase):
    def test_basic_format(self):
        self.assertEqual(_safe_name("filesystem", "read_file"), "mcp__filesystem__read_file")

    def test_strips_unsafe_chars(self):
        # Slashes / colons / dots get collapsed to underscores so the
        # name is OpenAI-function-call-safe.
        self.assertEqual(
            _safe_name("scope/server", "tool:variant.v2"),
            "mcp__scope_server__tool_variant_v2",
        )

    def test_empty_inputs_get_placeholders(self):
        self.assertEqual(_safe_name("", ""), "mcp__server__tool")


class McpToolAdapterTests(unittest.TestCase):
    def test_execute_proxies_to_client(self):
        client = MagicMock()
        client.call_tool.return_value = "all good"
        descriptor = McpToolDescriptor(
            server_id="fs",
            name="read_file",
            description="Read",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
        tool = McpTool(client, descriptor)
        result = tool.execute(path="/etc/hosts")
        self.assertEqual(result, "all good")
        client.call_tool.assert_called_once_with("read_file", {"path": "/etc/hosts"})

    def test_execute_converts_client_errors_to_text(self):
        client = MagicMock()
        client.call_tool.side_effect = McpClientError("server died")
        tool = McpTool(client, McpToolDescriptor(
            server_id="fs",
            name="read_file",
            description="Read",
            input_schema={},
        ))
        result = tool.execute(path="/x")
        self.assertIn("server died", result)
        self.assertIn("MCP server 'fs' error", result)

    def test_provenance_tag_format(self):
        tool = McpTool(MagicMock(), McpToolDescriptor(
            server_id="search-perplexity",
            name="search",
            description="",
            input_schema={},
        ))
        self.assertEqual(tool.provenance, "mcp:search-perplexity")

    def test_description_falls_back_when_empty(self):
        tool = McpTool(MagicMock(), McpToolDescriptor(
            server_id="fs",
            name="read_file",
            description="",
            input_schema={},
        ))
        self.assertIn("MCP server 'fs'", tool.description)


# ---------------------------------------------------------------------
# Subprocess round-trip — uses a Python -c snippet as a fake MCP server
# ---------------------------------------------------------------------


_FAKE_SERVER_SCRIPT = r"""
import json, sys

def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    method = msg.get("method")
    msg_id = msg.get("id")
    if method == "initialize":
        emit({"jsonrpc": "2.0", "id": msg_id, "result": {"protocolVersion": "2025-03-26", "capabilities": {}}})
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        emit({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": [{
            "name": "echo",
            "description": "Echo input back",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
        }]}})
    elif method == "tools/call":
        text = msg.get("params", {}).get("arguments", {}).get("text", "")
        emit({"jsonrpc": "2.0", "id": msg_id, "result": {"content": [{"type": "text", "text": f"echo: {text}"}]}})
    else:
        emit({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "Method not found"}})
"""


class McpClientRoundTripTests(unittest.TestCase):
    def _make_config(self, server_id: str = "fake") -> McpServerConfig:
        return McpServerConfig(
            id=server_id,
            command=sys.executable,
            args=("-c", _FAKE_SERVER_SCRIPT),
        )

    def test_initialize_then_list_then_call(self):
        config = self._make_config()
        client = McpClient(config, request_timeout=5.0)
        try:
            client.initialize(timeout=DEFAULT_INITIALIZE_TIMEOUT_S)
            tools = client.list_tools(timeout=5.0)
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].name, "echo")
            result = client.call_tool("echo", {"text": "hello world"}, timeout=5.0)
            self.assertEqual(result, "echo: hello world")
        finally:
            client.close()

    def test_list_tools_before_initialize_raises(self):
        client = McpClient(self._make_config(), request_timeout=5.0)
        with self.assertRaises(McpClientError):
            client.list_tools()
        client.close()

    def test_unknown_command_raises(self):
        config = McpServerConfig(id="bad", command="/nonexistent/binary")
        with self.assertRaises(McpClientError):
            McpClient(config).start()


class LoadMcpToolsTests(unittest.TestCase):
    def test_returns_tools_for_healthy_server(self):
        config = McpServerConfig(
            id="fake",
            command=sys.executable,
            args=("-c", _FAKE_SERVER_SCRIPT),
        )
        tools, clients = load_mcp_tools([config])
        try:
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].name, "mcp__fake__echo")
            self.assertEqual(tools[0].provenance, "mcp:fake")
        finally:
            close_all(clients)

    def test_skips_disabled_servers(self):
        config = McpServerConfig(
            id="fake",
            command=sys.executable,
            args=("-c", _FAKE_SERVER_SCRIPT),
            enabled=False,
        )
        tools, clients = load_mcp_tools([config])
        try:
            self.assertEqual(tools, [])
            self.assertEqual(clients, [])
        finally:
            close_all(clients)

    def test_isolates_failing_server(self):
        # One bad server + one good server: loader must return the
        # good server's tools and skip the bad one rather than aborting.
        good = McpServerConfig(
            id="ok",
            command=sys.executable,
            args=("-c", _FAKE_SERVER_SCRIPT),
        )
        bad = McpServerConfig(id="bad", command="/nonexistent/binary")
        log_calls: list[tuple[str, str]] = []
        tools, clients = load_mcp_tools([bad, good], log=lambda level, msg: log_calls.append((level, msg)))
        try:
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].name, "mcp__ok__echo")
            # Loader emitted a warning for the bad server.
            self.assertTrue(any(level == "warning" for level, _ in log_calls))
        finally:
            close_all(clients)


if __name__ == "__main__":
    unittest.main()
