import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import uvicorn

from backend_service.app import create_app
from backend_service.mcp_bridge import access

CLI = Path(__file__).resolve().parents[1] / "scripts" / "chaosengine-cli"
LINES = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "system_status", "arguments": {}}},
]


def relay(port: int) -> list[dict]:
    result = subprocess.run(
        [sys.executable, str(CLI), "--port", str(port), "mcp"],
        input="\n".join(json.dumps(m) for m in LINES) + "\n",
        capture_output=True, text=True, timeout=60,
    )
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class McpRelayTests(unittest.TestCase):
    def test_app_not_running_says_start_it(self):
        replies = relay(free_port())
        self.assertEqual([r["id"] for r in replies], [1, 2, 3])
        self.assertIn("result", replies[0])
        self.assertIn("Start ChaosEngineAI", replies[1]["error"]["message"])
        self.assertTrue(replies[2]["result"]["isError"])
        self.assertIn("Start ChaosEngineAI", replies[2]["result"]["content"][0]["text"])

    def test_forwards_to_running_app(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(access, "_path", lambda: Path(tmp.name) / "a.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(create_app(api_token="app-token"), host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 10)
        self.addCleanup(setattr, server, "should_exit", True)
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.1)
        self.assertTrue(server.started)

        off = relay(port)
        self.assertEqual(len(off[1]["result"]["tools"]), 11)
        self.assertIn("Agent access is off", off[2]["result"]["content"][0]["text"])

        access.set_enabled(True)
        on = relay(port)
        self.assertFalse(on[2]["result"]["isError"])
        self.assertIn("freeMemoryGb", on[2]["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
