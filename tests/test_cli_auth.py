"""chaosengine-cli sends the app token when the backend asks for it (requireApiAuth on).

Like the MCP relay, the CLI reads the loopback session token after a 401 and tries once more. The token goes only in
the Authorization header (never the URL, never stdout or stderr).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import uvicorn

from backend_service.app import create_app

CLI = Path(__file__).resolve().parents[1] / "scripts" / "chaosengine-cli"
TOKEN = "cli-test-token-7f3a"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_cli(port: int, *argv: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = {k: v for k, v in os.environ.items() if k != "CHAOSENGINE_API_TOKEN"}
    full_env.update(env or {})
    return subprocess.run(
        [sys.executable, str(CLI), "--port", str(port), *argv],
        capture_output=True, text=True, timeout=60, env=full_env,
    )


class _StubHandler(BaseHTTPRequestHandler):
    seen: list[tuple[str, str | None]] = []

    def log_message(self, *_args):
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        auth = self.headers.get("Authorization")
        type(self).seen.append((self.path, auth))
        if self.path == "/api/auth/session":
            self._send(200, {"apiToken": TOKEN, "tokenType": "Bearer"})
        elif auth == f"Bearer {TOKEN}":
            self._send(200, {"ok": True, "path": self.path})
        else:
            self._send(401, {"detail": "Unauthorized."})


class CliTokenStubTests(unittest.TestCase):
    def setUp(self):
        _StubHandler.seen = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.port = self.server.server_address[1]
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def test_reads_session_token_after_401_and_retries(self):
        result = run_cli(self.port, "call", "GET", "/api/version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["ok"], True)
        paths = [p for p, _ in _StubHandler.seen]
        self.assertEqual(paths, ["/api/version", "/api/auth/session", "/api/version"])
        self.assertIsNone(_StubHandler.seen[0][1])
        self.assertEqual(_StubHandler.seen[2][1], f"Bearer {TOKEN}")

    def test_token_never_in_url_or_output(self):
        result = run_cli(self.port, "call", "GET", "/api/workspace")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(TOKEN, result.stdout + result.stderr)
        self.assertTrue(all(TOKEN not in path for path, _ in _StubHandler.seen))

    def test_explicit_env_token_is_sent_up_front(self):
        result = run_cli(self.port, "call", "GET", "/api/version", env={"CHAOSENGINE_API_TOKEN": TOKEN})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([p for p, _ in _StubHandler.seen], ["/api/version"])

    def test_wrong_token_still_fails_cleanly(self):
        result = run_cli(self.port, "call", "GET", "/api/version", env={"CHAOSENGINE_API_TOKEN": "nope"})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("nope", result.stdout)


class CliTokenNonLoopbackTests(unittest.TestCase):
    def test_session_token_only_for_loopback(self):
        from importlib.machinery import SourceFileLoader
        import importlib.util

        loader = SourceFileLoader("chaosengine_cli_auth", str(CLI))
        spec = importlib.util.spec_from_loader("chaosengine_cli_auth", loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        env = {k: v for k, v in os.environ.items() if k != "CHAOSENGINE_API_TOKEN"}
        with unittest.mock.patch.dict(os.environ, env, clear=True), \
                unittest.mock.patch.object(module.urllib.request, "urlopen") as opener:
            self.assertIsNone(module._session_token("http://192.0.2.10:8876/api/version", refresh=True))
            opener.assert_not_called()


class CliTokenRealAppTests(unittest.TestCase):
    def test_call_works_with_auth_on(self):
        port = free_port()
        app = create_app(api_token=TOKEN)
        app.state.chaosengine_require_api_auth = True
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 10)
        self.addCleanup(setattr, server, "should_exit", True)
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.1)
        self.assertTrue(server.started)

        result = run_cli(port, "call", "GET", "/api/version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(TOKEN, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
