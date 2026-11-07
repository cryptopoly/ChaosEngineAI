"""Token scopes: read, generate, files, download, clone."""

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.mcp_bridge import access
from backend_service.routes import agent as agent_routes

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(access, "_path", lambda: Path(self.tmp.name) / "agent_access.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(create_app(api_token="app-token"))
        access.set_enabled(True)

    def headers(self, token=None):
        return {"Authorization": f"Bearer {token or access.get_token()}"}

    def call(self, name, args=None, token=None):
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args or {}}}
        result = self.client.post("/mcp", json=body, headers=self.headers(token)).json()["result"]
        return result["isError"], result["content"][0]["text"]

    def names(self, token=None):
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        return {t["name"] for t in self.client.post("/mcp", json=body, headers=self.headers(token)).json()["result"]["tools"]}

    # -- migration and defaults -------------------------------------------

    def test_existing_token_is_migrated_to_read_generate_files(self):
        path = access._path()
        path.write_text(json.dumps({"enabled": True, "token": "old-token"}))
        self.assertEqual(access.get_scopes(), ["read", "generate", "files"])
        self.assertEqual(json.loads(path.read_text())["scopes"], ["read", "generate", "files"])
        self.assertEqual(access.get_token(), "old-token")  # the token itself is unchanged

    def test_read_is_always_kept_and_unknown_scopes_dropped(self):
        self.assertEqual(access.set_scopes(["files", "bogus", "clone"]), ["read", "files", "clone"])

    # -- per scope ----------------------------------------------------------

    def test_full_token_cannot_download_or_clone_by_default(self):
        scopes = access.principal_for(access.get_token()).scopes
        self.assertEqual(scopes, {"read", "generate", "files"})

    def test_read_scope_allows_status_and_lists_only(self):
        token = access.make_read_token()
        self.assertFalse(self.call("system_status", token=token)[0])
        self.assertFalse(self.call("list_jobs", token=token)[0])
        self.assertFalse(self.call("storyboard_project", {"action": "list"}, token=token)[0])
        for name, args in (("generate_image", {"prompt": "x"}), ("speak", {"text": "hi"}),
                           ("cancel_job", {"job_id": "job-1"}),
                           ("storyboard_project", {"action": "create", "name": "n"}),
                           ("storyboard_project", {"action": "export", "project_id": "p"}),
                           ("upload_file", {"data_base64": base64.b64encode(PNG).decode()})):
            is_error, text = self.call(name, args, token=token)
            self.assertTrue(is_error, name)
            self.assertIn("scope", text)
        listed = self.names(token)
        self.assertIn("system_status", listed)
        self.assertNotIn("generate_image", listed)
        self.assertNotIn("upload_file", listed)
        self.assertIn("storyboard_project", listed)

    def test_generate_scope_is_needed_to_start_jobs(self):
        access.set_scopes(["read", "files"])
        is_error, text = self.call("speak", {"text": "hi"})
        self.assertTrue(is_error)
        self.assertIn("'generate' scope", text)
        self.assertNotIn("speak", self.names())
        access.set_scopes(["read", "generate", "files"])
        self.assertIn("speak", self.names())

    def test_files_scope_gates_upload_and_http_file_routes(self):
        access.set_scopes(["read", "generate"])
        is_error, text = self.call("upload_file", {"data_base64": base64.b64encode(PNG).decode()})
        self.assertTrue(is_error)
        self.assertIn("'files' scope", text)
        self.assertEqual(self.client.post("/api/agent/uploads", content=PNG, headers=self.headers()).status_code, 403)
        self.assertEqual(self.client.get("/api/agent/files", params={"path": "/x"}, headers=self.headers()).status_code, 403)
        access.set_scopes(["read", "generate", "files"])
        self.assertFalse(self.call("upload_file", {"data_base64": base64.b64encode(PNG).decode()})[0])
        self.assertFalse(self.call("upload_image", {"data_base64": base64.b64encode(PNG).decode()})[0])  # old name
        self.assertEqual(self.client.post("/api/agent/uploads", content=PNG, headers=self.headers()).status_code, 200)
        read_token = access.make_read_token()
        self.assertEqual(self.client.post("/api/agent/uploads", content=PNG, headers=self.headers(read_token)).status_code, 403)

    def test_app_token_keeps_every_scope(self):
        is_error, _ = self.call("system_status", token="app-token")
        self.assertFalse(is_error)

    # -- clone and remote ---------------------------------------------------

    def test_remote_caller_never_gets_clone_unless_ticked(self):
        access.set_scopes(["read", "generate", "files", "clone"])
        token = access.get_token()
        local = access.principal_for(token, loopback=True)
        remote = access.principal_for(token, loopback=False)
        self.assertIn("clone", local.scopes)
        self.assertNotIn("clone", remote.scopes)
        self.assertIn("generate", remote.scopes)
        access.set_remote_clone(True)
        self.assertIn("clone", access.principal_for(token, loopback=False).scopes)

    def test_request_from_another_computer_is_treated_as_remote(self):
        access.set_scopes(["read", "generate", "files", "clone"])
        seen = {}
        original = access.principal_for

        def spy(token, loopback=True):
            seen["loopback"] = loopback
            return original(token, loopback)

        with mock.patch.object(access, "principal_for", spy):
            self.client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers=self.headers())
        self.assertFalse(seen["loopback"])  # the test client's host is not loopback
        with mock.patch.object(agent_routes, "_is_loopback", lambda request: True), mock.patch.object(access, "principal_for", spy):
            self.client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers=self.headers())
        self.assertTrue(seen["loopback"])

    # -- settings endpoint --------------------------------------------------

    def test_access_endpoint_scopes_read_token_and_remote_clone(self):
        headers = {"Authorization": "Bearer app-token"}
        state = self.client.get("/api/agent/access", headers=headers).json()
        self.assertEqual(state["scopes"], ["read", "generate", "files"])
        self.assertIsNone(state["readToken"])
        self.assertFalse(state["remoteClone"])
        made = self.client.post("/api/agent/access", json={"readToken": True, "scopes": ["generate", "download"],
                                                           "remoteClone": True}, headers=headers).json()
        self.assertTrue(made["readToken"])
        self.assertEqual(made["scopes"], ["read", "generate", "download"])
        self.assertTrue(made["remoteClone"])
        self.assertEqual(access.principal_for(made["readToken"]).scopes, {"read"})
        self.assertNotEqual(made["readToken"], made["token"])
        gone = self.client.post("/api/agent/access", json={"readToken": False}, headers=headers).json()
        self.assertIsNone(gone["readToken"])
        self.assertIsNone(access.principal_for(made["readToken"]))
        rotated = self.client.post("/api/agent/access", json={"rotateToken": True}, headers=headers).json()
        self.assertEqual(rotated["scopes"], ["read", "generate", "download"])  # rotating keeps the permissions


if __name__ == "__main__":
    unittest.main()
