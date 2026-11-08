"""The agent guide (AGENTS.md, D-61): its tool list matches the real tools, and the three ways to read it work."""
from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.mcp_bridge import access, guide
from backend_service.mcp_bridge.protocol import INSTRUCTIONS
from backend_service.mcp_bridge.tools import TOOLS
from backend_service.routes import agent as agent_routes

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "AGENTS.md"
REAL = {t["name"] for t in TOOLS}


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert match, f"AGENTS.md lost its '## {heading}' section"
    return match.group(1)


def _listed_tools() -> list[str]:
    return re.findall(r"^- `([a-z_]+)`", _section(GUIDE.read_text(encoding="utf-8"), "Tools"), re.M)


class GuideContentTests(unittest.TestCase):
    def test_every_tool_in_the_guide_is_real(self) -> None:
        unknown = sorted(set(_listed_tools()) - REAL)
        self.assertEqual(unknown, [], f"AGENTS.md lists tools that do not exist: {unknown}")

    def test_every_real_tool_is_in_the_guide(self) -> None:
        missing = sorted(REAL - set(_listed_tools()))
        self.assertEqual(missing, [], f"tools missing from the AGENTS.md Tools section: {missing}")

    def test_no_tool_is_listed_twice(self) -> None:
        listed = _listed_tools()
        self.assertEqual(len(listed), len(set(listed)))

    def test_tool_like_names_anywhere_in_the_guide_are_real(self) -> None:
        text = GUIDE.read_text(encoding="utf-8")
        words = set(re.findall(
            r"`((?:storyboard|generate|make|list|speak|music|upload|system|job|wait|cancel|recommend)_[a-z_]+)`", text))
        words = {w for w in words if not w.endswith("_id")}  # job_id, project_id... are arguments
        self.assertEqual(sorted(words - REAL), [])

    def test_the_guide_stays_short_and_names_no_other_apps(self) -> None:
        text = GUIDE.read_text(encoding="utf-8")
        self.assertLess(len(text.splitlines()), 400)
        for banned in ("comfyui", "lm studio", "ollama", "automatic1111", "invokeai", "draw things", "pinokio"):
            self.assertNotIn(banned, text.lower())

    def test_the_instruction_string_points_at_the_guide(self) -> None:
        self.assertIn("chaosengine://guide", INSTRUCTIONS)
        self.assertIn("start_here", INSTRUCTIONS)


class GuideServedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        for patcher in (
            mock.patch.dict(os.environ, {"CHAOSENGINE_STORYBOARDS_DIR": str(base / "storyboards")}),
            mock.patch.object(access, "_path", lambda: base / "agent_access.json"),
            mock.patch.object(agent_routes, "_is_loopback", lambda request: True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(create_app(api_token="app-token"))
        access.set_enabled(True)
        self.token = access.get_token()
        self.text = GUIDE.read_text(encoding="utf-8")

    def rpc(self, method: str, params: dict | None = None) -> dict:
        body = {"jsonrpc": "2.0", "id": 1, "method": method, **({"params": params} if params is not None else {})}
        return self.client.post("/mcp", json=body, headers={"Authorization": f"Bearer {self.token}"}).json()

    def test_the_guide_file_is_what_the_loader_reads(self) -> None:
        self.assertEqual(guide.guide_path(), GUIDE)
        self.assertEqual(guide.guide_text(), self.text)

    def test_route_returns_the_guide(self) -> None:
        for token in (self.token, "app-token"):
            res = self.client.get("/api/agent/guide", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(res.status_code, 200)
            self.assertIn("text/markdown", res.headers["content-type"])
            self.assertEqual(res.text, self.text)

    def test_route_needs_a_token(self) -> None:
        self.assertEqual(self.client.get("/api/agent/guide").status_code, 401)
        self.assertEqual(self.client.get("/api/agent/guide", headers={"Authorization": "Bearer nope"}).status_code, 401)

    def test_resource_is_listed_and_readable(self) -> None:
        uris = [r["uri"] for r in self.rpc("resources/list")["result"]["resources"]]
        self.assertIn("chaosengine://guide", uris)
        content = self.rpc("resources/read", {"uri": "chaosengine://guide"})["result"]["contents"][0]
        self.assertEqual(content["mimeType"], "text/markdown")
        self.assertEqual(content["text"], self.text)

    def test_resource_reads_with_a_read_only_token(self) -> None:
        read = access.make_read_token()
        body = {"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": {"uri": "chaosengine://guide"}}
        res = self.client.post("/mcp", json=body, headers={"Authorization": f"Bearer {read}"}).json()
        self.assertEqual(res["result"]["contents"][0]["text"], self.text)

    def test_start_prompt_tells_the_agent_to_read_the_guide(self) -> None:
        names = [p["name"] for p in self.rpc("prompts/list")["result"]["prompts"]]
        self.assertIn("start_here", names)
        got = self.rpc("prompts/get", {"name": "start_here", "arguments": {"task": "a poster of a lighthouse"}})["result"]
        text = got["messages"][0]["content"]["text"]
        self.assertIn("chaosengine://guide", text)
        self.assertIn("lighthouse", text)
        bare = self.rpc("prompts/get", {"name": "start_here"})["result"]["messages"][0]["content"]["text"]
        self.assertNotIn("Then do this", bare)

    def test_a_missing_file_gives_a_short_pointer_not_an_error(self) -> None:
        with mock.patch.object(guide, "guide_path", lambda: Path(self.tmp.name) / "nope.md"):
            self.assertIn("missing", guide.guide_text())


class GuideStagedTests(unittest.TestCase):
    def test_the_guide_is_in_the_staged_runtime_list(self) -> None:
        source = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        match = re.search(r"const BUNDLED_ROOT_DOCS = \[(.*?)\];", source, re.S)
        self.assertTrue(match)
        self.assertIn('"AGENTS.md"', match.group(1))
        self.assertIn("...BUNDLED_ROOT_DOCS", source)

    def test_the_staged_layout_puts_it_where_the_loader_looks(self) -> None:
        # Staged: <backend>/AGENTS.md beside <backend>/backend_service/mcp_bridge/guide.py.
        self.assertEqual(Path(guide.__file__).resolve().parents[2], GUIDE.parent)


if __name__ == "__main__":
    unittest.main()
