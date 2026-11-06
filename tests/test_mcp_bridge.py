import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.mcp_bridge import access
from backend_service.mcp_bridge.tools import TOOLS, validate


def rpc(client, method, params=None, rid=1, headers=None):
    body = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        body["params"] = params
    return client.post("/mcp", json=body, headers=headers or {})


def call(client, name, arguments=None, headers=None):
    response = rpc(client, "tools/call", {"name": name, "arguments": arguments or {}}, headers=headers)
    result = response.json()["result"]
    return result["isError"], result["content"][0]["text"]


class McpBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(access, "_path", lambda: Path(self.tmp.name) / "agent_access.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        self.client = TestClient(create_app(api_token="app-token"))

    def auth(self):
        return {"Authorization": f"Bearer {access.get_token()}"}

    def test_off_by_default_answers_without_token(self):
        init = rpc(self.client, "initialize", {"protocolVersion": "2025-03-26"}).json()["result"]
        self.assertEqual(init["protocolVersion"], "2025-03-26")
        self.assertEqual(len(rpc(self.client, "tools/list").json()["result"]["tools"]), len(TOOLS))
        is_error, text = call(self.client, "system_status")
        self.assertTrue(is_error)
        self.assertIn("Settings > Integrations > Agent access", text)

    def test_on_requires_token(self):
        access.set_enabled(True)
        self.assertEqual(rpc(self.client, "tools/list").status_code, 401)
        self.assertEqual(rpc(self.client, "tools/list", headers={"Authorization": "Bearer nope"}).status_code, 401)
        self.assertEqual(rpc(self.client, "tools/list", headers=self.auth()).status_code, 200)
        self.assertEqual(rpc(self.client, "ping", headers={"Authorization": "Bearer app-token"}).status_code, 200)

    def test_notification_and_unknown_method(self):
        response = self.client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(rpc(self.client, "nope").json()["error"]["code"], -32601)
        self.assertEqual(self.client.get("/mcp").status_code, 405)

    def test_access_endpoint_round_trip(self):
        headers = {"Authorization": "Bearer app-token"}
        state = self.client.get("/api/agent/access", headers=headers).json()
        self.assertFalse(state["enabled"])
        on = self.client.post("/api/agent/access", json={"enabled": True}, headers=headers).json()
        self.assertTrue(on["enabled"])
        rotated = self.client.post("/api/agent/access", json={"rotateToken": True}, headers=headers).json()
        self.assertNotEqual(rotated["token"], on["token"])
        self.assertTrue(rotated["mcpUrl"].endswith("/mcp"))

    def test_bad_input_is_a_readable_tool_error(self):
        access.set_enabled(True)
        for name, args, needle in (
            ("generate_image", {}, "missing required argument 'prompt'"),
            ("generate_image", {"prompt": "x", "width": 99999}, "between"),
            ("recommend_models", {"kind": "audio"}, "one of"),
            ("job_status", {"job_id": 5}, "must be string"),
            ("system_status", {"extra": 1}, "unknown argument"),
        ):
            is_error, text = call(self.client, name, args, self.auth())
            self.assertTrue(is_error, name)
            self.assertIn(needle, text)

    def test_every_schema_is_well_formed(self):
        for tool in TOOLS:
            schema = tool["inputSchema"]
            self.assertEqual(schema["type"], "object")
            for required in schema["required"]:
                self.assertIn(required, schema["properties"])
            self.assertIsNone(validate(schema, {r: _sample(schema["properties"][r]) for r in schema["required"]}))

    def test_system_status_and_recommend(self):
        access.set_enabled(True)
        is_error, text = call(self.client, "system_status", headers=self.auth())
        self.assertFalse(is_error)
        self.assertIn("freeMemoryGb", text)
        is_error, text = call(self.client, "recommend_models", {"kind": "image"}, self.auth())
        self.assertFalse(is_error)
        self.assertIn('"models"', text)

    def test_generate_image_runs_job_to_a_path(self):
        access.set_enabled(True)
        row = {"id": "black-forest-labs/FLUX.1-schnell", "fit": "fits"}
        artifact = {"imagePath": "/tmp/x.png", "width": 512, "height": 512, "seed": 4,
                    "modelName": "FLUX", "durationSeconds": 1.0, "runtimeNote": "Rendered at 512x512"}
        with mock.patch("backend_service.mcp_bridge.models.installed_models", return_value=[row]), \
                mock.patch("backend_service.routes.images._generate_image_locked", return_value={"artifacts": [artifact]}):
            is_error, text = call(self.client, "generate_image", {"prompt": "red cube", "width": 500}, self.auth())
            self.assertFalse(is_error, text)
            job_id = _field(text, "jobId")
            is_error, text = call(self.client, "wait_for_job", {"job_id": job_id, "timeout_seconds": 5}, self.auth())
        self.assertIn('"state": "done"', text)
        self.assertIn("/tmp/x.png", text)
        self.assertIn("Rendered at 512x512", text)

    def test_generate_image_without_model_points_to_recommend(self):
        access.set_enabled(True)
        with mock.patch("backend_service.mcp_bridge.models.installed_models", return_value=[]):
            is_error, text = call(self.client, "generate_image", {"prompt": "x"}, self.auth())
        self.assertTrue(is_error)
        self.assertIn("recommend_models", text)

    def test_failed_job_carries_the_refusal(self):
        from fastapi import HTTPException

        access.set_enabled(True)
        row = {"id": "black-forest-labs/FLUX.1-schnell", "fit": "fits"}
        refusal = HTTPException(status_code=503, detail="Only 2.0 GB free; need 9.0 GB.")
        with mock.patch("backend_service.mcp_bridge.models.installed_models", return_value=[row]), \
                mock.patch("backend_service.routes.images._generate_image_locked", side_effect=refusal):
            _, text = call(self.client, "generate_image", {"prompt": "x"}, self.auth())
            _, text = call(self.client, "wait_for_job", {"job_id": _field(text, "jobId"), "timeout_seconds": 5}, self.auth())
        self.assertIn('"state": "failed"', text)
        self.assertIn("Only 2.0 GB free", text)

    def test_video_seconds_become_valid_frames(self):
        access.set_enabled(True)
        row = {"id": "Lightricks/LTX-Video", "fit": "fits"}
        seen = {}

        def fake(request, body):
            seen["frames"] = body.numFrames
            return {"artifact": {"videoPath": "/tmp/c.mp4", "width": 768, "height": 512,
                                 "clipDurationSeconds": 4.0, "hasAudio": False}}

        with mock.patch("backend_service.mcp_bridge.models.installed_models", return_value=[row]), \
                mock.patch("backend_service.routes.video._generate_video_locked", side_effect=fake):
            _, text = call(self.client, "generate_video", {"prompt": "waves", "seconds": 3}, self.auth())
            _, text = call(self.client, "wait_for_job", {"job_id": _field(text, "jobId"), "timeout_seconds": 5}, self.auth())
        self.assertEqual((seen["frames"] - 1) % 8, 0)
        self.assertIn("Frame count 72 changed", text)

    def test_cancel_queued_job(self):
        from backend_service.helpers.job_queue import HEAVY_JOBS

        access.set_enabled(True)
        holder = HEAVY_JOBS.enqueue("video", "ui", "busy")
        HEAVY_JOBS.wait_turn(holder)
        row = {"id": "black-forest-labs/FLUX.1-schnell", "fit": "fits"}
        try:
            with mock.patch("backend_service.mcp_bridge.models.installed_models", return_value=[row]):
                _, text = call(self.client, "generate_image", {"prompt": "x"}, self.auth())
            job_id = _field(text, "jobId")
            _, status = call(self.client, "job_status", {"job_id": job_id}, self.auth())
            self.assertIn('"queuePosition": 1', status)
            _, text = call(self.client, "cancel_job", {"job_id": job_id}, self.auth())
            _, text = call(self.client, "wait_for_job", {"job_id": job_id, "timeout_seconds": 5}, self.auth())
            self.assertIn('"state": "cancelled"', text)
        finally:
            HEAVY_JOBS.release(holder)


def _sample(spec):
    return {"string": "x", "integer": 1, "number": 1, "boolean": True}[spec["type"]] if "enum" not in spec else spec["enum"][0]


def _field(text, name):
    import json

    return json.loads(text)[name]


if __name__ == "__main__":
    unittest.main()
