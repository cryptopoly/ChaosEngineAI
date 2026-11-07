"""Music routes and agent tool, end to end through the real worker with the fake engine.

Route -> HEAVY_JOBS slot -> memory gate -> runtime -> client -> worker -> fake engine -> FLAC tags ->
Music Library, plus validation limits, the queue note, cancel, the gate refusal, the audio route's
Range support, the estimate strip, ``make_music`` over ``/mcp``, ``list_models kind music`` and the
unknown-kind fix in ``mcp_bridge/models.py``.
"""

from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_service import music_client, music_env, music_runtime
from backend_service.helpers.audio_metadata import read_audio_settings
from backend_service.helpers.job_queue import HEAVY_JOBS
from backend_service.progress import MUSIC_PROGRESS
from tests.music_fakes import MusicSandbox

INSTRUMENTAL = {"caption": "warm lo-fi piano", "instrumental": True, "seconds": 30, "seed": 5}


class _RouteBox(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_video_routes import make_client

        self.client, self._tmp, self._snapshot = make_client()
        self.box = MusicSandbox(flac=True).__enter__()
        self.worker = music_client.MusicWorker()
        self.box.patch(music_client, "get_worker", lambda: self.worker)
        self.box.patch(music_client, "_WORKER", self.worker)
        self.box.patch(music_env, "lane_unavailable_reason", lambda: None)
        self.box.patch(music_runtime, "embed_tag_enabled", lambda: True)
        self.box.patch(music_runtime, "_prepare", lambda variant: None)
        self.box.patch(music_runtime, "readiness_error", lambda variant: None)
        from backend_service.helpers import music as music_helpers

        self.box.patch(music_helpers, "weights_source", lambda variant: "checkpoints")

    def tearDown(self) -> None:
        from tests.test_video_routes import restore_env

        self.worker.stop()
        self.box.__exit__(None, None, None)
        restore_env(self._snapshot)
        self._tmp.cleanup()


class MusicRouteTests(_RouteBox):
    def test_generate_saves_tagged_tracks_into_the_library(self) -> None:
        response = self.client.post("/api/music/generate", json={**INSTRUMENTAL, "variations": 2, "title": "rain"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual([i["seed"] for i in body["items"]], [5, 6])
        self.assertEqual([i["title"] for i in body["items"]], ["rain (1)", "rain (2)"])
        self.assertEqual(body["seed"], 5)
        self.assertFalse(MUSIC_PROGRESS.snapshot()["active"])
        listed = self.client.get("/api/music/library").json()
        self.assertEqual(listed["count"], 2)
        item = body["items"][0]
        path = music_env.library_dir() / f"{item['id']}.flac"
        settings = read_audio_settings(path)
        self.assertEqual((settings["seed"], settings["model"], settings["machine_generated"]), (5, "ACE-Step/Ace-Step1.5", True))
        self.assertTrue(item["tagged"])
        self.assertFalse(list((self.box.data / "tmp").iterdir()))  # the job folder is cleaned up

    def test_the_tag_switch_off_leaves_files_untagged(self) -> None:
        with mock.patch.object(music_runtime, "embed_tag_enabled", lambda: False):
            item = self.client.post("/api/music/generate", json=INSTRUMENTAL).json()["items"][0]
        self.assertFalse(item["tagged"])
        self.assertIsNone(read_audio_settings(music_env.library_dir() / f"{item['id']}.flac"))

    def test_lyrics_are_kept_in_the_library_not_in_the_file(self) -> None:
        item = self.client.post("/api/music/generate", json={"caption": "folk", "lyrics": "[verse]\nsecret line", "seconds": 20}).json()["items"][0]
        self.assertEqual(item["lyrics"], "[verse]\nsecret line")
        flac = (music_env.library_dir() / f"{item['id']}.flac").read_bytes()
        self.assertNotIn(b"secret line", flac)

    def test_validation_limits(self) -> None:
        bad = [
            {**INSTRUMENTAL, "caption": "x" * 513},
            {**INSTRUMENTAL, "caption": "   "},
            {"caption": "song", "lyrics": "x" * 4097, "seconds": 30},
            {"caption": "song", "seconds": 30},  # no lyrics and not instrumental
            {**INSTRUMENTAL, "seconds": 9},
            {**INSTRUMENTAL, "seconds": 601},
            {**INSTRUMENTAL, "variations": 3},
            {**INSTRUMENTAL, "bpm": 29},
            {**INSTRUMENTAL, "bpm": 301},
            {**INSTRUMENTAL, "key": "H minor"},
            {**INSTRUMENTAL, "seed": -1},
            {**INSTRUMENTAL, "seed": 2**31},
            {**INSTRUMENTAL, "task": "cover"},
            {**INSTRUMENTAL, "reference_audio": "/tmp/x.wav"},
            {**INSTRUMENTAL, "timeSignature": 5},
        ]
        for payload in bad:
            self.assertEqual(self.client.post("/api/music/generate", json=payload).status_code, 422, payload)
        ok = self.client.post("/api/music/generate", json={**INSTRUMENTAL, "key": "F# major", "bpm": 300, "seconds": 600,
                                                           "timeSignature": 4, "seed": 2**31 - 1})
        self.assertEqual(ok.status_code, 200, ok.text)

    def test_unknown_model_and_not_ready(self) -> None:
        self.assertEqual(self.client.post("/api/music/generate", json={**INSTRUMENTAL, "modelId": "a/b"}).status_code, 400)
        with mock.patch.object(music_runtime, "readiness_error", lambda variant: "The music pack is not installed."):
            response = self.client.post("/api/music/generate", json=INSTRUMENTAL)
        self.assertEqual((response.status_code, response.json()["detail"]), (409, "The music pack is not installed."))

    def test_memory_gate_refuses_with_the_fix(self) -> None:
        with mock.patch("backend_service.helpers.memory_gate.snapshot_memory_signals", return_value=(10.0, 60.0)):
            response = self.client.post("/api/music/generate", json=INSTRUMENTAL)
        self.assertEqual(response.status_code, 503)
        self.assertIn("about 23 GB", response.json()["detail"])
        self.assertIn("Unload", response.json()["detail"])

    def test_queue_note_and_cancel_of_a_waiting_studio_job(self) -> None:
        holder = HEAVY_JOBS.enqueue("image", "agent", "busy")
        HEAVY_JOBS.wait_turn(holder)
        result: dict = {}
        thread = threading.Thread(target=lambda: result.update(r=self.client.post("/api/music/generate", json=INSTRUMENTAL)))
        try:
            thread.start()
            deadline = time.monotonic() + 5
            queue = {}
            while time.monotonic() < deadline:
                queue = self.client.get("/api/music/progress").json()["progress"]["queue"]
                if queue["waiting"]:
                    break
                time.sleep(0.05)
            self.assertEqual(queue, {"waiting": True, "position": 1, "agentAhead": True})
            self.assertTrue(self.client.post("/api/music/cancel").json()["cancelled"])
            thread.join(timeout=10)
        finally:
            HEAVY_JOBS.release(holder)
        self.assertEqual(result["r"].status_code, 409)

    def test_cancel_a_running_studio_job(self) -> None:
        result: dict = {}
        thread = threading.Thread(target=lambda: result.update(
            r=self.client.post("/api/music/generate", json={**INSTRUMENTAL, "caption": "delay=0.4"})))
        thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and MUSIC_PROGRESS.snapshot()["step"] < 1:
            time.sleep(0.05)
        self.assertTrue(self.client.post("/api/music/cancel").json()["cancelled"])
        thread.join(timeout=15)
        self.assertEqual(result["r"].status_code, 409)
        self.assertEqual(self.client.get("/api/music/library").json()["count"], 0)
        self.assertTrue(self.worker.running())  # a cooperative cancel keeps the worker

    def test_audio_route_supports_range_and_refuses_bad_ids(self) -> None:
        item = self.client.post("/api/music/generate", json=INSTRUMENTAL).json()["items"][0]
        full = self.client.get(f"/api/music/library/{item['id']}/audio")
        self.assertEqual((full.status_code, full.headers["content-type"]), (200, "audio/flac"))
        self.assertEqual(full.headers.get("accept-ranges"), "bytes")
        part = self.client.get(f"/api/music/library/{item['id']}/audio", headers={"Range": "bytes=0-99"})
        self.assertEqual(part.status_code, 206)
        self.assertEqual(len(part.content), 100)
        self.assertTrue(part.headers["content-range"].startswith("bytes 0-99/"))
        self.assertEqual(part.content, full.content[:100])
        self.assertEqual(self.client.get("/api/music/library/..%2F..%2Fetc/audio").status_code, 404)
        self.assertEqual(self.client.get("/api/music/library/123_zzz/audio").status_code, 400)
        self.assertEqual(self.client.get("/api/music/library/123_01234567/audio").status_code, 404)

    def test_library_rename_tag_and_delete_routes(self) -> None:
        item = self.client.post("/api/music/generate", json=INSTRUMENTAL).json()["items"][0]
        updated = self.client.post(f"/api/music/library/{item['id']}", json={"title": "New name", "tags": ["calm"]}).json()
        self.assertEqual((updated["title"], updated["tags"]), ("New name", ["calm"]))
        self.assertEqual(self.client.get("/api/music/library", params={"tag": "calm"}).json()["count"], 1)
        self.assertEqual(self.client.post("/api/music/library/delete", json={"ids": [item["id"]]}).json()["deleted"], [item["id"]])
        self.assertEqual(self.client.get(f"/api/music/library/{item['id']}").status_code, 404)
        self.assertEqual(self.client.post("/api/music/library/delete", json={"ids": ["../x"]}).status_code, 400)

    def test_catalog_runtime_estimate_and_unload(self) -> None:
        families = self.client.get("/api/music/catalog").json()["families"]
        variant = families[0]["variants"][0]
        for key in ("licence", "makes", "platforms", "estimates", "fit", "availableLocally", "packInstalled"):
            self.assertIn(key, variant)
        runtime = self.client.get("/api/music/runtime").json()["runtime"]
        for key in ("lane", "installed", "workerRunning", "loadedModel", "install", "engineVersion", "embedTag"):
            self.assertIn(key, runtime)
        cold = self.client.post("/api/music/estimate", json={"seconds": 60}).json()
        self.assertFalse(cold["estimate"]["warm"])
        self.assertEqual(cold["estimate"]["basis"], "measured")
        self.assertIn("fit", cold)
        self.client.post("/api/music/generate", json=INSTRUMENTAL)
        warm = self.client.post("/api/music/estimate", json={"seconds": 60}).json()["estimate"]
        self.assertTrue(warm["warm"])
        self.assertEqual(warm["seconds"], 5.9)
        self.assertTrue(self.client.post("/api/music/unload").json()["stopped"])
        self.assertFalse(self.worker.running())

    def test_install_and_download_routes(self) -> None:
        with mock.patch.object(music_env, "start_install", return_value={"phase": "preflight"}):
            self.assertEqual(self.client.post("/api/music/install").json()["phase"], "preflight")
        self.assertIn("phase", self.client.get("/api/music/install/status").json())
        self.assertEqual(self.client.post("/api/music/download", json={"repo": "black-forest-labs/FLUX.1-dev"}).status_code, 400)
        self.assertEqual(self.client.get("/api/music/download/status").json(), {"downloads": []})

    def test_an_idle_music_worker_makes_room_for_images(self) -> None:
        self.worker.request("ping")
        self.assertTrue(music_runtime.free_idle_worker())
        self.assertFalse(self.worker.running())


class MusicAgentTests(_RouteBox):
    def setUp(self) -> None:
        super().setUp()
        import tempfile

        from backend_service.mcp_bridge import access

        self._access_tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(access, "_path", lambda: Path(self._access_tmp.name) / "agent_access.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._access_tmp.cleanup)
        access.set_enabled(True)
        self.headers = {"Authorization": f"Bearer {access.get_token()}"}

    def _call(self, name: str, arguments: dict) -> tuple[bool, object]:
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
        result = self.client.post("/mcp", json=body, headers=self.headers).json()["result"]
        text = result["content"][0]["text"]
        try:
            return result["isError"], json.loads(text)
        except ValueError:
            return result["isError"], text

    def test_list_models_kind_music(self) -> None:
        is_error, out = self._call("list_models", {"kind": "music"})
        self.assertFalse(is_error, out)
        ids = [m["id"] for m in out["models"]]
        self.assertIn("ACE-Step/Ace-Step1.5", ids)
        self.assertEqual(out["models"][0]["licence"], "MIT")

    def test_make_music_job_to_a_fetchable_path(self) -> None:
        is_error, job = self._call("make_music", {"caption": "lo-fi", "instrumental": True, "seconds": 20, "seed": 9})
        self.assertFalse(is_error, job)
        self.assertEqual(job["kind"], "music")
        is_error, done = self._call("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 30})
        self.assertEqual(done["state"], "done", done)
        output = done["output"]
        self.assertEqual(output["seed"], 9)
        self.assertTrue(Path(output["path"]).is_relative_to(music_env.library_dir()))
        self.assertIn("download", output)
        fetched = self.client.get(output["download"]["url"])
        self.assertEqual(fetched.status_code, 200, fetched.text)
        self.assertEqual(len(fetched.content), output["download"]["bytes"])

    def test_make_music_errors_are_readable(self) -> None:
        self.assertEqual(self._call("make_music", {"caption": "x", "instrumental": True, "variations": 3}),
                         (True, "variations must be 1, 2 or 4."))
        is_error, text = self._call("make_music", {"caption": "x"})
        self.assertTrue(is_error)
        self.assertIn("lyrics", str(text))
        is_error, text = self._call("make_music", {"caption": "x", "instrumental": True, "model": "a/b"})
        self.assertTrue(is_error)
        self.assertIn("not an installed music model", str(text))

    def test_cancel_job_stops_a_running_agent_music_job(self) -> None:
        _, job = self._call("make_music", {"caption": "delay=0.4", "instrumental": True, "seconds": 20})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and MUSIC_PROGRESS.snapshot()["step"] < 1:
            time.sleep(0.05)
        # A Studio Cancel leaves the agent's job alone.
        self.assertFalse(self.client.post("/api/music/cancel").json()["cancelled"])
        self.assertTrue(self._call("cancel_job", {"job_id": job["jobId"]})[1]["cancelRequested"])
        _, done = self._call("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 20})
        self.assertEqual(done["state"], "cancelled")


class BridgeModelsTests(unittest.TestCase):
    def test_an_unknown_kind_raises_instead_of_becoming_video(self) -> None:
        from backend_service.mcp_bridge import models

        with self.assertRaises(ValueError):
            models.installed_models(object(), "sculpture")
        with self.assertRaises(ValueError):
            models._families(object(), "sculpture")
        with self.assertRaises(ValueError):
            models.recommend(object(), "sculpture")

    def test_tool_schema_and_roots(self) -> None:
        from backend_service.mcp_bridge import files
        from backend_service.mcp_bridge.tools import HANDLERS, TOOLS

        tool = next(t for t in TOOLS if t["name"] == "make_music")
        self.assertEqual(tool["inputSchema"]["required"], ["caption"])
        self.assertIn("make_music", HANDLERS)
        kind = next(t for t in TOOLS if t["name"] == "list_models")["inputSchema"]["properties"]["kind"]
        self.assertIn("music", kind["enum"])
        self.assertIn(music_env.library_dir().resolve(), files.output_roots())


if __name__ == "__main__":
    unittest.main()
