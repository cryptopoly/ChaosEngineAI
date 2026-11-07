"""Agent bridge v2, second wave: clone tools behind the clone scope, captions and OTIO tools, music_library,
resources and prompts with paging, and progress notifications over a short event stream."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from backend_service import music_env, voice_clone, voice_clone_safety, voice_env, voice_runtime
from backend_service.app import create_app
from backend_service.mcp_bridge import access, files, resources
from backend_service.routes import agent as agent_routes
from backend_service.mcp_bridge.jobs import STORE
from tests.storyboard_fixtures import fake_mp4, fake_wav


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.src = self.base / "src"
        self.src.mkdir()
        (self.base / "music").mkdir()
        patches = [
            mock.patch.dict(os.environ, {"CHAOSENGINE_STORYBOARDS_DIR": str(self.base / "storyboards")}),
            mock.patch.object(access, "_path", lambda: self.base / "agent_access.json"),
            mock.patch.object(voice_env, "VOICE_DATA_DIR", self.base / "voice"),
            mock.patch.object(music_env, "library_dir", lambda: self.base / "music"),
            mock.patch.object(files, "output_roots",
                              lambda: [p.resolve() for p in (self.src, self.base / "music", *files._storyboard_exports())]),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(create_app(api_token="app-token"))
        access.set_enabled(True)
        self.token = access.get_token()
        # The test client is not "this computer" to the bridge; these tests are about a local agent.
        patcher = mock.patch.object(agent_routes, "_is_loopback", lambda request: True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def rpc(self, method, params=None, token=None, headers=None):
        body = {"jsonrpc": "2.0", "id": 1, "method": method, **({"params": params} if params is not None else {})}
        head = {"Authorization": f"Bearer {token or self.token}", **(headers or {})}
        return self.client.post("/mcp", json=body, headers=head)

    def call(self, name, arguments=None, error=False, token=None):
        result = self.rpc("tools/call", {"name": name, "arguments": arguments or {}}, token).json()["result"]
        text = result["content"][0]["text"]
        self.assertEqual(result["isError"], error, text)
        return text if error else json.loads(text)


class CloneToolTests(_Base):
    def test_clone_tools_listed_only_with_the_clone_scope(self):
        names = lambda tok=None: {t["name"] for t in self.rpc("tools/list", token=tok).json()["result"]["tools"]}  # noqa: E731
        self.assertNotIn("speak_cloned", names())
        access.set_scopes(["read", "generate", "files", "clone"])
        self.assertIn("speak_cloned", names())
        self.assertIn("list_cloned_voices", names())
        self.assertNotIn("speak_cloned", names(access.make_read_token()))

    def test_scope_then_setting_both_needed(self):
        text = self.call("list_cloned_voices", error=True)
        self.assertIn("'clone' scope", text)
        access.set_scopes(["read", "generate", "files", "clone"])
        self.assertIn("Allow agents", self.call("list_cloned_voices", error=True))  # the setting is still off
        voice_runtime.save_settings({"allowAgentCloning": True})
        self.assertEqual(self.call("list_cloned_voices"), {"voices": []})

    def test_speak_cloned_needs_an_existing_consented_voice_and_never_creates_one(self):
        access.set_scopes(["read", "generate", "files", "clone"])
        voice_runtime.save_settings({"allowAgentCloning": True})
        self.assertIn("not found", self.call("speak_cloned", {"voice_id": "1234567890_deadbeef", "text": "Hi"}, error=True))
        self.assertFalse(hasattr(voice_clone, "create_voice_from_agent"))
        names = {t["name"] for t in self.rpc("tools/list").json()["result"]["tools"]}
        self.assertFalse({n for n in names if "clone" in n and n not in ("list_cloned_voices", "speak_cloned")})
        with mock.patch.object(voice_clone, "get_voice", return_value={"id": "1234567890_deadbeef"}), \
             mock.patch.object(voice_clone, "speak", side_effect=RuntimeError("no worker in tests")):
            job = self.call("speak_cloned", {"voice_id": "1234567890_deadbeef", "text": "Hi"})
            self.assertEqual(job["kind"], "voice")
            self.assertTrue(STORE.get(job["jobId"]).done.wait(10))

    def test_token_without_clone_scope_is_refused_by_the_helper_even_when_the_setting_is_on(self):
        voice_runtime.save_settings({"allowAgentCloning": True})
        read = access.make_read_token()
        request = SimpleNamespace(url=SimpleNamespace(path="/api/voice/clone/speak"), client=SimpleNamespace(host="127.0.0.1"),
                                  headers={"authorization": f"Bearer {read}"})
        self.assertFalse(voice_clone_safety.clone_allowed(request))
        access.set_scopes(["read", "generate", "files", "clone"])
        request.headers = {"authorization": f"Bearer {self.token}"}
        self.assertTrue(voice_clone_safety.clone_allowed(request))
        request.headers = {}  # the app's own window carries no agent token
        self.assertTrue(voice_clone_safety.clone_allowed(request))


class AgentFileRouteAuthTests(_Base):
    def test_agent_token_reaches_the_file_routes_while_app_routes_stay_app_token_only(self):
        self.client.app.state.chaosengine_require_api_auth = True
        head = {"Authorization": f"Bearer {self.token}"}
        # The file routes do their own token and scope check; before, the app-token middleware answered 401 first.
        self.assertEqual(self.client.get("/api/agent/files", params={"path": "/nope"}, headers=head).status_code, 404)
        self.assertEqual(self.client.get("/api/agent/files", params={"path": "/nope"}).status_code, 401)
        self.assertEqual(self.client.get("/api/agent/access", headers=head).status_code, 401)  # still the app's own


class StoryboardV2Tests(_Base):
    def project(self):
        info = self.call("storyboard_project", {"action": "create", "name": "Ad"})
        return self.call("storyboard_shot", {"action": "add", "project_id": info["projectId"], "revision": info["revision"],
                                             "fields": {"title": "s", "stillPrompt": "sea", "seconds": 3}})

    def test_edit_project_export_booleans_and_caption_ops(self):
        info = self.project()
        pid = info["projectId"]
        info = self.call("storyboard_project", {"action": "edit", "project_id": pid, "revision": info["revision"],
                                                "fields": {"export": {"captionsSrt": True, "captionsBurn": True, "stems": True}}})
        project = self.call("storyboard_project", {"action": "get", "project_id": pid})["project"]
        self.assertEqual({k: project["export"][k] for k in ("captionsSrt", "captionsBurn", "stems")},
                         {"captionsSrt": True, "captionsBurn": True, "stems": True})
        rev = info["revision"]
        self.assertIn("no captions", self.call("storyboard_captions", {"action": "export_srt", "project_id": pid}, error=True))
        self.call("storyboard_captions", {"action": "add", "project_id": pid, "revision": rev, "start": 0, "end": 1.5, "text": "Hello"})
        caption = self.call("storyboard_project", {"action": "get", "project_id": pid})["project"]["captions"][0]
        rev = self.call("storyboard_project", {"action": "get", "project_id": pid})["project"]["revision"]
        self.call("storyboard_captions", {"action": "edit", "project_id": pid, "revision": rev, "caption_id": caption["id"], "text": "Hi there"})
        self.assertIn("revision", self.call("storyboard_captions", {"action": "edit", "project_id": pid, "revision": rev,
                                                                    "caption_id": caption["id"], "text": "x"}, error=True))

    def test_set_remove_and_export_srt_download(self):
        info = self.project()
        pid = info["projectId"]
        info = self.call("storyboard_captions", {"action": "set", "project_id": pid, "revision": info["revision"],
                                                 "cues": [{"start": 0, "end": 1, "text": "One"}, {"start": 1, "end": 2, "text": "Two"}]})
        project = self.call("storyboard_project", {"action": "get", "project_id": pid})["project"]
        self.assertEqual(len(project["captions"]), 2)
        shot = self.call("storyboard_shot", {"action": "import", "project_id": pid, "revision": project["revision"],
                                             "shot_id": project["shots"][0]["id"], "path": str(fake_mp4(self.src / "c.mp4"))})
        srt = self.call("storyboard_captions", {"action": "export_srt", "project_id": pid})
        self.assertTrue(srt["file"].endswith(".srt"))
        self.assertEqual(self.client.get(srt["download"]["url"], headers={"Authorization": f"Bearer {self.token}"}).status_code, 200)
        rev = shot["revision"]
        cid = project["captions"][0]["id"]
        self.call("storyboard_captions", {"action": "remove", "project_id": pid, "revision": rev, "caption_id": cid})

    def test_export_otio_returns_download_not_path(self):
        info = self.project()
        pid = info["projectId"]
        info = self.call("storyboard_shot", {"action": "import", "project_id": pid, "revision": info["revision"],
                                             "shot_id": info["shots"][0]["id"], "path": str(fake_mp4(self.src / "c.mp4"))})
        out = self.call("storyboard_export_otio", {"project_id": pid})
        self.assertTrue(out["file"].endswith(".otio"))
        self.assertNotIn(str(self.base), json.dumps({k: v for k, v in out.items() if k != "download"}))
        self.assertIn("warnings", out)
        self.assertTrue(self.client.get(out["download"]["url"], headers={"Authorization": f"Bearer {self.token}"}).content)

    def test_render_passes_notify_and_reports_extras_and_resume(self):
        info = self.project()
        pid = info["projectId"]
        info = self.call("storyboard_shot", {"action": "import", "project_id": pid, "revision": info["revision"],
                                             "shot_id": info["shots"][0]["id"], "path": str(fake_mp4(self.src / "c.mp4"))})
        seen = {}
        from backend_service.routes import storyboard as routes

        real_start = routes.start_render

        def spy(project_id, revision, preview, notify=False):
            seen["notify"] = notify
            return real_start(project_id, revision, preview, notify)

        def fake_render(plan, progress, cancel_event):
            Path(plan["out_path"]).write_bytes(b"film")

        with mock.patch.object(routes, "start_render", spy), mock.patch("backend_service.storyboard.render_adapter.render", fake_render):
            job = self.call("storyboard_render", {"project_id": pid, "revision": info["revision"], "preview": False, "notify": True})
            done = self.call("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 10})
        self.assertTrue(seen["notify"])
        self.assertEqual(done["state"], "done", done)
        self.assertEqual(done["output"]["extras"], [])

    def test_imported_audio_with_the_cloned_tag_is_forced_to_clonedVoice(self):
        info = self.project()
        pid = info["projectId"]
        tagged = self.src / "tagged.wav"
        tagged.write_bytes(voice_clone_safety.tag_wav(fake_wav(self.src / "raw.wav").read_bytes(), {"voice": "x"}, "test"))
        plain = fake_wav(self.src / "plain.wav")
        info = self.call("storyboard_audio", {"action": "add", "project_id": pid, "revision": info["revision"], "kind": "voice",
                                              "path": str(tagged), "fields": {"clonedVoice": False}})
        info = self.call("storyboard_audio", {"action": "add", "project_id": pid, "revision": info["revision"], "kind": "voice", "path": str(plain)})
        audio = self.call("storyboard_project", {"action": "get", "project_id": pid})["project"]["audio"]
        self.assertEqual([a["source"]["clonedVoice"] for a in audio], [True, False])


class MusicLibraryToolTests(_Base):
    def make(self, title, tags=()):
        from backend_service import music_library

        audio = self.base / f"{title}.flac"
        audio.write_bytes(b"fLaC....")
        return music_library.create({"title": title, "caption": "calm", "tags": list(tags)}, audio)

    def test_list_get_rename_delete(self):
        a, b = self.make("Alpha", ["calm"]), self.make("Beta")
        listed = self.call("music_library", {"action": "list"})
        self.assertEqual(listed["total"], 2)
        self.assertEqual(self.call("music_library", {"action": "list", "tag": "calm"})["tracks"][0]["id"], a["id"])
        page = self.call("music_library", {"action": "list", "limit": 1})
        self.assertEqual(page["nextOffset"], 1)
        got = self.call("music_library", {"action": "get", "track_id": b["id"]})
        self.assertTrue(got["download"]["url"].startswith("/api/agent/files"))
        renamed = self.call("music_library", {"action": "rename", "track_id": b["id"], "title": "Gamma", "tags": ["x"]})
        self.assertEqual((renamed["title"], renamed["tags"]), ("Gamma", ["x"]))
        self.assertEqual(self.call("music_library", {"action": "delete", "track_id": a["id"]}), {"deleted": a["id"]})
        self.assertIn("Unknown track", self.call("music_library", {"action": "delete", "track_id": a["id"]}, error=True))
        self.assertIn("needs track_id", self.call("music_library", {"action": "get"}, error=True))

    def test_read_token_may_list_but_not_delete(self):
        track = self.make("Alpha")
        read = access.make_read_token()
        self.assertEqual(self.call("music_library", {"action": "list"}, token=read)["total"], 1)
        self.assertIn("scope", self.call("music_library", {"action": "delete", "track_id": track["id"]}, error=True, token=read))


class ResourcesAndPromptsTests(_Base):
    def test_initialize_advertises_resources_and_prompts(self):
        caps = self.rpc("initialize", {"protocolVersion": "2025-06-18"}).json()["result"]["capabilities"]
        self.assertIn("resources", caps)
        self.assertIn("prompts", caps)

    def test_resources_list_read_and_templates(self):
        info = self.call("storyboard_project", {"action": "create", "name": "Ad"})
        listed = self.rpc("resources/list").json()["result"]["resources"]
        uris = [r["uri"] for r in listed]
        self.assertIn("chaosengine://system/status", uris)
        self.assertIn(f"chaosengine://storyboard/{info['projectId']}", uris)
        film = self.rpc("resources/read", {"uri": f"chaosengine://storyboard/{info['projectId']}"}).json()["result"]["contents"][0]
        self.assertEqual(json.loads(film["text"])["project"]["name"], "Ad")
        status = self.rpc("resources/read", {"uri": "chaosengine://system/status"}).json()["result"]["contents"][0]
        self.assertIn("queue", json.loads(status["text"]))
        self.assertIn("outputs", json.loads(self.rpc("resources/read", {"uri": "chaosengine://gallery/recent"}).json()["result"]["contents"][0]["text"]))
        templates = self.rpc("resources/templates/list").json()["result"]["resourceTemplates"]
        self.assertEqual(len(templates), 2)

    def test_unknown_and_malformed_uris_are_errors_and_no_path_escapes(self):
        for uri in ("chaosengine://storyboard/../../etc", "chaosengine://storyboard/nope", "file:///etc/hosts", "chaosengine://music/x"):
            body = self.rpc("resources/read", {"uri": uri}).json()
            self.assertIn("error", body, uri)

    def test_paging_with_a_cursor(self):
        with mock.patch.object(resources, "PAGE", 2):
            first = self.rpc("resources/list").json()["result"]
            self.assertEqual(len(first["resources"]), 2)
            second = self.rpc("resources/list", {"cursor": first["nextCursor"]}).json()["result"]
            self.assertEqual(second["resources"][0]["uri"], "chaosengine://gallery/recent")
            self.assertNotIn("nextCursor", second)
            self.assertIn("error", self.rpc("resources/list", {"cursor": "abc"}).json())
        with mock.patch.object(resources, "PAGE", 2):
            page = self.rpc("prompts/list").json()["result"]
            self.assertEqual(page["nextCursor"], "2")
            self.assertEqual(self.rpc("prompts/list", {"cursor": "2"}).json()["result"]["prompts"][0]["name"], "soundtrack_for_project")

    def test_prompts(self):
        names = [p["name"] for p in self.rpc("prompts/list").json()["result"]["prompts"]]
        self.assertEqual(names, ["short_film_from_brief", "poster_set", "soundtrack_for_project"])
        got = self.rpc("prompts/get", {"name": "poster_set", "arguments": {"subject": "a lighthouse", "count": "4"}}).json()["result"]
        text = got["messages"][0]["content"]["text"]
        self.assertIn("lighthouse", text)
        self.assertIn("generate_image", text)
        self.assertIn("error", self.rpc("prompts/get", {"name": "poster_set", "arguments": {}}).json())
        self.assertIn("error", self.rpc("prompts/get", {"name": "nope"}).json())

    def test_resources_need_a_valid_token(self):
        body = {"jsonrpc": "2.0", "id": 1, "method": "resources/list"}
        self.assertEqual(self.client.post("/mcp", json=body).status_code, 401)
        self.assertEqual(self.rpc("resources/list", token=access.make_read_token()).status_code, 200)


class ProgressTests(_Base):
    def test_plain_json_without_a_token_and_event_stream_with_one(self):
        job = STORE.submit("voice", "fake", lambda: ({"ok": True}, []))
        job.done.wait(5)
        plain = self.rpc("tools/call", {"name": "wait_for_job", "arguments": {"job_id": job.id}})
        self.assertTrue(plain.headers["content-type"].startswith("application/json"))
        streamed = self.rpc("tools/call", {"name": "job_status", "arguments": {"job_id": job.id}, "_meta": {"progressToken": "t1"}})
        self.assertTrue(streamed.headers["content-type"].startswith("text/event-stream"))
        events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(events[-1]["id"], 1)
        self.assertIn("result", events[-1])

    def test_wait_for_job_sends_rising_progress_notifications(self):
        import threading

        gate = threading.Event()
        job = STORE.submit("voice", "slow", lambda: (gate.wait(10) and ({"ok": True}, [])) or ({"ok": True}, []))
        timer = threading.Timer(2.5, gate.set)
        timer.start()
        self.addCleanup(timer.cancel)
        streamed = self.rpc("tools/call", {"name": "wait_for_job", "arguments": {"job_id": job.id, "timeout_seconds": 20},
                                           "_meta": {"progressToken": 7}})
        events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
        notes = [e for e in events if e.get("method") == "notifications/progress"]
        self.assertGreaterEqual(len(notes), 1)
        values = [n["params"]["progress"] for n in notes]
        self.assertEqual(values, sorted(set(values)))
        self.assertTrue(all(n["params"]["progressToken"] == 7 for n in notes))
        final = events[-1]
        self.assertEqual(json.loads(final["result"]["content"][0]["text"])["state"], "done")

    def test_errors_still_end_the_stream_with_an_answer(self):
        streamed = self.rpc("tools/call", {"name": "wait_for_job", "arguments": {"job_id": "nope"}, "_meta": {"progressToken": "x"}})
        events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
        self.assertTrue(events[-1]["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
