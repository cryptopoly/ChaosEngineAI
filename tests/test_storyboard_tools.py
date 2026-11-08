"""Storyboard agent tools over ``/mcp``: grouped verbs, revisions, generate and render as jobs with downloads."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.mcp_bridge import access, files
from backend_service.mcp_bridge.storyboard_tools import STORYBOARD_TOOLS
from tests.storyboard_fixtures import FakeGenerators, fake_mp4, fake_wav, png
from tests.test_storyboard_routes import fake_wiring


class StoryboardToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.src = self.base / "src"
        self.src.mkdir()
        patches = [
            mock.patch.dict(os.environ, {"CHAOSENGINE_STORYBOARDS_DIR": str(self.base / "storyboards")}),
            mock.patch.object(access, "_path", lambda: self.base / "agent_access.json"),
            # The test's source folder stands in for the app's output folders (agents may import only from those).
            mock.patch.object(files, "output_roots", lambda: [p.resolve() for p in (self.src, *files._storyboard_exports())]),
            mock.patch.object(files, "inbox_dir", self._inbox),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(create_app(api_token="app-token"))
        access.set_enabled(True)
        self.headers = {"Authorization": f"Bearer {access.get_token()}"}

    def _inbox(self):
        path = self.base / "agent-inbox"
        path.mkdir(exist_ok=True)
        return path

    def call(self, name, arguments, error=False):
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
        result = self.client.post("/mcp", json=body, headers=self.headers).json()["result"]
        text = result["content"][0]["text"]
        self.assertEqual(result["isError"], error, text)
        return text if error else json.loads(text)

    def project(self, shots=2):
        info = self.call("storyboard_project", {"action": "create", "name": "Ad"})
        for i in range(shots):
            info = self.call("storyboard_shot", {"action": "add", "project_id": info["projectId"], "revision": info["revision"],
                                                 "fields": {"title": f"s{i}", "stillPrompt": "sea", "seconds": 3}})
        return info

    def test_descriptions_stay_short_and_names_are_grouped(self):
        self.assertEqual([t["name"] for t in STORYBOARD_TOOLS],
                         ["storyboard_project", "storyboard_shot", "storyboard_audio", "storyboard_generate", "storyboard_render",
                          "storyboard_captions", "storyboard_export_otio"])
        for tool in STORYBOARD_TOOLS:
            self.assertLess(len(tool["description"]), 220, tool["name"])

    def test_project_lifecycle(self):
        info = self.project(shots=1)
        pid = info["projectId"]
        self.assertEqual(self.call("storyboard_project", {"action": "list"})["projects"][0]["id"], pid)
        got = self.call("storyboard_project", {"action": "get", "project_id": pid})
        self.assertEqual(got["project"]["shots"][0]["title"], "s0")
        info = self.call("storyboard_project", {"action": "edit", "project_id": pid, "revision": info["revision"],
                                                "fields": {"defaults": {"fps": 25}, "export": {"aiLabel": {"embedTag": False}}}})
        info = self.call("storyboard_project", {"action": "undo", "project_id": pid, "revision": info["revision"]})
        info = self.call("storyboard_project", {"action": "redo", "project_id": pid, "revision": info["revision"]})
        project = self.call("storyboard_project", {"action": "get", "project_id": pid})["project"]
        self.assertEqual((project["defaults"]["fps"], project["export"]["aiLabel"]["embedTag"]), (25, False))
        exported = self.call("storyboard_project", {"action": "export", "project_id": pid})
        self.assertIn("/api/agent/files?path=", exported["download"]["url"])
        self.assertNotIn("path", exported)  # the agent gets the file name and the download, not a local path
        self.assertTrue(exported["file"].endswith(".chaosfilm") and "/" not in exported["file"])
        imported = self.call("storyboard_project", {"action": "import", "project_id": pid, "file": exported["file"]})
        self.assertNotEqual(imported["projectId"], pid)
        text = self.call("storyboard_project", {"action": "import", "project_id": pid, "file": "../../project.json"}, error=True)
        self.assertIn("not an export", text)
        self.call("storyboard_project", {"action": "delete", "project_id": pid, "revision": info["revision"]})

    def test_project_format_fields_and_the_old_shape(self):
        info = self.call("storyboard_project", {"action": "create", "name": "Wide", "aspect": "9:16",
                                                "resolution": "1080p", "fps": 25})
        defaults = self.call("storyboard_project", {"action": "get", "project_id": info["projectId"]})["project"]["defaults"]
        self.assertEqual((defaults["exportPreset"], defaults["resolution"], defaults["fps"], defaults["width"], defaults["height"]),
                         ("9:16", "1080p", 25, 1080, 1920))
        old = self.call("storyboard_project", {"action": "create", "name": "Old", "shape": "1:1"})
        edited = self.call("storyboard_project", {"action": "edit", "project_id": old["projectId"], "revision": old["revision"],
                                                  "shape": "16:9", "fps": 60, "upscale": "2160p"})
        project = self.call("storyboard_project", {"action": "get", "project_id": edited["projectId"]})["project"]
        self.assertEqual((project["defaults"]["exportPreset"], project["defaults"]["fps"], project["export"]["upscale"]),
                         ("16:9", 60, "2160p"))
        self.call("storyboard_project", {"action": "create", "name": "Bad", "resolution": "360p"}, error=True)

    def test_shot_and_audio_verbs(self):
        info = self.project(shots=2)
        pid, rev = info["projectId"], info["revision"]
        a, b = (s["id"] for s in info["shots"])
        args = {"project_id": pid}
        info = self.call("storyboard_shot", {**args, "revision": rev, "action": "edit", "shot_id": a, "fields": {"trimIn": 0.5}})
        self.assertAlmostEqual(info["shots"][0]["seconds"], 2.5)
        info = self.call("storyboard_shot", {**args, "revision": info["revision"], "action": "reorder", "order": [b, a]})
        info = self.call("storyboard_shot", {**args, "revision": info["revision"], "action": "lock", "shot_id": a})
        self.assertTrue(info["shots"][1]["locked"])
        info = self.call("storyboard_shot", {**args, "revision": info["revision"], "action": "import", "shot_id": b,
                                             "path": str(fake_mp4(self.src / "c.mp4"))})
        take = info["shots"][0]["activeTake"]
        info = self.call("storyboard_shot", {**args, "revision": info["revision"], "action": "pick_take", "shot_id": b, "take_id": take})
        info = self.call("storyboard_shot", {**args, "revision": info["revision"], "action": "redo", "shot_id": b})
        info = self.call("storyboard_audio", {**args, "revision": info["revision"], "action": "add", "kind": "voice",
                                              "path": str(fake_wav(self.src / "v.wav")), "fields": {"anchor": {"shotId": a, "offset": 1}}})
        clip = info["clipId"]
        self.assertAlmostEqual(info["audio"][0]["start"], 4.0)  # shot a now plays second, after b's 3 s
        info = self.call("storyboard_audio", {**args, "revision": info["revision"], "action": "edit", "clip_id": clip, "fields": {"gainDb": -3}})
        info = self.call("storyboard_audio", {**args, "revision": info["revision"], "action": "remove", "clip_id": clip})
        self.assertEqual(info["audio"], [])
        info = self.call("storyboard_shot", {**args, "revision": info["revision"], "action": "remove", "shot_id": a})
        self.assertEqual(len(info["shots"]), 1)

    def test_readable_errors(self):
        info = self.project(shots=1)
        pid, sid = info["projectId"], info["shots"][0]["id"]
        for name, arguments, needle in (
            ("storyboard_shot", {"action": "edit", "project_id": pid, "revision": 1, "shot_id": sid, "fields": {"title": "x"}}, "revision 2"),
            ("storyboard_shot", {"action": "edit", "project_id": pid, "revision": info["revision"]}, "shot_id"),
            ("storyboard_shot", {"action": "edit", "project_id": pid, "revision": info["revision"], "shot_id": sid,
                                 "fields": {"speed": 99}}, "speed"),
            ("storyboard_shot", {"action": "import", "project_id": pid, "revision": info["revision"], "shot_id": sid,
                                 "path": "/etc/hosts"}, "upload_file"),
            ("storyboard_shot", {"action": "import", "project_id": pid, "revision": info["revision"], "shot_id": sid,
                                 "path": str(self._text_file())}, "not a"),
            ("storyboard_project", {"action": "get", "project_id": "../x"}, "Unknown project"),
            ("storyboard_shot", {"action": "edit", "project_id": pid, "revision": 1, "fields": "x"}, "must be object"),
        ):
            self.assertIn(needle, self.call(name, arguments, error=True), arguments)

    def _text_file(self):
        path = self.src / "notes.txt"
        path.write_text("hi")
        return path

    def test_generate_is_a_job_and_respects_locks(self):
        info = self.project(shots=2)
        pid = info["projectId"]
        gens = FakeGenerators(self.src)
        with mock.patch("backend_service.storyboard.generators.studio_generators", fake_wiring(gens)):
            job = self.call("storyboard_generate", {"project_id": pid, "revision": info["revision"], "what": "both"})
            done = self.call("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 10})
            self.assertEqual(done["state"], "done", done)
            self.assertEqual(done["output"]["state"], "done")
            self.assertEqual(len(done["notes"]), 4)
            info = self.call("storyboard_project", {"action": "get", "project_id": pid})["project"]
            sid = info["shots"][0]["id"]
            info = self.call("storyboard_shot", {"action": "lock", "project_id": pid, "revision": info["revision"], "shot_id": sid})
            text = self.call("storyboard_generate", {"project_id": pid, "revision": info["revision"], "shot_ids": [sid]}, error=True)
            self.assertIn("locked", text)
            job = self.call("storyboard_generate", {"project_id": pid, "revision": info["revision"], "shot_ids": [sid],
                                                    "what": "clip", "force": True})
            self.assertEqual(self.call("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 10})["state"], "done")
        self.assertEqual(len(self.call("storyboard_project", {"action": "get", "project_id": pid})["project"]["shots"][0]["takes"]), 2)

    def test_render_job_output_carries_a_download(self):
        info = self.project(shots=1)
        pid = info["projectId"]
        info = self.call("storyboard_shot", {"action": "import", "project_id": pid, "revision": info["revision"],
                                             "shot_id": info["shots"][0]["id"], "path": str(fake_mp4(self.src / "c.mp4"))})

        def fake_render(plan, progress, cancel_event):
            Path(plan["out_path"]).write_bytes(b"the film")

        with mock.patch("backend_service.storyboard.render_adapter.render", fake_render):
            job = self.call("storyboard_render", {"project_id": pid, "revision": info["revision"], "preview": False})
            done = self.call("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 10})
        self.assertEqual(done["state"], "done", done)
        self.assertNotIn("path", done["output"])
        self.assertNotIn(str(self.base), json.dumps({k: v for k, v in done["output"].items() if k != "download"}))
        self.assertTrue(done["output"]["file"].endswith(".mp4"))
        download = done["output"]["download"]
        fetched = self.client.get(download["url"], headers=self.headers)
        self.assertEqual(fetched.content, b"the film")
        media = self.call("storyboard_project", {"action": "get", "project_id": pid})["project"]["shots"][0]["takes"][0]["clip"]
        folder = self.base / "storyboards" / pid
        self.assertEqual(self.client.get("/api/agent/files", params={"path": str(folder / media)}, headers=self.headers).status_code, 404)
        self.assertEqual(self.client.get("/api/agent/files", params={"path": str(folder / "project.json")}, headers=self.headers).status_code, 404)


    def test_agent_imports_from_outside_the_output_folders_are_refused(self):
        info = self.project(shots=1)
        pid, sid = info["projectId"], info["shots"][0]["id"]
        private = self.base / "elsewhere"
        clip, sound, still = fake_mp4(private / "home.mp4"), fake_wav(private / "x.wav"), png(private / "private.png")
        self.call("storyboard_project", {"action": "export", "project_id": pid})
        outside_bundle = private / "copy.chaosfilm"
        outside_bundle.write_bytes(next((self.base / "storyboards" / pid / "exports").glob("*.chaosfilm")).read_bytes())
        for name, arguments in (
            ("storyboard_shot", {"action": "import", "shot_id": sid, "path": str(still)}),
            ("storyboard_shot", {"action": "import", "shot_id": sid, "path": str(clip)}),
            ("storyboard_shot", {"action": "import", "shot_id": sid, "path": "~/../../etc/hosts"}),
            ("storyboard_audio", {"action": "add", "kind": "music", "path": str(sound)}),
            ("storyboard_audio", {"action": "add", "kind": "music", "path": str(self.src / ".." / "elsewhere" / "x.wav")}),
            ("storyboard_project", {"action": "import", "path": str(outside_bundle)}),
        ):
            text = self.call(name, {"project_id": pid, "revision": info["revision"], **arguments}, error=True)
            self.assertIn("upload_file", text, arguments)
            self.assertIn("upload: handle", text, arguments)
        project = self.call("storyboard_project", {"action": "get", "project_id": pid})["project"]
        self.assertEqual((project["shots"][0]["takes"], project["audio"]), ([], []))

    def test_agent_imports_upload_handles_for_stills_clips_and_sound(self):
        info = self.project(shots=1)
        pid, sid = info["projectId"], info["shots"][0]["id"]
        private = self.base / "elsewhere"
        mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200
        handles = {}
        for kind, data in (("still", png(private / "a.png").read_bytes()), ("clip", mp4),
                           ("sound", fake_wav(private / "b.wav").read_bytes())):
            response = self.client.post("/api/agent/uploads", content=data, headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            handles[kind] = response.json()["handle"]
        info = self.call("storyboard_shot", {"action": "import", "project_id": pid, "revision": info["revision"],
                                             "shot_id": sid, "path": handles["still"]})
        info = self.call("storyboard_shot", {"action": "import", "project_id": pid, "revision": info["revision"],
                                             "shot_id": sid, "path": handles["clip"]})
        info = self.call("storyboard_audio", {"action": "add", "project_id": pid, "revision": info["revision"],
                                              "kind": "voice", "path": handles["sound"]})
        takes = self.call("storyboard_project", {"action": "get", "project_id": pid})["project"]["shots"][0]["takes"]
        self.assertTrue(takes[0]["still"].endswith(".png") and takes[1]["clip"].endswith(".mp4"))
        self.assertEqual(len(info["audio"]), 1)


if __name__ == "__main__":
    unittest.main()
