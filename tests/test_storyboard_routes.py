"""``/api/storyboard`` routes: revisions over HTTP, files only from inside a project, batches, renders, bundles."""

import io
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.mcp_bridge import files
from backend_service.storyboard import store as store_module
from tests.storyboard_fixtures import FakeGenerators, fake_mp4, png

H = {"Authorization": "Bearer app-token"}


def fake_wiring(gens):
    def wiring(request, batch_id, folder):
        return {"make_still": gens.make_still, "make_clip": gens.make_clip, "unload": gens.unload,
                "last_frame": gens.last_frame, "stop_running": lambda kind: False}
    return wiring


class StoryboardRouteCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.src = self.base / "src"
        self.src.mkdir()
        env = mock.patch.dict(os.environ, {"CHAOSENGINE_STORYBOARDS_DIR": str(self.base / "storyboards")})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app(api_token="app-token"))

    def post(self, path, body=None, code=200):
        response = self.client.post(f"/api/storyboard{path}", json=body or {}, headers=H)
        self.assertEqual(response.status_code, code, response.text)
        return response.json()

    def get(self, path, code=200, **params):
        response = self.client.get(f"/api/storyboard{path}", params=params, headers=H)
        self.assertEqual(response.status_code, code, response.text)
        return response.json() if code == 200 else response

    def create(self, shots=2):
        data = self.post("/projects", {"name": "Lighthouse"})
        pid = data["project"]["id"]
        for i in range(shots):
            data = self.op(pid, data["project"]["revision"], "add_shot", {"title": f"s{i}", "stillPrompt": "sea"})
        return pid, data

    def op(self, pid, revision, op, args, code=200):
        return self.post(f"/projects/{pid}/ops", {"revision": revision, "op": op, "args": args}, code)

    def wait(self, check, seconds=10):
        deadline = time.time() + seconds
        while time.time() < deadline:
            result = check()
            if result:
                return result
            time.sleep(0.05)
        self.fail("timed out")


class ProjectRouteTests(StoryboardRouteCase):
    def test_needs_the_app_token(self):
        self.client.app.state.chaosengine_require_api_auth = True  # the default; a local settings file may turn it off
        self.assertEqual(self.client.get("/api/storyboard/projects").status_code, 401)

    def test_create_list_get_delete(self):
        pid, data = self.create(shots=1)
        self.assertEqual(self.get("/projects")["projects"][0]["id"], pid)
        got = self.get(f"/projects/{pid}")
        self.assertEqual(got["project"]["revision"], data["project"]["revision"])
        self.assertAlmostEqual(got["timeline"]["total"], 4.0)
        self.assertEqual(self.client.delete(f"/api/storyboard/projects/{pid}", headers=H).status_code, 422)
        self.assertEqual(self.client.delete(f"/api/storyboard/projects/{pid}", params={"revision": 1}, headers=H).status_code, 409)
        rev = got["project"]["revision"]
        busy = store_module.ProjectBusy("A render is running for this project. Cancel it or wait, then delete.")
        with mock.patch.object(store_module.ProjectStore, "delete", side_effect=busy):
            response = self.client.delete(f"/api/storyboard/projects/{pid}", params={"revision": rev}, headers=H)
        self.assertEqual((response.status_code, response.json()["detail"]), (409, str(busy)))
        self.assertEqual(self.client.delete(f"/api/storyboard/projects/{pid}", params={"revision": rev}, headers=H).status_code, 200)
        self.get(f"/projects/{pid}", code=404)
        self.get("/projects/..%2F..%2Fetc", code=404)

    def test_stale_revision_answers_409_with_the_current_project(self):
        pid, data = self.create(shots=1)
        sid, rev = data["project"]["shots"][0]["id"], data["project"]["revision"]
        self.op(pid, rev, "edit_shot", {"shotId": sid, "title": "ui"})
        conflict = self.op(pid, rev, "edit_shot", {"shotId": sid, "title": "agent"}, code=409)["detail"]
        self.assertEqual(conflict["revision"], rev + 1)
        self.assertEqual(conflict["project"]["shots"][0]["title"], "ui")
        self.post(f"/projects/{pid}/ops", {"op": "lock", "args": {"shotId": sid}}, code=422)  # revision required
        self.assertIn("Unknown shot", self.op(pid, rev + 1, "lock", {"shotId": "zz"}, code=400)["detail"])

    def test_undo_redo(self):
        pid, data = self.create(shots=1)
        sid = data["project"]["shots"][0]["id"]
        data = self.op(pid, data["project"]["revision"], "lock", {"shotId": sid})
        data = self.post(f"/projects/{pid}/undo", {"revision": data["project"]["revision"]})
        self.assertFalse(data["project"]["shots"][0]["locked"])
        data = self.post(f"/projects/{pid}/redo", {"revision": data["project"]["revision"]})
        self.assertTrue(data["project"]["shots"][0]["locked"])
        self.post(f"/projects/{pid}/redo", {"revision": data["project"]["revision"]}, code=400)

    def test_import_from_a_path_and_an_upload_handle(self):
        pid, data = self.create(shots=1)
        sid = data["project"]["shots"][0]["id"]
        inbox = self.base / "inbox"
        inbox.mkdir()
        with mock.patch.object(files, "inbox_dir", lambda: inbox):
            handle = files.save_upload(png(self.src / "u.png").read_bytes())["handle"]
            data = self.op(pid, data["project"]["revision"], "import_take",
                           {"shotId": sid, "still": handle, "clip": str(fake_mp4(self.src / "c.mp4"))})
        take = data["project"]["shots"][0]["takes"][0]
        self.assertTrue(take["still"].endswith(".png") and take["clip"].endswith(".mp4"))
        served = self.client.get(f"/api/storyboard/projects/{pid}/files", params={"path": take["clip"]}, headers=H)
        self.assertEqual(served.content, (self.src / "c.mp4").read_bytes())

    def test_files_only_from_inside_the_project(self):
        pid, _ = self.create(shots=0)
        (self.base / "secret.txt").write_text("secret")
        for path in ("project.json", "media/../project.json", "../secret.txt", "/etc/hosts", str(self.base / "secret.txt"),
                     "exports/../project.json", "exports/", "media/audio", "cache/x"):
            response = self.client.get(f"/api/storyboard/projects/{pid}/files", params={"path": path}, headers=H)
            self.assertEqual(response.status_code, 404, path)


class BatchRenderBundleRouteTests(StoryboardRouteCase):
    def test_a_live_batch_reports_progress_with_the_clips_made_so_far(self):
        import threading
        import time

        pid, data = self.create(shots=3)
        gens = FakeGenerators(self.src)
        second_clip, release = threading.Event(), threading.Event()
        original = gens.make_clip

        def slow_clip(shot, project, still):
            made = sum(1 for c in gens.calls if c[0] == "clip")
            if made == 1:  # hold the second clip so the poll sees a batch with one measured clip
                second_clip.set()
                release.wait(10)
            time.sleep(0.05)
            return original(shot, project, still)

        gens.make_clip = slow_clip
        with mock.patch("backend_service.storyboard.generators.studio_generators", fake_wiring(gens)):
            bid = self.post(f"/projects/{pid}/batches", {"revision": data["project"]["revision"], "what": "both"})["batchId"]
            self.assertTrue(second_clip.wait(10))
            live = self.get(f"/projects/{pid}/batches/{bid}")
            release.set()
            self.wait(lambda: self.get(f"/projects/{pid}/batches/{bid}")["batch"]["state"] == "done")
            finished = self.get(f"/projects/{pid}/batches/{bid}")
        progress = live["progress"]
        self.assertEqual((progress["kind"], progress["kindIndex"], progress["kindTotal"]), ("clip", 2, 3))
        self.assertEqual(progress["totalSteps"], 6)  # three stills, then three clips
        self.assertEqual(progress["etaBasis"], "clips")  # one clip was measured: the other two are priced from it
        self.assertIn("elapsedSeconds", progress)
        self.assertNotIn("progress", finished)

    def test_batch_runs_with_fake_generators_and_is_persisted(self):
        pid, data = self.create(shots=2)
        gens = FakeGenerators(self.src)
        with mock.patch("backend_service.storyboard.generators.studio_generators", fake_wiring(gens)):
            started = self.post(f"/projects/{pid}/batches", {"revision": data["project"]["revision"], "what": "both"})
            bid = started["batchId"]
            done = self.wait(lambda: (b := self.get(f"/projects/{pid}/batches/{bid}"))["batch"]["state"] == "done" and b)
        self.assertEqual([r["state"] for r in done["batch"]["results"]], ["done"] * 4)
        project = self.get(f"/projects/{pid}")["project"]
        self.assertEqual([s["status"] for s in project["shots"]], ["done", "done"])
        self.post(f"/projects/{pid}/batches/{bid}/cancel", code=200)  # finished: cancelled false
        self.post(f"/projects/{pid}/batches/b_00000000/cancel", code=404)
        self.post(f"/projects/{pid}/batches", {"revision": 1}, code=409)

    def test_render_runs_in_the_background_one_at_a_time(self):
        pid, data = self.create(shots=1)
        sid = data["project"]["shots"][0]["id"]
        data = self.op(pid, data["project"]["revision"], "import_take", {"shotId": sid, "clip": str(fake_mp4(self.src / "c.mp4"))})
        gate = threading.Event()
        seen = {}

        def fake_render(plan, progress, cancel_event):
            seen["plan"] = plan
            progress(0.5, "joining")
            gate.wait(5)
            Path(plan["out_path"]).write_bytes(b"film")

        rev = data["project"]["revision"]
        with mock.patch("backend_service.storyboard.render_adapter.render", fake_render):
            render = self.post(f"/projects/{pid}/render", {"revision": rev, "preview": True})["render"]
            self.assertEqual(render["state"], "running")
            self.post(f"/projects/{pid}/render", {"revision": rev, "preview": True}, code=409)
            gate.set()
            done = self.wait(lambda: (r := self.get(f"/projects/{pid}/render")["render"])["state"] == "done" and r)
        self.assertEqual(done["file"], "exports/preview.mp4")
        self.assertTrue(seen["plan"]["preview"])
        served = self.client.get(f"/api/storyboard/projects/{pid}/files", params={"path": done["file"]}, headers=H)
        self.assertEqual(served.content, b"film")
        self.post(f"/projects/{pid}/render", {"revision": rev - 1}, code=409)

    def test_render_with_no_clips_is_a_readable_400(self):
        pid, data = self.create(shots=1)
        self.assertIn("Nothing to render", self.post(f"/projects/{pid}/render", {"revision": data["project"]["revision"]}, code=400)["detail"])

    def test_bundle_export_and_import_by_path_and_body(self):
        pid, data = self.create(shots=1)
        sid = data["project"]["shots"][0]["id"]
        self.op(pid, data["project"]["revision"], "import_take", {"shotId": sid, "clip": str(fake_mp4(self.src / "c.mp4"))})
        exported = self.post(f"/projects/{pid}/bundle")
        raw = self.client.get(f"/api/storyboard/projects/{pid}/files", params={"path": exported["file"]}, headers=H).content
        self.assertEqual(len(raw), exported["bytes"])
        bundle_path = self.base / "film.chaosfilm"
        bundle_path.write_bytes(raw)
        by_path = self.post("/import", {"path": str(bundle_path)})
        by_body = self.client.post("/api/storyboard/import", content=raw, headers={**H, "content-type": "application/zip"})
        self.assertEqual(by_body.status_code, 200, by_body.text)
        ids = {pid, by_path["project"]["id"], by_body.json()["project"]["id"]}
        self.assertEqual(len(ids), 3)
        self.assertEqual(by_path["project"]["shots"][0]["status"], "done")
        hostile = io.BytesIO()
        import zipfile

        with zipfile.ZipFile(hostile, "w") as zf:
            zf.writestr("project.json", "{}")
            zf.writestr("../../evil.sh", "x")
        response = self.client.post("/api/storyboard/import", content=hostile.getvalue(), headers={**H, "content-type": "application/zip"})
        self.assertEqual(response.status_code, 400)
        self.post("/import", {"path": str(self.base / "nope.chaosfilm")}, code=400)
        self.assertEqual(len(self.get("/projects")["projects"]), 3)

    def test_a_bad_json_import_body_is_a_400(self):
        for body in ({"nope": 1}, {"path": ""}, {"path": 5}, [1]):
            response = self.client.post("/api/storyboard/import", json=body, headers=H)
            self.assertEqual(response.status_code, 400, (body, response.text))
            self.assertIn("path", response.json()["detail"])
        response = self.client.post("/api/storyboard/import", content=b"{not json", headers={**H, "content-type": "application/json"})
        self.assertEqual(response.status_code, 400, response.text)


if __name__ == "__main__":
    unittest.main()
