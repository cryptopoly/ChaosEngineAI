"""``/api/cleanup``: listing, deleting by listed id only, hostile ids, live batches.

Every folder is a temporary directory (storyboards via ``CHAOSENGINE_STORYBOARDS_DIR``, chat threads via a
state built on temp paths, gallery folders via patched output-dir helpers). Nothing here reads or writes the
real ``~/.chaosengine``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.state import ChaosEngineState
from backend_service.storyboard import store as store_module
from backend_service.storyboard.models import BatchRecord
from tests.test_backend_service import TEST_API_TOKEN, FakeRuntime, fake_library, fake_system_snapshot

H = {"Authorization": f"Bearer {TEST_API_TOKEN}"}


class CleanupRouteCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.image_dir = self.base / "image-outputs"
        self.video_dir = self.base / "video-outputs"
        self.image_dir.mkdir()
        self.video_dir.mkdir()
        self.outside = self.base / "outside"
        self.outside.mkdir()
        (self.outside / "precious.txt").write_text("keep me", encoding="utf-8")

        env = mock.patch.dict(os.environ, {"CHAOSENGINE_STORYBOARDS_DIR": str(self.base / "storyboards")})
        env.start()
        self.addCleanup(env.stop)
        for name, folder in (("_current_image_outputs_dir", self.image_dir), ("_current_video_outputs_dir", self.video_dir)):
            patcher = mock.patch(f"backend_service.app.{name}", return_value=folder)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=self.base / "settings.json",
            benchmarks_path=self.base / "benchmark-history.json",
            library_cache_path=self.base / "library-cache.json",
            chat_sessions_path=self.base / "chat-sessions.json",
        )
        self.state.runtime = FakeRuntime()
        self.client = TestClient(create_app(state=self.state, api_token=TEST_API_TOKEN))
        self.store = store_module.default_store()

    # ---- fixtures ----
    def make_project(self, name: str = "Lighthouse", shots: int = 2) -> str:
        project = self.store.create(name)
        for i in range(shots):
            self.store.mutate(project.id, None, lambda p, f, i=i: self._add_shot(p, i), "add_shot")
        (self.store.folder(project.id) / "media" / "blob.bin").write_bytes(b"x" * 1000)
        return project.id

    @staticmethod
    def _add_shot(project, index):
        from backend_service.storyboard.models import Shot

        project.shots.append(Shot(id=f"shot{index}", order=index, title=f"s{index}", stillPrompt="sea"))

    def make_thread(self, thread_id: str, title: str, messages: int = 2) -> None:
        self.state.chat_sessions.append({
            "id": thread_id, "title": title, "updatedAt": "2026-11-01 10:00:00",
            "messages": [{"role": "user", "text": "hi", "createdAt": 1_790_000_000_000 + i} for i in range(messages)],
        })

    def make_output(self, kind: str, artifact_id: str, day: str = "2026-11-01", prompt: str = "a lighthouse") -> Path:
        root = self.image_dir if kind == "image" else self.video_dir
        folder = root / day
        folder.mkdir(parents=True, exist_ok=True)
        ext = "png" if kind == "image" else "mp4"
        (folder / f"{artifact_id}.{ext}").write_bytes(b"m" * 500)
        meta = folder / f"{artifact_id}.json"
        path_key = "imagePath" if kind == "image" else "videoPath"
        meta.write_text(json.dumps({"artifactId": artifact_id, "prompt": prompt, "createdAt": f"{day}T10:00:00Z",
                                    path_key: str(folder / f"{artifact_id}.{ext}")}), encoding="utf-8")
        return folder / f"{artifact_id}.{ext}"

    def items(self) -> dict:
        response = self.client.get("/api/cleanup/items", headers=H)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def delete(self, code: int = 200, **ids) -> dict:
        response = self.client.post("/api/cleanup/delete", json=ids, headers=H)
        self.assertEqual(response.status_code, code, response.text)
        return response.json()


class ListingTests(CleanupRouteCase):
    def test_empty(self):
        self.assertEqual(self.items(), {"projects": [], "threads": [], "outputs": []})

    def test_lists_each_group_with_size_and_date(self):
        pid = self.make_project("Lighthouse", shots=2)
        self.make_thread("session-aa", "Plans", messages=3)
        self.make_output("image", "img1")
        self.make_output("video", "vid1", day="2026-11-02", prompt="a long " * 30)
        data = self.items()

        (project,) = data["projects"]
        self.assertEqual((project["id"], project["name"], project["shots"]), (pid, "Lighthouse", 2))
        self.assertGreater(project["bytes"], 1000)
        self.assertAlmostEqual(project["lastEdited"], time.time(), delta=60)

        (thread,) = data["threads"]
        self.assertEqual((thread["id"], thread["title"], thread["messages"]), ("session-aa", "Plans", 3))
        self.assertGreater(thread["bytes"], 0)
        self.assertEqual(thread["lastUsed"], 1_790_000_000.002)

        by_kind = {row["kind"]: row for row in data["outputs"]}
        self.assertEqual(by_kind["image"]["id"], "image:2026-11-01/img1.json")
        self.assertGreater(by_kind["image"]["bytes"], 500)
        self.assertEqual(by_kind["image"]["createdAt"], "2026-11-01T10:00:00Z")
        self.assertTrue(by_kind["video"]["name"].endswith("..."))
        self.assertLessEqual(len(by_kind["video"]["name"]), 83)
        # newest first
        self.assertEqual([row["kind"] for row in data["outputs"]], ["video", "image"])

    def test_unreadable_project_and_foreign_json_are_not_offered(self):
        pid = self.make_project()
        (self.store.folder(pid) / "project.json").write_text("{not json", encoding="utf-8")
        (self.image_dir / "notes.json").write_text(json.dumps({"hello": 1}), encoding="utf-8")
        (self.image_dir / "list.json").write_text("[1]", encoding="utf-8")
        data = self.items()
        self.assertEqual(data["projects"], [])
        self.assertEqual(data["outputs"], [])

    def test_requires_the_api_token(self):
        self.assertEqual(self.client.get("/api/cleanup/items").status_code, 401)
        self.assertEqual(self.client.post("/api/cleanup/delete", json={"projects": ["x"]}).status_code, 401)


class DeleteTests(CleanupRouteCase):
    def test_deletes_chosen_items_and_leaves_the_rest(self):
        keep, drop = self.make_project("Keep"), self.make_project("Drop")
        self.make_thread("session-keep", "Keep"), self.make_thread("session-drop", "Drop")
        image = self.make_output("image", "drop-img")
        self.make_output("image", "keep-img")
        video = self.make_output("video", "drop-vid")

        out = self.delete(projects=[drop], threads=["session-drop"],
                          outputs=["image:2026-11-01/drop-img.json", "video:2026-11-01/drop-vid.json"])
        self.assertEqual((out["deleted"], out["refused"]), (4, 0))
        self.assertTrue(all(r["ok"] for r in out["results"]))
        self.assertGreater(out["freedBytes"], 1000 + 500 + 500)

        self.assertFalse(self.store.folder(drop).exists())
        self.assertTrue(self.store.folder(keep).is_dir())
        self.assertFalse(image.exists() or image.with_suffix(".json").exists())
        self.assertFalse(video.exists() or video.with_suffix(".json").exists())
        data = self.items()
        self.assertEqual([p["id"] for p in data["projects"]], [keep])
        self.assertEqual([t["id"] for t in data["threads"]], ["session-keep"])
        self.assertEqual([o["id"] for o in data["outputs"]], ["image:2026-11-01/keep-img.json"])
        saved = json.loads((self.base / "chat-sessions.json").read_text(encoding="utf-8"))
        self.assertEqual([s["id"] for s in saved], ["session-keep"])

    def test_nothing_chosen_is_a_readable_400(self):
        self.assertIn("Nothing", self.delete(code=400)["detail"])

    def test_unknown_and_repeated_ids_are_reported_not_found(self):
        pid = self.make_project()
        out = self.delete(projects=[pid, pid, "0123456789abcdef"], threads=["session-nope"], outputs=["image:2026-01-01/none.json"])
        results = {(r["kind"], r["id"]): r for r in out["results"]}
        self.assertEqual(len(out["results"]), 4)  # the repeated id counts once
        self.assertTrue(results[("project", pid)]["ok"])
        for key in (("project", "0123456789abcdef"), ("thread", "session-nope"), ("output", "image:2026-01-01/none.json")):
            self.assertFalse(results[key]["ok"])
            self.assertIn("Not found", results[key]["reason"])
        self.assertEqual((out["deleted"], out["refused"]), (1, 3))

    def test_unknown_body_fields_and_paths_are_refused(self):
        # A path can only arrive as an id; a "paths" field does not exist and is refused.
        self.delete(code=422, paths=[str(self.outside / "precious.txt")])
        self.delete(code=422, outputs=[""])

    def test_hostile_ids_touch_nothing(self):
        pid = self.make_project()
        self.make_thread("session-aa", "Plans")
        self.make_output("image", "img1")
        victim = self.outside / "precious.txt"
        hostile = [
            "../outside/precious.txt", "../../outside", str(victim), str(self.outside), "/", "..", ".", "*", "**/*",
            "image:../outside/precious.txt", "image:../outside/x.json", f"image:{self.outside}/x.json",
            "image:2026-11-01/../../outside/precious.txt", "video:*", "image:*", "image:2026-11-01/*.json",
            "%2e%2e%2foutside", "..\\outside", "image:..\\outside", "a\x00b",
        ]
        out = self.delete(projects=hostile, threads=hostile, outputs=hostile)
        self.assertEqual(out["deleted"], 0)
        self.assertTrue(all(not r["ok"] for r in out["results"]))
        self.assertEqual(victim.read_text(encoding="utf-8"), "keep me")
        data = self.items()
        self.assertEqual([p["id"] for p in data["projects"]], [pid])
        self.assertEqual(len(data["threads"]), 1)
        self.assertEqual(len(data["outputs"]), 1)
        self.assertTrue(self.base.exists() and self.image_dir.is_dir() and self.video_dir.is_dir())

    def test_metadata_path_fields_are_never_followed(self):
        # A saved metadata file names an image path; a hand-edited one pointing outside the gallery must not be unlinked.
        victim = self.outside / "precious.txt"
        folder = self.image_dir / "2026-11-01"
        folder.mkdir()
        (folder / "evil.json").write_text(json.dumps({"artifactId": "evil", "prompt": "p", "imagePath": str(victim)}), encoding="utf-8")
        out = self.delete(outputs=["image:2026-11-01/evil.json"])
        self.assertEqual(out["deleted"], 1)
        self.assertEqual(victim.read_text(encoding="utf-8"), "keep me")
        self.assertFalse((folder / "evil.json").exists())

    def test_linked_media_and_linked_folders_are_left_alone(self):
        victim = self.outside / "precious.txt"
        folder = self.image_dir / "2026-11-01"
        folder.mkdir()
        (folder / "lnk.json").write_text(json.dumps({"artifactId": "lnk", "prompt": "p"}), encoding="utf-8")
        try:
            (folder / "lnk.png").symlink_to(victim)
            (self.image_dir / "linked-day").symlink_to(self.outside, target_is_directory=True)
            (self.outside / "far.json").write_text(json.dumps({"artifactId": "far", "prompt": "p"}), encoding="utf-8")
        except OSError:
            self.skipTest("symlinks are not available here")
        ids = [row["id"] for row in self.items()["outputs"]]
        self.assertEqual(ids, ["image:2026-11-01/lnk.json"])  # nothing under the linked folder is offered
        self.delete(outputs=ids + ["image:linked-day/far.json"])
        self.assertEqual(victim.read_text(encoding="utf-8"), "keep me")
        self.assertTrue((self.outside / "far.json").exists())
        self.assertTrue((folder / "lnk.png").is_symlink())  # the link stays: only real files inside the gallery go

    def test_project_folder_symlink_is_not_offered_or_followed(self):
        target = self.outside / "pretend-project"
        target.mkdir()
        try:
            self.store.root.mkdir(parents=True, exist_ok=True)
            (self.store.root / "0123456789abcdef").symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks are not available here")
        self.assertEqual(self.items()["projects"], [])
        self.delete(projects=["0123456789abcdef"])
        self.assertTrue(target.is_dir())

    def test_project_with_a_live_batch_is_refused_and_others_still_go(self):
        busy, free = self.make_project("Busy"), self.make_project("Free")
        batch_id = "batchlive1"
        self.store.mutate(busy, None, lambda p, f: p.batches.append(
            BatchRecord(id=batch_id, what="both", state="running", createdAt=time.time())), "batch", reversible=False)
        store_module.LIVE_BATCHES.add(batch_id)
        self.addCleanup(store_module.LIVE_BATCHES.discard, batch_id)

        out = self.delete(projects=[busy, free])
        by_id = {r["id"]: r for r in out["results"]}
        self.assertFalse(by_id[busy]["ok"])
        self.assertIn("batch is running", by_id[busy]["reason"])
        self.assertTrue(by_id[free]["ok"])
        self.assertTrue(self.store.folder(busy).is_dir())
        self.assertFalse(self.store.folder(free).exists())

        store_module.LIVE_BATCHES.discard(batch_id)  # the batch ended: the same request now works
        self.assertEqual(self.delete(projects=[busy])["deleted"], 1)
        self.assertFalse(self.store.folder(busy).exists())

    def test_project_with_a_running_render_is_refused(self):
        pid = self.make_project()
        from backend_service.storyboard.render_adapter import RENDERS

        class _Job:
            done = mock.Mock(is_set=lambda: False)

        with mock.patch.object(RENDERS, "status", return_value=_Job()):
            out = self.delete(projects=[pid])
        self.assertFalse(out["results"][0]["ok"])
        self.assertIn("render is running", out["results"][0]["reason"])
        self.assertTrue(self.store.folder(pid).is_dir())


if __name__ == "__main__":
    unittest.main()
