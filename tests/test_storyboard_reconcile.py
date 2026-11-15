"""A project left running or queued by a crash or a power cut is made honest again (FU-179, plan S3)."""

import json
import threading
import unittest
from unittest import mock

from backend_service.app import create_app
from backend_service.storyboard import store as store_module
from backend_service.storyboard.store import LIVE_BATCHES
from tests.storyboard_fixtures import StoreCase, png

BATCH_ID = "b_1234abcd"


class ReconcileCase(StoreCase):
    def stale_project(self, batch_state="running"):
        """Three shots: one running with a take, one queued with none, one idle; and a running batch record."""
        project = self.new(shots=3)
        folder = self.folder(project)
        png(folder / "media" / "a.png")
        path = folder / "project.json"
        data = json.loads(path.read_text())
        shots = data["shots"]
        shots[0]["status"] = "running"
        shots[0]["takes"] = [{"id": "t_00000001", "still": "media/a.png"}]
        shots[0]["activeTake"] = "t_00000001"
        shots[1]["status"] = "queued"
        data["batches"] = [{"id": BATCH_ID, "what": "both", "state": batch_state, "results": [
            {"shotId": shots[0]["id"], "kind": "still", "state": "done", "takeId": "t_00000001"},
            {"shotId": shots[0]["id"], "kind": "clip", "state": "running"},
            {"shotId": shots[1]["id"], "kind": "still", "state": "queued"},
            {"shotId": shots[2]["id"], "kind": "still", "state": "skipped", "error": "locked"},
        ]}]
        path.write_text(json.dumps(data))
        return project, path


class ReconcileOnLoadTests(ReconcileCase):
    def test_a_stale_run_is_reset_when_the_project_is_opened(self):
        project, path = self.stale_project()
        loaded = self.store.load(project.id)
        self.assertEqual([s.status for s in loaded.shots], ["done", "idle", "idle"])
        batch = loaded.batches[0]
        self.assertEqual(batch.state, "interrupted")
        self.assertIsNotNone(batch.finishedAt)
        self.assertEqual([r.state for r in batch.results], ["done", "cancelled", "cancelled", "skipped"])
        for result in batch.results[1:3]:
            self.assertEqual(result.error, store_module.INTERRUPTED_NOTE)
        self.assertEqual(batch.results[3].error, "locked")
        on_disk = json.loads(path.read_text())
        self.assertEqual(on_disk["batches"][0]["state"], "interrupted")
        self.assertEqual(on_disk["shots"][0]["status"], "done")

    def test_it_runs_again_on_a_later_open_not_only_the_first(self):
        project, path = self.stale_project()
        self.store.load(project.id)
        data = json.loads(path.read_text())
        data["shots"][1]["status"] = "running"
        path.write_text(json.dumps(data))
        self.assertEqual(self.store.load(project.id).shots[1].status, "idle")

    def test_a_clean_project_is_not_rewritten(self):
        project = self.new(shots=1)
        before = (self.folder(project) / "project.json").read_bytes()
        self.store.load(project.id)
        self.assertEqual((self.folder(project) / "project.json").read_bytes(), before)

    def test_a_batch_live_in_this_process_is_left_alone(self):
        project, _ = self.stale_project()
        LIVE_BATCHES.add(BATCH_ID)
        self.addCleanup(LIVE_BATCHES.discard, BATCH_ID)
        loaded = self.store.load(project.id)
        self.assertEqual([s.status for s in loaded.shots], ["running", "queued", "idle"])
        self.assertEqual(loaded.batches[0].state, "running")
        self.assertEqual([r.state for r in loaded.batches[0].results], ["done", "running", "queued", "skipped"])

    def test_a_dead_batch_is_reset_while_another_is_live(self):
        project, path = self.stale_project()
        data = json.loads(path.read_text())
        data["batches"].append({"id": "b_99999999", "what": "still", "state": "running", "results": [
            {"shotId": data["shots"][2]["id"], "kind": "still", "state": "running"}]})
        data["shots"][2]["status"] = "running"
        path.write_text(json.dumps(data))
        LIVE_BATCHES.add("b_99999999")
        self.addCleanup(LIVE_BATCHES.discard, "b_99999999")
        loaded = self.store.load(project.id)
        self.assertEqual([b.state for b in loaded.batches], ["interrupted", "running"])
        self.assertEqual([s.status for s in loaded.shots], ["done", "idle", "running"])


class ReconcileAtStartTests(ReconcileCase):
    def test_reconcile_all_resets_every_project(self):
        first, first_path = self.stale_project()
        second, second_path = self.stale_project("queued")
        self.assertEqual(self.store.reconcile_all(), 2)
        for path in (first_path, second_path):
            data = json.loads(path.read_text())
            self.assertEqual(data["batches"][0]["state"], "interrupted")
            self.assertEqual([s["status"] for s in data["shots"]], ["done", "idle", "idle"])
        self.assertEqual(self.store.reconcile_all(), 0)

    def test_a_broken_project_does_not_stop_the_others(self):
        project, path = self.stale_project()
        broken = self.root / "ffffffffffffffff"
        broken.mkdir()
        (broken / "project.json").write_text("{not json")
        self.assertEqual(self.store.reconcile_all(), 1)
        self.assertEqual(json.loads(path.read_text())["batches"][0]["state"], "interrupted")

    def test_the_app_reconciles_when_it_starts(self):
        from starlette.testclient import TestClient

        project, path = self.stale_project()
        done = threading.Event()
        real = store_module.ProjectStore.reconcile_all

        def spy(store):
            try:
                return real(store)
            finally:
                done.set()

        with mock.patch.object(store_module, "default_store", lambda: self.store), \
                mock.patch.object(store_module.ProjectStore, "reconcile_all", spy):
            with TestClient(create_app(state=mock.MagicMock(), api_token="t")):
                self.assertTrue(done.wait(10))
        self.assertEqual(json.loads(path.read_text())["batches"][0]["state"], "interrupted")


if __name__ == "__main__":
    unittest.main()
