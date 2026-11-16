"""The durable batch queue (S-M1): pause after a step, resume from the remaining steps, survive a restart."""

import json
import threading
from unittest import mock

from backend_service.storyboard import batch as batch_module
from backend_service.storyboard.batch import Batch, BatchRefused
from backend_service.storyboard.batch_queue import cancel_paused, paused_progress
from backend_service.storyboard.store import LIVE_BATCHES, ProjectBusy
from tests.storyboard_fixtures import FakeGenerators, StoreCase
from tests.test_storyboard_routes import StoryboardRouteCase, fake_wiring


class PausingGenerators(FakeGenerators):
    """Asks the batch to pause once ``after`` steps have been made."""

    def __init__(self, src, after):
        super().__init__(src)
        self.after, self.batch = after, None

    def _count(self):
        made = sum(1 for c in self.calls if c[0] in ("still", "clip"))
        if made == self.after and self.batch is not None:
            self.batch.pause()

    def make_still(self, shot, project):
        out = super().make_still(shot, project)
        self._count()
        return out, {"settings": {"model": "Img"}}

    def make_clip(self, shot, project, still):
        out = super().make_clip(shot, project, still)
        self._count()
        return out


class QueueCase(StoreCase):
    def batch(self, project, gens, what="both", **kwargs):
        return Batch(self.store, project.id, what, gens.make_still, gens.make_clip, gens.unload, gens.last_frame, **kwargs)

    def paused_after(self, shots, after):
        project = self.new(shots=shots)
        gens = PausingGenerators(self.src, after)
        first = self.batch(project, gens)
        gens.batch = first
        first.start(project.revision)
        first.run()
        return self.store.load(project.id), first, gens


class PauseTests(QueueCase):
    def test_pause_stops_after_the_step_in_progress_and_keeps_the_rest_queued(self):
        project, first, gens = self.paused_after(shots=2, after=1)
        record = project.batches[-1]
        self.assertEqual(record.state, "paused")
        self.assertEqual(record.pauseReason, "user")
        self.assertIsNotNone(record.pausedAt)
        self.assertIsNone(record.finishedAt)
        self.assertEqual([(r.kind, r.state) for r in record.results],
                         [("still", "done"), ("still", "queued"), ("clip", "queued"), ("clip", "queued")])
        self.assertEqual(len(gens.calls), 1)
        self.assertNotIn(first.id, LIVE_BATCHES)
        self.assertTrue(all(s.status in ("done", "idle") for s in project.shots))  # nothing shows as running

    def test_resume_runs_only_the_remaining_steps_and_the_clip_completes_the_paused_still(self):
        project, first, _ = self.paused_after(shots=2, after=1)
        still_take = project.batches[-1].results[0].takeId
        gens = FakeGenerators(self.src)
        again = self.batch(project, gens)
        again.resume(first.id, project.revision)
        again.run()
        done = self.store.load(project.id)
        record = done.batches[-1]
        self.assertEqual(len(done.batches), 1)  # the same record, not a new batch
        self.assertEqual(record.id, first.id)
        self.assertEqual(record.state, "done")
        self.assertEqual(record.resumes, 1)
        self.assertEqual([c[:2] for c in gens.calls if c[0] != "unload"],
                         [("still", done.shots[1].id), ("clip", done.shots[0].id), ("clip", done.shots[1].id)])
        shot = done.shots[0]
        self.assertEqual(len(shot.takes), 1)
        self.assertEqual(shot.activeTake, still_take)
        self.assertTrue(shot.active().still and shot.active().clip)
        self.assertEqual([s.status for s in done.shots], ["done", "done"])

    def test_progress_after_a_resume_counts_the_whole_batch(self):
        project, first, _ = self.paused_after(shots=2, after=1)
        again = self.batch(project, FakeGenerators(self.src))
        again.resume(first.id, project.revision)
        info = again.progress()
        self.assertEqual(info["totalSteps"], 4)
        self.assertEqual(info["step"], 1)
        again._current = (project.shots[1].id, "still", 0.0)
        info = again.progress()
        self.assertEqual(info["phase"], "still 2 of 2")
        self.assertEqual(info["step"], 2)
        again.run()

    def test_only_a_paused_batch_resumes(self):
        project, first, _ = self.paused_after(shots=1, after=1)
        again = self.batch(project, FakeGenerators(self.src))
        again.resume(first.id, project.revision)
        again.run()
        project = self.store.load(project.id)
        with self.assertRaises(ProjectBusy):
            self.batch(project, FakeGenerators(self.src)).resume(first.id, project.revision)
        self.assertNotIn(first.id, LIVE_BATCHES)

    def test_nothing_left_to_resume_is_refused_and_the_record_is_left_alone(self):
        project, first, _ = self.paused_after(shots=1, after=1)
        project = self.op(project, "lock", shotId=project.shots[0].id, locked=True)
        with self.assertRaises(BatchRefused):
            self.batch(project, FakeGenerators(self.src)).resume(first.id, project.revision)
        self.assertEqual(self.store.load(project.id).batches[-1].state, "paused")
        self.assertNotIn(first.id, LIVE_BATCHES)

    def test_a_paused_batch_can_be_cancelled(self):
        project, first, _ = self.paused_after(shots=2, after=1)
        self.assertTrue(cancel_paused(self.store, project.id, first.id))
        record = self.store.load(project.id).batches[-1]
        self.assertEqual(record.state, "cancelled")
        self.assertEqual([r.state for r in record.results], ["done", "cancelled", "cancelled", "cancelled"])
        with self.assertRaises(ProjectBusy):
            cancel_paused(self.store, project.id, first.id)

    def test_paused_bar_data_counts_steps_and_prices_what_is_left(self):
        project, first, _ = self.paused_after(shots=2, after=1)
        # The finished still recorded how long it took; no clip has been made anywhere in the project yet.
        info = paused_progress(project, project.batches[-1])
        self.assertEqual((info["totalSteps"], info["step"], info["remainingSteps"]), (4, 1, 3))
        self.assertEqual(info["remaining"], {"still": 1, "clip": 2})
        self.assertTrue(info["paused"])
        self.assertNotIn("etaSeconds", info)
        still = project.shots[0].takes[0]
        self.assertIn("stillSeconds", still.settings)
        still.settings = {**still.settings, "stillSeconds": 10.0}
        from backend_service.storyboard.models import Take

        project.shots[1].takes.append(Take(id="t_00000099", settings={"clipSeconds": 60.0}))
        info = paused_progress(project, project.batches[-1])
        self.assertEqual(info["etaSeconds"], 10 + 60 + 60)
        self.assertEqual(info["etaBasis"], "project")  # the clips are priced from the project's other take


class RestartTests(QueueCase):
    def test_a_batch_cut_off_by_a_restart_is_paused_then_resumes_where_it_stopped(self):
        project = self.new(shots=2)
        gens = FakeGenerators(self.src)
        first = self.batch(project, gens)
        first.start(project.revision)
        first._step(first.steps[0][0], "still")  # one step made, then the process dies mid-way through the next
        first._update(lambda p, f: setattr(first._result(p, project.shots[1].id, "still"), "state", "running"))
        LIVE_BATCHES.discard(first.id)
        with batch_module._BATCHES_LOCK:
            batch_module.BATCHES.pop(first.id, None)
        loaded = self.store.load(project.id)
        record = loaded.batches[-1]
        self.assertEqual(record.state, "paused")
        self.assertEqual(record.pauseReason, "restart")
        self.assertEqual([r.state for r in record.results], ["done", "queued", "queued", "queued"])
        self.assertEqual([s.status for s in loaded.shots], ["idle", "idle"])  # the still waits for its clip
        again = self.batch(loaded, FakeGenerators(self.src))
        again.resume(first.id, loaded.revision)
        again.run()
        done = self.store.load(project.id)
        self.assertEqual(done.batches[-1].state, "done")
        self.assertEqual([len(s.takes) for s in done.shots], [1, 1])
        self.assertTrue(all(s.active().clip for s in done.shots))


class QueueRouteTests(StoryboardRouteCase):
    def test_pause_resume_and_cancel_over_http(self):
        pid, data = self.create(shots=2)
        release = threading.Event()

        class Slow(FakeGenerators):
            def make_still(inner, shot, project):  # noqa: N805
                release.wait(5)
                return super().make_still(shot, project)

        gens = Slow(self.src)
        with mock.patch("backend_service.storyboard.generators.studio_generators", fake_wiring(gens)):
            started = self.post(f"/projects/{pid}/batches", {"revision": data["project"]["revision"], "what": "still"})
            bid = started["batchId"]
            self.assertEqual(self.post(f"/projects/{pid}/batches/{bid}/pause"), {"pausing": True})
            release.set()
            info = self.wait(lambda: (i := self.get(f"/projects/{pid}/batches/{bid}"))["batch"]["state"] == "paused" and i)
            self.assertTrue(info["progress"]["paused"])
            self.assertEqual(info["progress"]["remainingSteps"], 1)
            self.post(f"/projects/{pid}/batches/{bid}/pause", code=404)  # not running any more
            revision = self.get(f"/projects/{pid}")["project"]["revision"]
            self.post(f"/projects/{pid}/batches/{bid}/resume", {"revision": revision - 1}, code=409)
            resumed = self.post(f"/projects/{pid}/batches/{bid}/resume", {"revision": revision})
            self.assertEqual(resumed["batchId"], bid)
            done = self.wait(lambda: (i := self.get(f"/projects/{pid}/batches/{bid}"))["batch"]["state"] == "done" and i)
        self.assertEqual(done["batch"]["resumes"], 1)
        revision = self.get(f"/projects/{pid}")["project"]["revision"]
        self.post(f"/projects/{pid}/batches/{bid}/resume", {"revision": revision}, code=409)  # done, not paused

    def test_cancel_a_paused_batch_over_http(self):
        pid, data = self.create(shots=2)
        project = json.loads(json.dumps(data["project"]))
        from backend_service.storyboard import store as store_module

        store = store_module.default_store()

        def stale(p, folder):
            from backend_service.storyboard.models import BatchRecord, BatchResult

            p.batches = [BatchRecord(id="b_feedbeef", what="still", state="running", results=[
                BatchResult(shotId=project["shots"][0]["id"], kind="still", state="queued")])]

        store.mutate(pid, None, stale, "test", reversible=False)
        self.assertEqual(self.get(f"/projects/{pid}/batches/b_feedbeef")["batch"]["state"], "paused")
        self.assertEqual(self.post(f"/projects/{pid}/batches/b_feedbeef/cancel"), {"cancelled": True})
        self.assertEqual(self.get(f"/projects/{pid}/batches/b_feedbeef")["batch"]["state"], "cancelled")
        self.post(f"/projects/{pid}/batches/b_nothere1/cancel", code=404)
