"""Redo clip only and redo still only (S11): one half of a take again, with a new seed, the other half kept."""

from types import SimpleNamespace

from backend_service.storyboard import deps
from backend_service.storyboard.batch import Batch, BatchRefused, clip_outdated, needs_work
from backend_service.storyboard.estimate import estimate, plan_steps
from tests.storyboard_fixtures import FakeGenerators, StoreCase
from tests.test_storyboard_routes import StoryboardRouteCase, fake_wiring

HW = SimpleNamespace(kind="apple", bucket="apple-64", label="Apple Silicon, 64 GB", total_gb=64.0, usable_gb=48.0)
IMAGE = {"id": "img", "name": "Img One", "estimatedGenerationSeconds": 10.0,
         "estimateRef": {"width": 1280, "height": 720, "frames": 1, "steps": 4}}
VIDEO = {"id": "vid", "name": "Vid One", "estimatedGenerationSeconds": 40.0, "defaultDurationSeconds": 4.0}


class SeedGenerators(FakeGenerators):
    """Records the seed and size each call was given, and reports a seed back like the real makers do."""

    def __init__(self, src):
        super().__init__(src)
        self.seen = []
        self.next_seed = 100

    def make_still(self, shot, project):
        self.seen.append(("still", shot.id, shot.seed, project.defaults.width, project.defaults.height))
        self.next_seed += 1
        return super().make_still(shot, project), {"settings": {"model": "Img One", "seed": self.next_seed}}

    def make_clip(self, shot, project, still):
        self.seen.append(("clip", shot.id, shot.seed, project.defaults.width, project.defaults.height))
        self.next_seed += 1
        path, info = super().make_clip(shot, project, still)
        return path, {**info, "settings": {**info["settings"], "seed": self.next_seed}}


class RedoBatchTests(StoreCase):
    def _run(self, p, gens, what, **kwargs):
        batch = Batch(self.store, p.id, what, gens.make_still, gens.make_clip, gens.unload, gens.last_frame, **kwargs)
        batch.start(p.revision)
        batch.run()
        return batch, self.store.load(p.id)

    def _made(self, shots=1, **kwargs):
        p = self.new(shots=shots)
        _, p = self._run(p, SeedGenerators(self.src), "both", **kwargs)
        return p

    def test_redo_clip_keeps_the_still_and_makes_a_new_clip_take_with_a_new_seed(self):
        p = self._made()
        shot_id = p.shots[0].id
        p = self.op(p, "edit_shot", shotId=shot_id, seed=7)  # a fixed seed: the redo still asks for a new one
        old = p.shot(shot_id).active()
        p = self.op(p, "set_kept", shotId=shot_id, kept=False)
        gens = SeedGenerators(self.src)
        batch, p = self._run(p, gens, "redo-clip", shot_ids=[shot_id])
        self.assertEqual(batch.record().what, "redo-clip")
        self.assertEqual([s[:3] for s in gens.seen], [("clip", shot_id, None)])
        shot = p.shot(shot_id)
        new = shot.active()
        self.assertNotEqual(new.id, old.id)
        self.assertEqual(len(shot.takes), 2)
        self.assertEqual((new.still, new.hashes["still"]), (old.still, old.hashes["still"]))
        self.assertNotEqual(new.hashes["clip"], old.hashes["clip"])
        self.assertIn(old.still.split("/")[-2], str(gens.calls[-1][2]))  # the clip started from the kept still
        self.assertEqual(shot.status, "done")
        self.assertEqual(new.settings["clipSeed"], gens.next_seed)
        self.assertEqual(new.settings["clipStill"], old.hashes["still"])

    def test_redo_clip_of_a_draft_stays_a_draft_at_the_draft_size(self):
        p = self._made(draft=True, draft_sizes={"still": (640, 384), "clip": (640, 384)})
        shot_id = p.shots[0].id
        gens = SeedGenerators(self.src)
        _, p = self._run(p, gens, "redo-clip", shot_ids=[shot_id], draft_sizes={"still": (640, 384), "clip": (640, 384)})
        self.assertEqual(gens.seen[0][3:], (640, 384))
        self.assertEqual(p.shot(shot_id).active().renderPass, "draft")

    def test_redo_still_keeps_the_clip_until_remade_and_marks_it_stale(self):
        p = self._made()
        shot_id = p.shots[0].id
        old = p.shot(shot_id).active()
        gens = SeedGenerators(self.src)
        _, p = self._run(p, gens, "redo-still", shot_ids=[shot_id])
        self.assertEqual([s[:3] for s in gens.seen], [("still", shot_id, None)])
        shot = p.shot(shot_id)
        new = shot.active()
        self.assertNotEqual(new.hashes["still"], old.hashes["still"])
        self.assertEqual((new.clip, new.hashes["clip"]), (old.clip, old.hashes["clip"]))
        self.assertEqual(new.measuredSeconds, old.measuredSeconds)
        self.assertTrue(clip_outdated(shot))
        self.assertEqual(shot.status, "stale")
        self.assertIn(shot_id, deps.stale_shots(p))
        # The next Make (or Redo clip) remakes only the clip, from the new still.
        self.assertEqual([(s.id, k) for s, k in plan_steps(p, "both", [shot_id])], [(shot_id, "clip")])
        self.assertTrue(needs_work(shot, "both"))
        self.assertFalse(needs_work(shot, "still"))
        gens = SeedGenerators(self.src)
        _, p = self._run(p, gens, "both")
        self.assertEqual([s[0] for s in gens.seen], ["clip"])
        final = p.shot(shot_id).active()
        self.assertEqual(final.hashes["still"], new.hashes["still"])
        self.assertEqual(p.shot(shot_id).status, "done")
        self.assertFalse(clip_outdated(p.shot(shot_id)))

    def test_redo_still_of_a_still_only_take_makes_a_new_still_only_take(self):
        p = self.new(shots=1)
        _, p = self._run(p, SeedGenerators(self.src), "still")
        shot_id = p.shots[0].id
        _, p = self._run(p, SeedGenerators(self.src), "redo-still", shot_ids=[shot_id])
        shot = p.shot(shot_id)
        self.assertEqual(len(shot.takes), 2)
        self.assertIsNone(shot.active().clip)
        self.assertEqual(shot.status, "done")

    def test_redo_still_skips_shots_with_no_still_to_make(self):
        p = self._made(shots=2)
        b = p.shots[1].id
        p = self.op(p, "edit_shot", shotId=b, startFrom="text")
        with self.assertRaises(BatchRefused):
            Batch(self.store, p.id, "redo-still", lambda s, q: None, lambda s, q, i: None, shot_ids=[b]).start(p.revision)

    def test_redo_with_no_shot_ids_takes_every_unlocked_shot_that_has_a_take(self):
        p = self._made(shots=3)
        a, b, c = (s.id for s in p.shots)
        p = self.op(p, "lock", shotId=b, locked=True)
        p = self.op(p, "add_shot", title="empty")
        self.assertEqual([(s.id, k) for s, k in plan_steps(p, "redo-clip", None)], [(a, "clip"), (c, "clip")])
        gens = SeedGenerators(self.src)
        _, p = self._run(p, gens, "redo-clip")
        self.assertEqual([s[1] for s in gens.seen], [a, c])
        self.assertEqual([len(s.takes) for s in p.shots], [2, 1, 2, 0])

    def test_naming_a_locked_shot_is_refused_without_force(self):
        p = self._made()
        p = self.op(p, "lock", shotId=p.shots[0].id, locked=True)
        with self.assertRaises(BatchRefused):
            Batch(self.store, p.id, "redo-clip", lambda s, q: None, lambda s, q, i: None,
                  shot_ids=[p.shots[0].id]).start(p.revision)

    def test_takes_made_before_s11_are_never_clip_outdated(self):
        p = self._made()
        shot = p.shots[0]
        take = shot.active()
        take.settings.pop("clipStill", None)
        self.assertFalse(clip_outdated(shot))


class RedoEstimateTests(StoreCase):
    def _made(self, shots=2):
        p = self.new(shots=shots)
        gens = FakeGenerators(self.src)
        batch = Batch(self.store, p.id, "both", gens.make_still, gens.make_clip, gens.unload, gens.last_frame)
        batch.start(p.revision)
        batch.run()
        return self.store.load(p.id)

    def test_redo_kinds_are_priced_from_the_catalogue(self):
        p = self._made()
        ids = [p.shots[0].id]
        clip = estimate(p, self.folder(p), what="redo-clip", shot_ids=ids, image_variant=IMAGE, video_variant=VIDEO,
                        missing=[], hardware=HW)
        self.assertEqual((clip["stills"], clip["clips"]), (0, 1))
        self.assertGreater(clip["seconds"], 0)
        still = estimate(p, self.folder(p), what="redo-still", shot_ids=ids, image_variant=IMAGE, video_variant=VIDEO,
                         missing=[], hardware=HW)
        self.assertEqual((still["stills"], still["clips"]), (1, 0))
        self.assertGreater(still["seconds"], 0)

    def test_an_unknown_model_gives_no_time_rather_than_a_guess(self):
        p = self._made()
        out = estimate(p, self.folder(p), what="redo-clip", shot_ids=None, image_variant=IMAGE, video_variant=None,
                       missing=["No video model is installed."], hardware=HW)
        self.assertIsNone(out["seconds"])
        self.assertEqual(out["clips"], 2)


class RedoRouteTests(StoryboardRouteCase):
    def test_the_batch_and_estimate_routes_take_the_redo_kinds(self):
        from unittest import mock

        from tests.storyboard_fixtures import FakeGenerators as Gens

        pid, data = self.create(shots=1)
        shot_id = data["project"]["shots"][0]["id"]
        gens = Gens(self.src)
        with mock.patch("backend_service.storyboard.generators.studio_generators", fake_wiring(gens)):
            started = self.post(f"/projects/{pid}/batches", {"revision": data["project"]["revision"], "what": "both"})
            self._wait(pid, started["batchId"])
            project = self.get(f"/projects/{pid}")["project"]
            started = self.post(f"/projects/{pid}/batches",
                                {"revision": project["revision"], "what": "redo-clip", "shotIds": [shot_id]})
            done = self._wait(pid, started["batchId"])
        self.assertEqual(done["batch"]["what"], "redo-clip")
        project = self.get(f"/projects/{pid}")["project"]
        self.assertEqual(len(project["shots"][0]["takes"]), 2)
        out = self.post(f"/projects/{pid}/estimate", {"what": "redo-still", "shotIds": [shot_id]})
        self.assertEqual((out["stills"], out["clips"]), (1, 0))

    def _wait(self, pid, batch_id):
        import time

        for _ in range(200):
            info = self.get(f"/projects/{pid}/batches/{batch_id}")
            if info["batch"]["state"] not in ("queued", "running"):
                return info
            time.sleep(0.02)
        self.fail("the batch did not finish")


class RedoToolSchemaTests(StoreCase):
    def test_the_agent_can_ask_for_the_redo_kinds(self):
        from backend_service.mcp_bridge.storyboard_tools import STORYBOARD_TOOLS as TOOLS

        tool = next(t for t in TOOLS if t["name"] == "storyboard_generate")
        self.assertIn("redo-clip", tool["inputSchema"]["properties"]["what"]["enum"])
        self.assertIn("redo-still", tool["inputSchema"]["properties"]["what"]["enum"])
