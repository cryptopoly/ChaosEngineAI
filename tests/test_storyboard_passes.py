"""Draft and final passes (S6) and the fast default still model (S2)."""

from types import SimpleNamespace
from unittest import mock

from backend_service.storyboard.batch import Batch, BatchRefused, draft_size
from backend_service.storyboard.estimate import estimate, plan_steps
from backend_service.storyboard.models import Project, Take
from backend_service.storyboard.store import RevisionConflict
from tests.storyboard_fixtures import FakeGenerators, StoreCase
from tests.test_storyboard_routes import StoryboardRouteCase, fake_wiring

HW = SimpleNamespace(kind="apple", bucket="apple-64", label="Apple Silicon, 64 GB", total_gb=64.0, usable_gb=48.0)
FAST = {"id": "fast", "name": "Fast One", "estimatedGenerationSeconds": 10.0,
        "estimateRef": {"width": 1280, "height": 720, "frames": 1, "steps": 4}}
SLOW = {"id": "slow", "name": "Slow One", "estimatedGenerationSeconds": 400.0,
        "estimateRef": {"width": 1280, "height": 720, "frames": 1, "steps": 40}}
VIDEO = {"id": "vid", "name": "Vid One", "estimatedGenerationSeconds": 40.0, "defaultDurationSeconds": 4.0}


class RecordingGenerators(FakeGenerators):
    """Also records the project each call was given (the run's model and size)."""

    def __init__(self, src):
        super().__init__(src)
        self.seen = []

    def make_still(self, shot, project):
        self.seen.append(("still", shot.id, project.defaults.imageModel, project.defaults.width, project.defaults.height))
        return super().make_still(shot, project)

    def make_clip(self, shot, project, still):
        self.seen.append(("clip", shot.id, project.defaults.videoModel, project.defaults.width, project.defaults.height))
        return super().make_clip(shot, project, still)


class ModelTests(StoreCase):
    def test_old_takes_read_as_final_and_shots_as_not_kept(self):
        self.assertEqual(Take(id="t_1").renderPass, "final")
        p = self.new(shots=1)
        raw = p.model_dump(mode="json")
        for shot in raw["shots"]:
            shot.pop("kept")
        self.assertFalse(Project.model_validate(raw).shots[0].kept)

    def test_draft_size_halves_rounds_to_64_and_respects_a_minimum_side(self):
        self.assertEqual(draft_size(1280, 720), (640, 384))
        self.assertEqual(draft_size(720, 1280), (384, 640))
        self.assertEqual(draft_size(1024, 1024), (512, 512))
        self.assertEqual(draft_size(854, 480), (448, 256))  # never under 256
        # A model that needs 768 px on the short side: a draft never goes below it, nor above the project size.
        self.assertEqual(draft_size(1280, 720, 768), (1280, 720))
        self.assertEqual(draft_size(2560, 1440, 768), (1344, 768))


class KeptOpTests(StoreCase):
    def test_set_kept_is_revision_checked_and_undoable(self):
        p = self.new(shots=2)
        a = p.shots[0].id
        kept = self.op(p, "set_kept", shotId=a, kept=True)
        self.assertTrue(kept.shot(a).kept)
        with self.assertRaises(RevisionConflict):
            self.op(p, "set_kept", shotId=a, kept=False)  # p is the stale revision
        undone = self.store.undo(p.id, kept.revision)
        self.assertFalse(undone.shot(a).kept)


class PassBatchTests(StoreCase):
    def _run(self, p, gens, **kwargs):
        batch = Batch(self.store, p.id, kwargs.pop("what", "both"), gens.make_still, gens.make_clip, gens.unload,
                      gens.last_frame, **kwargs)
        batch.start(p.revision)
        batch.run()
        return batch, self.store.load(p.id)

    def test_drafts_are_marked_and_made_at_the_draft_size(self):
        p = self.new(shots=2)
        gens = RecordingGenerators(self.src)
        batch, p = self._run(p, gens, what="still", draft=True, draft_sizes={"still": (640, 384)})
        self.assertTrue(batch.record().draft)
        self.assertEqual([s[3:] for s in gens.seen], [(640, 384), (640, 384)])
        self.assertEqual((p.defaults.width, p.defaults.height), (1280, 720))  # the project itself is unchanged
        self.assertEqual({s.active().renderPass for s in p.shots}, {"draft"})
        self.assertEqual({s.status for s in p.shots}, {"done"})

    def test_final_pass_redoes_only_kept_shots_with_the_quality_model_then_clips_from_the_final_still(self):
        p = self.new(shots=2)
        a, b = p.shots[0].id, p.shots[1].id
        _, p = self._run(p, RecordingGenerators(self.src), what="still", draft=True, draft_sizes={"still": (640, 384)})
        p = self.op(p, "set_kept", shotId=a, kept=True)
        self.assertEqual([(s.id, k) for s, k in plan_steps(p, "final", None)], [(a, "still"), (a, "clip")])
        gens = RecordingGenerators(self.src)
        batch, p = self._run(p, gens, what="final", final_image_model="slow")
        self.assertEqual(batch.record().what, "final")
        self.assertEqual([s[:3] for s in gens.seen], [("still", a, "slow"), ("clip", a, None)])
        self.assertEqual(gens.seen[0][3:], (1280, 720))
        take = p.shot(a).active()
        self.assertEqual((take.renderPass, bool(take.still), bool(take.clip)), ("final", True, True))
        clip_call = next(c for c in gens.calls if c[0] == "clip")
        self.assertIn(take.id, str(clip_call[2]))  # the clip started from the new final still
        self.assertEqual(p.shot(b).active().renderPass, "draft")  # the shot not kept is untouched
        self.assertEqual(gens.unloads, ["image"])
        # Done: the kept shot now has a final take with a clip, so a second final pass has nothing to do.
        self.assertEqual(plan_steps(p, "final", None), [])
        with self.assertRaises(BatchRefused):
            Batch(self.store, p.id, "final", gens.make_still, gens.make_clip).start(p.revision)

    def test_final_pass_with_nothing_kept_is_refused(self):
        p = self.new(shots=1)
        with self.assertRaises(BatchRefused):
            Batch(self.store, p.id, "final", lambda s, q: None, lambda s, q, i: None).start(p.revision)

    def test_a_final_clip_never_completes_a_draft_still_only_take(self):
        p = self.new(shots=1)
        a = p.shots[0].id
        p = self.op(p, "edit_shot", shotId=a, startFrom="text")
        _, p = self._run(p, RecordingGenerators(self.src), what="clip", draft=True)
        draft_take = p.shot(a).active()
        self.assertEqual(draft_take.renderPass, "draft")
        p = self.op(p, "set_kept", shotId=a, kept=True)
        _, p = self._run(p, RecordingGenerators(self.src), what="final")
        self.assertNotEqual(p.shot(a).active().id, draft_take.id)
        self.assertEqual(p.shot(a).active().renderPass, "final")


class PassEstimateTests(StoreCase):
    def test_measured_times_count_only_for_their_own_pass(self):
        p = self.new(shots=2)
        _, p = PassBatchTests._run(self, p, RecordingGenerators(self.src), what="still", draft=True,
                                   draft_sizes={"still": (640, 384)})
        project = self.store.load(p.id)
        for shot in project.shots:
            shot.active().settings = {"stillSeconds": 2.0, "stillModel": "Fast One"}
        full = estimate(project, self.folder(p), what="still", shot_ids=[s.id for s in project.shots],
                        image_variant=FAST, video_variant=None, missing=[], hardware=HW)
        self.assertEqual((full["seconds"], full["basis"]), (20.0, "estimated"))  # draft takes do not price a full still
        drafts = estimate(project, self.folder(p), what="still", shot_ids=[s.id for s in project.shots],
                          image_variant=FAST, video_variant=None, missing=[], hardware=HW, render_pass="draft")
        self.assertEqual((drafts["seconds"], drafts["basis"]), (4.0, "measured"))

    def test_passes_price_drafts_small_and_fast_and_the_final_with_the_quality_model(self):
        from backend_service.storyboard.estimate import passes

        p = self.new(shots=2)
        out = passes(p, self.folder(p), shot_ids=None, draft_variant=FAST, final_variant=SLOW, video_variant=VIDEO,
                     draft_sizes={"still": (640, 384), "clip": (640, 384)})
        draft, final = out["draft"], out["final"]
        self.assertEqual((draft["stills"], draft["imageModel"], draft["width"], draft["height"]), (2, "Fast One", 640, 384))
        self.assertLess(draft["stillSeconds"], 20.0)  # smaller than the 10 s reference, twice
        self.assertEqual((final["stills"], final["clips"], final["imageModel"]), (2, 2, "Slow One"))
        self.assertEqual(final["stillSeconds"], 800.0)
        self.assertEqual(final["clipSeconds"], 80.0)
        self.assertTrue(final["assumedAllKept"])  # nothing kept yet: the final is priced for every shot
        self.assertEqual(final["seconds"], 880.0)
        # A shot with a still and no clip: Make drafts (stills) makes a new draft still, so it is priced.
        _, p = PassBatchTests._run(self, p, RecordingGenerators(self.src), what="still", draft=True,
                                   draft_sizes={"still": (640, 384)})
        p = self.store.load(p.id)
        out = passes(p, self.folder(p), shot_ids=[s.id for s in p.shots], draft_variant=FAST, final_variant=SLOW,
                     video_variant=VIDEO, draft_sizes={"still": (640, 384), "clip": (640, 384)})
        self.assertEqual((out["draft"]["stills"], out["draft"]["clips"]), (2, 2))
        self.assertIsNotNone(out["draft"]["seconds"])
        p = self.op(p, "set_kept", shotId=p.shots[0].id, kept=True)
        out = passes(p, self.folder(p), shot_ids=None, draft_variant=FAST, final_variant=SLOW, video_variant=VIDEO,
                     draft_sizes={"still": (640, 384), "clip": (640, 384)})
        self.assertEqual((out["final"]["stills"], out["final"]["assumedAllKept"]), (1, False))


class DefaultStillModelTests(StoreCase):
    def _rows(self):
        return [
            {"id": "qwen", "name": "Qwen", "fit": "fits", "secondsPerStill": 450.0},
            {"id": "gated", "name": "Gated", "fit": "fits", "secondsPerStill": 5.0, "territoryRestricted": True},
            {"id": "iris", "name": "Iris", "fit": "fits", "secondsPerStill": 15.0},
            {"id": "huge", "name": "Huge", "fit": "tight", "secondsPerStill": 1.0},
            {"id": "unknown", "name": "No figure", "fit": "fits", "secondsPerStill": None},
        ]

    def test_image_default_is_the_fastest_fitting_open_model_and_the_quality_pick_keeps_the_catalog_order(self):
        from backend_service.mcp_bridge import models as bridge_models

        with mock.patch.object(bridge_models, "installed_models", return_value=self._rows()):
            self.assertEqual(bridge_models.default_model(mock.Mock(), "image")["id"], "iris")
            self.assertEqual(bridge_models.quality_model(mock.Mock(), "image")["id"], "qwen")
            # Video keeps the catalogue's best-fit order.
            video = [{"id": "v1", "fit": "fits", "secondsPerStill": None}, {"id": "v2", "fit": "fits"}]
        with mock.patch.object(bridge_models, "installed_models", return_value=video):
            self.assertEqual(bridge_models.default_model(mock.Mock(), "video")["id"], "v1")

    def test_no_figure_at_all_falls_back_to_the_catalog_order(self):
        from backend_service.mcp_bridge import models as bridge_models

        rows = [{"id": "a", "fit": "fits", "secondsPerStill": None}, {"id": "b", "fit": "fits", "secondsPerStill": None}]
        with mock.patch.object(bridge_models, "installed_models", return_value=rows):
            self.assertEqual(bridge_models.default_model(mock.Mock(), "image")["id"], "a")


class PassRouteTests(StoryboardRouteCase):
    def _variants(self, request, kind, model):
        if kind == "video":
            return {**VIDEO}, None
        return ({**SLOW} if model == "slow" else {**FAST, "minSide": 768 if model == "tall" else None}), None

    def test_drafts_then_a_final_pass_through_the_routes(self):
        from backend_service.mcp_bridge import models as bridge_models

        pid, data = self.create(shots=2)
        gens = RecordingGenerators(self.src)
        quality = {"id": "slow", "name": "Slow One"}
        with mock.patch("backend_service.storyboard.generators.studio_generators", fake_wiring(gens)), \
                mock.patch("backend_service.routes.storyboard._project_variant", side_effect=self._variants), \
                mock.patch.object(bridge_models, "quality_model", lambda state, kind: quality):
            bid = self.post(f"/projects/{pid}/batches", {"revision": data["project"]["revision"], "what": "still",
                                                         "draft": True})["batchId"]
            done = self.wait(lambda: (b := self.get(f"/projects/{pid}/batches/{bid}"))["batch"]["state"] == "done" and b)
            self.assertTrue(done["batch"]["draft"])
            self.assertEqual({s[3:] for s in gens.seen}, {(640, 384)})
            project = self.get(f"/projects/{pid}")["project"]
            self.assertEqual({s["takes"][0]["renderPass"] for s in project["shots"]}, {"draft"})
            shot = project["shots"][0]["id"]
            data = self.op(pid, project["revision"], "set_kept", {"shotId": shot, "kept": True})
            estimate = self.post(f"/projects/{pid}/estimate", {"what": "final"})
            self.assertEqual((estimate["stills"], estimate["clips"], estimate["imageModel"]), (1, 1, "Slow One"))
            self.assertEqual(estimate["passes"]["final"]["imageModel"], "Slow One")
            self.assertEqual(estimate["passes"]["draft"]["imageModel"], "Fast One")
            gens.seen.clear()
            bid = self.post(f"/projects/{pid}/batches", {"revision": data["project"]["revision"], "what": "final"})["batchId"]
            self.wait(lambda: self.get(f"/projects/{pid}/batches/{bid}")["batch"]["state"] == "done")
        self.assertEqual([s[:3] for s in gens.seen], [("still", shot, "slow"), ("clip", shot, None)])
        project = self.get(f"/projects/{pid}")["project"]
        self.assertEqual(project["shots"][0]["takes"][-1]["renderPass"], "final")

    def test_a_draft_never_goes_under_the_models_smallest_side(self):
        pid, data = self.create(shots=1)
        data = self.op(pid, data["project"]["revision"], "edit_project", {"defaults": {"imageModel": "tall"}})
        gens = RecordingGenerators(self.src)
        with mock.patch("backend_service.storyboard.generators.studio_generators", fake_wiring(gens)), \
                mock.patch("backend_service.routes.storyboard._project_variant", side_effect=self._variants):
            bid = self.post(f"/projects/{pid}/batches", {"revision": data["project"]["revision"], "what": "still",
                                                         "draft": True})["batchId"]
            self.wait(lambda: self.get(f"/projects/{pid}/batches/{bid}")["batch"]["state"] == "done")
        self.assertEqual(gens.seen[0][3:], (1280, 720))

    def test_estimate_keeps_its_old_fields_and_adds_the_passes(self):
        pid, _ = self.create(shots=2)
        with mock.patch("backend_service.routes.storyboard._project_variant", side_effect=self._variants):
            out = self.post(f"/projects/{pid}/estimate", {"what": "both"})
        for key in ("what", "shots", "stills", "clips", "seconds", "peakGb", "totalGb", "diskMb", "basis", "missing",
                    "imageModel", "videoModel"):
            self.assertIn(key, out)
        self.assertEqual(out["passes"]["final"]["assumedAllKept"], True)
        self.assertEqual(out["passes"]["draft"]["stills"], 2)

    def test_still_models_names_the_fast_and_the_quality_pick(self):
        from backend_service.mcp_bridge import models as bridge_models

        rows = [{"id": "qwen", "name": "Qwen", "fit": "fits"}, {"id": "iris", "name": "Iris", "fit": "fits"}]
        variants = {"qwen": SLOW, "iris": FAST}
        with mock.patch.object(bridge_models, "installed_models", return_value=rows), \
                mock.patch.object(bridge_models, "default_model", lambda state, kind: rows[1]), \
                mock.patch("backend_service.helpers.images._find_image_variant", lambda vid: variants.get(vid)):
            out = self.get("/still-models", width=640, height=360)
        self.assertEqual((out["fast"], out["quality"]), ("iris", "qwen"))
        seconds = {m["id"]: m["secondsPerStill"] for m in out["models"]}
        self.assertLess(seconds["iris"], 10.0)  # smaller than the 1280x720 reference
