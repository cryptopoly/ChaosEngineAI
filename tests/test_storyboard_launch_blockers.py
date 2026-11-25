"""Launch blockers from the running-app review (2026-11): a superseded paused batch is not paused work, a final clip
at the size its model makes is not stale, and the Check stage reads as it should. No model runs."""

from backend_service.storyboard import guide, quality
from backend_service.storyboard.batch import Batch
from backend_service.storyboard.batch_queue import superseded
from backend_service.storyboard.timeline import compute_timeline
from tests.storyboard_fixtures import FakeGenerators, fake_mp4
from tests.test_storyboard_queue import PausingGenerators, QueueCase

LTX = "diffusers/LTX-2.3-Diffusers"  # makes 1280 x 720 as 1248 x 704 (multiples of 32 within 0.92 megapixels)


class Case(QueueCase):
    def guide(self, project):
        return guide.build(project, compute_timeline(project), self.folder(project))

    def stage(self, project, stage_id):
        return next(s for s in self.guide(project)["stages"]["stages"] if s["id"] == stage_id)


class SupersededPausedBatchTests(Case):
    def paused_draft(self, shots=2):
        project = self.new(shots=shots)
        gens = PausingGenerators(self.src, 1)
        first = self.batch(project, gens, draft=True, draft_sizes={"still": (640, 384), "clip": (640, 384)})
        gens.batch = first
        first.start(project.revision)
        first.run()
        return self.store.load(project.id), first

    def test_a_paused_batch_with_work_left_still_counts(self):
        project, _ = self.paused_draft()
        self.assertFalse(superseded(project, project.batches[-1]))
        drafts = self.stage(project, "drafts")
        self.assertEqual((drafts["message"], drafts["action"]), ("Making the shots is paused.", "resume-batch"))

    def test_a_later_finished_batch_supersedes_it(self):
        project, first = self.paused_draft()
        later = self.batch(project, FakeGenerators(self.src), "both")
        later.start(project.revision)
        later.run()
        project = self.store.load(project.id)
        self.assertEqual([b.state for b in project.batches], ["paused", "done"])
        self.assertTrue(superseded(project, project.batches[0]))
        drafts = self.stage(project, "drafts")
        self.assertNotEqual(drafts["action"], "resume-batch")
        self.assertEqual(drafts["status"], "done")
        self.assertNotEqual(self.guide(project)["stages"]["current"], "drafts")

    def test_final_takes_on_every_shot_supersede_a_paused_draft_batch(self):
        project, _ = self.paused_draft()
        for shot in project.shots:
            project = self.op(project, "import_take", shotId=shot.id, still=str(fake_png(self.src, shot.id)),
                              clip=str(fake_mp4(self.src / f"{shot.id}.mp4")))
        self.assertTrue(superseded(project, project.batches[0]))
        self.assertNotEqual(self.stage(project, "drafts")["action"], "resume-batch")

    def test_a_paused_final_batch_is_not_covered_by_draft_takes(self):
        project = self.new(shots=2)
        gens = PausingGenerators(self.src, 1)
        first = self.batch(project, gens)
        gens.batch = first
        first.start(project.revision)
        first.run()
        project = self.store.load(project.id)
        draft = self.batch(project, FakeGenerators(self.src), "both", draft=True,
                           draft_sizes={"still": (640, 384), "clip": (640, 384)})
        draft.start(project.revision)
        draft.run()
        project = self.store.load(project.id)
        self.assertFalse(superseded(project, project.batches[0]))  # a draft is not the final the paused batch makes
        self.assertEqual(self.stage(project, "drafts")["action"], "resume-batch")


def fake_png(src, name):
    from tests.storyboard_fixtures import png

    return png(src / f"{name}.png")


class ModelAlignedSizeTests(Case):
    def finished(self, width, height, **defaults):
        project = self.new(shots=2, **defaults)

        def still(shot, p):
            return FakeGenerators(self.src).make_still(shot, p), {"settings": {"width": 1280, "height": 720}}

        def clip(shot, p, image):
            path, info = FakeGenerators(self.src).make_clip(shot, p, image)
            return path, {**info, "settings": {"width": width, "height": height, "fps": 24}}

        batch = Batch(self.store, project.id, "both", still, clip, lambda kind: None, lambda c: None)
        batch.start(project.revision)
        batch.run()
        return self.store.load(project.id)

    def test_a_clip_at_the_size_the_model_makes_is_not_stale(self):
        project = self.finished(1248, 704, videoModel=LTX)
        self.assertEqual(quality.model_sizes(project)["clip"], (1248, 704))
        self.assertEqual(quality.quality(project)["staleShots"], [])
        g = self.guide(project)
        self.assertNotIn("stale-shots", [i["code"] for i in g["readiness"]["worthFixing"]])
        self.assertNotEqual(self.stage(project, "final")["action"], "remake-stale")

    def test_with_no_known_model_a_grid_step_of_rounding_is_not_stale(self):
        project = self.finished(1248, 704)
        self.assertEqual(quality.quality(project)["staleShots"], [])

    def test_a_clip_at_another_size_is_still_stale(self):
        project = self.finished(1920, 1080)
        self.assertEqual(len(quality.quality(self.op(project, "edit_project", defaults={"videoModel": LTX}))["staleShots"]), 2)
        self.assertEqual(len(quality.quality(project)["staleShots"]), 2)
        project = self.finished(1024, 576)  # the model would not make this for 1280 x 720
        project = self.op(project, "edit_project", defaults={"videoModel": LTX})
        self.assertEqual(len(quality.quality(project)["staleShots"]), 2)

    def test_the_project_size_itself_is_still_right(self):
        project = self.finished(1280, 720)
        project = self.op(project, "edit_project", defaults={"videoModel": LTX})
        self.assertEqual(quality.quality(project)["staleShots"], [])


class CheckStageTests(Case):
    def test_something_that_stops_the_export_is_not_worth_a_look(self):
        check = self.stage(self.new(), "check")
        self.assertEqual(check["message"], "1 point stops the export.")

    def test_check_is_done_once_the_film_is_exported_and_nothing_stops_it(self):
        project = self.new(shots=2)
        project = self.op(project, "import_take", shotId=project.shots[0].id, clip=str(fake_mp4(self.src / "a.mp4")))
        self.assertEqual(self.stage(project, "check")["status"], "not-started")  # a shot has no picture yet
        exports = self.folder(project) / "exports"
        exports.mkdir(exist_ok=True)
        (exports / "Lighthouse.mp4").write_bytes(b"x")
        g = self.guide(project)
        self.assertEqual(g["readiness"]["stops"], [])
        check = next(s for s in g["stages"]["stages"] if s["id"] == "check")
        self.assertEqual(check["status"], "done")
