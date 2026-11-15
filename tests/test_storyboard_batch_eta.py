"""FU-153 item 1: the first clip of a batch is priced from the catalogue (scaled by estimate.py) until one is measured."""

import time
from unittest import mock

from backend_service.storyboard.batch import Batch
from tests.storyboard_fixtures import FakeGenerators, StoreCase

IMAGE = {"id": "img", "name": "Img One", "estimatedGenerationSeconds": 5.0}
VIDEO = {"id": "vid", "name": "Vid One", "estimatedGenerationSeconds": 40.0, "defaultDurationSeconds": 4.0}


class BatchEtaTests(StoreCase):
    def batch(self, what="clip", shots=2, models=(IMAGE, VIDEO)):
        p = self.new(shots=shots)
        gens = FakeGenerators(self.src)
        batch = Batch(self.store, p.id, what, gens.make_still, gens.make_clip, gens.unload, gens.last_frame,
                      shot_ids=[s.id for s in p.shots], force=True)
        batch.start(p.revision)
        if models is not None:
            batch.use_models(*models)
        batch._began = time.monotonic()
        return batch, p

    def run_one(self, batch, seconds=2.0):
        """The state of a run with step 0 finished (took ``seconds``) and step 1 running."""
        batch._finished_steps.append((batch.steps[0][1], seconds))
        batch._processed = 1
        batch._current = (batch.steps[1][0], batch.steps[1][1], time.monotonic())

    def test_first_clip_uses_the_catalogue_figure(self):
        batch, p = self.batch()
        batch._current = (batch.steps[0][0], "clip", time.monotonic())
        info = batch.progress()
        # Two 4 s shots at the 40 s catalogue clip: the running one just began, so about 80 s are left.
        self.assertEqual(info["etaBasis"], "catalog")
        self.assertAlmostEqual(info["etaSeconds"], 80, delta=2)

    def test_not_started_batch_has_the_whole_catalogue_time(self):
        batch, _ = self.batch(what="both", shots=1)
        info = batch.progress()
        self.assertEqual((info["etaBasis"], info["etaSeconds"]), ("catalog", 45))  # one still 5 s, one clip 40 s

    def test_measured_clips_replace_the_catalogue_figure(self):
        batch, _ = self.batch()
        self.run_one(batch, seconds=3.0)
        info = batch.progress()
        self.assertEqual(info["etaBasis"], "clips")
        self.assertLessEqual(info["etaSeconds"], 3)  # the running clip is priced at the 3 s measured, not 40 s

    def test_a_mix_of_measured_and_catalogue_prices_reports_catalog(self):
        batch, _ = self.batch(what="both", shots=2)  # stills 0, 1 then clips 0, 1
        self.run_one(batch, seconds=2.0)  # one still done, second still running; the clips have no measure yet
        info = batch.progress()
        self.assertEqual(info["etaBasis"], "catalog")
        self.assertGreater(info["etaSeconds"], 70)  # two clips at the catalogue price

    def test_without_models_there_is_still_no_eta_for_the_first_clip(self):
        batch, _ = self.batch(models=None)
        batch._current = (batch.steps[0][0], "clip", time.monotonic())
        info = batch.progress()
        self.assertNotIn("etaSeconds", info)
        self.assertIn("elapsedSeconds", info)

    def test_a_pricing_failure_never_breaks_progress(self):
        batch, _ = self.batch()
        batch._current = (batch.steps[0][0], "clip", time.monotonic())
        with mock.patch("backend_service.storyboard.estimate.estimate", side_effect=RuntimeError("boom")):
            info = batch.progress()
        self.assertNotIn("etaSeconds", info)

    def test_a_model_without_a_catalogue_time_gives_no_eta(self):
        batch, _ = self.batch(models=(IMAGE, {"id": "vid", "name": "Vid One"}))
        batch._current = (batch.steps[0][0], "clip", time.monotonic())
        self.assertNotIn("etaSeconds", batch.progress())


class BatchInnerStepTests(BatchEtaTests):
    def test_the_denoise_step_of_the_running_still_is_reported(self):
        batch, _ = self.batch(what="still", shots=1)
        batch._current = (batch.steps[0][0], "still", time.monotonic())
        snap = {"active": True, "phase": "diffusing", "step": 27, "totalSteps": 40}
        with mock.patch("backend_service.progress.IMAGE_PROGRESS.snapshot", return_value=snap):
            info = batch.progress()
        self.assertEqual((info["innerStep"], info["innerTotal"]), (27, 40))

    def test_no_denoise_step_while_the_tracker_is_idle_or_loading(self):
        batch, _ = self.batch(what="still", shots=1)
        batch._current = (batch.steps[0][0], "still", time.monotonic())
        for snap in ({"active": False, "phase": "idle", "step": 0, "totalSteps": 0},
                     {"active": True, "phase": "loading", "step": 0, "totalSteps": 0}):
            with mock.patch("backend_service.progress.IMAGE_PROGRESS.snapshot", return_value=snap):
                self.assertNotIn("innerStep", batch.progress())
