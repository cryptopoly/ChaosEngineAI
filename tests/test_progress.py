"""Unit tests for ``backend_service.progress`` and the GET endpoints that
surface its snapshot to the frontend.

The module backs the live progress bars in the image + video generation
modals. A regression here would make the bars revert to client-side time
estimates — not catastrophic, but the user explicitly asked for the real
signal, so we lock the contract down.

We exercise:

* ``ProgressTracker`` lifecycle (idle -> begin -> set_phase -> set_step ->
  finish), including out-of-order calls that real callers might make.
* The snapshot shape — every key the frontend reads.
* The two route endpoints (``/api/images/progress`` and ``/api/video/progress``)
  reflecting the singleton trackers' live state.
"""

from __future__ import annotations

import time
import unittest

from backend_service.progress import (
    IMAGE_PROGRESS,
    PHASE_DECODING,
    PHASE_DIFFUSING,
    PHASE_ENCODING,
    PHASE_IDLE,
    PHASE_LOADING,
    PHASE_SAVING,
    ProgressTracker,
    VIDEO_PROGRESS,
)

from tests.test_video_routes import make_client, restore_env


# Every key the frontend reads from the snapshot. If a key disappears the
# UI silently breaks (the LiveProgress component falls back to estimates),
# so we assert presence rather than relying on type checks alone.
_REQUIRED_SNAPSHOT_KEYS = {
    "kind",
    "active",
    "phase",
    "message",
    "step",
    "totalSteps",
    "startedAt",
    "updatedAt",
    "elapsedSeconds",
    "runLabel",
}


class ProgressTrackerTests(unittest.TestCase):
    """Exercise the tracker in isolation — no FastAPI app, no singletons."""

    def setUp(self) -> None:
        # Use a fresh tracker per test rather than the module singletons so
        # tests don't bleed state into each other or into the real app.
        self.tracker = ProgressTracker(kind="image")

    def test_snapshot_before_begin_reports_idle(self):
        snap = self.tracker.snapshot()
        self.assertFalse(snap["active"])
        self.assertEqual(snap["phase"], PHASE_IDLE)
        self.assertEqual(snap["step"], 0)
        self.assertEqual(snap["totalSteps"], 0)
        self.assertEqual(snap["startedAt"], 0.0)
        self.assertEqual(snap["elapsedSeconds"], 0.0)
        self.assertIsNone(snap["runLabel"])

    def test_snapshot_keys_are_complete(self):
        # Lock the contract: any key removal here is a frontend break.
        self.assertEqual(set(self.tracker.snapshot().keys()), _REQUIRED_SNAPSHOT_KEYS)

    def test_kind_round_trips_on_snapshot(self):
        video = ProgressTracker(kind="video")
        self.assertEqual(self.tracker.snapshot()["kind"], "image")
        self.assertEqual(video.snapshot()["kind"], "video")

    def test_begin_marks_active_with_run_metadata(self):
        self.tracker.begin(run_label="SDXL · 1 image", total_steps=30, message="loading model")
        snap = self.tracker.snapshot()
        self.assertTrue(snap["active"])
        self.assertEqual(snap["phase"], PHASE_LOADING)
        self.assertEqual(snap["totalSteps"], 30)
        self.assertEqual(snap["runLabel"], "SDXL · 1 image")
        self.assertEqual(snap["message"], "loading model")
        self.assertGreater(snap["startedAt"], 0.0)

    def test_begin_resets_step_counters(self):
        # Simulate a previous run that left counters non-zero.
        self.tracker.begin(total_steps=10)
        self.tracker.set_step(7)
        self.tracker.finish()
        # Now begin a fresh run — step + total should reset, not carry over.
        self.tracker.begin(total_steps=20)
        snap = self.tracker.snapshot()
        self.assertEqual(snap["step"], 0)
        self.assertEqual(snap["totalSteps"], 20)

    def test_set_phase_clears_step_counter(self):
        # Step counts are per-phase: when diffusion ends and decoding starts,
        # the bar shouldn't keep showing "step 30 of 30" until the next
        # callback fires.
        self.tracker.begin(total_steps=30, phase=PHASE_DIFFUSING)
        self.tracker.set_step(30, total=30)
        self.tracker.set_phase(PHASE_DECODING, "decoding pixels")
        snap = self.tracker.snapshot()
        self.assertEqual(snap["phase"], PHASE_DECODING)
        self.assertEqual(snap["step"], 0)
        self.assertEqual(snap["message"], "decoding pixels")

    def test_set_phase_implicitly_begins_when_idle(self):
        # If a runtime jumps straight to ``set_phase`` without ``begin``, we
        # don't want elapsedSeconds to read as ~now (since startedAt would
        # otherwise be 0). The tracker stamps ``started_at`` on the implicit
        # begin instead.
        self.tracker.set_phase(PHASE_ENCODING, "encoding prompt")
        snap = self.tracker.snapshot()
        self.assertTrue(snap["active"])
        self.assertEqual(snap["phase"], PHASE_ENCODING)
        self.assertGreater(snap["startedAt"], 0.0)
        # Elapsed should be a tiny positive number, not the whole epoch.
        self.assertLess(snap["elapsedSeconds"], 1.0)

    def test_set_step_publishes_progress_during_diffusion(self):
        self.tracker.begin(total_steps=30, phase=PHASE_DIFFUSING)
        self.tracker.set_step(12, total=30)
        snap = self.tracker.snapshot()
        self.assertEqual(snap["step"], 12)
        self.assertEqual(snap["totalSteps"], 30)

    def test_set_step_after_finish_is_a_noop(self):
        # The diffusers callback can race with ``finish()`` — if a stray
        # callback fires after the tracker has been marked idle, it must not
        # resurrect the bar.
        self.tracker.begin(total_steps=10)
        self.tracker.finish()
        self.tracker.set_step(5, total=10)
        snap = self.tracker.snapshot()
        self.assertFalse(snap["active"])
        self.assertEqual(snap["step"], 0)

    def test_set_step_clamps_negative_values(self):
        self.tracker.begin(total_steps=10)
        self.tracker.set_step(-5, total=10)
        self.assertEqual(self.tracker.snapshot()["step"], 0)

    def test_finish_clears_run_label_and_steps(self):
        self.tracker.begin(run_label="LTX · 24f", total_steps=40)
        self.tracker.set_step(10, total=40)
        self.tracker.finish(message="done")
        snap = self.tracker.snapshot()
        self.assertFalse(snap["active"])
        self.assertEqual(snap["phase"], PHASE_IDLE)
        self.assertEqual(snap["step"], 0)
        self.assertEqual(snap["totalSteps"], 0)
        self.assertIsNone(snap["runLabel"])
        self.assertEqual(snap["message"], "done")

    def test_elapsed_seconds_grows_over_time(self):
        self.tracker.begin(total_steps=10)
        first = self.tracker.snapshot()["elapsedSeconds"]
        # Sleep long enough that we're confident elapsed advances even on a
        # heavily loaded CI runner; 50ms is plenty without slowing the suite.
        time.sleep(0.05)
        second = self.tracker.snapshot()["elapsedSeconds"]
        self.assertGreater(second, first)

    def test_elapsed_seconds_is_zero_when_idle(self):
        self.tracker.begin(total_steps=10)
        time.sleep(0.05)
        self.tracker.finish()
        snap = self.tracker.snapshot()
        self.assertEqual(snap["elapsedSeconds"], 0.0)

    def test_phase_constants_match_frontend_contract(self):
        # The frontend modal phase IDs must match these strings exactly —
        # see ``ImageGenerationModal.tsx`` / ``VideoGenerationModal.tsx``.
        self.assertEqual(PHASE_LOADING, "loading")
        self.assertEqual(PHASE_ENCODING, "encoding")
        self.assertEqual(PHASE_DIFFUSING, "diffusing")
        self.assertEqual(PHASE_DECODING, "decoding")
        self.assertEqual(PHASE_SAVING, "saving")
        self.assertEqual(PHASE_IDLE, "idle")


class ProgressEndpointTests(unittest.TestCase):
    """Exercise the GET endpoints that wrap ``snapshot()`` for the frontend."""

    def setUp(self) -> None:
        self.client, self.tempdir, self.env_snapshot = make_client()
        # Reset both singletons so a previous test (or a real backend run
        # during interactive dev) can't leak state into the assertions.
        IMAGE_PROGRESS.finish()
        VIDEO_PROGRESS.finish()

    def tearDown(self) -> None:
        IMAGE_PROGRESS.finish()
        VIDEO_PROGRESS.finish()
        restore_env(self.env_snapshot)
        self.tempdir.cleanup()

    def test_image_progress_endpoint_returns_idle_by_default(self):
        response = self.client.get("/api/images/progress")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("progress", payload)
        snap = payload["progress"]
        self.assertEqual(snap["kind"], "image")
        self.assertFalse(snap["active"])
        self.assertEqual(set(snap.keys()), _REQUIRED_SNAPSHOT_KEYS)

    def test_video_progress_endpoint_returns_idle_by_default(self):
        response = self.client.get("/api/video/progress")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("progress", payload)
        snap = payload["progress"]
        self.assertEqual(snap["kind"], "video")
        self.assertFalse(snap["active"])
        self.assertEqual(set(snap.keys()), _REQUIRED_SNAPSHOT_KEYS)

    def test_image_progress_endpoint_reflects_live_singleton(self):
        # Drive the singleton the same way the runtime would — the endpoint
        # must surface that state without any extra plumbing.
        IMAGE_PROGRESS.begin(run_label="SDXL · 1 image", total_steps=30, phase=PHASE_DIFFUSING)
        IMAGE_PROGRESS.set_step(11, total=30)
        snap = self.client.get("/api/images/progress").json()["progress"]
        self.assertTrue(snap["active"])
        self.assertEqual(snap["phase"], PHASE_DIFFUSING)
        self.assertEqual(snap["step"], 11)
        self.assertEqual(snap["totalSteps"], 30)
        self.assertEqual(snap["runLabel"], "SDXL · 1 image")
        self.assertGreater(snap["startedAt"], 0.0)

    def test_video_progress_endpoint_reflects_live_singleton(self):
        VIDEO_PROGRESS.begin(run_label="LTX · 24f", total_steps=50, phase=PHASE_DIFFUSING)
        VIDEO_PROGRESS.set_step(7, total=50)
        snap = self.client.get("/api/video/progress").json()["progress"]
        self.assertTrue(snap["active"])
        self.assertEqual(snap["phase"], PHASE_DIFFUSING)
        self.assertEqual(snap["step"], 7)
        self.assertEqual(snap["totalSteps"], 50)
        self.assertEqual(snap["runLabel"], "LTX · 24f")

    def test_image_progress_endpoint_returns_idle_after_finish(self):
        IMAGE_PROGRESS.begin(total_steps=30)
        IMAGE_PROGRESS.set_step(15, total=30)
        IMAGE_PROGRESS.finish()
        snap = self.client.get("/api/images/progress").json()["progress"]
        self.assertFalse(snap["active"])
        self.assertEqual(snap["phase"], PHASE_IDLE)
        self.assertEqual(snap["step"], 0)
        self.assertEqual(snap["totalSteps"], 0)

    def test_image_and_video_singletons_are_isolated(self):
        # The two modals can't run at once in practice, but the singletons
        # must not share state — toggling image must not affect video.
        IMAGE_PROGRESS.begin(run_label="image", total_steps=10, phase=PHASE_DIFFUSING)
        image_snap = self.client.get("/api/images/progress").json()["progress"]
        video_snap = self.client.get("/api/video/progress").json()["progress"]
        self.assertTrue(image_snap["active"])
        self.assertFalse(video_snap["active"])
        self.assertEqual(image_snap["kind"], "image")
        self.assertEqual(video_snap["kind"], "video")


if __name__ == "__main__":
    unittest.main()
