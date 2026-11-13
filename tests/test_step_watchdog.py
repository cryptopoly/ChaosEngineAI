import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers import step_watchdog
from backend_service.helpers.step_watchdog import StepWatchdog, stall_limit_seconds
from backend_service.progress import PHASE_DIFFUSING, PHASE_LOADING, GenerationCancelled, ProgressTracker


def _running(total: int = 4) -> ProgressTracker:
    progress = ProgressTracker(kind="video")
    progress.begin(run_label="t", total_steps=total, phase=PHASE_DIFFUSING)
    progress.set_step(0, total=total)
    return progress


def _wait_for(condition, seconds: float = 3.0) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


class StallLimitTests(unittest.TestCase):
    def test_before_two_steps_the_first_floor_applies(self):
        self.assertEqual(stall_limit_seconds([], first_floor=600), 600)
        self.assertEqual(stall_limit_seconds([30.0], first_floor=600), 600)

    def test_the_limit_scales_with_the_measured_step_time_and_never_drops_below_the_floor(self):
        self.assertEqual(stall_limit_seconds([1.2, 1.3, 1.2], factor=8, floor=120), 120)
        self.assertEqual(stall_limit_seconds([40.0, 42.0, 41.0], factor=8, floor=120), 8 * 41.0)


class WatchdogTests(unittest.TestCase):
    def test_a_step_that_never_ends_raises_the_cancel_flag_with_a_message(self):
        progress = _running()
        hard = mock.Mock()
        with StepWatchdog(progress, kind="video", first_floor=0.2, grace=60, poll=0.02, hard_stop=hard) as dog:
            self.assertTrue(_wait_for(lambda: progress.is_cancelled()))
        self.assertTrue(dog.stalled)
        self.assertIn("step 1 of 4", dog.message)
        self.assertEqual(progress.stall_message(), dog.message)
        hard.assert_not_called()

    def test_a_step_that_is_still_stuck_after_the_grace_runs_the_hard_stop_once(self):
        progress = _running()
        hard = mock.Mock()
        with StepWatchdog(progress, kind="video", first_floor=0.1, grace=0.2, poll=0.02, hard_stop=hard) as dog:
            self.assertTrue(_wait_for(lambda: hard.called))
        self.assertTrue(dog.hard_stopped)
        hard.assert_called_once()
        self.assertIn("without finishing", hard.call_args.args[0])

    def test_steady_progress_never_trips_it(self):
        progress = _running(total=6)
        hard = mock.Mock()
        with StepWatchdog(progress, kind="video", first_floor=0.3, floor=0.3, factor=8, grace=0.1, poll=0.02, hard_stop=hard) as dog:
            for step in range(1, 6):
                time.sleep(0.08)
                progress.set_step(step, total=6)
        self.assertFalse(dog.stalled)
        self.assertFalse(progress.is_cancelled())
        hard.assert_not_called()

    def test_loading_and_decoding_are_not_timed(self):
        progress = ProgressTracker(kind="image")
        progress.begin(run_label="t", total_steps=4, phase=PHASE_LOADING)
        hard = mock.Mock()
        with StepWatchdog(progress, kind="image", first_floor=0.1, grace=0.1, poll=0.02, hard_stop=hard) as dog:
            time.sleep(0.4)
        self.assertFalse(dog.stalled)
        # every step done, only the decode left: also not a stall
        progress.set_phase(PHASE_DIFFUSING)
        progress.set_step(4, total=4)
        with StepWatchdog(progress, kind="image", first_floor=0.1, grace=0.1, poll=0.02, hard_stop=hard) as dog:
            time.sleep(0.4)
        self.assertFalse(dog.stalled)
        hard.assert_not_called()

    def test_the_limit_is_measured_from_the_steps_already_seen(self):
        progress = _running(total=20)
        hard = mock.Mock()
        # steps take about 0.05 s, so 6x that is the limit (floor lowered for the test)
        with StepWatchdog(progress, kind="video", first_floor=5, floor=0.05, factor=6, grace=60, poll=0.01, hard_stop=hard) as dog:
            for step in range(1, 5):
                time.sleep(0.05)
                progress.set_step(step, total=20)
            self.assertTrue(_wait_for(lambda: progress.is_cancelled()))
        self.assertTrue(dog.stalled)


class FakePipelineTests(unittest.TestCase):
    """A fake pipeline whose second step sleeps: the run ends with the watchdog's message."""

    def test_a_slow_step_ends_the_run_with_a_readable_error(self):
        progress = _running(total=3)

        def on_step_end(step: int) -> None:
            progress.set_step(step + 1, total=3)
            if progress.is_cancelled():
                raise GenerationCancelled(progress.stall_message() or "cancelled by user")

        def fake_pipeline() -> None:
            for step in range(3):
                time.sleep(0.9 if step == 1 else 0.02)  # step 2 hangs past the limit
                on_step_end(step)

        with StepWatchdog(progress, kind="video", first_floor=0.3, floor=0.3, factor=8, grace=60, poll=0.02, hard_stop=mock.Mock()):
            with self.assertRaises(GenerationCancelled) as ctx:
                fake_pipeline()
        self.assertIn("The video stopped", str(ctx.exception))
        self.assertIn("step 2 of 3", str(ctx.exception))

    def test_a_user_cancel_keeps_its_own_message(self):
        progress = _running(total=3)
        progress.request_cancel()
        self.assertIsNone(progress.stall_message())

    def test_begin_clears_an_old_stall_message(self):
        progress = _running()
        progress.mark_stalled("old")
        progress.begin(run_label="next", total_steps=2, phase=PHASE_DIFFUSING)
        self.assertIsNone(progress.stall_message())


class EnvironmentTuningTests(unittest.TestCase):
    def test_limits_and_the_hard_stop_switch_come_from_the_environment(self):
        with mock.patch.dict(os.environ, {"CHAOSENGINE_STALL_FACTOR": "3", "CHAOSENGINE_STALL_FLOOR": "7",
                                          "CHAOSENGINE_STALL_FIRST_FLOOR": "9", "CHAOSENGINE_STALL_GRACE": "11",
                                          "CHAOSENGINE_STALL_HARD_STOP": "0"}):
            dog = StepWatchdog(ProgressTracker(kind="image"), kind="image")
        self.assertEqual((dog._factor, dog._floor, dog._first_floor, dog._grace), (3.0, 7.0, 9.0, 11.0))
        self.assertIsNone(dog._hard_stop("x"))  # switched off: no process exit

    def test_a_bad_value_falls_back_to_the_default(self):
        with mock.patch.dict(os.environ, {"CHAOSENGINE_STALL_FLOOR": "soon"}):
            dog = StepWatchdog(ProgressTracker(kind="image"), kind="image")
        self.assertEqual(dog._floor, step_watchdog.DEFAULT_FLOOR_SECONDS)


class HardStopTests(unittest.TestCase):
    def test_exit_backend_records_the_reason_and_exits_with_75(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"CHAOSENGINE_DATA_DIR": tmp}), \
                mock.patch.object(step_watchdog.os, "_exit") as exit_mock, mock.patch.object(step_watchdog.logging, "shutdown"):
            step_watchdog.exit_backend("step 3 of 20 ran for 900 s")
            record = json.loads((Path(tmp) / "last_stall.json").read_text(encoding="utf-8"))
        self.assertEqual(record["message"], "step 3 of 20 ran for 900 s")
        exit_mock.assert_called_once_with(75)


if __name__ == "__main__":
    unittest.main()
