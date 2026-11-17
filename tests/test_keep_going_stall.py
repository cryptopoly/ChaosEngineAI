"""FU-207: Keep going stops with ``job_stalled`` when a job the run waits on makes no progress, and only then."""

import threading
import unittest
from unittest import mock

from backend_service.keep_going.run import reason_text
from backend_service.keep_going.stall import TurnWatch
from tests.test_keep_going_controller import SID, Base

MIN = 60.0


class StubJob:
    """A chat job: ``step`` is its progress; ``finish`` ends it."""

    def __init__(self, job_id="job-36", label="captions 7271cd32dfa42cae"):
        self.id, self.label, self.state, self.kind, self.error = job_id, label, "running", "voice", None
        self.step = 0
        self.output = None
        self.done = threading.Event()

    def describe(self):
        info = {"jobId": self.id, "label": self.label, "state": self.state,
                "progress": {"phase": "transcribing", "step": self.step, "totalSteps": 3, "elapsedSeconds": 999}}
        if self.output is not None:
            info["output"] = self.output
        return info

    def finish(self):
        self.state, self.output = "done", {"captions": 3}
        self.done.set()


class TurnWatchTests(Base):
    def setUp(self):
        super().setUp()
        self.begin()
        self.job = StubJob()
        self.hooks.jobs = [self.job]
        self.watch = TurnWatch(self.ctrl, SID)

    def at(self, minutes):
        self.clock.now = 1000.0 + minutes * MIN
        return self.watch.stalled(self.job)

    def test_a_job_that_never_progresses_stops_the_run(self):
        self.assertFalse(self.at(0))
        self.assertFalse(self.at(10))
        status = self.ctrl.status(SID)
        self.assertEqual(status["waitingOn"]["label"], "captions 7271cd32dfa42cae")
        self.assertEqual(status["waitingOn"]["minutes"], 10)
        self.assertFalse(self.at(19.9))
        self.assertTrue(self.at(20))
        run = self.ctrl.get(SID)
        self.assertEqual(run.state, "stopped")
        self.assertEqual(run.stop_code, "job_stalled")
        self.assertEqual(run.stop_reason, "A job has made no progress for 20 minutes: captions 7271cd32dfa42cae")
        self.assertEqual(self.watch.reason, run.stop_reason)
        self.assertTrue(run.abort.is_set())  # the turn ends through the run's abort event
        self.assertEqual(self.job.state, "running")  # the job is left as it is
        self.assertIsNone(self.ctrl.status(SID)["waitingOn"])
        self.assertTrue(self.ctrl.status(SID)["resumable"])

    def test_a_job_that_progresses_slowly_never_stops_the_run(self):
        for minute in range(0, 120, 15):
            self.job.step = minute // 15
            self.assertFalse(self.at(minute))
        self.assertEqual(self.ctrl.get(SID).state, "running")

    def test_elapsed_time_alone_is_not_progress(self):
        self.job.describe = lambda: {"state": "running", "progress": {"elapsedSeconds": self.clock.now,
                                                                      "etaSeconds": 5000 - self.clock.now}}
        self.assertFalse(self.at(0))
        self.assertTrue(self.at(25))

    def test_another_chat_job_finishing_counts_as_progress(self):
        other = StubJob("job-37", "speech")
        self.hooks.jobs.append(other)
        self.assertFalse(self.at(0))
        other.finish()
        self.assertFalse(self.at(15))
        self.assertFalse(self.at(30))
        self.assertTrue(self.at(35))

    def test_limit_comes_from_the_run(self):
        self.ctrl.get(SID).stall_minutes = 5
        self.assertFalse(self.at(0))
        self.assertTrue(self.at(5))
        self.assertEqual(self.ctrl.get(SID).stop_reason, "A job has made no progress for 5 minutes: captions 7271cd32dfa42cae")

    def test_no_run_means_no_stall_guard(self):
        self.ctrl.stop(SID)
        self.assertFalse(self.at(0))
        self.assertFalse(self.at(500))

    def test_resume_after_a_stall(self):
        self.at(0)
        self.at(20)
        self.job.finish()
        run = self.ctrl.resume(SID, grant_mode="manual", stall_minutes=30)
        self.assertEqual(run.state, "running")
        self.assertEqual(run.stall_minutes, 30)
        self.assertEqual(len(self.hooks.turns), 1)  # the nudge after the resume

    def test_stall_minutes_clamped_and_saved(self):
        self.ctrl.stop(SID)
        self.ctrl.start("chat-2", stall_minutes=1)
        self.assertEqual(self.ctrl.get("chat-2").stall_minutes, 5)
        self.ctrl.start("chat-3", stall_minutes=10_000)
        self.assertEqual(self.ctrl.get("chat-3").stall_minutes, 240)
        self.assertEqual(self.make().get("chat-3").stall_minutes, 240)

    def test_reason_text(self):
        self.assertEqual(reason_text("job_stalled", detail="render p1", minutes=20),
                         "A job has made no progress for 20 minutes: render p1")


class WaitingRunTests(Base):
    """Between turns the run waits on chat jobs; a hung one stops it too."""

    def test_waiting_run_stops_on_a_stalled_job(self):
        self.begin()
        job = StubJob()
        self.hooks.jobs = [job]
        self.finish("Started captions.")
        self.assertEqual(self.ctrl.get(SID).state, "waiting")
        self.ctrl.poll_waiting()
        self.assertEqual(self.ctrl.status(SID)["waitingOn"]["label"], job.label)
        self.clock.now += 21 * MIN
        self.ctrl.poll_waiting()
        run = self.ctrl.get(SID)
        self.assertEqual(run.stop_code, "job_stalled")

    def test_waiting_run_with_progress_goes_on(self):
        self.begin()
        job = StubJob()
        self.hooks.jobs = [job]
        self.finish("Started captions.")
        for _ in range(6):
            self.ctrl.poll_waiting()
            job.step += 1
            self.clock.now += 10 * MIN
        self.assertEqual(self.ctrl.get(SID).state, "waiting")
        job.finish()
        self.ctrl.poll_waiting()
        self.assertEqual(self.ctrl.get(SID).state, "running")  # nudged once the job ended


class StudioWaitTests(Base):
    """The Chat ``wait_for_job`` with the watch: it returns at the stall, or when the job finishes."""

    def wait_tool(self, watch):
        from backend_service.tools.studio import StudioTool, _bridge

        tools, handlers = _bridge()
        spec = next(t for t in tools if t["name"] == "wait_for_job")
        return StudioTool(spec, handlers["wait_for_job"], None, watch=watch)

    def run_wait(self, job, tick_minutes):
        self.begin()
        self.hooks.jobs = [job]
        watch = TurnWatch(self.ctrl, SID)
        tool = self.wait_tool(watch)
        clock = self.clock

        def ticking():
            clock.now += tick_minutes * MIN
            return clock.now

        self.ctrl.clock = ticking
        with mock.patch("backend_service.tools.studio._WAIT_SLICE_SECONDS", 0.001), \
                mock.patch("backend_service.mcp_bridge.jobs.STORE.get", lambda _id: job):
            return tool._wait({"job_id": job.id})

    def test_wait_returns_at_the_stall(self):
        job = StubJob()
        out = self.run_wait(job, tick_minutes=1)
        self.assertTrue(out.data["stalled"])
        self.assertIn("no progress for 20 minutes: captions 7271cd32dfa42cae", out.data["next"])
        self.assertEqual(self.ctrl.get(SID).stop_code, "job_stalled")
        self.assertEqual(job.state, "running")

    def test_wait_on_a_job_that_finishes(self):
        job = StubJob()
        timer = threading.Timer(0.05, job.finish)
        timer.start()
        out = self.run_wait(job, tick_minutes=0)
        timer.join()
        self.assertNotIn("stalled", out.data)
        self.assertEqual(out.data["state"], "done")
        self.assertEqual(self.ctrl.get(SID).state, "running")
        self.assertIsNone(self.ctrl.get(SID).waiting_on)


if __name__ == "__main__":
    unittest.main()
