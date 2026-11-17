"""FU-206: a job that holds the heavy-job slot never waits behind itself, and a long wait says whom it waits on."""

import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers import job_queue
from backend_service.helpers.job_queue import HEAVY_JOBS, HeavyJobDeadlock, HeavyJobQueue
from backend_service.progress import GenerationCancelled


class ReentrantSlotTests(unittest.TestCase):
    def test_nested_slot_in_the_same_job_runs_at_once(self):
        queue = HeavyJobQueue()
        with queue.slot("voice", "agent", "captions") as outer:
            with queue.slot("voice", "ui", "captions") as inner:
                self.assertIs(inner, outer)
                self.assertEqual(queue.snapshot()["waiting"], [])
            self.assertEqual(queue.position(outer), 0)  # the nested exit leaves the outer slot held
        self.assertIsNone(queue.snapshot()["running"])

    def test_waiting_for_another_ticket_while_holding_fails_fast(self):
        queue = HeavyJobQueue()
        with queue.slot("voice", "agent", "captions"):
            other = queue.enqueue("image", "agent", "poster")
            started = time.monotonic()
            with self.assertRaises(HeavyJobDeadlock) as caught:
                queue.wait_turn(other)
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertIn("poster", str(caught.exception))
            self.assertEqual(queue.snapshot()["waiting"], [])  # the refused ticket left the line

    def test_another_thread_still_queues_behind_the_holder(self):
        queue = HeavyJobQueue()
        order: list[str] = []
        with queue.slot("voice", "agent", "captions"):
            def other() -> None:
                with queue.slot("image", "ui", "poster"):
                    order.append("poster")

            thread = threading.Thread(target=other)
            thread.start()
            time.sleep(0.2)
            order.append("captions done")
        thread.join(5)
        self.assertEqual(order, ["captions done", "poster"])

    def test_slot_released_on_exception_and_holder_cleared(self):
        queue = HeavyJobQueue()
        with self.assertRaises(ValueError):
            with queue.slot("voice", "agent", "captions"):
                with queue.slot("voice", "ui", "captions"):
                    raise ValueError("boom")
        self.assertIsNone(queue.snapshot()["running"])
        self.assertIsNone(job_queue._HELD.get())
        with queue.slot("image", "ui", "next"):  # the queue is free again
            pass

    def test_cancelled_waiter_raises_and_leaves_the_queue(self):
        queue = HeavyJobQueue()
        holder = queue.enqueue("video", "agent", "clip")
        outcome: list[str] = []

        def hold() -> None:
            queue.wait_turn(holder)

        threading.Thread(target=hold).start()
        time.sleep(0.1)

        def waiter() -> None:
            try:
                with queue.slot("voice", "ui", "captions"):
                    outcome.append("ran")
            except GenerationCancelled:
                outcome.append("cancelled")

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.1)
        self.assertEqual(queue.cancel_waiting(kind="voice"), 1)
        thread.join(5)
        self.assertEqual(outcome, ["cancelled"])
        queue.release(holder)
        self.assertEqual(queue.snapshot(), {"running": None, "waiting": []})


class LongWaitWatchdogTests(unittest.TestCase):
    def test_long_waiter_is_logged_once_and_marked_in_the_snapshot(self):
        queue = HeavyJobQueue()
        holder = queue.enqueue("voice", "agent", "captions 7271")
        queue.wait_turn(holder)
        holder.started_at = time.time() - 72 * 60
        done = threading.Event()

        def waiter() -> None:
            try:
                with queue.slot("voice", "ui", "captions"):
                    pass
            finally:
                done.set()

        with mock.patch.object(job_queue, "WAIT_WARN_SECONDS", 0.2), \
                self.assertLogs("backend_service.helpers.job_queue", level="WARNING") as logs:
            thread = threading.Thread(target=waiter)
            thread.start()
            time.sleep(1.5)
            row = queue.snapshot()["waiting"][0]
            self.assertEqual(row["waitNote"], "waiting behind captions 7271 (running 72 min)")
            queue.release(holder)
            thread.join(5)
        self.assertTrue(done.is_set())
        self.assertEqual(len(logs.output), 1)
        self.assertIn("waiting behind captions 7271", logs.output[0])

    def test_short_waiter_has_no_note(self):
        queue = HeavyJobQueue()
        holder = queue.enqueue("voice", "agent", "x")
        queue.wait_turn(holder)
        queue.enqueue("image", "ui", "y")
        self.assertNotIn("waitNote", queue.snapshot()["waiting"][0])


class AgentJobCaptionsTests(unittest.TestCase):
    """The rehearsal's deadlock: an agent captions job transcribes inside its own slot."""

    def test_captions_job_through_the_real_submit_path_finishes(self):
        from backend_service.mcp_bridge.jobs import STORE
        from backend_service.storyboard import captions

        calls: list[str] = []

        def fake_transcribe(path: Path, prompt: str, language: str | None) -> dict:
            calls.append(path.name)
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

        def run():
            reply = captions.transcribe_with_studio(Path("voice.wav"), "en")
            return {"segments": len(reply["segments"])}, []

        with mock.patch("backend_service.voice_runtime.transcribe_file", fake_transcribe):
            job = STORE.submit("voice", "captions test", run)
            finished = job.done.wait(5)
        if not finished:  # leave the shared queue usable for the other tests
            HEAVY_JOBS.cancel_waiting(source="ui")
            job.done.wait(5)
        self.assertTrue(finished, "the captions job waited behind its own slot")
        self.assertEqual(job.state, "done")
        self.assertEqual(job.output, {"segments": 1})
        self.assertEqual(calls, ["voice.wav"])
        self.assertIsNone(HEAVY_JOBS.snapshot()["running"])

    def test_a_studio_job_after_the_agent_job_runs(self):
        from backend_service.mcp_bridge.jobs import STORE

        def run():
            with HEAVY_JOBS.slot("voice", "ui", "nested"):
                time.sleep(0.1)
            return {}, []

        job = STORE.submit("voice", "agent voice", run)
        ran = threading.Event()

        def studio() -> None:
            with HEAVY_JOBS.slot("voice", "ui", "studio"):
                ran.set()

        thread = threading.Thread(target=studio)
        thread.start()
        self.assertTrue(job.done.wait(5))
        thread.join(5)
        self.assertTrue(ran.is_set())
        self.assertEqual(job.state, "done")

    def test_queue_view_carries_the_wait_note(self):
        from backend_service.mcp_bridge.jobs import STORE

        holder = HEAVY_JOBS.enqueue("voice", "agent", "captions abc")
        HEAVY_JOBS.wait_turn(holder)
        waiter = HEAVY_JOBS.enqueue("image", "ui", "poster")
        try:
            with mock.patch.object(job_queue, "WAIT_WARN_SECONDS", 0):
                view = STORE.queue_view()
            self.assertTrue(view["jobs"][1]["waitNote"].startswith("waiting behind captions abc (running"))
        finally:
            HEAVY_JOBS.release(waiter)
            HEAVY_JOBS.release(holder)


if __name__ == "__main__":
    unittest.main()
