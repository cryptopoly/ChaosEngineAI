import threading
import time
import unittest

from backend_service.helpers.job_queue import HeavyJobQueue
from backend_service.progress import GenerationCancelled


class HeavyJobQueueTests(unittest.TestCase):
    def test_jobs_run_one_at_a_time_in_order(self):
        queue = HeavyJobQueue()
        events: list[str] = []
        running = 0
        peak = 0
        lock = threading.Lock()

        def job(name: str, hold: float) -> None:
            nonlocal running, peak
            with queue.slot("image", "ui", name):
                with lock:
                    running += 1
                    peak = max(peak, running)
                events.append(f"start {name}")
                time.sleep(hold)
                events.append(f"end {name}")
                with lock:
                    running -= 1

        threads = []
        for name in ("a", "b", "c"):
            t = threading.Thread(target=job, args=(name, 0.1))
            t.start()
            threads.append(t)
            time.sleep(0.03)  # fix arrival order
        for t in threads:
            t.join(5)
        self.assertEqual(peak, 1)
        self.assertEqual(events, ["start a", "end a", "start b", "end b", "start c", "end c"])

    def test_positions_and_state(self):
        queue = HeavyJobQueue()
        first = queue.enqueue("video", "agent", "clip")
        second = queue.enqueue("image", "ui", "poster")
        queue.wait_turn(first)
        self.assertEqual(queue.position(first), 0)
        self.assertEqual(queue.position(second), 1)
        state = queue.queue_state("image")
        self.assertEqual(state, {"waiting": True, "position": 1, "agentAhead": True})
        self.assertFalse(queue.queue_state("voice")["waiting"])
        queue.release(first)
        queue.wait_turn(second)
        self.assertEqual(queue.position(second), 0)
        queue.release(second)
        self.assertEqual(queue.position(second), -1)

    def test_cancel_waiting_raises_for_waiter(self):
        queue = HeavyJobQueue()
        holder = queue.enqueue("video", "agent", "clip")
        queue.wait_turn(holder)
        outcome: list[str] = []

        def waiter() -> None:
            try:
                with queue.slot("image", "ui", "poster"):
                    outcome.append("ran")
            except GenerationCancelled:
                outcome.append("cancelled")

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)
        self.assertEqual(queue.cancel_waiting("image"), 1)
        t.join(5)
        self.assertEqual(outcome, ["cancelled"])
        self.assertIsNotNone(queue.snapshot()["running"])  # running job untouched
        queue.release(holder)

    def test_studio_cancel_leaves_agent_jobs_alone(self):
        queue = HeavyJobQueue()
        agent_running = queue.enqueue("image", "agent", "poster")
        queue.wait_turn(agent_running)
        agent_waiting = queue.enqueue("image", "agent", "second poster")
        ui_waiting = queue.enqueue("image", "ui", "studio poster")
        stopped: list[str] = []

        self.assertEqual(queue.running_source("image"), "agent")
        self.assertEqual(queue.running_source("video"), None)
        # The Studio's Cancel: nothing of the Studio's own is running, so the agent's run is not stopped;
        # its own waiting job is dropped and the agent's waiting job stays.
        self.assertTrue(queue.cancel_studio("image", lambda: stopped.append("stop") or True))
        self.assertEqual(stopped, [])
        self.assertTrue(ui_waiting.cancelled)
        self.assertFalse(agent_waiting.cancelled)
        self.assertEqual(queue.position(agent_waiting), 1)
        # Nothing of the Studio's left: Cancel reports there was nothing to cancel.
        self.assertFalse(queue.cancel_studio("image", lambda: False))
        queue.release(agent_running)

    def test_studio_cancel_stops_its_own_running_job(self):
        queue = HeavyJobQueue()
        mine = queue.enqueue("video", "ui", "clip")
        queue.wait_turn(mine)
        agent_waiting = queue.enqueue("video", "agent", "next clip")
        stopped: list[str] = []
        self.assertTrue(queue.cancel_studio("video", lambda: stopped.append("stop") or True))
        self.assertEqual(stopped, ["stop"])
        self.assertFalse(agent_waiting.cancelled)
        queue.release(mine)

    def test_cancel_waiting_can_filter_by_source(self):
        queue = HeavyJobQueue()
        holder = queue.enqueue("image", "agent", "a")
        queue.wait_turn(holder)
        mine = queue.enqueue("image", "ui", "b")
        other = queue.enqueue("image", "agent", "c")
        self.assertEqual(queue.cancel_waiting("image", source="ui"), 1)
        self.assertTrue(mine.cancelled)
        self.assertFalse(other.cancelled)
        queue.release(holder)

    def test_slot_released_after_error(self):
        queue = HeavyJobQueue()
        with self.assertRaises(ValueError):
            with queue.slot("voice", "ui"):
                raise ValueError("boom")
        self.assertIsNone(queue.snapshot()["running"])

    def test_cancel_batch_hits_only_that_batch(self):
        queue = HeavyJobQueue()
        running = queue.enqueue("image", "storyboard", "shot 1", batch="b_1")
        queue.wait_turn(running)
        mine = queue.enqueue("video", "storyboard", "shot 1 clip", batch="b_1")
        other_batch = queue.enqueue("image", "storyboard", "x", batch="b_2")
        studio = queue.enqueue("image", "ui", "poster")
        agent = queue.enqueue("image", "agent", "poster")
        stopped: list[str] = []
        self.assertEqual(queue.cancel_batch("b_1", lambda kind: stopped.append(kind) or True), 2)
        self.assertEqual(stopped, ["image"])
        self.assertTrue(mine.cancelled)
        self.assertFalse(any(t.cancelled for t in (other_batch, studio, agent)))
        self.assertEqual(queue.snapshot()["running"]["batch"], "b_1")
        self.assertEqual(queue.cancel_batch("b_9", lambda kind: True), 0)
        queue.release(running)

    def test_cancel_batch_never_stops_a_studio_job_that_starts_meanwhile(self):
        queue = HeavyJobQueue()
        mine = queue.enqueue("image", "storyboard", "shot still", batch="b_1")
        queue.wait_turn(mine)
        studio = queue.enqueue("image", "ui", "studio image")
        studio_started = threading.Event()
        running_when_stopped = []

        def finish_batch_step_and_start_studio():
            queue.release(mine)  # the batch's step ends at the worst moment
            queue.wait_turn(studio)
            studio_started.set()

        def stop_running(kind):
            # The interleaving the review found: the batch's ticket ends and the Studio job starts after
            # cancel_batch read the running ticket but before it stops the (global) tracker.
            threading.Thread(target=finish_batch_step_and_start_studio, daemon=True).start()
            studio_started.wait(0.5)
            running_when_stopped.append(queue._running)
            return True

        self.assertEqual(queue.cancel_batch("b_1", stop_running), 1)
        self.assertEqual(running_when_stopped, [mine], "stop_running must act while the batch still owns the slot")
        self.assertTrue(studio_started.wait(5))  # and the Studio job then runs normally
        queue.release(studio)

    def test_studio_cancel_leaves_a_storyboard_batch_alone(self):
        queue = HeavyJobQueue()
        running = queue.enqueue("image", "storyboard", "shot 1", batch="b_1")
        queue.wait_turn(running)
        waiting = queue.enqueue("image", "storyboard", "shot 2", batch="b_1")
        stopped: list[str] = []
        self.assertFalse(queue.cancel_studio("image", lambda: stopped.append("stop") or True))
        self.assertEqual(stopped, [])
        self.assertFalse(waiting.cancelled)
        queue.release(running)
        queue.release(waiting)



class ProgressRouteQueueTests(unittest.TestCase):
    def test_progress_routes_report_queue_state(self):
        from fastapi.testclient import TestClient

        from backend_service.app import create_app
        from backend_service.helpers.job_queue import HEAVY_JOBS

        client = TestClient(create_app(api_token="t"), headers={"Authorization": "Bearer t"})
        holder = HEAVY_JOBS.enqueue("video", "agent", "clip")
        HEAVY_JOBS.wait_turn(holder)
        waiting = HEAVY_JOBS.enqueue("image", "ui", "poster")
        try:
            body = client.get("/api/images/progress").json()["progress"]
            self.assertEqual(body["queue"], {"waiting": True, "position": 1, "agentAhead": True})
            self.assertFalse(client.get("/api/video/progress").json()["progress"]["queue"]["waiting"])
            self.assertTrue(client.post("/api/images/cancel").json()["cancelled"])
        finally:
            HEAVY_JOBS.release(waiting)
            HEAVY_JOBS.release(holder)


if __name__ == "__main__":
    unittest.main()
