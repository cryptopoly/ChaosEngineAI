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

    def test_slot_released_after_error(self):
        queue = HeavyJobQueue()
        with self.assertRaises(ValueError):
            with queue.slot("voice", "ui"):
                raise ValueError("boom")
        self.assertIsNone(queue.snapshot()["running"])



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
