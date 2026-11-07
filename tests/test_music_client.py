"""The music client against the real worker and the fake engine: events, status while busy, timeouts,
cooperative and hard cancel, restart after a kill, idle stop, the generate timeout and the stall watchdog."""

from __future__ import annotations

import threading
import time
import unittest

from backend_service import music_client
from tests.music_fakes import MusicSandbox


def _generate(worker: music_client.MusicWorker, box: MusicSandbox, job: str, caption: str = "x", **kwargs):
    return worker.request(
        "generate", timeout=kwargs.pop("timeout", 30.0), job=job, engine="acestep", model="acestep-v15-turbo",
        task="text2music", caption=caption, lyrics="[Instrumental]", instrumental=True, seconds=30, seed=1,
        variations=1, steps=kwargs.pop("steps", 4), out_dir=str(box.data / "tmp" / job), **kwargs,
    )


class MusicClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.box = MusicSandbox().__enter__()
        self.worker = music_client.MusicWorker()

    def tearDown(self) -> None:
        self.worker.stop()
        self.box.__exit__(None, None, None)

    def test_generate_with_events_and_reuse(self) -> None:
        events: list[dict] = []
        reply = _generate(self.worker, self.box, "mus_1", on_event=events.append)
        self.assertTrue(reply["ok"])
        self.assertEqual([e["step"] for e in events if e["event"] == "step"], [1, 2, 3, 4])
        pid = self.worker.pid
        self.assertTrue(_generate(self.worker, self.box, "mus_2")["ok"])
        self.assertEqual(self.worker.pid, pid)

    def test_status_answers_while_a_job_runs(self) -> None:
        result: dict = {}
        thread = threading.Thread(target=lambda: result.update(_generate(self.worker, self.box, "mus_3", "delay=0.2", steps=10)))
        thread.start()
        time.sleep(0.8)
        status = self.worker.status()
        thread.join(timeout=30)
        self.assertEqual(status["busy"], "mus_3")
        self.assertTrue(result["ok"])

    def test_status_never_starts_the_worker(self) -> None:
        self.assertIsNone(self.worker.status())
        self.assertIsNone(self.worker.pid)

    def test_errors_are_raised_and_the_worker_survives(self) -> None:
        with self.assertRaisesRegex(music_client.MusicWorkerError, "boom"):
            _generate(self.worker, self.box, "mus_4", "fail")
        self.assertTrue(_generate(self.worker, self.box, "mus_5")["ok"])

    def test_cooperative_cancel(self) -> None:
        flag = threading.Event()
        threading.Timer(0.5, flag.set).start()
        started = time.monotonic()
        with self.assertRaises(music_client.MusicCancelled):
            _generate(self.worker, self.box, "mus_6", "delay=0.2", steps=40, cancel_check=flag.is_set)
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertIsNotNone(self.worker.pid)  # still running: nothing was killed

    def test_hard_cancel_kills_a_job_that_ignores_the_flag(self) -> None:
        flag = threading.Event()
        threading.Timer(0.3, flag.set).start()
        pid_before = None
        started = time.monotonic()
        with self.assertRaisesRegex(music_client.MusicCancelled, "stopped"):
            _generate(self.worker, self.box, "mus_7", "stubborn", cancel_check=lambda: flag.is_set())
        elapsed = time.monotonic() - started
        self.assertGreater(elapsed, music_client.HARD_CANCEL_AFTER_S)
        self.assertLess(elapsed, music_client.HARD_CANCEL_AFTER_S + 6)
        self.assertIsNone(self.worker.pid)
        # The next request starts a fresh worker.
        self.assertTrue(_generate(self.worker, self.box, "mus_8")["ok"])
        self.assertNotEqual(self.worker.pid, pid_before)

    def test_a_timeout_kills_the_worker(self) -> None:
        with self.assertRaisesRegex(music_client.MusicWorkerError, "did not answer"):
            _generate(self.worker, self.box, "mus_9", "stubborn", timeout=1.0)
        self.assertIsNone(self.worker.pid)

    def test_not_installed_is_reported(self) -> None:
        self.box.installed = False
        with self.assertRaisesRegex(music_client.MusicWorkerError, "not installed"):
            self.worker.request("ping")

    def test_idle_expiry_ignores_status_polls(self) -> None:
        self.worker.request("ping")
        self.worker.idle_minutes = 1.0
        last = self.worker._last_used
        self.worker.status()
        self.assertEqual(self.worker._last_used, last)
        self.assertFalse(self.worker.idle_expired(now=last + 30))
        self.assertTrue(self.worker.idle_expired(now=last + 61))
        self.worker.idle_minutes = 0
        self.assertFalse(self.worker.idle_expired(now=last + 10_000))

    def test_stop_if_idle(self) -> None:
        self.assertFalse(self.worker.stop_if_idle())
        self.worker.request("ping")
        self.assertTrue(self.worker.stop_if_idle())
        self.assertIsNone(self.worker.pid)

    def test_generate_timeout_scales_with_the_model_estimate(self) -> None:
        from backend_service.music_runtime import estimate_run

        for model in ("ACE-Step/Ace-Step1.5", "ACE-Step/acestep-v15-xl-turbo"):
            for seconds, variations in ((60, 1), (600, 1), (600, 4)):
                warm = estimate_run(model, seconds, variations)["seconds"]
                cold = estimate_run(model, seconds, variations, warm=False)["coldStartSeconds"]
                timeout = music_client.generate_timeout_s(seconds, variations, model)
                self.assertGreaterEqual(timeout, 3 * warm + cold, (model, seconds, variations))
        # XL is slower than turbo, so it gets longer: the old 210 s for XL 600 s was a 1.27x margin.
        self.assertGreater(music_client.generate_timeout_s(600, 1, "ACE-Step/acestep-v15-xl-turbo"),
                           music_client.generate_timeout_s(600, 1, "ACE-Step/Ace-Step1.5"))
        self.assertGreaterEqual(music_client.generate_timeout_s(600, 1, "ACE-Step/acestep-v15-xl-turbo"), 3 * 165.0)

    def test_stall_watchdog_kills_a_job_with_no_events(self) -> None:
        started = time.monotonic()
        with self.assertRaisesRegex(music_client.MusicWorkerError, "no progress"):
            _generate(self.worker, self.box, "mus_10", "stubborn", timeout=60.0,
                      stall_limits={"decode": 1.0, "default": 30.0})
        self.assertLess(time.monotonic() - started, 10.0)
        self.assertIsNone(self.worker.pid)

    def test_stall_watchdog_leaves_a_slow_job_that_keeps_reporting(self) -> None:
        reply = _generate(self.worker, self.box, "mus_11", "delay=0.3", steps=8, timeout=60.0,
                          stall_limits={"default": 1.5, "decode": 1.5})
        self.assertTrue(reply["ok"])

    def test_stall_limits_are_generous_for_decode(self) -> None:
        limits = music_client.stall_limits(600, 4, "ACE-Step/acestep-v15-xl-turbo")
        self.assertGreaterEqual(limits["decode"], 3 * 0.08 * 600 * 4)
        self.assertGreater(limits["decode"], limits["default"])
        self.assertGreaterEqual(limits["load"], 30.0 * 3)


if __name__ == "__main__":
    unittest.main()
