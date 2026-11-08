"""Agent job records: restart safety, list_jobs, running voice cancel, the per-token cap, the queue view."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.mcp_bridge import access, jobs
from backend_service.mcp_bridge.jobs import JobLimitError, JobStore


def _wait(job, seconds=5):
    job.done.wait(seconds)


class JobRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "agent_jobs"
        patcher = mock.patch.object(jobs, "jobs_dir", lambda: self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_finished_job_is_written_and_survives_a_restart(self):
        store = JobStore()
        job = store.submit("voice", "speech", lambda: ({"text": "hi"}, ["note"]))
        _wait(job)
        self.assertEqual(json.loads((self.dir / f"{job.id}.json").read_text())["state"], "done")
        again = JobStore()
        loaded = again.get(job.id)
        self.assertEqual(loaded.state, "done")
        self.assertEqual(loaded.output, {"text": "hi"})
        self.assertTrue(loaded.done.is_set())
        # new ids continue after the old ones
        nxt = again.submit("voice", "x", lambda: ({}, []))
        self.assertNotEqual(nxt.id, job.id)
        self.assertGreater(int(nxt.id.split("-")[1]), int(job.id.split("-")[1]))
        _wait(nxt)

    def test_running_job_reads_interrupted_after_restart(self):
        self.dir.mkdir(parents=True)
        for jid, state in (("job-7", "running"), ("job-8", "queued")):
            (self.dir / f"{jid}.json").write_text(json.dumps({
                "id": jid, "kind": "video", "label": "clip", "state": state, "owner": "x", "output": None,
                "error": None, "notes": [], "createdAt": time.time(), "finishedAt": None}))
        store = JobStore()
        for jid in ("job-7", "job-8"):
            job = store.get(jid)
            self.assertEqual(job.state, "interrupted")
            self.assertIn("restarted", job.describe()["error"])
            self.assertTrue(job.done.is_set())
        self.assertEqual(json.loads((self.dir / "job-7.json").read_text())["state"], "interrupted")
        fresh = store.submit("voice", "n", lambda: ({}, []))
        self.assertEqual(fresh.id, "job-9")
        _wait(fresh)

    def test_records_older_than_seven_days_are_dropped(self):
        self.dir.mkdir(parents=True)
        old = time.time() - 8 * 24 * 3600
        (self.dir / "job-3.json").write_text(json.dumps({
            "id": "job-3", "kind": "image", "label": "a", "state": "done", "owner": "x", "output": {},
            "error": None, "notes": [], "createdAt": old, "finishedAt": old}))
        self.assertIsNone(JobStore().get("job-3"))
        self.assertFalse((self.dir / "job-3.json").exists())

    def test_list_filters_and_orders_newest_first(self):
        store = JobStore()
        first = store.submit("voice", "a", lambda: ({}, []))
        _wait(first)
        second = store.submit("voice", "b", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        _wait(second)
        self.assertEqual([j.id for j in store.list()], [second.id, first.id])
        self.assertEqual([j.id for j in store.list("failed")], [second.id])
        self.assertEqual(len(store.list(limit=1)), 1)

    def test_cap_refuses_the_21st_unfinished_job_per_token(self):
        store = JobStore()
        gate = threading.Event()
        started = []
        token = jobs.CURRENT_OWNER.set("tokenA")
        try:
            for _ in range(jobs.MAX_UNFINISHED_PER_TOKEN):
                started.append(store.submit("storyboard", "s", lambda: (gate.wait(5), ({}, []))[1], cancel=lambda: True))
            with self.assertRaises(JobLimitError) as ctx:
                store.submit("storyboard", "s", lambda: ({}, []), cancel=lambda: True)
            self.assertIn("20 unfinished", str(ctx.exception))
            jobs.CURRENT_OWNER.set("tokenB")  # another token is not affected
            other = store.submit("storyboard", "s", lambda: ({}, []), cancel=lambda: True)
            _wait(other)
        finally:
            gate.set()
            jobs.CURRENT_OWNER.reset(token)
        for job in started:
            _wait(job)
        jobs.CURRENT_OWNER.set("tokenA")
        try:
            store.submit("storyboard", "s", lambda: ({}, []), cancel=lambda: True)  # room again
        finally:
            jobs.CURRENT_OWNER.set("app")

    def test_running_voice_job_can_be_cancelled(self):
        store = JobStore()
        release = threading.Event()
        killed = []

        class Worker:
            def _kill(self):
                killed.append(True)
                release.set()

        def run():
            release.wait(5)
            if killed:
                raise RuntimeError("The voice worker exited.")
            return {}, []

        with mock.patch("backend_service.voice_client._WORKER", Worker()):
            job = store.submit("voice", "speech", run)
            for _ in range(100):
                if job.state == "running":
                    break
                time.sleep(0.02)
            self.assertEqual(job.state, "running")
            self.assertTrue(store.cancel(job))
            _wait(job)
        self.assertEqual(job.state, "cancelled")
        self.assertEqual(killed, [True])
        self.assertFalse(store.cancel(job))

    def test_queue_view_has_positions_and_labelled_estimates(self):
        store = JobStore()
        gate = threading.Event()
        made = [store.submit("image", f"m{i}", lambda: (gate.wait(5), ({}, []))[1], estimate_seconds=30 * (i + 1))
                for i in range(3)]
        try:
            for _ in range(100):
                if made[0].state == "running":
                    break
                time.sleep(0.02)
            with mock.patch.object(jobs, "STORE", store):
                view = store.queue_view()
            rows = view["jobs"]
            self.assertEqual([r["position"] for r in rows], [0, 1, 2])
            self.assertEqual([r["label"] for r in rows], ["m0", "m1", "m2"])
            self.assertEqual([r["estimatedSeconds"] for r in rows], [30, 60, 90])
            self.assertIsNone(rows[0]["estimatedStartInSeconds"])
            self.assertGreater(rows[1]["estimatedStartInSeconds"], 20)
            self.assertGreater(rows[2]["estimatedStartInSeconds"], rows[1]["estimatedStartInSeconds"])
            self.assertIn("estimated", view["estimateNote"])
        finally:
            gate.set()
            for job in made:
                _wait(job)


class ListJobsToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        for patcher in (
            mock.patch.object(access, "_path", lambda: Path(self.tmp.name) / "agent_access.json"),
            mock.patch.object(jobs, "jobs_dir", lambda: Path(self.tmp.name) / "agent_jobs"),
            mock.patch.object(jobs, "STORE", JobStore()),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        # tools.py bound STORE at import; point it at the fresh store too
        from backend_service.mcp_bridge import tools

        patcher = mock.patch.object(tools, "STORE", jobs.STORE)
        patcher.start()
        self.addCleanup(patcher.stop)
        access.set_enabled(True)
        self.client = TestClient(create_app(api_token="t"))
        self.headers = {"Authorization": f"Bearer {access.get_token()}"}

    def call(self, name, args=None):
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args or {}}}
        result = self.client.post("/mcp", json=body, headers=self.headers).json()["result"]
        return result["isError"], result["content"][0]["text"]

    def test_list_jobs_and_system_status_queue(self):
        from backend_service.mcp_bridge import tools

        job = tools.STORE.submit("voice", "speech", lambda: ({"x": 1}, []))
        _wait(job)
        is_error, text = self.call("list_jobs", {"state": "done"})
        self.assertFalse(is_error)
        self.assertIn(job.id, text)
        is_error, text = self.call("system_status")
        self.assertFalse(is_error)
        self.assertIn('"jobs"', text)
        self.assertIn("estimateNote", text)

    def test_unknown_job_message_points_to_list_jobs(self):
        is_error, text = self.call("job_status", {"job_id": "job-999"})
        self.assertTrue(is_error)
        self.assertIn("list_jobs", text)

    def test_cap_error_through_the_tool(self):
        with mock.patch.object(jobs, "MAX_UNFINISHED_PER_TOKEN", 0):
            is_error, text = self.call("speak", {"text": "hello"})
        self.assertTrue(is_error)
        self.assertIn("unfinished jobs", text)


if __name__ == "__main__":
    unittest.main()
