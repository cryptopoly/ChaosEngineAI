import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers.busy_advice import busy_advice, current_advice, describe_job
from backend_service.state import ChaosEngineState
from tests.test_backend_service import FakeRuntime, fake_library, fake_system_snapshot, make_test_client

VIDEO = {"kind": "video", "label": "Harbour: s2 clip", "source": "storyboard"}
IMAGE = {"kind": "image", "label": "load Flux", "source": "ui"}


class BusyAdviceTableTests(unittest.TestCase):
    def test_table(self):
        # (jobs, footprint, total, available, expected level)
        rows = [
            ([], None, 64, 40, "ok"),
            ([], 0, 64, 10, "ok"),
            ([], 8, 64, 40, "ok"),
            ([VIDEO], None, 64, 30, "slower"),
            ([VIDEO], 0, 64, 20, "slower"),
            ([IMAGE], 4, 64, 30, "slower"),
            ([VIDEO], 20, 64, 20, "risky"),  # 44 in use + 20 > 57.6
            ([], 30, 64, 20, "risky"),  # no job, but the model will not fit comfortably
            ([VIDEO, IMAGE], None, 16, 8, "risky"),  # peaks 36 > 14.4
        ]
        for jobs, footprint, total, available, level in rows:
            with self.subTest(jobs=[j["kind"] for j in jobs], footprint=footprint, available=available):
                self.assertEqual(busy_advice(jobs, footprint, total, available)["level"], level)

    def test_messages_name_the_job_and_the_numbers(self):
        slower = busy_advice([VIDEO], None, 64, 30)
        self.assertIn('video job "Harbour: s2 clip" (Storyboard)', slower["message"])
        self.assertIn("slower", slower["message"])
        self.assertIn("34.0 GB of 64.0 GB", slower["message"])  # 64 - 30 in use, above the 24 GB default peak
        risky = busy_advice([VIDEO], 20, 64, 20)
        self.assertIn("Estimated peak 64.0 GB of 64.0 GB", risky["message"])
        self.assertIn("20.0 GB for this model", risky["message"])
        self.assertIn("Wait for it to finish", risky["message"])
        self.assertEqual(risky["jobs"], [VIDEO])

    def test_ok_has_no_message_and_never_refuses(self):
        ok = busy_advice([], None, 64, 50)
        self.assertEqual((ok["level"], ok["message"]), ("ok", ""))
        # Even at no free memory it only advises; refusal belongs to memory_gate.
        self.assertEqual(busy_advice([VIDEO], 100, 64, 0)["level"], "risky")

    def test_job_peak_figure_wins_over_default(self):
        job = {**VIDEO, "peakGb": 50}
        self.assertEqual(busy_advice([job], 10, 64, 60)["combinedPeakGb"], 60.0)

    def test_unknown_total_does_not_cry_wolf(self):
        self.assertEqual(busy_advice([], 10, 0, 0)["level"], "ok")

    def test_describe_job_without_label(self):
        self.assertEqual(describe_job({"kind": "music", "source": "agent"}), "music job (agent)")

    def test_current_advice_reads_queue_read_only(self):
        snapshot = {"running": {"id": 1, "kind": "video", "source": "storyboard", "label": "L"}, "waiting": []}
        with mock.patch("backend_service.helpers.job_queue.HEAVY_JOBS.snapshot", return_value=snapshot):
            advice = current_advice(0)
        self.assertIsNotNone(advice)
        self.assertEqual(advice["jobs"][0]["kind"], "video")
        self.assertIn(advice["level"], ("slower", "risky"))


class BusyAdviceRouteTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=root / "settings.json",
            benchmarks_path=root / "benchmark-history.json",
            chat_sessions_path=root / "chat-sessions.json",
        )
        state.runtime = FakeRuntime()
        self.client = make_test_client(state)

    def tearDown(self):
        self.tempdir.cleanup()

    def _advice(self, level="slower"):
        return {"level": level, "message": "m", "jobs": [], "combinedPeakGb": 1.0, "totalGb": 2.0,
                "availableGb": 1.0, "footprintGb": 0.0}

    def test_busy_advice_route_shape(self):
        with mock.patch("backend_service.helpers.busy_advice.current_advice", return_value=self._advice()) as cur:
            body = self.client.get("/api/busy-advice", params={"footprintGb": 12.5}).json()
        cur.assert_called_once_with(12.5)
        self.assertEqual(body["advice"]["level"], "slower")
        self.assertEqual(self.client.get("/api/busy-advice", params={"footprintGb": -1}).status_code, 422)

    def test_model_load_response_carries_advice(self):
        with mock.patch("backend_service.helpers.busy_advice.current_advice", return_value=self._advice("risky")):
            response = self.client.post(
                "/api/models/load",
                json={"modelRef": "google/gemma-4-E4B-it", "modelName": "Gemma", "source": "catalog", "backend": "mock"},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("runtime", data)
        self.assertEqual(data["advice"]["level"], "risky")

    def test_chat_stream_sends_advice_event_only_when_not_ok(self):
        self.client.post("/api/models/load", json={"modelRef": "google/gemma-4-E4B-it", "modelName": "Gemma", "source": "catalog", "backend": "mock"})
        session = self.client.post("/api/chat/sessions", json={"title": "t"}).json()["session"]
        for level, expected in (("slower", True), ("ok", False)):
            with self.subTest(level=level), mock.patch("backend_service.helpers.busy_advice.current_advice", return_value=self._advice(level)):
                response = self.client.post("/api/chat/generate/stream", json={"sessionId": session["id"], "prompt": "hi"})
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ") and line[6:].startswith("{")]
                self.assertEqual(any("advice" in e for e in events), expected)
