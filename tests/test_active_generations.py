"""The workspace payload says which image, video or music run is in flight (FU-180 follow-up, plan S4)."""

import unittest

from backend_service.progress import IMAGE_PROGRESS, MUSIC_PROGRESS, VIDEO_PROGRESS
from backend_service.state.payloads import active_generations


class ActiveGenerationsTests(unittest.TestCase):
    def tearDown(self):
        for tracker in (IMAGE_PROGRESS, VIDEO_PROGRESS, MUSIC_PROGRESS):
            tracker.finish()

    def test_nothing_running_is_an_empty_list(self):
        self.assertEqual(active_generations(), [])

    def test_a_running_still_reports_label_and_step(self):
        IMAGE_PROGRESS.begin(run_label="Qwen-Image 2.1", total_steps=40)
        IMAGE_PROGRESS.set_phase("diffusing", "Denoising")
        IMAGE_PROGRESS.set_step(27, 40)
        (run,) = active_generations()
        self.assertEqual((run["kind"], run["runLabel"], run["phase"]), ("image", "Qwen-Image 2.1", "diffusing"))
        self.assertEqual((run["step"], run["totalSteps"]), (27, 40))
        self.assertNotIn("thumbnail", run)

    def test_image_and_video_both_listed_and_gone_when_finished(self):
        IMAGE_PROGRESS.begin(run_label="a")
        VIDEO_PROGRESS.begin(run_label="b")
        self.assertEqual([r["kind"] for r in active_generations()], ["image", "video"])
        IMAGE_PROGRESS.finish()
        self.assertEqual([r["kind"] for r in active_generations()], ["video"])

    def test_the_workspace_payload_carries_it_under_system(self):
        import tempfile
        from pathlib import Path

        from backend_service.state import ChaosEngineState
        from tests.test_backend_service import FakeRuntime, fake_library, fake_system_snapshot, make_test_client

        with tempfile.TemporaryDirectory() as tmp:
            state = ChaosEngineState(
                system_snapshot_provider=fake_system_snapshot,
                library_provider=fake_library,
                settings_path=Path(tmp) / "settings.json",
                benchmarks_path=Path(tmp) / "benchmarks.json",
                chat_sessions_path=Path(tmp) / "chat.json",
            )
            state.runtime = FakeRuntime()
            client = make_test_client(state)
            self.assertEqual(client.get("/api/workspace").json()["system"]["activeGenerations"], [])
            IMAGE_PROGRESS.begin(run_label="Iris-3B Turbo", total_steps=8)
            runs = client.get("/api/workspace").json()["system"]["activeGenerations"]
            self.assertEqual([r["runLabel"] for r in runs], ["Iris-3B Turbo"])


if __name__ == "__main__":
    unittest.main()
