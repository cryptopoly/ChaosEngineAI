"""Cancel for a speech-model download (FU-157): the job stops, the row resets, Delete is no longer refused."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_service import voice_clone, voice_env, voice_model_files

KEY = "qwen3-tts-0.6b"  # a cloning model, so a download also has the watermark step


def _sleeper(*_args, **_kwargs) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def _wait(predicate, seconds: float = 5.0) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class CancelDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        cache = Path(self.tmp.name) / "hub"
        cache.mkdir()
        self.env = mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(cache), "HUGGINGFACE_HUB_CACHE": str(cache)})
        self.env.start()
        self.downloads = mock.patch.dict(voice_clone._DOWNLOADS, {}, clear=True)
        self.downloads.start()
        self.avail = mock.patch.object(voice_clone, "available_here", return_value=True)
        self.avail.start()

    def tearDown(self) -> None:
        control = voice_clone._CONTROL.pop(KEY, None)
        if control:
            control["cancel"].set()
            if control.get("proc"):
                control["proc"].kill()
        self.avail.stop()
        self.downloads.stop()
        self.env.stop()
        self.tmp.cleanup()

    def _thread_done(self) -> bool:
        return not any(t.name == f"clone-download-{KEY}" for t in threading.enumerate())

    def test_cancel_stops_the_weights_download_and_resets_the_row(self) -> None:
        with mock.patch("backend_service.state._helpers._spawn_snapshot_download", _sleeper):
            voice_clone.start_download(KEY)
            self.assertTrue(_wait(lambda: voice_clone._CONTROL[KEY]["proc"] is not None))
            process = voice_clone._CONTROL[KEY]["proc"]
            job = voice_clone.cancel_download(KEY)
            self.assertEqual(job["state"], "cancelled")
            self.assertNotIn("token", job)
            self.assertTrue(_wait(lambda: process.poll() is not None), "the child process was not stopped")
            self.assertTrue(_wait(self._thread_done))
        self.assertEqual(voice_clone._DOWNLOADS[KEY]["state"], "cancelled")
        row = next(r for r in voice_clone.list_models() if r["key"] == KEY)
        self.assertEqual(row["download"]["state"], "cancelled")
        self.assertFalse(row["installed"])

    def test_cancel_during_the_watermark_step(self) -> None:
        entered = threading.Event()

        def install(cancel):
            entered.set()
            cancel.wait(5)
            raise InterruptedError("cancelled")

        with mock.patch.object(voice_clone, "_fetch_repo"), \
                mock.patch.object(voice_env, "watermark_installed", return_value=False), \
                mock.patch.object(voice_env, "install_watermark", install):
            voice_clone.start_download(KEY)
            self.assertTrue(entered.wait(5))
            self.assertEqual(voice_clone._DOWNLOADS[KEY]["phase"], "watermark")
            voice_clone.cancel_download(KEY)
            self.assertTrue(_wait(self._thread_done))
        self.assertEqual(voice_clone._DOWNLOADS[KEY]["state"], "cancelled")

    def test_a_late_failure_does_not_overwrite_a_cancel(self) -> None:
        release = threading.Event()

        def slow_fail(*_args):
            release.wait(5)
            raise RuntimeError("network dropped")

        with mock.patch.object(voice_clone, "_fetch_repo", slow_fail):
            voice_clone.start_download(KEY)
            voice_clone.cancel_download(KEY)
            release.set()
            self.assertTrue(_wait(self._thread_done))
        self.assertEqual(voice_clone._DOWNLOADS[KEY]["state"], "cancelled")

    def test_cancel_when_nothing_downloads_changes_nothing(self) -> None:
        self.assertEqual(voice_clone.cancel_download(KEY), {})
        voice_clone._DOWNLOADS[KEY] = {"state": "done", "error": None}
        self.assertEqual(voice_clone.cancel_download(KEY)["state"], "done")
        with self.assertRaises(ValueError):
            voice_clone.cancel_download("not-a-model")

    def test_a_cancelled_download_can_start_again(self) -> None:
        with mock.patch("backend_service.state._helpers._spawn_snapshot_download", _sleeper):
            voice_clone.start_download(KEY)
            self.assertTrue(_wait(lambda: voice_clone._CONTROL[KEY]["proc"] is not None))
            voice_clone.cancel_download(KEY)
            self.assertTrue(_wait(self._thread_done))
            again = voice_clone.start_download(KEY)
            self.assertEqual(again["state"], "downloading")
            voice_clone.cancel_download(KEY)
            self.assertTrue(_wait(self._thread_done))

    def test_delete_is_not_refused_after_a_cancel(self) -> None:
        with mock.patch("backend_service.state._helpers._spawn_snapshot_download", _sleeper):
            voice_clone.start_download(KEY)
            self.assertTrue(_wait(lambda: voice_clone._CONTROL[KEY]["proc"] is not None))
            with mock.patch.object(voice_model_files, "voice_busy_reason", return_value=None):
                with self.assertRaises(voice_model_files.ModelInUse):
                    voice_model_files.remove_speech_model(KEY)
                voice_clone.cancel_download(KEY)
                self.assertTrue(_wait(self._thread_done))
                result = voice_model_files.remove_speech_model(KEY)
        self.assertTrue(result["deleted"])
        self.assertNotIn(KEY, voice_clone._DOWNLOADS)

    def test_a_downloading_row_reports_progress(self) -> None:
        voice_clone._DOWNLOADS[KEY] = {"state": "downloading", "error": None, "token": "t"}
        row = next(r for r in voice_clone.list_models() if r["key"] == KEY)
        download = row["download"]
        self.assertEqual(download["state"], "downloading")
        for field in ("progress", "downloadedGb", "totalGb"):
            self.assertIn(field, download)
        self.assertNotIn("token", download)


class CancelRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_video_routes import make_client

        self.client, self._tmp, self._snapshot = make_client()

    def tearDown(self) -> None:
        from tests.test_video_routes import restore_env

        restore_env(self._snapshot)
        self._tmp.cleanup()

    def test_agents_and_other_machines_get_403(self) -> None:
        from backend_service import voice_clone_safety

        for origin in ("agent", "remote"):
            with mock.patch.object(voice_clone_safety, "request_origin", return_value=origin), \
                    mock.patch.object(voice_clone, "cancel_download") as cancel:
                response = self.client.post("/api/voice/clone/models/download/cancel", json={"key": KEY})
            self.assertEqual(response.status_code, 403, origin)
            cancel.assert_not_called()

    def test_the_window_cancels_and_the_state_resets(self) -> None:
        with mock.patch.dict(voice_clone._DOWNLOADS, {KEY: {"state": "downloading", "error": None, "token": "t"}}, clear=True):
            response = self.client.post("/api/voice/clone/models/download/cancel", json={"key": KEY})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["download"]["state"], "cancelled")
            self.assertEqual(voice_clone._DOWNLOADS[KEY]["state"], "cancelled")

    def test_unknown_key_is_404(self) -> None:
        response = self.client.post("/api/voice/clone/models/download/cancel", json={"key": "nope"})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
