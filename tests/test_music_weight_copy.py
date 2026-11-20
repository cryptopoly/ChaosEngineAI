"""The "Copy now" job for music weights (D-100, FU-276) with symlink and hard link forced to fail.

Covers: the machine-readable 409 fields, a completed copy with its progress numbers, a cancel and the
resume after it (finished files are not copied again), the free-space refusal, the load being held while a
copy runs, and the three routes.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_service import music_env, music_runtime, music_weight_copy
from tests.test_video_routes import make_client, restore_env  # imports the app before music_variants is patched

BIG = music_env._LINK_MIN_BYTES  # smallest size that is linked or copied rather than copied as a small file
NO_LINK = OSError(1314, "A required privilege is not held by the client")
NO_HARD_LINK = OSError(18, "Invalid cross-device link")


class _Box(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data = root / "music"
        self.snapshot = root / "hub" / "snap"
        self.snapshot.mkdir(parents=True)
        for name, fill in (("a.safetensors", b"a"), ("b.safetensors", b"b")):
            (self.snapshot / name).write_bytes(fill * BIG)
        (self.snapshot / "config.json").write_text("cfg")
        self.variant = {"repo": "ACE-Step/test", "revision": "r1", "checkpointSubdir": "dit", "weightFolders": ["dit"]}
        self.target = self.data / "checkpoints" / "dit"
        patches = [
            mock.patch.object(music_env, "MUSIC_DATA_DIR", self.data),
            mock.patch.object(music_env, "hf_snapshot", return_value=self.snapshot),
            mock.patch.object(music_env.os, "symlink", side_effect=NO_LINK),
            mock.patch.object(music_env.os, "link", side_effect=NO_HARD_LINK),
            mock.patch("backend_service.catalog.music_models.music_variants", return_value=[self.variant]),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        music_weight_copy._CANCEL.clear()
        music_weight_copy._reset()
        self.addCleanup(music_weight_copy._reset)

    def tearDown(self) -> None:
        self.wait()
        self.tmp.cleanup()

    def wait(self) -> dict:
        deadline = time.time() + 15
        while music_weight_copy.is_running() and time.time() < deadline:
            time.sleep(0.02)
        return music_weight_copy.status()


class CopyJobTests(_Box):
    def test_the_409_carries_the_machine_readable_fields(self) -> None:
        with mock.patch.object(music_runtime, "readiness_error", return_value=None):
            with self.assertRaises(music_runtime.MusicNotReady) as caught:
                music_runtime._prepare({"name": "test", "engine": "ace-step"})
        fields = caught.exception.fields
        self.assertTrue(fields["needsCopy"])
        self.assertEqual((fields["fileCount"], fields["totalBytes"]), (2, 2 * BIG))
        self.assertEqual(Path(fields["source"]), self.snapshot)
        self.assertEqual(Path(fields["target"]), self.data / "checkpoints")

    def test_copy_completes_with_progress_numbers_and_the_load_then_passes(self) -> None:
        done: list[dict] = []
        started = music_weight_copy.start(done.append)
        self.assertEqual((started["fileCount"], started["totalBytes"]), (2, 2 * BIG))
        final = self.wait()
        self.assertEqual(final["phase"], "done", final)
        self.assertEqual((final["bytesDone"], final["filesDone"]), (2 * BIG, 2))
        for name in ("a.safetensors", "b.safetensors"):
            copy = self.target / name
            self.assertTrue(copy.is_file() and not copy.is_symlink())
            self.assertEqual(copy.stat().st_size, BIG)
            self.assertFalse(copy.with_name(name + ".partial").exists())
        self.assertEqual(done[0]["fileCount"], 2)
        music_env.prepare_checkpoints([self.variant])  # nothing left to copy, so no notice
        self.assertEqual(music_weight_copy.start()["fileCount"], 0)

    def test_cancel_stops_cleanly_and_the_retry_copies_only_what_is_left(self) -> None:
        real_copy, calls = music_weight_copy._copy_file, []

        def cancel_during_second(real: Path, target: Path, on_bytes) -> None:
            calls.append(target.name)
            if len(calls) == 2:
                def hook(n: int) -> None:
                    on_bytes(n)
                    music_weight_copy._CANCEL.set()
                return real_copy(real, target, hook)
            return real_copy(real, target, on_bytes)

        with mock.patch.object(music_weight_copy, "_copy_file", cancel_during_second):
            music_weight_copy.start()
            cancelled = self.wait()
        self.assertEqual(cancelled["phase"], "cancelled")
        self.assertTrue((self.target / "a.safetensors").exists())
        self.assertFalse((self.target / "b.safetensors").exists())
        self.assertFalse((self.target / "b.safetensors.partial").exists(), "a cancelled partial is removed")
        first_stat = (self.target / "a.safetensors").stat().st_mtime_ns

        calls.clear()
        with mock.patch.object(music_weight_copy, "_copy_file", wraps=real_copy) as spy:
            retry = music_weight_copy.start()
            self.assertEqual((retry["fileCount"], retry["totalBytes"]), (1, BIG))
            final = self.wait()
        self.assertEqual(final["phase"], "done", final)
        self.assertEqual([c.args[1].name for c in spy.call_args_list], ["b.safetensors"])
        self.assertEqual((self.target / "a.safetensors").stat().st_mtime_ns, first_stat)

    def test_a_stale_partial_file_is_started_again(self) -> None:
        self.target.mkdir(parents=True)
        (self.target / "a.safetensors.partial").write_bytes(b"junk")
        music_weight_copy.start()
        self.assertEqual(self.wait()["phase"], "done")
        self.assertEqual((self.target / "a.safetensors").read_bytes()[:4], b"aaaa")
        self.assertFalse((self.target / "a.safetensors.partial").exists())

    def test_not_enough_space_refuses_before_copying_anything(self) -> None:
        usage = mock.Mock(free=BIG)  # the files need 2 x BIG plus the margin
        with mock.patch.object(music_weight_copy.shutil, "disk_usage", return_value=usage):
            with self.assertRaises(music_weight_copy.WeightCopyError) as caught:
                music_weight_copy.start()
        self.assertIn("Not enough free space", str(caught.exception))
        self.assertIn("GB free", str(caught.exception))
        self.assertFalse(music_weight_copy.is_running())
        self.assertFalse((self.target / "a.safetensors").exists())

    def test_required_space_has_a_margin(self) -> None:
        pending = [(self.snapshot / "a.safetensors", self.target / "a.safetensors")]
        self.assertEqual(music_weight_copy.required_bytes(pending), BIG + 1024 ** 3)

    def test_the_load_waits_while_a_copy_runs(self) -> None:
        with mock.patch.object(music_weight_copy, "is_running", return_value=True), \
                mock.patch.object(music_runtime, "readiness_error", return_value=None):
            with self.assertRaises(music_runtime.MusicNotReady) as caught:
                music_runtime._prepare({"name": "test", "engine": "ace-step"})
        self.assertIn("still being copied", str(caught.exception))

    def test_a_failing_source_ends_in_a_plain_error(self) -> None:
        with mock.patch.object(music_weight_copy, "_copy_file", side_effect=OSError("disk gone")):
            music_weight_copy.start()
            final = self.wait()
        self.assertEqual(final["phase"], "error")
        self.assertIn("disk gone", final["error"])


class CopyRouteTests(_Box):
    def setUp(self) -> None:
        super().setUp()
        self.client, self._tmpdir, self._snap = make_client()

    def tearDown(self) -> None:
        self.wait()
        restore_env(self._snap)
        self._tmpdir.cleanup()
        super().tearDown()

    def test_start_status_and_activity(self) -> None:
        response = self.client.post("/api/music/weights/copy")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["fileCount"], 2)
        self.wait()
        status = self.client.get("/api/music/weights/copy/status").json()
        self.assertEqual((status["phase"], status["bytesDone"], status["totalBytes"]), ("done", 2 * BIG, 2 * BIG))
        titles = [entry["title"] for entry in self.client.app.state.chaosengine.activity]
        self.assertIn("Music model files copied", titles)

    def test_space_refusal_is_a_409_with_a_plain_message(self) -> None:
        with mock.patch.object(music_weight_copy.shutil, "disk_usage", return_value=mock.Mock(free=1)):
            response = self.client.post("/api/music/weights/copy")
        self.assertEqual(response.status_code, 409)
        self.assertIn("Not enough free space", response.json()["detail"])

    def test_cancel_route_reports_status(self) -> None:
        body = self.client.post("/api/music/weights/copy/cancel").json()
        self.assertIn("phase", body)

    def test_preload_409_has_the_envelope_with_the_copy_fields(self) -> None:
        variant = {"id": "test-model", "name": "Test", "engine": "ace-step"}
        with mock.patch.object(music_runtime, "readiness_error", return_value=None),                 mock.patch.object(music_runtime, "resolve_variant", return_value=variant):
            response = self.client.post("/api/music/preload", json={"modelId": "test-model"})
        self.assertEqual(response.status_code, 409, response.text)
        detail = response.json()["detail"]
        self.assertTrue(detail["needsCopy"])
        self.assertEqual(detail["fileCount"], 2)
        self.assertIn("copy", detail["message"])


if __name__ == "__main__":
    unittest.main()
