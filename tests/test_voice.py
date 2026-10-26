"""Voice gallery persistence and the Kokoro voice-file download.

The speech lanes, worker and routes are covered in ``test_voice_io.py``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Pre-import numpy: the kokoro-onnx synth imports it lazily, and
# ``mock.patch.dict(sys.modules)`` would evict it on exit and trip
# numpy's "cannot load module more than once per process" guard.
import numpy  # noqa: F401

from backend_service import voice_gallery, voice_kokoro_onnx



# ---------------------------------------------------------------------------
# voice_runtime unit tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# voice_kokoro_onnx unit tests (non-Apple TTS lane, FU-083)
# ---------------------------------------------------------------------------

class KokoroOnnxTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self._dir_patch = mock.patch.object(
            voice_kokoro_onnx, "kokoro_dir", return_value=Path(self._tempdir.name)
        )
        self._dir_patch.start()
        # ``_status`` is module-global; reset so test order can't leak a
        # "failed"/"downloading" state from one case into the next.
        self._status_patch = mock.patch.dict(
            voice_kokoro_onnx._status, {"state": "idle", "progress": 0.0, "error": None}
        )
        self._status_patch.start()

    def tearDown(self) -> None:
        self._status_patch.stop()
        self._dir_patch.stop()
        self._tempdir.cleanup()

    def _install_files(self) -> None:
        for name, _url in voice_kokoro_onnx._FILES:
            (Path(self._tempdir.name) / name).write_bytes(b"fake-model-data")

    def test_is_installed_requires_both_files(self) -> None:
        self.assertFalse(voice_kokoro_onnx.is_installed())
        (Path(self._tempdir.name) / voice_kokoro_onnx._FILES[0][0]).write_bytes(b"x")
        self.assertFalse(voice_kokoro_onnx.is_installed())
        self._install_files()
        self.assertTrue(voice_kokoro_onnx.is_installed())


    def test_download_worker_writes_files_atomically(self) -> None:
        class FakeResponse:
            def __init__(self) -> None:
                self._chunks = [b"chunk-one", b"chunk-two", b""]

            def read(self, _size: int) -> bytes:
                return self._chunks.pop(0)

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *exc_info: object) -> None:
                return None

        with mock.patch.object(
            voice_kokoro_onnx.urllib.request, "urlopen", side_effect=lambda url, timeout: FakeResponse()
        ):
            voice_kokoro_onnx._download_worker()

        self.assertTrue(voice_kokoro_onnx.is_installed())
        for name, _url in voice_kokoro_onnx._FILES:
            self.assertEqual(
                (Path(self._tempdir.name) / name).read_bytes(), b"chunk-onechunk-two"
            )
        self.assertEqual(voice_kokoro_onnx.download_status()["state"], "completed")

    def test_download_worker_records_failure(self) -> None:
        with mock.patch.object(
            voice_kokoro_onnx.urllib.request, "urlopen", side_effect=OSError("network down")
        ):
            voice_kokoro_onnx._download_worker()
        status = voice_kokoro_onnx.download_status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("network down", status["error"])
        self.assertFalse(voice_kokoro_onnx.is_installed())


# ---------------------------------------------------------------------------
# voice_gallery unit tests
# ---------------------------------------------------------------------------

class GalleryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(
            voice_gallery, "_gallery_dir", return_value=Path(self._tempdir.name)
        )
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tempdir.cleanup()

    def test_transcript_roundtrip(self) -> None:
        meta = voice_gallery.save_transcript("hello world")
        self.assertEqual(meta["kind"], "transcript")
        items = voice_gallery.list_items()
        self.assertEqual([i["id"] for i in items], [meta["id"]])
        self.assertTrue(voice_gallery.delete_item(meta["id"]))
        self.assertEqual(voice_gallery.list_items(), [])

    def test_audio_roundtrip_with_wav_sidecar(self) -> None:
        meta = voice_gallery.save_audio("some text", "af_heart", b"RIFF-bytes")
        audio_path = voice_gallery.get_audio_path(meta["id"])
        self.assertIsNotNone(audio_path)
        self.assertEqual(audio_path.read_bytes(), b"RIFF-bytes")
        self.assertTrue(voice_gallery.delete_item(meta["id"]))
        self.assertIsNone(voice_gallery.get_audio_path(meta["id"]))

    def test_delete_missing_returns_false(self) -> None:
        self.assertFalse(voice_gallery.delete_item("1234567890_deadbeef"))

    def test_id_validation_rejects_traversal(self) -> None:
        self.assertFalse(voice_gallery.is_valid_id("../../etc/passwd"))
        self.assertFalse(voice_gallery.is_valid_id("nope"))
        self.assertTrue(voice_gallery.is_valid_id("1234567890_0123abcd"))

    def test_list_skips_corrupt_sidecars(self) -> None:
        (Path(self._tempdir.name) / "corrupt.json").write_text("{not json", encoding="utf-8")
        good = voice_gallery.save_transcript("ok")
        items = voice_gallery.list_items()
        self.assertEqual([i["id"] for i in items], [good["id"]])


# ---------------------------------------------------------------------------
# Route contract tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Extended coverage — platform detection, backend fallbacks, cache paths
# ---------------------------------------------------------------------------


class KokoroOnnxEdgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tempdir.name)
        self._dir_patch = mock.patch.object(voice_kokoro_onnx, "kokoro_dir", return_value=self.root)
        self._dir_patch.start()
        self._status_patch = mock.patch.dict(
            voice_kokoro_onnx._status, {"state": "idle", "progress": 0.0, "error": None}
        )
        self._status_patch.start()

    def tearDown(self) -> None:
        self._status_patch.stop()
        self._dir_patch.stop()
        self._tempdir.cleanup()

    def test_kokoro_dir_lives_under_home(self) -> None:
        self._dir_patch.stop()
        try:
            with mock.patch.object(voice_kokoro_onnx.Path, "home", return_value=self.root):
                path = voice_kokoro_onnx.kokoro_dir()
        finally:
            self._dir_patch.start()
        self.assertEqual(path, self.root / ".chaosengine" / "voice" / "kokoro")
        self.assertTrue(path.is_dir())

    def test_failed_download_leaves_no_partial_file(self) -> None:
        # Regression: the NamedTemporaryFile(delete=False) partial used to
        # be stranded on failure — each retry leaked another one.
        class ExplodingResponse:
            def read(self, _size: int) -> bytes:
                raise OSError("connection reset")

            def __enter__(self) -> "ExplodingResponse":
                return self

            def __exit__(self, *exc_info: object) -> None:
                return None

        with mock.patch.object(
            voice_kokoro_onnx.urllib.request, "urlopen", side_effect=lambda url, timeout: ExplodingResponse()
        ):
            voice_kokoro_onnx._download_worker()
        self.assertEqual(voice_kokoro_onnx.download_status()["state"], "failed")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_download_skips_files_already_present(self) -> None:
        first_name = voice_kokoro_onnx._FILES[0][0]
        (self.root / first_name).write_bytes(b"already-here")
        fetched: list[str] = []

        class FakeResponse:
            def __init__(self) -> None:
                self._chunks = [b"data", b""]

            def read(self, _size: int) -> bytes:
                return self._chunks.pop(0)

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *exc_info: object) -> None:
                return None

        def fake_urlopen(url: str, timeout: float) -> FakeResponse:
            fetched.append(url)
            return FakeResponse()

        with mock.patch.object(voice_kokoro_onnx.urllib.request, "urlopen", side_effect=fake_urlopen):
            voice_kokoro_onnx._download_worker()
        self.assertEqual(fetched, [voice_kokoro_onnx._FILES[1][1]])
        self.assertEqual((self.root / first_name).read_bytes(), b"already-here")
        status = voice_kokoro_onnx.download_status()
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["progress"], 1.0)

    def test_start_download_spawns_one_worker(self) -> None:
        with mock.patch.object(voice_kokoro_onnx.threading, "Thread") as thread_cls:
            first = voice_kokoro_onnx.start_download()
            second = voice_kokoro_onnx.start_download()
        self.assertEqual(first["state"], "downloading")
        self.assertEqual(second["state"], "downloading")
        thread_cls.assert_called_once()
        self.assertIs(thread_cls.call_args.kwargs["target"], voice_kokoro_onnx._download_worker)
        self.assertTrue(thread_cls.call_args.kwargs["daemon"])

    def test_start_download_resets_a_failed_state(self) -> None:
        voice_kokoro_onnx._status.update(state="failed", progress=0.4, error="boom")
        with mock.patch.object(voice_kokoro_onnx.threading, "Thread"):
            status = voice_kokoro_onnx.start_download()
        self.assertEqual(status, {"state": "downloading", "progress": 0.0, "error": None})

    def test_download_status_returns_a_copy(self) -> None:
        voice_kokoro_onnx.download_status()["state"] = "mutated"
        self.assertEqual(voice_kokoro_onnx.download_status()["state"], "idle")


class GalleryEdgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tempdir.name)
        self._patch = mock.patch.object(voice_gallery, "_gallery_dir", return_value=self.root)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tempdir.cleanup()

    def test_gallery_dir_lives_under_home(self) -> None:
        self._patch.stop()
        try:
            with mock.patch.object(voice_gallery.Path, "home", return_value=self.root):
                path = voice_gallery._gallery_dir()
        finally:
            self._patch.start()
        self.assertEqual(path, self.root / ".chaosengine" / "voice" / "gallery")
        self.assertTrue(path.is_dir())

    def test_generated_ids_are_valid_and_unique(self) -> None:
        ids = {voice_gallery.save_transcript(f"t{i}")["id"] for i in range(5)}
        self.assertEqual(len(ids), 5)
        self.assertTrue(all(voice_gallery.is_valid_id(item_id) for item_id in ids))

    def test_is_valid_id_rejects_near_misses(self) -> None:
        for bad in (
            "1234567890_deadbeef\n",  # regression: ``$`` matched before a newline
            "1234567890_DEADBEEF",
            "1234567890_deadbee",
            "1234567890_deadbeef0",
            "_deadbeef",
            "12345/_deadbeef",
            "",
        ):
            self.assertFalse(voice_gallery.is_valid_id(bad), repr(bad))

    def test_list_items_newest_first(self) -> None:
        for item_id, created in (("1_aaaaaaaa", 100), ("3_cccccccc", 300), ("2_bbbbbbbb", 200)):
            (self.root / f"{item_id}.json").write_text(
                f'{{"id": "{item_id}", "kind": "transcript", "createdAt": {created}, "text": "x"}}',
                encoding="utf-8",
            )
        self.assertEqual(
            [item["id"] for item in voice_gallery.list_items()],
            ["3_cccccccc", "2_bbbbbbbb", "1_aaaaaaaa"],
        )

    def test_save_audio_metadata(self) -> None:
        meta = voice_gallery.save_audio("spoken", "bm_george", b"RIFF")
        self.assertEqual(meta["kind"], "audio")
        self.assertEqual(meta["voice"], "bm_george")
        self.assertEqual(meta["text"], "spoken")
        self.assertIsInstance(meta["createdAt"], int)
        self.assertEqual(voice_gallery.list_items(), [meta])

    def test_transcript_has_no_audio_file(self) -> None:
        meta = voice_gallery.save_transcript("only text")
        self.assertIsNone(voice_gallery.get_audio_path(meta["id"]))
        self.assertTrue(voice_gallery.delete_item(meta["id"]))
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
