"""Voice Library storage and routes, speaker labels on transcripts, and the
transcription options (words, translation, speakers)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service import voice_env, voice_library, voice_runtime
from backend_service.voice_transcript import assign_speakers


class LibraryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data = Path(self._tmp.name)
        self._patch = mock.patch.object(voice_env, "VOICE_DATA_DIR", self.data)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def _audio(self, data: bytes = b"RIFF") -> Path:
        path = self.data / "upload.tmp"
        path.write_bytes(data)
        return path

    def test_transcript_round_trip_moves_the_audio_in(self) -> None:
        upload = self._audio(b"audio-bytes")
        item = voice_library.create(
            "transcript",
            {"text": "Hello there.", "segments": [{"start": 0, "end": 1, "text": "Hello there."}], "language": "en"},
            audio_file=upload,
            audio_ext=".m4a",
        )
        self.assertFalse(upload.exists())
        self.assertEqual(item["title"], "Hello there.")
        self.assertEqual(voice_library.audio_path(item["id"]).read_bytes(), b"audio-bytes")
        self.assertEqual(voice_library.get(item["id"])["segments"][0]["text"], "Hello there.")
        summary = voice_library.list_items()[0]
        self.assertNotIn("segments", summary)
        self.assertEqual(summary["preview"], "Hello there.")
        self.assertTrue(voice_library.delete(item["id"]))
        self.assertEqual(list(voice_library.library_dir().iterdir()), [])

    def test_search_and_kind_filter(self) -> None:
        voice_library.create("transcript", {"text": "Quarterly budget review", "title": "Meeting"})
        voice_library.create("speech", {"text": "Bedtime story", "voice": "af_heart"})
        self.assertEqual([i["title"] for i in voice_library.list_items("BUDGET")], ["Meeting"])
        self.assertEqual([i["kind"] for i in voice_library.list_items(kind="speech")], ["speech"])
        self.assertEqual(voice_library.list_items("nothing like this"), [])

    def test_update_only_touches_editable_fields(self) -> None:
        item = voice_library.create("transcript", {"text": "a"})
        updated = voice_library.update(item["id"], {"title": "New", "speakers": {"0": "Ana"}, "kind": "speech", "id": "x"})
        self.assertEqual((updated["title"], updated["speakers"], updated["kind"], updated["id"]), ("New", {"0": "Ana"}, "transcript", item["id"]))
        self.assertIsNone(voice_library.update("1_deadbeef", {"title": "x"}))

    def test_rejects_unknown_kind_and_audio_type(self) -> None:
        with self.assertRaises(ValueError):
            voice_library.create("video", {"text": "x"})
        with self.assertRaises(ValueError):
            voice_library.create("speech", {"text": "x"}, audio_file=self._audio(), audio_ext=".exe")

    def test_id_validation(self) -> None:
        for bad in ("../x", "1234567890_deadbeef\n", "1234567890_DEADBEEF", ""):
            self.assertFalse(voice_library.is_valid_id(bad), repr(bad))
        self.assertTrue(voice_library.is_valid_id("1234567890_deadbeef"))

    def test_old_gallery_items_move_in(self) -> None:
        old = self.data / "gallery"
        old.mkdir()
        (old / "100_aaaaaaaa.json").write_text(
            json.dumps({"id": "100_aaaaaaaa", "kind": "audio", "createdAt": 100, "text": "spoken", "voice": "bm_george"}),
            encoding="utf-8",
        )
        (old / "100_aaaaaaaa.wav").write_bytes(b"RIFF")
        (old / "200_bbbbbbbb.json").write_text(
            json.dumps({"id": "200_bbbbbbbb", "kind": "transcript", "createdAt": 200, "text": "notes"}), encoding="utf-8"
        )
        (old / "junk.json").write_text("{not json", encoding="utf-8")
        items = voice_library.list_items()
        self.assertEqual([(i["id"], i["kind"]) for i in items], [("200_bbbbbbbb", "transcript"), ("100_aaaaaaaa", "speech")])
        self.assertEqual(voice_library.audio_path("100_aaaaaaaa").read_bytes(), b"RIFF")
        self.assertFalse((old / "100_aaaaaaaa.json").exists())


class AssignSpeakersTests(unittest.TestCase):
    TURNS = [
        {"start": 0.0, "end": 2.0, "speaker": 0},
        {"start": 2.2, "end": 4.0, "speaker": 1},
    ]

    def test_splits_a_segment_where_the_speaker_changes(self) -> None:
        segment = {
            "start": 0.0, "end": 3.5, "text": "Hi there. Hello you.",
            "words": [
                {"start": 0.1, "end": 0.5, "text": "Hi"},
                {"start": 0.6, "end": 1.0, "text": "there."},
                {"start": 2.3, "end": 2.8, "text": "Hello"},
                {"start": 2.9, "end": 3.4, "text": "you."},
            ],
        }
        out = assign_speakers([segment], self.TURNS, "en")
        self.assertEqual([(s["speaker"], s["text"], s["start"], s["end"]) for s in out], [(0, "Hi there.", 0.1, 1.0), (1, "Hello you.", 2.3, 3.4)])

    def test_cjk_words_join_without_spaces(self) -> None:
        words = [{"start": 0.1, "end": 0.3, "text": "你好"}, {"start": 2.5, "end": 2.8, "text": "再见"}]
        out = assign_speakers([{"start": 0.1, "end": 2.8, "text": "你好再见", "words": words}], self.TURNS, "zh")
        self.assertEqual([s["text"] for s in out], ["你好", "再见"])

    def test_segment_without_words_and_words_in_gaps(self) -> None:
        out = assign_speakers([{"start": 2.5, "end": 3.0, "text": "x"}], self.TURNS)
        self.assertEqual(out[0]["speaker"], 1)
        # 2.05-2.15 falls in the gap; the nearer turn edge (2.0) wins.
        gap = assign_speakers([{"start": 2.05, "end": 2.15, "text": "y", "words": [{"start": 2.05, "end": 2.08, "text": "y"}]}], self.TURNS)
        self.assertEqual(gap[0]["speaker"], 0)
        self.assertIsNone(assign_speakers([{"start": 0, "end": 1, "text": "z"}], [])[0]["speaker"])


class WorkerWordTests(unittest.TestCase):
    def test_words_are_plain_floats_and_parakeet_tokens_join_at_spaces(self) -> None:
        import numpy as np

        from backend_service import voice_worker

        word = voice_worker._word(np.float64(1.234), np.float64(1.5), " hi ")
        self.assertEqual(word, {"start": 1.23, "end": 1.5, "text": "hi"})
        self.assertIs(type(word["start"]), float)  # json cannot write numpy floats
        token = lambda text, start, end: mock.Mock(text=text, start=start, end=end)  # noqa: E731
        words = voice_worker._parakeet_words([token(" Hel", 0.0, 0.2), token("lo", 0.2, 0.4), token(" there", 0.5, 0.9)])
        self.assertEqual([(w["text"], w["start"], w["end"]) for w in words], [("Hello", 0.0, 0.4), ("there", 0.5, 0.9)])


class TranscribeOptionsTests(unittest.TestCase):
    def _worker(self, replies: dict):
        worker = mock.Mock()
        worker.request.side_effect = lambda cmd, **kw: dict(replies[cmd], _kw=kw)
        return worker

    def test_speakers_label_segments_and_drop_words_unless_asked(self) -> None:
        worker = self._worker({
            "transcribe": {
                "text": "A B", "language": "en", "seconds": 1.0,
                "segments": [{"start": 0, "end": 3, "text": "A B", "words": [
                    {"start": 0.1, "end": 0.5, "text": "A"}, {"start": 2.5, "end": 2.9, "text": "B"}]}],
            },
            "diarize": {"turns": AssignSpeakersTests.TURNS, "seconds": 0.5},
        })
        with (
            mock.patch("backend_service.voice_client.get_worker", return_value=worker),
            mock.patch.object(voice_runtime, "resolve_stt_model", return_value=("mlx-whisper", "repo")),
            mock.patch.object(voice_runtime, "_hf_snapshot_path", return_value=None),
            mock.patch.object(voice_env, "diarize_installed", return_value=True),
            mock.patch.object(voice_env, "diarize_model_paths", return_value={"segmentation": Path("s"), "embedding": Path("e")}),
        ):
            result = voice_runtime.transcribe_file(Path("a.wav"), "", speakers=True, speaker_count=2)
        self.assertEqual([(s["speaker"], s["text"]) for s in result["segments"]], [(0, "A"), (1, "B")])
        self.assertNotIn("words", result["segments"][0])
        self.assertEqual(result["seconds"], 1.5)
        transcribe_call, diarize_call = worker.request.call_args_list
        self.assertTrue(transcribe_call.kwargs["words"])
        self.assertEqual(diarize_call.kwargs["speakers"], 2)

    def test_translate_refuses_parakeet_and_speakers_need_the_update(self) -> None:
        with mock.patch.object(voice_runtime, "resolve_stt_model", return_value=("parakeet-mlx", "repo")):
            with self.assertRaises(ValueError):
                voice_runtime.transcribe_file(Path("a.wav"), "", task="translate")
        with (
            mock.patch.object(voice_runtime, "resolve_stt_model", return_value=("mlx-whisper", "repo")),
            mock.patch.object(voice_env, "diarize_installed", return_value=False),
        ):
            with self.assertRaises(RuntimeError):
                voice_runtime.transcribe_file(Path("a.wav"), "", speakers=True)

    def test_only_models_trained_to_translate_may_translate(self) -> None:
        # Whisper Large v3 Turbo answers in the spoken language when asked to translate.
        self.assertFalse(voice_runtime._can_translate("mlx-community/whisper-large-v3-turbo-q4"))
        self.assertFalse(voice_runtime._can_translate("mobiuslabsgmbh/faster-whisper-large-v3-turbo"))
        self.assertTrue(voice_runtime._can_translate("mlx-community/whisper-small-mlx"))
        self.assertTrue(voice_runtime._can_translate("Systran/faster-whisper-base"))
        rows = {row["key"]: row["translate"] for row in voice_runtime.list_stt_models()}
        self.assertFalse(rows["whisper-large-v3-turbo"])
        self.assertTrue(rows["whisper-small"])

    def test_missing_features_lists_what_an_old_pack_lacks(self) -> None:
        with (
            mock.patch.object(voice_env, "is_installed", return_value=True),
            mock.patch.object(voice_env, "read_marker", return_value={"features": ["vad"]}),
        ):
            self.assertEqual(voice_env.missing_features(), ["speakers"])
        with mock.patch.object(voice_env, "is_installed", return_value=False):
            self.assertEqual(voice_env.missing_features(), [])


class LibraryRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_video_routes import make_client

        self.client, self._tmp, self._snapshot = make_client()
        self._data = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(self._data.name))
        self._patch.start()

    def tearDown(self) -> None:
        from tests.test_video_routes import restore_env

        self._patch.stop()
        self._data.cleanup()
        restore_env(self._snapshot)
        self._tmp.cleanup()

    def test_transcribe_with_save_keeps_the_file_in_the_library(self) -> None:
        def fake(path, model, language=None, **options):
            return {"text": "hi", "segments": [{"start": 0, "end": 1, "text": "hi", "speaker": 0}], "language": "en"}

        with mock.patch.object(voice_runtime, "transcribe_file", side_effect=fake):
            resp = self.client.post(
                "/api/voice/transcribe",
                files={"audio": ("interview.mp3", b"mp3-bytes", "audio/mpeg")},
                data={"save": "true", "speakers": "true", "task": "transcribe"},
            )
        item_id = resp.json()["libraryId"]
        item = self.client.get(f"/api/voice/library/{item_id}").json()
        self.assertEqual((item["title"], item["source"], item["audio"]), ("interview", "interview.mp3", ".mp3"))
        audio = self.client.get(f"/api/voice/library/{item_id}/audio")
        self.assertEqual((audio.content, audio.headers["content-type"]), (b"mp3-bytes", "audio/mpeg"))
        self.assertEqual(self.client.post("/api/voice/transcribe", files={"audio": ("a.wav", b"RIFF")}, data={"task": "dance"}).status_code, 422)

    def test_speech_update_list_and_bulk_delete(self) -> None:
        speech = self.client.post(
            "/api/voice/library/speech",
            files={"audio": ("speech.wav", b"RIFF", "audio/wav")},
            data={"text": "Once upon a time", "voice": "af_heart", "speed": "1.2"},
        ).json()
        transcript = self.client.post(
            "/api/voice/library/speech", files={"audio": ("b.wav", b"RIFF", "audio/wav")}, data={"text": "other"}
        ).json()
        updated = self.client.post(
            f"/api/voice/library/{speech['id']}",
            json={"title": "Story", "speakers": {"0": "  Ana   Lopez ", "x": "bad"}, "segments": [{"start": 0, "end": 1, "text": "x"}]},
        ).json()
        self.assertEqual((updated["title"], updated["speakers"]), ("Story", {"0": "Ana Lopez"}))
        listed = self.client.get("/api/voice/library", params={"q": "once", "kind": "speech"}).json()["items"]
        self.assertEqual([i["id"] for i in listed], [speech["id"]])
        self.assertEqual(self.client.get("/api/voice/library", params={"kind": "video"}).status_code, 400)
        deleted = self.client.post("/api/voice/library/delete", json={"ids": [speech["id"], transcript["id"]]}).json()
        self.assertEqual(sorted(deleted["deleted"]), sorted([speech["id"], transcript["id"]]))
        self.assertEqual(self.client.get("/api/voice/library").json()["items"], [])

    def test_ids_and_missing_items(self) -> None:
        self.assertEqual(self.client.get("/api/voice/library/not-an-id").status_code, 400)
        self.assertEqual(self.client.get("/api/voice/library/1234567890_deadbeef").status_code, 404)
        self.assertEqual(self.client.delete("/api/voice/library/1234567890_deadbeef").status_code, 404)
        self.assertEqual(self.client.post("/api/voice/library/delete", json={"ids": ["../x"]}).status_code, 400)

    def test_mp3_export_goes_through_the_worker(self) -> None:
        item = self.client.post("/api/voice/library/speech", files={"audio": ("s.wav", b"RIFF", "audio/wav")}).json()

        def fake_encode(source, target):
            target.write_bytes(b"ID3" + source.read_bytes())

        with mock.patch.object(voice_runtime, "encode_mp3", side_effect=fake_encode):
            resp = self.client.get(f"/api/voice/library/{item['id']}/audio", params={"format": "mp3"})
            encoded = self.client.post("/api/voice/encode", files={"audio": ("x.wav", b"RIFF", "audio/wav")})
        self.assertEqual((resp.content, resp.headers["content-type"]), (b"ID3RIFF", "audio/mpeg"))
        self.assertEqual(encoded.content, b"ID3RIFF")
        self.assertEqual(self.client.get(f"/api/voice/library/{item['id']}/audio", params={"format": "ogg"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
