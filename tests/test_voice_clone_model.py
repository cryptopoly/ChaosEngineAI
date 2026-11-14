"""The cloning model chosen when a voice is made (FU-154): recorded with the voice, the default when none is named,
used by Speak and read-aloud when the request names none."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service import voice_clone, voice_env, voice_library, voice_runtime, voice_speak
from backend_service.catalog import voice_models
from tests.test_voice_clone import make_wav

_PATCHES: list = []


def setUpModule() -> None:
    for patch in (mock.patch.object(voice_env, "lane", return_value="mlx"),
                  mock.patch.object(voice_env, "speech_lanes", return_value=["mlx"])):
        patch.start()
        _PATCHES.append(patch)


def tearDownModule() -> None:
    while _PATCHES:
        _PATCHES.pop().stop()


def installed(*keys: str):
    rows = {voice_models.voice_model(k)["repo"] for k in keys}
    return lambda row: Path("/snap") / row["key"] if row["repo"] in rows else None


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data = Path(self._tmp.name)
        patch = mock.patch.object(voice_env, "VOICE_DATA_DIR", self.data)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self._tmp.cleanup)

    def create(self, **kwargs):
        sample = self.data / "sample.wav"
        sample.write_bytes(make_wav())
        args = dict(speaker_name="Dan", basis="own", name="Me", sample_audio=sample, sample_ext=".wav", sample_text=None,
                    permission_confirmed=True, origin="ui",
                    transcribe=mock.Mock(return_value={"text": "Hello, this is my voice sample.", "durationS": 8.0}))
        args.update(kwargs)
        return voice_clone.create_voice(**args)


class CreateTests(_Base):
    def test_the_default_is_one_constant_the_form_reads(self) -> None:
        listed = {r["key"]: r for r in voice_clone.list_models()}
        self.assertEqual([k for k, r in listed.items() if r["default"]], [voice_models.DEFAULT_CLONE_MODEL])

    def test_the_chosen_model_is_saved_with_the_voice(self) -> None:
        with mock.patch.object(voice_clone, "snapshot_dir", installed("qwen3-tts-0.6b", "voxcpm2")):
            voice = self.create(model_key="voxcpm2")
        self.assertEqual(voice["model"], "voxcpm2")
        self.assertEqual(voice_library.get(voice["id"])["model"], "voxcpm2")

    def test_an_empty_model_with_two_downloaded_takes_the_default_and_is_not_refused(self) -> None:
        both = installed("qwen3-tts-0.6b", "qwen3-tts-1.7b", "voxcpm2")
        with mock.patch.object(voice_clone, "snapshot_dir", both):
            voice = self.create(model_key="")
        self.assertEqual(voice["model"], voice_models.DEFAULT_CLONE_MODEL)

    def test_the_default_gives_way_to_the_only_downloaded_model(self) -> None:
        with mock.patch.object(voice_clone, "snapshot_dir", installed("chatterbox-turbo")):
            self.assertEqual(self.create(model_key="")["model"], "chatterbox-turbo")

    def test_an_unknown_model_is_a_plain_error(self) -> None:
        with self.assertRaises(ValueError):
            self.create(model_key="no-such-model")

    def test_voices_made_before_the_model_was_recorded_report_none(self) -> None:
        with mock.patch.object(voice_clone, "snapshot_dir", installed()):
            voice = self.create()
        item = voice_library.get(voice["id"])
        item.pop("model")
        self.assertIsNone(voice_clone._public(item)["model"])


class SpeakTests(_Base):
    def _plan_model(self, voice, request_model: str, downloaded) -> str:
        with mock.patch.object(voice_clone, "snapshot_dir", downloaded), \
             mock.patch.object(voice_clone, "available_here", return_value=True), \
             mock.patch.object(voice_env, "is_installed", return_value=True), \
             mock.patch.object(voice_env, "watermark_installed", return_value=True):
            _plan, row, _voice, _notes = voice_clone.clone_plan(voice["id"], request_model, origin="ui")
        return row["key"]

    def test_speak_uses_the_model_chosen_at_creation_unless_one_is_named(self) -> None:
        both = installed("qwen3-tts-0.6b", "qwen3-tts-1.7b", "voxcpm2")
        with mock.patch.object(voice_clone, "snapshot_dir", both):
            voice = self.create(model_key="voxcpm2")
        self.assertEqual(self._plan_model(voice, "", both), "voxcpm2")
        self.assertEqual(self._plan_model(voice, "qwen3-tts-0.6b", both), "qwen3-tts-0.6b")

    def test_a_recorded_model_that_is_no_longer_downloaded_falls_back_to_the_default(self) -> None:
        with mock.patch.object(voice_clone, "snapshot_dir", installed("voxcpm2")):
            voice = self.create(model_key="voxcpm2")
        self.assertEqual(self._plan_model(voice, "", installed("qwen3-tts-0.6b", "qwen3-tts-1.7b")),
                         voice_models.DEFAULT_CLONE_MODEL)

    def test_the_voice_list_shows_each_voice_with_its_own_model(self) -> None:
        both = installed("qwen3-tts-0.6b", "voxcpm2")
        with mock.patch.object(voice_clone, "snapshot_dir", both):
            first = self.create(model_key="voxcpm2")
            second = self.create(model_key="qwen3-tts-0.6b")
        with mock.patch.object(voice_clone, "snapshot_dir", both), \
             mock.patch.object(voice_speak, "_installed", side_effect=lambda row: both(row) is not None), \
             mock.patch.object(voice_env, "watermark_installed", return_value=True):
            models = {v["id"]: v["model"] for v in voice_speak.list_voices() if v["kind"] == "clone"}
        self.assertEqual(models[voice_speak.CLONE_PREFIX + first["id"]], "voxcpm2")
        self.assertEqual(models[voice_speak.CLONE_PREFIX + second["id"]], "qwen3-tts-0.6b")


class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_video_routes import make_client, restore_env

        self.client, self._tmp, self._snapshot = make_client()
        self._data = tempfile.TemporaryDirectory()
        patch = mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(self._data.name))
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self._data.cleanup)
        self.addCleanup(restore_env, self._snapshot)

    def post(self, **extra):
        data = {"speakerName": "Dan", "basis": "own", "permissionConfirmed": "true", "name": "Me", **extra}
        heard = {"text": "Hello, this is my voice sample.", "durationS": 6.0}
        with mock.patch.object(voice_runtime, "transcribe_file", return_value=heard):
            return self.client.post("/api/voice/clone/voices", data=data,
                                    files={"sample": ("sample.wav", make_wav(), "audio/wav")})

    def test_create_takes_a_model_and_an_omitted_one_uses_the_default(self) -> None:
        both = installed("qwen3-tts-0.6b", "qwen3-tts-1.7b", "voxcpm2")
        with mock.patch.object(voice_clone, "snapshot_dir", both):
            chosen = self.post(model="voxcpm2")
            omitted = self.post()
            unknown = self.post(model="nope")
        self.assertEqual((chosen.status_code, chosen.json()["model"]), (200, "voxcpm2"))
        self.assertEqual((omitted.status_code, omitted.json()["model"]), (200, voice_models.DEFAULT_CLONE_MODEL))
        self.assertEqual(unknown.status_code, 400)
        self.assertIn("Unknown voice cloning model", unknown.json()["detail"])


if __name__ == "__main__":
    unittest.main()
