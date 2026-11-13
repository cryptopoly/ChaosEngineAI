"""Kokoro as an optional add-on of the voice runtime, and the Fallback voice setting (D-75).

Install without Kokoro, remove it and add it again, refusal while voice is busy, the checked delete paths, and which
voice reads when the chosen one cannot: the fallback, none (a plain error), or a cloned fallback refused for calls.
Every test runs on a throwaway Hugging Face cache, voice venv and voice data folder; nothing touches the real ones.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service import voice_chat, voice_client, voice_env, voice_model_files, voice_runtime, voice_runtime_disk
from backend_service import voice_speak

KOKORO_ONNX_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")


class _Sandbox(unittest.TestCase):
    """A temporary HF cache, voice venv (with a marker) and voice data folder, on the lane ``LANE``."""

    LANE = "mlx"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cache = root / "hub"
        self.cache.mkdir()
        self.data = root / "voice"
        self.data.mkdir()
        venv = root / "voice-venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        self.venv = venv
        self.marker = venv / "chaosengine-voice.json"
        self.marker.write_text(json.dumps({"schema": 1, "lane": self.LANE, "extras": ["kokoro"], "features": voice_env.FEATURES}),
                               encoding="utf-8")
        self.patches = [
            mock.patch.object(voice_env, "VOICE_VENV_DIR", venv),
            mock.patch.object(voice_env, "VOICE_DATA_DIR", self.data),
            mock.patch.object(voice_env, "_MARKER", self.marker),
            mock.patch.object(voice_env, "_is_windows", return_value=False),
            mock.patch.dict(os.environ, {"CHAOSENGINE_VOICE_LANE": self.LANE, "HF_HUB_CACHE": str(self.cache),
                                         "HUGGINGFACE_HUB_CACHE": str(self.cache)}),
            mock.patch.object(voice_model_files, "voice_busy_reason", return_value=None),
            mock.patch.object(voice_client, "stop_all"),
            mock.patch.object(voice_client, "stop_speech_worker"),
        ]
        for patch in self.patches:
            patch.start()
        voice_runtime_disk._PARTS_CACHE.clear()

    def tearDown(self) -> None:
        for patch in reversed(self.patches):
            patch.stop()
        voice_runtime_disk._PARTS_CACHE.clear()
        voice_env._JOB.__init__()  # type: ignore[misc]
        self.tmp.cleanup()

    def put_kokoro(self) -> Path:
        """Kokoro's files for this lane; returns the folder that holds them."""
        if self.LANE == "mlx":
            snapshot = self.cache / f"models--{voice_env.KOKORO_MLX_REPO.replace('/', '--')}" / "snapshots" / "abc"
            snapshot.mkdir(parents=True)
            (snapshot / "kokoro-v1_0.safetensors").write_bytes(b"k" * 1_000_000)
            return snapshot.parent.parent
        folder = self.data / "kokoro"
        folder.mkdir()
        for name in KOKORO_ONNX_FILES:
            (folder / name).write_bytes(b"k" * 500_000)
        return folder


class InstallWithoutKokoroTests(_Sandbox):
    def run_install(self, extras: list[str]) -> list[str]:
        """Run the install job with every command and download mocked; returns the Kokoro downloads asked for."""
        asked: list[str] = []
        with mock.patch.object(voice_env, "_run"), \
                mock.patch.object(voice_env, "download_kokoro", side_effect=asked.append), \
                mock.patch("huggingface_hub.hf_hub_download"), \
                mock.patch.object(voice_env.shutil, "disk_usage", return_value=mock.Mock(free=100e9)):
            voice_env._run_install(extras)
        self.assertEqual(voice_env.job_status()["phase"], "done", voice_env.job_status()["error"])
        return asked

    def test_unticked_kokoro_is_not_downloaded(self) -> None:
        self.assertEqual(self.run_install([]), [])
        self.assertEqual(json.loads(self.marker.read_text())["extras"], [])
        self.assertNotIn("kokoro", voice_env.installed_extras())
        self.assertTrue(voice_env.is_installed())

    def test_ticked_kokoro_is_downloaded(self) -> None:
        self.assertEqual(self.run_install(["kokoro"]), ["mlx"])

    def test_the_add_on_is_offered_on_every_lane_and_reports_its_files(self) -> None:
        extras = {e["id"]: e for e in voice_runtime.get_voice_capabilities()["extras"]}
        self.assertFalse(extras["kokoro"]["installed"])
        self.put_kokoro()
        extras = {e["id"]: e for e in voice_runtime.get_voice_capabilities()["extras"]}
        self.assertTrue(extras["kokoro"]["installed"])
        with mock.patch.object(voice_env, "lane", return_value="portable"):
            self.assertIn("kokoro", [e["id"] for e in voice_runtime.get_voice_capabilities()["extras"]])

    def test_a_runtime_without_kokoro_lists_its_voices_as_not_installed(self) -> None:
        quick = [v for v in voice_speak.list_voices(include_clones=False) if v["kind"] == "builtin"]
        self.assertTrue(quick)
        self.assertFalse(any(v["available"] for v in quick))
        self.assertIn("Kokoro quick voices are not installed", quick[0]["needs"])
        with self.assertRaises(RuntimeError):
            voice_runtime.synthesize("Hello.", "af_heart", 1.0)

    def test_read_aloud_is_available_with_a_speech_model_and_no_kokoro(self) -> None:
        self.assertFalse(voice_runtime.get_voice_capabilities()["ttsAvailable"])
        with mock.patch.object(voice_speak, "natural_voice_installed", return_value=True):
            caps = voice_runtime.get_voice_capabilities()
        self.assertTrue(caps["ttsAvailable"])
        self.assertFalse(caps["ttsModelInstalled"])


class RemoveKokoroTests(_Sandbox):
    def test_untick_removes_the_files_and_tick_adds_them_back(self) -> None:
        folder = self.put_kokoro()
        self.assertIn("kokoro", voice_env.installed_extras())
        result = voice_runtime_disk.remove_extra("kokoro")
        self.assertFalse(folder.exists())
        self.assertEqual(result["files"], [voice_env.KOKORO_MLX_REPO])
        self.assertNotIn("kokoro", voice_env.installed_extras())
        self.assertNotIn("kokoro", json.loads(self.marker.read_text())["extras"])
        self.assertTrue(self.venv.exists(), "the runtime stays")
        with mock.patch.object(voice_env.threading, "Thread") as thread:
            job = voice_env.start_install(["kokoro"])
        self.assertEqual(job["extras"], ["kokoro"])
        thread.assert_called_once()
        self.assertEqual(thread.call_args.kwargs["args"], (["kokoro"],))

    def test_a_reinstall_keeps_kokoro_when_it_is_there(self) -> None:
        self.put_kokoro()
        with mock.patch.object(voice_env.threading, "Thread"):
            self.assertIn("kokoro", voice_env.start_install([])["extras"])

    def test_refused_while_voice_is_busy(self) -> None:
        folder = self.put_kokoro()
        with mock.patch.object(voice_model_files, "voice_busy_reason", return_value="A voice model is speaking."):
            with self.assertRaises(RuntimeError):
                voice_runtime_disk.remove_extra("kokoro")
        self.assertTrue(folder.exists())

    def test_refused_while_an_install_runs(self) -> None:
        folder = self.put_kokoro()
        with mock.patch.object(voice_env, "job_status", return_value={"running": True}):
            with self.assertRaises(RuntimeError):
                voice_runtime_disk.remove_extra("kokoro")
        self.assertTrue(folder.exists())

    def test_not_installed_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            voice_runtime_disk.remove_extra("kokoro")

    def test_parts_say_kokoro_is_empty_after_removal(self) -> None:
        self.put_kokoro()
        before = {p["id"]: p["sizeGb"] for p in voice_runtime_disk.pack_parts()}
        self.assertGreater(before["kokoro"], 0)
        voice_runtime_disk.remove_extra("kokoro")
        after = {p["id"]: p["sizeGb"] for p in voice_runtime_disk.pack_parts()}
        self.assertEqual(after["kokoro"], 0)


class RemoveKokoroPortableTests(_Sandbox):
    LANE = "portable"

    def test_untick_removes_the_onnx_folder(self) -> None:
        folder = self.put_kokoro()
        self.assertTrue(voice_env.kokoro_installed())
        result = voice_runtime_disk.remove_extra("kokoro")
        self.assertFalse(folder.exists())
        self.assertEqual(result["files"], ["kokoro"])
        self.assertFalse(voice_env.kokoro_installed())
        self.assertFalse(folder.exists(), "asking whether Kokoro is installed does not create its folder")

    def test_a_folder_outside_the_voice_data_folder_is_never_deleted(self) -> None:
        outside = Path(self.tmp.name) / "elsewhere" / "kokoro"
        outside.mkdir(parents=True)
        for name in KOKORO_ONNX_FILES:
            (outside / name).write_bytes(b"k")
        with mock.patch.object(voice_env, "kokoro_onnx_dir", return_value=outside):
            with self.assertRaises(RuntimeError):
                voice_runtime_disk.remove_extra("kokoro")
        self.assertTrue(outside.exists())

    def test_remove_runtime_with_its_models_deletes_kokoro_through_the_checked_path(self) -> None:
        folder = self.put_kokoro()
        result = voice_runtime_disk.remove_runtime(True)
        self.assertFalse(folder.exists())
        self.assertIn("kokoro", result["removedModels"])


def _voice(voice_id: str, kind: str, *, available: bool = True) -> dict:
    return {"id": voice_id, "name": voice_id, "kind": kind, "available": available, "chatUse": voice_speak.chat_uses(kind)}


VOICES = [
    _voice("af_heart", "builtin"),
    _voice("qwen3:Ryan", "preset"),
    _voice("qwen3:Vivian", "preset", available=False),
    _voice("design", "design"),
    _voice("clone:1234567890_deadbeef", "clone"),
]


class FallbackTests(unittest.TestCase):
    """``voice_chat`` with a fixed voice list: which voice reads when the chosen one cannot."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(self.tmp.name)),
            mock.patch.object(voice_env, "kokoro_installed", return_value=True),
            mock.patch.object(voice_speak, "list_voices",
                              side_effect=lambda include_clones=True: [v for v in VOICES if include_clones or v["kind"] != "clone"]),
        ]
        for patch in self.patches:
            patch.start()
        voice_chat.forget_failures()

    def tearDown(self) -> None:
        voice_chat.forget_failures()
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    def set_fallback(self, voice: str) -> None:
        path = voice_runtime._settings_path()
        path.write_text(json.dumps({"fallbackVoice": voice}), encoding="utf-8")

    def ryan_fails(self):
        return mock.patch.object(voice_speak, "build_plan", side_effect=RuntimeError("Qwen3-TTS is not downloaded"))

    def test_the_default_fallback_is_kokoro_heart_while_kokoro_is_installed(self) -> None:
        self.assertEqual(voice_chat.fallback_for("read"), "af_heart")
        self.assertEqual(voice_chat.fallback_for("call"), "af_heart")
        with mock.patch.object(voice_env, "kokoro_installed", return_value=False), \
                mock.patch.object(voice_speak, "list_voices", return_value=[_voice("af_heart", "builtin", available=False)]):
            self.assertIsNone(voice_chat.fallback_for("read"))

    def test_the_chosen_fallback_reads_with_a_note_naming_it(self) -> None:
        self.set_fallback("qwen3:Ryan")
        ryan = voice_speak.Plan("preset", "qwen3:Ryan", "qwen3-tts-custom-1.7b", "Qwen3-TTS")
        def build(voice: str, *_a, **_k):
            if voice != "qwen3:Ryan":
                raise RuntimeError("Qwen3-TTS is not downloaded")
            return ryan

        with mock.patch.object(voice_speak, "build_plan", side_effect=build), \
                mock.patch.object(voice_speak, "prepare_text", return_value=(["Hello."], [])):
            chat = voice_chat.plan_stream("Hello.", "qwen3:Vivian", use="call")
        self.assertEqual(chat.voice, "qwen3:Ryan")
        self.assertTrue(chat.fallback)
        self.assertIn("Vivian could not speak here", chat.notes[-1])
        self.assertIn("the fallback voice Ryan reads instead", chat.notes[-1])

    def test_with_no_fallback_the_chosen_voice_failing_is_a_plain_error(self) -> None:
        self.set_fallback("")
        with self.ryan_fails():
            with self.assertRaises(voice_chat.NoFallbackVoice) as caught:
                voice_chat.plan_stream("Hello.", "qwen3:Ryan")
        self.assertEqual(str(caught.exception), voice_chat.NO_FALLBACK)

    def test_a_fallback_that_is_not_installed_counts_as_none(self) -> None:
        self.set_fallback("qwen3:Vivian")
        with self.ryan_fails(), self.assertRaises(voice_chat.NoFallbackVoice):
            voice_chat.plan_stream("Hello.", "qwen3:Ryan")

    def test_a_cloned_fallback_reads_aloud_but_never_answers_calls(self) -> None:
        self.set_fallback("clone:1234567890_deadbeef")
        self.assertEqual(voice_chat.fallback_for("read"), "clone:1234567890_deadbeef")
        self.assertIsNone(voice_chat.fallback_for("call"))
        self.assertIsNone(voice_chat.fallback_for("read", origin="agent"))
        with self.ryan_fails(), self.assertRaises(voice_chat.NoFallbackVoice):
            voice_chat.plan_stream("Hello.", "qwen3:Ryan", use="call")

    def test_the_fallback_failing_too_gives_its_own_reason(self) -> None:
        with mock.patch.object(voice_runtime, "synthesize", side_effect=RuntimeError("no voice")):
            chat = voice_chat.plan_stream("Hello there.", "af_heart")
            with self.assertRaises(RuntimeError) as caught:
                voice_chat.switch_to_fallback(chat, chat.sentences, "no voice")
        self.assertNotIsInstance(caught.exception, voice_chat.NoFallbackVoice)
        self.assertEqual(str(caught.exception), "no voice")

    def test_setting_validation(self) -> None:
        voice_chat.check_fallback_setting("", include_clones=True)
        voice_chat.check_fallback_setting("af_heart", include_clones=True)
        voice_chat.check_fallback_setting("clone:1234567890_deadbeef", include_clones=True)
        for bad in ("qwen3:Nobody", "design", "qwen3:Vivian"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                voice_chat.check_fallback_setting(bad, include_clones=True)
        with self.assertRaises(ValueError):
            voice_chat.check_fallback_setting("clone:1234567890_deadbeef", include_clones=False)


class FallbackRouteTests(unittest.TestCase):
    """The settings key is validated at the request edge; a stream with no fallback answers 409 in plain words."""

    def setUp(self) -> None:
        from tests.test_video_routes import make_client

        self.client, self._tmp, self._snapshot = make_client()
        self._data = tempfile.TemporaryDirectory()
        self._patches = [
            mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(self._data.name)),
            mock.patch.object(voice_env, "kokoro_installed", return_value=False),
        ]
        for patch in self._patches:
            patch.start()
        voice_chat.forget_failures()

    def tearDown(self) -> None:
        from tests.test_video_routes import restore_env

        voice_chat.forget_failures()
        for patch in reversed(self._patches):
            patch.stop()
        self._data.cleanup()
        restore_env(self._snapshot)
        self._tmp.cleanup()

    def test_settings_validate_the_fallback_voice(self) -> None:
        self.assertEqual(self.client.post("/api/voice/settings", json={"fallbackVoice": "qwen3:Nobody"}).status_code, 400)
        self.assertEqual(self.client.post("/api/voice/settings", json={"fallbackVoice": "design"}).status_code, 400)
        refused = self.client.post("/api/voice/settings", json={"fallbackVoice": "af_heart"})
        self.assertEqual(refused.status_code, 400, "Kokoro is not installed here")
        self.assertIn("must be installed", refused.json()["detail"])
        self.assertEqual(self.client.post("/api/voice/settings", json={"fallbackVoice": "x" * 65}).status_code, 422)
        cleared = self.client.post("/api/voice/settings", json={"fallbackVoice": ""})
        self.assertEqual(cleared.status_code, 200)
        self.assertEqual(cleared.json()["fallbackVoice"], "")
        with mock.patch.object(voice_env, "kokoro_installed", return_value=True):
            self.assertEqual(self.client.post("/api/voice/settings", json={"fallbackVoice": "af_heart"}).status_code, 200)

    def test_read_aloud_without_kokoro_or_a_fallback_is_a_plain_409(self) -> None:
        response = self.client.post("/api/voice/speak/stream", json={"text": "Hello.", "voice": "af_heart"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], voice_chat.NO_FALLBACK)

    def test_the_extras_route_refuses_kokoro_removal_while_busy(self) -> None:
        with mock.patch.object(voice_env, "kokoro_installed", return_value=True), \
                mock.patch.object(voice_model_files, "voice_busy_reason", return_value="A voice model is speaking."):
            response = self.client.post("/api/voice/extras/remove", json={"extra": "kokoro"})
        self.assertEqual(response.status_code, 409)

    def test_the_stream_hands_over_to_the_fallback_mid_way(self) -> None:
        with mock.patch.object(voice_env, "kokoro_installed", return_value=True), \
                mock.patch.object(voice_runtime, "synthesize", side_effect=lambda t, v, s: (v.encode(), 1.0)), \
                mock.patch.object(voice_speak, "build_plan", side_effect=RuntimeError("out of memory")):
            response = self.client.post("/api/voice/speak/stream", json={"text": "One. Two.", "voice": "qwen3:Ryan"})
        lines = [json.loads(line) for line in response.text.splitlines()]
        self.assertIn("fallback voice Heart (Kokoro)", lines[0]["note"])
        self.assertEqual(base64.b64decode(lines[1]["audio"]), b"af_heart")


if __name__ == "__main__":
    unittest.main()
