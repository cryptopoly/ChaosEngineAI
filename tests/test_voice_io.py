"""Voice I/O: text clean-up, catalogs, the voice venv, the worker protocol and the routes.

The worker tests run the real ``voice_worker.py`` with this interpreter for
the commands that need no speech packages (ping, status, unload, errors);
transcription and speech are exercised live on the voice venv, not here.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import textwrap
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

from backend_service import voice_client, voice_env, voice_runtime
from backend_service.voice_text import MAX_SPEAK_CHARS, clean_for_speech, speech_sentences, split_sentences


class CleanForSpeechTests(unittest.TestCase):
    def test_code_blocks_are_announced_not_read(self) -> None:
        text = "Here it is:\n```python\nprint('hi')\n```\nDone."
        cleaned = clean_for_speech(text)
        self.assertIn("code block omitted", cleaned)
        self.assertNotIn("print", cleaned)

    def test_links_read_their_text_and_bare_urls_become_link(self) -> None:
        cleaned = clean_for_speech("See [the docs](https://x.y/z) or https://example.com/a?b=c now.")
        self.assertIn("the docs", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertIn("link", cleaned)

    def test_markdown_markers_and_reasoning_are_removed(self) -> None:
        cleaned = clean_for_speech("<think>plan</think>## Title\n- **bold** item\n> quoted `code`")
        self.assertNotIn("plan", cleaned)
        for marker in ("#", "**", "- ", "> ", "`"):
            self.assertNotIn(marker, cleaned)
        self.assertIn("Title.", cleaned)  # a heading ends a sentence
        self.assertIn("bold item.", cleaned)

    def test_tool_calls_are_not_read(self) -> None:
        cleaned = clean_for_speech('Sure.<tool_call>{"name": "calc"}</tool_call> Result is 4.')
        self.assertNotIn("calc", cleaned)


class SplitSentencesTests(unittest.TestCase):
    def test_splits_on_end_punctuation_and_paragraphs(self) -> None:
        self.assertEqual(split_sentences("One. Two! Three?\n\nFour"), ["One.", "Two!", "Three?", "Four"])

    def test_abbreviations_stay_in_their_sentence(self) -> None:
        self.assertEqual(split_sentences("Use tools, e.g. a hammer. Then stop."), ["Use tools, e.g. a hammer.", "Then stop."])

    def test_long_sentences_are_cut_at_commas(self) -> None:
        long = ", ".join(["word"] * 200) + "."
        pieces = split_sentences(long, max_chars=100)
        self.assertTrue(all(len(p) <= 101 for p in pieces))
        self.assertEqual(" ".join(pieces).replace(" ", "").replace(",", ""), long.replace(" ", "").replace(",", ""))

    def test_cjk_punctuation_ends_sentences(self) -> None:
        self.assertEqual(split_sentences("你好。今天好吗？ 好"), ["你好。", "今天好吗？", "好"])

    def test_speech_sentences_caps_the_text(self) -> None:
        sentences = speech_sentences("a. " * MAX_SPEAK_CHARS)
        self.assertLessEqual(sum(len(s) for s in sentences), MAX_SPEAK_CHARS)


class LaneTests(unittest.TestCase):
    def test_forced_lane(self) -> None:
        with mock.patch.dict(os.environ, {"CHAOSENGINE_VOICE_LANE": "portable"}):
            self.assertEqual(voice_env.lane(), "portable")

    def test_apple_silicon_is_mlx_everything_else_portable(self) -> None:
        cases = [("Darwin", "arm64", "mlx"), ("Darwin", "x86_64", "portable"), ("Windows", "AMD64", "portable"), ("Linux", "x86_64", "portable")]
        for system, machine, expected in cases:
            with mock.patch.dict(os.environ, {"CHAOSENGINE_VOICE_LANE": ""}), \
                    mock.patch("platform.system", return_value=system), \
                    mock.patch("platform.machine", return_value=machine):
                self.assertEqual(voice_env.lane(), expected, (system, machine))

    def test_packages_per_lane(self) -> None:
        mlx = voice_env.packages_for("mlx")
        self.assertTrue(any(p.startswith("parakeet-mlx") for p in mlx))
        self.assertTrue(any(p.startswith("en_core_web_sm @") for p in mlx))
        with mock.patch.object(voice_env, "has_nvidia_gpu", return_value=False):
            portable = voice_env.packages_for("portable")
        self.assertTrue(any(p.startswith("faster-whisper") for p in portable))
        self.assertFalse(any("mlx" in p for p in portable))
        self.assertFalse(any(p.startswith("nvidia-") for p in portable))
        with mock.patch.object(voice_env, "has_nvidia_gpu", return_value=True), \
                mock.patch.object(voice_env.platform, "machine", return_value="x86_64"):
            self.assertTrue(any(p.startswith("nvidia-cudnn") for p in voice_env.packages_for("portable")))
        # Linux ARM64 (DGX Spark): CTranslate2's aarch64 wheel is CPU-only, so no CUDA libraries.
        with mock.patch.object(voice_env, "has_nvidia_gpu", return_value=True), \
                mock.patch.object(voice_env.platform, "machine", return_value="aarch64"):
            self.assertFalse(any(p.startswith("nvidia-") for p in voice_env.packages_for("portable")))
        # Conversation mode's VAD runs on onnxruntime on both lanes.
        self.assertTrue(any(p.startswith("onnxruntime") for p in mlx))
        self.assertTrue(any(p.startswith("onnxruntime") for p in portable))

    def test_extras_apply_only_to_their_lane(self) -> None:
        self.assertTrue(any("misaki[zh]" in p for p in voice_env.packages_for("mlx", ["chinese"])))
        with mock.patch.object(voice_env, "has_nvidia_gpu", return_value=False):
            self.assertFalse(any("misaki" in p for p in voice_env.packages_for("portable", ["chinese"])))

    def test_venv_python_path_per_os(self) -> None:
        with mock.patch.object(voice_env.os, "name", "nt"):
            self.assertEqual(voice_env.venv_python().parts[-2:], ("Scripts", "python.exe"))
        with mock.patch.object(voice_env.os, "name", "posix"):
            self.assertEqual(voice_env.venv_python().parts[-2:], ("bin", "python"))


class BasePythonTests(unittest.TestCase):
    def test_a_normal_python_is_used_as_is(self) -> None:
        with mock.patch.object(voice_env.sys, "base_prefix", "/usr/local"):
            self.assertEqual(voice_env._base_python(), sys.executable)

    def test_the_packaged_python_is_copied_to_a_stable_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "chaosengine-embedded-runtime" / "darwin-aarch64-abc" / "python" / "home"
            (base / "bin").mkdir(parents=True)
            (base / "bin" / "python3").write_text("#!", encoding="utf-8")
            home = Path(tmp) / "home"
            with mock.patch.object(voice_env.sys, "base_prefix", str(base)), \
                    mock.patch.object(voice_env.Path, "home", return_value=home):
                first = voice_env._base_python()
                second = voice_env._base_python()
            self.assertEqual(first, second)
            self.assertTrue(first.startswith(str(home / ".chaosengine" / "voice-python")))
            self.assertTrue(Path(first).exists())


class InstalledMarkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        venv = Path(self.tmp.name) / "voice-venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        self.patches = [
            mock.patch.object(voice_env, "VOICE_VENV_DIR", venv),
            mock.patch.object(voice_env, "_MARKER", venv / "chaosengine-voice.json"),
            mock.patch.object(voice_env.os, "name", "posix"),
            mock.patch.dict(os.environ, {"CHAOSENGINE_VOICE_LANE": "portable"}),
        ]
        for patch in self.patches:
            patch.start()
        self.venv = venv

    def tearDown(self) -> None:
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    def test_needs_a_marker_for_this_lane(self) -> None:
        self.assertFalse(voice_env.is_installed())
        (self.venv / "chaosengine-voice.json").write_text(json.dumps({"schema": 1, "lane": "mlx"}), encoding="utf-8")
        self.assertFalse(voice_env.is_installed())
        (self.venv / "chaosengine-voice.json").write_text(
            json.dumps({"schema": 1, "lane": "portable", "extras": ["chinese", "bogus"]}), encoding="utf-8"
        )
        self.assertTrue(voice_env.is_installed())
        with mock.patch.object(voice_env, "kokoro_installed", return_value=False):
            self.assertEqual(voice_env.installed_extras(), ["chinese"])
        with mock.patch.object(voice_env, "kokoro_installed", return_value=True):
            self.assertEqual(voice_env.installed_extras(), ["kokoro", "chinese"])  # Kokoro from its files (D-75)

    def test_capabilities_report_when_the_pack_passed_its_check(self) -> None:
        (self.venv / "chaosengine-voice.json").write_text(
            json.dumps({"schema": 1, "lane": "portable", "installedAt": "2026-11-07T20:00:00"}), encoding="utf-8"
        )
        self.assertEqual(voice_runtime.get_voice_capabilities()["installedAt"], "2026-11-07T20:00:00")

    def test_a_corrupt_marker_is_not_installed(self) -> None:
        (self.venv / "chaosengine-voice.json").write_text("{nope", encoding="utf-8")
        self.assertFalse(voice_env.is_installed())

    def test_a_pack_from_before_wave2_offers_an_update_on_the_mlx_lane_only(self) -> None:
        marker = self.venv / "chaosengine-voice.json"
        old = {"schema": 1, "features": ["vad", "speakers"]}
        marker.write_text(json.dumps({**old, "lane": "portable"}), encoding="utf-8")
        self.assertEqual(voice_env.missing_features(), [])  # nothing for the portable lane yet
        with mock.patch.object(voice_env, "lane", return_value="mlx"):
            marker.write_text(json.dumps({**old, "lane": "mlx"}), encoding="utf-8")
            self.assertEqual(voice_env.missing_features(), ["speech-wave2"])
            marker.write_text(json.dumps({**old, "lane": "mlx", "features": voice_env.FEATURES}), encoding="utf-8")
            self.assertEqual(voice_env.missing_features(), [])
        self.assertIn("mlx-audio>=0.6.4,<0.7", voice_env.packages_for("mlx"))

    def test_vad_needs_the_marker_feature_and_the_model_file(self) -> None:
        model = self.venv / "model.onnx"  # not Path(): os.name is patched to posix
        model.write_bytes(b"onnx")
        marker = self.venv / "chaosengine-voice.json"
        marker.write_text(json.dumps({"schema": 1, "lane": "portable"}), encoding="utf-8")
        with mock.patch.object(voice_env, "vad_model_path", return_value=model):
            self.assertFalse(voice_env.vad_installed())  # a voice pack from before conversation mode
            marker.write_text(json.dumps({"schema": 1, "lane": "portable", "features": ["vad"]}), encoding="utf-8")
            self.assertTrue(voice_env.vad_installed())
        with mock.patch.object(voice_env, "vad_model_path", return_value=None):
            self.assertFalse(voice_env.vad_installed())


class CatalogTests(unittest.TestCase):
    def _rows(self, lane: str) -> list[dict]:
        with mock.patch.object(voice_env, "lane", return_value=lane), \
                mock.patch.object(voice_env, "is_installed", return_value=True), \
                mock.patch.object(voice_runtime, "_hf_snapshot_path", return_value=None):
            return voice_runtime.list_stt_models()

    def test_mlx_lane_defaults_to_parakeet(self) -> None:
        rows = self._rows("mlx")
        self.assertEqual(rows[0]["backend"], "parakeet-mlx")
        self.assertEqual([r["key"] for r in rows if r["default"]], ["parakeet-tdt-v3"])

    def test_nemotron_asr_is_offered_on_the_mlx_lane_only(self) -> None:
        mlx = {r["key"]: r for r in self._rows("mlx")}
        self.assertEqual(mlx["nemotron-3.5-asr"]["backend"], "nemotron-asr-mlx")
        self.assertEqual(mlx["nemotron-3.5-asr"]["id"], "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
        self.assertFalse(mlx["nemotron-3.5-asr"]["default"])
        self.assertFalse(mlx["nemotron-3.5-asr"]["translate"])
        self.assertNotIn("nemotron-3.5-asr", {r["key"] for r in self._rows("portable")})

    def test_nemotron_asr_resolves_by_key_and_by_repo(self) -> None:
        with mock.patch.object(voice_env, "lane", return_value="mlx"), \
                mock.patch.object(voice_env, "is_installed", return_value=True), \
                mock.patch.object(voice_runtime, "_hf_snapshot_path", return_value=None):
            expected = ("nemotron-asr-mlx", "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
            self.assertEqual(voice_runtime.resolve_stt_model("nemotron-3.5-asr"), expected)
            self.assertEqual(voice_runtime.resolve_stt_model(expected[1]), expected)

    def test_portable_lane_has_no_mlx_ids_and_defaults_to_whisper_turbo(self) -> None:
        rows = self._rows("portable")
        self.assertTrue(all(r["backend"] == "faster-whisper" for r in rows))
        self.assertFalse(any("mlx" in r["id"] for r in rows))
        self.assertEqual([r["key"] for r in rows if r["default"]], ["whisper-large-v3-turbo"])

    def test_resolution_maps_ids_across_lanes_and_refuses_paths(self) -> None:
        with mock.patch.object(voice_env, "lane", return_value="portable"), \
                mock.patch.object(voice_env, "is_installed", return_value=True), \
                mock.patch.object(voice_runtime, "_hf_snapshot_path", return_value=None):
            self.assertEqual(
                voice_runtime.resolve_stt_model("mlx-community/whisper-small-mlx"),
                ("faster-whisper", "Systran/faster-whisper-small"),
            )
            # Parakeet has no portable build: the default Whisper answers.
            self.assertEqual(voice_runtime.resolve_stt_model("parakeet-tdt-v3")[0], "faster-whisper")
            for bad in ("/etc/passwd", "../x", "someone/model"):
                with self.assertRaises(voice_runtime.UnknownSttModelError):
                    voice_runtime.resolve_stt_model(bad)

    def test_all_54_kokoro_voices_and_language_availability(self) -> None:
        with mock.patch.object(voice_env, "lane", return_value="mlx"), \
                mock.patch.object(voice_env, "installed_extras", return_value=[]):
            voices = {v["id"]: v for v in voice_runtime.list_tts_voices()}
        self.assertEqual(len(voices), 54)
        self.assertTrue(voices["af_heart"]["available"])
        self.assertTrue(voices["ef_dora"]["available"])
        self.assertFalse(voices["zf_xiaobei"]["available"])
        self.assertFalse(voices["jf_alpha"]["available"])
        self.assertEqual(voices["bm_george"]["gender"], "male")
        with mock.patch.object(voice_env, "lane", return_value="portable"):
            self.assertTrue(all(v["available"] for v in voice_runtime.list_tts_voices()))


class SettingsTests(unittest.TestCase):
    def test_round_trip_ignores_unknown_and_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(tmp)):
            self.assertEqual(voice_runtime.load_settings(), voice_runtime.DEFAULT_SETTINGS)
            saved = voice_runtime.save_settings({"ttsVoice": "bf_emma", "autoRead": True, "speed": None, "evil": 1})
            self.assertEqual(saved["ttsVoice"], "bf_emma")
            self.assertTrue(saved["autoRead"])
            self.assertEqual(saved["speed"], 1.0)
            self.assertNotIn("evil", voice_runtime.load_settings())


_FAKE_WORKER = textwrap.dedent(
    """
    import atexit, json, subprocess, sys, time
    print(json.dumps({"id": None, "ok": True, "ready": True}), flush=True)
    stubborn = False
    for line in sys.stdin:
        req = json.loads(line)
        if req["cmd"] in ("child", "stubborn-child"):
            # Like the Orpheus lane's llama-server: a child the worker cleans up in atexit.
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
            if req["cmd"] == "child":
                atexit.register(child.kill)
            else:
                stubborn = True
            print(json.dumps({"id": req["id"], "ok": True, "pid": child.pid}), flush=True)
            continue
        if req["cmd"] == "sleep":
            time.sleep(5)
        if req["cmd"] == "die":
            sys.exit(3)
        if req["cmd"] == "fail":
            print(json.dumps({"id": req["id"], "ok": False, "error": "boom"}), flush=True)
            continue
        print(json.dumps({"id": req["id"], "ok": True, "echo": req.get("value")}), flush=True)
    if stubborn:
        time.sleep(120)  # ignores stdin closing, so only the forced path can end it
    """
)


class VoiceClientTests(unittest.TestCase):
    def setUp(self) -> None:
        # Windows can hold worker.log for a moment after the process exits.
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        script = Path(self.tmp.name) / "fake_worker.py"
        script.write_text(_FAKE_WORKER, encoding="utf-8")
        self.patches = [
            mock.patch.object(voice_client, "_WORKER_SCRIPT", script),
            mock.patch.object(voice_env, "is_installed", return_value=True),
            mock.patch.object(voice_env, "venv_python", return_value=Path(sys.executable)),
            mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(self.tmp.name)),
            mock.patch.object(voice_env, "worker_env", return_value=dict(os.environ)),
        ]
        for patch in self.patches:
            patch.start()
        self.worker = voice_client.VoiceWorker()

    def tearDown(self) -> None:
        self.worker.stop()
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    def test_round_trip_and_reuse(self) -> None:
        self.assertEqual(self.worker.request("echo", value=1)["echo"], 1)
        pid = self.worker.pid
        self.assertEqual(self.worker.request("echo", value=2)["echo"], 2)
        self.assertEqual(self.worker.pid, pid)

    def test_errors_are_raised(self) -> None:
        with self.assertRaisesRegex(voice_client.VoiceWorkerError, "boom"):
            self.worker.request("fail")
        self.assertEqual(self.worker.request("echo", value=3)["echo"], 3)

    def test_a_dead_worker_is_restarted(self) -> None:
        with self.assertRaisesRegex(voice_client.VoiceWorkerError, "exited"):
            self.worker.request("die")
        self.assertIsNone(self.worker.pid)
        self.assertEqual(self.worker.request("echo", value=4)["echo"], 4)

    def test_a_dead_worker_is_reported_before_the_os_reaps_it(self) -> None:
        # Windows: stdout closes before poll() sees the exit. Was reported as
        # "did not answer within 300 s".
        self.worker.request("echo", value=1)
        with mock.patch.object(self.worker._process, "poll", return_value=None):
            with self.assertRaisesRegex(voice_client.VoiceWorkerError, "exited"):
                self.worker.request("die")

    def test_a_timeout_kills_the_worker(self) -> None:
        with self.assertRaisesRegex(voice_client.VoiceWorkerError, "did not answer"):
            self.worker.request("sleep", timeout=0.5)
        self.assertIsNone(self.worker.pid)

    @staticmethod
    def _gone(pid: int, within: float = 10.0) -> bool:
        import psutil

        deadline = time.monotonic() + within
        while time.monotonic() < deadline:
            if not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
            time.sleep(0.2)
        return False

    def test_stop_lets_the_worker_clean_up_its_children(self) -> None:
        # RTX 4090 run: Stop left Orpheus's llama-server running, because terminate() (TerminateProcess on
        # Windows) skipped the worker's atexit cleanup. Closing stdin first lets the worker exit on its own.
        pid = self.worker.request("child")["pid"]
        self.worker.stop()
        self.assertTrue(self._gone(pid), "the worker's child survived Stop")

    def test_a_worker_that_ignores_stdin_still_loses_its_children(self) -> None:
        pid = self.worker.request("stubborn-child")["pid"]
        with mock.patch.object(voice_client, "_GRACEFUL_EXIT_S", 0.5):
            self.worker.stop()
        self.assertTrue(self._gone(pid), "a forced stop left the worker's child running")

    def test_not_installed_is_reported(self) -> None:
        self.worker.stop()
        with mock.patch.object(voice_env, "is_installed", return_value=False):
            with self.assertRaisesRegex(voice_client.VoiceWorkerError, "not installed"):
                self.worker.request("echo")


class RealWorkerProtocolTests(unittest.TestCase):
    """The shipped worker, run with this interpreter, for commands that need no speech packages."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(voice_env, "is_installed", return_value=True),
            mock.patch.object(voice_env, "venv_python", return_value=Path(sys.executable)),
            mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(self.tmp.name)),
            mock.patch.object(voice_env, "worker_env", return_value=dict(os.environ)),
        ]
        for patch in self.patches:
            patch.start()
        self.worker = voice_client.VoiceWorker()

    def tearDown(self) -> None:
        self.worker.stop()
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    def test_ping_status_unload_and_unknown_command(self) -> None:
        self.assertTrue(self.worker.request("ping")["pong"])
        self.assertEqual(self.worker.request("status")["stt"], None)
        self.worker.request("unload", what="all")
        with self.assertRaisesRegex(voice_client.VoiceWorkerError, "Unknown command"):
            self.worker.request("nope")
        with self.assertRaisesRegex(voice_client.VoiceWorkerError, "Unknown speech-to-text backend"):
            self.worker.request("transcribe", backend="evil", source="x", path="y")

    def test_wav_bytes_is_valid_16_bit_mono(self) -> None:
        from backend_service.voice_worker import wav_bytes

        data = wav_bytes([0.0, 0.5, -0.5, 2.0], 24_000)
        with wave.open(io.BytesIO(data)) as handle:
            self.assertEqual((handle.getnchannels(), handle.getsampwidth(), handle.getframerate()), (1, 2, 24_000))
            self.assertEqual(handle.getnframes(), 4)


class VadCommandTests(unittest.TestCase):
    """The worker's VAD loop with a stand-in model: frame size, context and state carry-over."""

    def test_frames_context_and_sessions(self) -> None:
        import numpy as np

        from backend_service import voice_worker

        calls: list[tuple] = []

        class FakeSession:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def run(self, _outputs, feeds):
                frame, state = feeds["input"], feeds["state"]
                calls.append((frame.shape, float(frame[0, 0]), float(state[0, 0, 0])))
                return np.array([[float(frame[0, -1])]], dtype=np.float32), state + 1

        fake_ort = mock.MagicMock()
        fake_ort.InferenceSession = FakeSession
        pcm = (np.arange(1100) % 100).astype("<i2")  # two whole frames and 76 left over
        b64 = base64.b64encode(pcm.tobytes()).decode("ascii")
        voice_worker._vad.update(path=None, session=None, streams={})
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            first = voice_worker.vad({"path": "m.onnx", "session": "a", "pcm": b64})["probs"]
            second = voice_worker.vad({"path": "m.onnx", "session": "a", "pcm": b64})["probs"]
            reset = voice_worker.vad({"path": "m.onnx", "session": "a", "pcm": b64, "reset": True})["probs"]
        self.assertEqual(len(first), 2)
        self.assertEqual(first, reset)
        self.assertEqual(calls[0][0], (1, 64 + 512))
        self.assertEqual((calls[0][1], calls[0][2]), (0.0, 0.0))  # fresh stream: zero context and state
        # The second frame starts with the last 64 samples of the first.
        self.assertAlmostEqual(calls[1][1], float(pcm[512 - 64]) / 32768.0, places=5)
        self.assertEqual(calls[2][2], 2.0)  # the session's state carried into the next call
        self.assertEqual(calls[4][2], 0.0)  # reset starts over
        self.assertNotEqual(second, [])


class VoiceRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_video_routes import make_client

        self.client, self._tmp, self._snapshot = make_client()
        # A throwaway settings folder, and the Kokoro quick voices installed (an add-on since D-75).
        self._data = tempfile.TemporaryDirectory()
        self._patches = [
            mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(self._data.name)),
            mock.patch.object(voice_env, "kokoro_installed", return_value=True),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self) -> None:
        from tests.test_video_routes import restore_env

        for patch in reversed(self._patches):
            patch.stop()
        self._data.cleanup()
        restore_env(self._snapshot)
        self._tmp.cleanup()

    def test_runtime_shape(self) -> None:
        body = self.client.get("/api/voice/runtime").json()
        for key in ("lane", "installed", "sttModels", "ttsVoices", "settings", "install"):
            self.assertIn(key, body)

    def test_transcribe_rejects_empty_and_unknown_model(self) -> None:
        empty = self.client.post("/api/voice/transcribe", files={"audio": ("a.wav", b"", "audio/wav")})
        self.assertEqual(empty.status_code, 400)
        with mock.patch.object(voice_runtime, "transcribe_file", side_effect=voice_runtime.UnknownSttModelError("Unknown")):
            bad = self.client.post("/api/voice/transcribe", files={"audio": ("a.wav", b"RIFF", "audio/wav")}, data={"model": "x"})
        self.assertEqual(bad.status_code, 400)

    def test_transcribe_passes_a_temp_file_and_cleans_it_up(self) -> None:
        seen: dict = {}

        def fake(path, model, language=None, **options):
            seen.update(path=path, exists=path.exists(), model=model, language=language, data=path.read_bytes())
            self.assertEqual(options, {"task": "transcribe", "words": False, "speakers": False, "speaker_count": 0})
            return {"text": "hi", "segments": []}

        with mock.patch.object(voice_runtime, "transcribe_file", side_effect=fake):
            resp = self.client.post(
                "/api/voice/transcribe",
                files={"audio": ("clip.m4a", b"audio-bytes", "audio/mp4")},
                data={"model": "whisper-base", "language": "de"},
            )
        self.assertEqual(resp.json()["text"], "hi")
        self.assertTrue(seen["exists"])
        self.assertEqual(seen["path"].suffix, ".m4a")
        self.assertEqual((seen["model"], seen["language"], seen["data"]), ("whisper-base", "de", b"audio-bytes"))
        self.assertFalse(seen["path"].exists())

    def test_worker_failures_are_503(self) -> None:
        with mock.patch.object(voice_runtime, "transcribe_file", side_effect=voice_client.VoiceWorkerError("not installed")):
            resp = self.client.post("/api/voice/transcribe", files={"audio": ("a.wav", b"RIFF", "audio/wav")})
        self.assertEqual(resp.status_code, 503)

    def test_speak_stream_sends_one_line_per_sentence(self) -> None:
        with mock.patch.object(voice_runtime, "synthesize", side_effect=lambda text, voice, speed: (text.encode(), 1.0)):
            resp = self.client.post("/api/voice/speak/stream", json={"text": "One. Two.", "voice": "af_heart"})
        lines = [json.loads(line) for line in resp.text.splitlines()]
        self.assertEqual([line.get("text") for line in lines[:2]], ["One.", "Two."])
        self.assertEqual(base64.b64decode(lines[0]["audio"]), b"One.")
        self.assertEqual(lines[-1], {"done": True})

    def test_speak_stream_falls_back_to_kokoro_when_the_chosen_voice_cannot_speak(self) -> None:
        from backend_service import voice_chat

        voice_chat.forget_failures()
        with mock.patch.object(voice_runtime, "synthesize", side_effect=lambda text, voice, speed: (voice.encode(), 1.0)):
            resp = self.client.post("/api/voice/speak/stream", json={"text": "One. Two.", "voice": "qwen3:Ryan", "use": "call"})
        lines = [json.loads(line) for line in resp.text.splitlines()]
        self.assertIn("fallback voice Heart (Kokoro) reads instead", lines[0]["note"])
        self.assertEqual([base64.b64decode(line["audio"]) for line in lines[1:3]], [b"af_heart", b"af_heart"])
        self.assertEqual(lines[-1], {"done": True})
        self.assertEqual(self.client.post("/api/voice/speak/stream", json={"text": "Hi", "use": "shout"}).status_code, 422)

    def test_speak_stream_reports_errors_in_stream(self) -> None:
        with mock.patch.object(voice_runtime, "synthesize", side_effect=RuntimeError("no voice")):
            resp = self.client.post("/api/voice/speak/stream", json={"text": "One. Two."})
        self.assertEqual(json.loads(resp.text.splitlines()[0])["error"], "no voice")

    def test_limits_and_validation(self) -> None:
        self.assertEqual(self.client.post("/api/voice/synthesize", json={"text": "x" * 20_001}).status_code, 422)
        self.assertEqual(self.client.post("/api/voice/synthesize", json={"text": "hi", "speed": 9}).status_code, 422)
        self.assertEqual(self.client.post("/api/voice/settings", json={"idleUnloadMinutes": -1}).status_code, 422)
        self.assertEqual(self.client.post("/api/voice/install", json={"extras": ["bogus"]}).status_code, 400)

    def test_vad_route(self) -> None:
        pcm = base64.b64encode(b"\x00\x00" * 1024).decode("ascii")
        with mock.patch.object(voice_runtime, "detect_speech", return_value=[0.1, 0.9]) as detect:
            resp = self.client.post("/api/voice/vad", json={"session": "call-1", "pcm": pcm, "reset": True})
        self.assertEqual(resp.json(), {"probs": [0.1, 0.9]})
        detect.assert_called_once_with("call-1", pcm, True)
        self.assertEqual(self.client.post("/api/voice/vad", json={"session": "../x", "pcm": ""}).status_code, 422)
        self.assertEqual(self.client.post("/api/voice/vad", json={"pcm": "A" * 90_001}).status_code, 422)
        with mock.patch.object(voice_env, "vad_installed", return_value=False):
            self.assertEqual(self.client.post("/api/voice/vad", json={"pcm": pcm}).status_code, 503)

    def test_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(tmp)):
            self.assertEqual(self.client.post("/api/voice/settings", json={"autoSend": True}).json()["autoSend"], True)
            self.assertTrue(self.client.get("/api/voice/settings").json()["autoSend"])

    def test_speak_voice_is_its_own_setting_and_takes_a_cloned_voice_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(tmp)):
            fresh = self.client.get("/api/voice/settings").json()
            self.assertEqual(fresh["speakVoice"], "")
            saved = self.client.post("/api/voice/settings", json={"speakVoice": "clone:voice-1234567890abcdef"}).json()
            self.assertEqual(saved["speakVoice"], "clone:voice-1234567890abcdef")
            self.assertEqual(saved["ttsVoice"], fresh["ttsVoice"])  # the chat voice is untouched
            self.assertEqual(self.client.get("/api/voice/settings").json()["speakVoice"], "clone:voice-1234567890abcdef")
            self.assertEqual(self.client.post("/api/voice/settings", json={"speakVoice": "x" * 65}).status_code, 422)

    def test_chat_and_call_voices_take_any_listed_voice(self) -> None:
        # D-72: Chat read-aloud (ttsVoice) and calls (callVoice) take any voice list_voices returns, wave 2 included;
        # unknown ids, "Describe a voice", and cloned voices for calls are refused; old Kokoro settings keep working.
        from backend_service import voice_clone

        clone = {"id": "1234567890_deadbeef", "name": "Me", "consent": {}, "durationS": 6.0}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(tmp)), \
                mock.patch.object(voice_clone, "list_voices", return_value=[clone]):
            fresh = self.client.get("/api/voice/settings").json()
            self.assertEqual((fresh["ttsVoice"], fresh["callVoice"]), ("af_heart", ""))
            for voice in ("qwen3:Ryan", "chatterbox-turbo", "chatterbox", "orpheus:tara", "bf_emma"):
                saved = self.client.post("/api/voice/settings", json={"ttsVoice": voice, "callVoice": voice})
                self.assertEqual(saved.status_code, 200, voice)
                self.assertEqual((saved.json()["ttsVoice"], saved.json()["callVoice"]), (voice, voice))
            self.assertEqual(self.client.post("/api/voice/settings", json={"callVoice": ""}).json()["callVoice"], "")
            self.assertEqual(self.client.post("/api/voice/settings", json={"ttsVoice": "clone:1234567890_deadbeef"}).status_code, 200)
            refused = self.client.post("/api/voice/settings", json={"callVoice": "clone:1234567890_deadbeef"})
            self.assertEqual(refused.status_code, 400)
            self.assertIn("calls", refused.json()["detail"])
            for bad in ("qwen3:Nobody", "nonsense", "design", "clone:0000000000_00000000"):
                self.assertEqual(self.client.post("/api/voice/settings", json={"ttsVoice": bad}).status_code, 400, bad)
            self.assertEqual(self.client.post("/api/voice/settings", json={"ttsVoice": "x" * 65}).status_code, 422)
            # An agent never sees cloned voices, so it cannot make one the chat voice.
            with mock.patch("backend_service.mcp_bridge.access.token_ok", return_value=True):
                agent = self.client.post("/api/voice/settings", json={"ttsVoice": "clone:1234567890_deadbeef"},
                                         headers={"x-chaosengine-token": "agent-token"})
            self.assertEqual(agent.status_code, 400)
            self.assertEqual(self.client.get("/api/voice/settings").json()["ttsVoice"], "clone:1234567890_deadbeef")


if __name__ == "__main__":
    unittest.main()


class NemotronLanguageTests(unittest.TestCase):
    """The UI sends ISO codes; Nemotron prompts by locale key."""

    class _Model:
        prompt_dictionary = {"auto": 0, "en-US": 1, "de-DE": 2, "pt-BR": 3, "zh-CN": 4}

    def test_iso_codes_map_onto_locale_keys(self) -> None:
        from backend_service import voice_worker

        model = self._Model()
        self.assertEqual(voice_worker._nemotron_language(model, "de"), "de-DE")
        self.assertEqual(voice_worker._nemotron_language(model, "PT"), "pt-BR")
        self.assertEqual(voice_worker._nemotron_language(model, "en-US"), "en-US")

    def test_missing_or_unknown_language_is_auto_detected(self) -> None:
        from backend_service import voice_worker

        model = self._Model()
        self.assertEqual(voice_worker._nemotron_language(model, None), "auto")
        self.assertEqual(voice_worker._nemotron_language(model, "xx"), "auto")
        self.assertEqual(voice_worker._nemotron_language(object(), "de"), "auto")


class CudaDllPathTests(unittest.TestCase):
    """Windows: ctranslate2 loads cuBLAS with plain LoadLibrary, which only searches PATH (RTX 4090 run)."""

    def test_nvidia_bin_folders_go_on_path_as_well_as_the_dll_directories(self) -> None:
        from backend_service import voice_worker

        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "nvidia" / "cublas" / "bin"
            bin_dir.mkdir(parents=True)
            added: list[str] = []
            with mock.patch.object(voice_worker.os, "name", "nt"), \
                 mock.patch.object(voice_worker.os, "add_dll_directory", side_effect=added.append, create=True), \
                 mock.patch.object(voice_worker.sys, "path", [tmp]), \
                 mock.patch.dict(voice_worker.os.environ, {"PATH": "C:/Windows"}, clear=False):
                voice_worker._add_cuda_dll_dirs()
                voice_worker._add_cuda_dll_dirs()  # a second call must not stack duplicates
                path = voice_worker.os.environ["PATH"].split(os.pathsep)
        self.assertIn(str(bin_dir), added)
        self.assertEqual(path[0], str(bin_dir))
        self.assertEqual(path.count(str(bin_dir)), 1)
        self.assertIn("C:/Windows", path)

    def test_nothing_changes_off_windows(self) -> None:
        from backend_service import voice_worker

        with mock.patch.object(voice_worker.os, "name", "posix"), \
             mock.patch.dict(voice_worker.os.environ, {"PATH": "/usr/bin"}, clear=False):
            voice_worker._add_cuda_dll_dirs()
            self.assertEqual(voice_worker.os.environ["PATH"], "/usr/bin")
