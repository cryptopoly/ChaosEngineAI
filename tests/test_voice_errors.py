"""Voice error codes (voice studio UX audit, change 6): every known worker or setup failure maps to a stable code,
one plain sentence and an action hint; the routes keep their HTTP status and ``detail`` and add the three fields.

The worker is stubbed: no server, no model and no GPU run here.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service import voice_client, voice_clone, voice_clone_safety, voice_env, voice_errors, voice_runtime, voice_speak
from backend_service.voice_errors import ACTIONS, classify


def worker(text: str) -> voice_client.VoiceWorkerError:
    return voice_client.VoiceWorkerError(text)


# (raw text as the code raises it, code, action, extra)
CASES = [
    ("The voice runtime is not installed. Install it from Voice > Models.", "runtime_missing", "install-voice-runtime", None),
    ("Speaker labels need a voice runtime update. Update it from Voice > Models.", "runtime_outdated", "install-voice-runtime", None),
    ("Conversation mode needs a voice runtime update. Update it from Voice > Models.", "runtime_outdated", "install-voice-runtime", None),
    (voice_runtime.KOKORO_MISSING, "addon_missing", "install-addon", "kokoro"),
    (voice_clone.NOT_HERE, "addon_missing", "install-addon", "speech-gpu"),
    ("Preset and designed voices need an Apple Silicon Mac, or the speech GPU add-on with an NVIDIA GPU. Use a built-in voice here.",
     "addon_missing", "install-addon", "speech-gpu"),
    ('The speech GPU add-on has no Qwen3-TTS package yet. Untick and tick "More speech models on the NVIDIA GPU" under '
     "Voice runtime > Add-ons to update it.", "addon_missing", "install-addon", "speech-gpu"),
    ('Transformers 4.1 cannot run this model. Untick and tick "More speech-to-text models on the NVIDIA GPU" under '
     "Voice runtime > Add-ons to update it.", "addon_missing", "install-addon", "stt-gpu"),
    ("VoiceWorkerError: ModuleNotFoundError: No module named 'chatterbox'", "addon_missing", "install-addon", "speech-gpu"),
    ("Mandarin Chinese voices need an extra install on this machine.", "addon_missing", "install-addon", "chinese"),
    ("Qwen3-TTS 1.7B is not downloaded. Download it in Voice > Models.", "model_missing", "download-model", None),
    ("Parakeet is not downloaded. Download it under Voice > Models > Dictation models.", "model_missing", "download-model", None),
    (voice_clone_safety.WATERMARK_MISSING, "watermark_missing", "download-model", None),
    ("Qwen3-TTS 1.7B needs about 8 GB of GPU memory; this GPU has 6 GB.", "gpu_memory", "smaller-model", None),
    ("OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", "gpu_memory", "smaller-model", None),
    ("Chatterbox needs an NVIDIA GPU of compute capability 7 or newer; this one is 6.1.", "gpu_unsupported", "pick-another-model", None),
    ("Qwen3-TTS runs on Apple Silicon Macs only, so it cannot be used on this computer.", "unsupported_here", "pick-another-model", None),
    ("Orpheus is not offered on this version yet.", "unsupported_here", "pick-another-model", None),
    ("Changing speed or pitch for this voice needs ffmpeg. Install it from Settings, or leave both at their defaults.",
     "ffmpeg_missing", "install-ffmpeg", None),
    ("FileNotFoundError: [WinError 2] No such file or directory: 'ffmpeg'", "ffmpeg_missing", "install-ffmpeg", None),
    ("Qwen3-TTS needs a voice sample of at least 5 s; this one is 3.2 s. Pick another cloning model or record a longer sample.",
     "reference_too_short", "change-sample", None),
    ("The voice sample must be 3 to 30 seconds long; this one is 1.4 s.", "reference_too_short", "change-sample", None),
    ("The voice sample must be 3 to 30 seconds long; this one is 41.0 s.", "reference_too_long", "change-sample", None),
    ("No speech was heard in the voice sample.", "reference_unreadable", "change-sample", None),
    ("The voice sample must be a WAV, MP3, M4A, FLAC or OGG file.", "reference_unreadable", "change-sample", None),
    ("This voice's sample is missing. Delete the voice and make it again.", "reference_unreadable", "change-sample", None),
    ("ValueError: The file has no audio track.", "audio_unreadable", "change-sample", None),
    ("av.error.InvalidDataError: [Errno 1094995529] Invalid data found when processing input", "audio_unreadable", "change-sample", None),
    ("Chatterbox speaks en, de, fr; 'xx' is not one of them.", "language_unsupported", "pick-another-language", None),
    ("ValueError: Unsupported language 'xx' for Qwen3-ASR.", "language_unsupported", "pick-another-language", None),
    (voice_clone.PERMISSION_REQUIRED, "permission_required", "confirm-permission", None),
    ("Confirm that this is your own voice or that the speaker gave permission.", "permission_required", "confirm-permission", None),
    (voice_clone_safety.REFUSED_MESSAGE, "cloning_not_allowed", "none", None),
    ("Cloned voice not found. Its permission record is missing or it was deleted.", "voice_not_found", "none", None),
    ("Downloading Qwen3-TTS failed: 401 Client Error: Unauthorized for url: https://huggingface.co/x", "download_blocked", "add-hf-token", None),
    ("Downloading Chatterbox failed: Cannot access gated repo for url https://huggingface.co/y", "download_blocked", "add-hf-token", None),
    ("Downloading Orpheus failed: HTTPSConnectionPool(host='huggingface.co'): Max retries exceeded", "download_failed", "retry", None),
    ("Another voice is speaking. Try again when it has finished.", "worker_busy", "retry", None),
    ("Another transcription is running. Try again when it has finished.", "worker_busy", "retry", None),
    ("The voice worker did not answer within 300 s.", "worker_timeout", "retry", None),
    ("The voice worker exited. See C:\\Users\\x\\.chaosengine\\voice\\worker.log.", "worker_stopped", "retry", None),
    ("The voice worker did not start. See C:\\Users\\x\\worker.log.", "worker_stopped", "retry", None),
    ("The voice worker stopped: [Errno 32] Broken pipe", "worker_stopped", "retry", None),
    ("KeyError: 'something nobody planned for'", "unknown", "retry", None),
]


class ClassifyTests(unittest.TestCase):
    def test_every_known_failure_maps_to_its_code_action_and_extra(self) -> None:
        for raw, code, action, extra in CASES:
            with self.subTest(raw=raw[:70]):
                info = classify(raw)
                self.assertEqual((info.code, info.action, info.extra), (code, action, extra))

    def test_every_code_has_one_plain_sentence_and_a_listed_action(self) -> None:
        for code, (message, action) in voice_errors._TABLE.items():
            with self.subTest(code=code):
                self.assertIn(action, ACTIONS)
                self.assertTrue(message.endswith("."))
                self.assertEqual(message.count(". "), 0, "one sentence")
                self.assertNotIn("Voice > Models", message)  # the message names no screen path; the button does the work

    def test_accepts_an_exception_and_empty_text(self) -> None:
        self.assertEqual(classify(worker("The voice runtime is not installed.")).code, "runtime_missing")
        self.assertEqual(classify("").code, "unknown")
        self.assertEqual(classify(None).code, "unknown")

    def test_an_unknown_client_error_has_no_retry_button(self) -> None:
        self.assertEqual(classify("Unknown voice 'x'.", 400).action, "none")
        self.assertEqual(classify("Unknown voice 'x'.", 503).action, "retry")

    def test_fields_shape(self) -> None:
        self.assertEqual(set(voice_errors.error_fields("x")), {"code", "message", "action", "extra"})

    def test_the_real_gpu_problem_sentences_map(self) -> None:
        row = {"name": "Big model"}
        with mock.patch("backend_service.voice_gpu.facts", return_value={"kind": "memory", "need": 8.0, "have": 6.0}):
            from backend_service import voice_gpu

            self.assertEqual(classify(voice_gpu.problem(row)).code, "gpu_memory")
        with mock.patch("backend_service.voice_gpu.facts", return_value={"kind": "compute", "need": 8.0, "have": 6.1}):
            from backend_service import voice_gpu

            self.assertEqual(classify(voice_gpu.problem(row)).code, "gpu_unsupported")


class RouteErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_video_routes import make_client

        self.client, self._tmp, self._snapshot = make_client()
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

    def assertVoiceError(self, resp, status: int, code: str, action: str, detail_has: str) -> dict:
        self.assertEqual(resp.status_code, status, resp.text)
        body = resp.json()
        self.assertEqual((body["code"], body["action"]), (code, action))
        self.assertIsInstance(body["detail"], str)  # as before: the raw text, for callers that read ``detail``
        self.assertIn(detail_has, body["detail"])
        self.assertTrue(body["message"])
        self.assertNotEqual(body["message"], body["detail"])
        self.assertIn("extra", body)
        return body

    def transcribe(self, error: Exception):
        with mock.patch.object(voice_runtime, "transcribe_file", side_effect=error):
            return self.client.post("/api/voice/transcribe", files={"audio": ("a.wav", b"RIFF", "audio/wav")})

    def speak(self, error: Exception, voice: str = "af_heart", **extra):
        with mock.patch.object(voice_speak, "speak", side_effect=error):
            return self.client.post("/api/voice/speak", json={"text": "Hello", "voice": voice, **extra})

    def test_transcribe_worker_failures_keep_503_and_gain_codes(self) -> None:
        self.assertVoiceError(self.transcribe(worker("The voice runtime is not installed. Install it from Voice > Models.")),
                              503, "runtime_missing", "install-voice-runtime", "not installed")
        self.assertVoiceError(self.transcribe(worker("Parakeet is not downloaded. Download it under Voice > Models > Dictation models.")),
                              503, "model_missing", "download-model", "Parakeet")
        self.assertVoiceError(self.transcribe(worker("OutOfMemoryError: CUDA out of memory.")),
                              503, "gpu_memory", "smaller-model", "out of memory")
        self.assertVoiceError(self.transcribe(worker("The voice worker exited. See worker.log.")),
                              503, "worker_stopped", "retry", "exited")
        self.assertVoiceError(self.transcribe(worker("The voice worker did not answer within 300 s.")),
                              503, "worker_timeout", "retry", "300 s")
        unknown = self.assertVoiceError(self.transcribe(worker("KeyError: 'x'")), 503, "unknown", "retry", "KeyError")
        self.assertIsNone(unknown["extra"])

    def test_transcribe_bad_audio_keeps_400(self) -> None:
        body = self.assertVoiceError(self.transcribe(worker("InvalidDataError: Invalid data found when processing input")),
                                     400, "audio_unreadable", "change-sample", "could not be read as audio")
        self.assertIn("WAV, MP3", body["detail"])
        self.assertVoiceError(self.transcribe(ValueError("Unsupported language 'xx' for Qwen3-ASR.")),
                              400, "language_unsupported", "pick-another-language", "Unsupported language")

    def test_add_on_missing_names_the_tick_box(self) -> None:
        body = self.assertVoiceError(self.speak(RuntimeError(voice_runtime.KOKORO_MISSING)),
                                     503, "addon_missing", "install-addon", "Kokoro")
        self.assertEqual(body["extra"], "kokoro")
        body = self.assertVoiceError(self.speak(ValueError(voice_clone.NOT_HERE)), 400, "addon_missing", "install-addon", "Apple Silicon")
        self.assertEqual(body["extra"], "speech-gpu")

    def test_speak_errors(self) -> None:
        self.assertVoiceError(self.speak(RuntimeError("Qwen3-TTS 1.7B is not downloaded. Download it in Voice > Models.")),
                              503, "model_missing", "download-model", "not downloaded")
        self.assertVoiceError(self.speak(worker("RuntimeError: CUDA out of memory. Tried to allocate 3 GiB")),
                              503, "gpu_memory", "smaller-model", "out of memory")
        self.assertVoiceError(self.speak(RuntimeError("Changing speed or pitch for this voice needs ffmpeg. Install it from Settings.")),
                              503, "ffmpeg_missing", "install-ffmpeg", "ffmpeg")
        self.assertVoiceError(self.speak(ValueError("Chatterbox speaks en, de; 'xx' is not one of them."), language="xx"),
                              400, "language_unsupported", "pick-another-language", "not one of them")
        self.assertVoiceError(self.speak(LookupError("Cloned voice not found. Its permission record is missing or it was deleted."),
                                         voice="clone:abc"),
                              404, "voice_not_found", "none", "not found")

    def test_cloning_refusals(self) -> None:
        with mock.patch.object(voice_clone_safety, "clone_allowed", return_value=False):
            refused = self.client.post("/api/voice/speak", json={"text": "Hello", "voice": "clone:abc"})
        self.assertVoiceError(refused, 403, "cloning_not_allowed", "none", "Cloned voices can only be used")
        no_tick = self.client.post(
            "/api/voice/clone/voices",
            files={"sample": ("s.wav", b"RIFF", "audio/wav")},
            data={"speakerName": "Dan", "basis": "own", "permissionConfirmed": "false", "name": "Me"},
        )
        self.assertVoiceError(no_tick, 400, "permission_required", "confirm-permission", "permission")
        with mock.patch.object(voice_clone, "speak", side_effect=ValueError(voice_clone.PERMISSION_REQUIRED)):
            unconfirmed = self.client.post("/api/voice/clone/speak", json={"voiceId": "abc", "text": "Hi"})
        self.assertVoiceError(unconfirmed, 400, "permission_required", "confirm-permission", "permission_confirmed")

    def test_clone_sample_lengths(self) -> None:
        fields = {"speakerName": "Dan", "basis": "own", "permissionConfirmed": "true", "name": "Me"}
        for seconds, code in ((1.2, "reference_too_short"), (45.0, "reference_too_long")):
            with self.subTest(seconds=seconds), mock.patch.object(voice_runtime, "transcribe_file",
                                                                  return_value={"text": "x", "durationS": seconds}):
                resp = self.client.post("/api/voice/clone/voices", files={"sample": ("s.wav", b"RIFF", "audio/wav")}, data=fields)
            self.assertVoiceError(resp, 400, code, "change-sample", "voice sample")

    def test_stream_failure_line_carries_the_code(self) -> None:
        with mock.patch.object(voice_runtime, "synthesize", side_effect=RuntimeError("OutOfMemoryError: CUDA out of memory.")):
            resp = self.client.post("/api/voice/speak/stream", json={"text": "One. Two."})
        line = json.loads(resp.text.splitlines()[0])
        self.assertEqual((line["code"], line["action"]), ("gpu_memory", "smaller-model"))
        self.assertIn("out of memory", line["error"])  # the raw text, as before
        self.assertEqual(line["index"], 0)

    def test_a_download_failure_carries_its_code_in_the_model_row(self) -> None:
        rows = voice_clone.list_models(every_model=True)
        key = rows[0]["key"]
        with mock.patch.dict(voice_clone._DOWNLOADS, {key: {"state": "failed", "error": "Downloading X failed: 403 Forbidden, gated repo"}}):
            row = next(r for r in voice_clone.list_models(every_model=True) if r["key"] == key)
        self.assertEqual(row["download"]["errorInfo"]["code"], "download_blocked")
        self.assertIsNone(next(r for r in rows if r["key"] == key)["download"]["errorInfo"])


if __name__ == "__main__":
    unittest.main()
