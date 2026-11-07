"""Voice cloning: consent check, AI tag, watermark hook, audit log, agent guard, store, worker branch, routes, agent tool."""

from __future__ import annotations

import base64
import io
import json
import re
import sys
import tempfile
import time
import types
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend_service import voice_clone, voice_clone_consent, voice_clone_safety, voice_env, voice_library, voice_runtime
from backend_service.catalog import voice_models


def make_wav(seconds: float = 0.1, rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x01\x00" * int(seconds * rate))
    return buffer.getvalue()


def fake_request(path: str = "/api/voice/clone/speak", host: str = "127.0.0.1", headers: dict | None = None):
    return SimpleNamespace(url=SimpleNamespace(path=path), headers=headers or {}, client=SimpleNamespace(host=host))


class _DataDir(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data = Path(self._tmp.name)
        self._patch = mock.patch.object(voice_env, "VOICE_DATA_DIR", self.data)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def audit_lines(self) -> list[dict]:
        path = voice_clone_safety.audit_path()
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class ConsentTests(unittest.TestCase):
    def test_spoken_digits_reads_digits_and_number_words(self) -> None:
        for said in ("my code is 4 7 2 1", "four seven two one", "forty-seven twenty one", "4721",
                     "four thousand seven hundred twenty one"):
            self.assertIn("4721", voice_clone_consent.spoken_digits(said), said)
        self.assertEqual(voice_clone_consent.spoken_digits("one oh five"), "105")

    def test_verify_needs_code_and_agreement_words(self) -> None:
        good = "I, Dan, agree that my voice may be copied by AI on this computer. My code is four seven two one."
        self.assertIsNone(voice_clone_consent.verify(good, "4721"))
        self.assertIn("4 7 2 1", voice_clone_consent.verify(good.replace("four", "five"), "4721"))
        self.assertIn("words", voice_clone_consent.verify("my code is 4 7 2 1", "4721"))

    def test_challenge_is_random_single_use_and_expires(self) -> None:
        a = voice_clone_consent.start("  Dan   Smith ", "own")
        b = voice_clone_consent.start("Dan", "permission")
        self.assertRegex(a["code"], r"^\d{4}$")
        self.assertNotEqual(a["id"], b["id"])
        self.assertEqual(a["speakerName"], "Dan Smith")
        self.assertIn(" ".join(a["code"]), a["sentence"])
        self.assertIsNotNone(voice_clone_consent.peek(a["id"]))
        voice_clone_consent.consume(a["id"])
        self.assertIsNone(voice_clone_consent.peek(a["id"]))
        with mock.patch.object(time, "time", return_value=time.time() + voice_clone_consent.CHALLENGE_TTL_S + 1):
            self.assertIsNone(voice_clone_consent.peek(b["id"]))
        with self.assertRaises(ValueError):
            voice_clone_consent.start("", "own")
        with self.assertRaises(ValueError):
            voice_clone_consent.start("Dan", "celebrity")


class CatalogTests(unittest.TestCase):
    def test_rows_are_pinned_and_apache(self) -> None:
        self.assertEqual({r["key"] for r in voice_models.CLONE_MODELS}, {"qwen3-tts-0.6b", "qwen3-tts-1.7b"})
        for row in voice_models.CLONE_MODELS:
            self.assertRegex(row["revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(row["repo"].startswith("mlx-community/Qwen3-TTS-12Hz-") and row["repo"].endswith("-Base-bf16"))
        self.assertEqual(voice_models.CLONE_LICENCE["id"], "apache-2.0")
        self.assertEqual(sum(r["default"] for r in voice_models.CLONE_MODELS), 1)
        self.assertEqual(voice_models.clone_model("")["key"], "qwen3-tts-0.6b")
        with self.assertRaises(ValueError):
            voice_models.clone_model("/tmp/some/local/path")


class SafetyTests(_DataDir):
    def test_tag_round_trip_keeps_the_audio(self) -> None:
        raw = make_wav()
        tagged = voice_clone_safety.tag_wav(raw, {"model": "m", "machine_generated": False, "clonedVoice": False}, "ChaosEngineAI 1")
        tag = voice_clone_safety.read_wav_tag(tagged)
        self.assertEqual((tag["machine_generated"], tag["clonedVoice"], tag["model"]), (True, True, "m"))
        with wave.open(io.BytesIO(tagged)) as handle:
            frames = handle.readframes(handle.getnframes())
        with wave.open(io.BytesIO(raw)) as handle:
            self.assertEqual(frames, handle.readframes(handle.getnframes()))
        # Re-tagging replaces the old tag rather than adding a second one.
        twice = voice_clone_safety.tag_wav(tagged, {"model": "n"}, "x")
        self.assertEqual(twice.count(b"chaosengine:"), 1)
        self.assertEqual(voice_clone_safety.read_wav_tag(twice)["model"], "n")
        self.assertIsNone(voice_clone_safety.read_wav_tag(raw))
        self.assertIsNone(voice_clone_safety.read_wav_tag(b"not a wav"))

    def test_apply_watermark_refuses_without_the_component_and_calls_the_worker(self) -> None:
        path = self.data / "a.wav"
        path.write_bytes(make_wav())
        with mock.patch.object(voice_env, "watermark_model_paths", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "watermark"):
                voice_clone_safety.apply_watermark(path)
        worker = mock.Mock()
        worker.request.return_value = {"probability": 0.99, "message": "CE5A", "matches": True, "seconds": 0.1}
        models = {"generator": Path("/g.pth"), "detector": Path("/d.pth")}
        with mock.patch.object(voice_env, "watermark_model_paths", return_value=models), \
             mock.patch.object(voice_env, "watermark_installed", return_value=True), \
             mock.patch("backend_service.voice_client.get_worker", return_value=worker):
            self.assertTrue(voice_clone_safety.apply_watermark(path))
            self.assertEqual(worker.request.call_args.args[0], "watermark")
            self.assertEqual(worker.request.call_args.kwargs["generator"], "/g.pth")
            found = voice_clone_safety.detect_watermark(path)
        self.assertEqual(found, {"probability": 0.99, "message": "CE5A", "matches": True})
        self.assertEqual(worker.request.call_args.args[0], "detectWatermark")

    def test_audit_appends_json_lines_locally(self) -> None:
        voice_clone_safety.audit("speak", voiceId="1_abcdef12", textLength=5)
        voice_clone_safety.audit("delete", voiceId="1_abcdef12")
        lines = self.audit_lines()
        self.assertEqual([l["event"] for l in lines], ["speak", "delete"])
        self.assertTrue(str(voice_clone_safety.audit_path()).startswith(str(self.data)))

    def test_request_origin_and_clone_allowed(self) -> None:
        self.assertEqual(voice_clone_safety.request_origin(fake_request()), "ui")
        self.assertEqual(voice_clone_safety.request_origin(fake_request(path="/mcp")), "agent")
        self.assertEqual(voice_clone_safety.request_origin(fake_request(host="192.168.1.9")), "remote")
        with mock.patch("backend_service.mcp_bridge.access.token_ok", return_value=True):
            agent = fake_request(headers={"authorization": "Bearer agent-token"})
            self.assertEqual(voice_clone_safety.request_origin(agent), "agent")
        with mock.patch("backend_service.mcp_bridge.access.token_ok", return_value=False):
            self.assertEqual(voice_clone_safety.request_origin(fake_request(headers={"authorization": "Bearer app"})), "ui")
        self.assertTrue(voice_clone_safety.clone_allowed(fake_request()))
        self.assertFalse(voice_clone_safety.clone_allowed(fake_request(path="/mcp")))
        self.assertFalse(voice_clone_safety.clone_allowed(fake_request(host="10.0.0.2")))
        voice_runtime.save_settings({"allowAgentCloning": True})
        self.assertTrue(voice_clone_safety.clone_allowed(fake_request(path="/mcp")))
        self.assertFalse(voice_runtime.DEFAULT_SETTINGS["allowAgentCloning"])


class StoreTests(_DataDir):
    def _challenge(self) -> dict:
        return voice_clone_consent.start("Dan", "own")

    def _create(self, challenge: dict, heard: str | None = None, duration: float = 8.0, **kwargs):
        consent_file = self.data / "consent.webm"
        consent_file.write_bytes(b"consent-audio")
        said = heard if heard is not None else f"I Dan agree that my voice may be copied. My code is {' '.join(challenge['code'])}"
        transcribe = mock.Mock(return_value={"text": said, "durationS": duration})
        args = dict(challenge_id=challenge["id"], name="My voice", consent_audio=consent_file, consent_ext=".webm",
                    reference_audio=None, reference_ext=None, reference_text=None, statement_accepted=True,
                    origin="ui", transcribe=transcribe)
        args.update(kwargs)
        return voice_clone.create_voice(**args)

    def test_create_stores_consent_with_the_voice_and_delete_removes_all(self) -> None:
        challenge = self._challenge()
        voice = self._create(challenge)
        item = voice_library.get(voice["id"])
        record = item["consent"]
        self.assertEqual((item["kind"], record["speakerName"], record["basis"], record["code"]), ("clone", "Dan", "own", challenge["code"]))
        self.assertRegex(record["referenceSha256"], r"^[0-9a-f]{64}$")
        self.assertIn("appVersion", record)
        consent_copy = voice_library.library_dir() / f"{voice['id']}.consent.webm"
        self.assertEqual(consent_copy.read_bytes(), b"consent-audio")
        self.assertEqual(voice_library.audio_path(voice["id"]).read_bytes(), b"consent-audio")
        # Not in the normal Library list; listed by kind.
        self.assertEqual(voice_library.list_items(), [])
        self.assertEqual([v["id"] for v in voice_clone.list_voices()], [voice["id"]])
        # The challenge is single use.
        with self.assertRaises(ValueError):
            self._create(challenge)
        self.assertTrue(voice_clone.delete_voice(voice["id"]))
        self.assertEqual(list(voice_library.library_dir().iterdir()), [])
        self.assertEqual([l["event"] for l in self.audit_lines()], ["create", "delete"])

    def test_mismatch_and_bad_samples_are_refused_plainly(self) -> None:
        with self.assertRaisesRegex(ValueError, "code"):
            self._create(self._challenge(), heard="I agree my voice may be copied, code 0000")
        with self.assertRaisesRegex(ValueError, "3 to 30 seconds"):
            self._create(self._challenge(), duration=1.5)
        with self.assertRaisesRegex(ValueError, "3 to 30 seconds"):
            self._create(self._challenge(), duration=45)
        with self.assertRaisesRegex(ValueError, "permission"):
            self._create(self._challenge(), statement_accepted=False)
        with self.assertRaisesRegex(ValueError, "expired"):
            self._create({"id": "nope", "code": "1234"})
        self.assertEqual(voice_clone.list_voices(), [])
        self.assertIn("consent-rejected", [l["event"] for l in self.audit_lines()])

    def test_library_refuses_a_clone_without_consent(self) -> None:
        with self.assertRaises(ValueError):
            voice_library.create("clone", {"text": "x"})

    def test_speak_tags_saves_and_logs(self) -> None:
        voice = self._create(self._challenge())
        snapshot = self.data / "snap"
        worker = mock.Mock()
        worker.request.return_value = {"wavB64": base64.b64encode(make_wav()).decode(), "durationS": 0.1, "seconds": 1.0}
        with mock.patch.object(voice_clone, "snapshot_dir", return_value=snapshot), \
             mock.patch.object(voice_clone, "available_here", return_value=True), \
             mock.patch.object(voice_env, "is_installed", return_value=True), \
             mock.patch.object(voice_env, "watermark_installed", return_value=True), \
             mock.patch("backend_service.voice_client.get_worker", return_value=worker), \
             mock.patch.object(voice_clone_safety, "apply_watermark", return_value=True) as mark:
            item = voice_clone.speak(voice["id"], "Hello there.", "", disclaimer=True, origin="ui")
        kwargs = worker.request.call_args.kwargs
        self.assertEqual(kwargs["source"], str(snapshot))
        self.assertTrue(kwargs["text"].startswith(voice_clone.DISCLAIMER_LINE))
        self.assertEqual(kwargs["refText"], voice_library.get(voice["id"])["text"])
        self.assertNotIn("voice", kwargs)
        mark.assert_called_once()
        self.assertEqual((item["kind"], item["clonedVoice"], item["cloneVoiceId"]), ("speech", True, voice["id"]))
        tag = voice_clone_safety.read_wav_tag(voice_library.audio_path(item["id"]))
        self.assertTrue(tag["machine_generated"] and tag["clonedVoice"])
        self.assertEqual(tag["model"], voice_models.clone_model("")["repo"])
        spoken = [l for l in self.audit_lines() if l["event"] == "speak"][0]
        self.assertEqual((spoken["textLength"], spoken["voiceId"]), (len("Hello there."), voice["id"]))
        self.assertRegex(spoken["outputSha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("Hello", json.dumps(spoken))

    def test_speak_needs_a_stored_voice_and_the_model(self) -> None:
        with self.assertRaises(LookupError):
            voice_clone.speak("1234567890_deadbeef", "Hi", "", disclaimer=False, origin="ui")
        voice = self._create(self._challenge())
        with mock.patch.object(voice_clone, "snapshot_dir", return_value=None), \
             mock.patch.object(voice_clone, "available_here", return_value=True), \
             mock.patch.object(voice_env, "is_installed", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "not downloaded"):
                voice_clone.speak(voice["id"], "Hi", "", disclaimer=False, origin="ui")
        # No watermark component, no cloned speech: there is no way to switch it off.
        with mock.patch.object(voice_clone, "snapshot_dir", return_value=self.data), \
             mock.patch.object(voice_clone, "available_here", return_value=True), \
             mock.patch.object(voice_env, "is_installed", return_value=True), \
             mock.patch.object(voice_env, "watermark_installed", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "watermark"):
                voice_clone.speak(voice["id"], "Hi", "", disclaimer=False, origin="ui")


class DownloadTests(unittest.TestCase):
    def _run(self, install_error: Exception | None):
        class SyncThread:
            def __init__(self, target, **_kwargs):
                self.target = target

            def start(self):
                self.target()

        install = mock.Mock(side_effect=install_error)
        with mock.patch.object(voice_clone, "available_here", return_value=True), \
             mock.patch("huggingface_hub.snapshot_download") as weights, \
             mock.patch.object(voice_env, "watermark_installed", return_value=False), \
             mock.patch.object(voice_env, "install_watermark", install), \
             mock.patch.object(voice_clone.threading, "Thread", SyncThread):
            voice_clone._DOWNLOADS.clear()
            voice_clone.start_download("qwen3-tts-0.6b")
        row = voice_models.clone_model("qwen3-tts-0.6b")
        weights.assert_called_once_with(row["repo"], revision=row["revision"])
        install.assert_called_once()
        return voice_clone._DOWNLOADS["qwen3-tts-0.6b"]

    def test_download_installs_the_watermark_component_automatically(self) -> None:
        self.assertEqual(self._run(None)["state"], "done")

    def test_a_failed_component_install_says_so_plainly(self) -> None:
        job = self._run(RuntimeError("pip exited 1"))
        self.assertEqual(job["state"], "failed")
        self.assertTrue(job["error"].startswith("Downloading the watermark component failed"), job)


class WorkerTests(unittest.TestCase):
    def test_synthesize_routes_reference_audio_to_the_clone_branch(self) -> None:
        from backend_service import voice_worker

        import numpy as np

        with mock.patch.object(voice_worker, "_load_tts", return_value=object()), \
             mock.patch.object(voice_worker, "synthesize_cloned", return_value=(np.zeros(2400, dtype=np.float32), 24000)) as clone:
            reply = voice_worker.synthesize({"engine": "mlx-audio", "source": "/snap", "text": "Hi",
                                             "refAudioPath": "/ref.wav", "refText": "sample words"})
        clone.assert_called_once()
        self.assertEqual(clone.call_args.args[1:], ("Hi", "/ref.wav", "sample words"))
        self.assertEqual(reply["durationS"], 0.1)
        with mock.patch.object(voice_worker, "_load_tts", return_value=object()):
            with self.assertRaises(ValueError):
                voice_worker.synthesize({"engine": "kokoro-onnx", "source": "/k", "text": "Hi", "refAudioPath": "/r.wav"})

    def test_clone_branch_passes_ref_audio_and_text_and_no_voice(self) -> None:
        from backend_service import voice_worker

        import numpy as np

        seen = {}

        class Model:
            sample_rate = 24000

            def generate(self, text, **kwargs):
                seen.update(kwargs, text=text)
                yield SimpleNamespace(audio=np.ones(10, dtype=np.float32))

        fake_mx = types.ModuleType("mlx.core")
        fake_mx.array = lambda value: value
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_mx
        with mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_mx}), \
             mock.patch.object(voice_worker, "decode_audio", return_value=np.zeros(5, dtype=np.float32)) as decode:
            samples, rate = voice_worker.synthesize_cloned(Model(), "Hello", "/ref.wav", "words")
        decode.assert_called_once_with("/ref.wav", 24000)
        self.assertEqual((seen["text"], seen["ref_text"], rate, len(samples)), ("Hello", "words", 24000, 10))
        self.assertNotIn("voice", seen)


class RouteTests(unittest.TestCase):
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

    def test_models_listing(self) -> None:
        body = self.client.get("/api/voice/clone/models").json()
        self.assertEqual(len(body["models"]), 2)
        self.assertTrue(body["allowedHere"])
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", m["revision"]) for m in body["models"]))

    def test_consent_and_create_through_the_api(self) -> None:
        challenge = self.client.post("/api/voice/clone/consent", json={"speakerName": "Dan", "basis": "own"}).json()
        self.assertEqual(set(challenge), {"id", "code", "sentence", "speakerName", "basis"})
        self.assertEqual(self.client.post("/api/voice/clone/consent", json={"speakerName": "Dan", "basis": "star"}).status_code, 422)
        said = {"text": f"I agree that my voice may be copied. My code is {' '.join(challenge['code'])}.", "durationS": 6.0}
        wrong = {"text": "I agree my voice may be copied. My code is 1 1 1 1.", "durationS": 6.0}
        files = {"consent": ("consent.wav", make_wav(), "audio/wav")}
        with mock.patch.object(voice_runtime, "transcribe_file", return_value=wrong):
            refused = self.client.post("/api/voice/clone/voices", files=files,
                                       data={"challengeId": challenge["id"], "statementAccepted": "true"})
        self.assertEqual(refused.status_code, 400)
        self.assertIn("code", refused.json()["detail"])
        with mock.patch.object(voice_runtime, "transcribe_file", return_value=said):
            no_statement = self.client.post("/api/voice/clone/voices", files=files, data={"challengeId": challenge["id"]})
            created = self.client.post("/api/voice/clone/voices", files=files,
                                       data={"challengeId": challenge["id"], "statementAccepted": "true", "name": "Me"})
        self.assertEqual(no_statement.status_code, 400)
        self.assertEqual(created.status_code, 200, created.text)
        voice = created.json()
        self.assertEqual((voice["name"], voice["consent"]["speakerName"]), ("Me", "Dan"))
        self.assertEqual([v["id"] for v in self.client.get("/api/voice/clone/voices").json()["voices"]], [voice["id"]])
        self.assertEqual(self.client.delete(f"/api/voice/clone/voices/{voice['id']}").json(), {"deleted": True})
        self.assertEqual(self.client.delete(f"/api/voice/clone/voices/{voice['id']}").status_code, 404)

    def test_agents_are_refused_until_the_person_allows_it(self) -> None:
        with mock.patch("backend_service.mcp_bridge.access.token_ok", return_value=True):
            headers = {"x-chaosengine-token": "agent-token"}
            speak = {"voiceId": "1234567890_deadbeef", "text": "Hi"}
            self.assertEqual(self.client.post("/api/voice/clone/speak", json=speak, headers=headers).status_code, 403)
            self.assertEqual(self.client.post("/api/voice/clone/consent", json={"speakerName": "A", "basis": "own"}, headers=headers).status_code, 403)
            self.assertEqual(self.client.get("/api/voice/clone/voices", headers=headers).status_code, 403)
            # An agent cannot grant itself cloning.
            self.assertEqual(self.client.post("/api/voice/settings", json={"allowAgentCloning": True}, headers=headers).status_code, 403)
        self.assertTrue(self.client.post("/api/voice/settings", json={"allowAgentCloning": True}).json()["allowAgentCloning"])
        with mock.patch("backend_service.mcp_bridge.access.token_ok", return_value=True):
            self.assertEqual(self.client.get("/api/voice/clone/voices", headers=headers).status_code, 200)
        events = [json.loads(l)["event"] for l in voice_clone_safety.audit_path().read_text().splitlines()]
        self.assertIn("refused", events)
        self.assertIn("setting", events)

    def test_speak_route_maps_errors(self) -> None:
        body = {"voiceId": "1234567890_deadbeef", "text": "Hi"}
        self.assertEqual(self.client.post("/api/voice/clone/speak", json=body).status_code, 404)
        with mock.patch.object(voice_clone, "speak", return_value={"id": "1_abcdef12", "kind": "speech", "clonedVoice": True}):
            self.assertTrue(self.client.post("/api/voice/clone/speak", json=body).json()["clonedVoice"])

    def test_mp3_of_cloned_speech_carries_the_tag(self) -> None:
        tagged = voice_clone_safety.tag_wav(make_wav(), {"model": "m"}, "ChaosEngineAI")
        item = self.client.post("/api/voice/library/speech", files={"audio": ("s.wav", tagged, "audio/wav")}).json()
        self.assertTrue(item["clonedVoice"])
        seen = {}

        def fake_encode(source, target, comment=None):
            seen["comment"] = comment
            target.write_bytes(b"ID3")

        with mock.patch.object(voice_runtime, "encode_mp3", side_effect=fake_encode):
            self.client.get(f"/api/voice/library/{item['id']}/audio", params={"format": "mp3"})
        self.assertTrue(seen["comment"].startswith("chaosengine:"))
        self.assertTrue(json.loads(seen["comment"][len("chaosengine:"):])["clonedVoice"])


class AgentToolTests(_DataDir):
    def test_tools_refuse_until_allowed_then_submit(self) -> None:
        from backend_service.mcp_bridge import voice_clone_tool
        from backend_service.mcp_bridge.tools import ToolError

        ctx = SimpleNamespace(request=fake_request(path="/mcp"))
        with self.assertRaises(ToolError):
            voice_clone_tool.list_cloned_voices(ctx, {})
        voice_runtime.save_settings({"allowAgentCloning": True})
        self.assertEqual(voice_clone_tool.list_cloned_voices(ctx, {}), {"voices": []})
        with self.assertRaises(ToolError):
            voice_clone_tool.speak_cloned(ctx, {"voice_id": "1234567890_deadbeef", "text": "Hi"})
        with mock.patch.object(voice_clone, "get_voice", return_value={"id": "1234567890_deadbeef"}), \
             mock.patch.object(voice_clone_tool.STORE, "submit", return_value=SimpleNamespace(describe=lambda: {"jobId": "job-1"})) as submit:
            reply = voice_clone_tool.speak_cloned(ctx, {"voice_id": "1234567890_deadbeef", "text": "Hi"})
        self.assertEqual(reply["jobId"], "job-1")
        self.assertEqual(submit.call_args.args[0], "voice")
        self.assertEqual({t["name"] for t in voice_clone_tool.CLONE_TOOLS}, set(voice_clone_tool.CLONE_HANDLERS))

    def test_tool_schemas_validate(self) -> None:
        from backend_service.mcp_bridge import tools, voice_clone_tool

        schema = voice_clone_tool.CLONE_TOOLS[1]["inputSchema"]
        self.assertIsNone(tools.validate(schema, {"voice_id": "x", "text": "Hi"}))
        self.assertIsNotNone(tools.validate(schema, {"text": "Hi"}))


if __name__ == "__main__":
    unittest.main()
