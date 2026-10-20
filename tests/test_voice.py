"""Voice I/O tests — runtime helpers, gallery persistence, API routes.

The runtime unit tests fake the heavy backends (mlx_audio / mlx_whisper)
via ``sys.modules`` injection so no audio stack is required; route tests
patch the ``backend_service.voice_runtime`` source module because the
handlers lazy-import inside the function body.
"""

from __future__ import annotations

import array
import base64
import io
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

# Pre-import numpy: the kokoro-onnx synth imports it lazily, and
# ``mock.patch.dict(sys.modules)`` would evict it on exit and trip
# numpy's "cannot load module more than once per process" guard.
import numpy  # noqa: F401

from backend_service import voice_gallery, voice_kokoro_onnx, voice_runtime

from tests.test_video_routes import make_client, restore_env


# ---------------------------------------------------------------------------
# voice_runtime unit tests
# ---------------------------------------------------------------------------

class CapabilitiesTests(unittest.TestCase):
    def test_capabilities_shape(self) -> None:
        caps = voice_runtime.get_voice_capabilities()
        for key in (
            "sttAvailable", "ttsAvailable", "platform", "sttBackend",
            "ttsBackend", "sttInstallPackages", "ttsInstallPackages",
            "ttsModelRepo", "ttsModelInstalled",
        ):
            self.assertIn(key, caps)

    def test_install_packages_none_when_available(self) -> None:
        with mock.patch.object(voice_runtime, "_is_available", return_value=True):
            caps = voice_runtime.get_voice_capabilities()
        self.assertIsNone(caps["sttInstallPackages"])
        self.assertIsNone(caps["ttsInstallPackages"])

    def test_install_packages_apple_silicon(self) -> None:
        with mock.patch.object(voice_runtime, "_platform", return_value="apple_silicon"):
            self.assertEqual(voice_runtime._stt_install_packages(), ["mlx-whisper", "parakeet-mlx"])
            tts = voice_runtime._tts_install_packages()
        self.assertEqual(tts[0], "mlx-audio")
        self.assertIn("misaki", tts)
        self.assertIn("unidic-lite", tts)

    def test_install_packages_other_platforms(self) -> None:
        with mock.patch.object(voice_runtime, "_platform", return_value="windows"):
            self.assertEqual(voice_runtime._stt_install_packages(), ["faster-whisper"])
            self.assertEqual(voice_runtime._tts_install_packages(), ["kokoro-onnx"])

    def test_mime_to_suffix(self) -> None:
        self.assertEqual(voice_runtime._mime_to_suffix("audio/wav"), ".wav")
        self.assertEqual(voice_runtime._mime_to_suffix("audio/webm;codecs=opus"), ".webm")
        self.assertEqual(voice_runtime._mime_to_suffix("application/unknown"), ".webm")


class _FakeMlxAudio:
    """Installs a fake ``mlx_audio.tts.generate`` module tree while active."""

    def __init__(self, generate_audio) -> None:
        self._generate_audio = generate_audio
        self._saved: dict[str, types.ModuleType | None] = {}

    def __enter__(self) -> "_FakeMlxAudio":
        pkg = types.ModuleType("mlx_audio")
        tts = types.ModuleType("mlx_audio.tts")
        gen = types.ModuleType("mlx_audio.tts.generate")
        gen.generate_audio = self._generate_audio
        pkg.tts = tts
        tts.generate = gen
        for name, module in (("mlx_audio", pkg), ("mlx_audio.tts", tts), ("mlx_audio.tts.generate", gen)):
            self._saved[name] = sys.modules.get(name)
            sys.modules[name] = module
        return self

    def __exit__(self, *exc_info: object) -> None:
        for name, original in self._saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class SynthesizeSpeechTests(unittest.TestCase):
    def setUp(self) -> None:
        self._platform = mock.patch.object(
            voice_runtime, "_platform", return_value="apple_silicon"
        )
        self._available = mock.patch.object(
            voice_runtime, "_is_available", side_effect=lambda name: name == "mlx_audio"
        )
        self._platform.start()
        self._available.start()

    def tearDown(self) -> None:
        self._platform.stop()
        self._available.stop()

    def test_happy_path_returns_wav_bytes(self) -> None:
        calls: dict[str, object] = {}

        def fake_generate(**kwargs: object) -> None:
            calls.update(kwargs)
            out = Path(str(kwargs["output_path"])) / f"{kwargs['file_prefix']}.wav"
            out.write_bytes(b"RIFF-fake-wav")

        with _FakeMlxAudio(fake_generate):
            result = voice_runtime.synthesize_speech("hello", "af_heart", 1.0)
        self.assertEqual(result, b"RIFF-fake-wav")
        self.assertEqual(calls["voice"], "af_heart")
        self.assertEqual(calls["lang_code"], "a")

    def test_lang_code_derived_from_voice_prefix(self) -> None:
        calls: dict[str, object] = {}

        def fake_generate(**kwargs: object) -> None:
            calls.update(kwargs)
            out = Path(str(kwargs["output_path"])) / f"{kwargs['file_prefix']}.wav"
            out.write_bytes(b"x")

        with _FakeMlxAudio(fake_generate):
            voice_runtime.synthesize_speech("テスト", "jf_alpha", 1.0)
        self.assertEqual(calls["lang_code"], "j")

    def test_segment_filename_fallback(self) -> None:
        def fake_generate(**kwargs: object) -> None:
            out = Path(str(kwargs["output_path"])) / f"{kwargs['file_prefix']}_000.wav"
            out.write_bytes(b"segment-bytes")

        with _FakeMlxAudio(fake_generate):
            result = voice_runtime.synthesize_speech("hello", "af_heart", 1.0)
        self.assertEqual(result, b"segment-bytes")

    def test_no_output_file_raises(self) -> None:
        def fake_generate(**kwargs: object) -> None:
            return None  # upstream swallowed an internal error

        with _FakeMlxAudio(fake_generate):
            with self.assertRaises(RuntimeError):
                voice_runtime.synthesize_speech("hello", "af_heart", 1.0)


class TranscribeAudioTests(unittest.TestCase):
    def test_apple_silicon_mlx_whisper_path(self) -> None:
        fake_whisper = types.ModuleType("mlx_whisper")
        fake_whisper.transcribe = lambda path, path_or_hf_repo: {"text": "  hi there  "}
        with mock.patch.object(voice_runtime, "_platform", return_value="apple_silicon"), \
             mock.patch.object(voice_runtime, "_is_available", side_effect=lambda n: n == "mlx_whisper"), \
             mock.patch.dict(sys.modules, {"mlx_whisper": fake_whisper}):
            text = voice_runtime.transcribe_audio(b"fake-audio", "audio/wav", "mlx-community/whisper-base-mlx")
        self.assertEqual(text, "hi there")

    def test_no_backend_raises(self) -> None:
        with mock.patch.object(voice_runtime, "_platform", return_value="apple_silicon"), \
             mock.patch.object(voice_runtime, "_is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                voice_runtime.transcribe_audio(b"fake-audio", "audio/wav", "mlx-community/whisper-base-mlx")


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

    def test_synthesize_raises_when_files_missing(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            voice_kokoro_onnx.synthesize("hi", "af_heart", 1.0)
        self.assertIn("not downloaded", str(ctx.exception))

    def test_synthesize_constructs_kokoro_with_file_paths_and_lang(self) -> None:
        self._install_files()
        calls: dict[str, object] = {}

        class FakeKokoro:
            def __init__(self, model_path: str, voices_path: str) -> None:
                calls["model_path"] = model_path
                calls["voices_path"] = voices_path

            def create(self, text: str, voice: str, speed: float, lang: str):
                calls["lang"] = lang
                return [0.0, 0.5, -1.0, 2.0], 24000

        fake_pkg = types.ModuleType("kokoro_onnx")
        fake_pkg.Kokoro = FakeKokoro
        # ``soundfile: None`` makes any import of it raise — regression for
        # the lane depending on a package kokoro-onnx doesn't pull in.
        with mock.patch.dict(sys.modules, {"kokoro_onnx": fake_pkg, "soundfile": None}):
            result = voice_kokoro_onnx.synthesize("hello", "bm_george", 1.0)
        with wave.open(io.BytesIO(result), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 24000)
            frames = array.array("h", wav.readframes(wav.getnframes()))
        # Float samples scale to int16 and out-of-range values clip.
        self.assertEqual(list(frames), [0, 16383, -32767, 32767])
        self.assertTrue(str(calls["model_path"]).endswith("kokoro-v1.0.onnx"))
        self.assertTrue(str(calls["voices_path"]).endswith("voices-v1.0.bin"))
        self.assertEqual(calls["lang"], "en-gb")

    def test_lang_mapping_by_voice_prefix(self) -> None:
        self.assertEqual(voice_kokoro_onnx._LANG_BY_PREFIX.get("a"), "en-us")
        self.assertEqual(voice_kokoro_onnx._LANG_BY_PREFIX.get("b"), "en-gb")
        self.assertEqual(voice_kokoro_onnx._LANG_BY_PREFIX.get("j"), "ja")

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

    def test_runtime_delegates_non_apple_tts(self) -> None:
        with mock.patch.object(voice_runtime, "_platform", return_value="windows"), \
             mock.patch.object(voice_runtime, "_is_available", side_effect=lambda n: n == "kokoro_onnx"), \
             mock.patch.object(voice_kokoro_onnx, "synthesize", return_value=b"wav") as synth:
            result = voice_runtime.synthesize_speech("hi", "af_heart", 1.0)
        self.assertEqual(result, b"wav")
        synth.assert_called_once_with("hi", "af_heart", 1.0)

    def test_capabilities_report_kokoro_files_on_non_apple(self) -> None:
        with mock.patch.object(voice_runtime, "_platform", return_value="windows"):
            caps = voice_runtime.get_voice_capabilities()
        self.assertIsNone(caps["ttsModelRepo"])
        self.assertFalse(caps["ttsModelInstalled"])
        self._install_files()
        with mock.patch.object(voice_runtime, "_platform", return_value="windows"):
            caps = voice_runtime.get_voice_capabilities()
        self.assertTrue(caps["ttsModelInstalled"])


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

class VoiceRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client, self._tempdir, self._env = make_client()
        self._gallery_tempdir = tempfile.TemporaryDirectory()
        self._gallery_patch = mock.patch.object(
            voice_gallery, "_gallery_dir", return_value=Path(self._gallery_tempdir.name)
        )
        self._gallery_patch.start()

    def tearDown(self) -> None:
        self._gallery_patch.stop()
        self._gallery_tempdir.cleanup()
        self.client.close()
        restore_env(self._env)
        self._tempdir.cleanup()

    def test_runtime_endpoint_shape(self) -> None:
        response = self.client.get("/api/voice/runtime")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        for key in ("sttAvailable", "ttsAvailable", "sttModels", "ttsVoices", "ttsModelRepo"):
            self.assertIn(key, payload)
        self.assertIsInstance(payload["sttModels"], list)
        self.assertIsInstance(payload["ttsVoices"], list)

    def test_synthesize_rejects_empty_text(self) -> None:
        response = self.client.post("/api/voice/synthesize", json={"text": "   "})
        self.assertEqual(response.status_code, 400)

    def test_synthesize_rejects_unknown_voice(self) -> None:
        response = self.client.post(
            "/api/voice/synthesize", json={"text": "hi", "voice": "zz_nope"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown voice", response.json()["detail"])

    def test_synthesize_happy_path_returns_wav(self) -> None:
        with mock.patch.object(voice_runtime, "synthesize_speech", return_value=b"RIFF-wav"):
            response = self.client.post(
                "/api/voice/synthesize", json={"text": "hi", "voice": "af_heart"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, b"RIFF-wav")

    def test_synthesize_maps_runtime_error_to_503(self) -> None:
        with mock.patch.object(
            voice_runtime, "synthesize_speech", side_effect=RuntimeError("no backend")
        ):
            response = self.client.post(
                "/api/voice/synthesize", json={"text": "hi", "voice": "af_heart"}
            )
        self.assertEqual(response.status_code, 503)

    def test_transcribe_rejects_empty_upload(self) -> None:
        response = self.client.post(
            "/api/voice/transcribe", files={"audio": ("a.wav", b"", "audio/wav")}
        )
        self.assertEqual(response.status_code, 400)

    def test_transcribe_happy_path(self) -> None:
        with mock.patch.object(voice_runtime, "transcribe_audio", return_value="hello"):
            response = self.client.post(
                "/api/voice/transcribe", files={"audio": ("a.wav", b"data", "audio/wav")}
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["text"], "hello")
        self.assertIn("duration_s", payload)

    def test_gallery_roundtrip(self) -> None:
        saved = self.client.post(
            "/api/voice/gallery/transcript", json={"text": "route test"}
        )
        self.assertEqual(saved.status_code, 200)
        item_id = saved.json()["id"]

        listed = self.client.get("/api/voice/gallery")
        self.assertEqual([i["id"] for i in listed.json()["items"]], [item_id])

        deleted = self.client.delete(f"/api/voice/gallery/{item_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get("/api/voice/gallery").json()["items"], [])

    def test_gallery_audio_roundtrip(self) -> None:
        encoded = base64.b64encode(b"RIFF-audio").decode("ascii")
        saved = self.client.post(
            "/api/voice/gallery/audio",
            json={"text": "t", "voice": "af_heart", "audioBase64": encoded},
        )
        self.assertEqual(saved.status_code, 200)
        item_id = saved.json()["id"]

        audio = self.client.get(f"/api/voice/gallery/{item_id}/audio")
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(audio.content, b"RIFF-audio")

    def test_gallery_rejects_invalid_audio_payloads(self) -> None:
        bad_b64 = self.client.post(
            "/api/voice/gallery/audio",
            json={"text": "t", "voice": "v", "audioBase64": "!!not-base64!!"},
        )
        self.assertEqual(bad_b64.status_code, 400)
        empty = self.client.post(
            "/api/voice/gallery/audio",
            json={"text": "t", "voice": "v", "audioBase64": ""},
        )
        self.assertEqual(empty.status_code, 400)

    def test_gallery_rejects_invalid_ids(self) -> None:
        response = self.client.delete("/api/voice/gallery/not-a-valid-id")
        self.assertEqual(response.status_code, 400)
        missing = self.client.delete("/api/voice/gallery/1234567890_deadbeef")
        self.assertEqual(missing.status_code, 404)


# ---------------------------------------------------------------------------
# Extended coverage — platform detection, backend fallbacks, cache paths
# ---------------------------------------------------------------------------

class PlatformDetectionTests(unittest.TestCase):
    def _detect(self, system: str, machine: str) -> str:
        with mock.patch.object(voice_runtime.platform, "system", return_value=system), \
             mock.patch.object(voice_runtime.platform, "machine", return_value=machine):
            return voice_runtime._platform()

    def test_platform_mapping(self) -> None:
        self.assertEqual(self._detect("Darwin", "arm64"), "apple_silicon")
        self.assertEqual(self._detect("Darwin", "x86_64"), "macos_x86")
        self.assertEqual(self._detect("Windows", "AMD64"), "windows")
        self.assertEqual(self._detect("Linux", "x86_64"), "linux")
        self.assertEqual(self._detect("Linux", "aarch64"), "linux")


class BackendSelectionTests(unittest.TestCase):
    def _patch(self, plat: str, available: set[str]):
        return (
            mock.patch.object(voice_runtime, "_platform", return_value=plat),
            mock.patch.object(voice_runtime, "_is_available", side_effect=lambda n: n in available),
        )

    def _run(self, plat: str, available: set[str], fn):
        p1, p2 = self._patch(plat, available)
        with p1, p2:
            return fn()

    def test_apple_silicon_prefers_mlx_backends(self) -> None:
        avail = {"mlx_whisper", "mlx_audio", "faster_whisper", "kokoro_onnx"}
        self.assertEqual(self._run("apple_silicon", avail, voice_runtime._stt_backend), "mlx-whisper")
        self.assertEqual(self._run("apple_silicon", avail, voice_runtime._tts_backend), "mlx-audio")

    def test_apple_silicon_falls_back_to_portable_backends(self) -> None:
        avail = {"faster_whisper", "kokoro_onnx"}
        self.assertEqual(self._run("apple_silicon", avail, voice_runtime._stt_backend), "faster-whisper")
        self.assertEqual(self._run("apple_silicon", avail, voice_runtime._tts_backend), "kokoro-onnx")

    def test_non_apple_ignores_mlx_modules(self) -> None:
        avail = {"mlx_whisper", "mlx_audio"}
        self.assertIsNone(self._run("linux", avail, voice_runtime._stt_backend))
        self.assertIsNone(self._run("linux", avail, voice_runtime._tts_backend))
        self.assertFalse(self._run("linux", avail, voice_runtime._stt_available))
        self.assertFalse(self._run("linux", avail, voice_runtime._tts_available))

    def test_availability_tracks_platform_backend(self) -> None:
        self.assertTrue(self._run("apple_silicon", {"mlx_whisper"}, voice_runtime._stt_available))
        self.assertFalse(self._run("apple_silicon", {"faster_whisper"}, voice_runtime._stt_available))
        self.assertTrue(self._run("windows", {"kokoro_onnx"}, voice_runtime._tts_available))

    def test_capabilities_list_install_packages_when_missing(self) -> None:
        caps = self._run("windows", set(), voice_runtime.get_voice_capabilities)
        self.assertFalse(caps["sttAvailable"])
        self.assertEqual(caps["sttInstallPackages"], ["faster-whisper"])
        self.assertEqual(caps["ttsInstallPackages"], ["kokoro-onnx"])
        self.assertIsNone(caps["sttBackend"])
        self.assertEqual(caps["platform"], "windows")

    def test_capabilities_report_tts_repo_on_apple_silicon(self) -> None:
        p1, p2 = self._patch("apple_silicon", {"mlx_audio"})
        with p1, p2, mock.patch.object(voice_runtime, "_hf_snapshot_path", return_value=Path("/snap")):
            caps = voice_runtime.get_voice_capabilities()
        self.assertEqual(caps["ttsModelRepo"], voice_runtime._TTS_MODEL_REPO)
        self.assertTrue(caps["ttsModelInstalled"])


def _make_snapshot(hub: Path, repo_id: str, revision: str = "abc123") -> Path:
    repo_dir = hub / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(revision, encoding="utf-8")
    return snapshot


class HfSnapshotPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.hub = Path(self._tempdir.name) / "custom-hub"
        self._env = mock.patch.dict(
            "os.environ", {"HF_HUB_CACHE": str(self.hub)}, clear=False
        )
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        self._tempdir.cleanup()

    def test_honours_relocated_hf_cache(self) -> None:
        # Regression: the path used to be hardcoded to ~/.cache/huggingface,
        # so a relocated cache (Settings → hfCachePath → HF_HOME) always
        # reported voice models as not installed.
        snapshot = _make_snapshot(self.hub, "prince-canuma/Kokoro-82M")
        self.assertEqual(voice_runtime._hf_snapshot_path("prince-canuma/Kokoro-82M"), snapshot)

    def test_missing_repo_returns_none(self) -> None:
        self.assertIsNone(voice_runtime._hf_snapshot_path("nobody/nothing"))

    def test_list_stt_models_installed_flags(self) -> None:
        _make_snapshot(self.hub, "mlx-community/whisper-small-mlx")
        with mock.patch.object(voice_runtime, "_platform", return_value="apple_silicon"):
            models = {m["id"]: m for m in voice_runtime.list_stt_models()}
        self.assertTrue(models["mlx-community/whisper-small-mlx"]["installed"])
        self.assertFalse(models["mlx-community/whisper-base-mlx"]["installed"])

    def test_non_apple_rows_are_ctranslate2_repos(self) -> None:
        # FU-092: Windows/Linux must never be offered MLX weights.
        _make_snapshot(self.hub, "Systran/faster-whisper-small")
        with mock.patch.object(voice_runtime, "_platform", return_value="linux"):
            models = {m["id"]: m for m in voice_runtime.list_stt_models()}
        self.assertEqual(
            set(models),
            {
                "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
                "Systran/faster-whisper-small",
                "Systran/faster-whisper-base",
            },
        )
        self.assertTrue(models["Systran/faster-whisper-small"]["installed"])
        self.assertTrue(all(m["backend"] == "faster-whisper" for m in models.values()))
        self.assertTrue(models["mobiuslabsgmbh/faster-whisper-large-v3-turbo"]["default"])


class CatalogInvariantTests(unittest.TestCase):
    def test_exactly_one_default_stt_model(self) -> None:
        defaults = [m for m in voice_runtime._STT_MODELS if m["default"]]
        self.assertEqual(len(defaults), 1)

    def test_every_voice_prefix_has_a_kokoro_onnx_lang(self) -> None:
        # Both TTS lanes key language off the voice id's first letter; a
        # voice added to the catalog without a lang mapping would silently
        # fall back to en-us G2P on Windows/Linux.
        for voice in voice_runtime.list_tts_voices():
            self.assertIn(voice["id"][0], voice_kokoro_onnx._LANG_BY_PREFIX, voice["id"])
            self.assertIn(voice["id"][0], "abefhipjz", voice["id"])

    def test_install_packages_are_all_allowlisted(self) -> None:
        # The Voice Models "Install" button posts each key to
        # /api/setup/install-package, which 400s on anything missing from
        # the allowlist — so every key must be installable on every platform.
        from backend_service.routes.setup import _INSTALLABLE_PIP_PACKAGES

        for plat in ("apple_silicon", "windows", "linux", "macos_x86"):
            with mock.patch.object(voice_runtime, "_platform", return_value=plat):
                keys = voice_runtime._stt_install_packages() + voice_runtime._tts_install_packages()
            for key in keys:
                self.assertIn(key, _INSTALLABLE_PIP_PACKAGES, f"{plat}: {key}")

    def test_list_tts_voices_returns_a_copy(self) -> None:
        voices = voice_runtime.list_tts_voices()
        voices.clear()
        self.assertTrue(voice_runtime.list_tts_voices())


class SynthesizeSpeechEdgeTests(unittest.TestCase):
    def _capture(self, voice: str) -> dict[str, object]:
        calls: dict[str, object] = {}

        def fake_generate(**kwargs: object) -> None:
            calls.update(kwargs)
            (Path(str(kwargs["output_path"])) / "tts.wav").write_bytes(b"x")

        with mock.patch.object(voice_runtime, "_platform", return_value="apple_silicon"), \
             mock.patch.object(voice_runtime, "_is_available", side_effect=lambda n: n == "mlx_audio"), \
             _FakeMlxAudio(fake_generate):
            voice_runtime.synthesize_speech("hello", voice, 1.25)
        return calls

    def test_unknown_prefix_falls_back_to_american_english(self) -> None:
        self.assertEqual(self._capture("xx_custom")["lang_code"], "a")

    def test_empty_voice_does_not_crash(self) -> None:
        # Regression: ``"" in "abefhipjz"`` is True, so an empty voice id
        # used to reach ``voice[0]`` and raise IndexError.
        self.assertEqual(self._capture("")["lang_code"], "a")

    def test_forwards_generation_arguments(self) -> None:
        calls = self._capture("bf_emma")
        self.assertEqual(calls["model"], voice_runtime._TTS_MODEL_REPO)
        self.assertEqual(calls["speed"], 1.25)
        self.assertEqual(calls["lang_code"], "b")
        self.assertEqual(calls["audio_format"], "wav")
        self.assertTrue(calls["join_audio"])

    def test_apple_silicon_without_mlx_audio_uses_kokoro_onnx(self) -> None:
        with mock.patch.object(voice_runtime, "_platform", return_value="apple_silicon"), \
             mock.patch.object(voice_runtime, "_is_available", side_effect=lambda n: n == "kokoro_onnx"), \
             mock.patch.object(voice_kokoro_onnx, "synthesize", return_value=b"onnx-wav"):
            self.assertEqual(voice_runtime.synthesize_speech("hi", "af_heart", 1.0), b"onnx-wav")

    def test_no_backend_raises_runtime_error(self) -> None:
        with mock.patch.object(voice_runtime, "_platform", return_value="linux"), \
             mock.patch.object(voice_runtime, "_is_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                voice_runtime.synthesize_speech("hi", "af_heart", 1.0)
        self.assertIn("No TTS backend", str(ctx.exception))


class TranscribeAudioEdgeTests(unittest.TestCase):
    def _fake_mlx_whisper(self, seen: dict[str, object], result: object):
        module = types.ModuleType("mlx_whisper")

        def transcribe(path: str, path_or_hf_repo: str):
            seen["path"] = path
            seen["repo"] = path_or_hf_repo
            seen["existed"] = Path(path).is_file()
            seen["bytes"] = Path(path).read_bytes()
            if isinstance(result, Exception):
                raise result
            return result

        module.transcribe = transcribe
        return module

    def _apple(self):
        return (
            mock.patch.object(voice_runtime, "_platform", return_value="apple_silicon"),
            mock.patch.object(voice_runtime, "_is_available", side_effect=lambda n: n == "mlx_whisper"),
        )

    def test_tempfile_has_mime_suffix_and_is_removed(self) -> None:
        seen: dict[str, object] = {}
        p1, p2 = self._apple()
        with p1, p2, mock.patch.dict(sys.modules, {"mlx_whisper": self._fake_mlx_whisper(seen, {"text": "ok"})}):
            voice_runtime.transcribe_audio(b"payload", "audio/ogg; codecs=opus", "mlx-community/whisper-base-mlx")
        self.assertTrue(seen["existed"])
        self.assertEqual(seen["bytes"], b"payload")
        self.assertTrue(str(seen["path"]).endswith(".ogg"))
        self.assertEqual(seen["repo"], "mlx-community/whisper-base-mlx")
        self.assertFalse(Path(str(seen["path"])).exists())

    def test_tempfile_removed_when_backend_raises(self) -> None:
        seen: dict[str, object] = {}
        p1, p2 = self._apple()
        fake = self._fake_mlx_whisper(seen, ValueError("decode failed"))
        with p1, p2, mock.patch.dict(sys.modules, {"mlx_whisper": fake}):
            with self.assertRaises(ValueError):
                voice_runtime.transcribe_audio(b"payload", "audio/wav", "mlx-community/whisper-base-mlx")
        self.assertFalse(Path(str(seen["path"])).exists())

    def test_missing_text_returns_empty_string(self) -> None:
        p1, p2 = self._apple()
        with p1, p2, mock.patch.dict(sys.modules, {"mlx_whisper": self._fake_mlx_whisper({}, {"text": None})}):
            self.assertEqual(voice_runtime.transcribe_audio(b"x", "audio/wav", "mlx-community/whisper-base-mlx"), "")

    def test_faster_whisper_path_joins_segments(self) -> None:
        seen: dict[str, object] = {}

        class FakeSegment:
            def __init__(self, text: str) -> None:
                self.text = text

        class FakeWhisperModel:
            def __init__(self, model: str, device: str, compute_type: str) -> None:
                seen["model"] = model
                seen["device"] = device

            def transcribe(self, path: str):
                return iter([FakeSegment(" hello"), FakeSegment(" world ")]), {"language": "en"}

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = FakeWhisperModel
        with mock.patch.object(voice_runtime, "_platform", return_value="linux"), \
             mock.patch.object(voice_runtime, "_is_available", side_effect=lambda n: n == "faster_whisper"), \
             mock.patch.dict(sys.modules, {"faster_whisper": fake}):
            text = voice_runtime.transcribe_audio(b"x", "audio/webm", "small")
        # Regression: " ".join doubled the separator ("hello  world").
        self.assertEqual(text, "hello world")
        # FU-092: the size token resolves to the CTranslate2 repo
        # faster-whisper itself maps "small" to — never an MLX id.
        self.assertEqual(seen["model"], "Systran/faster-whisper-small")
        self.assertEqual(seen["device"], "auto")


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

    def test_synthesize_unknown_prefix_uses_en_us_in_memory(self) -> None:
        for name, _url in voice_kokoro_onnx._FILES:
            (self.root / name).write_bytes(b"fake")
        seen: dict[str, str] = {}

        class FakeKokoro:
            def __init__(self, *_args: str) -> None:
                pass

            def create(self, text: str, voice: str, speed: float, lang: str):
                seen["lang"] = lang
                return [0.0], 22050

        fake_pkg = types.ModuleType("kokoro_onnx")
        fake_pkg.Kokoro = FakeKokoro
        with mock.patch.dict(sys.modules, {"kokoro_onnx": fake_pkg}):
            result = voice_kokoro_onnx.synthesize("hi", "zz_voice", 1.0)
        self.assertEqual(seen["lang"], "en-us")
        self.assertTrue(result.startswith(b"RIFF"))
        # Encoded in memory — nothing written next to the model files.
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()),
            sorted(name for name, _url in voice_kokoro_onnx._FILES),
        )


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


class VoiceRouteEdgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client, self._tempdir, self._env = make_client()
        self._gallery_tempdir = tempfile.TemporaryDirectory()
        self._gallery_patch = mock.patch.object(
            voice_gallery, "_gallery_dir", return_value=Path(self._gallery_tempdir.name)
        )
        self._gallery_patch.start()

    def tearDown(self) -> None:
        self._gallery_patch.stop()
        self._gallery_tempdir.cleanup()
        self.client.close()
        restore_env(self._env)
        self._tempdir.cleanup()

    def test_models_endpoint_shape(self) -> None:
        payload = self.client.get("/api/voice/models").json()
        self.assertEqual(set(payload), {"sttModels", "ttsVoices"})
        self.assertEqual(len(payload["sttModels"]), len(voice_runtime.list_stt_models()))
        self.assertIn("af_heart", {v["id"] for v in payload["ttsVoices"]})

    def test_transcribe_forwards_mime_and_model(self) -> None:
        with mock.patch.object(voice_runtime, "transcribe_audio", return_value="ok") as fn:
            self.client.post(
                "/api/voice/transcribe",
                files={"audio": ("a.ogg", b"bytes", "audio/ogg")},
                data={"model": "mlx-community/whisper-small-mlx"},
            )
        fn.assert_called_once_with(b"bytes", "audio/ogg", "mlx-community/whisper-small-mlx")

    def test_transcribe_defaults_to_platform_default(self) -> None:
        # Empty model → voice_runtime picks the platform default.
        with mock.patch.object(voice_runtime, "transcribe_audio", return_value="ok") as fn:
            self.client.post("/api/voice/transcribe", files={"audio": ("a.wav", b"b", "audio/wav")})
        self.assertEqual(fn.call_args.args[2], "")

    def test_transcribe_rejects_unknown_model(self) -> None:
        # The id reaches loaders that also accept filesystem paths, so it
        # must be a catalog id — anything else is a 400, not a load.
        response = self.client.post(
            "/api/voice/transcribe",
            files={"audio": ("a.wav", b"b", "audio/wav")},
            data={"model": "/etc/passwd"},
        )
        self.assertEqual(response.status_code, 400)

    def test_transcribe_error_mapping(self) -> None:
        for exc, status in ((RuntimeError("no backend"), 503), (ValueError("bad audio"), 500)):
            with mock.patch.object(voice_runtime, "transcribe_audio", side_effect=exc):
                response = self.client.post(
                    "/api/voice/transcribe", files={"audio": ("a.wav", b"b", "audio/wav")}
                )
            self.assertEqual(response.status_code, status)
            self.assertIn(str(exc), response.json()["detail"])

    def test_synthesize_clamps_speed(self) -> None:
        for requested, expected in ((0.0, 0.1), (-3.0, 0.1), (9.5, 4.0), (1.5, 1.5)):
            with mock.patch.object(voice_runtime, "synthesize_speech", return_value=b"w") as fn:
                self.client.post(
                    "/api/voice/synthesize", json={"text": "hi", "voice": "af_heart", "speed": requested}
                )
            self.assertEqual(fn.call_args.args[2], expected, requested)

    def test_synthesize_defaults_voice_and_speed(self) -> None:
        with mock.patch.object(voice_runtime, "synthesize_speech", return_value=b"w") as fn:
            response = self.client.post("/api/voice/synthesize", json={"text": "hi"})
        self.assertEqual(response.status_code, 200)
        fn.assert_called_once_with("hi", "af_heart", 1.0)

    def test_synthesize_unexpected_error_maps_to_500(self) -> None:
        with mock.patch.object(voice_runtime, "synthesize_speech", side_effect=ValueError("kaboom")):
            response = self.client.post(
                "/api/voice/synthesize", json={"text": "hi", "voice": "af_heart"}
            )
        self.assertEqual(response.status_code, 500)
        self.assertIn("kaboom", response.json()["detail"])

    def test_kokoro_download_routes(self) -> None:
        fake_status = {"state": "downloading", "progress": 0.25, "error": None}
        with mock.patch.object(voice_kokoro_onnx, "start_download", return_value=fake_status) as start:
            started = self.client.post("/api/voice/kokoro/download")
        self.assertEqual(started.json(), fake_status)
        start.assert_called_once()

        with mock.patch.object(voice_kokoro_onnx, "download_status", return_value=dict(fake_status)), \
             mock.patch.object(voice_kokoro_onnx, "is_installed", return_value=False):
            status = self.client.get("/api/voice/kokoro/download/status").json()
        self.assertEqual(status["progress"], 0.25)
        self.assertIs(status["installed"], False)

    def test_gallery_transcript_is_stripped_and_rejects_blank(self) -> None:
        blank = self.client.post("/api/voice/gallery/transcript", json={"text": "  \n "})
        self.assertEqual(blank.status_code, 400)
        saved = self.client.post("/api/voice/gallery/transcript", json={"text": "  padded  "})
        self.assertEqual(saved.json()["text"], "padded")

    def test_gallery_audio_download_rejects_bad_and_missing_ids(self) -> None:
        self.assertEqual(self.client.get("/api/voice/gallery/nope/audio").status_code, 400)
        self.assertEqual(
            self.client.get("/api/voice/gallery/1234567890_deadbeef/audio").status_code, 404
        )

    def test_gallery_audio_download_404_for_transcript_items(self) -> None:
        item_id = self.client.post("/api/voice/gallery/transcript", json={"text": "t"}).json()["id"]
        self.assertEqual(self.client.get(f"/api/voice/gallery/{item_id}/audio").status_code, 404)


if __name__ == "__main__":
    unittest.main()


class SttModelResolutionTests(unittest.TestCase):
    """FU-092 + Parakeet: catalog ids map onto the backend the host runs."""

    def setUp(self) -> None:
        voice_runtime.release_stt_model()
        self._snap = mock.patch.object(voice_runtime, "_hf_snapshot_path", return_value=None)
        self._snap.start()

    def tearDown(self) -> None:
        self._snap.stop()
        voice_runtime.release_stt_model()

    def _resolve(self, plat: str, available: set[str], model_id: str | None):
        with mock.patch.object(voice_runtime, "_platform", return_value=plat), \
             mock.patch.object(voice_runtime, "_is_available", side_effect=lambda n: n in available):
            return voice_runtime.resolve_stt_model(model_id)

    def test_mlx_id_on_linux_maps_to_ctranslate2_sibling(self) -> None:
        # The exact FU-092 failure: an MLX id reaching faster-whisper.
        self.assertEqual(
            self._resolve("linux", {"faster_whisper"}, "mlx-community/whisper-large-v3-turbo-q4"),
            ("faster-whisper", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"),
        )

    def test_ctranslate2_id_on_apple_maps_to_mlx_sibling(self) -> None:
        self.assertEqual(
            self._resolve("apple_silicon", {"mlx_whisper"}, "Systran/faster-whisper-base"),
            ("mlx-whisper", "mlx-community/whisper-base-mlx"),
        )

    def test_default_prefers_parakeet_when_installed(self) -> None:
        self.assertEqual(
            self._resolve("apple_silicon", {"mlx_whisper", "parakeet_mlx"}, ""),
            ("parakeet-mlx", "mlx-community/parakeet-tdt-0.6b-v3"),
        )

    def test_default_is_whisper_turbo_without_parakeet(self) -> None:
        self.assertEqual(
            self._resolve("apple_silicon", {"mlx_whisper"}, None),
            ("mlx-whisper", "mlx-community/whisper-large-v3-turbo-q4"),
        )

    def test_parakeet_rejected_off_apple(self) -> None:
        with self.assertRaises(RuntimeError):
            self._resolve("windows", {"faster_whisper"}, "mlx-community/parakeet-tdt-0.6b-v3")

    def test_unknown_and_path_ids_rejected(self) -> None:
        for bad in ("some/model", "/tmp/evil", "..\\x"):
            with self.assertRaises(voice_runtime.UnknownSttModelError):
                self._resolve("linux", {"faster_whisper"}, bad)

    def test_exactly_one_default_row_per_platform(self) -> None:
        for plat, avail in (
            ("apple_silicon", {"mlx_whisper"}),
            ("apple_silicon", {"mlx_whisper", "parakeet_mlx"}),
            ("linux", {"faster_whisper"}),
        ):
            with mock.patch.object(voice_runtime, "_platform", return_value=plat), \
                 mock.patch.object(voice_runtime, "_is_available", side_effect=lambda n, a=avail: n in a):
                rows = voice_runtime.list_stt_models()
            self.assertEqual(sum(1 for r in rows if r["default"]), 1, (plat, avail))

    def test_parakeet_transcribe_loads_once_and_reuses(self) -> None:
        loads: list[str] = []

        class FakeResult:
            text = "  bonjour  "

        class FakeModel:
            def transcribe(self, path: str) -> FakeResult:
                self.path = path
                return FakeResult()

        def from_pretrained(source: str) -> FakeModel:
            loads.append(source)
            return FakeModel()

        fake = types.ModuleType("parakeet_mlx")
        fake.from_pretrained = from_pretrained
        with mock.patch.object(voice_runtime, "_platform", return_value="apple_silicon"), \
             mock.patch.object(voice_runtime, "_is_available", side_effect=lambda n: n in {"parakeet_mlx", "mlx_whisper"}), \
             mock.patch.dict(sys.modules, {"parakeet_mlx": fake}):
            first = voice_runtime.transcribe_audio(b"x", "audio/webm", "mlx-community/parakeet-tdt-0.6b-v3")
            second = voice_runtime.transcribe_audio(b"y", "audio/webm", "")
        self.assertEqual((first, second), ("bonjour", "bonjour"))
        self.assertEqual(loads, ["mlx-community/parakeet-tdt-0.6b-v3"])
