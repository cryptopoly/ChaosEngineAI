"""One speak request for every voice: catalog rows and capability flags, model choice, request validation,
chunking, speed and pitch post-processing, the old routes as wrappers, the combined voice list, agent tools."""

from __future__ import annotations

import base64
import io
import json
import math
import re
import shutil
import struct
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock

from backend_service import voice_clone, voice_clone_safety, voice_env, voice_library, voice_runtime, voice_speak
from backend_service import voice_speak_audio
from backend_service.catalog import voice_models


def tone_wav(seconds: float = 0.5, rate: int = 24_000, hz: float = 440.0) -> bytes:
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * hz * i / rate))) for i in range(int(seconds * rate))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return buffer.getvalue()


def seconds_of(data: bytes) -> float:
    return voice_speak.wav_info(data)[3]


def installed(*keys: str):
    """Pretend only these catalog rows are downloaded."""
    rows = {voice_models.voice_model(k)["repo"] for k in keys}

    def snapshot(row):
        return Path("/snap") / row["key"] if row["repo"] in rows else None

    return snapshot


class Machine:
    """An Apple Silicon box with the voice pack installed."""

    def __enter__(self):
        self.patches = [
            mock.patch.object(voice_clone, "available_here", return_value=True),
            mock.patch.object(voice_env, "is_installed", return_value=True),
        ]
        for patch in self.patches:
            patch.start()
        return self

    def __exit__(self, *_exc):
        for patch in reversed(self.patches):
            patch.stop()


QWEN_ROWS = [r for r in voice_models.VOICE_MODELS if r["family"] == "qwen3-tts"]


class CatalogTests(unittest.TestCase):
    def test_new_rows_are_pinned_apache_and_unique(self) -> None:
        keys = [r["key"] for r in voice_models.VOICE_MODELS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            {r["key"] for r in QWEN_ROWS},
            {"qwen3-tts-0.6b", "qwen3-tts-1.7b", "qwen3-tts-custom-0.6b", "qwen3-tts-custom-1.7b", "qwen3-tts-design-1.7b"},
        )
        for row in QWEN_ROWS:
            self.assertRegex(row["revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(row["repo"].startswith("mlx-community/Qwen3-TTS-12Hz-"))
            self.assertTrue(row["upstream"].startswith("Qwen/Qwen3-TTS-12Hz-"))
            self.assertEqual(row["lanes"], ["mlx"])
            self.assertGreater(row["sizeGb"], 1.0)
        self.assertEqual(voice_models.CLONE_LICENCE["id"], "apache-2.0")

    def test_every_row_records_a_measured_run(self) -> None:
        for row in voice_models.listed_models():
            est = row["estimates"]
            self.assertEqual(est["basis"], "measured")
            self.assertIn("Apple M4 Max", est["measuredOn"])
            self.assertGreater(est["peakGb"], est["idleGb"] - 0.01)
            self.assertGreater(est["secondsPer10sAudio"], 0)
            self.assertNotIn("Faster", row["summary"])  # the 0.6B rows were not quicker than the 1.7B ones

    def test_capability_flags(self) -> None:
        for row in voice_models.VOICE_MODELS:
            self.assertEqual(
                set(row["capabilities"]),
                {"speak", "style", "clone", "design", "expression", "tags", "presets", "languageSelect"},
            )
        caps = {r["key"]: r["capabilities"] for r in QWEN_ROWS}
        self.assertFalse(caps["qwen3-tts-custom-0.6b"]["style"])
        self.assertTrue(caps["qwen3-tts-custom-1.7b"]["style"])
        self.assertTrue(caps["qwen3-tts-custom-0.6b"]["speak"] and caps["qwen3-tts-custom-1.7b"]["speak"])
        self.assertEqual([k for k, f in caps.items() if f["design"]], ["qwen3-tts-design-1.7b"])
        self.assertEqual([k for k, f in caps.items() if f["clone"]], ["qwen3-tts-0.6b", "qwen3-tts-1.7b"])

    def test_cloning_models_are_every_listed_clone_row_default_first(self) -> None:
        keys = [r["key"] for r in voice_models.CLONE_MODELS]
        self.assertEqual(keys[0], voice_models.DEFAULT_CLONE_MODEL)
        self.assertEqual(set(keys), {r["key"] for r in voice_models.listed_models() if r["capabilities"]["clone"]})
        self.assertIn("qwen3-tts-1.7b", keys)
        self.assertEqual(sum(r["default"] for r in voice_models.VOICE_MODELS), 1)
        with self.assertRaises(ValueError):
            voice_models.clone_model("qwen3-tts-custom-1.7b")
        self.assertEqual(voice_models.voice_model("qwen3-tts-custom-1.7b")["kind"], "custom")
        with self.assertRaises(ValueError):
            voice_models.voice_model("nope")

    def test_nine_preset_speakers(self) -> None:
        ids = [s["id"] for s in voice_models.CUSTOM_SPEAKERS]
        self.assertEqual(len(ids), 9)
        self.assertIn("Ryan", ids)
        self.assertEqual({s["language"] for s in voice_models.CUSTOM_SPEAKERS}, {"en", "zh", "ja", "ko"})


class PlanTests(unittest.TestCase):
    def test_the_installed_style_model_runs_with_or_without_style_text(self) -> None:
        # No model swap (a reload) when the style box gets text: the 1.7B model runs both times (D-67 W2).
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed("qwen3-tts-custom-0.6b", "qwen3-tts-custom-1.7b")):
            plain = voice_speak.build_plan("qwen3:Ryan")
            styled = voice_speak.build_plan("qwen3:ryan", style="  Very   happy ")
            listed = next(v for v in voice_speak.list_voices(include_clones=False) if v["id"] == "qwen3:Ryan")
        self.assertEqual(plain.model_key, "qwen3-tts-custom-1.7b")
        self.assertNotIn("instruct", plain.extra)
        self.assertEqual(styled.model_key, "qwen3-tts-custom-1.7b")
        self.assertEqual(styled.extra, {"mode": "preset", "speaker": "Ryan", "language": "auto", "instruct": "Very happy"})
        self.assertEqual(listed["model"], "qwen3-tts-custom-1.7b")

    def test_every_voice_says_when_ffmpeg_is_missing(self) -> None:
        # Pitch (every voice) and speed (all but the built-in ones) are ffmpeg steps: the screen warns before Generate.
        from backend_service.storyboard import ffmpeg_bin

        for exe, expected in ((None, True), ("/usr/bin/ffmpeg", False)):
            with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed("qwen3-tts-custom-1.7b")), \
                    mock.patch.object(ffmpeg_bin, "optional_ffmpeg_exe", return_value=exe):
                voices = voice_speak.list_voices(include_clones=False)
            self.assertTrue(voices)
            self.assertEqual({v["needsFfmpeg"] for v in voices}, {expected})

    def test_the_0_6b_model_runs_only_when_it_is_the_only_one_installed(self) -> None:
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed("qwen3-tts-custom-0.6b")):
            plain = voice_speak.build_plan("qwen3:Ryan")
            listed = next(v for v in voice_speak.list_voices(include_clones=False) if v["id"] == "qwen3:Ryan")
        self.assertEqual(plain.model_key, "qwen3-tts-custom-0.6b")
        self.assertEqual(listed["model"], "qwen3-tts-custom-0.6b")
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed("qwen3-tts-custom-1.7b")):
            self.assertEqual(voice_speak.build_plan("qwen3:Ryan").model_key, "qwen3-tts-custom-1.7b")

    def test_the_0_6b_model_never_gets_the_style_text(self) -> None:
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed("qwen3-tts-custom-0.6b")):
            plan = voice_speak.build_plan("qwen3:Aiden", style="Whisper")
            forced = voice_speak.build_plan("qwen3:Aiden", style="Whisper", model_key="qwen3-tts-custom-0.6b")
        for item in (plan, forced):
            self.assertNotIn("instruct", item.extra)
            self.assertTrue(any("0.6B" in n for n in item.notes))

    def test_missing_model_and_wrong_model_are_plain_errors(self) -> None:
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed()):
            with self.assertRaisesRegex(RuntimeError, "not downloaded"):
                voice_speak.build_plan("qwen3:Ryan")
            with self.assertRaisesRegex(RuntimeError, "VoiceDesign"):
                voice_speak.build_plan("design", style="A calm older woman")
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed("qwen3-tts-design-1.7b")):
            with self.assertRaisesRegex(ValueError, "cannot speak"):
                voice_speak.build_plan("qwen3:Ryan", model_key="qwen3-tts-design-1.7b")
            with self.assertRaisesRegex(ValueError, "Unknown voice"):
                voice_speak.build_plan("qwen3:Nobody")
        with mock.patch.object(voice_clone, "available_here", return_value=False):
            with self.assertRaisesRegex(ValueError, "Apple Silicon"):
                voice_speak.build_plan("qwen3:Ryan")

    def test_design_needs_a_description(self) -> None:
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed("qwen3-tts-design-1.7b")):
            with self.assertRaisesRegex(ValueError, "Describe"):
                voice_speak.build_plan("design")
            plan = voice_speak.build_plan("design", style="Calm woman in her fifties")
        self.assertEqual((plan.kind, plan.extra["mode"], plan.extra["instruct"]), ("design", "design", "Calm woman in her fifties"))

    def test_kokoro_voice_keeps_working_and_ignores_style(self) -> None:
        plan = voice_speak.build_plan("af_heart", style="Angry")
        self.assertEqual((plan.kind, plan.native_speed, plan.model_key), ("kokoro", True, "kokoro"))
        self.assertTrue(any("no style" in n for n in plan.notes))
        with self.assertRaisesRegex(ValueError, "Unknown voice"):
            voice_speak.build_plan("zz_nobody")


class AudioTests(unittest.TestCase):
    def test_atempo_chain_splits_out_of_range_factors(self) -> None:
        def product(filters):
            out = 1.0
            for f in filters:
                out *= float(f.split("=")[1])
            return out

        for factor in (0.5, 0.8, 1.25, 2.0, 3.0, 4.5, 0.2):
            chain = voice_speak.atempo_chain(factor)
            self.assertAlmostEqual(product(chain), factor, places=4)
            self.assertTrue(all(0.5 <= float(f.split("=")[1]) <= 2.0 for f in chain))
        self.assertEqual(voice_speak.atempo_chain(1.0), [])
        with self.assertRaises(ValueError):
            voice_speak.atempo_chain(0)

    def test_filter_graph(self) -> None:
        self.assertEqual(voice_speak.filter_graph(24_000, 1.0, 0.0), "")
        self.assertEqual(voice_speak.filter_graph(24_000, 1.5, 0.0), "atempo=1.500000")
        pitched = voice_speak.filter_graph(24_000, 1.0, 12.0)
        self.assertTrue(pitched.startswith("asetrate=48000.000,aresample=24000,atempo=0.5"))

    def test_neutral_settings_pass_the_audio_through_untouched(self) -> None:
        wav = tone_wav()
        with mock.patch.object(voice_speak_audio.subprocess, "run", side_effect=AssertionError("ffmpeg must not run")):
            self.assertIs(voice_speak.process_audio(wav, 1.0, 0.0), wav)

    def test_changes_need_ffmpeg(self) -> None:
        from backend_service.storyboard import ffmpeg_bin

        with mock.patch.object(ffmpeg_bin, "optional_ffmpeg_exe", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg"):
                voice_speak.process_audio(tone_wav(), 1.2, 0.0)

    def test_ffmpeg_is_called_list_form_with_the_filter(self) -> None:
        from backend_service.storyboard import ffmpeg_bin

        seen = {}
        real_run = voice_speak_audio.subprocess.run

        def fake_run(command, **kwargs):
            # ``subprocess.run`` is patched for the whole process: a background thread's call (the memory check's
            # ``sysctl -n kern.memorystatus_level``) must not be answered by writing a WAV under its last argument.
            if command[0] != "/bin/ffmpeg":
                return real_run(command, **kwargs)
            seen["command"], seen["kwargs"] = command, kwargs
            Path(command[-1]).write_bytes(tone_wav(0.25))
            return mock.Mock(returncode=0, stderr="")

        with mock.patch.object(ffmpeg_bin, "optional_ffmpeg_exe", return_value="/bin/ffmpeg"), \
             mock.patch.object(voice_speak_audio.subprocess, "run", fake_run):
            out = voice_speak.process_audio(tone_wav(0.5), 2.0, 0.0)
        self.assertIsInstance(seen["command"], list)
        self.assertNotIn("shell", seen["kwargs"])
        self.assertEqual(seen["command"][seen["command"].index("-af") + 1], "atempo=2.000000")
        self.assertAlmostEqual(seconds_of(out), 0.25, places=2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_real_ffmpeg_speed_and_pitch(self) -> None:
        wav = tone_wav(1.0)
        with mock.patch("backend_service.storyboard.ffmpeg_bin.optional_ffmpeg_exe", return_value=shutil.which("ffmpeg")):
            faster = voice_speak.process_audio(wav, 2.0, 0.0)
            slower = voice_speak.process_audio(wav, 0.5, 0.0)
            pitched = voice_speak.process_audio(wav, 1.0, 5.0)
            both = voice_speak.process_audio(wav, 1.5, -3.0)
        self.assertAlmostEqual(seconds_of(faster), 0.5, delta=0.05)
        self.assertAlmostEqual(seconds_of(slower), 2.0, delta=0.05)
        self.assertAlmostEqual(seconds_of(pitched), 1.0, delta=0.05)
        self.assertAlmostEqual(seconds_of(both), 1.0 / 1.5, delta=0.06)
        self.assertEqual(voice_speak.wav_info(faster)[2], 24_000)

    def test_join_adds_a_gap_and_rejects_mixed_formats(self) -> None:
        joined = voice_speak.join_wavs([tone_wav(0.2), tone_wav(0.2)], gap_s=0.1)
        self.assertAlmostEqual(seconds_of(joined), 0.5, places=2)
        self.assertEqual(voice_speak.join_wavs([tone_wav(0.2)]), voice_speak.join_wavs([tone_wav(0.2)]))
        with self.assertRaises(ValueError):
            voice_speak.join_wavs([tone_wav(0.1, rate=24_000), tone_wav(0.1, rate=16_000)])

    def test_quiet_speech_is_lifted_and_normal_speech_is_not(self) -> None:
        def peak_of(data: bytes) -> float:
            with wave.open(io.BytesIO(data), "rb") as handle:
                raw = handle.readframes(handle.getnframes())
            values = struct.unpack("<%dh" % (len(raw) // 2), raw)
            return max(abs(v) for v in values) / 32768.0

        quiet = bytearray()
        for i in range(2400):
            quiet += struct.pack("<h", int(800 * math.sin(i / 5)))  # peak about 0.024
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24_000)
            handle.writeframes(bytes(quiet))
        lifted = voice_speak.lift_quiet(buffer.getvalue())
        self.assertAlmostEqual(peak_of(lifted), 0.024 * voice_speak.MAX_LIFT, delta=0.02)  # capped lift
        normal = tone_wav(0.2)  # peak about 0.24
        self.assertEqual(voice_speak.lift_quiet(normal), normal)
        self.assertEqual(seconds_of(lifted), 0.1)


class RenderTests(unittest.TestCase):
    def test_text_is_tidied_and_cut_into_chunks_for_every_model(self) -> None:
        text = "# Title\n\nSee [the docs](https://x.y/z). " + "A sentence about speech. " * 40 + "\n\n```py\nprint(1)\n```"
        chunks = voice_speak.chunk_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= voice_speak.CHUNK_CHARS for c in chunks))
        joined = " ".join(chunks)
        self.assertNotIn("```", joined)
        self.assertNotIn("https://", joined)
        self.assertNotIn("#", joined)
        self.assertIn("the docs", joined)

    def test_worker_gets_one_request_per_chunk_and_post_processing_runs_once(self) -> None:
        worker = mock.Mock()
        worker.request.return_value = {"wavB64": base64.b64encode(tone_wav(0.2)).decode(), "seconds": 0.5}
        plan = voice_speak.Plan("preset", "qwen3:Ryan", "k", "Name", "mlx-audio", "/snap", {"mode": "preset", "speaker": "Ryan"})
        with mock.patch("backend_service.voice_client.get_worker", return_value=worker), \
             mock.patch.object(voice_speak, "process_audio", side_effect=lambda w, s, p: w) as post:
            out = voice_speak.render(plan, "One. Two. Three.", speed=1.5, pitch=2.0, seed=7)
        self.assertEqual(worker.request.call_count, 3)
        first = worker.request.call_args_list[0].kwargs
        self.assertEqual((first["speaker"], first["seed"], first["engine"], first["text"]), ("Ryan", 7, "mlx-audio", "One."))
        post.assert_called_once()
        self.assertEqual(post.call_args.args[1:], (1.5, 2.0))  # non-Kokoro models get speed after the fact
        self.assertEqual((out.chunks, out.seconds), (3, 1.5))
        self.assertAlmostEqual(out.duration_s, 0.2 * 3 + 0.15 * 2, places=2)

    def test_kokoro_applies_speed_itself_and_only_pitch_is_post_processing(self) -> None:
        plan = voice_speak.build_plan("af_heart")
        calls = []

        def synth(text, voice, speed):
            calls.append((text, voice, speed))
            return tone_wav(0.2), 0.2

        with mock.patch.object(voice_runtime, "synthesize", synth), \
             mock.patch.object(voice_speak, "process_audio", side_effect=lambda w, s, p: w) as post:
            voice_speak.render(plan, "Hello there. General.", speed=1.3, pitch=-2.0)
        self.assertEqual(calls, [("Hello there.", "af_heart", 1.3), ("General.", "af_heart", 1.3)])
        self.assertEqual(post.call_args.args[1:], (1.0, -2.0))

    def test_nothing_to_speak_after_tidying(self) -> None:
        with self.assertRaisesRegex(ValueError, "Nothing"):
            voice_speak.render(voice_speak.build_plan("af_heart"), "   ")

    def test_speak_validates_before_any_work(self) -> None:
        for kwargs, message in (
            ({"text": " "}, "Text is required"),
            ({"text": "x" * 20_001}, "characters"),
            ({"text": "x", "speed": 3.0}, "Speed"),
            ({"text": "x", "pitch": 13.0}, "Pitch"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                voice_speak.speak(kwargs.pop("text"), "af_heart", origin="ui", **kwargs)


class WorkerModeTests(unittest.TestCase):
    def test_preset_and_design_modes_call_the_matching_model_method(self) -> None:
        import numpy as np

        from backend_service import voice_worker

        seen = {}

        class Model:
            sample_rate = 24_000

            def generate_custom_voice(self, **kwargs):
                seen["preset"] = kwargs
                yield types_namespace(np.ones(2400, dtype=np.float32))

            def generate_voice_design(self, **kwargs):
                seen["design"] = kwargs
                yield types_namespace(np.ones(4800, dtype=np.float32))

        with mock.patch.object(voice_worker, "_load_tts", return_value=Model()):
            preset = voice_worker.synthesize({"engine": "mlx-audio", "source": "/s", "text": "Hi", "mode": "preset",
                                              "speaker": "Ryan", "instruct": "Happy", "language": "auto"})
            design = voice_worker.synthesize({"engine": "mlx-audio", "source": "/s", "text": "Hi", "mode": "design",
                                              "instruct": "Calm woman"})
            with self.assertRaises(ValueError):
                voice_worker.synthesize({"engine": "kokoro-onnx", "source": "/s", "text": "Hi", "mode": "preset", "speaker": "Ryan"})
        self.assertEqual(seen["preset"], {"text": "Hi", "speaker": "Ryan", "language": "auto", "instruct": "Happy"})
        self.assertEqual(seen["design"], {"text": "Hi", "instruct": "Calm woman", "language": "auto"})
        self.assertEqual((preset["durationS"], design["durationS"]), (0.1, 0.2))

    def test_a_seed_is_applied_before_generating(self) -> None:
        import sys
        import types

        import numpy as np

        from backend_service import voice_worker

        order = []
        fake_mx = types.ModuleType("mlx.core")
        fake_mx.random = types.SimpleNamespace(seed=lambda value: order.append(("seed", value)))
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_mx

        class Model:
            sample_rate = 24_000

            def generate_custom_voice(self, **_kwargs):
                order.append(("generate", None))
                yield types_namespace(np.ones(10, dtype=np.float32))

        with mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_mx}), \
             mock.patch.object(voice_worker, "_load_tts", return_value=Model()):
            voice_worker.synthesize({"engine": "mlx-audio", "source": "/s", "text": "Hi", "mode": "preset",
                                     "speaker": "Ryan", "seed": 42})
        self.assertEqual(order, [("seed", 42), ("generate", None)])

    def test_the_old_model_is_freed_before_the_next_one_loads(self) -> None:
        from backend_service import voice_worker

        order = []
        voice_worker._tts["key"], voice_worker._tts["model"] = ("mlx-audio", "old"), object()
        fake = mock.Mock(side_effect=lambda source: order.append(("load", source)) or object())
        import sys
        import types

        loader = types.ModuleType("mlx_audio.tts.utils")
        loader.load_model = fake
        try:
            with mock.patch.object(voice_worker, "_free_tts", side_effect=lambda: order.append(("free", None))), \
                 mock.patch.object(voice_worker, "_short_espeak_data_path"), \
                 mock.patch.dict(sys.modules, {"mlx_audio": types.ModuleType("mlx_audio"),
                                               "mlx_audio.tts": types.ModuleType("mlx_audio.tts"),
                                               "mlx_audio.tts.utils": loader}):
                voice_worker._load_tts("mlx-audio", "new")
        finally:
            voice_worker._tts["key"], voice_worker._tts["model"] = None, None
        self.assertEqual(order, [("free", None), ("load", "new")])


def types_namespace(audio):
    from types import SimpleNamespace

    return SimpleNamespace(audio=audio)


class VoiceListTests(unittest.TestCase):
    def test_one_list_with_builtin_preset_design_and_clone_voices(self) -> None:
        clones = [{"id": "1_abcdef12", "name": "Me", "consent": {"speakerName": "Dan"}}]
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed("qwen3-tts-custom-1.7b")), \
             mock.patch.object(voice_clone, "list_voices", return_value=clones), \
             mock.patch.object(voice_env, "watermark_installed", return_value=False):
            voices = voice_speak.list_voices()
        by_kind = {}
        for v in voices:
            by_kind.setdefault(v["kind"], []).append(v)
        self.assertEqual(set(by_kind), {"builtin", "preset", "design", "clone"})
        self.assertEqual(len(by_kind["builtin"]), 49)  # 54 Kokoro voices less the five Japanese ones (never on MLX)
        # Nine Qwen3-TTS speakers, Turbo's and Chatterbox's own voices, eight Orpheus voices.
        self.assertEqual(len(by_kind["preset"]), 9 + 1 + 1 + 8)
        fields = {"id", "name", "kind", "group", "family", "language", "model", "capabilities"}
        self.assertTrue(all(fields <= set(v) for v in voices))
        self.assertEqual(len({v["id"] for v in voices}), len(voices))
        builtin = by_kind["builtin"][0]
        self.assertEqual((builtin["model"], builtin["capabilities"]["style"]), ("kokoro", False))
        preset = by_kind["preset"][0]
        self.assertTrue(preset["id"].startswith("qwen3:"))
        self.assertTrue(preset["available"])
        self.assertTrue(preset["capabilities"]["style"])
        design = by_kind["design"][0]
        self.assertFalse(design["available"])
        self.assertIn("VoiceDesign", design["needs"])
        self.assertTrue(design["capabilities"]["design"])
        clone = by_kind["clone"][0]
        self.assertEqual(clone["id"], "clone:1_abcdef12")
        self.assertTrue(clone["capabilities"]["clone"] and not clone["capabilities"]["style"])
        self.assertFalse(clone["available"])  # watermark component missing

    def test_a_machine_with_only_the_1_7b_cloning_model_can_clone(self) -> None:
        clones = [{"id": "1_abcdef12", "name": "Me", "consent": {"speakerName": "Dan"}}]
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed("qwen3-tts-1.7b")), \
             mock.patch.object(voice_clone, "list_voices", return_value=clones), \
             mock.patch.object(voice_env, "watermark_installed", return_value=True):
            clone = [v for v in voice_speak.list_voices() if v["kind"] == "clone"][0]
            self.assertTrue(clone["available"])
            self.assertEqual(clone["model"], "qwen3-tts-1.7b")
            self.assertEqual(voice_clone.pick_clone_model("")["key"], "qwen3-tts-1.7b")
            self.assertEqual(voice_clone.pick_clone_model(None)["key"], "qwen3-tts-1.7b")
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed("qwen3-tts-0.6b", "qwen3-tts-1.7b")):
            self.assertEqual(voice_clone.pick_clone_model("")["key"], "qwen3-tts-0.6b")  # the default wins when both are there
            self.assertEqual(voice_clone.pick_clone_model("qwen3-tts-1.7b")["key"], "qwen3-tts-1.7b")  # a named model is honoured
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed()):
            self.assertEqual(voice_clone.pick_clone_model("")["key"], "qwen3-tts-0.6b")  # nothing installed: the default is named in the error

    def test_clones_can_be_left_out(self) -> None:
        with mock.patch.object(voice_clone, "list_voices", side_effect=AssertionError("not read")):
            voices = voice_speak.list_voices(include_clones=False)
        self.assertNotIn("clone", {v["kind"] for v in voices})


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

    def result(self, **extra):
        base = {"wav": tone_wav(0.1), "durationS": 0.1, "seconds": 0.2, "chunks": 1, "voice": "af_heart",
                "kind": "builtin", "model": "kokoro", "modelName": "Kokoro", "clonedVoice": False,
                "libraryId": None, "notes": []}
        return {**base, **extra}

    def test_request_shape_and_defaults_reach_speak(self) -> None:
        with mock.patch.object(voice_speak, "speak", return_value=self.result()) as run:
            body = self.client.post("/api/voice/speak", json={"text": "Hello"}).json()
            self.client.post("/api/voice/speak", json={
                "text": "Hello", "voice": "qwen3:Ryan", "style": "Calm", "speed": 1.2, "pitch": -3, "seed": 5,
                "model": "qwen3-tts-custom-1.7b",
            })
        self.assertEqual(base64.b64decode(body["audio"])[:4], b"RIFF")
        self.assertEqual({"audio", "durationS", "seconds", "chunks", "voice", "kind", "model", "modelName",
                          "clonedVoice", "libraryId", "notes"}, set(body))
        first, second = run.call_args_list
        self.assertEqual(first.args, ("Hello", "af_heart"))
        self.assertEqual((first.kwargs["style"], first.kwargs["speed"], first.kwargs["pitch"], first.kwargs["seed"]), ("", 1.0, 0.0, None))
        self.assertEqual(second.args, ("Hello", "qwen3:Ryan"))
        self.assertEqual((second.kwargs["style"], second.kwargs["speed"], second.kwargs["pitch"], second.kwargs["seed"],
                          second.kwargs["model"]), ("Calm", 1.2, -3.0, 5, "qwen3-tts-custom-1.7b"))
        self.assertEqual(second.kwargs["origin"], "ui")

    def test_clone_ids_never_skip_the_clone_check(self) -> None:
        from backend_service import voice_clone_safety as safety

        with mock.patch.object(voice_speak, "speak", return_value=self.result()) as run, \
                mock.patch.object(safety, "clone_allowed", return_value=False):
            for voice in ("clone:abc", " clone:abc", "  clone:abc  "):
                response = self.client.post("/api/voice/speak", json={"text": "Hi", "voice": voice})
                self.assertEqual(response.status_code, 403, voice)
            old = self.client.post("/api/voice/synthesize", json={"text": "Hi", "voice": " clone:abc"})
            self.assertEqual(old.status_code, 400)
        run.assert_not_called()

    def test_speak_needs_an_explicit_origin(self) -> None:
        with self.assertRaises(TypeError):
            voice_speak.speak("Hi", "af_heart")  # type: ignore[call-arg]

    def test_request_validation(self) -> None:
        post = lambda **body: self.client.post("/api/voice/speak", json=body).status_code  # noqa: E731
        self.assertEqual(post(text=""), 422)
        self.assertEqual(post(text="x" * 20_001), 422)
        self.assertEqual(post(text="x", speed=0.4), 422)
        self.assertEqual(post(text="x", speed=2.1), 422)
        self.assertEqual(post(text="x", pitch=12.5), 422)
        self.assertEqual(post(text="x", seed=-1), 422)
        self.assertEqual(post(text="x", style="s" * 501), 422)
        self.assertEqual(post(text="x", voice=""), 422)

    def test_errors_map_to_plain_statuses(self) -> None:
        for error, status in ((ValueError("Unknown voice 'x'."), 400), (RuntimeError("not downloaded"), 503),
                              (LookupError("Cloned voice not found."), 404)):
            with mock.patch.object(voice_speak, "speak", side_effect=error):
                response = self.client.post("/api/voice/speak", json={"text": "Hi"})
            self.assertEqual(response.status_code, status)
        self.assertEqual(self.client.post("/api/voice/speak", json={"text": "Hi", "voice": "zz_none"}).status_code, 400)
        self.assertEqual(
            self.client.post("/api/voice/speak", json={"text": "Hi", "voice": "design"}).status_code, 400)

    def test_cloned_voices_keep_the_agent_rules(self) -> None:
        body = {"text": "Hi", "voice": "clone:1234567890_deadbeef"}
        with mock.patch("backend_service.mcp_bridge.access.token_ok", return_value=True):
            headers = {"x-chaosengine-token": "agent-token"}
            self.assertEqual(self.client.post("/api/voice/speak", json=body, headers=headers).status_code, 403)
        self.assertTrue(self.client.post("/api/voice/settings", json={"allowAgentCloning": True}).json()["allowAgentCloning"])
        with mock.patch("backend_service.mcp_bridge.access.token_ok", return_value=True), \
             mock.patch.object(voice_clone, "get_voice", return_value={"id": "1234567890_deadbeef"}):
            unconfirmed = self.client.post("/api/voice/speak", json=body, headers=headers)
        self.assertEqual(unconfirmed.status_code, 400)
        self.assertIn("permission_confirmed", unconfirmed.json()["detail"])
        events = [json.loads(l)["event"] for l in voice_clone_safety.audit_path().read_text().splitlines()]
        self.assertIn("refused", events)
        # The app's own window needs no confirmation, but the voice must exist.
        self.assertEqual(self.client.post("/api/voice/speak", json=body).status_code, 404)

    def test_cloned_speech_is_tagged_watermarked_saved_and_returned(self) -> None:
        voice = voice_library.create(
            "clone", {"title": "Me", "text": "Hello, this is my sample.", "durationS": 6.0,
                      "consent": {"speakerName": "Dan", "basis": "own", "consentedAt": "now", "permissionConfirmed": True}},
            audio_file=self._sample(), audio_ext=".wav",
        )
        worker = mock.Mock()
        worker.request.return_value = {"wavB64": base64.b64encode(tone_wav(0.2)).decode(), "seconds": 1.0}
        with mock.patch.object(voice_clone, "snapshot_dir", return_value=Path(self._data.name)), \
             mock.patch.object(voice_clone, "available_here", return_value=True), \
             mock.patch.object(voice_env, "is_installed", return_value=True), \
             mock.patch.object(voice_env, "watermark_installed", return_value=True), \
             mock.patch("backend_service.voice_client.get_worker", return_value=worker), \
             mock.patch.object(voice_clone_safety, "apply_watermark", return_value=True) as mark:
            response = self.client.post("/api/voice/speak", json={"text": "Hi there. Again.", "voice": f"clone:{voice['id']}", "style": "Angry"})
        body = response.json()
        self.assertEqual(response.status_code, 200, body)
        mark.assert_called_once()
        self.assertEqual(worker.request.call_count, 2)
        self.assertTrue(body["clonedVoice"] and body["libraryId"])
        self.assertTrue(any("style" in n for n in body["notes"]))
        tag = voice_clone_safety.read_wav_tag(base64.b64decode(body["audio"]))
        self.assertTrue(tag["machine_generated"] and tag["clonedVoice"])
        saved = voice_library.get(body["libraryId"])
        self.assertTrue(saved["clonedVoice"])

    def _sample(self) -> Path:
        path = Path(self._data.name) / "sample.wav"
        path.write_bytes(tone_wav(1.0))
        return path

    def test_cancel_stops_the_running_job_and_the_next_one_does_not_wait(self) -> None:
        from backend_service.helpers.job_queue import HEAVY_JOBS
        from backend_service.progress import GenerationCancelled

        started, cancelled = threading.Event(), threading.Event()
        plan = voice_speak.Plan("preset", "qwen3:Ryan", "k", "Name", "mlx-audio", "/snap", {})
        calls = []

        def slow_chunk(_plan, sentence, _speed, _seed):
            calls.append(sentence)
            started.set()
            cancelled.wait(5)  # a chunk in progress; the worker is ended by the cancel
            raise RuntimeError("The voice worker exited.")

        outcome = {}

        def job():
            try:
                with HEAVY_JOBS.slot("voice", "ui", "speech"):
                    voice_speak.render(plan, "One. Two. Three.")
            except GenerationCancelled:
                outcome["cancelled"] = True

        with mock.patch.object(voice_speak, "_speak_chunk", side_effect=slow_chunk), \
             mock.patch("backend_service.voice_client.interrupt_all", side_effect=cancelled.set) as interrupt:
            thread = threading.Thread(target=job)
            thread.start()
            self.assertTrue(started.wait(5))
            self.assertTrue(self.client.post("/api/voice/speak/cancel").json()["cancelled"])
            thread.join(5)
        self.assertFalse(thread.is_alive())
        interrupt.assert_called_once()
        self.assertTrue(outcome.get("cancelled"))
        self.assertEqual(len(calls), 1)  # nothing after the cancel
        self.assertIsNone(HEAVY_JOBS.running_source("voice"))  # the slot is free, the next Generate starts at once
        with mock.patch.object(voice_speak, "_speak_chunk", return_value=(tone_wav(0.1), 0.0)) as again:
            voice_speak.render(plan, "Fresh start.")
        again.assert_called_once()

    def test_cancel_with_nothing_running_leaves_the_worker_alone(self) -> None:
        with mock.patch("backend_service.voice_client.interrupt_all") as interrupt:
            body = self.client.post("/api/voice/speak/cancel").json()
        self.assertFalse(body["cancelled"])
        interrupt.assert_not_called()

    def test_an_agents_speech_is_not_stopped_by_the_screen_cancel(self) -> None:
        from backend_service.helpers.job_queue import HEAVY_JOBS

        with HEAVY_JOBS.slot("voice", "agent", "speech"), mock.patch("backend_service.voice_client.interrupt_all") as interrupt:
            self.client.post("/api/voice/speak/cancel")
        interrupt.assert_not_called()

    def test_a_cancelled_request_answers_499(self) -> None:
        from backend_service.progress import GenerationCancelled

        with mock.patch.object(voice_speak, "speak", side_effect=GenerationCancelled("x")):
            self.assertEqual(self.client.post("/api/voice/speak", json={"text": "Hi"}).status_code, 499)

    def test_old_routes_still_work_as_wrappers(self) -> None:
        with mock.patch.object(voice_speak, "speak", return_value=self.result()) as run:
            response = self.client.post("/api/voice/synthesize", json={"text": "Hello", "voice": "af_bella", "speed": 1.4})
        self.assertEqual((response.status_code, response.headers["content-type"]), (200, "audio/wav"))
        self.assertEqual(response.content[:4], b"RIFF")
        self.assertEqual((run.call_args.args, run.call_args.kwargs), (("Hello", "af_bella"), {"speed": 1.4, "origin": "ui"}))
        with mock.patch.object(voice_runtime, "synthesize", return_value=(tone_wav(0.1), 0.1)):
            stream = self.client.post("/api/voice/speak/stream", json={"text": "One. Two."})
        lines = [json.loads(l) for l in stream.text.splitlines()]
        self.assertEqual([l.get("index") for l in lines[:-1]], [0, 1])
        self.assertTrue(lines[-1]["done"])
        with mock.patch.object(voice_clone, "speak", return_value={"id": "1_abcdef12", "kind": "speech", "clonedVoice": True}):
            ok = self.client.post("/api/voice/clone/speak", json={"voiceId": "1234567890_deadbeef", "text": "Hi"})
        self.assertTrue(ok.json()["clonedVoice"])

    def test_old_synthesize_route_now_tidies_the_text(self) -> None:
        spoken = []
        with mock.patch.object(voice_runtime, "synthesize", side_effect=lambda t, v, s: spoken.append(t) or (tone_wav(0.1), 0.1)):
            response = self.client.post("/api/voice/synthesize", json={"text": "**Hello** [there](https://a.b)."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(spoken, ["Hello there."])

    def test_voice_list_route(self) -> None:
        body = self.client.get("/api/voice/voices").json()
        kinds = {v["kind"] for v in body["voices"]}
        self.assertTrue({"builtin", "preset", "design"} <= kinds)
        with mock.patch("backend_service.mcp_bridge.access.token_ok", return_value=True):
            agent = self.client.get("/api/voice/voices", headers={"x-chaosengine-token": "agent-token"}).json()
        self.assertNotIn("clone", {v["kind"] for v in agent["voices"]})

    def test_models_route_lists_every_row_with_capabilities(self) -> None:
        body = self.client.get("/api/voice/models").json()
        self.assertEqual(len(body["models"]), len(voice_models.listed_models()))
        by_key = {m["key"]: m for m in body["models"]}
        self.assertFalse(by_key["qwen3-tts-custom-0.6b"]["capabilities"]["style"])
        self.assertTrue(by_key["qwen3-tts-custom-1.7b"]["capabilities"]["style"])
        self.assertEqual(by_key["qwen3-tts-design-1.7b"]["kind"], "design")
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", m["revision"]) for m in body["models"]))
        # The cloning screen lists every model that can clone.
        self.assertEqual(len(self.client.get("/api/voice/clone/models").json()["models"]), len(voice_models.CLONE_MODELS))


class ChatStreamTests(unittest.TestCase):
    """``/api/voice/speak/stream`` with any voice (D-67 W6, D-72): plans through ``voice_speak`` per sentence, falls
    back to Kokoro with a note, cloned voices only for read-aloud in this window."""

    def setUp(self) -> None:
        RouteTests.setUp(self)  # type: ignore[arg-type]
        self.addCleanup(RouteTests.tearDown, self)
        from backend_service import voice_chat

        voice_chat.forget_failures()
        self.addCleanup(voice_chat.forget_failures)

    def stream(self, voice: str, use: str = "read", text: str = "Hello there. How are you?", **kwargs):
        response = self.client.post("/api/voice/speak/stream", json={"text": text, "voice": voice, "use": use}, **kwargs)
        return response, [json.loads(line) for line in response.text.splitlines()] if response.status_code == 200 else []

    def kokoro(self):
        return mock.patch.object(voice_runtime, "synthesize", side_effect=lambda t, v, s: (tone_wav(0.1), 0.1))

    def test_a_preset_voice_reads_aloud_through_its_own_model(self) -> None:
        plans = []

        def chunk(plan, sentence, speed, seed, extra=None):
            plans.append((plan.kind, plan.model_key, sentence, speed))
            return tone_wav(0.2), 0.1

        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", side_effect=installed("qwen3-tts-custom-1.7b")), \
                mock.patch.object(voice_speak, "_speak_chunk", side_effect=chunk):
            response, lines = self.stream("qwen3:Ryan", use="call")
        self.assertEqual(response.status_code, 200)
        self.assertEqual([l.get("voice") for l in lines[:-1]], ["qwen3:Ryan", "qwen3:Ryan"])
        self.assertFalse(any("note" in l for l in lines))
        self.assertEqual(plans, [("preset", "qwen3-tts-custom-1.7b", "Hello there.", 1.0),
                                 ("preset", "qwen3-tts-custom-1.7b", "How are you?", 1.0)])
        self.assertTrue(lines[-1]["done"])

    def test_turbo_keeps_its_sound_tags_and_others_strip_them(self) -> None:
        spoken = []
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", side_effect=installed("chatterbox-turbo")), \
                mock.patch.object(voice_speak, "_speak_chunk", side_effect=lambda p, s, *a, **k: spoken.append(s) or (tone_wav(0.1), 0.1)):
            self.stream("chatterbox-turbo", text="Ha [laugh] that is funny.")
        self.assertIn("[laugh]", spoken[0])

    def test_a_voice_whose_model_is_missing_falls_back_to_kokoro_with_a_note(self) -> None:
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", return_value=None), self.kokoro() as kokoro:
            response, lines = self.stream("qwen3:Ryan")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Ryan could not speak here", lines[0]["note"])
        self.assertIn("Kokoro", lines[0]["note"])
        self.assertEqual({l.get("voice") for l in lines[1:-1]}, {"af_heart"})
        self.assertEqual(kokoro.call_count, 2)

    def test_a_model_that_fails_to_load_hands_over_to_kokoro_and_is_skipped_next_time(self) -> None:
        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", side_effect=installed("chatterbox-turbo")), \
                mock.patch.object(voice_speak, "_speak_chunk", side_effect=RuntimeError("out of memory")) as chunk, \
                self.kokoro() as kokoro:
            _r, lines = self.stream("chatterbox-turbo")
            self.assertIn("out of memory", lines[0]["note"])
            self.assertEqual([l.get("text") for l in lines[1:-1]], ["Hello there.", "How are you?"])
            self.assertEqual(kokoro.call_count, 2)
            _r, again = self.stream("chatterbox-turbo")
        self.assertEqual(chunk.call_count, 1)  # the failed load is not tried again within RETRY_AFTER_S
        self.assertIn("note", again[0])

    def test_kokoro_never_reads_a_sound_tag_aloud(self) -> None:
        spoken = []
        with mock.patch.object(voice_runtime, "synthesize", side_effect=lambda t, v, s: spoken.append(t) or (tone_wav(0.1), 0.1)):
            self.stream("af_heart", text="Ha [laugh] that is funny.")
        self.assertEqual(spoken, ["Ha that is funny."])

    def test_a_kokoro_failure_is_still_an_error(self) -> None:
        with mock.patch.object(voice_runtime, "synthesize", side_effect=RuntimeError("no voice")):
            _r, lines = self.stream("af_heart")
        self.assertEqual(lines[0]["error"], "no voice")

    def test_describe_a_voice_cannot_be_the_chat_voice(self) -> None:
        with self.kokoro():
            _r, lines = self.stream("design")
        self.assertIn("description", lines[0]["note"])

    def test_a_cloned_voice_reads_aloud_marked_and_audited_but_is_refused_for_calls_and_agents(self) -> None:
        plan = voice_speak.Plan("clone", "clone:1234567890_deadbeef", "qwen3-tts-0.6b", "Qwen3-TTS 0.6B Base")
        row = voice_models.voice_model("qwen3-tts-0.6b")
        with mock.patch.object(voice_clone, "clone_plan", return_value=(plan, row, {}, [])) as checks, \
                mock.patch.object(voice_speak, "_speak_chunk", return_value=(tone_wav(0.1), 0.1)), \
                mock.patch.object(voice_clone, "mark_cloned", side_effect=lambda wav, r, vid: b"MARKED" + wav) as mark, \
                mock.patch.object(voice_clone_safety, "audit") as audit:
            _r, lines = self.stream("clone:1234567890_deadbeef")
            self.assertEqual(checks.call_args.kwargs["origin"], "ui")
            self.assertEqual(mark.call_count, 2)
            self.assertTrue(base64.b64decode(lines[0]["audio"]).startswith(b"MARKED"))
            self.assertEqual(audit.call_args.kwargs["use"], "read-aloud")
            with self.kokoro():
                _r, call = self.stream("clone:1234567890_deadbeef", use="call")
            self.assertIn("not used in voice calls", call[0]["note"])
            self.assertEqual({l.get("voice") for l in call[1:-1]}, {"af_heart"})
            self.assertEqual(checks.call_count, 1)  # the call never reached the clone
            with mock.patch("backend_service.mcp_bridge.access.token_ok", return_value=True):
                agent, _ = self.stream("clone:1234567890_deadbeef", headers={"x-chaosengine-token": "agent-token"})
        self.assertEqual(agent.status_code, 403)
        self.assertEqual(mark.call_count, 2)

    def test_a_cloned_voice_whose_permission_record_is_gone_falls_back(self) -> None:
        with mock.patch.object(voice_clone, "get_voice", return_value=None), self.kokoro():
            _r, lines = self.stream("clone:1234567890_deadbeef")
        self.assertIn("permission record is missing", lines[0]["note"])

    def test_voices_say_where_they_may_speak_and_what_chat_costs(self) -> None:
        voices = {v["id"]: v for v in voice_speak.list_voices()}
        self.assertEqual(voices["af_heart"]["chatUse"], {"read": True, "call": True})
        self.assertEqual(voices["qwen3:Ryan"]["chatUse"], {"read": True, "call": True})
        self.assertEqual(voices["design"]["chatUse"], {"read": False, "call": False})
        self.assertEqual(voice_speak.chat_uses("clone"), {"read": True, "call": False})
        self.assertEqual(voices["af_heart"]["chatCost"]["peakGb"], voice_models.KOKORO_CHAT_ESTIMATES["chatPeakGb"])
        self.assertFalse(voices["af_heart"]["chatCost"]["slowForCalls"])
        self.assertIsNotNone(voices["chatterbox-turbo"]["chatCost"]["firstAudioColdS"])
        with mock.patch.dict(voice_models.voice_model("chatterbox-turbo")["estimates"], {"chatFirstAudioS": 2.5}):
            self.assertTrue(voice_speak.chat_cost("chatterbox-turbo")["slowForCalls"])
        self.assertIsNone(voice_speak.chat_cost("orpheus-3b"))  # no chat measurement yet

    def test_agents_never_default_to_a_cloned_chat_voice(self) -> None:
        from backend_service.mcp_bridge import tools

        jobs = []
        with mock.patch.object(voice_runtime, "load_settings", return_value={"ttsVoice": "clone:1234567890_deadbeef", "speed": 1.0}), \
                mock.patch.object(tools.STORE, "submit", side_effect=lambda *a, **k: jobs.append(a[2]) or {"id": "j"}), \
                mock.patch.object(tools, "_submitted", side_effect=lambda job: job), \
                mock.patch.object(voice_speak, "speak", side_effect=RuntimeError("stop")) as speak:
            tools.speak(None, {"text": "Hi"})
            with self.assertRaises(RuntimeError):
                jobs[0]()
        self.assertEqual(speak.call_args.args[1], voice_runtime.DEFAULT_VOICE)


class DownloadTests(unittest.TestCase):
    def test_a_speaking_model_downloads_without_the_watermark_component(self) -> None:
        class SyncThread:
            def __init__(self, target, **_kwargs):
                self.target = target

            def start(self):
                self.target()

        install = mock.Mock()
        with mock.patch.object(voice_clone, "available_here", return_value=True), \
             mock.patch.object(voice_clone, "_fetch_repo") as weights, \
             mock.patch.object(voice_env, "watermark_installed", return_value=False), \
             mock.patch.object(voice_env, "install_watermark", install), \
             mock.patch.object(voice_clone.threading, "Thread", SyncThread):
            voice_clone._DOWNLOADS.clear()
            voice_clone.start_download("qwen3-tts-custom-1.7b")
        self.assertEqual(weights.call_args_list[0].args[2], voice_models.voice_model("qwen3-tts-custom-1.7b")["revision"])
        install.assert_not_called()
        self.assertEqual(voice_clone._DOWNLOADS["qwen3-tts-custom-1.7b"]["state"], "done")
        voice_clone._DOWNLOADS.clear()


class AgentToolTests(unittest.TestCase):
    def test_speak_tool_schema_has_style_and_pitch(self) -> None:
        from backend_service.mcp_bridge.tools import TOOLS

        props = next(t for t in TOOLS if t["name"] == "speak")["inputSchema"]["properties"]
        self.assertEqual((props["pitch"]["minimum"], props["pitch"]["maximum"]), (-12, 12))
        self.assertEqual(props["style"]["maxLength"], 500)

    def test_list_voices_tool_adds_the_kind_and_only_ready_voices(self) -> None:
        from backend_service.mcp_bridge import models

        with Machine(), mock.patch.object(voice_clone, "snapshot_dir", installed("qwen3-tts-custom-0.6b")):
            voices = models.list_voices()
        kinds = {v["kind"] for v in voices}
        self.assertEqual(kinds, {"builtin", "preset"})
        self.assertTrue(all({"id", "name", "gender", "language", "kind"} <= set(v) for v in voices))


if __name__ == "__main__":
    unittest.main()
