"""The speech GPU add-on and CUDA rows (D-70 Phase 6, FU-149): install commands per OS, catalog rows, routing,
the PerTh note in the AI tag, and the GPU worker's pure parts. Nothing here runs on a GPU; the RTX 4090 session
does (docs/plans/rtx4090-test.md section 2)."""

from __future__ import annotations

import base64
import io
import importlib
import os
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy  # noqa: F401 - loaded before any sys.modules patch

from backend_service import voice_clone, voice_clone_safety, voice_env, voice_library, voice_speak
from backend_service import voice_worker_torch as wt
from backend_service.catalog import voice_models, voice_models_cuda

CUDA_KEYS = ["chatterbox-turbo-cuda", "chatterbox-mtl-cuda", "voxcpm2-cuda", "orpheus-3b-gguf"]


def make_wav(rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x10\x00" * 2400)
    return buffer.getvalue()


class InstallTests(unittest.TestCase):
    def test_commands_install_cuda_torch_then_the_hand_picked_set_then_the_models_without_deps(self) -> None:
        commands = voice_env.speech_gpu_commands("/venv/bin/python")
        self.assertEqual(commands[0], ["/venv/bin/python", "-m", "pip", "install", "torch==2.9.1", "torchaudio==2.9.1",
                                       "--index-url", "https://download.pytorch.org/whl/cu128"])
        self.assertEqual(commands[1][:4], ["/venv/bin/python", "-m", "pip", "install"])
        self.assertEqual(commands[2], ["/venv/bin/python", "-m", "pip", "install", "--no-deps", "chatterbox-tts==0.1.7", "voxcpm==2.0.3"])
        self.assertEqual(commands[3][:3], ["/venv/bin/python", "-I", "-c"])
        self.assertIn("torch.cuda.is_available()", commands[3][3])
        names = " ".join(commands[1]).lower()
        for unwanted in ("gradio", "funasr", "modelscope", "datasets", "matplotlib", "pykakasi", "vllm"):
            self.assertNotIn(unwanted, names)
        self.assertTrue(all(isinstance(part, str) for command in commands for part in command))  # list form only

    def test_windows_path_uses_the_venv_python(self) -> None:
        commands = voice_env.speech_gpu_commands(r"C:\Users\x\.chaosengine\voice-venv\Scripts\python.exe")
        self.assertTrue(all(c[0].endswith("python.exe") for c in commands))

    def test_supported_only_on_windows_or_linux_x86_with_an_nvidia_gpu(self) -> None:
        cases = [
            ("mlx", True, "x86_64", False),
            ("portable", False, "x86_64", False),
            ("portable", True, "aarch64", False),
            ("portable", True, "AMD64", True),
            ("portable", True, "x86_64", True),
        ]
        for lane, gpu, machine, expected in cases:
            with self.subTest(lane=lane, gpu=gpu, machine=machine), \
                    mock.patch.object(voice_env, "lane", return_value=lane), \
                    mock.patch.object(voice_env, "has_nvidia_gpu", return_value=gpu), \
                    mock.patch.object(voice_env.platform, "machine", return_value=machine):
                ok, why = voice_env.speech_gpu_supported()
                self.assertEqual(ok, expected)
                self.assertEqual(bool(why), not expected)

    def test_the_install_job_runs_the_add_on_commands_only_when_asked(self) -> None:
        ran: list[list[str]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(voice_env, "VOICE_VENV_DIR", Path(tmp)), \
                mock.patch.object(voice_env, "_MARKER", Path(tmp) / "m.json"), \
                mock.patch.object(voice_env, "lane", return_value="portable"), \
                mock.patch.object(voice_env, "has_nvidia_gpu", return_value=True), \
                mock.patch.object(voice_env.platform, "machine", return_value="x86_64"), \
                mock.patch.object(voice_env, "venv_python", return_value=Path(tmp) / "py"), \
                mock.patch.object(voice_env, "_run", side_effect=lambda c: ran.append(c)), \
                mock.patch("backend_service.voice_kokoro_onnx.is_installed", return_value=True), \
                mock.patch("huggingface_hub.hf_hub_download"):
            (Path(tmp) / "py").write_text("")  # the venv's python; _run is mocked
            python = str(Path(tmp) / "py")
            voice_env._run_install(["speech-gpu"])
            self.assertEqual(voice_env.job_status()["phase"], "done", voice_env.job_status()["error"])
            for command in voice_env.speech_gpu_commands(python):
                self.assertIn(command, ran)
            ran.clear()
            voice_env._run_install([])
            self.assertNotIn(voice_env.speech_gpu_commands(python)[0], ran)

    def test_runtime_offers_the_add_on_only_with_a_gpu(self) -> None:
        from backend_service import voice_runtime

        for gpu in (True, False):
            with mock.patch.object(voice_env, "lane", return_value="portable"), \
                    mock.patch.object(voice_env, "has_nvidia_gpu", return_value=gpu), \
                    mock.patch.object(voice_env.platform, "machine", return_value="x86_64"):
                ids = [e["id"] for e in voice_runtime.get_voice_capabilities()["extras"]]
            self.assertEqual("speech-gpu" in ids, gpu)


class CatalogTests(unittest.TestCase):
    def test_rows_are_pinned_list_their_files_and_say_which_add_their_own_watermark(self) -> None:
        for key in CUDA_KEYS:
            row = voice_models.voice_model(key)
            self.assertEqual(row["lanes"], ["cuda"])
            self.assertRegex(row["revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(row["files"])
            self.assertIn(row["licence"]["id"], ("mit", "apache-2.0"))
            self.assertIsNone(row["estimates"])  # measured on the 4090 first
        self.assertEqual(voice_models.voice_model("chatterbox-turbo-cuda")["builtinWatermark"], "perth")
        self.assertEqual(voice_models.voice_model("chatterbox-mtl-cuda")["builtinWatermark"], "perth")
        self.assertIsNone(voice_models.voice_model("voxcpm2-cuda")["builtinWatermark"])
        self.assertIn("t3_mtl23ls_v3.safetensors", voice_models.voice_model("chatterbox-mtl-cuda")["files"])
        self.assertEqual(voice_models.voice_model("orpheus-3b-gguf")["engine"], "llama-gguf")
        self.assertFalse(voice_models.voice_model("orpheus-3b-gguf")["capabilities"]["clone"])

    def test_unmeasured_cuda_rows_are_not_offered_and_never_on_a_mac(self) -> None:
        for key in CUDA_KEYS:
            self.assertFalse(voice_models.voice_model(key)["listed"])
        rows = [dict(r, listed=True) for r in voice_models.VOICE_MODELS]
        with mock.patch.object(voice_models, "VOICE_MODELS", rows):
            with mock.patch.object(voice_env, "lane", return_value="mlx"):
                self.assertFalse(any("cuda" in r["lanes"] for r in voice_models.listed_models()))
            with mock.patch.object(voice_env, "lane", return_value="portable"):
                self.assertTrue(set(CUDA_KEYS) <= {r["key"] for r in voice_models.listed_models()})

    def test_the_4090_session_switch_lists_them(self) -> None:
        with mock.patch.dict(os.environ, {"CHAOSENGINE_VOICE_UNMEASURED": "1"}):
            fresh = importlib.reload(voice_models)
            try:
                self.assertTrue(all(fresh.voice_model(k)["listed"] for k in CUDA_KEYS))
            finally:
                os.environ.pop("CHAOSENGINE_VOICE_UNMEASURED", None)
                importlib.reload(voice_models)
                importlib.reload(voice_models_cuda)

    def test_readiness_uses_the_listed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.assertFalse(voice_clone.weights_ready(path, ["a.gguf"]))
            (path / "a.gguf").write_text("x")
            self.assertTrue(voice_clone.weights_ready(path, ["a.gguf"]))


class RoutingTests(unittest.TestCase):
    def _gpu_box(self):
        rows = [dict(r, listed=True) for r in voice_models.VOICE_MODELS]
        return [
            mock.patch.object(voice_models, "listed_models", return_value=rows),
            mock.patch.object(voice_env, "lane", return_value="portable"),
            mock.patch.object(voice_env, "speech_lanes", return_value=["cuda"]),
            mock.patch.object(voice_env, "is_installed", return_value=True),
            mock.patch.object(voice_clone, "snapshot_dir", side_effect=lambda row: Path("/snap") / row["key"] if "cuda" in row["lanes"] else None),
            mock.patch.object(voice_clone, "companion_file", return_value=Path("/c/onnx/decoder_model.onnx")),
            mock.patch("backend_service.inference.binaries._resolve_llama_server", return_value="/bin/llama-server"),
        ]

    def test_a_gpu_box_runs_the_cuda_row_of_each_family(self) -> None:
        patches = self._gpu_box()
        for patch in patches:
            patch.start()
        try:
            turbo = voice_speak.build_plan("chatterbox-turbo")
            orpheus = voice_speak.build_plan("orpheus:zoe")
            design = voice_speak.build_plan("design", style="A calm man", model_key="voxcpm2-cuda")
            mtl = voice_speak.build_plan("chatterbox", language="fr")
        finally:
            for patch in reversed(patches):
                patch.stop()
        self.assertEqual((turbo.model_key, turbo.engine, turbo.extra["device"]), ("chatterbox-turbo-cuda", "torch", "cuda"))
        self.assertEqual(orpheus.engine, "llama-gguf")
        self.assertEqual(orpheus.extra["gguf"], "/snap/orpheus-3b-gguf/orpheus-3b-0.1-ft-Q8_0.gguf")
        self.assertEqual(orpheus.extra["snacDecoderPath"], "/c/onnx/decoder_model.onnx")
        self.assertEqual(orpheus.extra["llamaServerPath"], "/bin/llama-server")
        self.assertEqual(orpheus.tag_style, "angle")
        self.assertEqual(design.extra["instruct"], "A calm man")
        self.assertEqual((mtl.model_key, mtl.extra["language"]), ("chatterbox-mtl-cuda", "fr"))
        self.assertNotIn("defaultVoicePath", mtl.extra)  # conds.pt is in the upstream folder

    def test_cloning_is_open_on_a_gpu_box_and_closed_without_one(self) -> None:
        with mock.patch.object(voice_env, "speech_lanes", return_value=["cuda"]):
            self.assertTrue(voice_clone.available_here())
            self.assertTrue(voice_clone.runs_here(voice_models.voice_model("voxcpm2-cuda")))
            self.assertFalse(voice_clone.runs_here(voice_models.voice_model("voxcpm2")))
        with mock.patch.object(voice_env, "speech_lanes", return_value=[]):
            self.assertFalse(voice_clone.available_here())

    def test_speech_lanes(self) -> None:
        with mock.patch.object(voice_env, "lane", return_value="mlx"):
            self.assertEqual(voice_env.speech_lanes(), ["mlx"])
        with mock.patch.object(voice_env, "lane", return_value="portable"), \
                mock.patch.object(voice_env, "speech_gpu_installed", return_value=True), \
                mock.patch.object(voice_env, "speech_gpu_supported", return_value=(True, "")):
            self.assertEqual(voice_env.speech_lanes(), ["cuda"])
        with mock.patch.object(voice_env, "lane", return_value="portable"), \
                mock.patch.object(voice_env, "speech_gpu_installed", return_value=False):
            self.assertEqual(voice_env.speech_lanes(), [])


class PerthTagTests(unittest.TestCase):
    def test_cloned_speech_on_a_perth_row_records_both_marks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(voice_env, "VOICE_DATA_DIR", Path(tmp)):
            sample = Path(tmp) / "s.m4a"
            sample.write_bytes(b"x")
            voice = voice_clone.create_voice(
                speaker_name="Dan", basis="own", name="Me", sample_audio=sample, sample_ext=".m4a", sample_text=None,
                permission_confirmed=True, origin="ui", transcribe=mock.Mock(return_value={"text": "Hello there.", "durationS": 9.0}),
            )
            row = dict(voice_models.voice_model("chatterbox-turbo-cuda"), listed=True)
            worker = mock.Mock()
            worker.request.return_value = {"wavB64": base64.b64encode(make_wav()).decode(), "seconds": 0.1}
            with mock.patch.object(voice_clone, "CLONE_MODELS", [row]), mock.patch.object(voice_models, "CLONE_MODELS", [row]), \
                    mock.patch.object(voice_clone, "snapshot_dir", return_value=Path(tmp)), \
                    mock.patch.object(voice_clone, "available_here", return_value=True), \
                    mock.patch.object(voice_env, "is_installed", return_value=True), \
                    mock.patch.object(voice_env, "watermark_installed", return_value=True), \
                    mock.patch("backend_service.voice_client.get_worker", return_value=worker), \
                    mock.patch.object(voice_clone_safety, "apply_watermark", return_value=True) as mark:
                item = voice_clone.speak(voice["id"], "Hello.", "chatterbox-turbo-cuda", origin="ui")
            mark.assert_called_once()
            tag = voice_clone_safety.read_wav_tag(voice_library.audio_path(item["id"]))
            self.assertEqual(tag["builtinWatermark"], "perth")
            self.assertEqual(tag["watermark"], "audioseal-0.2.1:CE5A")
            self.assertEqual(worker.request.call_args.kwargs["engine"], "torch")


class GpuWorkerTests(unittest.TestCase):
    def test_orpheus_prompt_codes_and_snac_layers(self) -> None:
        self.assertEqual(wt.orpheus_prompt([1, 2]), [128259, 1, 2, 128009, 128260, 128261, 128257])
        frame = [wt.AUDIO_TOKEN_BASE + slot * wt.CODEBOOK + value for slot, value in enumerate([10, 11, 12, 13, 14, 15, 16])]
        tokens = [128000, *frame, *frame, wt.AUDIO_TOKEN_BASE + 5]  # a text token, two frames, a partial frame
        codes = wt.codes_from_tokens(tokens)
        self.assertEqual(codes, [10, 11, 12, 13, 14, 15, 16] * 2)
        self.assertEqual(wt.snac_layers(codes), ([10, 10], [11, 14, 11, 14], [12, 13, 15, 16, 12, 13, 15, 16]))

    def test_llama_server_command_is_local_and_list_form(self) -> None:
        command = wt.server_command("/bin/llama-server", "/m/o.gguf", 41234)
        self.assertEqual(command[:3], ["/bin/llama-server", "--model", "/m/o.gguf"])
        self.assertIn("127.0.0.1", command)
        self.assertEqual(command[command.index("--port") + 1], "41234")

    def test_chatterbox_and_voxcpm_arguments(self) -> None:
        model = mock.Mock(sr=24_000)
        model.generate.return_value = [0.0, 0.1]
        wt.chatterbox_call(model, {"family": "chatterbox", "text": "Bonjour.", "language": "fr", "expression": 0.8,
                                   "guidance": 0.2, "refAudioPath": "/clip.wav"})
        self.assertEqual(model.generate.call_args.kwargs, {"audio_prompt_path": "/clip.wav", "language_id": "fr",
                                                           "exaggeration": 0.8, "cfg_weight": 0.2, "temperature": 0.8})
        vox = mock.Mock()
        vox.tts_model.sample_rate = 48_000
        vox.generate.return_value = [0.0]
        _samples, rate = wt.voxcpm_call(vox, {"text": "Hi.", "instruct": "A calm woman"})
        self.assertEqual((vox.generate.call_args.kwargs["text"], rate), ("(A calm woman)Hi.", 48_000))
        wt.voxcpm_call(vox, {"text": "Hi.", "refAudioPath": "/c.wav", "refText": "Hello", "mode": "exact"})
        self.assertEqual(vox.generate.call_args.kwargs["prompt_wav_path"], "/c.wav")
        self.assertEqual(vox.generate.call_args.kwargs["reference_wav_path"], "/c.wav")
        wt.voxcpm_call(vox, {"text": "Hi.", "refAudioPath": "/c.wav", "mode": "steered", "instruct": "cheerful"})
        self.assertNotIn("prompt_wav_path", vox.generate.call_args.kwargs)
        self.assertEqual(vox.generate.call_args.kwargs["text"], "(cheerful)Hi.")
        with self.assertRaisesRegex(ValueError, "does not clone"):
            wt.orpheus_call({"refAudioPath": "/c.wav", "text": "x"})

    def test_the_worker_sends_gpu_engines_to_the_gpu_module(self) -> None:
        from backend_service import voice_worker

        with mock.patch.object(wt, "synthesize", return_value=([0.0] * 240, 24_000)) as gpu, \
                mock.patch.object(voice_worker, "_load_tts") as mlx:
            reply = voice_worker.synthesize({"engine": "torch", "family": "chatterbox-turbo", "source": "/s", "text": "Hi."})
        gpu.assert_called_once()
        mlx.assert_not_called()
        self.assertEqual(reply["sampleRate"], 24_000)


if __name__ == "__main__":
    unittest.main()
