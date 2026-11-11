"""Qwen3-TTS on an NVIDIA GPU (the CUDA lane): install commands, catalog rows, routing next to the Apple rows, the
GPU gate (memory and compute capability read from the driver, never assumed), and the worker's pure parts. Nothing
here needs a GPU; the live run is in docs/progress/rtx4090-progress.md."""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:  # imported before any sys.modules patch: torch cannot be imported twice in one process
    import torch
except ImportError:  # a box without torch skips the two loader tests
    torch = None

from backend_service import voice_clone, voice_env, voice_gpu, voice_speak
from backend_service import voice_worker_torch as wt
from backend_service.catalog import voice_models

QWEN_CUDA = {
    "qwen3-tts-0.6b-cuda": ("base", "qwen3-tts-0.6b"),
    "qwen3-tts-1.7b-cuda": ("base", "qwen3-tts-1.7b"),
    "qwen3-tts-custom-0.6b-cuda": ("custom", "qwen3-tts-custom-0.6b"),
    "qwen3-tts-custom-1.7b-cuda": ("custom", "qwen3-tts-custom-1.7b"),
    "qwen3-tts-design-1.7b-cuda": ("design", "qwen3-tts-design-1.7b"),
}


class InstallTests(unittest.TestCase):
    def test_qwen_tts_goes_into_its_own_folder_with_the_transformers_it_pins(self) -> None:
        # qwen-tts 0.1.1 needs transformers 4.57.3 (on the add-on's 5.2 it loads and speaks hum, RTX 4090 run);
        # Chatterbox pins 5.2.0. The two live side by side: the venv keeps 5.2, this folder holds 4.57.
        self.assertIn("transformers==4.57.3", voice_env.QWEN_TTS_SITE_PACKAGES)
        self.assertIn("huggingface-hub==0.36.0", voice_env.QWEN_TTS_SITE_PACKAGES)  # 4.57 refuses hub 1.x
        self.assertIn("qwen-tts==0.1.1", voice_env.QWEN_TTS_SITE_PACKAGES)
        self.assertTrue(all("==" in p for p in voice_env.QWEN_TTS_SITE_PACKAGES))  # every package pinned
        commands = voice_env.speech_gpu_commands("/venv/bin/python")
        install = next(c for c in commands if "--target" in c)
        self.assertEqual(install[install.index("--target") + 1], str(voice_env.qwen_site_dir()))
        self.assertIn("--no-deps", install)
        self.assertEqual(install[-len(voice_env.QWEN_TTS_SITE_PACKAGES):], voice_env.QWEN_TTS_SITE_PACKAGES)
        everything = " ".join(" ".join(c) for c in commands)
        self.assertNotIn("gradio", everything)
        # The add-on's own transformers is untouched.
        self.assertIn("transformers==5.2.0", commands[1])

    def test_the_folder_is_inside_the_voice_venv_so_removing_the_venv_removes_it(self) -> None:
        self.assertEqual(voice_env.qwen_site_dir().parent, voice_env.VOICE_VENV_DIR)

    def test_the_last_check_imports_qwen_tts_from_that_folder_and_asserts_the_version(self) -> None:
        check = voice_env.speech_gpu_commands("/venv/bin/python")[-1]
        self.assertEqual(check[:3], ["/venv/bin/python", "-I", "-c"])
        self.assertIn("qwen_tts", check[3])
        self.assertIn("4.57.3", check[3])
        self.assertIn("qwen-site", check[3])

    def test_the_qwen_worker_gets_the_folder_first_on_pythonpath_and_stays_offline(self) -> None:
        with mock.patch.dict("os.environ", {"PYTHONPATH": "/the/backends/own/packages"}):
            env = voice_env.qwen_worker_env()
            full = {**voice_env.worker_env(), **env}
        # Only the Qwen folder: the backend's own PYTHONPATH (another torch and transformers) must not leak in.
        self.assertEqual(env["PYTHONPATH"], str(voice_env.qwen_site_dir()))
        self.assertEqual(full["PYTHONPATH"], str(voice_env.qwen_site_dir()))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")  # it loads pinned local snapshots only

    def test_installed_means_the_package_and_the_pinned_transformers_are_in_the_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(voice_env, "VOICE_VENV_DIR", Path(tmp)):
            self.assertFalse(voice_env.qwen_installed())
            site = voice_env.qwen_site_dir()
            (site / "qwen_tts").mkdir(parents=True)
            self.assertFalse(voice_env.qwen_installed())
            (site / "transformers-4.57.3.dist-info").mkdir()
            self.assertTrue(voice_env.qwen_installed())


class RemoveTests(unittest.TestCase):
    def test_unticking_the_add_on_removes_the_qwen_folder_and_only_that_folder(self) -> None:
        from backend_service import voice_runtime_disk

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(voice_env, "VOICE_VENV_DIR", Path(tmp)):
            site = voice_env.qwen_site_dir()
            (site / "qwen_tts").mkdir(parents=True)
            keep = Path(tmp) / "Lib" / "site-packages"
            keep.mkdir(parents=True)
            voice_runtime_disk.remove_qwen_site()
            self.assertFalse(site.exists())
            self.assertTrue(keep.is_dir())
            voice_runtime_disk.remove_qwen_site()  # nothing there: no error
        with mock.patch.object(voice_env, "qwen_site_dir", return_value=Path("/home/x/.chaosengine")):
            with self.assertRaisesRegex(RuntimeError, "Refusing"):
                voice_runtime_disk.remove_qwen_site()


class CatalogTests(unittest.TestCase):
    def test_each_gpu_row_pins_the_upstream_repo_and_matches_its_apple_twin(self) -> None:
        for key, (kind, twin) in QWEN_CUDA.items():
            row = voice_models.voice_model(key)
            apple = voice_models.voice_model(twin)
            with self.subTest(key=key):
                self.assertEqual((row["lanes"], row["engine"], row["family"], row["kind"]),
                                 (["cuda"], "torch", "qwen3-tts", kind))
                self.assertEqual(row["repo"], row["upstream"])
                self.assertEqual(row["upstream"], apple["upstream"])
                self.assertRegex(row["revision"], r"^[0-9a-f]{40}$")
                self.assertEqual(row["capabilities"], apple["capabilities"])
                self.assertEqual(row["licence"]["id"], "apache-2.0")
                self.assertIn("speech_tokenizer/model.safetensors", row["files"])
                self.assertIn("model.safetensors", row["files"])
                self.assertNotIn("README.md", row["files"])
                self.assertTrue(row["listed"])
                self.assertEqual(row["companions"], [])

    def test_the_clone_default_is_the_same_model_on_both_lanes(self) -> None:
        defaults = {r["key"] for r in voice_models.VOICE_MODELS if r["default"]}
        self.assertEqual(defaults, {"qwen3-tts-0.6b", "qwen3-tts-0.6b-cuda"})

    def test_measured_numbers_are_there_and_the_gate_clears_the_peak(self) -> None:
        for key in QWEN_CUDA:
            row = voice_models.voice_model(key)
            est = row["estimates"]
            with self.subTest(key=key):
                self.assertEqual(est["basis"], "measured")
                self.assertIn("RTX 4090", est["measuredOn"])
                self.assertGreater(est["secondsPer10sAudio"], 0)
                self.assertGreater(est["peakGb"], 0)
                self.assertGreaterEqual(row["minVramGb"], est["peakGb"])  # the gate never refuses a card that ran it
                self.assertEqual(row["minComputeCap"], 7.0)

    def test_a_mac_does_not_list_the_gpu_rows(self) -> None:
        with mock.patch.object(voice_env, "lane", return_value="mlx"):
            listed = {r["key"] for r in voice_models.listed_models()}
        self.assertFalse(set(QWEN_CUDA) & listed)
        with mock.patch.object(voice_env, "lane", return_value="portable"):
            listed = {r["key"] for r in voice_models.listed_models()}
        self.assertTrue(set(QWEN_CUDA) <= listed)


def _gpu_box(*, installed: tuple[str, ...] = tuple(QWEN_CUDA), vram: float | None = 24.0, cap: float | None = 8.9):
    """Patches for a Windows or Linux box with the speech GPU add-on and the given GPU."""
    return [
        mock.patch.object(voice_env, "lane", return_value="portable"),
        mock.patch.object(voice_env, "speech_lanes", return_value=["cuda"]),
        mock.patch.object(voice_env, "is_installed", return_value=True),
        mock.patch.object(voice_gpu, "vram_total_gb", return_value=vram),
        mock.patch.object(voice_gpu, "compute_capability", return_value=cap),
        mock.patch.object(voice_clone, "snapshot_dir",
                          side_effect=lambda row: Path("/snap") / row["key"] if row["key"] in installed else None),
    ]


class _Box:
    def __init__(self, **kwargs) -> None:
        self.patches = _gpu_box(**kwargs)

    def __enter__(self):
        for patch in self.patches:
            patch.start()
        return self

    def __exit__(self, *exc) -> None:
        for patch in reversed(self.patches):
            patch.stop()


class RoutingTests(unittest.TestCase):
    def test_a_preset_voice_with_a_style_goes_to_the_1_7b_gpu_row(self) -> None:
        with _Box():
            plan = voice_speak.build_plan("qwen3:Ryan", "sound sunny and fast")
        self.assertEqual((plan.model_key, plan.engine, plan.family), ("qwen3-tts-custom-1.7b-cuda", "torch", "qwen3-tts"))
        self.assertEqual(plan.source, str(Path("/snap") / "qwen3-tts-custom-1.7b-cuda"))
        self.assertEqual(plan.extra["device"], "cuda")
        self.assertEqual(plan.extra["family"], "qwen3-tts")
        self.assertEqual((plan.extra["mode"], plan.extra["speaker"], plan.extra["instruct"]),
                         ("preset", "Ryan", "sound sunny and fast"))

    def test_the_0_6b_row_ignores_the_style_and_says_so(self) -> None:
        with _Box(installed=("qwen3-tts-custom-0.6b-cuda",)):
            plan = voice_speak.build_plan("qwen3:Aiden", "whisper")
        self.assertEqual(plan.model_key, "qwen3-tts-custom-0.6b-cuda")
        self.assertNotIn("instruct", plan.extra)
        self.assertTrue(any("0.6B" in note for note in plan.notes))

    def test_a_described_voice_goes_to_voicedesign(self) -> None:
        with _Box():
            plan = voice_speak.build_plan("design", "A calm older man with a slow pace")
        self.assertEqual((plan.model_key, plan.engine), ("qwen3-tts-design-1.7b-cuda", "torch"))
        self.assertEqual((plan.extra["mode"], plan.extra["instruct"], plan.extra["language"]),
                         ("design", "A calm older man with a slow pace", "auto"))

    def test_the_apple_rows_never_shadow_the_gpu_rows(self) -> None:
        with _Box():
            keys = [r["key"] for r in voice_speak._rows(lambda r: r["family"] == "qwen3-tts")]
        self.assertEqual(sorted(keys), sorted(QWEN_CUDA))

    def test_without_the_add_on_every_row_is_still_listed_so_the_screen_can_say_what_to_install(self) -> None:
        with mock.patch.object(voice_env, "lane", return_value="portable"), \
                mock.patch.object(voice_env, "speech_lanes", return_value=[]):
            keys = {r["key"] for r in voice_speak._rows(lambda r: r["family"] == "qwen3-tts")}
        self.assertTrue(set(QWEN_CUDA) <= keys and "qwen3-tts-custom-1.7b" in keys)

    def test_a_clone_runs_on_the_gpu_default_and_sends_the_sample(self) -> None:
        with _Box(installed=("qwen3-tts-0.6b-cuda",)):
            self.assertEqual(voice_clone.pick_clone_model("")["key"], "qwen3-tts-0.6b-cuda")
            row = voice_models.voice_model("qwen3-tts-0.6b-cuda")
            fields, notes = voice_clone.clone_fields(row, "/v/ref.wav", "Hello there.", "", "auto", 0.5, 0.5)
            plan = voice_speak.plan_for("clone", "clone:v1", row, fields)
        self.assertEqual((fields["refAudioPath"], fields["refText"]), ("/v/ref.wav", "Hello there."))
        self.assertEqual(notes, [])
        self.assertEqual((plan.engine, plan.extra["family"], plan.extra["device"]), ("torch", "qwen3-tts", "cuda"))

    def test_the_apple_requests_are_unchanged(self) -> None:
        # Apple rows send no family or device: their worker call is picked from the mode.
        self.assertEqual(voice_speak.worker_fields(voice_models.voice_model("qwen3-tts-custom-1.7b")), {})

    def test_the_agent_tool_takes_the_gpu_clone_models(self) -> None:
        import backend_service.mcp_bridge.tools  # noqa: F401 - loaded first, the two import each other
        from backend_service.mcp_bridge.voice_clone_tool import CLONE_TOOLS

        tool = next(t for t in CLONE_TOOLS if t["name"] == "speak_cloned")
        allowed = tool["inputSchema"]["properties"]["model"]["enum"]
        self.assertIn("qwen3-tts-0.6b-cuda", allowed)
        self.assertIn("qwen3-tts-1.7b-cuda", allowed)


class _FakeWorker:
    """What ``voice_client`` needs of a worker."""

    def __init__(self, tts: str = "", running: bool = True, busy: bool = False) -> None:
        self.loaded = {"tts": tts} if tts else {}
        self.pid = 4242 if running else None
        self.busy = busy
        self.calls: list[tuple] = []

    def request(self, cmd, timeout=300.0, **payload):
        self.calls.append((cmd, payload))
        if cmd == "unload":
            self.loaded = {}
        return {"ok": True}

    def stop(self) -> None:
        self.calls.append(("stop", {}))

    def interrupt(self) -> None:
        self.calls.append(("interrupt", {}))


class WorkerRoutingTests(unittest.TestCase):
    def _clients(self, main: _FakeWorker, qwen: _FakeWorker):
        from backend_service import voice_client

        return voice_client, [
            mock.patch.object(voice_client, "_WORKER", main),
            mock.patch.object(voice_client, "_QWEN_WORKER", qwen),
        ]

    def _run(self, main, qwen, test) -> None:
        client, patches = self._clients(main, qwen)
        for patch in patches:
            patch.start()
        try:
            test(client)
        finally:
            for patch in reversed(patches):
                patch.stop()

    def test_qwen_on_the_gpu_has_its_own_worker_and_everything_else_the_main_one(self) -> None:
        main, qwen = _FakeWorker(), _FakeWorker()

        def check(client):
            self.assertTrue(client.uses_qwen_worker("torch", "qwen3-tts"))
            for engine, family in (("torch", "chatterbox"), ("torch", "voxcpm2"), ("llama-gguf", "orpheus"), ("mlx-audio", "qwen3-tts")):
                self.assertFalse(client.uses_qwen_worker(engine, family), (engine, family))
            self.assertIs(client.speech_worker_for("torch", "qwen3-tts"), qwen)
            self.assertIs(client.speech_worker_for("torch", "chatterbox"), main)
            self.assertIs(client.speech_worker_for("mlx-audio", "qwen3-tts"), main)

        self._run(main, qwen, check)

    def test_a_model_held_by_the_other_worker_is_unloaded_first_so_two_never_share_the_gpu(self) -> None:
        main, qwen = _FakeWorker(tts="/snap/chatterbox-turbo-cuda"), _FakeWorker()

        def check(client):
            self.assertIs(client.speech_worker_for("torch", "qwen3-tts"), qwen)
            self.assertEqual(main.calls, [("unload", {"what": "tts"})])
            self.assertEqual(main.loaded, {})

        self._run(main, qwen, check)
        main, qwen = _FakeWorker(), _FakeWorker(tts="/snap/qwen3-tts-custom-1.7b-cuda")

        def other_way(client):
            client.speech_worker_for("torch", "voxcpm2")
            self.assertEqual(qwen.calls, [("unload", {"what": "tts"})])

        self._run(main, qwen, other_way)

    def test_kokoro_stays_loaded_and_a_stopped_worker_is_left_alone_and_a_busy_one_is_never_interrupted(self) -> None:
        main, qwen = _FakeWorker(tts="C:/Users/x/.chaosengine/voice/kokoro"), _FakeWorker()

        def kokoro(client):
            client.speech_worker_for("torch", "qwen3-tts")
            self.assertEqual(main.calls, [])

        self._run(main, qwen, kokoro)
        main, qwen = _FakeWorker(tts="/snap/x", running=False), _FakeWorker()
        self._run(main, qwen, lambda client: (client.speech_worker_for("torch", "qwen3-tts"), self.assertEqual(main.calls, [])))
        main, qwen = _FakeWorker(tts="/snap/x", busy=True), _FakeWorker()

        def busy(client):
            with self.assertRaisesRegex(client.VoiceWorkerError, "Another voice is speaking"):
                client.speech_worker_for("torch", "qwen3-tts")
            self.assertEqual(main.calls, [])

        self._run(main, qwen, busy)

    def test_stop_interrupt_busy_and_in_memory_cover_both_speech_workers(self) -> None:
        main, qwen = _FakeWorker(tts="/snap/a"), _FakeWorker(tts="/snap/b", busy=True)

        def check(client):
            self.assertTrue(client.worker_busy())
            self.assertEqual(client.loaded_sources(), {"tts": "/snap/b"})
            client.interrupt_all()
            client.stop_speech_worker()
            self.assertIn(("interrupt", {}), main.calls)
            self.assertIn(("interrupt", {}), qwen.calls)
            self.assertIn(("stop", {}), main.calls)
            self.assertIn(("stop", {}), qwen.calls)

        self._run(main, qwen, check)

    def test_speaking_goes_to_the_worker_the_plan_names(self) -> None:
        from backend_service import voice_client

        qwen = _FakeWorker()
        qwen.request = mock.Mock(return_value={"wavB64": base64.b64encode(b"RIFF").decode(), "seconds": 1.5})
        plan = voice_speak.Plan("preset", "qwen3:Ryan", "qwen3-tts-custom-1.7b-cuda", "Qwen", "torch", "/snap/q",
                                {"family": "qwen3-tts", "device": "cuda", "mode": "preset", "speaker": "Ryan"},
                                family="qwen3-tts")
        with mock.patch.object(voice_client, "_QWEN_WORKER", qwen), mock.patch.object(voice_client, "_WORKER", None):
            wav, took = voice_speak._speak_chunk(plan, "Hello.", 1.0, 7)
        self.assertEqual((wav, took), (b"RIFF", 1.5))
        args = qwen.request.call_args
        self.assertEqual(args.args, ("synthesize",))
        self.assertEqual((args.kwargs["engine"], args.kwargs["source"], args.kwargs["family"], args.kwargs["seed"]),
                         ("torch", "/snap/q", "qwen3-tts", 7))


class GpuGateTests(unittest.TestCase):
    def setUp(self) -> None:
        voice_gpu.reset()

    def test_problem_reads_the_memory_and_the_compute_capability(self) -> None:
        row = {"name": "Big", "minVramGb": 9.0, "minComputeCap": 7.0}
        cases = [
            (24.0, 8.9, None),        # RTX 4090
            (32.0, 12.0, None),       # RTX 5090
            (8.0, 8.6, "9 GB"),       # 8 GB card: too small
            (16.0, 6.1, "compute capability 7"),  # Pascal: too old
            (None, 8.9, None),        # unified memory or unreadable: never blocks
            (24.0, None, None),
        ]
        for vram, cap, expected in cases:
            with self.subTest(vram=vram, cap=cap), \
                    mock.patch.object(voice_gpu, "vram_total_gb", return_value=vram), \
                    mock.patch.object(voice_gpu, "compute_capability", return_value=cap):
                why = voice_gpu.problem(row)
                if expected is None:
                    self.assertIsNone(why)
                else:
                    self.assertIn(expected, why)

    def test_facts_are_data_for_the_screen_compute_first(self) -> None:
        row = {"name": "Big", "minVramGb": 9.0, "minComputeCap": 7.0}
        for vram, cap, expected in ((8.0, 8.6, {"kind": "memory", "need": 9.0, "have": 8}),
                                    (16.0, 6.1, {"kind": "compute", "need": 7.0, "have": 6.1}),
                                    (4.0, 6.1, {"kind": "compute", "need": 7.0, "have": 6.1}),
                                    (24.0, 8.9, None)):
            with self.subTest(vram=vram, cap=cap), mock.patch.object(voice_gpu, "vram_total_gb", return_value=vram),                     mock.patch.object(voice_gpu, "compute_capability", return_value=cap):
                self.assertEqual(voice_gpu.facts(row), expected)

    def test_a_row_with_no_stated_need_is_never_checked(self) -> None:
        with mock.patch.object(voice_gpu, "vram_total_gb", side_effect=AssertionError), \
                mock.patch.object(voice_gpu, "compute_capability", side_effect=AssertionError):
            self.assertIsNone(voice_gpu.problem({"name": "Apple row"}))

    def test_compute_capability_comes_from_nvidia_smi_and_is_cached(self) -> None:
        done = types.SimpleNamespace(returncode=0, stdout="8.9\n")
        with mock.patch.object(voice_gpu.subprocess, "run", return_value=done) as run:
            self.assertEqual(voice_gpu.compute_capability(), 8.9)
            self.assertEqual(voice_gpu.compute_capability(), 8.9)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][0], "nvidia-smi")  # list form, no shell
        voice_gpu.reset()
        with mock.patch.object(voice_gpu.subprocess, "run", side_effect=OSError):
            self.assertIsNone(voice_gpu.compute_capability())

    def test_an_8_gb_card_runs_every_row_and_a_4_gb_card_only_the_0_6b_custom_one(self) -> None:
        everything = set(QWEN_CUDA)
        with _Box(vram=8.0, cap=8.6):
            keys = {r["key"] for r in voice_speak._rows(lambda r: r["family"] == "qwen3-tts")}
            self.assertEqual(keys, everything)  # the measured 1.7B peak is 5.3 GB at most
        with _Box(vram=4.0, cap=8.6):
            keys = {r["key"] for r in voice_speak._rows(lambda r: r["family"] == "qwen3-tts")}
            self.assertIn("qwen3-tts-custom-0.6b-cuda", keys)
            self.assertNotIn("qwen3-tts-custom-1.7b-cuda", keys)
            plan = voice_speak.build_plan("qwen3:Ryan", "")
            self.assertEqual(plan.model_key, "qwen3-tts-custom-0.6b-cuda")
            with self.assertRaisesRegex(ValueError, "GB of GPU memory"):
                voice_speak.build_plan("qwen3:Ryan", "", model_key="qwen3-tts-custom-1.7b-cuda")
            with self.assertRaisesRegex(ValueError, "GB of GPU memory"):
                voice_clone.start_download("qwen3-tts-1.7b-cuda")

    def test_the_models_list_says_why_a_row_cannot_run_on_this_gpu(self) -> None:
        with _Box(vram=3.0, cap=8.6), mock.patch.object(voice_clone, "_ready", return_value=False):
            rows = {r["key"]: r for r in voice_clone.list_models(every_model=True)}
        small = rows["qwen3-tts-custom-0.6b-cuda"]
        self.assertFalse(small["supported"])
        self.assertIn("GB of GPU memory", small["unsupportedReason"])
        self.assertEqual(small["unsupportedFacts"]["kind"], "memory")
        with _Box(vram=24.0, cap=8.9), mock.patch.object(voice_clone, "_ready", return_value=False):
            rows = {r["key"]: r for r in voice_clone.list_models(every_model=True)}
        self.assertTrue(rows["qwen3-tts-custom-1.7b-cuda"]["supported"])
        self.assertIsNone(rows["qwen3-tts-custom-1.7b-cuda"]["unsupportedReason"])


class WorkerTests(unittest.TestCase):
    def _model(self):
        model = mock.Mock()
        wave = np.zeros(2400, dtype=np.float32)
        for name in ("generate_custom_voice", "generate_voice_design", "generate_voice_clone"):
            getattr(model, name).return_value = ([wave], 24_000)
        return model

    def test_a_preset_speaker_with_and_without_a_style(self) -> None:
        model = self._model()
        samples, rate = wt.qwen_call(model, {"mode": "preset", "text": "Hi.", "speaker": "Ryan", "language": "english",
                                             "instruct": "cheerful"})
        self.assertEqual((len(samples), rate), (2400, 24_000))
        self.assertEqual(model.generate_custom_voice.call_args.kwargs,
                         {"text": "Hi.", "speaker": "Ryan", "language": "english", "instruct": "cheerful"})
        wt.qwen_call(model, {"mode": "preset", "text": "Hi.", "speaker": "Ryan"})
        self.assertEqual(model.generate_custom_voice.call_args.kwargs["language"], "auto")
        self.assertIsNone(model.generate_custom_voice.call_args.kwargs["instruct"])

    def test_a_described_voice(self) -> None:
        model = self._model()
        wt.qwen_call(model, {"mode": "design", "text": "Hi.", "instruct": "A calm woman"})
        self.assertEqual(model.generate_voice_design.call_args.kwargs,
                         {"text": "Hi.", "instruct": "A calm woman", "language": "auto"})

    def test_a_voice_sample_with_a_transcript_clones_in_context_and_without_one_uses_the_speaker_embedding(self) -> None:
        model = self._model()
        with mock.patch.object(wt, "readable_audio", return_value="/tmp/ref.wav") as readable:
            wt.qwen_call(model, {"text": "Hi.", "refAudioPath": "/v/ref.m4a", "refText": " Hello there. "})
            kwargs = model.generate_voice_clone.call_args.kwargs
            self.assertEqual((kwargs["ref_audio"], kwargs["ref_text"], kwargs["x_vector_only_mode"]),
                             ("/tmp/ref.wav", "Hello there.", False))
            wt.qwen_call(model, {"text": "Hi.", "refAudioPath": "/v/ref.m4a", "refText": ""})
            kwargs = model.generate_voice_clone.call_args.kwargs
            self.assertEqual((kwargs["ref_text"], kwargs["x_vector_only_mode"]), (None, True))
        readable.assert_called_with("/v/ref.m4a")  # an m4a is decoded first, like the other GPU engines

    def test_a_request_with_no_voice_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "preset voice"):
            wt.qwen_call(self._model(), {"text": "Hi."})

    @unittest.skipUnless(torch, "torch is not installed here")
    def test_the_loader_reads_the_local_snapshot_straight_onto_the_gpu(self) -> None:
        loader = mock.Mock(return_value="wrapper")
        fake = types.ModuleType("qwen_tts")
        fake.Qwen3TTSModel = types.SimpleNamespace(from_pretrained=loader)
        for bf16, expected in ((True, torch.bfloat16), (False, torch.float16)):
            with self.subTest(bf16=bf16), mock.patch.dict(sys.modules, {"qwen_tts": fake}),                     mock.patch.object(torch.cuda, "is_bf16_supported", return_value=bf16):
                self.assertEqual(wt.load_qwen("/snap/qwen", "cuda"), "wrapper")
                self.assertEqual(loader.call_args.args, ("/snap/qwen",))
                kwargs = loader.call_args.kwargs
                self.assertTrue(kwargs["local_files_only"])  # never fetched by repo id
                self.assertEqual(kwargs["attn_implementation"], "sdpa")  # flash-attn is not installed
                self.assertEqual(kwargs["device_map"], "cuda:0")
                self.assertEqual(kwargs["dtype"], expected)  # a card without bf16 gets fp16
        with mock.patch.dict(sys.modules, {"qwen_tts": fake}):
            wt.load_qwen("/snap/qwen", "cuda:1")
            self.assertEqual(loader.call_args.kwargs["device_map"], "cuda:1")

    @unittest.skipUnless(torch, "torch is not installed here")
    def test_an_add_on_from_before_qwen3_tts_says_how_to_update_it(self) -> None:
        with mock.patch.dict(sys.modules, {"qwen_tts": None}):  # None makes the import fail
            with self.assertRaisesRegex(RuntimeError, "Add-ons to update it"):
                wt.load_qwen("/snap/qwen", "cuda")

    def test_synthesize_loads_the_family_and_calls_qwen(self) -> None:
        wt.unload()
        model = self._model()
        with mock.patch.object(wt, "load_qwen", return_value=model) as load:
            samples, rate = wt.synthesize({"engine": "torch", "family": "qwen3-tts", "source": "/s", "device": "cuda",
                                           "mode": "design", "text": "Hi.", "instruct": "A calm man", "seed": None})
            wt.synthesize({"engine": "torch", "family": "qwen3-tts", "source": "/s", "device": "cuda",
                           "mode": "design", "text": "Again.", "instruct": "A calm man"})
        load.assert_called_once_with("/s", "cuda")  # the second request reuses the loaded model
        self.assertEqual(rate, 24_000)
        wt.unload()


if __name__ == "__main__":
    unittest.main()
