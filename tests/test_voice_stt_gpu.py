"""Speech-to-text on an NVIDIA GPU (the ``stt-gpu`` add-on, ``transformers-asr`` backend): lane selection, catalogue
rows, download file lists, request guards, worker routing and the worker's own pure helpers. No network, no GPU."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend_service import voice_client, voice_env, voice_env_asr, voice_runtime, voice_runtime_disk, voice_worker
from backend_service import voice_worker_asr
from backend_service.catalog import voice_stt_cuda

GPU = "transformers-asr"


def _rows(*, ready: bool, installed: set[str] = frozenset(), facts: dict | None = None) -> list[dict]:
    with mock.patch.object(voice_env, "lane", return_value="portable"), \
            mock.patch.object(voice_env, "is_installed", return_value=True), \
            mock.patch.object(voice_env_asr, "stt_gpu_ready", return_value=ready), \
            mock.patch("backend_service.voice_gpu.facts", side_effect=lambda row: (facts or {}).get(row["repo"])), \
            mock.patch.object(voice_runtime, "_hf_snapshot_path",
                              side_effect=lambda repo: Path("snap") if repo in installed else None):
        return voice_runtime.list_stt_models()


class LaneTests(unittest.TestCase):
    def test_portable_without_the_add_on_offers_faster_whisper_only(self) -> None:
        with mock.patch.object(voice_env, "lane", return_value="portable"), \
                mock.patch.object(voice_env_asr, "stt_gpu_ready", return_value=False):
            self.assertEqual(voice_runtime._lane_backends(), ("faster-whisper",))
        rows = _rows(ready=False)
        self.assertTrue(all(r["backend"] == "faster-whisper" for r in rows))
        self.assertEqual([r["key"] for r in rows if r["default"]], ["whisper-large-v3-turbo"])

    def test_portable_with_the_add_on_puts_the_gpu_first_and_keeps_whisper(self) -> None:
        with mock.patch.object(voice_env, "lane", return_value="portable"), \
                mock.patch.object(voice_env_asr, "stt_gpu_ready", return_value=True):
            self.assertEqual(voice_runtime._lane_backends(), (GPU, "faster-whisper"))
        rows = {r["key"]: r for r in _rows(ready=True)}
        for key in ("qwen3-asr-1.7b", "qwen3-asr-0.6b", "parakeet-tdt-v3", "nemotron-3.5-asr",
                    "granite-speech-5.0-2b", "nemotron-3.5-transcribe-30b"):
            self.assertEqual(rows[key]["backend"], GPU, key)
        for key in ("whisper-large-v3-turbo", "whisper-small", "whisper-base"):
            self.assertEqual(rows[key]["backend"], "faster-whisper", key)
        self.assertEqual(list(rows)[:2], ["qwen3-asr-1.7b", "qwen3-asr-0.6b"])

    def test_mlx_lane_is_unchanged(self) -> None:
        with mock.patch.object(voice_env, "lane", return_value="mlx"), \
                mock.patch.object(voice_env, "is_installed", return_value=True), \
                mock.patch.object(voice_runtime, "_hf_snapshot_path", return_value=None):
            rows = voice_runtime.list_stt_models()
        self.assertFalse(any(r["backend"] == GPU for r in rows))
        self.assertNotIn("qwen3-asr-1.7b", {r["key"] for r in rows})
        self.assertEqual([r["key"] for r in rows if r["default"]], ["parakeet-tdt-v3"])

    def test_default_is_whisper_until_a_gpu_default_is_downloaded_and_fits(self) -> None:
        self.assertEqual([r["key"] for r in _rows(ready=True) if r["default"]], ["whisper-large-v3-turbo"])
        qwen = "Qwen/Qwen3-ASR-1.7B-hf"
        self.assertEqual([r["key"] for r in _rows(ready=True, installed={qwen}) if r["default"]], ["qwen3-asr-1.7b"])
        too_small = {qwen: {"kind": "memory", "need": 6.0, "have": 4}}
        rows = _rows(ready=True, installed={qwen}, facts=too_small)
        self.assertEqual([r["key"] for r in rows if r["default"]], ["whisper-large-v3-turbo"])
        self.assertEqual(next(r for r in rows if r["id"] == qwen)["gpuFacts"]["kind"], "memory")

    def test_resolution_by_key_and_repo_and_fallback_without_the_add_on(self) -> None:
        with mock.patch.object(voice_env, "lane", return_value="portable"), \
                mock.patch.object(voice_env, "is_installed", return_value=True), \
                mock.patch.object(voice_env_asr, "stt_gpu_ready", return_value=True), \
                mock.patch.object(voice_runtime, "_hf_snapshot_path", return_value=None):
            self.assertEqual(voice_runtime.resolve_stt_model("qwen3-asr-1.7b"), (GPU, "Qwen/Qwen3-ASR-1.7B-hf"))
            self.assertEqual(voice_runtime.resolve_stt_model("parakeet-tdt-v3"), (GPU, "nvidia/parakeet-tdt-0.6b-v3"))
            # The MLX conversion's id maps onto the GPU source of the same row.
            self.assertEqual(voice_runtime.resolve_stt_model("mlx-community/parakeet-tdt-0.6b-v3")[0], GPU)
        with mock.patch.object(voice_env, "lane", return_value="portable"), \
                mock.patch.object(voice_env, "is_installed", return_value=True), \
                mock.patch.object(voice_env_asr, "stt_gpu_ready", return_value=False), \
                mock.patch.object(voice_runtime, "_hf_snapshot_path", return_value=None):
            self.assertEqual(voice_runtime.resolve_stt_model("qwen3-asr-1.7b")[0], "faster-whisper")


class CatalogueTests(unittest.TestCase):
    EXPECTED = {
        "qwen3-asr-1.7b": ("Qwen/Qwen3-ASR-1.7B-hf", "qwen3-asr", "Apache-2.0"),
        "qwen3-asr-0.6b": ("Qwen/Qwen3-ASR-0.6B-hf", "qwen3-asr", "Apache-2.0"),
        "parakeet-tdt-v3": ("nvidia/parakeet-tdt-0.6b-v3", "parakeet-tdt", "CC-BY-4.0"),
        "nemotron-3.5-asr": ("nvidia/nemotron-3.5-asr-streaming-0.6b", "nemotron-asr", "OpenMDW-1.1"),
        "granite-speech-5.0-2b": ("ibm-granite/granite-speech-5.0-2b", "granite-speech", "Apache-2.0"),
        "nemotron-3.5-transcribe-30b": ("nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B", "nemotron-transcribe", "OpenMDW-1.1"),
    }

    def _entries(self) -> dict[str, dict]:
        return {e["key"]: e for e in voice_runtime._STT_MODELS if GPU in e["sources"]}

    def test_repos_families_and_licences(self) -> None:
        entries = self._entries()
        self.assertEqual(set(entries), set(self.EXPECTED))
        for key, (repo, family, licence) in self.EXPECTED.items():
            self.assertEqual(entries[key]["sources"][GPU][0], repo, key)
            self.assertEqual(entries[key]["gpu"]["family"], family, key)
            self.assertEqual(entries[key]["gpu"]["licence"], licence, key)
            self.assertFalse(entries[key]["translate"], key)

    def test_files_are_the_loaded_ones_only(self) -> None:
        for key, entry in self._entries().items():
            files = entry["gpu"]["files"]
            self.assertIn("model.safetensors", files, key)
            self.assertIn("config.json", files, key)
            self.assertFalse(any(f.endswith((".nemo", ".gguf", ".png")) for f in files), key)

    def test_fit_needs_and_the_large_row_is_marked(self) -> None:
        entries = self._entries()
        for key, entry in entries.items():
            self.assertGreater(entry["gpu"]["minVramGb"], 0, key)
            self.assertEqual(entry["gpu"]["minComputeCap"], 7.0, key)
        transcribe = entries["nemotron-3.5-transcribe-30b"]
        self.assertGreater(transcribe["gpu"]["minVramGb"], 24)  # listed for bigger cards, fit-gated here
        self.assertIn("Untested here", transcribe["gpu"]["note"])
        self.assertGreater(transcribe["sources"][GPU][1], 60)
        self.assertTrue(entries["qwen3-asr-1.7b"]["default"])
        self.assertFalse(entries["nemotron-3.5-transcribe-30b"]["default"])

    def test_mac_rows_keep_their_mlx_sources(self) -> None:
        entries = self._entries()
        self.assertIn("parakeet-mlx", entries["parakeet-tdt-v3"]["sources"])
        self.assertIn("nemotron-asr-mlx", entries["nemotron-3.5-asr"]["sources"])
        self.assertEqual(len([e for e in voice_runtime._STT_MODELS if e["key"] == "parakeet-tdt-v3"]), 1)

    def test_rows_carry_licence_note_and_need(self) -> None:
        rows = {r["key"]: r for r in _rows(ready=True)}
        self.assertEqual(rows["parakeet-tdt-v3"]["licence"], "CC-BY-4.0")
        self.assertIsNone(rows["qwen3-asr-1.7b"]["note"])
        self.assertIn("Untested here", rows["nemotron-3.5-transcribe-30b"]["note"])
        self.assertNotIn("licence", rows["whisper-base"])

    def test_download_patterns(self) -> None:
        patterns = voice_runtime.stt_repo_allow_patterns("nvidia/parakeet-tdt-0.6b-v3")
        self.assertIn("model.safetensors", patterns)
        self.assertFalse(any("nemo" in p or "gguf" in p for p in patterns))
        self.assertIsNone(voice_runtime.stt_repo_allow_patterns("mobiuslabsgmbh/faster-whisper-large-v3-turbo"))
        self.assertIsNone(voice_runtime.stt_repo_allow_patterns("someone/else"))
        self.assertEqual(voice_stt_cuda.BACKEND, GPU)

    def test_every_gpu_repo_is_pinned_to_a_full_commit(self) -> None:
        repos = {r["sources"][GPU][0]: r["gpu"] for r in voice_stt_cuda.GPU_STT_MODELS}
        repos.update({v["source"][0]: v["gpu"] for v in voice_stt_cuda.GPU_STT_SOURCES.values()})
        self.assertEqual(len(repos), 6)
        for repo, gpu in repos.items():
            self.assertRegex(gpu["revision"], r"^[0-9a-f]{40}$", repo)
            self.assertEqual(voice_runtime.stt_repo_revision(repo), gpu["revision"])
        self.assertEqual(len({g["revision"] for g in repos.values()}), 6)
        self.assertIsNone(voice_runtime.stt_repo_revision("mobiuslabsgmbh/faster-whisper-large-v3-turbo"))
        self.assertIsNone(voice_runtime.stt_repo_revision("someone/else"))

    def test_the_forced_aligner_has_its_own_files_and_pin_but_no_row(self) -> None:
        aligner = voice_stt_cuda.ALIGNER
        self.assertEqual(aligner["licence"], "Apache-2.0")
        self.assertRegex(aligner["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(voice_runtime.stt_repo_revision(aligner["repo"]), aligner["revision"])
        patterns = voice_runtime.stt_repo_allow_patterns(aligner["repo"])
        self.assertIn("model.safetensors", patterns)
        self.assertNotIn("generation_config.json", patterns)  # the repo has none
        self.assertFalse(any(r["key"].startswith("qwen3-forced") for r in voice_runtime.list_stt_models()))
        self.assertEqual(set(aligner["families"]),
                         {"qwen3-asr", "granite-speech", "nemotron-transcribe", "nemotron-asr"})
        self.assertEqual(aligner["optionalFor"], ("nemotron-asr",))  # it has timings of its own

    def test_the_pinned_snapshot_is_used_when_it_is_there(self) -> None:
        pinned = voice_stt_cuda.GPU_STT_MODELS[0]["gpu"]["revision"]
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "models--Qwen--Qwen3-ASR-1.7B-hf"
            (repo_dir / "snapshots" / "other").mkdir(parents=True)
            with mock.patch("backend_service.helpers.hf_cache_paths._hf_repo_cache_dir", return_value=repo_dir):
                self.assertEqual(voice_runtime._hf_snapshot_path("Qwen/Qwen3-ASR-1.7B-hf"), repo_dir / "snapshots" / "other")
                (repo_dir / "snapshots" / pinned).mkdir()
                self.assertEqual(voice_runtime._hf_snapshot_path("Qwen/Qwen3-ASR-1.7B-hf"), repo_dir / "snapshots" / pinned)


class GuardTests(unittest.TestCase):
    def _transcribe(self, model: str, *, snapshot: Path | None, problem: str | None = None, reply: dict | None = None,
                    **kwargs):
        worker = mock.Mock()
        worker.request.return_value = reply or {"text": "hello there", "segments": [], "language": "en", "seconds": 0.5}
        with mock.patch.object(voice_env, "lane", return_value="portable"), \
                mock.patch.object(voice_env, "is_installed", return_value=True), \
                mock.patch.object(voice_env_asr, "stt_gpu_ready", return_value=True), \
                mock.patch.object(voice_runtime, "_hf_snapshot_path", return_value=snapshot), \
                mock.patch("backend_service.voice_gpu.problem", return_value=problem), \
                mock.patch("backend_service.voice_gpu.facts", return_value=None), \
                mock.patch.object(voice_client, "stt_worker_for", return_value=worker) as chooser:
            result = voice_runtime.transcribe_file(Path("a.wav"), model, **kwargs)
        return result, worker, chooser

    def test_a_gpu_row_must_be_downloaded(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not downloaded"):
            self._transcribe("qwen3-asr-1.7b", snapshot=None)

    def test_a_gpu_row_must_fit_the_gpu(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "GPU memory"):
            self._transcribe("nemotron-3.5-transcribe-30b", snapshot=Path("snap"),
                             problem="Nemotron needs about 72 GB of GPU memory; this GPU has 24 GB.")

    def test_translation_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot translate"):
            self._transcribe("qwen3-asr-1.7b", snapshot=Path("snap"), task="translate")

    def test_request_goes_to_the_asr_worker_with_the_family(self) -> None:
        result, worker, chooser = self._transcribe("parakeet-tdt-v3", snapshot=Path("snap"), language="de")
        chooser.assert_called_with(GPU)
        payload = worker.request.call_args.kwargs
        self.assertEqual(payload["backend"], GPU)
        self.assertEqual(payload["family"], "parakeet-tdt")
        self.assertEqual(payload["source"], "snap")
        self.assertEqual(result["model"], "nvidia/parakeet-tdt-0.6b-v3")
        self.assertEqual(result["text"], "hello there")

    def test_speaker_labels_send_the_speaker_turns_as_spans(self) -> None:
        turns = {"turns": [{"start": 0.0, "end": 2.0, "speaker": 0}, {"start": 2.1, "end": 3.0, "speaker": 0},
                           {"start": 3.2, "end": 6.0, "speaker": 1}], "seconds": 0.4}
        with mock.patch.object(voice_env, "diarize_installed", return_value=True), \
                mock.patch.object(voice_runtime, "aligner_snapshot", return_value=None), \
                mock.patch.object(voice_runtime, "diarize_file", return_value=turns) as diarize:
            result, worker, _chooser = self._transcribe("qwen3-asr-1.7b", snapshot=Path("snap"), speakers=True, words=True)
        diarize.assert_called_once()  # once, before the worker; not again afterwards
        self.assertEqual(worker.request.call_args.kwargs["spans"], [[0.0, 3.0], [3.2, 6.0]])
        self.assertIn("Qwen/Qwen3-ForcedAligner-0.6B-hf", result["note"])

    def test_word_timings_ask_the_worker_with_the_downloaded_aligner(self) -> None:
        aligner = Path("aligner-snap")
        with mock.patch.object(voice_runtime, "aligner_snapshot", return_value=aligner):
            result, worker, _chooser = self._transcribe("qwen3-asr-1.7b", snapshot=Path("snap"), words=True)
            payload = worker.request.call_args.kwargs
            self.assertTrue(payload["words"])
            self.assertEqual(payload["alignerSource"], str(aligner))
            result, _worker, _chooser = self._transcribe(
                "qwen3-asr-1.7b", snapshot=Path("snap"), words=True,
                reply={"text": "hi", "segments": [], "wordTimings": "aligner", "seconds": 0.1})
        self.assertIsNone(result["note"])  # word timings arrived: nothing to explain

    def test_a_transducer_needs_no_aligner_and_nothing_is_asked_without_the_flag(self) -> None:
        with mock.patch.object(voice_runtime, "aligner_snapshot", side_effect=AssertionError("not needed")):
            _result, worker, _chooser = self._transcribe("parakeet-tdt-v3", snapshot=Path("snap"), words=True)
            self.assertIsNone(worker.request.call_args.kwargs["alignerSource"])
            _result, worker, _chooser = self._transcribe("qwen3-asr-1.7b", snapshot=Path("snap"))
            self.assertFalse(worker.request.call_args.kwargs["words"])

    def test_nemotron_asr_offers_the_aligner_but_does_not_need_it(self) -> None:
        reply = {"text": "hi", "segments": [], "wordTimings": "native", "seconds": 0.1}
        with mock.patch.object(voice_runtime, "aligner_snapshot", return_value=None):
            result, worker, _chooser = self._transcribe("nemotron-3.5-asr", snapshot=Path("snap"), words=True, reply=reply)
        self.assertIsNone(worker.request.call_args.kwargs["alignerSource"])
        self.assertIsNone(result["note"])  # its own timings, no nag about the aligner
        with mock.patch.object(voice_runtime, "aligner_snapshot", return_value=Path("al")):
            _result, worker, _chooser = self._transcribe("nemotron-3.5-asr", snapshot=Path("snap"), words=True, reply=reply)
        self.assertEqual(worker.request.call_args.kwargs["alignerSource"], "al")

    def test_why_there_are_no_word_timings(self) -> None:
        self.assertIn("forced aligner", voice_runtime._no_word_timings_note("aligner-missing"))
        self.assertIn("does not speak this language", voice_runtime._no_word_timings_note("language"))
        self.assertIn("could not be worked out", voice_runtime._no_word_timings_note("failed"))

    def test_whisper_requests_carry_no_family(self) -> None:
        _result, worker, chooser = self._transcribe("whisper-base", snapshot=None)
        chooser.assert_called_with("faster-whisper")
        self.assertIsNone(worker.request.call_args.kwargs["family"])


class WorkerChoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.main, self.asr = mock.Mock(pid=1, busy=False), mock.Mock(pid=2, busy=False)
        self.main.loaded, self.asr.loaded = {"stt": "whisper"}, {"stt": "qwen"}
        patches = [
            mock.patch.object(voice_client, "_WORKER", self.main),
            mock.patch.object(voice_client, "_ASR_WORKER", self.asr),
            mock.patch.object(voice_client, "get_worker", return_value=self.main),
            mock.patch.object(voice_client, "get_asr_worker", return_value=self.asr),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_gpu_backend_uses_the_asr_worker_and_frees_whisper(self) -> None:
        self.assertIs(voice_client.stt_worker_for(GPU), self.asr)
        self.main.request.assert_called_once_with("unload", timeout=60.0, what="stt")

    def test_whisper_frees_the_gpu_model(self) -> None:
        self.assertIs(voice_client.stt_worker_for("faster-whisper"), self.main)
        self.asr.request.assert_called_once_with("unload", timeout=60.0, what="stt")

    def test_a_busy_other_worker_is_not_interrupted(self) -> None:
        self.main.busy = True
        with self.assertRaises(voice_client.VoiceWorkerError):
            voice_client.stt_worker_for(GPU)

    def test_the_asr_worker_counts_as_a_speech_worker(self) -> None:
        self.assertIn(self.asr, voice_client._speech_workers())


class InstallTests(unittest.TestCase):
    def test_commands_install_torch_then_the_pinned_site_then_check(self) -> None:
        commands = voice_env_asr.stt_gpu_commands("py")
        self.assertIn(voice_env.SPEECH_GPU_TORCH_INDEX, commands[0])
        self.assertIn("torchaudio==2.9.1", commands[0])
        site = commands[1]
        self.assertIn("--no-deps", site)
        self.assertEqual(site[site.index("--target") + 1], str(voice_env_asr.asr_site_dir()))
        self.assertIn("transformers==5.23.0", site)
        self.assertTrue(all("==" in spec for spec in voice_env_asr.ASR_SITE_PACKAGES))
        self.assertEqual(commands[2][:2], ["py", "-I"])
        self.assertIn("asr ok", commands[2][3])

    def test_the_add_on_is_known_and_portable_only(self) -> None:
        lanes, packages, _label = voice_env.EXTRAS["stt-gpu"]
        self.assertEqual(lanes, ("portable",))
        self.assertEqual(packages, [])

    def test_worker_env_puts_the_site_first_and_stays_offline(self) -> None:
        env = voice_env_asr.asr_worker_env()
        self.assertEqual(env["PYTHONPATH"], str(voice_env_asr.asr_site_dir()))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(voice_env_asr.asr_site_dir().parent, voice_env.VOICE_VENV_DIR)

    def test_installed_needs_the_marker_and_the_site(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(voice_env, "VOICE_VENV_DIR", Path(tmp)), \
                mock.patch.object(voice_env, "is_installed", return_value=True), \
                mock.patch.object(voice_env, "installed_extras", return_value=["stt-gpu"]):
            self.assertFalse(voice_env_asr.stt_gpu_installed())
            site = Path(tmp) / "asr-site"
            (site / "transformers").mkdir(parents=True)
            (site / "transformers-5.23.0.dist-info").mkdir()
            self.assertTrue(voice_env_asr.stt_gpu_installed())

    def test_untick_keeps_torch_while_the_other_gpu_add_on_needs_it(self) -> None:
        with mock.patch.object(voice_env, "installed_extras", return_value=["stt-gpu", "speech-gpu"]):
            self.assertEqual(voice_runtime_disk.extra_packages_to_remove("stt-gpu"), [])
            speech = voice_runtime_disk.extra_packages_to_remove("speech-gpu")
            self.assertNotIn("torch", speech)
            self.assertIn("chatterbox-tts", speech)
        with mock.patch.object(voice_env, "installed_extras", return_value=["stt-gpu"]):
            self.assertEqual(voice_runtime_disk.extra_packages_to_remove("stt-gpu"), ["torch", "torchaudio"])
        with mock.patch.object(voice_env, "installed_extras", return_value=["speech-gpu"]):
            self.assertIn("torch", voice_runtime_disk.extra_packages_to_remove("speech-gpu"))

    def test_remove_asr_site_refuses_a_foreign_folder(self) -> None:
        with mock.patch.object(voice_env_asr, "asr_site_dir", return_value=Path("C:/elsewhere/asr-site")):
            with self.assertRaises(RuntimeError):
                voice_runtime_disk.remove_asr_site()


class WorkerHelperTests(unittest.TestCase):
    def test_short_audio_is_one_window(self) -> None:
        self.assertEqual(voice_worker_asr.windows(np.zeros(16_000, dtype=np.float32), 30.0), [(0, 16_000)])
        self.assertEqual(voice_worker_asr.windows(np.zeros(0, dtype=np.float32), 30.0), [])

    def test_long_audio_is_cut_at_the_quiet_point(self) -> None:
        rate = voice_worker_asr.SAMPLE_RATE
        audio = np.ones(70 * rate, dtype=np.float32)
        audio[int(26.0 * rate):int(26.3 * rate)] = 0.0  # a pause in the last quarter of the first window
        spans = voice_worker_asr.windows(audio, 30.0)
        self.assertEqual(spans[0][0], 0)
        self.assertTrue(26.0 * rate <= spans[0][1] <= 26.3 * rate, spans)
        self.assertEqual(spans[-1][1], len(audio))
        for (_a, end), (start, _b) in zip(spans, spans[1:]):
            self.assertEqual(end, start)
        self.assertTrue(all(b - a <= 30 * rate for a, b in spans))

    def test_spans_become_windows_of_their_own(self) -> None:
        rate = voice_worker_asr.SAMPLE_RATE
        audio = np.ones(10 * rate, dtype=np.float32)
        spans = voice_worker_asr._request_windows(audio, 30.0, [[0.0, 3.0], [3.2, 6.0], [9.99, 10.0]])
        self.assertEqual(spans, [(0, 3 * rate), (int(3.2 * rate), 6 * rate)])  # the 10 ms span is dropped
        self.assertEqual(voice_worker_asr._request_windows(audio, 30.0, None), [(0, 10 * rate)])

    def test_languages(self) -> None:
        self.assertIsNone(voice_worker_asr._qwen_language(None))
        self.assertEqual(voice_worker_asr._qwen_language("de-DE"), "de")
        with self.assertRaisesRegex(ValueError, "Unsupported language"):
            voice_worker_asr._qwen_language("xx")
        processor = mock.Mock(prompt_dictionary={"en-US": 0, "de-DE": 1})
        self.assertEqual(voice_worker_asr._nemotron_language(processor, None), "auto")
        self.assertEqual(voice_worker_asr._nemotron_language(processor, "de"), "de-DE")
        with self.assertRaisesRegex(ValueError, "Unsupported language"):
            voice_worker_asr._nemotron_language(processor, "fi")

    def test_transcribe_joins_windows_and_refuses_translation(self) -> None:
        rate = voice_worker_asr.SAMPLE_RATE
        audio = np.ones(50 * rate, dtype=np.float32)
        with mock.patch.object(voice_worker_asr, "load", return_value=("m", "p", "cpu")), \
                mock.patch.object(voice_worker_asr, "_run_window", side_effect=[("one", "en"), ("two", "en")]) as run:
            reply = voice_worker_asr.transcribe(
                {"family": "qwen3-asr", "source": "s", "path": "a.wav", "language": ""}, lambda _p, _r: audio
            )
        self.assertEqual(run.call_count, 2)
        self.assertEqual(reply["text"], "one two")
        self.assertEqual(reply["language"], "en")
        self.assertEqual(len(reply["segments"]), 2)
        self.assertEqual(reply["durationS"], 50.0)
        with self.assertRaisesRegex(ValueError, "Whisper"):
            voice_worker_asr.transcribe({"family": "qwen3-asr", "task": "translate"}, lambda _p, _r: audio)

    def test_parakeet_ignores_the_language_setting(self) -> None:
        audio = np.ones(16_000, dtype=np.float32)
        with mock.patch.object(voice_worker_asr, "load", return_value=("m", "p", "cpu")), \
                mock.patch.object(voice_worker_asr, "_run_window", return_value=("hallo", None)) as run:
            voice_worker_asr.transcribe(
                {"family": "parakeet-tdt", "source": "s", "path": "a.wav", "language": "de"}, lambda _p, _r: audio
            )
        self.assertIsNone(run.call_args.args[5])

    def test_token_words_join_subwords_and_punctuation_keeps_the_end(self) -> None:
        tokens = [
            {"token": "The", "start": 0.0, "end": 0.24}, {"token": " region", "start": 0.24, "end": 0.48},
            {"token": "al", "start": 0.48, "end": 0.72}, {"token": ",", "start": 0.72, "end": 0.8},
            {"token": " ", "start": 0.96, "end": 1.04}, {"token": "ra", "start": 0.96, "end": 1.04},
            {"token": "il", "start": 1.12, "end": 1.2}, {"token": " ", "start": 1.3, "end": 1.4},
        ]
        words = voice_worker_asr._token_words(tokens, 10.0)
        self.assertEqual([w["text"] for w in words], ["The", "regional,", "rail"])
        self.assertEqual((words[1]["start"], words[1]["end"]), (10.24, 10.72))  # the comma did not stretch the word
        self.assertEqual((words[2]["start"], words[2]["end"]), (10.96, 11.2))  # a bare space token starts a word

    def _words_transcribe(self, request: dict, audio: np.ndarray):
        with mock.patch.object(voice_worker_asr, "load", return_value=("m", "p", "cpu")):
            return voice_worker_asr.transcribe(request, lambda _p, _r: audio)

    def test_transducer_words_are_native_and_offset_by_their_window(self) -> None:
        rate = voice_worker_asr.SAMPLE_RATE
        audio = np.ones(10 * rate, dtype=np.float32)
        words = [{"start": 0.5, "end": 0.9, "text": "hi"}]
        with mock.patch.object(voice_worker_asr, "_transducer_window_words", return_value=("hi", words)) as run, \
                mock.patch.object(voice_worker_asr, "_run_window", side_effect=AssertionError("plain path")):
            reply = self._words_transcribe({"family": "parakeet-tdt", "source": "s", "path": "a.wav", "words": True,
                                            "language": "de", "spans": [[2.0, 4.0]]}, audio)
        self.assertEqual(reply["wordTimings"], "native")
        self.assertEqual(reply["segments"][0]["words"], words)
        self.assertEqual(run.call_args.args[6], 2.0)  # the window's offset into the file
        # Without the flag the plain path runs and the reply says nothing about words.
        with mock.patch.object(voice_worker_asr, "_run_window", return_value=("hi", None)):
            plain = self._words_transcribe({"family": "parakeet-tdt", "source": "s", "path": "a.wav"}, audio)
        self.assertNotIn("wordTimings", plain)
        self.assertNotIn("words", plain["segments"][0])

    def test_audio_llm_words_come_from_the_aligner_with_the_window_offset(self) -> None:
        rate = voice_worker_asr.SAMPLE_RATE
        audio = np.ones(50 * rate, dtype=np.float32)  # two windows
        aligned = [{"start": 0.1, "end": 0.4, "text": "x"}]
        with mock.patch.object(voice_worker_asr, "_run_window", side_effect=[("one", "en"), ("two", "en")]), \
                mock.patch.object(voice_worker_asr, "_align_window", return_value=aligned) as align:
            reply = self._words_transcribe({"family": "qwen3-asr", "source": "s", "path": "a.wav", "words": True,
                                            "alignerSource": "al"}, audio)
        self.assertEqual(reply["wordTimings"], "aligner")
        self.assertEqual(align.call_count, 2)
        second = reply["segments"][1]
        self.assertEqual(second["words"][0]["start"], round(second["start"] + 0.1, 2))
        self.assertEqual(align.call_args.args[4], "en")

    def test_nemotron_asr_prefers_the_aligner_and_keeps_its_own_timings_otherwise(self) -> None:
        audio = np.ones(16_000, dtype=np.float32)
        base = {"family": "nemotron-asr", "source": "s", "path": "a.wav", "words": True}
        own = [{"start": 0.3, "end": 0.5, "text": "hi"}]
        aligned = [{"start": 0.0, "end": 0.2, "text": "hi"}]
        with mock.patch.object(voice_worker_asr, "_transducer_window_words", return_value=("hi", own)), \
                mock.patch.object(voice_worker_asr, "_run_window", return_value=("hi", None)), \
                mock.patch.object(voice_worker_asr, "_align_window", return_value=aligned):
            without = self._words_transcribe(base, audio)
            with_aligner = self._words_transcribe({**base, "alignerSource": "al"}, audio)
            german = self._words_transcribe({**base, "alignerSource": "al", "language": "nl"}, audio)
        with mock.patch.object(voice_worker_asr, "_transducer_window_words", return_value=("hi", own)), \
                mock.patch.object(voice_worker_asr, "_run_window", return_value=("hi", None)), \
                mock.patch.object(voice_worker_asr, "_align_window", side_effect=RuntimeError("boom")):
            failed = self._words_transcribe({**base, "alignerSource": "al"}, audio)
        self.assertEqual((without["wordTimings"], without["segments"][0]["words"]), ("native", own))
        self.assertEqual((with_aligner["wordTimings"], with_aligner["segments"][0]["words"]), ("aligner", aligned))
        self.assertEqual((german["wordTimings"], german["segments"][0]["words"]), ("native", own))  # not an aligner language
        self.assertEqual((failed["wordTimings"], failed["segments"][0]["words"]), ("native", own))

    def test_audio_llm_words_say_why_they_are_missing_and_keep_the_text(self) -> None:
        audio = np.ones(16_000, dtype=np.float32)
        base = {"family": "qwen3-asr", "source": "s", "path": "a.wav", "words": True}
        with mock.patch.object(voice_worker_asr, "_run_window", return_value=("hello", "en")):
            missing = self._words_transcribe(base, audio)  # no aligner downloaded
            with mock.patch.object(voice_worker_asr, "_align_window", side_effect=RuntimeError("boom")):
                failed = self._words_transcribe({**base, "alignerSource": "al"}, audio)
        with mock.patch.object(voice_worker_asr, "_run_window", return_value=("salaam", "nl")):
            language = self._words_transcribe({**base, "alignerSource": "al"}, audio)
        self.assertEqual(missing["wordsUnavailable"], "aligner-missing")
        self.assertEqual(failed["wordsUnavailable"], "failed")
        self.assertEqual(language["wordsUnavailable"], "language")
        for reply in (missing, failed, language):
            self.assertIsNone(reply["wordTimings"])
            self.assertTrue(reply["text"])
            self.assertNotIn("words", reply["segments"][0])

    def test_unknown_family_is_refused(self) -> None:
        fake = mock.Mock(__version__="5.23.0")
        with mock.patch.dict(sys.modules, {"transformers": fake}):
            with self.assertRaisesRegex(ValueError, "Unknown speech-to-text model family"):
                voice_worker_asr._auto_class("nope")

    def test_the_worker_hands_gpu_requests_to_the_asr_module(self) -> None:
        fake = mock.Mock()
        fake.transcribe.return_value = {"text": "x"}
        with mock.patch.object(voice_worker, "_asr_module", return_value=fake):
            self.assertEqual(voice_worker.transcribe({"backend": GPU, "family": "qwen3-asr"}), {"text": "x"})
        fake.transcribe.assert_called_once()


class DownloadPinTests(unittest.TestCase):
    """The download of a GPU dictation repo (and of the forced aligner) is handed its pinned commit (FU-271)."""

    def setUp(self) -> None:
        from backend_service.state import ChaosEngineState

        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.state = ChaosEngineState(settings_path=tmp / "settings.json", benchmarks_path=tmp / "benchmarks.json",
                                      chat_sessions_path=tmp / "chats.json")
        self.addCleanup(self._tmp.cleanup)

    def _start(self, repo: str):
        process = mock.Mock()
        process.poll.return_value = 0
        process.returncode = 0
        process.wait.return_value = 0

        created: list[object] = []

        class ImmediateThread:  # the first thread (the download) runs at once; the progress monitor never starts
            def __init__(self, *, target=None, daemon=None):
                self.target = target
                self.run_target = not created
                created.append(self)

            def start(self):
                if self.run_target and self.target is not None:
                    self.target()

            def join(self, timeout=None):
                return None

        with mock.patch("backend_service.state.downloads._spawn_snapshot_download", return_value=process) as spawn, \
                mock.patch("backend_service.helpers.huggingface._hf_repo_downloaded_bytes", return_value=0), \
                mock.patch("backend_service.helpers.huggingface._hf_repo_preflight_size_gb", return_value=1.0), \
                mock.patch("backend_service.helpers.huggingface._hub_repo_files", return_value={"files": []}), \
                mock.patch("backend_service.state.downloads.threading.Thread", side_effect=ImmediateThread):
            self.state.start_download(repo, allow_patterns=voice_runtime.stt_repo_allow_patterns(repo))
        return spawn

    def test_a_gpu_row_downloads_its_pinned_commit(self) -> None:
        spawn = self._start("Qwen/Qwen3-ASR-0.6B-hf")
        self.assertEqual(spawn.call_args.kwargs["revision"], "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c")
        self.assertIn("model.safetensors", spawn.call_args.kwargs["allow_patterns"])

    def test_the_aligner_downloads_its_pinned_commit(self) -> None:
        spawn = self._start(voice_stt_cuda.ALIGNER["repo"])
        self.assertEqual(spawn.call_args.kwargs["revision"], voice_stt_cuda.ALIGNER["revision"])

    def test_other_repos_stay_unpinned(self) -> None:
        self.assertIsNone(self._start("someone/else").call_args.kwargs["revision"])


class RouteTests(unittest.TestCase):
    def test_download_route_passes_the_row_files(self) -> None:
        from types import SimpleNamespace

        from backend_service.routes import models as model_routes

        state = mock.Mock()
        state.start_download.return_value = {"repo": "x"}
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(chaosengine=state)))
        body = model_routes.DownloadModelRequest(repo="Qwen/Qwen3-ASR-1.7B-hf")
        model_routes.download_model(request, body)
        patterns = state.start_download.call_args.kwargs["allow_patterns"]
        self.assertIn("model.safetensors", patterns)
        state.start_download.reset_mock()
        model_routes.download_model(request, model_routes.DownloadModelRequest(repo="Qwen/Qwen3-0.6B"))
        self.assertIsNone(state.start_download.call_args.kwargs["allow_patterns"])


if __name__ == "__main__":
    unittest.main()
