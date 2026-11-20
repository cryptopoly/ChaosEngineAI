"""Sounds (Stable Audio Open Small): catalog row, licence, gated download message, sounds pack env, worker limits,
the pure audio helpers, routes, the library kind and the ``make_sound`` agent tool.

No real model runs: the worker is the real ``music_worker.py`` with the fake engine from ``music_fakes`` plus
the two attributes a sound engine file declares (``SECONDS_RANGE``, ``TASKS``). The real engine's wiring was
checked by hand against stable-audio-tools with a random mini model (see ``docs/hub/progress/v-audio.md``).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend_service import music_client, music_env, music_runtime, music_sound_env
from backend_service.catalog.music_models import music_variants
from backend_service.helpers import music as music_helpers
from backend_service.helpers.audio_metadata import read_audio_settings
from backend_service.progress import MUSIC_PROGRESS
from tests.music_fakes import FAKE_ENGINE, MusicSandbox

SOUND = "stabilityai/stable-audio-open-small"
SOUND_ENGINE = FAKE_ENGINE + "\nSECONDS_RANGE = (1, 11)\nTASKS = ('text2sound',)\n"


def _row() -> dict:
    return next(v for v in music_variants() if v["id"] == SOUND)


def _load_engine_module():
    path = Path(__file__).resolve().parents[1] / "backend_service" / "music_worker_stableaudio.py"
    spec = importlib.util.spec_from_file_location("sound_engine_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CatalogTests(unittest.TestCase):
    def test_the_sound_row(self) -> None:
        row = _row()
        self.assertEqual((row["kind"], row["pack"], row["engine"], row["makes"]), ("sound", "sound", "stableaudio", ["sfx", "loops"]))
        self.assertEqual((row["minSeconds"], row["maxSeconds"]), (1, 11))
        self.assertEqual(len(row["revision"]), 40)
        self.assertEqual(len(row["textEncoder"]["revision"]), 40)
        licence = row["licence"]
        self.assertEqual((licence["id"], licence["gated"], licence["commercial"]), ("stability-community", True, "limited"))
        self.assertEqual(licence["attribution"], "Powered by Stability AI")
        self.assertEqual(licence["revenueCapUsd"], 1_000_000)
        text = " ".join(licence["summaryLines"])
        for needle in ("US$1 million", "Powered by Stability AI", "foundation model", "register"):
            self.assertIn(needle, text)
        self.assertIn("/blob/" + row["revision"] + "/LICENSE", licence["url"])
        self.assertEqual(row["estimates"]["basis"], "measured")  # first real run on an M4 Max, 2026-11-07

    def test_only_the_needed_files_are_fetched(self) -> None:
        patterns = music_helpers.music_repo_allow_patterns(SOUND)
        self.assertEqual(patterns, ["model.safetensors", "model_config.json", "LICENSE", "README.md"])
        self.assertIsNone(music_helpers.music_repo_allow_patterns("ACE-Step/Ace-Step1.5"))
        self.assertEqual(music_helpers.music_repo_revision(SOUND), _row()["revision"])
        companions = _row()["downloadCompanions"]
        self.assertEqual(companions[0]["repo"], "google-t5/t5-base")
        self.assertNotIn("pytorch_model.bin", companions[0]["patterns"])

    def test_music_defaults_are_unchanged(self) -> None:
        self.assertEqual(music_helpers.default_music_variant()["id"], "ACE-Step/Ace-Step1.5")
        self.assertEqual(music_runtime.is_sound(music_helpers.default_music_variant()), False)

    def test_annotation_uses_the_sounds_pack(self) -> None:
        with mock.patch.object(music_sound_env, "is_installed", lambda: True), \
                mock.patch.object(music_env, "is_installed", lambda: False), \
                mock.patch.object(music_sound_env, "text_encoder_ready", lambda v: True), \
                mock.patch.object(music_helpers, "weights_source", lambda v: "hf-cache"):
            rows = {v["id"]: v for f in music_helpers.music_model_payloads() for v in f["variants"]}
        self.assertTrue(rows[SOUND]["packInstalled"] and rows[SOUND]["ready"])
        self.assertFalse(rows["ACE-Step/Ace-Step1.5"]["packInstalled"])

    def test_the_sound_row_needs_its_text_encoder(self) -> None:
        with mock.patch.object(music_sound_env, "is_installed", lambda: True), \
                mock.patch.object(music_sound_env, "text_encoder_ready", lambda v: False), \
                mock.patch.object(music_helpers, "weights_source", lambda v: "hf-cache"):
            row = next(v for f in music_helpers.music_model_payloads() for v in f["variants"] if v["id"] == SOUND)
        self.assertFalse(row["availableLocally"])


class GatedMessageTests(unittest.TestCase):
    def test_a_refusal_becomes_the_licence_message(self) -> None:
        for error in ("401 Client Error: Unauthorized", "Cannot access gated repo for url ...", "HTTP 403",
                      "Access to model stabilityai/stable-audio-open-small is restricted."):
            message = music_helpers.gated_music_message(SOUND, error)
            self.assertIn("accept the licence", message)
            self.assertIn("huggingface.co/stabilityai/stable-audio-open-small", message)
            self.assertIn("token", message)

    def test_other_failures_and_other_repos_keep_their_own_text(self) -> None:
        self.assertIsNone(music_helpers.gated_music_message(SOUND, "Connection reset by peer"))
        self.assertIsNone(music_helpers.gated_music_message("ACE-Step/Ace-Step1.5", "401 Client Error"))
        self.assertIsNone(music_helpers.gated_music_message("some/other", "401 Client Error"))

    def test_the_shared_download_error_path_uses_it(self) -> None:
        from backend_service.helpers.hf_errors import _friendly_hf_download_error

        self.assertIn("accept the licence", _friendly_hf_download_error(SOUND, "401 Client Error: Unauthorized for url"))
        self.assertIn("Hugging Face refused access", _friendly_hf_download_error("some/other", "HTTP 401"))


class _SoundEnvBox(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data = root / "music"
        self.venv = root / "sound-venv"
        for patch in (
            mock.patch.object(music_env, "MUSIC_DATA_DIR", self.data),
            mock.patch.object(music_sound_env, "SOUND_VENV_DIR", self.venv),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)


class SoundEnvTests(_SoundEnvBox):
    def test_lane_follows_the_music_lane(self) -> None:
        with mock.patch.object(music_env, "lane", lambda: "mlx"):
            self.assertEqual((music_sound_env.lane(), music_sound_env.lane_unavailable_reason()), ("mps", None))
        with mock.patch.object(music_env, "lane", lambda: "unsupported"):
            self.assertEqual(music_sound_env.lane(), "unsupported")
            self.assertEqual(music_sound_env.lane_unavailable_code(), music_env.UNSUPPORTED_PLATFORM_CODE)
        # FU-274: where music runs on an NVIDIA GPU, sounds run there too.
        with mock.patch.object(music_env, "lane", lambda: "cuda"):
            self.assertEqual(music_sound_env.lane(), "cuda")
            self.assertIsNone(music_sound_env.lane_unavailable_reason())
            self.assertIsNone(music_sound_env.lane_unavailable_code())

    def test_marker_needs_the_release_the_venv_and_its_python(self) -> None:
        self.assertFalse(music_sound_env.is_installed())
        python = music_sound_env.venv_python()
        python.parent.mkdir(parents=True)
        python.write_text("")
        music_sound_env._write_marker({"torch": "2.7.1"})
        self.assertTrue(music_sound_env.is_installed())
        marker = music_sound_env.read_marker()
        for key, stale in (("version", "0.0.1"), ("sa3", "0" * 40), ("lane", "elsewhere"), ("schema", 1)):
            music_sound_env._marker_path().write_text(json.dumps({**marker, key: stale}))
            self.assertFalse(music_sound_env.is_installed(), key)  # a pack without Stable Audio 3, or for another lane
        music_sound_env._marker_path().write_text(json.dumps(marker))
        self.assertTrue(music_sound_env.is_installed())

    def test_worker_environment_selects_the_sound_engine_and_goes_offline(self) -> None:
        env = music_sound_env.worker_env()
        self.assertEqual(env["CHAOSENGINE_MUSIC_ENGINE_MODULE"], str(music_sound_env.ENGINE_FILE))
        self.assertTrue(music_sound_env.ENGINE_FILE.is_file())
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["CHAOSENGINE_MUSIC_DATA"], str(self.data))

    def test_the_pinned_requirements_cover_the_engine_imports(self) -> None:
        lines = music_sound_env.REQUIREMENTS.read_text(encoding="utf-8").lower().splitlines()
        text = "\n".join(line for line in lines if not line.startswith("#"))
        for name in ("torch==", "torchaudio==", "transformers==", "pytorch-lightning==", "einops==", "alias-free-torch==",
                     "soundfile==", "safetensors==", "sentencepiece==", "vector-quantize-pytorch==", "v-diffusion-pytorch=="):
            self.assertIn(name, text)
        self.assertNotIn("stable-audio-tools", text)  # installed afterwards, --no-deps

    def test_text_encoder_is_mirrored_from_the_cache(self) -> None:
        row = _row()
        cache = Path(self.tmp.name) / "hub"
        # The real Hub cache may already hold the encoder (it does once the model is downloaded).
        with mock.patch("backend_service.helpers.hf_cache_paths._hf_repo_cache_dir", lambda repo: cache / "models--google-t5--t5-base"):
            self.assertFalse(music_sound_env.text_encoder_ready(row))
        snapshot = cache / "models--google-t5--t5-base" / "snapshots" / row["textEncoder"]["revision"]
        snapshot.mkdir(parents=True)
        for name in row["textEncoder"]["files"]:
            (snapshot / name).write_bytes(b"x" * 10)
        with mock.patch("backend_service.helpers.hf_cache_paths._hf_repo_cache_dir", lambda repo: cache / "models--google-t5--t5-base"):
            self.assertTrue(music_sound_env.text_encoder_ready(row))
            self.assertTrue(music_sound_env.prepare_text_encoder(row))
        for name in row["textEncoder"]["files"]:
            self.assertTrue((music_sound_env.text_encoder_dir(row) / name).is_file())
        self.assertTrue(music_sound_env.text_encoder_ready(row))  # now from the layout, cache not needed

    def test_install_job_order_and_the_no_deps_engine_step(self) -> None:
        commands: list[list[str]] = []
        python = music_sound_env.venv_python()

        def fake_run(command: list[str]) -> str:
            commands.append(command)
            if "venv" in command:
                python.parent.mkdir(parents=True, exist_ok=True)
                python.write_text("")
            return ""

        with mock.patch.object(music_env, "lane", lambda: "mlx"), \
                mock.patch.object(music_env, "_python_supported", lambda *a: True), \
                mock.patch.object(music_sound_env, "_run", fake_run), \
                mock.patch.object(music_sound_env, "_verify", lambda: {"torch": "2.7.1"}), \
                mock.patch("backend_service.voice_env._base_python", lambda: "python3"):
            music_sound_env._JOB.__init__()
            music_sound_env._run_install()
            self.assertTrue(music_sound_env.is_installed())  # the marker records the lane it was built for
        status = music_sound_env.job_status()
        self.assertEqual((status["phase"], status["error"]), ("done", None))
        flat = [" ".join(c) for c in commands]
        self.assertTrue(any("-m venv" in c for c in flat))
        self.assertTrue(any("-r" in c and "sound_requirements.txt" in c for c in flat))
        engine = [c for c in flat if "stable-audio-tools==" in c][0]
        self.assertIn("--no-deps", engine)
        self.assertIn("--ignore-requires-python", engine)
        # FU-493: Stable Audio 3 rows are CUDA-only, so the Mac lane never installs it.
        self.assertFalse(any("stable-audio-3" in c for c in flat))
        self.assertIsNone(music_sound_env.read_marker()["sa3"])
        self.assertFalse(any("download.pytorch.org" in c for c in flat))  # the Mac lane takes torch from PyPI

    def test_the_cuda_lane_installs_cuda_torch_and_checks_the_gpu(self) -> None:
        """FU-274: on an NVIDIA GPU the requirements add the PyTorch CUDA index and the check needs a CUDA torch."""
        commands: list[list[str]] = []
        python = music_sound_env.venv_python()

        def fake_run(command: list[str]) -> str:
            commands.append(command)
            if "venv" in command:
                python.parent.mkdir(parents=True, exist_ok=True)
                python.write_text("")
            return '{"torch": "2.7.1+cu128"}'

        with mock.patch.object(music_env, "lane", lambda: "cuda"), \
                mock.patch.object(music_env, "_python_supported", lambda *a: True), \
                mock.patch.object(music_sound_env, "_run", fake_run), \
                mock.patch("backend_service.voice_env._base_python", lambda: "python3"):
            music_sound_env._JOB.__init__()
            music_sound_env._run_install()
            self.assertEqual(music_sound_env.job_status()["error"], None)
            self.assertTrue(music_sound_env.is_installed())
            self.assertEqual(music_sound_env.read_marker()["lane"], "cuda")
        requirements = [c for c in commands if "-r" in c][0]
        self.assertEqual(requirements[-2:], ["--extra-index-url", music_env.CUDA_TORCH_INDEX])
        sa3 = [" ".join(c) for c in commands if any("stable-audio-3 @ " in part for part in c)][0]
        self.assertIn("--no-deps", sa3)
        self.assertIn(f"/archive/{music_sound_env.SA3_REF}.zip", sa3)
        self.assertEqual(len(music_sound_env.SA3_REF), 40)
        self.assertEqual(music_sound_env.read_marker()["sa3"], music_sound_env.SA3_REF)
        check = [c for c in commands if "-c" in c][-1][-1]
        self.assertIn("torch.cuda.is_available()", check)
        self.assertIn("stable_audio_3", check)
        self.assertIn("T5GemmaEncoderModel", check)
        with mock.patch.object(music_env, "lane", lambda: "mlx"):
            self.assertFalse(music_sound_env.is_installed())  # a CUDA pack is not a Mac pack

    def test_simulated_hosts_only_cuda_installs_or_imports_stable_audio_3(self) -> None:
        """FU-493: simulate each host through the real lane check (platform and nvidia-smi patched): an Apple
        Silicon Mac neither installs nor imports ``stable_audio_3`` (so it cannot break Stable Audio Open there);
        a Windows or Linux PC with an NVIDIA GPU still does both."""
        hosts = {
            "mac-arm64": ("Darwin", "arm64", None, "mps"),
            "windows-nvidia": ("Windows", "AMD64", "C:/nvidia-smi.exe", "cuda"),
            "linux-nvidia": ("Linux", "x86_64", "/usr/bin/nvidia-smi", "cuda"),
        }
        for host, (system, machine, smi, expected_lane) in hosts.items():
            with self.subTest(host=host):
                commands: list[list[str]] = []
                python = music_sound_env.venv_python()
                shutil.rmtree(self.venv, ignore_errors=True)  # each host builds its pack from nothing

                def fake_run(command: list[str]) -> str:
                    commands.append(command)
                    if "venv" in command:
                        python.parent.mkdir(parents=True, exist_ok=True)
                        python.write_text("")
                    return '{"torch": "2.7.1"}'

                with mock.patch("platform.system", lambda: system), \
                        mock.patch("platform.machine", lambda: machine), \
                        mock.patch("shutil.which", lambda name: smi if name == "nvidia-smi" else None), \
                        mock.patch.object(music_env, "_python_supported", lambda *a: True), \
                        mock.patch.object(music_sound_env, "_run", fake_run), \
                        mock.patch("shutil.disk_usage", lambda p: mock.Mock(free=100e9)), \
                        mock.patch("backend_service.voice_env._base_python", lambda: "python3"):
                    self.assertEqual(music_sound_env.lane(), expected_lane)
                    music_sound_env._JOB.__init__()
                    music_sound_env._run_install()
                    self.assertEqual(music_sound_env.job_status()["error"], None)
                    self.assertTrue(music_sound_env.is_installed())
                checks = [c[-1] for c in commands if "-c" in c]
                installs = [" ".join(c) for c in commands]
                on_cuda = expected_lane == "cuda"
                self.assertEqual(any("stable-audio-3 @ " in c for c in installs), on_cuda)
                self.assertEqual(all("stable_audio_3" in c for c in checks), on_cuda)
                self.assertTrue(checks and all("stable_audio_tools" in c for c in checks))  # Open is checked everywhere
                self.assertEqual(any("T5GemmaEncoderModel" in c for c in checks), on_cuda)

    def test_the_mac_worker_does_not_ask_for_stable_audio_3(self) -> None:
        engine = _load_engine_module().StableAudioEngine(Path(self.tmp.name))
        seen: list[str] = []

        def find_spec(name: str) -> object:
            seen.append(name)
            return object()

        for platform_name, wants_sa3 in (("darwin", False), ("win32", True), ("linux", True)):
            seen.clear()
            with mock.patch("sys.platform", platform_name), mock.patch("importlib.util.find_spec", find_spec):
                self.assertIsNone(engine.unavailable_reason())
            self.assertEqual("stable_audio_3" in seen, wants_sa3, platform_name)
            self.assertIn("stable_audio_tools", seen)

    def test_install_refuses_off_apple_silicon(self) -> None:
        with mock.patch.object(music_env, "lane", lambda: "unsupported"):
            music_sound_env._JOB.__init__()
            music_sound_env._run_install()
        status = music_sound_env.job_status()
        self.assertEqual(status["phase"], "error")
        self.assertIn("Apple Silicon", status["error"])


class AudioHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = _load_engine_module()

    def test_a_loop_runs_straight_from_its_last_sample_into_its_first(self) -> None:
        sr = 8000
        t = np.arange(sr, dtype=np.float32) / sr  # one second
        clip = np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t)], axis=1).astype(np.float32)
        looped = self.engine.make_loop(clip, sr)
        fade = int(self.engine.LOOP_FADE_SECONDS * sr)
        self.assertEqual(looped.shape[0], clip.shape[0] - fade)
        # The sample after the last one is the first one: it must be the clip's own next sample after the cut.
        step = np.abs(clip[1:] - clip[:-1]).max()
        join = np.abs(looped[0] - looped[-1]).max()
        self.assertLessEqual(join, step * 1.5)
        # Without the cross-fade a plain cut at the same place jumps more.
        plain = np.abs(clip[0] - clip[clip.shape[0] - fade - 1]).max()
        self.assertGreater(plain, join)

    def test_a_short_clip_is_never_emptied(self) -> None:
        tiny = np.ones((6, 2), dtype=np.float32)
        self.assertEqual(self.engine.make_loop(tiny, 8000).shape, tiny.shape)
        self.assertEqual(self.engine.edge_fades(tiny, 8000).shape, tiny.shape)

    def test_edge_fades_start_and_end_at_zero(self) -> None:
        faded = self.engine.edge_fades(np.ones((4000, 2), dtype=np.float32), 8000)
        self.assertEqual((float(faded[0, 0]), float(faded[-1, 0])), (0.0, 0.0))
        self.assertEqual(float(faded[2000, 0]), 1.0)

    def test_peak_normalisation(self) -> None:
        quiet = np.full((100, 2), 0.1, dtype=np.float32)
        out = self.engine.normalise_peak(quiet, -1.0)
        self.assertAlmostEqual(float(np.abs(out).max()), 10 ** (-1 / 20), places=4)
        silence = np.zeros((10, 2), dtype=np.float32)
        self.assertEqual(float(np.abs(self.engine.normalise_peak(silence, -1.0)).max()), 0.0)
        self.assertIs(self.engine.normalise_peak(quiet, None), quiet)

    def test_the_engine_declares_what_the_worker_checks(self) -> None:
        self.assertEqual((self.engine.SECONDS_RANGE, self.engine.TASKS), ((1, 380), ("text2sound",)))
        self.assertEqual({k: v["max_seconds"] for k, v in self.engine.MODEL_PROFILES.items()},
                         {"stable-audio-open-small": 11, "stable-audio-open-1.0": 47, "stable-audio-3-small-sfx": 120,
                          "stable-audio-3-small-music": 120, "stable-audio-3-medium": 380})
        # Every catalog sound row has an engine profile with the same longest clip.
        for row in music_variants():
            if row.get("kind") == "sound":
                self.assertEqual(self.engine.MODEL_PROFILES[row["engineModel"]]["max_seconds"], row["maxSeconds"], row["id"])

    def test_a_full_gpu_gets_a_plain_sentence(self) -> None:
        """FU-274: a Stable Audio 3 run checks free VRAM first (counting what the worker already holds)."""
        need = self.engine.MODEL_PROFILES["stable-audio-3-medium"]["vram_gb"]
        message = self.engine.vram_shortfall_message("stable-audio-3-medium", 1.0, 0.5)
        self.assertIn("1.0 GB free", message)
        self.assertIn(f"{need:.0f} GB", message)
        self.assertIn("Another program may be using the GPU", message)
        self.assertIsNone(self.engine.vram_shortfall_message("stable-audio-3-medium", need, 0.0))
        self.assertIsNone(self.engine.vram_shortfall_message("stable-audio-3-medium", 1.0, need))  # already loaded here
        self.assertIsNone(self.engine.vram_shortfall_message("stable-audio-open-small", 0.1, 0.0))  # no figure, no check

    def test_stable_audio_3_reads_its_own_encoder_folder_offline(self) -> None:
        """The T5Gemma conditioner's Hub repo and subfolder are replaced by the local folder inside the model's layout."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "checkpoints" / "stable-audio-3-small-sfx"
            (folder / "t5gemma-b-b-ul2").mkdir(parents=True)
            (folder / "model.safetensors").write_bytes(b"x")
            (folder / "t5gemma-b-b-ul2" / "model.safetensors").write_bytes(b"x")
            (folder / "model_config.json").write_text(json.dumps({
                "sample_rate": 44100, "sample_size": 5292032,
                "model": {"conditioning": {"configs": [
                    {"id": "prompt", "type": "t5gemma", "config": {"max_length": 256, "repo_id": "stabilityai/stable-audio-3-small-sfx",
                                                                  "subfolder": "t5gemma-b-b-ul2"}},
                    {"id": "seconds_total", "type": "number", "config": {"min_val": 0, "max_val": 384}},
                ]}},
            }))
            engine = self.engine.create_engine(Path(tmp))
            seen: dict = {}
            with mock.patch.object(engine, "_pick_device", lambda: "cuda"), \
                    mock.patch.object(engine, "_build_sa3", lambda config, weights: seen.update(config=config, weights=weights) or "model"):
                info = engine.load("stable-audio-3-small-sfx")
        prompt = seen["config"]["model"]["conditioning"]["configs"][0]["config"]
        self.assertEqual(prompt["model_path"], str(folder / "t5gemma-b-b-ul2"))
        self.assertNotIn("repo_id", prompt)
        self.assertNotIn("subfolder", prompt)
        self.assertEqual(seen["weights"], folder / "model.safetensors")
        self.assertEqual((info["device"], info["sample_rate"], engine.sample_size), ("cuda", 44100, 5292032))


class StableAudio3CatalogTests(unittest.TestCase):
    """FU-274: the three Stable Audio 3 rows, gated like Stable Audio Open, measured on an RTX 4090."""

    IDS = ("stabilityai/stable-audio-3-small-sfx", "stabilityai/stable-audio-3-small-music", "stabilityai/stable-audio-3-medium")

    def test_the_rows(self) -> None:
        rows = {v["id"]: v for v in music_variants()}
        for model_id in self.IDS:
            row = rows[model_id]
            self.assertEqual((row["kind"], row["pack"], row["engine"], row["engineLibrary"]), ("sound", "sound", "stableaudio", "stable-audio-3"))
            self.assertEqual(len(row["revision"]), 40)
            self.assertIn("cuda", row["platforms"])
            self.assertNotIn("textEncoder", row)  # the T5Gemma encoder is a folder of the same repo
            self.assertEqual(row["weightFolders"], [row["checkpointSubdir"], row["checkpointSubdir"] + "/t5gemma-b-b-ul2"])
            self.assertIn("t5gemma-b-b-ul2/*", row["downloadPatterns"])
            self.assertNotIn("Stable_Audio_3.0_Thumbnail_1x1.png", row["downloadPatterns"])
            licence = row["licence"]
            self.assertTrue(licence["gated"])
            self.assertEqual(licence["attribution"], "Powered by Stability AI")
            text = " ".join(licence["summaryLines"])
            for needle in ("US$1 million", "Gemma Terms of Use", "Prohibited Use Policy", "foundation model"):
                self.assertIn(needle, text)
            self.assertIn(f"/blob/{row['revision']}/LICENSE.md", licence["url"])
            self.assertIn(f"/blob/{row['revision']}/LICENSE_GEMMA.md", licence["gemmaUrl"])
            cuda = row["estimates"]["cuda"]
            self.assertEqual(cuda["basis"], "measured")
            self.assertTrue(cuda["wallSecondsAt"] and cuda["peakGb"] and cuda["hostPeakGb"])
            self.assertFalse(row["estimates"].get("fixedWindow"))  # variable length: a short clip is cheaper
        self.assertEqual(rows[self.IDS[0]]["makes"], ["sfx", "loops"])
        self.assertNotIn("sfx", rows[self.IDS[1]]["makes"])
        self.assertEqual((rows[self.IDS[2]]["maxSeconds"], rows[self.IDS[0]]["maxSeconds"]), (380, 120))

    def test_text_encoder_needs_nothing_extra_and_the_engine_label(self) -> None:
        row = next(v for v in music_variants() if v["id"] == self.IDS[0])
        self.assertTrue(music_sound_env.text_encoder_ready(row))
        self.assertTrue(music_sound_env.prepare_text_encoder(row))
        self.assertTrue(music_sound_env.engine_version(row).startswith("stable-audio-3 "))
        self.assertEqual(music_sound_env.engine_version(_row()), music_sound_env.ENGINE_VERSION)

    def test_a_refused_download_says_to_accept_the_licence(self) -> None:
        message = music_helpers.gated_music_message(self.IDS[2], "403 Client Error: Forbidden")
        self.assertIn("huggingface.co/stabilityai/stable-audio-3-medium", message)
        self.assertIn("accept the licence", message)


class _SoundBox(unittest.TestCase):
    """The real worker with a fake sound engine, patched in as the sounds worker."""

    def setUp(self) -> None:
        from tests.test_video_routes import make_client

        self.client, self._tmp, self._snapshot = make_client()
        self.box = MusicSandbox(flac=True).__enter__()
        engine_file = self.box.root / "fake_sound_engine.py"
        engine_file.write_text(SOUND_ENGINE, encoding="utf-8")
        sound_env = {**self.box.env, "CHAOSENGINE_MUSIC_ENGINE_MODULE": str(engine_file)}
        pack = types.SimpleNamespace(is_installed=lambda: True, venv_python=lambda: Path(sys.executable),
                                     worker_env=lambda: dict(sound_env))
        self.sound_worker = music_client.MusicWorker(pack=pack, log_name="sound-worker.log", label="sounds")
        self.music_worker = music_client.MusicWorker()
        self.box.patch(music_client, "get_sound_worker", lambda: self.sound_worker)
        self.box.patch(music_client, "_SOUND_WORKER", self.sound_worker)
        self.box.patch(music_client, "get_worker", lambda: self.music_worker)
        self.box.patch(music_client, "_WORKER", self.music_worker)
        self.box.patch(music_env, "lane_unavailable_reason", lambda: None)
        self.box.patch(music_sound_env, "lane_unavailable_reason", lambda: None)
        self.box.patch(music_runtime, "embed_tag_enabled", lambda: True)
        self.box.patch(music_runtime, "_prepare", lambda variant: None)
        self.box.patch(music_runtime, "readiness_error", lambda variant: None)
        self.box.patch(music_helpers, "weights_source", lambda variant: "checkpoints")

    def tearDown(self) -> None:
        from tests.test_video_routes import restore_env

        self.sound_worker.stop()
        self.music_worker.stop()
        self.box.__exit__(None, None, None)
        restore_env(self._snapshot)
        self._tmp.cleanup()


class SoundRouteTests(_SoundBox):
    def test_generate_saves_a_tagged_sound_in_the_library(self) -> None:
        response = self.client.post("/api/music/sound/generate", json={"caption": "wooden door creak", "seconds": 4, "loop": True, "seed": 11})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        item = body["items"][0]
        self.assertEqual((item["kind"], item["loop"], item["seed"], item["engine"]), ("sound", True, 11, "stableaudio"))
        self.assertIn("Powered by Stability AI", item["licence"])
        self.assertEqual(item["title"], "wooden door creak")
        settings = read_audio_settings(music_env.library_dir() / f"{item['id']}.flac")
        self.assertEqual((settings["kind"], settings["loop"], settings["seed"], settings["machine_generated"]),
                         ("sound", True, 11, True))
        self.assertEqual(settings["model"], SOUND)
        self.assertEqual(settings["prompt"], "wooden door creak")
        self.assertIn("Stability AI Community Licence", settings["licence"])
        self.assertNotIn(str(self.box.root), json.dumps(settings))  # no local path in the tag
        self.assertFalse(MUSIC_PROGRESS.snapshot()["active"])
        self.assertFalse(list((self.box.data / "tmp").iterdir()))

    def test_variations_use_consecutive_seeds(self) -> None:
        body = self.client.post("/api/music/sound/generate", json={"caption": "rain", "seconds": 2, "variations": 2, "seed": 3, "title": "rain"}).json()
        self.assertEqual([i["seed"] for i in body["items"]], [3, 4])
        self.assertEqual([i["title"] for i in body["items"]], ["rain (1)", "rain (2)"])

    def test_the_library_lists_sounds_next_to_tracks(self) -> None:
        self.client.post("/api/music/sound/generate", json={"caption": "footsteps on gravel", "seconds": 3})
        self.client.post("/api/music/generate", json={"caption": "piano", "instrumental": True, "seconds": 30, "modelId": "ACE-Step/Ace-Step1.5"})
        kinds = sorted(i["kind"] for i in self.client.get("/api/music/library").json()["items"])
        self.assertEqual(kinds, ["sound", "track"])

    def test_validation(self) -> None:
        for body in ({"caption": "x", "seconds": 381}, {"caption": "x", "seconds": 0.5}, {"caption": "  "}, {"caption": "x", "bpm": 90},
                     {"caption": "x", "variations": 3}, {"caption": "x" * 301}):
            self.assertEqual(self.client.post("/api/music/sound/generate", json=body).status_code, 422, body)

    def test_a_music_row_cannot_make_a_sound_and_the_reverse(self) -> None:
        response = self.client.post("/api/music/sound/generate", json={"caption": "x", "modelId": "ACE-Step/Ace-Step1.5"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("makes music", response.json()["detail"])
        response = self.client.post("/api/music/generate", json={"caption": "x", "instrumental": True, "seconds": 20, "modelId": SOUND})
        self.assertEqual(response.status_code, 400)
        self.assertIn("sound effects", response.json()["detail"])

    def test_not_ready_is_a_409_with_the_fix(self) -> None:
        with mock.patch.object(music_runtime, "readiness_error", lambda v: "The sounds pack is not installed. Install it from Music > Models."):
            response = self.client.post("/api/music/sound/generate", json={"caption": "x"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("sounds pack", response.json()["detail"])

    def test_estimate_is_per_model_range(self) -> None:
        ok = self.client.post("/api/music/estimate", json={"modelId": SOUND, "seconds": 5})
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertEqual(ok.json()["estimate"]["basis"], "measured")
        self.assertEqual(self.client.post("/api/music/estimate", json={"modelId": SOUND, "seconds": 30}).status_code, 400)
        self.assertEqual(self.client.post("/api/music/estimate", json={"modelId": "ACE-Step/Ace-Step1.5", "seconds": 5}).status_code, 400)

    def test_cancel_stops_a_running_sound(self) -> None:
        result: dict = {}
        thread = __import__("threading").Thread(
            target=lambda: result.update(r=self.client.post("/api/music/sound/generate", json={"caption": "delay=0.4", "seconds": 3})))
        thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and MUSIC_PROGRESS.snapshot()["step"] < 1:
            time.sleep(0.05)
        self.assertTrue(self.client.post("/api/music/cancel").json()["cancelled"])
        thread.join(20)
        self.assertEqual(result["r"].status_code, 409)

    def test_the_worker_limits_come_from_the_engine(self) -> None:
        # The sounds worker takes 5 s and refuses 12 s; the music worker (no limits declared) still wants 10 s or more.
        self.sound_worker.request("load", engine="stableaudio", model="stable-audio-open-small")
        out = self.box.data / "tmp" / "limit"
        ok = self.sound_worker.request("generate", job="snd_ok", caption="a", seconds=5, seed=1, out_dir=str(out), variations=1)
        self.assertTrue(ok["ok"])
        with self.assertRaisesRegex(music_client.MusicWorkerError, "between 1 and 11"):
            self.sound_worker.request("generate", job="snd_long", caption="a", seconds=12, seed=1, out_dir=str(out))
        with self.assertRaisesRegex(music_client.MusicWorkerError, "between 10 and 600"):
            self.music_worker.request("generate", job="mus_short", caption="a", seconds=5, seed=1, out_dir=str(out))

    def test_unload_stops_both_workers_and_idle_ones_make_room(self) -> None:
        self.sound_worker.request("ping")
        self.music_worker.request("ping")
        self.assertTrue(music_runtime.free_idle_worker(keep="sound"))
        self.assertTrue(self.sound_worker.running())
        self.assertFalse(self.music_worker.running())
        self.assertEqual(music_runtime.unload(), {"stopped": True})
        self.assertFalse(self.sound_worker.running())

    def test_runtime_reports_the_sounds_pack(self) -> None:
        runtime = self.client.get("/api/music/runtime").json()["runtime"]
        self.assertEqual(runtime["sound"]["engine"], "stableaudio")
        self.assertIn("install", runtime["sound"])
        self.assertIn("installed", runtime["sound"])

    def test_sound_install_routes(self) -> None:
        with mock.patch.object(music_sound_env, "start_install", lambda: {"phase": "preflight", "running": True}):
            self.assertEqual(self.client.post("/api/music/sound/install").json()["phase"], "preflight")
        self.assertIn("phase", self.client.get("/api/music/sound/install/status").json())
        self.assertIn("phase", self.client.post("/api/music/sound/install/cancel").json())


class MakeSoundToolTests(_SoundBox):
    """``make_sound`` is registered in the bridge (generate scope, counts as a job)."""

    def setUp(self) -> None:
        super().setUp()
        from backend_service.mcp_bridge import access

        access_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(access_tmp.cleanup)
        patcher = mock.patch.object(access, "_path", lambda: Path(access_tmp.name) / "agent_access.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        access.set_enabled(True)
        self.headers = {"Authorization": f"Bearer {access.get_token()}"}

    def _call(self, name: str, arguments: dict) -> tuple[bool, object]:
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
        result = self.client.post("/mcp", json=body, headers=self.headers).json()["result"]
        text = result["content"][0]["text"]
        try:
            return result["isError"], json.loads(text)
        except ValueError:
            return result["isError"], text

    def test_make_sound_job_to_a_fetchable_path(self) -> None:
        is_error, job = self._call("make_sound", {"caption": "door slam", "seconds": 3, "loop": True, "seed": 4})
        self.assertFalse(is_error, job)
        self.assertEqual(job["kind"], "music")
        _, done = self._call("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 30})
        self.assertEqual(done["state"], "done", done)
        output = done["output"]
        self.assertEqual((output["seed"], output["tracks"][0]["loop"]), (4, True))
        self.assertIn("Powered by Stability AI", output["licence"])
        self.assertTrue(Path(output["path"]).is_relative_to(music_env.library_dir()))
        self.assertEqual(self.client.get(output["download"]["url"]).status_code, 200)

    def test_make_sound_errors_are_readable(self) -> None:
        self.assertEqual(self._call("make_sound", {"caption": "x", "variations": 3}), (True, "variations must be 1, 2 or 4."))
        is_error, text = self._call("make_sound", {"caption": "x", "seconds": 30})
        self.assertTrue(is_error)
        self.assertIn("1 to 11 seconds", str(text))
        with mock.patch.object(music_runtime, "readiness_error", lambda v: "The sounds pack is not installed. Install it from Music > Models."):
            is_error, text = self._call("make_sound", {"caption": "x"})
        self.assertTrue(is_error)
        self.assertIn("sounds pack", str(text))

    def test_make_music_does_not_see_the_sound_model(self) -> None:
        with mock.patch.object(music_sound_env, "text_encoder_ready", lambda v: True):
            is_error, text = self._call("make_music", {"caption": "x", "instrumental": True, "model": SOUND})
        self.assertTrue(is_error)
        self.assertIn("not an installed music model", str(text))

    def test_the_tool_description_is_short_and_registered(self) -> None:
        from backend_service.mcp_bridge.sound_tool import SOUND_TOOLS

        self.assertLess(len(SOUND_TOOLS[0]["description"]), 330)
        self.assertEqual(SOUND_TOOLS[0]["inputSchema"]["required"], ["caption"])
        from backend_service.mcp_bridge import tools

        self.assertIn("make_sound", tools.HANDLERS)


if __name__ == "__main__":
    unittest.main()
