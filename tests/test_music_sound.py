"""Sounds (Stable Audio Open Small): catalog row, licence, gated download message, sounds pack env, worker limits,
the pure audio helpers, routes, the library kind and the ``make_sound`` agent tool.

No real model runs: the worker is the real ``music_worker.py`` with the fake engine from ``music_fakes`` plus
the two attributes a sound engine file declares (``SECONDS_RANGE``, ``TASKS``). The real engine's wiring was
checked by hand against stable-audio-tools with a random mini model (see ``docs/hub/progress/v-audio.md``).
"""

from __future__ import annotations

import importlib.util
import json
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
            self.assertIn("Apple Silicon", music_sound_env.lane_unavailable_reason())

    def test_marker_needs_the_release_the_venv_and_its_python(self) -> None:
        self.assertFalse(music_sound_env.is_installed())
        python = music_sound_env.venv_python()
        python.parent.mkdir(parents=True)
        python.write_text("")
        music_sound_env._write_marker({"torch": "2.7.1"})
        self.assertTrue(music_sound_env.is_installed())
        marker = music_sound_env.read_marker()
        marker["version"] = "0.0.1"
        music_sound_env._marker_path().write_text(json.dumps(marker))
        self.assertFalse(music_sound_env.is_installed())

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
        status = music_sound_env.job_status()
        self.assertEqual((status["phase"], status["error"]), ("done", None))
        flat = [" ".join(c) for c in commands]
        self.assertTrue(any("-m venv" in c for c in flat))
        self.assertTrue(any("-r" in c and "sound_requirements.txt" in c for c in flat))
        engine = [c for c in flat if "stable-audio-tools==" in c][0]
        self.assertIn("--no-deps", engine)
        self.assertIn("--ignore-requires-python", engine)
        self.assertTrue(music_sound_env.is_installed())

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
        self.assertEqual((self.engine.SECONDS_RANGE, self.engine.TASKS), ((1, 47), ("text2sound",)))
        self.assertEqual({k: v["max_seconds"] for k, v in self.engine.MODEL_PROFILES.items()},
                         {"stable-audio-open-small": 11, "stable-audio-open-1.0": 47})


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
        for body in ({"caption": "x", "seconds": 48}, {"caption": "x", "seconds": 0.5}, {"caption": "  "}, {"caption": "x", "bpm": 90},
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
