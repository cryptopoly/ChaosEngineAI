"""Stable Audio Open 1.0 as the second sound row: catalog row and licence, the request and estimate limits per
model, the command line the runtime sends the worker, the engine's per-model pipeline arguments, and the
``make_sound`` model choice. No weights and no torch: the worker is the real ``music_worker.py`` with the fake
engine, and the real engine file runs against stubbed ``torch`` and ``stable_audio_tools`` modules.
"""

from __future__ import annotations

import sys
import types
import unittest
from contextlib import nullcontext
from unittest import mock

from backend_service import music_runtime
from backend_service.catalog.music_models import music_variants, sound_max_seconds
from backend_service.helpers import music as music_helpers
from backend_service.helpers.hardware_class import host_hardware_class  # noqa: F401  (route import side effects)
from tests.music_fakes import FAKE_ENGINE
from tests.test_music_sound import SOUND, _SoundBox, _load_engine_module

FULL = "stabilityai/stable-audio-open-1.0"
FULL_ENGINE = FAKE_ENGINE + "\nSECONDS_RANGE = (1, 47)\nTASKS = ('text2sound',)\n"


def _row(model_id: str) -> dict:
    return next(v for v in music_variants() if v["id"] == model_id)


class CatalogTests(unittest.TestCase):
    def test_the_full_model_row(self) -> None:
        row = _row(FULL)
        self.assertEqual((row["kind"], row["pack"], row["engine"], row["engineModel"]), ("sound", "sound", "stableaudio", "stable-audio-open-1.0"))
        self.assertEqual((row["minSeconds"], row["maxSeconds"]), (1, 47))
        self.assertEqual(row["revision"], "f21265c1e2710b3bd2386596943f0007f55f802e")
        self.assertEqual(row["checkpointSubdir"], "stable-audio-open-1.0")
        self.assertEqual(row["weightFolders"], ["stable-audio-open-1.0"])
        self.assertEqual(row["defaultSettings"]["steps"], 100)
        self.assertEqual(row["defaultSettings"]["cfgScale"], 7.0)
        self.assertEqual(row["defaultSettings"]["sampler"], "dpmpp-3m-sde")
        self.assertLessEqual(row["defaultSettings"]["seconds"], row["maxSeconds"])
        self.assertEqual(row["defaultSettings"]["sampleRate"], 44100)
        self.assertEqual(sound_max_seconds(), 47)

    def test_the_text_encoder_is_the_one_small_uses(self) -> None:
        self.assertEqual(_row(FULL)["textEncoder"], _row(SOUND)["textEncoder"])
        self.assertEqual(_row(FULL)["downloadCompanions"], _row(SOUND)["downloadCompanions"])

    def test_only_the_stable_audio_tools_checkpoint_is_fetched(self) -> None:
        patterns = music_helpers.music_repo_allow_patterns(FULL)
        self.assertEqual(patterns, ["model.safetensors", "model_config.json", "LICENSE.md", "README.md"])
        self.assertEqual(music_helpers.music_repo_revision(FULL), _row(FULL)["revision"])

    def test_the_licence_is_the_gated_community_licence_with_the_attribution(self) -> None:
        licence = _row(FULL)["licence"]
        small = _row(SOUND)["licence"]
        for key in ("id", "name", "commercial", "gated", "attribution", "revenueCapUsd", "summaryLines", "note"):
            self.assertEqual(licence[key], small[key], key)
        self.assertEqual((licence["gated"], licence["attribution"]), (True, "Powered by Stability AI"))
        self.assertEqual(licence["url"], f"https://huggingface.co/{FULL}/blob/{_row(FULL)['revision']}/LICENSE.md")
        self.assertIn("/blob/" + _row(SOUND)["revision"] + "/LICENSE", small["url"])  # Small's link is unchanged

    def test_estimates_are_the_real_run_numbers(self) -> None:
        est = _row(FULL)["estimates"]
        self.assertEqual(est["basis"], "measured")
        self.assertIn("2026-11-08", est["measuredOn"])
        self.assertEqual(est["peakGb"], 22.8)
        self.assertGreater(est["peakGb"], est["idleGb"])
        self.assertEqual(est["cuda"]["peakGb"], None)
        self.assertEqual(_row(SOUND)["estimates"]["basis"], "measured")  # Small keeps its real numbers

    def test_the_default_sound_row_is_still_small(self) -> None:
        family = next(f for f in __import__("backend_service.catalog.music_models", fromlist=["x"]).MUSIC_MODEL_FAMILIES if f.get("kind") == "sound")
        self.assertEqual(family["defaultVariantId"], SOUND)
        self.assertEqual([v["id"] for v in family["variants"]], [SOUND, FULL])


class _Hooks:
    def __init__(self) -> None:
        self.steps: list[tuple[int, int]] = []
        self.devices: list[tuple[str, bool]] = []

    def step(self, done: int, total: int) -> None:
        self.steps.append((done, total))

    def check_cancel(self) -> None:
        pass

    def stage(self, *args: object) -> None:
        pass

    def device(self, name: str, fallback: bool = False) -> None:
        self.devices.append((name, fallback))


class _FakeTensor:
    pass


class EngineArgumentTests(unittest.TestCase):
    """The real engine file with ``generate_diffusion_cond`` captured: what each model is called with."""

    def setUp(self) -> None:
        self.engine_module = _load_engine_module()
        self.calls: list[dict] = []

        def fake_generate(model: object, **kwargs: object) -> object:
            self.calls.append(kwargs)
            callback = kwargs["callback"]
            for i in range(int(kwargs["steps"])):  # type: ignore[call-overload]
                callback({"i": i})
            return "audio"

        torch = types.ModuleType("torch")
        torch.inference_mode = lambda: nullcontext()  # type: ignore[attr-defined]
        generation = types.ModuleType("stable_audio_tools.inference.generation")
        generation.generate_diffusion_cond = fake_generate  # type: ignore[attr-defined]
        modules = {
            "torch": torch,
            "stable_audio_tools": types.ModuleType("stable_audio_tools"),
            "stable_audio_tools.inference": types.ModuleType("stable_audio_tools.inference"),
            "stable_audio_tools.inference.generation": generation,
        }
        patcher = mock.patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _engine(self, name: str, tmp: str = "/nonexistent") -> object:
        engine = self.engine_module.StableAudioEngine(tmp)  # type: ignore[arg-type]
        engine.model, engine.name, engine.device = object(), name, "cpu"
        return engine

    def test_the_full_model_gets_its_card_arguments(self) -> None:
        engine = self._engine("stable-audio-open-1.0")
        hooks = _Hooks()
        request = {"caption": "rain", "steps": 100, "cfg_scale": 7.0, "sampler": "dpmpp-3m-sde"}
        engine._run_one(request, 5, 30.0, hooks)
        call = self.calls[0]
        self.assertEqual(call["conditioning"], [{"prompt": "rain", "seconds_total": 30.0, "seconds_start": 0}])
        self.assertEqual((call["steps"], call["cfg_scale"], call["sampler_type"]), (100, 7.0, "dpmpp-3m-sde"))
        self.assertEqual((call["sigma_min"], call["sigma_max"]), (0.3, 500))
        self.assertEqual(hooks.steps[-1], (100, 100))

    def test_small_keeps_its_own_arguments(self) -> None:
        engine = self._engine("stable-audio-open-small")
        engine._run_one({"caption": "door", "steps": 8, "cfg_scale": 1.0, "sampler": "pingpong"}, 5, 4.0, _Hooks())
        call = self.calls[0]
        self.assertEqual(call["conditioning"], [{"prompt": "door", "seconds_total": 4.0}])
        self.assertNotIn("sigma_min", call)
        self.assertNotIn("sigma_max", call)
        self.assertEqual((call["steps"], call["sampler_type"]), (8, "pingpong"))

    def test_a_run_on_the_cpu_is_reported_so_the_ui_can_warn(self) -> None:
        engine = self._engine("stable-audio-open-1.0")  # the fake engine sits on the CPU
        hooks = _Hooks()
        engine._run_one({"caption": "rain", "steps": 4}, 5, 30.0, hooks)
        self.assertEqual(hooks.devices, [("cpu", False)])

    def test_a_gpu_failure_retries_on_the_cpu_and_says_so(self) -> None:
        engine = self._engine("stable-audio-open-1.0")
        engine.device = "mps"
        engine.model = mock.MagicMock()
        runs = []

        def run(request: dict, seed: int, seconds: float, hooks: object) -> str:
            runs.append(engine.device)
            if engine.device == "mps":
                raise RuntimeError("operation not implemented on MPS")
            hooks.device("cpu", fallback=engine._fell_back)  # type: ignore[attr-defined]
            return "audio"

        engine._run_one = run
        hooks = _Hooks()
        self.assertEqual(engine._generate_audio({}, 1, 30.0, hooks), "audio")
        self.assertEqual(runs, ["mps", "cpu"])
        self.assertEqual(hooks.devices, [("cpu", True)])

    def test_each_model_reads_its_own_folder(self) -> None:
        engine = self._engine("stable-audio-open-1.0", "/data")
        config, weights, encoder = engine._model_files("stable-audio-open-1.0")
        self.assertEqual(config.as_posix(), "/data/checkpoints/stable-audio-open-1.0/model_config.json")
        self.assertEqual(weights.as_posix(), "/data/checkpoints/stable-audio-open-1.0/model.safetensors")
        self.assertEqual(encoder.as_posix(), "/data/checkpoints/t5-base")
        self.assertEqual(engine._model_files("stable-audio-open-small")[1].as_posix(), "/data/checkpoints/stable-audio-open-small/model.safetensors")

    def test_the_engine_holds_a_clip_to_the_loaded_models_length(self) -> None:
        small = self._engine("stable-audio-open-small")
        with self.assertRaisesRegex(ValueError, "up to 11 s"):
            small.generate({"model": "stable-audio-open-small", "seconds": 12, "seed": 1, "out_dir": "/x", "job": "j"}, _Hooks())
        full = self._engine("stable-audio-open-1.0")
        with self.assertRaisesRegex(ValueError, "up to 47 s"):
            full.generate({"model": "stable-audio-open-1.0", "seconds": 48, "seed": 1, "out_dir": "/x", "job": "j"}, _Hooks())

    def test_an_unknown_model_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown sound model"):
            self._engine("stable-audio-open-small").load("stable-audio-open-2")


class _FullBox(_SoundBox):
    """``_SoundBox`` with a fake engine that allows 47 s, as the real engine file now declares."""

    def setUp(self) -> None:
        super().setUp()
        engine_file = self.box.root / "fake_sound_engine.py"
        engine_file.write_text(FULL_ENGINE, encoding="utf-8")


class RouteTests(_FullBox):
    def test_a_long_clip_is_refused_by_small_and_made_by_the_full_model(self) -> None:
        refused = self.client.post("/api/music/sound/generate", json={"caption": "x", "seconds": 30})
        self.assertEqual(refused.status_code, 400)
        self.assertIn("Stable Audio Open Small makes sounds of 1 to 11 seconds", refused.json()["detail"])
        made = self.client.post("/api/music/sound/generate", json={"caption": "rain", "seconds": 45, "modelId": FULL, "seed": 3})
        self.assertEqual(made.status_code, 200, made.text)
        item = made.json()["items"][0]
        self.assertEqual((item["model"], item["modelName"]), (FULL, "Stable Audio Open 1.0"))
        self.assertEqual(item["durationS"], 45.0)
        self.assertIn("Powered by Stability AI", item["licence"])

    def test_nothing_longer_than_the_widest_model_is_accepted(self) -> None:
        response = self.client.post("/api/music/sound/generate", json={"caption": "x", "seconds": 48, "modelId": FULL})
        self.assertEqual(response.status_code, 422)
        too_short = self.client.post("/api/music/sound/generate", json={"caption": "x", "seconds": 0.5, "modelId": FULL})
        self.assertEqual(too_short.status_code, 422)

    def test_estimate_follows_the_selected_model(self) -> None:
        ok = self.client.post("/api/music/estimate", json={"modelId": FULL, "seconds": 40})
        self.assertEqual(ok.status_code, 200, ok.text)
        body = ok.json()
        self.assertEqual(body["estimate"]["basis"], "measured")
        self.assertEqual(body["estimate"]["peakGb"], 22.8)
        self.assertGreater(body["estimate"]["seconds"], 200.0)  # about 208 s whatever the length, plus the cold start
        self.assertEqual(self.client.post("/api/music/estimate", json={"modelId": FULL, "seconds": 47}).status_code, 200)
        self.assertEqual(self.client.post("/api/music/estimate", json={"modelId": SOUND, "seconds": 40}).status_code, 400)

    def test_the_catalog_lists_both_sound_rows(self) -> None:
        families = self.client.get("/api/music/catalog").json()["families"]
        rows = [v for f in families if f.get("kind") == "sound" for v in f["variants"]]
        self.assertEqual([(r["id"], r["maxSeconds"]) for r in rows], [(SOUND, 11), (FULL, 47)])
        full = rows[1]
        self.assertEqual((full["licence"]["gated"], full["estimates"]["basis"]), (True, "measured"))

    def test_the_runtime_sends_the_worker_the_models_own_settings(self) -> None:
        with mock.patch.object(self.sound_worker, "request", wraps=self.sound_worker.request) as spy:
            response = self.client.post("/api/music/sound/generate", json={"caption": "wind", "seconds": 20, "modelId": FULL, "seed": 9, "loop": True})
        self.assertEqual(response.status_code, 200, response.text)
        calls = {c.args[0]: c.kwargs for c in spy.call_args_list if c.args[0] in ("load", "generate")}
        self.assertEqual(calls["load"]["model"], "stable-audio-open-1.0")
        gen = calls["generate"]
        self.assertEqual((gen["engine"], gen["model"], gen["task"]), ("stableaudio", "stable-audio-open-1.0", "text2sound"))
        self.assertEqual((gen["steps"], gen["cfg_scale"], gen["sampler"]), (100, 7.0, "dpmpp-3m-sde"))
        self.assertEqual((gen["seconds"], gen["loop"], gen["seed"], gen["format"]), (20.0, True, 9, "flac24"))
        # The timeouts are sized from this row's estimate: sized from the measured time.
        self.assertGreater(gen["timeout"], 525.0)

    def test_small_still_gets_eight_steps_and_pingpong(self) -> None:
        with mock.patch.object(self.sound_worker, "request", wraps=self.sound_worker.request) as spy:
            self.client.post("/api/music/sound/generate", json={"caption": "door", "seconds": 3})
        gen = next(c.kwargs for c in spy.call_args_list if c.args[0] == "generate")
        self.assertEqual((gen["model"], gen["steps"], gen["cfg_scale"], gen["sampler"]), ("stable-audio-open-small", 8, 1.0, "pingpong"))

    def test_the_encoder_tag_names_the_model(self) -> None:
        self.assertIn("Stable Audio Open 1.0", music_runtime._encoder_label(_row(FULL)))
        self.assertIn("Stable Audio Open Small", music_runtime._encoder_label(_row(SOUND)))


class MakeSoundModelTests(_FullBox):
    def setUp(self) -> None:
        super().setUp()
        import tempfile
        from pathlib import Path

        from backend_service.mcp_bridge import access

        access_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(access_tmp.cleanup)
        patcher = mock.patch.object(access, "_path", lambda: Path(access_tmp.name) / "agent_access.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        access.set_enabled(True)
        self.headers = {"Authorization": f"Bearer {access.get_token()}"}

    def _call(self, name: str, arguments: dict) -> tuple[bool, object]:
        import json

        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
        result = self.client.post("/mcp", json=body, headers=self.headers).json()["result"]
        text = result["content"][0]["text"]
        try:
            return result["isError"], json.loads(text)
        except ValueError:
            return result["isError"], text

    def test_the_schema_allows_the_widest_model_and_names_the_choice(self) -> None:
        from backend_service.mcp_bridge.sound_tool import SOUND_TOOLS

        props = SOUND_TOOLS[0]["inputSchema"]["properties"]
        self.assertEqual(props["seconds"]["maximum"], 47)
        self.assertIn("modelId", props)
        self.assertIn("47 s", SOUND_TOOLS[0]["description"])

    def test_a_long_sound_needs_the_full_model(self) -> None:
        is_error, text = self._call("make_sound", {"caption": "x", "seconds": 30})
        self.assertTrue(is_error)
        self.assertIn("Stable Audio Open Small makes sounds of 1 to 11 seconds", str(text))
        is_error, text = self._call("make_sound", {"caption": "x", "seconds": 48, "modelId": FULL})
        self.assertTrue(is_error)
        self.assertIn("between 1 and 47", str(text))  # the schema's own bound, before the model check

    def test_make_sound_with_the_full_model_runs_to_a_path(self) -> None:
        is_error, job = self._call("make_sound", {"caption": "rain", "seconds": 40, "modelId": FULL, "seed": 2})
        self.assertFalse(is_error, job)
        _, done = self._call("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 30})
        self.assertEqual(done["state"], "done", done)
        self.assertEqual(done["output"]["model"], "Stable Audio Open 1.0")
        self.assertEqual(done["output"]["tracks"][0]["durationSeconds"], 40.0)

    def test_the_model_choice_must_be_a_sound_row(self) -> None:
        is_error, text = self._call("make_sound", {"caption": "x", "modelId": "ACE-Step/Ace-Step1.5"})
        self.assertTrue(is_error)
        self.assertIn("make_music", str(text))
        is_error, text = self._call("make_sound", {"caption": "x", "modelId": "nope/none"})
        self.assertTrue(is_error)


if __name__ == "__main__":
    unittest.main()
