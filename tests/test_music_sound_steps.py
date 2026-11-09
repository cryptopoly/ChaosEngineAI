"""Sound steps (D-74): the per-row range, the 422 at the request edge, the estimate and timeout scaling, and the
steps reaching the progress total, the worker request, the library item and the FLAC tag."""

from __future__ import annotations

import unittest
from unittest import mock

from backend_service import music_client, music_env, music_runtime
from backend_service.catalog.music_models import music_variants
from backend_service.helpers.audio_metadata import read_audio_settings
from backend_service.mcp_bridge import tools as _bridge_tools  # noqa: F401  (loads the tool modules in order)
from backend_service.mcp_bridge.sound_tool import SOUND_TOOLS
from backend_service.progress import MUSIC_PROGRESS
from tests.test_music_sound import _SoundBox

SMALL = "stabilityai/stable-audio-open-small"
FULL = "stabilityai/stable-audio-open-1.0"


def _row(model_id: str) -> dict:
    return next(v for v in music_variants() if v["id"] == model_id)


class StepsCatalogTests(unittest.TestCase):
    def test_each_sound_row_has_a_range_that_holds_its_default_and_presets(self) -> None:
        for model_id, (low, high, presets) in {SMALL: (1, 32, [4, 8, 16]), FULL: (10, 150, [25, 50, 100])}.items():
            row = _row(model_id)
            options = row["stepsOptions"]
            self.assertEqual((options["min"], options["max"], options["presets"]), (low, high, presets))
            self.assertIn(row["defaultSettings"]["steps"], options["presets"])
            self.assertTrue(all(low <= p <= high for p in options["presets"]))

    def test_music_rows_have_no_steps_setting(self) -> None:
        self.assertFalse(any("stepsOptions" in v for v in music_variants() if v.get("kind") != "sound"))

    def test_check_steps(self) -> None:
        self.assertIsNone(music_runtime.check_steps(_row(FULL), None))
        self.assertEqual(music_runtime.check_steps(_row(FULL), 10), 10)
        self.assertEqual(music_runtime.check_steps(_row(FULL), 150), 150)
        for steps in (9, 151):
            with self.assertRaisesRegex(ValueError, "10 to 150"):
                music_runtime.check_steps(_row(FULL), steps)
        with self.assertRaises(ValueError):
            music_runtime.check_steps(_row(SMALL), 33)


class StepsEstimateTests(unittest.TestCase):
    def test_default_steps_keep_the_measured_numbers_and_basis(self) -> None:
        for steps in (None, 100):
            run = music_runtime.estimate_run(FULL, 5, 1, steps=steps)
            self.assertAlmostEqual(run["seconds"], 207.9, delta=1.0)
            self.assertEqual((run["basis"], run["basisNote"], run["steps"]), ("measured", None, 100))

    def test_fifty_steps_is_about_half_of_a_hundred_plus_the_fixed_costs(self) -> None:
        full = music_runtime.estimate_run(FULL, 5, 1)["seconds"]
        half = music_runtime.estimate_run(FULL, 5, 1, steps=50)
        self.assertGreater(half["seconds"], full / 2)  # the VAE decode does not shrink
        self.assertLess(half["seconds"], full / 2 + 10)
        self.assertEqual((half["basis"], half["steps"]), ("estimated", 50))
        self.assertIn("measured run", half["basisNote"])

    def test_small_scales_too_and_cold_start_is_not_scaled(self) -> None:
        warm8 = music_runtime.estimate_run(SMALL, 5, 1)["seconds"]
        warm16 = music_runtime.estimate_run(SMALL, 5, 1, steps=16)["seconds"]
        warm1 = music_runtime.estimate_run(SMALL, 5, 1, steps=1)["seconds"]
        self.assertGreater(warm16, warm8)
        self.assertLess(warm1, warm8)
        cold = music_runtime.estimate_run(FULL, 5, 1, warm=False, steps=50)
        self.assertEqual(cold["coldStartSeconds"], 7.0)
        self.assertAlmostEqual(cold["seconds"], music_runtime.estimate_run(FULL, 5, 1, steps=50)["seconds"] + 7.0, places=1)

    def test_variations_multiply_and_music_rows_ignore_steps(self) -> None:
        one = music_runtime.estimate_run(FULL, 5, 1, steps=50)["seconds"]
        self.assertAlmostEqual(music_runtime.estimate_run(FULL, 5, 2, steps=50)["seconds"], 2 * one, delta=0.2)
        self.assertEqual(music_runtime.estimate_run("ACE-Step/Ace-Step1.5", 60, 1, steps=3)["basis"],
                         music_runtime.estimate_run("ACE-Step/Ace-Step1.5", 60, 1)["basis"])

    def test_timeout_and_stall_follow_the_steps(self) -> None:
        short = music_client.generate_timeout_s(5, 1, FULL, 50)
        long = music_client.generate_timeout_s(5, 1, FULL, 100)
        self.assertLess(short, long)
        warm50 = music_runtime.estimate_run(FULL, 5, 1, steps=50)["seconds"]
        self.assertGreaterEqual(short, 3 * warm50)
        self.assertGreaterEqual(music_client.generate_timeout_s(5, 1, FULL, 150), 3 * 300.0)
        self.assertEqual(music_client.stall_limits(5, 1, FULL, 50)["default"], music_client.STALL_DEFAULT_S)


class StepsRouteTests(_SoundBox):
    def test_estimate_route_scales_and_refuses_out_of_range(self) -> None:
        base = {"modelId": FULL, "seconds": 5}
        default = self.client.post("/api/music/estimate", json=base).json()["estimate"]
        fifty = self.client.post("/api/music/estimate", json={**base, "steps": 50}).json()["estimate"]
        self.assertLess(fifty["seconds"], default["seconds"])
        self.assertEqual((default["basis"], fifty["basis"]), ("measured", "estimated"))
        self.assertEqual(self.client.post("/api/music/estimate", json={**base, "steps": 5}).status_code, 422)
        self.assertEqual(self.client.post("/api/music/estimate", json={**base, "steps": 151}).status_code, 422)

    def test_generate_validates_steps_per_row(self) -> None:
        for body in ({"modelId": FULL, "steps": 9}, {"modelId": FULL, "steps": 151}, {"modelId": SMALL, "steps": 33},
                     {"modelId": SMALL, "steps": 0}, {"steps": 33}, {"modelId": "ACE-Step/Ace-Step1.5", "steps": 8}):
            response = self.client.post("/api/music/sound/generate", json={"caption": "x", "seconds": 2, **body})
            self.assertEqual(response.status_code, 422, body)

    def test_missing_steps_uses_the_row_default(self) -> None:
        with mock.patch.object(MUSIC_PROGRESS, "begin", wraps=MUSIC_PROGRESS.begin) as begin:
            body = self.client.post("/api/music/sound/generate", json={"caption": "rain", "seconds": 2, "seed": 1}).json()
        self.assertEqual(begin.call_args.kwargs["total_steps"], 8)
        self.assertEqual(body["items"][0]["steps"], 8)

    def test_chosen_steps_reach_progress_library_and_tag(self) -> None:
        with mock.patch.object(MUSIC_PROGRESS, "begin", wraps=MUSIC_PROGRESS.begin) as begin:
            response = self.client.post("/api/music/sound/generate", json={"caption": "rain", "seconds": 2, "seed": 4, "steps": 16})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(begin.call_args.kwargs["total_steps"], 16)
        item = response.json()["items"][0]
        self.assertEqual((item["steps"], item["seed"]), (16, 4))
        settings = read_audio_settings(music_env.library_dir() / f"{item['id']}.flac")
        self.assertEqual(str(settings["steps"]), "16")


class StepsAgentToolTests(unittest.TestCase):
    def test_make_sound_declares_steps(self) -> None:
        props = SOUND_TOOLS[0]["inputSchema"]["properties"]
        self.assertEqual((props["steps"]["type"], props["steps"]["minimum"], props["steps"]["maximum"]), ("integer", 1, 150))

    def test_make_sound_refuses_steps_outside_the_row_range(self) -> None:
        from backend_service.mcp_bridge.sound_tool import make_sound
        from backend_service.mcp_bridge.tools import ToolError

        with mock.patch.object(music_runtime, "readiness_error", lambda v: None):
            with self.assertRaisesRegex(ToolError, "10 to 150"):
                make_sound(mock.Mock(), {"caption": "x", "modelId": FULL, "steps": 5})


if __name__ == "__main__":
    unittest.main()
