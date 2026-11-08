"""Project format (D-64): aspect ratio, resolution, frame rate, the old-shape migration, the export upscale."""

import unittest

from backend_service.catalog.size_limits import apply_video_size_limits
from backend_service.storyboard import formats
from backend_service.storyboard.assemble_graphs import build_mix_graph, build_shot_graph, upscale_filter
from backend_service.storyboard.models import Defaults, ExportSettings, Project, dump
from backend_service.storyboard.ops import EditProject, edit_project
from backend_service.storyboard.render_plan import RenderPlan, ShotInput
from backend_service.storyboard.timeline import build_render_plan, output_size, upscale_target
from tests.test_storyboard_timeline import project, shot


class SizeTableTests(unittest.TestCase):
    def test_sizes_per_aspect_and_resolution(self):
        self.assertEqual(
            [formats.size_for("16:9", r) for r in formats.RESOLUTIONS],
            [(854, 480), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)],
        )
        self.assertEqual(formats.size_for("9:16", "1080p"), (1080, 1920))
        self.assertEqual(formats.size_for("1:1", "480p"), (480, 480))
        self.assertEqual(formats.size_for("1:1", "2160p"), (2160, 2160))

    def test_every_size_is_even(self):
        for aspect in formats.ASPECTS:
            for res in formats.RESOLUTIONS:
                self.assertTrue(all(side % 2 == 0 for side in formats.size_for(aspect, res)), (aspect, res))

    def test_unknown_choices_are_refused(self):
        with self.assertRaises(ValueError):
            formats.size_for("4:3", "720p")
        with self.assertRaises(ValueError):
            formats.size_for("16:9", "360p")

    def test_frame_rates_offered(self):
        self.assertEqual(formats.FPS_CHOICES, (24, 25, 30, 48, 50, 60))


class MigrationTests(unittest.TestCase):
    def test_old_shapes_read_as_aspect_at_720p(self):
        for old, aspect, size in [
            ({"width": 1280, "height": 720, "exportPreset": "16:9"}, "16:9", (1280, 720)),
            ({"width": 720, "height": 1280, "exportPreset": "9:16"}, "9:16", (720, 1280)),
            ({"width": 1024, "height": 1024, "exportPreset": "1:1"}, "1:1", (720, 720)),
        ]:
            d = Defaults.model_validate({**old, "fps": 30})
            self.assertEqual((d.exportPreset, d.resolution, (d.width, d.height), d.fps), (aspect, "720p", size, 30))

    def test_a_stored_project_without_resolution_opens(self):
        stored = dump(project([shot("a")]))
        del stored["defaults"]["resolution"]
        stored["defaults"].update(width=1024, height=1024, exportPreset="1:1")
        self.assertEqual(Project.model_validate(stored).defaults.resolution, "720p")

    def test_a_custom_size_is_kept(self):
        d = Defaults.model_validate({"width": 832, "height": 480, "exportPreset": "16:9"})
        self.assertEqual((d.width, d.height), (832, 480))

    def test_aspect_and_old_shape_are_names_for_the_preset(self):
        self.assertEqual(Defaults(aspect="9:16", resolution="1080p").model_dump()["width"], 1080)
        self.assertEqual(Defaults(shape="1:1").exportPreset, "1:1")
        self.assertEqual((Defaults(aspect="16:9", resolution="480p").width, Defaults(aspect="16:9", resolution="480p").height), (854, 480))

    def test_edit_follows_the_format(self):
        p = project([shot("a")])
        edit_project(p, None, EditProject(defaults={"aspect": "9:16"}))
        self.assertEqual((p.defaults.width, p.defaults.height), (720, 1280))
        edit_project(p, None, EditProject(defaults={"resolution": "1080p"}))
        self.assertEqual((p.defaults.exportPreset, p.defaults.width, p.defaults.height), ("9:16", 1080, 1920))
        edit_project(p, None, EditProject(defaults={"fps": 50}))
        self.assertEqual((p.defaults.fps, p.defaults.width), (50, 1080))


class ExportScalingTests(unittest.TestCase):
    def test_joined_clips_are_fitted_to_the_project_size(self):
        plan = RenderPlan(width=1920, height=1080, fps=25, out_path="/x/o.mp4", work_dir="/x/w",
                          shots=[ShotInput(id="a", video="/x/a.mp4", hash="h")])
        graph = build_shot_graph(plan.shots[0], plan, True, 50, 0.0, 2.0)
        self.assertIn("fps=25,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:", graph)

    def test_upscale_is_a_lanczos_scale_in_the_final_graph(self):
        self.assertEqual(upscale_filter((2560, 1440)), "scale=2560:1440:flags=lanczos,setsar=1")
        graph, _ = build_mix_graph([], 48000, None, None, (2560, 1440))
        self.assertIn("[0:v]scale=2560:1440:flags=lanczos,setsar=1[vout]", graph)
        graph, _ = build_mix_graph([], 48000, None, "c.srt", (2560, 1440))
        self.assertIn("[0:v]subtitles=filename=c.srt,scale=2560:1440:flags=lanczos,setsar=1[vout]", graph)
        graph, _ = build_mix_graph([], 48000, None)
        self.assertNotIn("[vout]", graph)

    def test_upscale_target_by_aspect_and_never_downward(self):
        p = project([shot("a")])
        self.assertIsNone(upscale_target(p, (1280, 720)))  # same size
        p.export = ExportSettings(upscale="2160p")
        self.assertEqual(upscale_target(p, (1280, 720)), [3840, 2160])
        p.defaults = Defaults(aspect="9:16")
        self.assertEqual(upscale_target(p, (720, 1280)), [2160, 3840])
        p.export = ExportSettings(upscale="1440p")
        self.assertIsNone(upscale_target(p, (2560, 1440)))  # not larger than the film

    def test_plan_carries_the_upscale_but_a_preview_does_not(self):
        import tempfile
        from pathlib import Path

        p = project([shot("a")])
        p.export = ExportSettings(upscale="1440p")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "media" / "a" / "ta").mkdir(parents=True)
            (folder / "media" / "a" / "ta" / "clip.mp4").write_bytes(b"x")
            (folder / "exports").mkdir()
            (folder / "cache" / "render").mkdir(parents=True)
            self.assertEqual(build_render_plan(p, preview=False, folder=folder)["upscale_to"], [2560, 1440])
            self.assertIsNone(build_render_plan(p, preview=True, folder=folder)["upscale_to"])
            plan = build_render_plan(p, preview=False, folder=folder)
            self.assertEqual(RenderPlan.from_dict(plan).upscale_to, (2560, 1440))
            with self.assertRaises(Exception):
                RenderPlan.from_dict({**plan, "upscale_to": [2561, 1440]})

    def test_output_size_follows_the_format(self):
        p = project([shot("a")])
        p.defaults = Defaults(aspect="9:16", resolution="480p")
        self.assertEqual(output_size(p, False), (480, 854))
        p.defaults = Defaults(aspect="1:1", resolution="1080p")
        self.assertEqual(output_size(p, False), (1080, 1080))


class NativeFpsTests(unittest.TestCase):
    def test_catalog_states_a_rate_only_where_a_card_does(self):
        families = [
            {"id": "wan-2-2", "variants": [{"id": "w", "familyId": "wan-2-2"}]},
            {"id": "ltx-2", "variants": [{"id": "l", "familyId": "ltx-2"}]},
            {"id": "cogvideox", "variants": [{"id": "THUDM/CogVideoX1.5-5B", "familyId": "cogvideox"}]},
        ]
        apply_video_size_limits(families)
        got = {f["variants"][0]["id"]: f["variants"][0].get("nativeFps") for f in families}
        self.assertEqual(got, {"w": 16, "l": None, "THUDM/CogVideoX1.5-5B": 16})


if __name__ == "__main__":
    unittest.main()
