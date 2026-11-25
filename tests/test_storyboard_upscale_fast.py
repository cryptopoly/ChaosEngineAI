"""Storyboard export ``upscaleMethod``: lanczos stays the default; "fast" upscales each shot before the film is joined
at the final size. The AI model is a pixel-doubling fake run through the real video upscale pipeline."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from backend_service.mcp_bridge import tools as _tools  # noqa: F401 - loads storyboard_tools in its import order
from backend_service.mcp_bridge.storyboard_tools import _format_fields
from backend_service.storyboard import assemble, store as store_module
from backend_service.storyboard.models import ExportSettings, Project
from backend_service.storyboard.render_plan import PlanError, RenderPlan
from backend_service.video_upscale import addon, pipeline
from tests.video_upscale_fakes import DoublePixels, count_frames, find_ffmpeg, make_clip

FFMPEG = find_ffmpeg()


class FormatTests(unittest.TestCase):
    def test_default_is_lanczos_and_old_projects_load_unchanged(self):
        self.assertEqual(ExportSettings().upscaleMethod, "lanczos")
        old = {"schemaVersion": 3, "id": "0123456789abcdef", "name": "Old", "export": {"upscale": "1440p"}}
        project = Project.model_validate(old)
        self.assertEqual((project.export.upscale, project.export.upscaleMethod), ("1440p", "lanczos"))
        self.assertEqual(ExportSettings(upscaleMethod="fast").upscaleMethod, "fast")
        with self.assertRaises(ValidationError):
            ExportSettings(upscaleMethod="quality")

    def test_plan_checks_the_method(self):
        base = {"width": 160, "height": 90, "fps": 24, "out_path": str(Path.home() / "o.mp4"),
                "work_dir": str(Path.home()), "shots": [{"id": "a", "video": __file__, "hash": "h"}]}
        self.assertEqual(RenderPlan.from_dict(base).upscale_method, "lanczos")
        with self.assertRaises(PlanError):
            RenderPlan.from_dict({**base, "upscale_method": "magic"})
        plan = RenderPlan.from_dict({**base, "upscale_method": "fast", "upscale_to": [320, 180]})
        self.assertEqual(RenderPlan.from_dict(plan.to_dict()), plan)
        # The fast names join the resume list only when the film is upscaled that way.
        self.assertEqual(len(assemble.intermediate_names(plan)), 2)
        self.assertEqual(len(assemble.intermediate_names(RenderPlan.from_dict(base))), 1)
        preview = RenderPlan.from_dict({**base, "upscale_method": "fast", "upscale_to": [320, 180], "preview": True})
        self.assertEqual(len(assemble.intermediate_names(preview)), 1)

    def test_agent_tool_folds_the_method_into_the_export(self):
        fields = _format_fields({"upscale": "2160p", "upscale_method": "fast"})
        self.assertEqual(fields["export"], {"upscale": "2160p", "upscaleMethod": "fast"})


def fake_shot_upscaler(calls: list):
    """What ``storyboard_hook.shot_upscaler`` returns, with the fake model in place of the worker."""

    def run(src, dst, size, cancel_event, report):
        calls.append((Path(src).name, size))
        spec = pipeline.make_spec(FFMPEG, Path(src), Path(dst), model_id="fake", weights="none", size=size,
                                  avtb=True, codec_args=["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt",
                                                         "yuv420p"])
        pipeline.run_job(spec, pipeline.local_runner(DoublePixels()), exe=FFMPEG, work=Path(dst).parent / "upwork",
                         progress=lambda done, total, i, n: report(done / total),
                         is_cancelled=cancel_event.is_set if cancel_event else (lambda: False))

    return run


@unittest.skipUnless(FFMPEG, "ffmpeg is not available")
class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name).resolve()
        make_clip(FFMPEG, cls.root / "a.mp4", seconds=1.0, fps=24, size="160x90")
        make_clip(FFMPEG, cls.root / "b.mp4", seconds=1.0, fps=24, size="160x90")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def plan(self, name: str, crossfade: bool = False, **over) -> RenderPlan:
        work = self.root / f"w-{name}"
        work.mkdir(exist_ok=True)
        second = {"id": "b", "video": str(self.root / "b.mp4"), "hash": "hb"}
        if crossfade:
            second["transition_in"] = {"type": "crossfade", "seconds": 0.25}
        data = {"width": 160, "height": 90, "fps": 24, "out_path": str(self.root / f"{name}.mp4"), "work_dir": str(work),
                "loudness_lufs": None, "upscale_to": [320, 180], "upscale_method": "fast",
                "shots": [{"id": "a", "video": str(self.root / "a.mp4"), "hash": "ha"}, second]}
        data.update(over)
        return RenderPlan.from_dict(data)

    def test_fast_upscales_each_shot_and_joins_at_the_final_size(self):
        calls: list = []
        plan = self.plan("cuts")
        result = assemble.render(plan, shot_upscaler=fake_shot_upscaler(calls))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(size == (320, 180) for _name, size in calls))
        frames, width, height = count_frames(FFMPEG, Path(result.path))
        self.assertEqual((width, height), (320, 180))
        self.assertEqual(frames, 48)
        self.assertAlmostEqual(result.duration_s, 2.0, delta=0.1)
        # A second export reuses the upscaled shots.
        again: list = []
        assemble.render(plan, shot_upscaler=fake_shot_upscaler(again))
        self.assertEqual(again, [])

    def test_fast_with_a_crossfade(self):
        result = assemble.render(self.plan("xfade", crossfade=True), shot_upscaler=fake_shot_upscaler([]))
        self.assertEqual(count_frames(FFMPEG, Path(result.path))[1:], (320, 180))

    def test_lanczos_is_unchanged_and_never_calls_the_upscaler(self):
        calls: list = []
        result = assemble.render(self.plan("lanczos", upscale_method="lanczos"), shot_upscaler=fake_shot_upscaler(calls))
        self.assertEqual(calls, [])
        self.assertEqual(count_frames(FFMPEG, Path(result.path))[1:], (320, 180))

    def test_without_the_addon_the_render_says_how_to_install_it(self):
        with mock.patch.object(addon, "site_installed", return_value=False):
            with self.assertRaises(assemble.RenderError) as caught:
                assemble.render(self.plan("missing"))
        self.assertIn("Install video upscale", str(caught.exception))

    def test_cancel_during_the_upscale(self):
        cancel = threading.Event()

        def upscaler(src, dst, size, cancel_event, report):
            cancel.set()
            raise pipeline.Cancelled("cancelled")

        with self.assertRaises(assemble.RenderCancelled):
            assemble.render(self.plan("cancel"), cancel_event=cancel, shot_upscaler=upscaler)


@unittest.skipUnless(FFMPEG, "ffmpeg is not available")
class StartRenderTests(unittest.TestCase):
    def test_export_is_refused_before_a_job_exists_without_the_addon(self):
        from backend_service.routes import storyboard as routes

        fake_store = mock.Mock()
        plan = {"upscale_method": "fast", "upscale_to": [2560, 1440]}
        with mock.patch.object(routes, "get_store", return_value=fake_store), \
                mock.patch("backend_service.storyboard.timeline.build_render_plan", return_value=plan), \
                mock.patch.object(addon, "installed", return_value=False), \
                mock.patch("backend_service.storyboard.render_adapter.RENDERS") as renders:
            with self.assertRaises(store_module.StoreError) as caught:
                routes.start_render("0123456789abcdef", None, preview=False)
        self.assertIn("Install video upscale", str(caught.exception))
        self.assertIn("Lanczos", str(caught.exception))
        renders.start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
