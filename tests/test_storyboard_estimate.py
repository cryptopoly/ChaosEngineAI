"""The batch estimate (D-65 phase 3): sums from the catalog or this project's measured takes, never invented."""

from types import SimpleNamespace

from backend_service.storyboard.estimate import estimate, plan_steps
from tests.storyboard_fixtures import StoreCase, fake_mp4, png
from tests.test_storyboard_routes import StoryboardRouteCase

IMAGE = {"id": "img", "name": "Img One", "estimatedGenerationSeconds": 5.0, "runtimeFootprintMpsGb": 9.0,
         "runtimeFootprintGb": 9.0, "sizeGb": 6.0}
VIDEO = {"id": "vid", "name": "Vid One", "estimatedGenerationSeconds": 40.0, "defaultDurationSeconds": 4.0,
         "runtimeFootprintMpsGb": 20.0, "runtimeFootprintGb": 20.0, "sizeGb": 15.0}
HW = SimpleNamespace(kind="apple", bucket="apple-64", label="Apple Silicon, 64 GB", total_gb=64.0, usable_gb=48.0)


class EstimateTests(StoreCase):
    def test_catalog_sum_peak_and_unknown_disk(self):
        p = self.new(shots=2)
        p = self.op(p, "edit_shot", shotId=p.shots[1].id, seconds=8)
        out = estimate(p, self.folder(p), image_variant=IMAGE, video_variant=VIDEO, missing=[], hardware=HW)
        # Two stills at 5 s, clips scaled from the 4 s catalog clip: 40 + 80.
        self.assertEqual((out["stills"], out["clips"], out["seconds"]), (2, 2, 130.0))
        self.assertEqual((out["peakGb"], out["totalGb"], out["basis"]), (20.0, 48.0, "estimated"))
        self.assertIsNone(out["diskMb"])  # no still or clip on disk yet to measure from

    def test_measured_takes_win_and_disk_comes_from_real_files(self):
        p = self.new(shots=2)
        a = p.shots[0].id
        p = self.op(p, "import_take", shotId=a, still=str(png(self.src / "s.png")), clip=str(fake_mp4(self.src / "c.mp4")))
        project = self.store.load(p.id)
        take = project.shot(a).active()
        take.settings = {"stillSeconds": 3.0, "stillModel": "Img One", "clipSeconds": 20.0, "clipModel": "Vid One"}
        take.measuredSeconds = 4.0
        out = estimate(project, self.folder(p), what="both", shot_ids=[project.shots[1].id],
                       image_variant=IMAGE, video_variant=VIDEO, missing=[], hardware=HW)
        self.assertEqual((out["seconds"], out["basis"]), (3.0 + 5.0 * 4.0, "measured"))
        self.assertIsNotNone(out["diskMb"])  # from the real still and clip sizes (tiny here)

    def test_still_only_shot_needs_only_its_clip_and_a_missing_model_is_unknown(self):
        p = self.new(shots=1)
        p = self.op(p, "import_take", shotId=p.shots[0].id, still=str(png(self.src / "s.png")))
        self.assertEqual([k for _, k in plan_steps(p, "both", None)], ["clip"])
        out = estimate(p, self.folder(p), image_variant=None, video_variant=None,
                       missing=["No video model is installed."], hardware=HW)
        self.assertEqual((out["clips"], out["stills"], out["seconds"], out["peakGb"]), (1, 0, None, None))
        self.assertEqual(out["missing"], ["No video model is installed."])

    def test_nothing_to_make(self):
        p = self.new()
        out = estimate(p, self.folder(p), image_variant=IMAGE, video_variant=VIDEO, missing=[], hardware=HW)
        self.assertEqual((out["shots"], out["seconds"], out["basis"]), (0, None, None))


class EstimateRouteTests(StoryboardRouteCase):
    def test_no_model_installed_answers_with_what_is_missing(self):
        from unittest import mock

        from backend_service.mcp_bridge import models as bridge_models

        pid, data = self.create(shots=2)
        with mock.patch.object(bridge_models, "default_model", lambda state, kind: None):
            out = self.post(f"/projects/{pid}/estimate", {"what": "both"})
        self.assertEqual((out["stills"], out["clips"], out["seconds"]), (2, 2, None))
        self.assertTrue(any("image model" in line for line in out["missing"]), out["missing"])
        self.assertTrue(any("video model" in line for line in out["missing"]), out["missing"])
