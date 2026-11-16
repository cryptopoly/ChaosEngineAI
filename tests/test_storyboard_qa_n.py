"""Regression tests for the S-N QA pass (v0.10.0)."""
from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import backend_service.app  # noqa: F401  (loads the routes package, so routes.images can be patched)
from backend_service.storyboard.models import Project, Shot


class PlaceholderStillTests(unittest.TestCase):
    def test_a_placeholder_picture_fails_the_step_in_plain_words(self):
        """When the image engine cannot run, the runtime returns its placeholder SVG; the batch step said
        "'.svg' is not a image file: img-....svg". It now names the model and the runtime's reason."""
        from backend_service.storyboard import generators

        folder = Path(tempfile.mkdtemp())
        project = Project(id="0123456789abcdef", name="Film", shots=[Shot(id="s_1", stillPrompt="a harbour")])
        artifact = {"imagePath": str(folder / "img-1.svg"), "runtimeNote": "mflux is not installed."}
        variant = {"id": "iris", "name": "Iris-3B Turbo"}
        with mock.patch.object(generators, "_variant", return_value=variant), \
                mock.patch("backend_service.mcp_bridge.tools._recommended", return_value={}), \
                mock.patch("backend_service.storyboard.style.apply_to_still", return_value=([], "a harbour", "none")), \
                mock.patch("backend_service.helpers.job_queue.HEAVY_JOBS.slot", return_value=nullcontext()), \
                mock.patch("backend_service.routes.images._generate_image_locked", return_value={"artifacts": [artifact]}):
            make_still = generators.studio_generators(SimpleNamespace(), "b1", folder)["make_still"]
            with self.assertRaises(RuntimeError) as caught:
                make_still(project.shots[0], project)
        self.assertEqual(str(caught.exception), "Iris-3B Turbo could not run here, so no still was made. mflux is not installed.")


class DraftTimeLeftTests(unittest.TestCase):
    """A draft batch on a project whose still model is slow showed "about 32 min left" (the project's model at full
    size) while each Iris draft took 9 s: the time left now prices a draft step with the draft model and size."""

    def setUp(self):
        from tests.test_storyboard_passes import PassRouteTests

        self.case = PassRouteTests(methodName="_variants")
        self.case.setUp()
        self.addCleanup(self.case.tearDown)

    def test_first_draft_step_is_priced_with_the_fast_model(self):
        from backend_service.mcp_bridge import models as bridge_models
        from backend_service.routes import storyboard as routes
        from tests.test_storyboard_passes import FAST, SLOW

        case = self.case
        pid, data = case.create(shots=2)
        data = case.op(pid, data["project"]["revision"], "edit_project", {"defaults": {"imageModel": "slow"}})
        request = SimpleNamespace(app=case.client.app)
        with mock.patch.object(routes, "_project_variant", side_effect=case._variants), \
                mock.patch.object(bridge_models, "default_model", lambda state, kind: {"id": "fast"}):
            batch = routes.build_batch(request, pid, "still", None, False, draft=True)
            batch.start(data["project"]["revision"])
            shot_id = data["project"]["shots"][0]["id"]
            seconds, _ = batch._catalog_price(shot_id, "still")
        self.assertEqual(batch._variants[0]["id"], FAST["id"])
        self.assertLess(seconds, FAST["estimatedGenerationSeconds"])  # the fast model at the smaller draft size
        self.assertLess(seconds, SLOW["estimatedGenerationSeconds"] / 10)


class LyricsPastTheFilmTests(unittest.TestCase):
    """The sample film is 7 s; a 64 s song's lyrics were timed from 7.4 s on. Every line fell after the film's end:
    the panel said the lines were captions, and the .srt export answered only "There are no captions to export"."""

    def setUp(self):
        import os

        from fastapi.testclient import TestClient

        from backend_service.app import create_app
        from backend_service.storyboard.ffmpeg_bin import ffmpeg_exe

        try:
            ffmpeg_exe()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        env = mock.patch.dict(os.environ, {"CHAOSENGINE_STORYBOARDS_DIR": str(self.base / "storyboards")})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(create_app(api_token="app-token"))

    def post(self, path, body, code=200):
        from tests.test_storyboard_song import H

        response = self.client.post(f"/api/storyboard{path}", json=body, headers=H)
        self.assertEqual(response.status_code, code, response.text)
        return response.json()

    def test_lines_past_the_films_end_are_named_and_the_export_says_why(self):
        from backend_service.storyboard import lyrics
        from tests.test_storyboard_song import song_samples, write_wav

        data = self.post("/projects", {"name": "Short film"})
        pid = data["project"]["id"]
        wav = write_wav(self.base / "in" / "song.wav", song_samples())
        attached = self.post(f"/projects/{pid}/song", {"revision": data["project"]["revision"], "path": str(wav)})
        lyric = self.post(f"/projects/{pid}/ops", {"revision": attached["project"]["revision"], "op": "edit_song",
                                                   "args": {"lyrics": ["first line", "second line"]}})
        with mock.patch.object(lyrics, "speech_status", return_value=(False, None)):
            made = self.post(f"/projects/{pid}/lyrics/captions", {"revision": lyric["project"]["revision"]})
        self.assertEqual(len(made["project"]["captions"]), 2)
        self.assertTrue(any("2 of 2 lines fall after the film's current end (0.0 s)" in n for n in made["notes"]), made["notes"])
        refused = self.post(f"/projects/{pid}/captions/export", {}, code=400)
        self.assertIn("The captions start after the film ends (0.0 s)", refused["detail"])


if __name__ == "__main__":
    unittest.main()
