"""Song to lip-sync shots (FU-371): the plan per shot (lead-in, 3 to 8 s rule, stem silence), the one-op attach,
and the route that slices a stem or the song with ffmpeg. Synthetic audio; the route tests skip without ffmpeg."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.storyboard import song, song_vocals as sv
from backend_service.storyboard.ffmpeg_bin import ffmpeg_exe
from backend_service.storyboard.timeline import compute_timeline
from tests.storyboard_fixtures import StoreCase, fake_wav
from tests.test_storyboard_song import song_args, song_samples, write_wav

H = {"Authorization": "Bearer app-token"}


def _stem(seconds: float, sung: list[tuple[float, float]], rate: int = song.SAMPLE_RATE):
    """Silence with a loud tone where someone sings."""
    t = np.arange(int(seconds * rate)) / rate
    x = np.zeros_like(t)
    for a, b in sung:
        x[(t >= a) & (t < b)] = 0.3 * np.sin(2 * np.pi * 330 * t[(t >= a) & (t < b)])
    return x.astype(np.float32)


class PlanTests(StoreCase):
    def project(self, lengths: list[float]):
        project = self.new(shots=len(lengths))
        for shot, seconds in zip(project.shots, lengths):
            project = self.op(project, "edit_shot", shotId=shot.id, seconds=seconds, trimOut=seconds)
        return self.op(project, "set_song", **song_args(fake_wav(self.src / "song.wav")))

    def test_each_shot_gets_its_window_with_a_lead_in_except_at_the_start(self):
        project = self.project([4.0, 4.5, 3.0])
        plan = sv.plan_vocals(project)
        self.assertEqual([i["lead"] for i in plan["items"]], [0.0, 0.4, 0.4])
        self.assertEqual([i["songStart"] for i in plan["items"]], [0.0, 3.6, 8.1])
        self.assertEqual([i["length"] for i in plan["items"]], [4.0, 4.5, 3.0])
        self.assertEqual(plan["skipped"], [])

    def test_too_short_too_long_and_past_the_song_are_named(self):
        project = self.project([2.0, 9.0, 4.0])
        plan = sv.plan_vocals(project)
        self.assertEqual([i["number"] for i in plan["items"]], [3])
        reasons = [s["reason"] for s in plan["skipped"]]
        self.assertIn("Shot 1 lasts 2.0 s", reasons[0])
        self.assertIn("Shot 2 lasts 9.0 s", reasons[1])

    def test_silent_stem_windows_get_no_vocal(self):
        project = self.project([4.0, 4.0, 4.0])
        voiced = sv.voiced_frames(_stem(40.0, [(4.2, 7.5)]), song.SAMPLE_RATE)
        plan = sv.plan_vocals(project, voiced)
        self.assertEqual([i["number"] for i in plan["items"]], [2])
        self.assertEqual([s["number"] for s in plan["skipped"]], [1, 3])
        self.assertIn("no singing", plan["skipped"][0]["reason"])

    def test_the_song_clip_position_moves_the_windows(self):
        project = self.project([4.0, 4.0])
        clip = project.clip(project.song.clipId)
        project = self.op(project, "edit_audio", clipId=clip.id, trimIn=10.0)
        self.assertEqual([i["songStart"] for i in sv.plan_vocals(project)["items"]], [9.6, 13.6])

    def test_attaching_is_one_op_and_the_film_keeps_its_length(self):
        project = self.project([4.0, 4.0])
        before = compute_timeline(project).total
        items = [{"shotId": s.id, "audio": str(fake_wav(self.src / f"{s.id}.wav")), "lead": lead, "length": 4.0}
                 for s, lead in zip(project.shots, [0.0, 0.4])]
        done = self.op(project, "set_shot_vocals", items=items)
        second = done.shots[1]
        self.assertTrue(second.vocalClip.startswith(f"media/{second.id}/vocal-"))
        self.assertEqual((second.seconds, second.trimIn, second.trimOut, second.vocalSinging, second.vocalStart),
                         (4.4, 0.4, 4.4, True, 0.0))
        self.assertAlmostEqual(compute_timeline(done).total, before)
        self.assertEqual(sv.plan_vocals(done)["items"][1]["songStart"], 3.6)  # applying again gives the same windows
        undone = self.store.undo(done.id, done.revision)
        self.assertIsNone(undone.shots[1].vocalClip)


class RouteTests(unittest.TestCase):
    def setUp(self):
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
        response = self.client.post(f"/api/storyboard{path}", json=body, headers=H)
        self.assertEqual(response.status_code, code, response.text)
        return response.json()

    def project_with_song(self):
        data = self.post("/projects", {"name": "Video"})
        pid = data["project"]["id"]
        for _ in range(3):
            data = self.post(f"/projects/{pid}/ops", {"revision": data["project"]["revision"], "op": "add_shot",
                                                     "args": {"seconds": 4.0, "trimOut": 4.0}})
        wav = write_wav(self.base / "in" / "song.wav", song_samples(seconds=20.0, split=10.0))
        data = self.post(f"/projects/{pid}/song", {"revision": data["project"]["revision"], "path": str(wav)})
        return pid, data

    def test_the_song_is_sliced_when_there_is_no_stem(self):
        pid, data = self.project_with_song()
        out = self.post(f"/projects/{pid}/song/vocals", {"revision": data["project"]["revision"]})
        self.assertEqual((out["vocals"]["source"], out["vocals"]["attached"]), ("song", 3))
        shots = out["project"]["shots"]
        self.assertEqual([s["trimIn"] for s in shots], [0.0, 0.4, 0.4])
        self.assertTrue(all(s["vocalSinging"] for s in shots))
        self.assertAlmostEqual(out["timeline"]["total"], 12.0, places=2)
        folder = self.base / "storyboards" / pid
        from backend_service.storyboard.media import probe_seconds

        self.assertAlmostEqual(probe_seconds(folder / shots[1]["vocalClip"]), 4.4, places=1)

    def test_a_stem_skips_the_silent_shots(self):
        pid, data = self.project_with_song()
        stem = write_wav(self.base / "in" / "vocals.wav", _stem(20.0, [(4.5, 7.8)]))
        out = self.post(f"/projects/{pid}/song/vocals", {"revision": data["project"]["revision"], "stemPath": str(stem)})
        self.assertEqual((out["vocals"]["source"], out["vocals"]["attached"]), ("stem", 1))
        self.assertEqual([s["number"] for s in out["vocals"]["skipped"]], [1, 3])
        self.assertIsNotNone(out["project"]["shots"][1]["vocalClip"])
        self.assertIsNone(out["project"]["shots"][0]["vocalClip"])

    def test_no_song_is_refused(self):
        data = self.post("/projects", {"name": "Video"})
        self.post(f"/projects/{data['project']['id']}/song/vocals", {"revision": data["project"]["revision"]}, code=400)


if __name__ == "__main__":
    unittest.main()
