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

    def test_room_after_the_phrase_is_made_then_cut_off_by_the_out_point(self):
        # FU-609: the slice runs 0.5 s past the shot; trimOut cuts it off, so the film keeps its length.
        project = self.project([4.0, 4.0])
        before = compute_timeline(project).total
        items = [{"shotId": s.id, "audio": str(fake_wav(self.src / f"{s.id}.wav")), "lead": lead, "length": 4.0,
                  "tail": 0.5} for s, lead in zip(project.shots, [0.0, 0.4])]
        done = self.op(project, "set_shot_vocals", items=items)
        second = done.shots[1]
        self.assertEqual((second.seconds, second.trimIn, second.trimOut), (4.9, 0.4, 4.4))
        self.assertAlmostEqual(compute_timeline(done).total, before)


class PhraseTests(StoreCase):
    """FU-607 (cuts on phrase starts) and FU-609 (room after the phrase) from a synthetic stem."""

    def project(self, lengths: list[float]):
        return PlanTests.project(self, lengths)

    def voiced(self, sung):
        return sv.voiced_frames(_stem(40.0, sung), song.SAMPLE_RATE)

    def test_phrase_starts_follow_silence(self):
        voiced = self.voiced([(0.0, 3.3), (4.6, 8.0), (8.2, 9.0), (8.5, 11.5)])
        self.assertEqual(sv.phrase_starts(voiced), [0.0, 4.6])  # 8.2 follows only 0.2 s of silence

    def test_cuts_move_to_the_nearest_phrase_start_and_the_film_keeps_its_length(self):
        project = self.project([4.0, 4.0, 4.0])
        plan = sv.plan_vocals(project, self.voiced([(0.2, 3.3), (4.6, 8.0), (8.5, 11.5)]))
        self.assertEqual(plan["snapped"], 2)
        self.assertEqual([round(i["songStart"] + i["lead"], 3) for i in plan["items"]], [0.0, 4.6, 8.5])
        self.assertEqual([i["length"] for i in plan["items"]], [4.6, 3.9, 3.5])
        self.assertAlmostEqual(sum(i["length"] for i in plan["items"]), 12.0)

    def test_a_cut_stays_when_moving_it_would_leave_3_to_5_s(self):
        project = self.project([4.0, 3.2])
        plan = sv.plan_vocals(project, self.voiced([(0.2, 3.3), (4.6, 7.0)]))  # 4.6 would leave the second 2.6 s
        self.assertEqual(plan["snapped"], 0)
        self.assertEqual([i["length"] for i in plan["items"]], [4.0, 3.2])

    def test_a_cut_next_to_a_shot_without_singing_stays(self):
        project = self.project([4.0, 4.0, 4.0])
        plan = sv.plan_vocals(project, self.voiced([(0.2, 3.3), (4.6, 7.5)]))
        self.assertEqual([s["number"] for s in plan["skipped"]], [3])
        self.assertEqual(plan["snapped"], 1)  # 4.0 -> 4.6; the cut at 8.0 borders the silent shot and stays
        self.assertEqual(plan["items"][1]["length"], 3.4)

    def test_a_shot_ending_on_its_phrase_gets_room_after_it(self):
        project = self.project([4.0, 4.0])
        plan = sv.plan_vocals(project, self.voiced([(0.1, 8.0)]))  # one long phrase: no cut moves
        first, second = plan["items"]
        self.assertEqual((first["tail"], first["natural"]), (0.5, False))  # the stem sings on: silence
        self.assertEqual((second["tail"], second["natural"]), (0.5, True))  # the stem is silent after 8 s: itself

    def test_silence_already_after_the_phrase_needs_less_or_no_room(self):
        project = self.project([4.0, 4.0])
        plan = sv.plan_vocals(project, self.voiced([(0.1, 3.8), (4.05, 7.2)]))
        first, second = plan["items"]
        self.assertAlmostEqual(first["tail"], 0.3, delta=0.051)  # 0.2 s of silence after it (to the 50 ms frame)
        self.assertEqual(second["tail"], 0.0)  # 0.8 s of silence after it

    def test_without_a_stem_every_slice_gets_the_room_as_silence(self):
        plan = sv.plan_vocals(self.project([4.0, 4.0]))
        self.assertEqual([(i["tail"], i["natural"]) for i in plan["items"]], [(0.5, False), (0.5, False)])
        self.assertEqual(plan["snapped"], 0)

    def test_the_room_never_takes_a_shot_past_8_s(self):
        self.assertEqual(sv.room_after(None, 0.0, 7.5, 8.0 - 7.5 - 0.4), (0.1, False))
        self.assertEqual(sv.room_after(None, 0.0, 7.6, 0.0), (0.0, False))


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

        self.assertAlmostEqual(probe_seconds(folder / shots[1]["vocalClip"]), 4.9, places=1)  # lead + shot + room
        self.assertEqual([(s["seconds"], s["trimOut"]) for s in shots], [(4.5, 4.0), (4.9, 4.4), (4.9, 4.4)])

    def test_a_stem_skips_the_silent_shots(self):
        pid, data = self.project_with_song()
        stem = write_wav(self.base / "in" / "vocals.wav", _stem(20.0, [(4.5, 7.8)]))
        out = self.post(f"/projects/{pid}/song/vocals", {"revision": data["project"]["revision"], "stemPath": str(stem)})
        self.assertEqual((out["vocals"]["source"], out["vocals"]["attached"]), ("stem", 1))
        self.assertEqual([s["number"] for s in out["vocals"]["skipped"]], [1, 3])
        self.assertIsNotNone(out["project"]["shots"][1]["vocalClip"])
        self.assertIsNone(out["project"]["shots"][0]["vocalClip"])

    def test_a_stem_moves_cuts_to_phrases_and_pads_a_phrase_that_runs_on(self):
        pid, data = self.project_with_song()
        stem = write_wav(self.base / "in" / "vocals.wav", _stem(20.0, [(0.2, 3.3), (4.6, 12.0)]))
        out = self.post(f"/projects/{pid}/song/vocals", {"revision": data["project"]["revision"], "stemPath": str(stem)})
        self.assertEqual((out["vocals"]["attached"], out["vocals"]["snapped"]), (3, 1))
        shots = out["project"]["shots"]
        self.assertEqual([(s["trimIn"], s["trimOut"], s["seconds"]) for s in shots],
                         [(0.0, 4.6, 4.6), (0.4, 3.8, 4.3), (0.4, 4.4, 4.9)])  # the 1st ends 1.3 s after its phrase
        self.assertAlmostEqual(out["timeline"]["total"], 12.0, places=2)
        from backend_service.storyboard.media import probe_seconds

        slice_ = self.base / "storyboards" / pid / shots[1]["vocalClip"]
        self.assertAlmostEqual(probe_seconds(slice_), 4.3, places=1)
        import soundfile as sf

        audio, rate = sf.read(str(slice_))
        self.assertGreater(float(np.abs(audio[int(3.5 * rate): int(3.75 * rate)]).max()), 0.1)
        self.assertLess(float(np.abs(audio[int(3.85 * rate):]).max()), 1e-3)  # the stem sings on; the room is silence

    def test_no_song_is_refused(self):
        data = self.post("/projects", {"name": "Video"})
        self.post(f"/projects/{data['project']['id']}/song/vocals", {"revision": data["project"]["revision"]}, code=400)


if __name__ == "__main__":
    unittest.main()


class SongSoundTests(StoreCase):
    """A film cut to a song plays the song: clips made under it start with their own sound muted."""

    def test_shots_made_under_a_song_start_muted_and_others_do_not(self):
        project = self.new(shots=1)
        self.assertEqual(project.shots[0].nativeSoundDb, 0.0)
        project = self.op(project, "set_song", **song_args(fake_wav(self.src / "song.wav")))
        project = self.op(project, "add_shot", title="later")
        project = self.op(project, "add_shot", title="loud", nativeSoundDb=-6.0)
        self.assertEqual([s.nativeSoundDb for s in project.shots], [0.0, -60.0, -6.0])

    def test_a_cut_plan_applied_under_a_song_makes_muted_shots(self):
        project = self.op(self.new(shots=0), "set_song", **song_args(fake_wav(self.src / "song.wav")))
        project = self.op(project, "apply_plan", shots=[{"title": "a", "seconds": 4.0}, {"title": "b", "seconds": 4.0}])
        self.assertEqual({s.nativeSoundDb for s in project.shots}, {-60.0})

    def test_a_lip_sync_slice_mutes_the_clips_own_copy_of_the_vocal(self):
        project = self.op(self.new(shots=1), "set_song", **song_args(fake_wav(self.src / "song.wav")))
        project = self.op(project, "set_shot_vocals", items=[{"shotId": project.shots[0].id,
                          "audio": str(fake_wav(self.src / "slice.wav")), "lead": 0.0, "length": 4.0, "tail": 0.5,
                          "singing": True}])
        self.assertEqual(project.shots[0].nativeSoundDb, -60.0)
