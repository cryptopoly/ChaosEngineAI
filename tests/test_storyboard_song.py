"""A song on a storyboard project (S-M1): the file check, the analysis, the song ops, the cut plan and lyric captions.

Audio is synthetic: a quiet sine part then a loud noise part, with a click on every beat, so the tempo and the one
section change are known. The analysis tests run without ffmpeg (samples in); the route tests decode a WAV with the
bundled ffmpeg and skip without it.
"""

import json
import os
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.storyboard import captions, cutplan, lyrics, song
from backend_service.storyboard.ffmpeg_bin import ffmpeg_exe
from backend_service.storyboard.ops import OpError
from backend_service.storyboard.timeline import compute_timeline
from tests.storyboard_fixtures import StoreCase, fake_wav

H = {"Authorization": "Bearer app-token"}


def song_samples(seconds=40.0, bpm=120.0, split=20.0, rate=song.SAMPLE_RATE, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    x = 0.1 * np.sin(2 * np.pi * 220 * t)
    loud = t >= split
    x[loud] = 0.25 * rng.standard_normal(int(loud.sum())) + 0.3 * np.sin(2 * np.pi * 440 * t[loud])
    period = 60.0 / bpm
    for k in range(int(seconds / period)):
        i, n = int(k * period * rate), int(0.03 * rate)
        x[i:i + n] += 0.8 * np.exp(-np.arange(n) / (0.005 * rate)) * rng.standard_normal(n)[: x[i:i + n].size]
    return x.astype(np.float32)


def write_wav(path: Path, samples, rate=song.SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    return path


def song_args(path, seconds=40.0, bpm=120.0, sections=None, **extra):
    beats = [round(k * 60.0 / bpm, 3) for k in range(int(seconds * bpm / 60.0))] if bpm else []
    sections = sections or [{"start": 0.0, "end": 20.0, "label": "A", "energy": 0.2},
                            {"start": 20.0, "end": seconds, "label": "B", "energy": 1.0}]
    return {"path": str(path), "title": "Night drive", "seconds": seconds, "bpm": bpm, "beats": beats,
            "sections": sections, "note": song.ROUGH_NOTE, **extra}


class AnalysisTests(unittest.TestCase):
    def test_tempo_beats_and_the_section_change_are_found(self):
        for bpm in (90.0, 128.0):
            found = song.analyse_samples(song_samples(bpm=bpm))
            self.assertAlmostEqual(found["bpm"], bpm, delta=2.0)
            gaps = np.diff(found["beats"])
            self.assertAlmostEqual(float(np.median(gaps)), 60.0 / bpm, delta=0.03)
            self.assertEqual(len(found["sections"]), 2, found["sections"])
            self.assertAlmostEqual(found["sections"][1]["start"], 20.0, delta=1.5)
            self.assertLess(found["sections"][0]["energy"], found["sections"][1]["energy"])
            self.assertTrue(found["rough"])
            self.assertIn("estimated", found["note"])

    def test_parts_that_sound_alike_share_a_letter_and_quiet_ends_are_intro_and_outro(self):
        rate, rng = song.SAMPLE_RATE, np.random.default_rng(1)

        def part(seconds, kind):
            t = np.arange(int(seconds * rate)) / rate
            if kind == "quiet":
                return 0.05 * np.sin(2 * np.pi * 220 * t)
            if kind == "verse":
                return 0.2 * np.sin(2 * np.pi * 330 * t) + 0.05 * rng.standard_normal(t.size)
            return 0.4 * rng.standard_normal(t.size) + 0.3 * np.sin(2 * np.pi * 880 * t)

        x = np.concatenate([part(10, "quiet"), part(30, "verse"), part(25, "chorus"), part(30, "verse"),
                            part(25, "chorus"), part(10, "quiet")]).astype(np.float32)
        labels = [s["label"] for s in song.analyse_samples(x)["sections"]]
        self.assertEqual(labels, ["intro", "A", "B", "A", "B", "outro"])

    def test_a_known_tempo_from_the_library_is_used_and_said(self):
        found = song.analyse_samples(song_samples(bpm=120.0), bpm_hint=121.0)
        self.assertEqual(found["bpm"], 121.0)
        self.assertIn(song.HINT_NOTE, found["note"])

    def test_no_steady_beat_gives_no_beats_and_says_so(self):
        found = song.analyse_samples((0.1 * np.sin(2 * np.pi * 220 * np.arange(20 * song.SAMPLE_RATE) / song.SAMPLE_RATE)).astype(np.float32))
        self.assertEqual(found["beats"], [])
        self.assertIsNone(found["bpm"])
        self.assertIn(song.NO_BEAT_NOTE, found["note"])

    def test_lyrics_drop_structure_tags_and_blank_lines(self):
        self.assertEqual(song.lyric_lines("[verse]\n  Hello   night \n\n(instrumental)\nDrive on\n"), ["Hello night", "Drive on"])


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_audio_is_checked_by_its_bytes_not_its_name(self):
        good = fake_wav(self.dir / "a.wav")
        self.assertEqual(song.check_source(str(good)), good.resolve())
        (self.dir / "b.mp3").write_bytes(b"ID3\x04" + b"\x00" * 64)
        self.assertTrue(song.looks_like_audio(self.dir / "b.mp3"))
        (self.dir / "c.flac").write_bytes(b"fLaC" + b"\x00" * 64)
        self.assertTrue(song.looks_like_audio(self.dir / "c.flac"))
        fake = self.dir / "notes.wav"
        fake.write_text("this is a text file, not a song " * 4)
        with self.assertRaises(song.SongError) as caught:
            song.check_source(str(fake))
        self.assertIn("Not an audio file", str(caught.exception))
        with self.assertRaises(song.SongError):
            song.check_source(str(self.dir / "picture.png"))

    def test_a_library_item_gives_its_file_title_and_lyrics(self):
        audio = fake_wav(self.dir / "1763290000_ab12cd34.wav")
        meta = {"id": "1763290000_ab12cd34", "title": "Night drive", "lyrics": "[verse]\nline one", "bpm": 118}
        with mock.patch("backend_service.music_library.get", return_value=meta), \
                mock.patch("backend_service.music_library.audio_path", return_value=audio):
            path, got = song.library_source("1763290000_ab12cd34")
        self.assertEqual(path, audio.resolve())
        self.assertEqual(got["title"], "Night drive")
        with self.assertRaises(song.SongError):
            song.library_source("../../etc/passwd")


class SongOpTests(StoreCase):
    def attach(self, project, **extra):
        return self.op(project, "set_song", **song_args(fake_wav(self.src / "song.wav"), **extra))

    def test_set_song_copies_the_file_puts_it_on_the_mixer_and_undo_takes_both_off(self):
        project = self.new(shots=1)
        with_song = self.attach(project)
        self.assertTrue(with_song.song.path.startswith("media/song/"))
        self.assertTrue((self.folder(with_song) / with_song.song.path).is_file())
        clip = with_song.clip(with_song.song.clipId)
        self.assertEqual((clip.kind, clip.start, clip.source.path), ("music", 0.0, with_song.song.path))
        undone = self.store.undo(with_song.id, with_song.revision)
        self.assertIsNone(undone.song)
        self.assertEqual(undone.audio, [])
        redone = self.store.redo(undone.id, undone.revision)
        self.assertEqual(redone.song.clipId, clip.id)

    def test_a_new_song_replaces_the_old_one_and_its_clip(self):
        project = self.attach(self.new())
        again = self.attach(project)
        self.assertEqual(len(again.audio), 1)
        self.assertEqual(again.audio[0].id, again.song.clipId)
        kept = self.attach(again, addToMixer=False)
        self.assertIsNone(kept.song.clipId)
        self.assertEqual(len(kept.audio), 1)  # the earlier clip stays as an ordinary music clip

    def test_a_bundle_carries_the_song_file(self):
        from backend_service.storyboard import bundle

        project = self.attach(self.new(), addToMixer=False)
        self.assertEqual(bundle.referenced(project)[project.song.path], project.song.hash)
        path = bundle.export_bundle(self.store, project.id)
        copy = bundle.import_bundle(self.store, path)
        self.assertEqual(copy.song.path, project.song.path)
        self.assertTrue((self.folder(copy) / copy.song.path).is_file())

    def test_edit_song_and_clear_song(self):
        project = self.attach(self.new())
        edited = self.op(project, "edit_song", lyrics=["one", "two"], sections=[
            {"start": 10.0, "end": 40.0, "label": "chorus"}, {"start": 0.0, "end": 10.0, "label": "verse"}])
        self.assertEqual([s.label for s in edited.song.sections], ["verse", "chorus"])
        self.assertEqual(edited.song.lyrics, ["one", "two"])
        with self.assertRaises(OpError):
            self.op(edited, "edit_song", sections=[{"start": 0.0, "end": 99.0, "label": "too long"}])
        with self.assertRaises(OpError):
            self.op(edited, "edit_song")
        cleared = self.op(edited, "clear_song")
        self.assertIsNone(cleared.song)
        self.assertEqual(cleared.audio, [])
        with self.assertRaises(OpError):
            self.op(cleared, "clear_song")


class CutPlanTests(StoreCase):
    def project_with_song(self, shots=0, **extra):
        return self.op(self.new(shots=shots), "set_song", **song_args(fake_wav(self.src / "song.wav"), **extra))

    def test_shots_follow_the_music_and_never_pass_the_clip_length(self):
        project = self.project_with_song()
        plan = cutplan.plan_cuts(project, clip_seconds=4.0)
        lengths = [s["seconds"] for s in plan["shots"]]
        self.assertAlmostEqual(sum(lengths), 40.0, places=2)
        self.assertTrue(all(2.0 <= x <= 4.0 + 1e-6 for x in lengths), lengths)
        cuts = np.cumsum(lengths)[:-1]
        beats = set(project.song.beats)
        self.assertTrue(all(round(float(c), 3) in beats for c in cuts), cuts)  # every cut on a beat
        self.assertIn(20.0, [round(float(c), 3) for c in cuts])  # and one on the section change
        self.assertTrue(all(s["trimOut"] == s["seconds"] for s in plan["shots"]))
        self.assertEqual([s["shots"] for s in plan["sections"]], [5, 5])
        self.assertEqual(plan["shots"][0]["title"], "A 1")
        self.assertEqual(plan["mode"], "replace")

    def test_applying_it_is_one_undoable_op_and_the_film_lines_up_with_the_song(self):
        project = self.project_with_song(shots=2)
        plan = cutplan.plan_cuts(project, clip_seconds=3.0)
        self.assertEqual(plan["shots"][0]["stillPrompt"], "still 0")  # kept from the shots already there
        applied = self.op(project, "apply_plan", mode="replace", shots=plan["shots"])
        self.assertEqual(len(applied.shots), len(plan["shots"]))
        self.assertGreater(len(applied.shots), 12)  # more than a chat-model plan may carry
        self.assertAlmostEqual(compute_timeline(applied).total, 40.0, places=2)
        self.assertEqual(applied.shots[0].trimOut, applied.shots[0].seconds)
        undone = self.store.undo(applied.id, applied.revision)
        self.assertEqual([s.title for s in undone.shots], ["shot 0", "shot 1"])

    def test_a_length_cuts_only_the_songs_first_seconds(self):
        # S-P: "cut it into a 30 s ad" with a 240 s song; before, the plan always covered the whole song, so a chat
        # model built the shots one by one (12 ops, 12 undos).
        project = self.project_with_song()
        plan = cutplan.plan_cuts(project, clip_seconds=4.0, length_seconds=25.0)
        lengths = [s["seconds"] for s in plan["shots"]]
        self.assertAlmostEqual(sum(lengths), 25.0, places=2)
        self.assertTrue(all(2.0 <= x <= 4.0 + 1e-6 for x in lengths), lengths)
        self.assertEqual(plan["totalSeconds"], 25.0)
        self.assertEqual(plan["sections"][-1]["end"], 25.0)
        self.assertIn("first 25 s of 40 s", plan["note"])
        self.assertEqual(cutplan.plan_cuts(project, 4.0, length_seconds=99.0)["totalSeconds"], 40.0)  # longer: all of it
        applied = self.op(project, "apply_plan", mode="replace", shots=plan["shots"])
        self.assertAlmostEqual(compute_timeline(applied).total, 25.0, places=2)

    def test_no_beats_split_evenly_and_no_song_is_refused(self):
        project = self.project_with_song(bpm=None)
        plan = cutplan.plan_cuts(project, clip_seconds=3.0)
        self.assertTrue(all(abs(s["seconds"] - 20.0 / 7) < 0.01 for s in plan["shots"]), plan["shots"])
        self.assertIn("No beats", plan["note"])
        with self.assertRaises(cutplan.CutPlanError):
            cutplan.plan_cuts(self.new(), clip_seconds=4.0)

    def test_the_cap_comes_from_the_video_model(self):
        self.assertEqual(cutplan.clip_seconds_for({"defaultDurationSeconds": 5.0}, 24, None), (5.0, "model"))
        self.assertEqual(cutplan.clip_seconds_for({"defaultDurationSeconds": 5.0}, 24, 3.0), (3.0, "request"))
        self.assertEqual(cutplan.clip_seconds_for({"defaultDurationSeconds": 5.0}, 24, 9.0), (5.0, "model"))
        self.assertEqual(cutplan.clip_seconds_for(None, 24, None), (4.0, "default"))
        self.assertAlmostEqual(cutplan.clip_seconds_for({"defaultDurationSeconds": 20.0}, 24, None)[0], 257 / 24)


class LyricTests(StoreCase):
    def project_with_lyrics(self, lines):
        project = self.op(self.new(), "add_shot", title="whole song", seconds=40)  # captions end with the film
        project = self.op(project, "set_song", **song_args(fake_wav(self.src / "song.wav"), lyrics=lines,
                                                              sections=[{"start": 0.0, "end": 4.0, "label": "intro"},
                                                                        {"start": 4.0, "end": 20.0, "label": "A"},
                                                                        {"start": 20.0, "end": 36.0, "label": "B"},
                                                                        {"start": 36.0, "end": 40.0, "label": "outro"}]))
        return project

    def test_section_timing_shares_the_lines_over_the_sung_parts(self):
        project = self.project_with_lyrics(["one", "two", "three", "four"])
        saved, used, notes = lyrics.make(self.store, project.id, project.revision, "auto",
                                         status=lambda: (False, "Speech-to-text is not installed."))
        self.assertEqual(used, "sections")
        self.assertIn("Speech-to-text is not installed.", notes[0])
        rows = [c for c in saved.captions if c.clipId == saved.song.clipId]
        self.assertEqual([c.text for c in rows], ["one", "two", "three", "four"])
        self.assertEqual([c.start for c in rows], [4.0, 12.0, 20.0, 28.0])
        self.assertTrue(all(c.end - c.start <= lyrics.MAX_LINE_SECONDS for c in rows))
        srt = captions.to_srt(captions.cues(saved, compute_timeline(saved)))
        self.assertIn("00:00:04,000 --> ", srt)
        undone = self.store.undo(saved.id, saved.revision)
        self.assertEqual(undone.captions, [])

    def test_speech_timing_matches_the_words_in_order_and_spaces_the_rest(self):
        project = self.project_with_lyrics(["hello night", "we drive on", "nothing heard here", "under the stars"])
        segments = [{"start": 5.0, "end": 7.0, "text": "Hello, night!"}, {"start": 9.0, "end": 11.0, "text": "we drive on"},
                    {"start": 21.0, "end": 23.0, "text": "under the stars"}]
        saved, used, notes = lyrics.make(self.store, project.id, project.revision, "speech",
                                         transcribe=lambda path, language: {"segments": segments},
                                         status=lambda: (True, None))
        self.assertEqual(used, "speech")
        self.assertIn("placed 3 of 4", notes[0])
        rows = {c.text: c for c in saved.captions}
        self.assertAlmostEqual(rows["hello night"].start, 5.0)
        self.assertAlmostEqual(rows["we drive on"].start, 9.0)
        self.assertAlmostEqual(rows["under the stars"].start, 21.0)
        self.assertTrue(11.0 <= rows["nothing heard here"].start < 21.0)

    def test_speech_asked_for_without_speech_to_text_is_refused_with_a_code(self):
        project = self.project_with_lyrics(["one"])
        with self.assertRaises(lyrics.LyricsError) as caught:
            lyrics.make(self.store, project.id, project.revision, "speech", status=lambda: (False, "not here"))
        self.assertEqual(caught.exception.code, "no-stt")

    def test_no_lyrics_no_song_or_no_clip_is_refused(self):
        bare = self.op(self.new(), "add_shot", title="x", seconds=40)
        with self.assertRaises(lyrics.LyricsError):
            lyrics.make(self.store, bare.id, bare.revision)
        project = self.op(bare, "set_song", **song_args(fake_wav(self.src / "s.wav")))
        with self.assertRaises(lyrics.LyricsError):
            lyrics.make(self.store, project.id, project.revision)
        project = self.op(project, "set_song", **song_args(fake_wav(self.src / "t.wav"), lyrics=["x"], addToMixer=False))
        project = self.op(project, "clear_song", removeClip=True)
        project = self.op(project, "set_song", **song_args(fake_wav(self.src / "u.wav"), lyrics=["x"], addToMixer=False))
        with self.assertRaises(lyrics.LyricsError) as caught:
            lyrics.make(self.store, project.id, project.revision, status=lambda: (False, None))
        self.assertIn("mixer", str(caught.exception))


class SongRouteTests(unittest.TestCase):
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

    def test_attach_a_file_then_cut_plan_then_lyric_captions_and_srt(self):
        data = self.post("/projects", {"name": "Video"})
        pid, revision = data["project"]["id"], data["project"]["revision"]
        wav = write_wav(self.base / "in" / "song.wav", song_samples())
        attached = self.post(f"/projects/{pid}/song", {"revision": revision, "path": str(wav)})
        analysis = attached["analysis"]
        self.assertAlmostEqual(analysis["bpm"], 120.0, delta=2.0)
        self.assertEqual(len(analysis["sections"]), 2)
        self.assertTrue(analysis["rough"])
        stored = attached["project"]["song"]
        self.assertEqual(stored["title"], "song")
        self.assertAlmostEqual(stored["seconds"], 40.0, delta=0.1)
        self.assertEqual(attached["project"]["audio"][0]["id"], stored["clipId"])
        # S-P: the Plan tab's Length (filmSeconds) limits the cut; an asked length wins over it.
        planned = self.post(f"/projects/{pid}/ops", {"revision": attached["project"]["revision"], "op": "set_plan",
                                                     "args": {"filmSeconds": 20}})
        short = self.post(f"/projects/{pid}/cut-plan", {"maxShotSeconds": 4})
        self.assertAlmostEqual(sum(s["seconds"] for s in short["shots"]), 20.0, places=1)
        asked = self.post(f"/projects/{pid}/cut-plan", {"maxShotSeconds": 4, "lengthSeconds": 12})
        self.assertAlmostEqual(sum(s["seconds"] for s in asked["shots"]), 12.0, places=1)
        attached["project"] = self.post(f"/projects/{pid}/ops", {"revision": planned["project"]["revision"], "op": "set_plan",
                                                                 "args": {"clearFilmSeconds": True}})["project"]
        plan = self.post(f"/projects/{pid}/cut-plan", {"maxShotSeconds": 4})
        self.assertIn(plan["clipSecondsSource"], ("request", "model"))  # the installed video model may make shorter clips
        self.assertTrue(all(s["seconds"] <= plan["clipSeconds"] + 1e-6 for s in plan["shots"]))
        self.assertAlmostEqual(sum(s["seconds"] for s in plan["shots"]), stored["seconds"], places=1)
        revision = attached["project"]["revision"]
        applied = self.post(f"/projects/{pid}/ops", {"revision": revision, "op": "apply_plan",
                                                     "args": {"mode": "replace", "shots": plan["shots"]}})
        revision = applied["project"]["revision"]
        clip = self.base / "in" / "clip.mp4"  # a small real clip, so the shots are ready and the .srt has a film
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-nostdin", "-f", "lavfi", "-i",
                        "testsrc2=s=64x64:r=8:d=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
        for shot in applied["project"]["shots"]:
            revision = self.post(f"/projects/{pid}/ops", {"revision": revision, "op": "import_take",
                                                          "args": {"shotId": shot["id"], "clip": str(clip)}})["project"]["revision"]
        lyric = self.post(f"/projects/{pid}/ops", {"revision": revision, "op": "edit_song",
                                                   "args": {"lyrics": ["first line", "second line"]}})
        with mock.patch.object(lyrics, "speech_status", return_value=(False, "Speech-to-text is not installed.")):
            self.post(f"/projects/{pid}/lyrics/captions", {"revision": lyric["project"]["revision"], "timing": "speech"}, code=409)
            made = self.post(f"/projects/{pid}/lyrics/captions", {"revision": lyric["project"]["revision"]})
        self.assertEqual(made["timing"], "sections")
        self.assertEqual(len(made["project"]["captions"]), 2)
        exported = self.post(f"/projects/{pid}/captions/export", {})
        srt = (self.base / "storyboards" / pid / exported["file"]).read_text()
        self.assertIn("first line", srt)

    def test_a_text_file_named_like_a_song_is_refused(self):
        data = self.post("/projects", {"name": "Video"})
        fake = self.base / "fake.mp3"
        fake.write_text("not a song at all " * 10)
        out = self.post(f"/projects/{data['project']['id']}/song", {"revision": 1, "path": str(fake)}, code=400)
        self.assertIn("Not an audio file", json.dumps(out))
        self.post(f"/projects/{data['project']['id']}/song", {"revision": 1}, code=422)
        self.post(f"/projects/{data['project']['id']}/cut-plan", {}, code=400)  # no song yet

    def test_speech_status_route(self):
        with mock.patch.object(lyrics, "speech_status", return_value=(False, "why")):
            response = self.client.get("/api/storyboard/lyrics/speech", headers=H)
        self.assertEqual(response.json(), {"ready": False, "reason": "why"})


if __name__ == "__main__":
    unittest.main()
