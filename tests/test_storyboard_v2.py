"""Storyboard v2: captions, resumable renders, OpenTimelineIO export, lane stems, keep-awake and the banner."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.storyboard import captions, longrun, otio_export
from backend_service.storyboard.ffmpeg_bin import ffmpeg_exe
from backend_service.storyboard.models import Project, SCHEMA_VERSION
from backend_service.storyboard.store import migrate
from backend_service.storyboard.timeline import compute_timeline

H = {"Authorization": "Bearer app-token"}
REPO = Path(__file__).resolve().parents[1]


def _ffmpeg(*args):
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-nostdin", *args], check=True, capture_output=True)


class Base(unittest.TestCase):
    shots = 3

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
        self.voice = self.base / "voice.wav"
        _ffmpeg("-f", "lavfi", "-i", "sine=f=800:d=3.0", str(self.voice))
        self.music = self.base / "music.wav"
        _ffmpeg("-f", "lavfi", "-i", "sine=f=220:d=4.0", str(self.music))
        self.pid, self.data = self._film()

    def _op(self, op, args):
        response = self.client.post(f"/api/storyboard/projects/{self.pid}/ops", headers=H,
                                    json={"revision": self.data["project"]["revision"], "op": op, "args": args})
        self.assertEqual(response.status_code, 200, response.text)
        self.data = response.json()
        return self.data

    def _film(self):
        data = self.client.post("/api/storyboard/projects", json={"name": "Caption film"}, headers=H).json()
        self.pid = data["project"]["id"]
        self.data = data
        for i in range(self.shots):
            self._op("add_shot", {"title": f"s{i}", "seconds": 1.5})
            clip = self.base / f"c{i}.mp4"
            _ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=1.5", "-f", "lavfi", "-i", "sine=f=440:d=1.5",
                    "-vf", f"hue=h={i * 90}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clip))
            self._op("import_take", {"shotId": self.data["project"]["shots"][i]["id"], "clip": str(clip)})
        return self.pid, self.data

    def project(self) -> Project:
        from backend_service.storyboard.store import default_store

        return default_store().load(self.pid)

    def render(self, preview=False, timeout=90):
        started = self.client.post(f"/api/storyboard/projects/{self.pid}/render", headers=H,
                                   json={"revision": self.project().revision, "preview": preview, "notify": False})
        self.assertEqual(started.status_code, 200, started.text)
        deadline = time.time() + timeout
        state = {}
        while time.time() < deadline:
            state = self.client.get(f"/api/storyboard/projects/{self.pid}/render", headers=H).json()["render"] or {}
            if state.get("state") in ("done", "failed", "cancelled", "interrupted"):
                return state
            time.sleep(0.2)
        self.fail(f"render did not finish: {state}")

    def exports(self):
        return self.base / "storyboards" / self.pid / "exports"


def _fake_stt(path, language):
    return {"segments": [{"start": 0.1, "end": 1.4, "text": "Hello from the lighthouse."},
                         {"start": 1.5, "end": 2.8, "text": "{\\fs99}Second --> line"}], "note": None}


class CaptionLogicTests(unittest.TestCase):
    def test_srt_stamps_and_text_safety(self):
        items = [captions.Cue("a", 0.0, 1.5, "Hi {\\b1}there --> now"), captions.Cue("b", 3661.25, 3662.0, "Two\n\n\nlines")]
        srt = captions.to_srt(items)
        self.assertIn("1\n00:00:00,000 --> 00:00:01,500\nHi (\\b1)there -> now\n", srt)
        self.assertIn("2\n01:01:01,250 --> 01:01:02,000\nTwo\nlines\n", srt)

    def test_long_segment_is_split_in_proportion(self):
        words = " ".join(["word"] * 40)
        parts = captions.split_segment(0.0, 10.0, words)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p["text"]) <= captions.MAX_CUE_CHARS for p in parts))
        self.assertAlmostEqual(parts[0]["start"], 0.0)
        self.assertAlmostEqual(parts[-1]["end"], 10.0)
        for a, b in zip(parts, parts[1:]):
            self.assertAlmostEqual(a["end"], b["start"], places=2)

    def test_tiny_and_empty_segments(self):
        out = captions.cues_from_segments([{"start": 1.0, "end": 1.01, "text": "ok"}, {"start": 2, "end": 3, "text": "  "},
                                           {"start": "x", "end": 3, "text": "bad"}], limit=None)
        self.assertEqual(len(out), 1)
        self.assertGreaterEqual(out[0]["end"] - out[0]["start"], captions.MIN_CUE_SECONDS)

    def test_version_1_projects_migrate(self):
        data = migrate({"schemaVersion": 1, "revision": 1, "id": "0123456789abcdef", "name": "old"})
        self.assertEqual((data["schemaVersion"], data["captions"]), (SCHEMA_VERSION, []))
        project = Project.model_validate(data)
        self.assertFalse(project.export.captionsBurn)


class CaptionOpsTests(Base):
    def _voice(self, anchor=True):
        first = self.data["project"]["shots"][0]["id"]
        args = {"kind": "voice", "path": str(self.voice)}
        args["anchor"] = {"shotId": first, "offset": 0.5}
        self._op("add_audio", args)
        return self.data["project"]["audio"][-1]["id"]

    def test_generate_places_captions_on_the_timeline_and_follows_the_clip(self):
        clip = self._voice()
        store = __import__("backend_service.storyboard.store", fromlist=["x"]).default_store()
        project, notes = captions.generate(store, self.pid, self.project().revision, None, "", transcribe=_fake_stt)
        self.assertEqual(len(project.captions), 2)
        items = captions.cues(project, compute_timeline(project, ready_only=True))
        self.assertAlmostEqual(items[0].start, 0.6, places=2)  # anchor offset 0.5 + 0.1
        # Move the clip: the captions follow it.
        self.data = {"project": {"revision": project.revision}}
        self._op("edit_audio", {"clipId": clip, "start": 2.0})
        moved = captions.cues(self.project(), compute_timeline(self.project(), ready_only=True))
        self.assertAlmostEqual(moved[0].start, 2.1, places=2)
        # Trim the first second off: the caption wholly inside it disappears, the other is cut.
        self._op("edit_audio", {"clipId": clip, "trimIn": 1.45})
        trimmed = captions.cues(self.project(), compute_timeline(self.project(), ready_only=True))
        self.assertEqual(len(trimmed), 1)
        self.assertEqual(notes, [])

    def test_edit_undo_and_remove_with_the_clip(self):
        clip = self._voice()
        self._op("set_captions", {"clipId": clip, "cues": [{"start": 0.2, "end": 1.0, "text": "one"}]})
        cid = self.data["project"]["captions"][0]["id"]
        self._op("edit_caption", {"captionId": cid, "text": "uno"})
        self.assertEqual(self.data["project"]["captions"][0]["text"], "uno")
        undone = self.client.post(f"/api/storyboard/projects/{self.pid}/undo", headers=H,
                                  json={"revision": self.data["project"]["revision"]}).json()
        self.assertEqual(undone["project"]["captions"][0]["text"], "one")
        self.data = undone
        self._op("add_caption", {"start": 0.0, "end": 0.5, "text": "manual"})
        self._op("remove_audio", {"clipId": clip})
        self.assertEqual([c["text"] for c in self.data["project"]["captions"]], ["manual"])

    def test_bad_captions_are_refused(self):
        response = self.client.post(f"/api/storyboard/projects/{self.pid}/ops", headers=H,
                                    json={"revision": self.data["project"]["revision"], "op": "add_caption",
                                          "args": {"start": 2.0, "end": 1.0, "text": "backwards"}})
        self.assertEqual(response.status_code, 400)

    def test_generate_route_refuses_without_voice(self):
        response = self.client.post(f"/api/storyboard/projects/{self.pid}/captions/generate", headers=H,
                                    json={"revision": self.data["project"]["revision"]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("voice", response.json()["detail"])

    def test_generate_route_and_srt_export(self):
        self._voice()
        with mock.patch("backend_service.storyboard.captions.transcribe_with_studio", _fake_stt):
            response = self.client.post(f"/api/storyboard/projects/{self.pid}/captions/generate", headers=H,
                                        json={"revision": self.data["project"]["revision"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["project"]["captions"]), 2)
        out = self.client.post(f"/api/storyboard/projects/{self.pid}/captions/export", headers=H).json()
        text = (self.exports() / Path(out["file"]).name).read_text()
        self.assertIn("-->", text)
        served = self.client.get(f"/api/storyboard/projects/{self.pid}/files", params={"path": out["file"]}, headers=H)
        self.assertEqual(served.status_code, 200)


class RenderV2Tests(Base):
    def _captioned(self):
        first = self.data["project"]["shots"][0]["id"]
        self._op("add_audio", {"kind": "voice", "path": str(self.voice), "anchor": {"shotId": first, "offset": 0.0}})
        self._op("add_audio", {"kind": "music", "path": str(self.music), "start": 0.0, "duck": True})
        clip = self.data["project"]["audio"][0]["id"]
        self._op("set_captions", {"clipId": clip, "cues": [{"start": 0.2, "end": 1.6, "text": "Burned into the picture"}]})

    def test_export_with_burn_in_sidecar_and_stems(self):
        self._captioned()
        self._op("edit_project", {"export": {"captionsBurn": True, "captionsSrt": True, "stems": True}})
        state = self.render()
        self.assertEqual(state["state"], "done", state)
        names = sorted(p.name for p in self.exports().iterdir())
        film = next(n for n in names if n.endswith(".mp4"))
        stem = film[:-4]
        self.assertIn(f"{stem}.srt", names)
        for lane in ("shot-sound", "voice", "music"):
            self.assertIn(f"{stem}-stem-{lane}.wav", names)
        self.assertNotIn(f"{stem}-stem-sfx.wav", names)
        self.assertEqual(sorted(state["extras"]), sorted(f"exports/{n}" for n in names if n != film))
        # Burn-in changed the picture: a frame at 1 s differs from the same film without captions.
        with_caps = self.base / "with.png"
        _ffmpeg("-ss", "1.0", "-i", str(self.exports() / film), "-frames:v", "1", str(with_caps))
        self._op("edit_project", {"export": {"captionsBurn": False, "captionsSrt": False, "stems": False}})
        state2 = self.render()
        self.assertEqual(state2["state"], "done", state2)
        plain = self.base / "plain.png"
        _ffmpeg("-ss", "1.0", "-i", str(self.exports() / Path(state2["file"]).name), "-frames:v", "1", str(plain))
        self.assertNotEqual(with_caps.read_bytes(), plain.read_bytes())
        # Stems are the full film length.
        info = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(self.exports() / f"{stem}-stem-voice.wav")],
                              capture_output=True, text=True).stderr
        self.assertIn("Duration: 00:00:04", info)  # 3 x 1.5 s shots

    def test_missing_subtitles_filter_falls_back_to_srt_only(self):
        self._captioned()
        self._op("edit_project", {"export": {"captionsBurn": True, "captionsSrt": True}})
        from backend_service.storyboard import ffmpeg_bin

        real = ffmpeg_bin.ffmpeg_filters()
        with mock.patch.object(ffmpeg_bin, "ffmpeg_filters", return_value=frozenset(real - {"subtitles"})):
            state = self.render()
        self.assertEqual(state["state"], "done", state)
        self.assertTrue(any("subtitles" in w for w in state["warnings"]), state)
        self.assertTrue(any(e.endswith(".srt") for e in state["extras"]))

    def test_preview_never_writes_sidecars_or_stems(self):
        self._captioned()
        self._op("edit_project", {"export": {"captionsSrt": True, "stems": True}})
        state = self.render(preview=True)
        self.assertEqual(state["state"], "done", state)
        self.assertNotIn("extras", state)

    def test_a_killed_render_is_interrupted_and_resumes_from_its_finished_shots(self):
        script = textwrap.dedent(f"""
            import sys, time
            from pathlib import Path
            from backend_service.storyboard import assemble
            from backend_service.routes.storyboard import start_render
            real, calls = assemble._run, []
            def slow(*a, **k):
                calls.append(1)
                if len(calls) == 3:           # two shots are finished; stop in the third
                    Path({str(self.base / "flag")!r}).write_text("x")
                    time.sleep(120)
                return real(*a, **k)
            assemble._run = slow
            start_render({self.pid!r}, None, False)
            time.sleep(120)
        """)
        proc = subprocess.Popen([sys.executable, "-c", script], cwd=str(REPO), env=os.environ.copy(),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.time() + 60
            while not (self.base / "flag").exists() and time.time() < deadline:
                time.sleep(0.1)
            self.assertTrue((self.base / "flag").exists(), "the child never reached shot 3")
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
        status = self.client.get(f"/api/storyboard/projects/{self.pid}/render", headers=H).json()["render"]
        self.assertEqual(status["state"], "interrupted", status)
        self.assertTrue(status["resumable"])
        self.assertEqual((status["shotsReady"], status["shotsTotal"]), (2, 3))
        from backend_service.storyboard import assemble
        from backend_service.storyboard.render_adapter import RENDERS

        cached = []
        real = assemble.render

        def spy(plan, *a, **k):
            result = real(plan, *a, **k)
            cached.append((result.shots_cached, result.shots_rendered))
            return result

        with mock.patch.object(assemble, "render", spy):
            state = self.render()
        self.assertEqual(state["state"], "done", state)
        self.assertEqual(cached, [(2, 1)])
        self.assertTrue(RENDERS.status(self.pid).out_path.is_file())
        self.assertEqual([p.name for p in (self.exports()).glob(".*partial*")], [])


class OtioTests(Base):
    def test_document_shape_crossfade_and_lanes(self):
        self._op("edit_shot", {"shotId": self.data["project"]["shots"][1]["id"],
                               "transition": {"type": "crossfade", "seconds": 0.5}})
        self._op("add_audio", {"kind": "music", "path": str(self.music), "start": 0.0, "gainDb": -6})
        self._op("add_audio", {"kind": "music", "path": str(self.music), "start": 1.0})
        response = self.client.post(f"/api/storyboard/projects/{self.pid}/otio", headers=H)
        self.assertEqual(response.status_code, 200, response.text)
        doc = json.loads((self.exports() / Path(response.json()["file"]).name).read_text())
        self.assertEqual(doc["OTIO_SCHEMA"], "Timeline.1")
        tracks = doc["tracks"]["children"]
        self.assertEqual([t["name"] for t in tracks], ["Picture", "Shot sound", "Music 1", "Music 2"])  # overlap -> 2 tracks
        kinds = [c["OTIO_SCHEMA"] for c in tracks[0]["children"]]
        self.assertEqual(kinds, ["Clip.1", "Transition.1", "Clip.1", "Clip.1"])
        durations = [c["source_range"]["duration"]["value"] for c in tracks[0]["children"] if c["OTIO_SCHEMA"] == "Clip.1"]
        # 24 fps: 1.5 s shots; a 0.5 s crossfade takes 0.25 s off each side of the joint.
        self.assertEqual(durations, [30, 30, 36])
        self.assertEqual(sum(durations), 36 * 3 - 12)  # the track is as long as the rendered film
        music = tracks[2]["children"][0]
        self.assertEqual(music["metadata"]["chaosengine"]["gainDb"], -6)
        self.assertTrue(music["media_reference"]["target_url"].startswith("file://"))

    def test_nothing_to_export(self):
        empty = self.client.post("/api/storyboard/projects", json={"name": "Empty"}, headers=H).json()["project"]["id"]
        response = self.client.post(f"/api/storyboard/projects/{empty}/otio", headers=H)
        self.assertEqual(response.status_code, 400)


class LongRunTests(unittest.TestCase):
    def test_banner_text_is_passed_as_arguments_not_script(self):
        evil = 'x" & do shell script "touch /tmp/pwned'
        with mock.patch.object(longrun.sys, "platform", "darwin"), \
                mock.patch.object(longrun.shutil, "which", return_value="/usr/bin/osascript"), \
                mock.patch.object(longrun.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertTrue(longrun.notify("Film", evil))
        argv = run.call_args[0][0]
        self.assertEqual(argv[-2:], ["Film", evil])
        self.assertNotIn(evil, " ".join(argv[:-2]))

    def test_keep_awake_never_raises_without_helpers(self):
        with mock.patch.object(longrun.shutil, "which", return_value=None):
            with longrun.keep_awake("test") as held:
                self.assertIn(held, (True, False))

    def test_short_or_preview_renders_get_no_banner(self):
        from backend_service.storyboard.render_adapter import RenderJob, RenderManager

        job = RenderJob("p", {"out_path": "/x/a.mp4", "preview": False}, notify=True)
        job.state, job.finished_at = "done", job.started_at + 5
        with mock.patch.object(longrun, "notify") as banner:
            RenderManager._announce(job)
            banner.assert_not_called()
            job.finished_at = job.started_at + 120
            RenderManager._announce(job)
            banner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
