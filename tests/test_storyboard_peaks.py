"""Waveform peaks for the Mixer: real ffmpeg-made audio in, bucketed 0..1 peaks out, cached, and path-safe."""

import array
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.storyboard import peaks
from backend_service.storyboard.ffmpeg_bin import ffmpeg_exe

H = {"Authorization": "Bearer app-token"}


def _ffmpeg(*args):
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-nostdin", *args], check=True, capture_output=True)


class FoldSamplesTests(unittest.TestCase):
    def test_largest_absolute_value_per_bucket(self):
        block = array.array("f", [0.1, -0.5, 0.2, 0.0, 0.9, -0.3, 0.0, 0.0])
        self.assertEqual([round(p, 2) for p in peaks.fold_samples([block], 8, 4)], [0.5, 0.2, 0.9, 0.0])

    def test_chunk_boundaries_do_not_change_the_result(self):
        samples = [((i * 37) % 100) / 100 * (-1 if i % 3 == 0 else 1) for i in range(1000)]
        whole = peaks.fold_samples([array.array("f", samples)], 1000, 32)
        pieces = [array.array("f", samples[i:i + 77]) for i in range(0, 1000, 77)]
        self.assertEqual(peaks.fold_samples(pieces, 1000, 32), whole)

    def test_more_samples_than_announced_land_in_the_last_bucket(self):
        out = peaks.fold_samples([array.array("f", [0.0, 0.0, 0.0, 0.0, 0.8])], 4, 4)
        self.assertEqual(len(out), 4)
        self.assertAlmostEqual(out[3], 0.8, places=5)

    def test_bucket_count_is_clamped(self):
        self.assertEqual(peaks.clamp_buckets(1), peaks.MIN_BUCKETS)
        self.assertEqual(peaks.clamp_buckets(10**6), peaks.MAX_BUCKETS)


class PeaksRouteTests(unittest.TestCase):
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
        # one second of silence then one second of a loud tone
        self.wav = self.base / "tone.wav"
        _ffmpeg("-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono:d=1", "-f", "lavfi", "-i", "sine=f=440:sample_rate=8000:d=1",
                "-filter_complex", "[0][1]concat=n=2:v=0:a=1", str(self.wav))
        self.clip = self.base / "clip.mp4"
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x90:r=24:d=1", "-f", "lavfi", "-i", "sine=f=330:d=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(self.clip))
        self.silent = self.base / "silent.mp4"
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=s=160x90:r=24:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.silent))
        data = self.client.post("/api/storyboard/projects", json={"name": "Peaks"}, headers=H).json()
        self.pid = data["project"]["id"]
        self.revision = data["project"]["revision"]
        data = self._op("add_audio", {"kind": "voice", "path": str(self.wav)})
        self.audio_rel = data["project"]["audio"][0]["source"]["path"]
        data = self._op("add_shot", {"title": "one", "seconds": 1})
        shot = data["project"]["shots"][0]["id"]
        data = self._op("import_take", {"shotId": shot, "clip": str(self.clip)})
        self.clip_rel = data["project"]["shots"][0]["takes"][0]["clip"]
        data = self._op("add_shot", {"title": "two", "seconds": 1})
        shot = data["project"]["shots"][1]["id"]
        data = self._op("import_take", {"shotId": shot, "clip": str(self.silent)})
        self.silent_rel = data["project"]["shots"][1]["takes"][0]["clip"]
        self.folder = self.base / "storyboards" / self.pid

    def _op(self, op, args):
        response = self.client.post(f"/api/storyboard/projects/{self.pid}/ops", headers=H,
                                    json={"revision": self.revision, "op": op, "args": args})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.revision = data["project"]["revision"]
        return data

    def peaks(self, rel, buckets=None, code=200):
        params = {"path": rel}
        if buckets is not None:
            params["buckets"] = buckets
        response = self.client.get(f"/api/storyboard/projects/{self.pid}/peaks", params=params, headers=H)
        self.assertEqual(response.status_code, code, response.text)
        return response.json()

    def test_peaks_follow_the_loudness_over_time(self):
        data = self.peaks(self.audio_rel, 64)
        self.assertEqual(len(data["peaks"]), 64)
        self.assertAlmostEqual(data["durationS"], 2.0, delta=0.1)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in data["peaks"]))
        self.assertLess(max(data["peaks"][:28]), 0.05)  # the silent first half
        self.assertGreater(min(data["peaks"][36:60]), 0.8)  # the tone, scaled so the loudest bucket is 1.0
        self.assertAlmostEqual(max(data["peaks"]), 1.0, places=3)

    def test_the_sound_of_a_video_clip(self):
        data = self.peaks(self.clip_rel, 40)
        self.assertEqual(len(data["peaks"]), 40)
        self.assertGreater(max(data["peaks"]), 0.9)
        self.assertAlmostEqual(data["durationS"], 1.0, delta=0.2)

    def test_a_clip_without_sound_has_no_peaks(self):
        data = self.peaks(self.silent_rel, 40)
        self.assertEqual(data["peaks"], [])
        self.assertGreater(data["durationS"], 0.5)

    def test_buckets_are_bounded(self):
        self.peaks(self.audio_rel, 31, code=422)
        self.peaks(self.audio_rel, 4001, code=422)
        self.assertEqual(len(self.peaks(self.audio_rel, 4000)["peaks"]), 4000)
        self.assertEqual(len(self.peaks(self.audio_rel)["peaks"]), 400)

    def test_second_request_comes_from_the_cache_keyed_by_hash_and_buckets(self):
        first = self.peaks(self.audio_rel, 64)
        cached = sorted((self.folder / "cache" / "peaks").glob("*-64.json"))
        self.assertEqual(len(cached), 1)
        self.assertEqual(json.loads(cached[0].read_text()), first)
        with mock.patch.object(peaks, "compute_peaks", side_effect=AssertionError("decoded again")):
            self.assertEqual(self.peaks(self.audio_rel, 64), first)
        self.peaks(self.audio_rel, 96)
        self.assertEqual(len(list((self.folder / "cache" / "peaks").glob("*.json"))), 2)

    def test_a_changed_file_gets_new_peaks(self):
        first = self.peaks(self.audio_rel, 64)
        target = self.folder / self.audio_rel
        loud = self.base / "loud.wav"
        _ffmpeg("-f", "lavfi", "-i", "sine=f=800:sample_rate=8000:d=2", str(loud))
        target.write_bytes(loud.read_bytes())
        second = self.peaks(self.audio_rel, 64)
        self.assertNotEqual(first["peaks"], second["peaks"])

    def test_paths_must_stay_inside_the_project(self):
        for bad in ("../../etc/passwd", "media/../../x", "/etc/passwd", "C:/x.wav", "media/audio/missing.wav", "project.json"):
            self.peaks(bad, 64, code=404)

    def test_a_symlink_out_of_the_project_is_refused(self):
        outside = self.base / "outside.wav"
        outside.write_bytes(self.wav.read_bytes())
        link = self.folder / "media" / "audio" / "escape.wav"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available")
        self.peaks("media/audio/escape.wav", 64, code=404)

    def test_needs_the_app_token_and_a_known_project(self):
        self.client.app.state.chaosengine_require_api_auth = True  # the default; a local settings file may turn it off
        self.assertEqual(self.client.get(f"/api/storyboard/projects/{self.pid}/peaks",
                                         params={"path": self.audio_rel}).status_code, 401)
        response = self.client.get("/api/storyboard/projects/0123456789abcdef/peaks", params={"path": self.audio_rel}, headers=H)
        self.assertEqual(response.status_code, 404)

    def test_an_unreadable_file_is_a_readable_400(self):
        junk = self.folder / "media" / "audio" / "junk.wav"
        junk.write_bytes(b"not audio at all")
        data = self.peaks("media/audio/junk.wav", 64, code=400)
        self.assertIn("junk.wav", data["detail"])


if __name__ == "__main__":
    unittest.main()
