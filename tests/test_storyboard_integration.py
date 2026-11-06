"""Store, routes and the real assembler together: tiny real clips in, one finished MP4 out."""

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.storyboard.ffmpeg_bin import ffmpeg_exe

H = {"Authorization": "Bearer app-token"}


def _ffmpeg(*args):
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-nostdin", *args], check=True, capture_output=True)


class StoryboardIntegrationTests(unittest.TestCase):
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
        self.clips = []
        for i in range(3):
            clip = self.base / f"c{i}.mp4"
            _ffmpeg("-f", "lavfi", "-i", f"testsrc2=s=160x90:r=24:d=1.5", "-f", "lavfi", "-i", "sine=f=440:d=1.5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clip))
            self.clips.append(clip)
        self.voice = self.base / "voice.wav"
        _ffmpeg("-f", "lavfi", "-i", "sine=f=2500:d=1.0", str(self.voice))

    def _op(self, pid, revision, op, args):
        response = self.client.post(f"/api/storyboard/projects/{pid}/ops", headers=H,
                                    json={"revision": revision, "op": op, "args": args})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_a_project_renders_to_a_tagged_mp4(self):
        data = self.client.post("/api/storyboard/projects", json={"name": "Mini film"}, headers=H).json()
        pid = data["project"]["id"]
        for i in range(3):
            data = self._op(pid, data["project"]["revision"], "add_shot", {"title": f"s{i}", "seconds": 1.5})
        for shot, clip in zip(data["project"]["shots"], self.clips):
            data = self._op(pid, data["project"]["revision"], "import_take", {"shotId": shot["id"], "clip": str(clip)})
        second = data["project"]["shots"][1]["id"]
        data = self._op(pid, data["project"]["revision"], "edit_shot",
                        {"shotId": second, "transition": {"type": "crossfade", "seconds": 0.5}})
        first = data["project"]["shots"][0]["id"]
        data = self._op(pid, data["project"]["revision"], "add_audio",
                        {"kind": "voice", "path": str(self.voice), "anchor": {"shotId": first, "offset": 0.2}})

        started = self.client.post(f"/api/storyboard/projects/{pid}/render", headers=H,
                                   json={"revision": data["project"]["revision"], "preview": True})
        self.assertEqual(started.status_code, 200, started.text)
        deadline = time.time() + 60
        state = {}
        while time.time() < deadline:
            state = self.client.get(f"/api/storyboard/projects/{pid}/render", headers=H).json()["render"] or {}
            if state.get("state") in ("done", "failed", "cancelled"):
                break
            time.sleep(0.3)
        self.assertEqual(state.get("state"), "done", state)

        folder = self.base / "storyboards" / pid / "exports"
        outputs = sorted(folder.glob("*.mp4"))
        self.assertTrue(outputs, "no exported MP4")
        info = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(outputs[-1])], capture_output=True, text=True).stderr
        self.assertIn("Audio:", info)
        self.assertIn("Video:", info)
        comment = next(line for line in info.splitlines() if line.strip().startswith("comment"))
        self.assertIn("chaosengine:", comment)        # the AI-label tag is on by default
        self.assertNotIn("Mini film", comment)        # no project name in the tag
        self.assertNotIn(str(self.base), comment)     # no local paths in the tag


if __name__ == "__main__":
    unittest.main()
