"""Remote use of the agent bridge: image upload handles and output downloads."""

import base64
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.helpers.media_input import InputImageError, resolve_input_image
from backend_service.mcp_bridge import access, files


def _image_bytes(fmt="PNG"):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(buffer, fmt)
    return buffer.getvalue()


class AgentFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ("images", "videos", "voice", "settings", "elsewhere"):
            (self.root / name).mkdir()
        self.addCleanup(self.tmp.cleanup)
        patches = [
            mock.patch.object(access, "_path", lambda: self.root / "agent_access.json"),
            mock.patch.object(files, "inbox_dir", self._inbox),
            mock.patch.object(files, "output_roots", lambda: [(self.root / n).resolve() for n in ("images", "videos", "voice")]),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(create_app(api_token="app-token"))
        access.set_enabled(True)
        self.headers = {"Authorization": f"Bearer {access.get_token()}"}

    def _inbox(self):
        path = self.root / "settings" / "agent-inbox"
        path.mkdir(exist_ok=True)
        return path

    # ---- upload ----
    def test_upload_returns_a_handle_that_resolves_like_a_path(self):
        response = self.client.post("/api/agent/uploads", content=_image_bytes(), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        handle = response.json()["handle"]
        self.assertTrue(handle.startswith("upload:"))
        path = resolve_input_image(handle)
        self.assertEqual(path.suffix, ".png")
        self.assertTrue(path.is_relative_to(self._inbox().resolve()))

    def test_type_comes_from_the_bytes(self):
        for fmt, extension in (("JPEG", ".jpg"), ("WEBP", ".webp")):
            handle = files.save_upload(_image_bytes(fmt))["handle"]
            self.assertEqual(files.resolve_handle(handle).suffix, extension)

    def test_bad_uploads_are_refused(self):
        for data, needle in ((b"", "empty"), (b"not an image at all", "not a PNG")):
            response = self.client.post("/api/agent/uploads", content=data, headers=self.headers)
            self.assertEqual(response.status_code, 400)
            self.assertIn(needle, response.json()["detail"])
        with mock.patch("backend_service.helpers.media_input.MAX_INPUT_IMAGE_BYTES", 20), \
                mock.patch("backend_service.mcp_bridge.files.MAX_INPUT_IMAGE_BYTES", 20):
            self.assertEqual(self.client.post("/api/agent/uploads", content=_image_bytes(), headers=self.headers).status_code, 413)

    def test_sound_and_video_uploads_are_typed_from_their_bytes(self):
        samples = {
            ".wav": b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 64,
            ".mp3": b"ID3\x04\x00" + b"\x00" * 64,
            ".flac": b"fLaC" + b"\x00" * 64,
            ".ogg": b"OggS" + b"\x00" * 64,
            ".m4a": b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 64,
            ".mp4": b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64,
            ".mov": b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 64,
            ".webm": b"\x1a\x45\xdf\xa3" + b"\x00" * 64,
        }
        for extension, data in samples.items():
            response = self.client.post("/api/agent/uploads", content=data, headers=self.headers)
            self.assertEqual(response.status_code, 200, extension)
            handle = response.json()["handle"]
            self.assertEqual(files.resolve_handle(handle, files.UPLOAD_EXTENSIONS).suffix, extension)
            with self.assertRaises(InputImageError):  # an image input never takes a sound or video handle
                resolve_input_image(handle)
        self.assertEqual([p.name for p in self._inbox().iterdir() if p.name.startswith(".part-")], [])

    def test_each_type_has_its_own_size_cap(self):
        wav = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 200
        mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200
        with mock.patch.object(files, "MAX_AUDIO_UPLOAD_BYTES", 100):
            self.assertEqual(self.client.post("/api/agent/uploads", content=wav, headers=self.headers).status_code, 413)
            self.assertEqual(self.client.post("/api/agent/uploads", content=mp4, headers=self.headers).status_code, 200)
            with self.assertRaises(InputImageError):
                files.save_upload(wav)
        with mock.patch.object(files, "MAX_VIDEO_UPLOAD_BYTES", 100):
            self.assertEqual(self.client.post("/api/agent/uploads", content=mp4, headers=self.headers).status_code, 413)
        self.assertEqual(files.max_upload_bytes(".png"), 30 * 1024 * 1024)
        self.assertEqual(files.max_upload_bytes(".flac"), 200 * 1024 * 1024)
        self.assertEqual(files.max_upload_bytes(".mov"), 1024 * 1024 * 1024)
        self.assertEqual([p.name for p in self._inbox().iterdir() if p.name.startswith(".part-")], [])

    def test_upload_needs_the_token_and_access_on(self):
        self.assertEqual(self.client.post("/api/agent/uploads", content=_image_bytes()).status_code, 401)
        access.set_enabled(False)
        response = self.client.post("/api/agent/uploads", content=_image_bytes(), headers=self.headers)
        self.assertEqual(response.status_code, 403)
        self.assertIn("Agent access is off", response.json()["detail"])

    def test_handles_cannot_escape_the_inbox(self):
        for raw in ("upload:../../etc/passwd", "upload:", "upload:zzzzzzzzzzzz", "upload:0123456789ab"):
            with self.assertRaises(InputImageError):
                resolve_input_image(raw)

    def test_old_inbox_files_are_removed(self):
        handle = files.save_upload(_image_bytes())["handle"]
        path = files.resolve_handle(handle)
        old = time.time() - files.INBOX_MAX_AGE_SECONDS - 60
        os.utime(path, (old, old))
        files.prune_inbox()
        self.assertFalse(path.exists())

    def test_upload_tool_round_trip(self):
        text = self._tool("upload_image", {"data_base64": base64.b64encode(_image_bytes()).decode()})
        self.assertTrue(json.loads(text)["handle"].startswith("upload:"))
        bad = self.client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "upload_image", "arguments": {"data_base64": "!!!!not base64!!!!"}}}, headers=self.headers).json()["result"]
        self.assertTrue(bad["isError"])

    def _tool(self, name, arguments):
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
        result = self.client.post("/mcp", json=body, headers=self.headers).json()["result"]
        self.assertFalse(result["isError"], result)
        return result["content"][0]["text"]

    # ---- download ----
    def test_outputs_can_be_fetched_with_size_and_checksum(self):
        target = self.root / "images" / "a.png"
        target.write_bytes(_image_bytes())
        info = files.download_info(target)
        self.assertEqual(info["bytes"], target.stat().st_size)
        response = self.client.get(info["url"], headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, target.read_bytes())
        import hashlib

        self.assertEqual(hashlib.sha256(response.content).hexdigest(), info["sha256"])

    def test_only_output_folders_can_be_fetched(self):
        outside = self.root / "elsewhere" / "secret.txt"
        outside.write_text("secret")
        inbox_file = files.resolve_handle(files.save_upload(_image_bytes())["handle"])
        traversal = str(self.root / "images" / ".." / "elsewhere" / "secret.txt")
        for path in (str(outside), str(inbox_file), traversal, "/etc/hosts", str(self.root / "images")):
            response = self.client.get("/api/agent/files", params={"path": path}, headers=self.headers)
            self.assertEqual(response.status_code, 404, path)
        self.assertIsNone(files.download_info(outside))

    def test_symlink_out_of_an_output_folder_is_refused(self):
        outside = self.root / "elsewhere" / "secret.txt"
        outside.write_text("secret")
        link = self.root / "images" / "link.png"
        link.symlink_to(outside)
        self.assertIsNone(files.fetchable(link))

    def test_download_needs_the_token(self):
        target = self.root / "videos" / "a.mp4"
        target.write_bytes(b"x" * 10)
        self.assertEqual(self.client.get("/api/agent/files", params={"path": str(target)}).status_code, 401)

    def test_a_finished_job_carries_its_download_info(self):
        from backend_service.mcp_bridge.jobs import STORE

        target = self.root / "images" / "job.png"
        target.write_bytes(_image_bytes())
        job = STORE.submit("voice", "test", lambda: ({"path": str(target)}, []))
        job.done.wait(5)
        self.assertEqual(job.state, "done")
        self.assertIn("/api/agent/files?path=", job.output["download"]["url"])
        self.assertEqual(job.output["download"]["bytes"], target.stat().st_size)


if __name__ == "__main__":
    unittest.main()
