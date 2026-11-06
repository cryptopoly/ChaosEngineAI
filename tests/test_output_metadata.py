import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from backend_service.helpers.image_artifacts import _save_image_artifact
from backend_service.helpers.output_metadata import (
    _ffmpeg_exe,
    embed_mp4_settings,
    read_mp4_settings,
    embed_png_settings,
    generation_settings,
    read_png_settings,
)


def _tiny_png() -> bytes:
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


class OutputMetadataTests(unittest.TestCase):
    def test_png_round_trip(self):
        settings = {"prompt": "a café — 猫", "seed": 7, "steps": 8}
        out = embed_png_settings(_tiny_png(), settings)
        self.assertEqual(read_png_settings(out), settings)
        self.assertTrue(out.endswith(b"IEND\xaeB`\x82"))

    def test_png_still_decodes(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow missing")
        import io

        out = embed_png_settings(_tiny_png(), {"seed": 1})
        self.assertEqual(Image.open(io.BytesIO(out)).size, (1, 1))

    def test_non_png_unchanged(self):
        self.assertEqual(embed_png_settings(b"nope", {"seed": 1}), b"nope")

    def test_saved_image_carries_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = _save_image_artifact(
                {"artifactId": "a1", "prompt": "red cube", "modelId": "m", "seed": 3,
                 "steps": 8, "width": 1, "height": 1, "imageBytes": _tiny_png()},
                Path(tmp),
            )
            got = read_png_settings(Path(saved["imagePath"]).read_bytes())
            self.assertEqual(got["prompt"], "red cube")
            self.assertEqual(got["seed"], 3)
            self.assertIn("appVersion", got)

    def test_generation_settings_skips_empty(self):
        self.assertNotIn("negativePrompt", generation_settings({"prompt": "x", "negativePrompt": ""}))

    def test_mp4_round_trip(self):
        import subprocess

        exe = _ffmpeg_exe()
        if not exe:
            self.skipTest("no ffmpeg")
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "c.mp4"
            subprocess.run([exe, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=0.5",
                            "-pix_fmt", "yuv420p", str(clip)], check=True)
            self.assertTrue(embed_mp4_settings(clip, {"prompt": "red", "seed": 9}))
            self.assertEqual(read_mp4_settings(clip), {"prompt": "red", "seed": 9})
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["c.mp4"])


if __name__ == "__main__":
    unittest.main()
