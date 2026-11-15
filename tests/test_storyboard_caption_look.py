"""Burnt-in caption look (S13): the default look, the S15 project look when present, the ffmpeg
``force_style`` text, the render-plan checks, and a real burn-in whose pixels are read back."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend_service.storyboard import captions, ffmpeg_bin
from backend_service.storyboard.assemble import render
from backend_service.storyboard.assemble_graphs import build_mix_graph, caption_force_style
from backend_service.storyboard.render_plan import PlanError, RenderPlan


def _ffmpeg() -> str | None:
    try:
        return ffmpeg_bin.ffmpeg_exe()
    except RuntimeError:
        return None


class LookTests(unittest.TestCase):
    def test_default_look_is_white_bottom_medium(self):
        look = captions.look_for(SimpleNamespace(), "Hello there", platform="darwin")
        self.assertEqual((look.position, look.size, look.colour), ("bottom", "medium", "#FFFFFF"))
        self.assertEqual(look.font, "Helvetica Neue")

    def test_cjk_text_picks_a_font_with_those_glyphs(self):
        self.assertEqual(captions.look_for(SimpleNamespace(), "お茶の時間", platform="darwin").font, "Hiragino Sans")
        self.assertEqual(captions.look_for(SimpleNamespace(), "喝茶", platform="win32").font, "Microsoft YaHei")
        self.assertEqual(captions.look_for(SimpleNamespace(), "차", platform="linux").font, "Noto Sans CJK SC")
        self.assertEqual(captions.look_for(SimpleNamespace(), "tea", platform="win32").font, "Segoe UI")

    def test_project_caption_look_is_used_when_the_project_has_one(self):
        brand = SimpleNamespace(captionLook={"position": "top", "size": "large", "colour": "#ffd700", "font": "Avenir Next"})
        look = captions.look_for(SimpleNamespace(brand=brand), "Hi", platform="darwin")
        self.assertEqual((look.position, look.size, look.colour, look.font), ("top", "large", "#FFD700", "Avenir Next"))

    def test_a_bad_project_look_falls_back_field_by_field(self):
        brand = SimpleNamespace(captionLook=SimpleNamespace(position="left", size="huge", color="red", font="Evil',x=1"))
        look = captions.look_for(SimpleNamespace(brand=brand), "Hi", platform="darwin")
        self.assertEqual((look.position, look.size, look.colour, look.font), ("bottom", "medium", "#FFFFFF", "Helvetica Neue"))


class ForceStyleTests(unittest.TestCase):
    def test_bottom_white_at_720p(self):
        style = caption_force_style({"position": "bottom", "size": "medium", "colour": "#FFFFFF", "font": "Helvetica Neue"}, 1280, 720)
        self.assertEqual(
            style,
            "FontName=Helvetica Neue,Bold=1,Fontsize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BackColour=&H80000000,BorderStyle=1,Outline=1.4,Shadow=0.6,Alignment=2,MarginV=22,MarginL=24,MarginR=24",
        )

    def test_position_size_and_colour(self):
        top = caption_force_style({"position": "top", "size": "large", "colour": "#FFD700"}, 1280, 720)
        self.assertIn("Alignment=6", top)
        self.assertIn("Fontsize=23", top)
        self.assertIn("PrimaryColour=&H0000D7FF", top)  # ASS colours are BGR
        self.assertNotIn("FontName", top)
        middle = caption_force_style({"position": "middle", "size": "small"}, 1280, 720)
        self.assertIn("Alignment=10", middle)
        self.assertIn("MarginV=0", middle)
        self.assertIn("Fontsize=15", middle)

    def test_portrait_frames_get_smaller_type(self):
        portrait = caption_force_style({"size": "medium"}, 720, 1280)
        self.assertIn("Fontsize=10", portrait)  # 18 x 720 / 1280, so a line still fits the narrow frame

    def test_unsafe_values_never_reach_the_filter(self):
        for bad in ({"font": "A',original_size=1x1"}, {"colour": "white"}, {"position": "left"}, {"size": "xl"}):
            with self.assertRaises(ValueError):
                caption_force_style(bad, 1280, 720)

    def test_mix_graph_burns_with_the_style(self):
        graph, _ = build_mix_graph([], 48000, None, "run-captions.srt", None, "Alignment=2,Fontsize=18")
        self.assertIn("[0:v]subtitles=filename=run-captions.srt:force_style='Alignment=2,Fontsize=18'[vout]", graph)
        plain, _ = build_mix_graph([], 48000, None, "run-captions.srt", None)
        self.assertIn("[0:v]subtitles=filename=run-captions.srt[vout]", plain)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "a.mp4").write_bytes(b"x")
        (root / "c.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
        self.base = {
            "width": 160, "height": 90, "fps": 24, "out_path": str(root / "out.mp4"), "work_dir": str(root / "work"),
            "shots": [{"id": "a", "video": str(root / "a.mp4"), "hash": "h"}], "captions_srt": str(root / "c.srt"),
        }

    def test_valid_style_round_trips(self):
        style = {"position": "top", "size": "small", "colour": "#00FF00", "font": "Hiragino Sans"}
        plan = RenderPlan.from_dict({**self.base, "captions_style": style})
        self.assertEqual(plan.captions_style, style)
        self.assertEqual(RenderPlan.from_dict(plan.to_dict()).captions_style, style)
        self.assertIsNone(RenderPlan.from_dict(self.base).captions_style)

    def test_bad_style_is_refused(self):
        for bad in ({"colour": "#FFF"}, {"font": "x,y"}, {"position": "side"}, {"shadow": "1"}, "bottom"):
            with self.assertRaises(PlanError):
                RenderPlan.from_dict({**self.base, "captions_style": bad})


class BurnInRenderTests(unittest.TestCase):
    """A real render: yellow captions at the top must put yellow pixels in the top band and none in the bottom band."""

    def setUp(self):
        self.exe = _ffmpeg()
        if not self.exe:
            self.skipTest("ffmpeg not available")
        if "subtitles" not in ffmpeg_bin.ffmpeg_filters():
            self.skipTest("this ffmpeg has no subtitles filter")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        subprocess.run([self.exe, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x202830:s=640x360:d=2:r=24",
                        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(self.root / "a.mp4")], check=True)
        (self.root / "c.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\nTEA TIME TEA TIME\n", encoding="utf-8")

    def _band_has_yellow(self, film: Path, top: bool) -> bool:
        crop = "crop=640:90:0:0" if top else "crop=640:90:0:270"
        raw = subprocess.run([self.exe, "-loglevel", "error", "-ss", "1", "-i", str(film), "-frames:v", "1", "-vf",
                              f"{crop},format=rgb24", "-f", "rawvideo", "-"], check=True, capture_output=True).stdout
        pixels = [raw[i:i + 3] for i in range(0, len(raw), 3)]
        return sum(1 for r, g, b in pixels if r > 180 and g > 150 and b < 90) > 50

    def test_top_yellow_captions(self):
        plan = RenderPlan.from_dict({
            "width": 640, "height": 360, "fps": 24, "loudness_lufs": None,
            "out_path": str(self.root / "film.mp4"), "work_dir": str(self.root / "work"),
            "shots": [{"id": "a", "video": str(self.root / "a.mp4"), "hash": "h"}],
            "captions_srt": str(self.root / "c.srt"),
            "captions_style": {"position": "top", "size": "large", "colour": "#FFD700", "font": "Helvetica Neue"},
        })
        render(plan)
        film = self.root / "film.mp4"
        self.assertTrue(self._band_has_yellow(film, top=True))
        self.assertFalse(self._band_has_yellow(film, top=False))


if __name__ == "__main__":
    unittest.main()
