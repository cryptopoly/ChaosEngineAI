"""The brand kit on the exported film (S15 export): title and end cards, the corner logo, the plan checks, the
ffmpeg graphs (no typed text ever reaches a filter), the captions moved behind the title card, and a real render
whose frames are read back."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from backend_service.storyboard import cards, ffmpeg_bin
from backend_service.storyboard.assemble import render
from backend_service.storyboard.assemble_graphs import (
    LogoOverlay,
    build_card_graph,
    build_mix_graph,
    card_frames,
    card_layout,
    logo_chain,
    shift_unanchored,
)
from backend_service.storyboard.models import Brand, CaptionLook, Caption, Project, RefImage, Reference, Shot, Take
from backend_service.storyboard.render_plan import AudioInput, CardInput, PlanError, RenderPlan
from backend_service.storyboard.timeline import build_render_plan
from tests.storyboard_fixtures import png

NASTY = "Tea'; x=1:enable=0, %{pts} [v] \\ drawtext=text=evil"


def _ffmpeg() -> str | None:
    try:
        return ffmpeg_bin.ffmpeg_exe()
    except RuntimeError:
        return None


class PlanBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "a.mp4").write_bytes(b"x")
        self.logo = png(root / "logo.png")
        (root / "font.ttf").write_bytes(b"x")
        (root / "font.sh").write_bytes(b"x")
        self.root = root
        self.base = {"width": 160, "height": 90, "fps": 24, "out_path": str(root / "out.mp4"), "work_dir": str(root / "work"),
                     "shots": [{"id": "a", "video": str(root / "a.mp4"), "hash": "h"}]}

    def card(self, **over):
        return {"text": "Tea Shop", "seconds": 2, "background": "#112233", "colour": "#ffffff",
                "font_file": str(self.root / "font.ttf"), "logo": str(self.logo), **over}

    def test_cards_and_logo_round_trip(self):
        plan = RenderPlan.from_dict({**self.base, "title_card": self.card(), "end_card": self.card(text=""),
                                     "logo": {"path": str(self.logo), "corner": "top-left"}})
        self.assertEqual(plan.title_card.colour, "#FFFFFF")
        self.assertEqual(plan.end_card.text, "")
        again = RenderPlan.from_dict(plan.to_dict())
        self.assertEqual(again.title_card, plan.title_card)
        self.assertEqual(again.logo, plan.logo)
        self.assertIsNone(RenderPlan.from_dict(self.base).title_card)

    def test_unsafe_card_fields_are_refused(self):
        bad = [
            {"text": "two\nlines"}, {"text": "bell\x07"}, {"text": "x" * 201}, {"text": 5}, {"text": "sep "},
            {"background": "red"}, {"colour": "#FFF"}, {"seconds": 0.1}, {"seconds": 99},
            {"font_file": str(self.root / "font.sh")}, {"font_file": "relative.ttf"}, {"font_file": "http://x/f.ttf"},
            {"logo": str(self.root / "a.mp4")}, {"logo": "-i.png"},
        ]
        for over in bad:
            with self.assertRaises(PlanError, msg=over):
                RenderPlan.from_dict({**self.base, "title_card": self.card(**over)})

    def test_bad_logo_is_refused(self):
        for logo in ({"path": str(self.logo), "corner": "centre"}, {"path": str(self.root / "missing.png")}, "x"):
            with self.assertRaises(PlanError):
                RenderPlan.from_dict({**self.base, "logo": logo})


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "a.mp4").write_bytes(b"x")
        self.plan = RenderPlan.from_dict({"width": 1280, "height": 720, "fps": 24, "out_path": str(root / "o.mp4"),
                                          "work_dir": str(root / "w"),
                                          "shots": [{"id": "a", "video": str(root / "a.mp4"), "hash": "h"}]})

    def test_card_graph_reads_the_text_from_a_file_with_expansion_off(self):
        card = CardInput(text=NASTY, seconds=2.0, background="#112233", colour="#FFD700")
        layout = card_layout(card, 1280, 720, None, has_logo=True)
        graph = build_card_graph(card, self.plan, card_frames(card, 24), layout, "abc-title.txt", "abc-font.ttf", 0)
        self.assertNotIn("Tea", graph)
        self.assertNotIn("evil", graph)
        self.assertIn("textfile=abc-title.txt:expansion=none", graph)
        self.assertIn("fontfile=abc-font.ttf", graph)
        self.assertIn("color=c=0x112233:s=1280x720:r=24", graph)
        self.assertIn("fontcolor=0xFFD700", graph)
        self.assertIn("[0:v]format=rgba,scale=-1:158[cardlogo]", graph)
        self.assertIn("trim=end_frame=48", graph)
        self.assertIn("atrim=end_sample=96000", graph)

    def test_card_graph_refuses_unsafe_file_names(self):
        card = CardInput(text="Hi")
        layout = card_layout(card, 1280, 720, None, False)
        for name in ("a b.txt", "../x.txt", "x:y.txt", "-x.txt", "x'.txt"):
            with self.assertRaises(ValueError, msg=name):
                build_card_graph(card, self.plan, 48, layout, name, None, None)

    def test_card_without_text_or_logo(self):
        card = CardInput(text="")
        graph = build_card_graph(card, self.plan, 48, card_layout(card, 1280, 720, None, False), None, None, None)
        self.assertNotIn("drawtext", graph)
        self.assertNotIn("overlay", graph)

    def test_layout_shrinks_long_text_to_the_frame(self):
        card = CardInput(text="A rather long title for a short film about tea")
        layout = card_layout(card, 1280, 720, lambda size: size * 0.55 * len(card.text), False)
        self.assertLessEqual(layout.font_size * 0.55 * len(card.text), 1280 * 0.84)
        self.assertEqual(card_layout(CardInput(text="Tea"), 1280, 720, lambda size: size * 2.0, False).font_size, 65)
        with_logo = card_layout(CardInput(text="Tea"), 1280, 720, None, True)
        self.assertLess(with_logo.logo_y + with_logo.logo_height, with_logo.text_y)

    def test_logo_corners_and_window(self):
        expected = {"top-left": "x=29:y=29", "top-right": "x=W-w-29:y=29", "bottom-left": "x=29:y=H-h-29",
                    "bottom-right": "x=W-w-29:y=H-h-29"}
        for corner, place in expected.items():
            chain = logo_chain(LogoOverlay(3, corner, 1280, 720), "0:v", "vl")
            self.assertIn(f"[3:v]format=rgba,scale=-1:86[logo]", chain)
            self.assertIn(f"overlay={place}:eof_action=repeat[vl]", chain)
            self.assertNotIn("enable", chain)
        windowed = logo_chain(LogoOverlay(1, "top-left", 1280, 720, 2.0, 7.5), "0:v", "vl")
        self.assertIn(":enable='between(t,2,7.5)'", windowed)
        with self.assertRaises(ValueError):
            logo_chain(LogoOverlay(1, "middle", 1280, 720), "0:v", "vl")

    def test_mix_graph_draws_the_logo_under_the_captions(self):
        graph, _ = build_mix_graph([], 48000, None, "x-captions.srt", None, "Bold=1",
                                   LogoOverlay(1, "bottom-right", 1280, 720, 2, 6), "x-fonts")
        self.assertIn("[0:v][logo]overlay", graph)
        self.assertIn("[vlogo]subtitles=filename=x-captions.srt:fontsdir=x-fonts:force_style='Bold=1'[vout]", graph)
        logo_only, _ = build_mix_graph([], 48000, None, logo=LogoOverlay(1, "top-left", 1280, 720))
        self.assertIn("[vlogo]null[vout]", logo_only)

    def test_clips_placed_by_time_move_behind_the_title_card(self):
        loose = AudioInput(id="m", kind="music", path="/x", start=1.0)
        anchored = AudioInput(id="v", kind="voice", path="/x", start=1.0, anchor_shot="a")
        moved = shift_unanchored([loose, anchored], 2.0)
        self.assertEqual((moved[0].start, moved[1].start), (3.0, 1.0))
        self.assertEqual(shift_unanchored([loose], 0.0)[0].start, 1.0)


def _project(**export) -> Project:
    take = Take(id="t1", clip="media/a/t1/clip.mp4", measuredSeconds=3.0, hashes={"clip": "a" * 64})
    p = Project(id="0123456789abcdef", name="Tea Shop", shots=[Shot(id="a", takes=[take], activeTake="t1", status="done")])
    for key, value in export.items():
        setattr(p.export, key, value)
    return p


class ProjectChoiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / "0123456789abcdef"
        clip = self.folder / "media/a/t1/clip.mp4"
        clip.parent.mkdir(parents=True)
        clip.write_bytes(b"x")
        png(self.folder / "media/brand/logo.png")

    def test_colours_follow_the_style_palette_and_keep_contrast(self):
        p = _project()
        self.assertEqual(cards.card_colours(p), ("#000000", "#FFFFFF"))
        p.references.styles.append(Reference(id="s1", palette=["#f5e6c8", "#123456"]))
        self.assertEqual(cards.card_colours(p), ("#F5E6C8", "#000000"))  # white on cream does not read: black
        p.brand = Brand(captions=CaptionLook(colour="#5a2d0c"))
        self.assertEqual(cards.card_colours(p), ("#F5E6C8", "#5A2D0C"))  # the brand colour reads: kept
        p.references.styles.insert(0, Reference(id="s0"))  # a style with no palette is skipped
        self.assertEqual(cards.card_colours(p)[0], "#F5E6C8")

    def test_text_defaults_to_the_project_name(self):
        p = _project(titleText="  ", endText="Visit us")
        self.assertEqual((cards.card_text(p, "title"), cards.card_text(p, "end")), ("Tea Shop", "Visit us"))

    def test_plan_entries(self):
        p = _project()
        self.assertEqual(cards.plan_entries(p, self.folder), {"title_card": None, "end_card": None, "logo": None})
        p.brand = Brand(logo=RefImage(path="media/brand/logo.png"), logoCorner="top-left")
        p.export.titleCard = True
        p.export.cardSeconds = 1.5
        out = cards.plan_entries(p, self.folder)
        self.assertEqual(out["logo"], {"path": str((self.folder / "media/brand/logo.png").resolve()), "corner": "top-left"})
        self.assertEqual((out["title_card"]["text"], out["title_card"]["seconds"]), ("Tea Shop", 1.5))
        self.assertEqual(out["title_card"]["logo"], out["logo"]["path"])
        self.assertIsNone(out["end_card"])
        p.export.showLogo = False
        self.assertIsNone(cards.plan_entries(p, self.folder)["logo"])

    def test_render_plan_moves_captions_behind_the_title_card(self):
        p = _project(titleCard=True, cardSeconds=2.0, captionsSrt=True, captionsBurn=True)
        p.captions.append(Caption(id="c1", start=0.5, end=1.5, text="Hello"))
        plan = build_render_plan(p, preview=False, folder=self.folder)
        srt = Path(plan["captions_srt"]).read_text(encoding="utf-8")
        self.assertIn("00:00:02,500 --> 00:00:03,500", srt)
        self.assertIn("00:00:02,500", plan["sidecar_srt"]["text"])
        preview = build_render_plan(p, preview=True, folder=self.folder)
        self.assertIsNone(preview["title_card"])  # the Mixer's preview stays on its timeline
        self.assertIn("00:00:00,500", Path(preview["captions_srt"]).read_text(encoding="utf-8"))
        (self.folder / "exports").mkdir()
        RenderPlan.from_dict(plan)  # the plan passes its own checks


class BrandRenderTests(unittest.TestCase):
    """A real render: a red title card with text and the logo, a blue shot with the logo in its corner, an end card."""

    def setUp(self):
        self.exe = _ffmpeg()
        if not self.exe:
            self.skipTest("ffmpeg not available")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        subprocess.run([self.exe, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x0000C0:s=320x180:d=2:r=24",
                        "-f", "lavfi", "-i", "sine=f=440:r=48000:d=2", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-ac", "2", str(self.root / "a.mp4")], check=True)
        self.logo = png(self.root / "logo.png", (0, 220, 0))
        from PIL import Image

        Image.new("RGB", (64, 64), (0, 220, 0)).save(self.logo)
        font = cards.card_font_file(_project(), "Tea")
        self.font = font

    def grab(self, film: Path, at: float):
        raw = subprocess.run([self.exe, "-loglevel", "error", "-ss", str(at), "-i", str(film), "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
        return lambda x, y: tuple(raw[(y * 320 + x) * 3:(y * 320 + x) * 3 + 3])

    def test_cards_and_corner_logo(self):
        card = {"text": "TEA TIME", "seconds": 1.0, "background": "#C00000", "colour": "#FFFFFF",
                "font_file": self.font, "logo": str(self.logo)}
        plan = RenderPlan.from_dict({
            "width": 320, "height": 180, "fps": 24, "loudness_lufs": None,
            "out_path": str(self.root / "film.mp4"), "work_dir": str(self.root / "work"),
            "shots": [{"id": "a", "video": str(self.root / "a.mp4"), "hash": "h"}],
            "audio": [{"id": "m", "kind": "music", "path": str(self.root / "a.mp4"), "start": 0.0, "duration": 1.0}],
            "title_card": card, "end_card": {**card, "text": "BYE", "logo": None},
            "logo": {"path": str(self.logo), "corner": "bottom-right"},
        })
        result = render(plan)
        self.assertAlmostEqual(result.duration_s, 4.0, delta=0.06)  # 1 s + 2 s + 1 s
        counted = subprocess.run([self.exe, "-hide_banner", "-i", result.path, "-map", "0:v", "-f", "null", "-"],
                                 capture_output=True, text=True).stderr
        self.assertIn("frame=   96", counted)  # whole frames: 24 + 48 + 24, none lost to the logo overlay
        title = self.grab(Path(result.path), 0.5)
        self.assertGreater(title(5, 5)[0], 150)  # red background
        self.assertGreater(title(160, 60)[1], 150)  # the logo, centred above the text
        if "drawtext" in ffmpeg_bin.ffmpeg_filters() and self.font:
            band = [title(x, y) for y in range(100, 140) for x in range(40, 280)]
            self.assertGreater(sum(1 for r, g, b in band if r > 200 and g > 200 and b > 200), 30)  # white text
        shot = self.grab(Path(result.path), 2.0)
        self.assertGreater(shot(160, 60)[2], 150)  # blue shot
        self.assertGreater(shot(320 - 12 - 8, 180 - 7 - 8)[1], 150)  # green logo in the bottom-right corner
        self.assertLess(title(320 - 12 - 8, 180 - 7 - 8)[1], 100)  # not over the title card's corner
        end = self.grab(Path(result.path), 3.5)
        self.assertGreater(end(5, 5)[0], 150)
        self.assertLess(end(320 - 12 - 8, 180 - 7 - 8)[1], 100)
        # The rendered cards are cached like shots.
        again = render(plan)
        self.assertEqual(again.shots_cached, 3)


if __name__ == "__main__":
    unittest.main()
