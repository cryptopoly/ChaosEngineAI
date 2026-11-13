"""ffmpeg lookup order, FfmpegMissing mapping, and the H.264 encoder chooser over fake ``-encoders`` output."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.storyboard import encoders, ffmpeg_bin

ENC_LIBX264 = """Encoders:
 V..... = Video
 ------
 V....D libx264              libx264 H.264 (codec h264)
 V....D libx265              libx265 H.265 (codec hevc)
 V....D h264_videotoolbox    VideoToolbox H.264 Encoder (codec h264)
 A....D aac                  AAC (Advanced Audio Coding)
"""
ENC_MAC_LGPL = """ V....D h264_videotoolbox    VideoToolbox H.264 Encoder (codec h264)
 V....D hevc_videotoolbox    VideoToolbox H.265 Encoder (codec hevc)
 A....D aac                  AAC
"""
ENC_WIN_BTBN = """ V....D libopenh264          OpenH264 H.264 / AVC (codec h264)
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V....D h264_amf             AMD AMF H.264 Encoder (codec h264)
 V....D h264_qsv             H.264 / AVC / MPEG-4 AVC (Intel Quick Sync Video acceleration) (codec h264)
 V....D libvpx-vp9           libvpx VP9 (codec vp9)
"""
ENC_NONE = " V....D libvpx-vp9           libvpx VP9 (codec vp9)\n A....D aac AAC\n"


class ParseEncoders(unittest.TestCase):
    def test_lists_video_encoders_only(self):
        names = encoders.parse_encoders(ENC_LIBX264)
        self.assertEqual(names, {"libx264", "libx265", "h264_videotoolbox"})

    def test_empty(self):
        self.assertEqual(encoders.parse_encoders(""), frozenset())


class ChooseH264(unittest.TestCase):
    def pick(self, text, platform, **kw):
        return encoders.choose_h264(encoders.parse_encoders(text), platform, **kw)

    def test_libx264_wins_everywhere(self):
        for platform in ("darwin", "win32", "linux"):
            self.assertEqual(self.pick(ENC_LIBX264, platform), "libx264")

    def test_mac_lgpl_uses_videotoolbox(self):
        self.assertEqual(self.pick(ENC_MAC_LGPL, "darwin"), "h264_videotoolbox")

    def test_windows_order_is_nvenc_then_amf_then_qsv_then_openh264(self):
        self.assertEqual(self.pick(ENC_WIN_BTBN, "win32"), "h264_nvenc")
        self.assertEqual(self.pick(ENC_WIN_BTBN, "win32", trial=lambda n: n != "h264_nvenc"), "h264_amf")
        self.assertEqual(self.pick(ENC_WIN_BTBN, "win32", trial=lambda n: n in ("h264_qsv", "libopenh264")), "h264_qsv")
        self.assertEqual(self.pick(ENC_WIN_BTBN, "win32", trial=lambda n: n == "libopenh264"), "libopenh264")

    def test_linux_skips_amf(self):
        self.assertEqual(self.pick(ENC_WIN_BTBN, "linux", trial=lambda n: n != "h264_nvenc"), "h264_qsv")
        self.assertEqual(self.pick(ENC_WIN_BTBN, "linux", trial=lambda n: n == "h264_amf"), None)

    def test_listed_but_absent_hardware_falls_through_to_openh264(self):
        self.assertEqual(self.pick(ENC_WIN_BTBN, "linux", trial=lambda n: n == "libopenh264"), "libopenh264")

    def test_libx264_is_never_trialled(self):
        trial = mock.Mock(return_value=False)
        self.assertEqual(self.pick(ENC_LIBX264, "darwin", trial=trial), "libx264")
        trial.assert_not_called()

    def test_no_h264_encoder(self):
        self.assertIsNone(self.pick(ENC_NONE, "linux"))
        self.assertIsNone(self.pick(ENC_MAC_LGPL, "linux"))

    def test_override_wins_when_listed_and_working(self):
        self.assertEqual(self.pick(ENC_WIN_BTBN, "win32", override="libopenh264", trial=lambda n: True), "libopenh264")
        self.assertEqual(self.pick(ENC_WIN_BTBN, "win32", override="h264_amf", trial=lambda n: n != "h264_amf"), "h264_nvenc")
        self.assertEqual(self.pick(ENC_WIN_BTBN, "win32", override="auto", trial=lambda n: True), "h264_nvenc")
        self.assertEqual(self.pick(ENC_LIBX264, "win32", override="nope"), "libx264")

    def test_platform_keys(self):
        self.assertEqual(encoders.platform_key("win32"), "win32")
        self.assertEqual(encoders.platform_key("darwin"), "darwin")
        self.assertEqual(encoders.platform_key("linux"), "linux")
        self.assertEqual(encoders.platform_key("freebsd13"), "linux")


class CodecArgs(unittest.TestCase):
    def test_libx264_matches_the_old_presets(self):
        self.assertEqual(encoders.codec_args("libx264", False),
                         ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"])
        self.assertEqual(encoders.codec_args("libx264", True),
                         ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p"])

    def test_every_encoder_has_final_and_preview_with_yuv420p(self):
        for name in ("libx264", "h264_videotoolbox", "h264_nvenc", "h264_amf", "h264_qsv", "libopenh264"):
            final, preview = encoders.codec_args(name, False), encoders.codec_args(name, True)
            self.assertEqual(final[:2], ["-c:v", name])
            self.assertNotEqual(final, preview, name)
            self.assertEqual(final[-2:], ["-pix_fmt", "yuv420p"])

    def test_videotoolbox_quality_and_realtime_preview(self):
        self.assertIn("65", encoders.codec_args("h264_videotoolbox", False))
        preview = encoders.codec_args("h264_videotoolbox", True)
        self.assertEqual(preview[preview.index("-realtime") + 1], "1")

    def test_nvenc_constant_quality(self):
        args = encoders.codec_args("h264_nvenc", False)
        self.assertEqual(args[args.index("-cq") + 1], "19")
        self.assertEqual(args[args.index("-b:v") + 1], "0")

    def test_openh264_bitrate_scales_with_size_and_is_clamped(self):
        small = int(encoders.codec_args("libopenh264", False, width=160, height=90, fps=24)[3])
        full = int(encoders.codec_args("libopenh264", False, width=1920, height=1080, fps=30)[3])
        huge = int(encoders.codec_args("libopenh264", False, width=7680, height=4320, fps=120)[3])
        self.assertEqual(small, 500_000)
        self.assertEqual(full, int(1920 * 1080 * 30 * 0.12))
        self.assertEqual(huge, 40_000_000)
        preview = int(encoders.codec_args("libopenh264", True, width=1920, height=1080, fps=30)[3])
        self.assertLess(preview, full)

    def test_unknown_encoder(self):
        with self.assertRaises(ValueError):
            encoders.codec_args("h264_mf", False)


class Lookup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "ffmpeg"
        self.root.mkdir()
        env = {"CHAOSENGINE_FFMPEG_DIR": str(self.root)}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("CHAOSENGINE_FFMPEG", None)
        self.imageio = Path(self.tmp.name) / "imageio-ffmpeg"
        self.imageio.write_text("x")

    def download(self, version="1.0"):
        folder = self.root / version
        folder.mkdir()
        exe = folder / ffmpeg_bin.binary_name()
        exe.write_text("x")
        (self.root / "current").write_text(version)
        return exe

    def find(self, **kw):
        with mock.patch.object(ffmpeg_bin, "_imageio_exe", return_value=str(self.imageio)):
            return ffmpeg_bin.find_ffmpeg(**kw)

    def test_order_env_custom_download_imageio(self):
        self.assertEqual(self.find(), (str(self.imageio), "imageio"))
        exe = self.download()
        self.assertEqual(self.find(), (str(exe), "download"))
        chosen = Path(self.tmp.name) / "mine"
        chosen.write_text("x")
        (self.root / "custom").write_text(str(chosen))
        self.assertEqual(self.find(), (str(chosen), "custom"))
        with mock.patch.dict(os.environ, {"CHAOSENGINE_FFMPEG": str(self.imageio)}):
            self.assertEqual(self.find(), (str(self.imageio), "env"))

    def test_missing_file_behind_pointer_falls_through(self):
        (self.root / "current").write_text("9.9")
        self.assertEqual(self.find()[1], "imageio")
        (self.root / "custom").write_text(str(Path(self.tmp.name) / "gone"))
        self.assertEqual(self.find()[1], "imageio")

    def test_pointer_cannot_escape_the_folder(self):
        for bad in ("../x", "..", "a/b", "a\\b", ".hidden"):
            (self.root / "current").write_text(bad)
            self.assertIsNone(ffmpeg_bin.downloaded_exe(), bad)

    def test_path_is_last_resort_and_opt_in(self):
        with mock.patch.object(ffmpeg_bin, "_imageio_exe", return_value=None), \
                mock.patch.object(ffmpeg_bin.shutil, "which", return_value="/usr/bin/ffmpeg"):
            self.assertIsNone(ffmpeg_bin.find_ffmpeg())
            self.assertEqual(ffmpeg_bin.find_ffmpeg(allow_path=True), ("/usr/bin/ffmpeg", "path"))
            self.assertIsNone(ffmpeg_bin.optional_ffmpeg_exe(allow_path=False))
            self.assertEqual(ffmpeg_bin.optional_ffmpeg_exe(), "/usr/bin/ffmpeg")

    def test_download_beats_path_and_imageio_beats_path(self):
        with mock.patch.object(ffmpeg_bin.shutil, "which", return_value="/usr/bin/ffmpeg"):
            self.assertEqual(self.find(allow_path=True)[1], "imageio")

    def test_none_raises_ffmpeg_missing_with_action(self):
        with mock.patch.object(ffmpeg_bin, "_imageio_exe", return_value=None):
            with self.assertRaises(ffmpeg_bin.FfmpegMissing) as ctx:
                ffmpeg_bin.ffmpeg_exe()
        self.assertEqual(ctx.exception.action, "install-ffmpeg")
        self.assertIsInstance(ctx.exception, RuntimeError)

    def test_media_helper_returns_none_instead_of_raising(self):
        from backend_service.storyboard import media

        with mock.patch.object(ffmpeg_bin, "_imageio_exe", return_value=None), \
                mock.patch.object(ffmpeg_bin.shutil, "which", return_value="/usr/bin/ffmpeg"):
            self.assertIsNone(media.ffmpeg_exe())

    def test_metadata_helper_may_use_path(self):
        from backend_service.helpers import output_metadata

        with mock.patch.object(ffmpeg_bin, "_imageio_exe", return_value=None), \
                mock.patch.object(ffmpeg_bin.shutil, "which", return_value="/usr/bin/ffmpeg"):
            self.assertEqual(output_metadata._ffmpeg_exe(), "/usr/bin/ffmpeg")


class FilterListing(unittest.TestCase):
    """``ffmpeg -filters`` has three flag columns up to ffmpeg 7 and two from ffmpeg 9; both must be read."""

    def names(self, listing: str) -> frozenset[str]:
        ffmpeg_bin._filters_for.cache_clear()
        try:
            with mock.patch.object(ffmpeg_bin.subprocess, "run", return_value=mock.Mock(stdout=listing)):
                return ffmpeg_bin._filters_for("ffmpeg-test")
        finally:
            ffmpeg_bin._filters_for.cache_clear()

    def test_three_flag_columns(self):
        listing = (
            "Filters:\n  T.. = Timeline support\n ------\n"
            " TSC adelay            A->A       Delay audio.\n"
            " ... scale             V->V       Scale.\n"
        )
        self.assertEqual(self.names(listing), frozenset({"adelay", "scale"}))

    def test_two_flag_columns(self):
        listing = (
            "Filters:\n  T.. = Timeline support\n  ------\n"
            " TS adelay            A->A       Delay audio.\n"
            " .. scale             V->V       Scale.\n"
        )
        self.assertEqual(self.names(listing), frozenset({"adelay", "scale"}))


class FilterScriptFlag(unittest.TestCase):
    """ffmpeg 9 removed ``-filter_complex_script``; the file-reading option follows the binary's major version."""

    def flag(self, version: str) -> str:
        from backend_service.storyboard import assemble

        with mock.patch.object(assemble.ffmpeg_bin, "ffmpeg_version", return_value=version):
            return assemble._filter_script_flag()

    def test_old_and_new_binaries(self):
        self.assertEqual(self.flag("6.1.1"), "-filter_complex_script")
        self.assertEqual(self.flag("7.1"), "-/filter_complex")
        self.assertEqual(self.flag("9.0.2"), "-/filter_complex")
        self.assertEqual(self.flag("unknown"), "-/filter_complex")


class RouteMapping(unittest.TestCase):
    def test_ffmpeg_missing_is_409_with_code(self):
        from backend_service.routes import storyboard as routes

        http = routes._http(ffmpeg_bin.FfmpegMissing())
        self.assertEqual(http.status_code, 409)
        self.assertEqual(http.detail["code"], "install-ffmpeg")
        self.assertIn("Video tools", http.detail["message"])


if __name__ == "__main__":
    unittest.main()
