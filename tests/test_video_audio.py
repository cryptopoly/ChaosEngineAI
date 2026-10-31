import subprocess
import tempfile
import unittest
from pathlib import Path

from backend_service.helpers.video_artifacts import _hydrate_video_artifact
from backend_service.helpers.video_audio import mp4_has_audio_bytes, mp4_has_audio_file


def _box(kind: bytes, payload: bytes) -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


def _track(handler: bytes) -> bytes:
    hdlr = _box(b"hdlr", b"\x00" * 4 + b"\x00" * 4 + handler + b"\x00" * 12)
    return _box(b"trak", _box(b"mdia", hdlr))


def _mp4(*handlers: bytes) -> bytes:
    ftyp = _box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
    mdat = _box(b"mdat", b"soun" * 64)  # the word inside media data must not fool the scan
    moov = _box(b"moov", b"".join(_track(h) for h in handlers))
    return ftyp + mdat + moov


class Mp4AudioTests(unittest.TestCase):
    def test_a_sound_track_is_found_after_the_media_data(self) -> None:
        self.assertTrue(mp4_has_audio_bytes(_mp4(b"vide", b"soun")))

    def test_a_video_only_file_has_no_sound(self) -> None:
        self.assertFalse(mp4_has_audio_bytes(_mp4(b"vide")))

    def test_the_word_in_media_data_is_not_a_track(self) -> None:
        self.assertFalse(mp4_has_audio_bytes(_box(b"ftyp", b"isom") + _box(b"mdat", b"soun" * 100)))

    def test_garbage_and_truncated_data_say_no(self) -> None:
        self.assertFalse(mp4_has_audio_bytes(b""))
        self.assertFalse(mp4_has_audio_bytes(b"not a video at all, just text"))
        self.assertFalse(mp4_has_audio_bytes(_mp4(b"vide", b"soun")[:-20]))

    def test_a_box_that_claims_a_huge_size_does_not_hang(self) -> None:
        self.assertFalse(mp4_has_audio_bytes((0xFFFFFFFF).to_bytes(4, "big") + b"moov" + b"\x00" * 16))

    def test_the_file_reader_matches_the_bytes_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sound, silent = Path(tmp) / "a.mp4", Path(tmp) / "b.mp4"
            sound.write_bytes(_mp4(b"vide", b"soun"))
            silent.write_bytes(_mp4(b"vide"))
            self.assertTrue(mp4_has_audio_file(sound))
            self.assertFalse(mp4_has_audio_file(silent))
            self.assertFalse(mp4_has_audio_file(Path(tmp) / "missing.mp4"))

    def test_an_older_saved_clip_without_the_field_is_read_from_its_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "old.mp4"
            clip.write_bytes(_mp4(b"vide", b"soun"))
            hydrated = _hydrate_video_artifact({"artifactId": "x", "videoPath": str(clip)})
            self.assertTrue(hydrated["hasAudio"])
            self.assertFalse(_hydrate_video_artifact({"artifactId": "x"})["hasAudio"])
            self.assertFalse(_hydrate_video_artifact({"artifactId": "x", "videoPath": str(clip), "hasAudio": False})["hasAudio"])

    def test_real_ffmpeg_files(self) -> None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            self.skipTest("imageio-ffmpeg not installed")
        with tempfile.TemporaryDirectory() as tmp:
            silent, sound = Path(tmp) / "silent.mp4", Path(tmp) / "sound.mp4"
            video = ["-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1"]
            run = lambda args: subprocess.run([exe, "-v", "error", "-y", *args], check=True, capture_output=True)  # noqa: E731
            run([*video, "-pix_fmt", "yuv420p", str(silent)])
            run([*video, "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(sound)])
            self.assertFalse(mp4_has_audio_file(silent))
            self.assertTrue(mp4_has_audio_file(sound))
            self.assertTrue(mp4_has_audio_bytes(sound.read_bytes()))


if __name__ == "__main__":
    unittest.main()
