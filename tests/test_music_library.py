"""Music Library (id safety, CRUD, tags) and the FLAC Vorbis-comment writer (round trip, no lyrics or paths)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service import music_env, music_library
from backend_service.helpers.audio_metadata import (
    embed_flac_tags,
    music_tag_settings,
    read_audio_settings,
    read_flac_tags,
)
from tests.music_fakes import make_flac


class LibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patch = mock.patch.object(music_env, "MUSIC_DATA_DIR", self.root / "music")
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()
        self.tmp.cleanup()

    def _audio(self, name: str = "a.flac") -> Path:
        path = self.root / name
        path.write_bytes(b"fLaC" + b"\0" * 100)
        return path

    def test_create_get_list_rename_tag_delete(self) -> None:
        item = music_library.create({"caption": "warm lo-fi piano", "lyrics": "la la", "seed": 3, "tags": ["Lo-Fi", "lo-fi", " piano "]},
                                    self._audio())
        self.assertTrue(music_library.is_valid_id(item["id"]))
        self.assertEqual(item["title"], "warm lo-fi piano")
        self.assertEqual(item["tags"], ["Lo-Fi", "piano"])
        self.assertEqual(item["sizeBytes"], 104)
        self.assertEqual(music_library.audio_path(item["id"]).parent, music_library.library_dir())
        second = music_library.create({"caption": "orchestral", "title": "Big"}, self._audio("b.flac"))
        listed = music_library.list_items()
        self.assertCountEqual([i["id"] for i in listed], [second["id"], item["id"]])
        self.assertGreaterEqual(listed[0]["createdAt"], listed[1]["createdAt"])  # newest first
        self.assertEqual([i["id"] for i in music_library.list_items("piano")], [item["id"]])
        self.assertEqual([i["id"] for i in music_library.list_items(tag="LO-FI")], [item["id"]])
        renamed = music_library.update(item["id"], title="  Rainy   day ", tags=["rain", "x" * 41, "\x01bad"])
        self.assertEqual((renamed["title"], renamed["tags"]), ("Rainy day", ["rain"]))
        self.assertTrue(music_library.delete(item["id"]))
        self.assertIsNone(music_library.get(item["id"]))
        self.assertFalse(music_library.delete(item["id"]))
        self.assertGreater(music_library.disk_bytes(), 0)

    def test_ids_from_outside_are_never_trusted(self) -> None:
        for bad in ("../../etc/passwd", "1_abc", "123_0123456g", "123_01234567\n", "", "123_01234567/../x"):
            self.assertFalse(music_library.is_valid_id(bad), bad)
            self.assertIsNone(music_library.get(bad))
            self.assertIsNone(music_library.audio_path(bad))
            self.assertFalse(music_library.delete(bad))
            self.assertIsNone(music_library.update(bad, title="x"))

    def test_only_known_audio_types_and_unknown_files_ignored(self) -> None:
        with self.assertRaises(ValueError):
            music_library.create({"caption": "x"}, self._audio("c.exe"), ".exe")
        (music_library.library_dir() / "notes.json").write_text("{}")
        (music_library.library_dir() / "123_01234567.json").write_text('{"id": "999_01234567"}')  # id mismatch
        self.assertEqual(music_library.list_items(), [])


class FlacTagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.flac = make_flac(Path(self.tmp.name) / "t.flac")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_round_trip_keeps_the_audio_readable(self) -> None:
        settings = {"appVersion": "1.2.3", "model": "ACE-Step/Ace-Step1.5", "seed": 11, "durationSeconds": 1.0,
                    "bpm": 110, "key": "G major", "prompt": "indie pop, fünf", "lyrics": "secret words",
                    "path": "/Users/someone/x.flac"}
        self.assertTrue(embed_flac_tags(self.flac, settings, "ChaosEngineAI 1.2.3 / ACE-Step 1.5"))
        tags = read_flac_tags(self.flac)
        self.assertEqual(tags["ENCODER"], "ChaosEngineAI 1.2.3 / ACE-Step 1.5")
        self.assertTrue(tags["COMMENT"].startswith("chaosengine:{"))
        read = read_audio_settings(self.flac)
        self.assertEqual(read["seed"], 11)
        self.assertEqual(read["prompt"], "indie pop, fünf")
        self.assertIs(read["machine_generated"], True)
        self.assertNotIn("lyrics", read)
        self.assertNotIn("secret words", str(tags))
        self.assertNotIn("/Users/", str(tags))
        # Writing twice replaces the block, it does not stack.
        self.assertTrue(embed_flac_tags(self.flac, {"seed": 12}, "e"))
        self.assertEqual(read_audio_settings(self.flac)["seed"], 12)
        # The pinned ffmpeg still decodes it and sees the same tags.
        import subprocess

        from backend_service.storyboard.ffmpeg_bin import ffmpeg_exe

        result = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(self.flac), "-f", "null", "-"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-400:])
        self.assertIn("COMMENT", result.stderr)
        self.assertIn("24 bit", result.stderr)

    def test_non_flac_and_missing_files_fail_softly(self) -> None:
        other = Path(self.tmp.name) / "x.wav"
        other.write_bytes(b"RIFF....")
        self.assertFalse(embed_flac_tags(other, {"seed": 1}, "e"))
        self.assertEqual(other.read_bytes(), b"RIFF....")
        self.assertFalse(embed_flac_tags(Path(self.tmp.name) / "missing.flac", {}, "e"))
        self.assertEqual(read_flac_tags(other), {})
        self.assertIsNone(read_audio_settings(other))

    def test_tag_settings_whitelist(self) -> None:
        self.assertEqual(music_tag_settings({"seed": 1, "lyrics": "x", "bpm": None}), {"seed": 1, "machine_generated": True})


if __name__ == "__main__":
    unittest.main()
