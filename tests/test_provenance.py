import json
import tempfile
import unittest
from pathlib import Path

from backend_service.helpers.image_artifacts import _load_image_outputs, _save_image_artifact
from backend_service.helpers.output_metadata import generation_settings, read_png_settings
from backend_service.helpers.provenance import provenance, stamp
from tests.test_output_metadata import _tiny_png


def _artifact(artifact_id: str) -> dict:
    return {"artifactId": artifact_id, "modelId": "m", "modelName": "M", "prompt": "p", "width": 8, "height": 8,
            "seed": 1, "createdAt": "2026-11-08T00:00:00Z", "imageBytes": _tiny_png(), "imageExtension": "png"}


class ProvenanceTests(unittest.TestCase):
    def test_stamp_is_a_no_op_outside_a_run(self):
        self.assertEqual(stamp({"a": 1}), {"a": 1})

    def test_stamp_inside_and_after(self):
        with provenance("proj-1", "Harbour", "s2"):
            self.assertEqual(stamp({})["shotId"], "s2")
        self.assertEqual(stamp({}), {})

    def test_saved_still_carries_project_in_file_json_and_gallery_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with provenance("proj-1", "Harbour", "s2"):
                saved = _save_image_artifact(_artifact("a1"), folder)
            _save_image_artifact(_artifact("a2"), folder)  # an ordinary Studio image
            self.assertEqual((saved["projectId"], saved["projectName"], saved["shotId"]), ("proj-1", "Harbour", "s2"))
            embedded = read_png_settings(Path(saved["imagePath"]).read_bytes())
            self.assertEqual(embedded["projectName"], "Harbour")
            self.assertEqual(embedded["shotId"], "s2")
            self.assertNotIn(tmp, json.dumps(embedded))
            rows = {r["artifactId"]: r for r in _load_image_outputs(folder)}
            self.assertEqual(rows["a1"]["projectId"], "proj-1")
            self.assertIsNone(rows["a2"]["projectId"])

    def test_settings_keys_only_when_present(self):
        self.assertNotIn("projectId", generation_settings({"prompt": "x"}))


class AudioLibraryProvenanceTests(unittest.TestCase):
    def test_music_and_voice_items_carry_the_project_and_the_bridge_scope_sets_it(self):
        from unittest import mock

        from backend_service import music_env, music_library, voice_env, voice_library
        from backend_service.mcp_bridge.tools import ToolError, project_scope

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def flac(name):
                path = root / name
                path.write_bytes(b"fLaC" + b"\0" * 8)
                return path

            wav = root / "a.wav"
            wav.write_bytes(b"RIFF")
            with mock.patch.object(music_env, "MUSIC_DATA_DIR", root / "music"), \
                    mock.patch.object(voice_env, "VOICE_DATA_DIR", root / "voice"):
                with project_scope({"project_id": "proj-9", "project_name": "Alpine"}):
                    track = music_library.create({"caption": "calm"}, flac("a.flac"))
                    speech = voice_library.create("speech", {"text": "hi"}, audio_file=wav, audio_ext=".wav")
                self.assertEqual((track["projectId"], track["projectName"]), ("proj-9", "Alpine"))
                self.assertNotIn("shotId", track)
                self.assertEqual(voice_library.list_items(kind="speech")[0]["projectId"], "proj-9")
                plain = music_library.create({"caption": "x"}, flac("b.flac"))
                self.assertNotIn("projectId", plain)
                self.assertNotIn("projectId", music_library.get(plain["id"]))
                self.assertEqual(speech["projectName"], "Alpine")
            with project_scope({}):
                self.assertEqual(stamp({}), {})
            with self.assertRaises(ToolError):
                project_scope({"project_name": "Orphan"})
