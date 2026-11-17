"""What the Storyboard tools tell an agent (FU-200 to FU-203): the size, rate and models a project got, cast entries
that need a name and a picture, render warnings (silent film, unburned captions, a preview), and audio that landed."""

import unittest
from pathlib import Path
from unittest import mock

from tests import test_storyboard_tools as tool_tests
from backend_service.mcp_bridge.storyboard_tools import STORYBOARD_TOOLS
from tests.storyboard_fixtures import fake_mp4, fake_wav, png


def _schema(name):
    return next(t for t in STORYBOARD_TOOLS if t["name"] == name)["inputSchema"]["properties"]


class StoryboardGuidanceTests(unittest.TestCase):
    setUp = tool_tests.StoryboardToolTests.setUp
    _inbox = tool_tests.StoryboardToolTests._inbox
    call = tool_tests.StoryboardToolTests.call
    project = tool_tests.StoryboardToolTests.project

    def get(self, pid):
        return self.call("storyboard_project", {"action": "get", "project_id": pid})["project"]

    # FU-200
    def test_create_takes_an_exact_size_models_and_voice_and_echoes_them(self):
        info = self.call("storyboard_project", {"action": "create", "name": "Film", "size": "1024x576", "fps": 24,
                                                "image_model": "img-x", "video_model": "vid-y", "voice": "af_heart"})
        settings = info["settings"]
        self.assertEqual((settings["size"], settings["aspect"], settings["fps"]), ("1024x576", "16:9", 24))
        self.assertEqual((settings["imageModel"], settings["videoModel"], settings["voice"]), ("img-x", "vid-y", "af_heart"))
        self.assertFalse(settings["captionsBurn"])
        notes = " ".join(info["notes"])
        self.assertIn("'img-x' is not an installed image model", notes)  # the test machine has none installed
        self.assertIn("captionsBurn", notes)
        self.assertIn("make_music", notes)
        self.assertNotIn("multiples of 64", notes)  # 1024x576 is already LTX-sized
        defaults = self.get(info["projectId"])["defaults"]
        self.assertEqual((defaults["width"], defaults["height"], defaults["imageModel"], defaults["videoModel"]),
                         (1024, 576, "img-x", "vid-y"))

    def test_a_size_that_is_not_a_multiple_of_64_says_what_clips_are_made_at(self):
        info = self.call("storyboard_project", {"action": "create", "name": "Default"})
        self.assertEqual(info["settings"]["size"], "1280x720")
        self.assertIn("1280x704", " ".join(info["notes"]))
        self.assertIn("imageModel", " ".join(info["notes"]))

    def test_edit_sets_size_and_portrait_aspect_and_echoes_settings(self):
        info = self.call("storyboard_project", {"action": "create", "name": "Tall"})
        info = self.call("storyboard_project", {"action": "edit", "project_id": info["projectId"], "revision": info["revision"],
                                                "size": "576x1024", "fields": {"export": {"captionsBurn": True}}})
        self.assertEqual((info["settings"]["size"], info["settings"]["aspect"], info["settings"]["captionsBurn"]),
                         ("576x1024", "9:16", True))
        renamed = self.call("storyboard_project", {"action": "edit", "project_id": info["projectId"], "revision": info["revision"],
                                                   "fields": {"name": "Tall one"}})
        self.assertNotIn("settings", renamed)  # only format, model or export edits echo them

    def test_bad_sizes_are_readable_errors(self):
        for size, needle in (("big", "WIDTHxHEIGHT"), ("1025x576", "even"), ("32x32", "64 to 4096")):
            self.assertIn(needle, self.call("storyboard_project", {"action": "create", "name": "x", "size": size}, error=True))
        self.assertIn("does not fit aspect", self.call("storyboard_project", {"action": "create", "name": "x", "size": "1024x576",
                                                                              "aspect": "9:16"}, error=True))

    def test_schema_names_the_fields_for_size_models_and_export(self):
        props = _schema("storyboard_project")
        for key in ("size", "image_model", "video_model", "voice", "fps"):
            self.assertIn(key, props)
        self.assertIn("captionsBurn", props["fields"]["description"])
        self.assertIn("imageModel", props["fields"]["description"])

    # FU-201
    def test_add_ref_needs_a_name_and_warns_without_a_picture(self):
        info = self.project(shots=0)
        pid = info["projectId"]
        text = self.call("storyboard_project", {"action": "add_ref", "project_id": pid, "revision": info["revision"],
                                                "kind": "character"}, error=True)
        self.assertIn("needs a name", text)
        self.assertIn("name", self.call("storyboard_project", {"action": "add_ref", "project_id": pid, "revision": info["revision"],
                                                               "kind": "character", "name": "  "}, error=True))
        info = self.call("storyboard_project", {"action": "add_ref", "project_id": pid, "revision": info["revision"],
                                                "kind": "character", "name": "Mara"})
        self.assertIn("no picture", info["warning"])
        self.assertIn(info["refId"], info["next"])
        info = self.call("storyboard_project", {"action": "add_ref", "project_id": pid, "revision": info["revision"],
                                                "kind": "character", "name": "Ivo", "images": [str(png(self.src / "ivo.png"))]})
        self.assertNotIn("warning", info)
        self.assertIn("generate_image", _schema("storyboard_project")["images"]["description"])

    # FU-202
    def test_render_warns_about_silence_unburned_captions_and_a_preview(self):
        info = self.project(shots=1)
        pid = info["projectId"]
        info = self.call("storyboard_shot", {"action": "import", "project_id": pid, "revision": info["revision"],
                                             "shot_id": info["shots"][0]["id"], "path": str(fake_mp4(self.src / "c.mp4"))})
        info = self.call("storyboard_captions", {"action": "add", "project_id": pid, "revision": info["revision"],
                                                 "start": 0, "end": 1, "text": "Hello"})

        def fake_render(plan, progress, cancel_event):
            Path(plan["out_path"]).write_bytes(b"film")

        with mock.patch("backend_service.storyboard.render_adapter.render", fake_render):
            job = self.call("storyboard_render", {"project_id": pid, "revision": info["revision"]})
            text = " ".join(job["warnings"])
            for needle in ("no audio lanes", "captionsBurn is off", "preview"):
                self.assertIn(needle, text)
            done = self.call("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 10})
            self.assertTrue(done["output"]["preview"])
            self.assertIn("preview: false", " ".join(done["output"]["warnings"]))
            info = self.call("storyboard_project", {"action": "edit", "project_id": pid, "revision": info["revision"],
                                                    "fields": {"export": {"captionsBurn": True}}})
            info = self.call("storyboard_audio", {"action": "add", "project_id": pid, "revision": info["revision"],
                                                  "kind": "music", "path": str(fake_wav(self.src / "m.wav"))})
            job = self.call("storyboard_render", {"project_id": pid, "revision": info["revision"], "preview": False})
            self.assertNotIn("warnings", job)
            self.call("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 10})
        self.assertIn("preview: false for the final MP4", _schema("storyboard_render")["preview"]["description"])

    # FU-203
    def test_audio_add_shows_the_lanes_and_takes_a_library_id(self):
        from backend_service import music_library

        info = self.project(shots=1)
        pid = info["projectId"]
        info = self.call("storyboard_audio", {"action": "add", "project_id": pid, "revision": info["revision"], "kind": "voice",
                                              "path": str(fake_wav(self.src / "v.wav"))})
        self.assertEqual(info["lanes"], {"voice": 1})
        self.assertIn("one of them", self.call("storyboard_audio", {"action": "add", "project_id": pid,
                                                                     "revision": info["revision"], "kind": "music"}, error=True))
        track = fake_wav(self.src / "track.wav")
        with mock.patch.object(music_library, "is_valid_id", return_value=True), \
                mock.patch.object(music_library, "audio_path", return_value=track):
            info = self.call("storyboard_audio", {"action": "add", "project_id": pid, "revision": info["revision"],
                                                  "kind": "music", "library_id": "abc123"})
        self.assertEqual(info["lanes"], {"voice": 1, "music": 1})
        self.assertIn("No Music library track", self.call("storyboard_audio", {"action": "add", "project_id": pid,
                                                                                "revision": info["revision"], "kind": "music",
                                                                                "library_id": "nope"}, error=True))
        self.assertIn("make_music", _schema("storyboard_audio")["path"]["description"])

    # FU-209
    def _rendered_project(self):
        info = self.project(shots=1)
        pid = info["projectId"]
        info = self.call("storyboard_shot", {"action": "import", "project_id": pid, "revision": info["revision"],
                                             "shot_id": info["shots"][0]["id"], "path": str(fake_mp4(self.src / "c.mp4"))})
        return pid, info

    def test_render_in_chat_carries_the_exact_path_and_the_bridge_does_not(self):
        from backend_service.mcp_bridge.jobs import CURRENT_OWNER, STORE
        from backend_service.mcp_bridge.storyboard_tools import storyboard_render

        pid, info = self._rendered_project()

        def fake_render(plan, progress, cancel_event):
            Path(plan["out_path"]).write_bytes(b"film")

        with mock.patch("backend_service.storyboard.render_adapter.render", fake_render):
            token = CURRENT_OWNER.set("chat")
            try:
                submitted = storyboard_render(None, {"project_id": pid, "revision": info["revision"], "preview": False})
            finally:
                CURRENT_OWNER.reset(token)
            job = STORE.get(submitted["jobId"])
            self.assertTrue(job.done.wait(10))
        path = Path(job.output["path"])
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.parent, (self.base / "storyboards" / pid / "exports").resolve())
        self.assertEqual(job.notes[0], f"The film is saved at {path}. Tell the user this exact path.")
        with mock.patch("backend_service.storyboard.render_adapter.render", fake_render):
            job = self.call("storyboard_render", {"project_id": pid, "revision": info["revision"], "preview": False})
            done = self.call("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 10})
        self.assertNotIn("path", done["output"])
        self.assertFalse(any("exact path" in n for n in done["notes"]))

    def test_render_warns_when_burn_in_is_on_but_there_are_no_captions(self):
        pid, info = self._rendered_project()
        info = self.call("storyboard_project", {"action": "edit", "project_id": pid, "revision": info["revision"],
                                                "fields": {"export": {"captionsBurn": True}}})

        def fake_render(plan, progress, cancel_event):
            Path(plan["out_path"]).write_bytes(b"film")

        with mock.patch("backend_service.storyboard.render_adapter.render", fake_render):
            job = self.call("storyboard_render", {"project_id": pid, "revision": info["revision"]})
            self.call("wait_for_job", {"job_id": job["jobId"], "timeout_seconds": 10})
        self.assertIn("captions burn-in is on but the project has no captions: add them with storyboard_captions first",
                      " ".join(job["warnings"]).lower())

    def test_captions_generate_with_shot_ids_names_the_audio_clips(self):
        info = self.project(shots=2)
        pid = info["projectId"]
        info = self.call("storyboard_audio", {"action": "add", "project_id": pid, "revision": info["revision"],
                                              "kind": "voice", "path": str(fake_wav(self.src / "v.wav"))})
        voice = self.get(pid)["audio"][0]["id"]
        shots = [s["id"] for s in self.get(pid)["shots"]]
        error = self.call("storyboard_captions", {"action": "generate", "project_id": pid, "revision": info["revision"],
                                                  "clip_ids": shots}, error=True)
        self.assertIn(f"Unknown audio clip '{shots[0]}'", error)
        self.assertIn(f"voice: {voice}", error)
        self.assertIn("Omit clip_ids to caption every voice lane.", error)


if __name__ == "__main__":
    unittest.main()
