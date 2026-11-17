"""Keep going after the 2026-11-17 rehearsal (FU-197, FU-198): what the nudge says, when the model counts as done,
what the chat already made, what "finished" needs, when the project is bound, and how a stall is explained.

The controller tests use fake hooks and a fake film; the scripted runs at the end drive the real chat path, the
real studio tools and a real Storyboard project with a stub model that writes the rehearsal's broken call."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend_service import keep_going
from backend_service.keep_going import adapters as kg_adapters
from backend_service.keep_going.hooks import AppHooks
from backend_service.keep_going.nudge import claims_done
from backend_service.keep_going.project_state import FilmState, _render_state
from backend_service.mcp_bridge import files
from tests.storyboard_fixtures import fake_wav
from tests.test_chat_studio_tools import call, result
from tests.test_keep_going_controller import SID, Base, done_film, film, tool
from tests.test_keep_going_integration import Harness, StubModel

REPLIES = json.loads((Path(__file__).parent / "fixtures" / "keep_going_qwen_unclosed_calls.json").read_text("utf-8"))["replies"]
UNREADABLE = ("Error: your tool call could not be read, so nothing ran: the JSON is missing 1 closing brace; "
              "send it again as <tool_call>...</tool_call>.")


def made(kind, path, at=None):
    return SimpleNamespace(id=f"job-{kind}", kind=kind, state="done", error=None, output={"path": str(path)},
                           created_at=time.time() if at is None else at)


class DoneClaimTests(unittest.TestCase):
    def test_a_claim_needs_a_final_done_statement_and_no_word_about_going_on(self):
        self.assertTrue(claims_done("The film is ready."))
        self.assertTrue(claims_done("I rendered the film to exports/ad.mp4. Everything is finished."))
        self.assertFalse(claims_done("Music is done. Now let me add it to the project and add the voice line."))
        self.assertFalse(claims_done("The film is ready. Next, I will add captions."))
        self.assertFalse(claims_done("The music is done. The voice line still needs work, and the captions too."))
        for text in REPLIES:  # every real reply tried a call: never a claim
            self.assertFalse(claims_done(text), text)


class NudgeWordingTests(Base):
    def test_step_limit(self):
        self.begin()
        self.finish(text="(Reached maximum tool-use iterations)", max_iterations=True,
                    calls=[tool("storyboard_shot", "old error"), tool("storyboard_project")])
        prompt = self.hooks.turns[0].prompt
        self.assertIn("Your last turn stopped at the step limit.", prompt)
        self.assertNotIn("tool error", prompt)
        self.assertNotIn("old error", prompt)

    def test_a_stale_error_is_never_repeated(self):
        # The rehearsal's nudge 4: 'terminal' failed mid-turn, then captions and a render worked.
        self.begin()
        self.finish(text="Rendered.", calls=[tool("terminal", "unknown tool 'terminal'"), tool("storyboard_captions"),
                                             tool("storyboard_render")])
        prompt = self.hooks.turns[0].prompt
        self.assertNotIn("terminal", prompt)
        self.assertNotIn("Fix that error first", prompt)
        self.assertNotIn("tool error", prompt)

    def test_the_last_result_an_error_is_repeated(self):
        self.begin()
        self.finish(calls=[tool("storyboard_project"), tool("storyboard_render", "no shots have clips")])
        prompt = self.hooks.turns[0].prompt
        self.assertIn("Your last turn ended after a tool error.", prompt)
        self.assertIn("Last error (storyboard_render): no shots have clips", prompt)
        self.assertIn("Fix that error first", prompt)

    def test_a_partial_done_is_not_called_a_claim_and_a_claim_is_not_accused(self):
        self.begin()
        self.finish(text="Music is done. Now let me add it.")
        self.assertNotIn("not finished yet", self.hooks.turns[0].prompt)
        self.film = film(revision=4)
        self.finish(text="The film is finished.")
        prompt = self.hooks.turns[1].prompt
        self.assertIn("The project is not finished yet.", prompt)
        self.assertNotIn("You said", prompt)

    def test_an_unreadable_call_in_the_last_reply(self):
        for text in REPLIES[:2]:
            with self.subTest(text[:30]):
                self.setUp()
                self.begin()
                self.finish(text=text, calls=[tool("make_music")])
                prompt = self.hooks.turns[0].prompt
                self.assertIn("Your last tool call (storyboard_audio) could not be read, so it did not run", prompt)
                self.assertIn("Send it again as a tool call: storyboard_audio.", prompt)
                self.assertNotIn("Fix that error first", prompt)
                self.assertNotIn("not finished yet", prompt)

    def test_an_unreadable_call_reported_by_the_loop(self):
        self.begin()
        self.finish(text="", calls=[{"name": "storyboard_audio", "result": UNREADABLE, "data": None}])
        prompt = self.hooks.turns[0].prompt
        self.assertIn("Your last tool call (storyboard_audio) could not be read, so it did not run: the JSON is "
                      "missing 1 closing brace.", prompt)


class AssetTests(Base):
    def setUp(self):
        super().setUp()
        self.dir = Path(self.tmp.name)
        self.music = fake_wav(self.dir / "music" / "track.wav")
        self.voice = fake_wav(self.dir / "voice" / "line.wav")

    def test_files_the_chat_made_are_named_and_attached_not_made_again(self):
        self.begin(brief="Make a 12 s ad with music.")
        self.hooks.jobs = [made("music", self.music), made("voice", self.voice), made("storyboard", self.dir / "x.mp4")]
        self.film = film(needs_clip=[])
        self.finish(text="Working.")
        prompt = self.hooks.turns[0].prompt
        line = next(x for x in prompt.splitlines() if x.startswith("Already made in this chat and not in the project yet: "))
        self.assertIn(f"music {self.music}", line)
        self.assertIn(f"voice {self.voice}", line)
        self.assertNotIn("x.mp4", line)  # storyboard jobs are already in the project
        self.assertIn("do not make it again", line)
        self.assertIn(f'storyboard_audio add with project_id p1, kind "music", path "{self.music}"', prompt)
        self.assertNotIn("make_music for the film's length", prompt)

    def test_attached_files_and_files_from_before_the_run_are_not_listed(self):
        from backend_service.storyboard.media import sha256_file

        self.begin()
        self.hooks.jobs = [made("music", self.music), made("voice", self.voice, at=1.0)]
        self.film = film(needs_clip=[], hashes={sha256_file(self.music)})
        self.finish()
        self.assertNotIn("Already made", self.hooks.turns[0].prompt)

    def test_voice_the_run_made_must_be_in_the_finished_film(self):
        self.begin(brief="Make a 12 s ad with music.")  # the brief never says voice
        self.hooks.jobs = [made("voice", self.voice)]
        self.film = done_film()
        self.finish(text="Here it is.")
        self.assertEqual(self.state().state, "running")
        self.assertIn("Still missing: voice", self.hooks.turns[0].prompt)
        self.film = done_film()
        self.film.voice = True
        self.finish(text="Here it is.")
        self.assertEqual(self.state().stop_code, "complete")


class FinishedFilmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        (self.folder / "cache" / "render").mkdir(parents=True)
        (self.folder / "exports").mkdir()

    def manifest(self, name, **data):
        (self.folder / "cache" / "render" / name).write_text(json.dumps(data), encoding="utf-8")

    def test_a_preview_is_never_the_finished_film(self):
        (self.folder / "exports" / "preview.mp4").write_bytes(b"x")
        self.manifest("manifest-preview.json", state="done", out="preview.mp4", finishedAt=time.time())
        self.assertEqual(_render_state(self.folder, 0.0)[0], "preview")
        # Even a manifest under the export name that says preview, or names preview.mp4, is not the film.
        self.manifest("manifest-export.json", state="done", out="preview.mp4", preview=True, finishedAt=time.time())
        self.assertEqual(_render_state(self.folder, 0.0)[0], "preview")
        (self.folder / "exports" / "Ad.mp4").write_bytes(b"x")
        self.manifest("manifest-export.json", state="done", out="Ad.mp4", preview=False, finishedAt=time.time())
        self.assertEqual(_render_state(self.folder, 0.0)[0], "done")

    def test_captions_must_show_in_the_film(self):
        state = done_film()
        state.captions = 3
        goal = "A 12 s ad with music and burned-in captions."
        self.assertFalse(state.complete(goal))
        self.assertIn("captions burned into the picture", state.lacks(goal))
        self.assertIn('"captionsBurn": true', state.next_step(goal))
        state.captions_burn = True
        self.assertTrue(state.complete(goal))
        srt = done_film()
        srt.captions, srt.captions_srt = 2, True
        self.assertTrue(srt.complete("Captions as an SRT file please."))
        self.assertFalse(srt.complete(goal))


class ProjectBindingTests(Base):
    def test_the_first_turn_binds_the_project_from_its_first_tool_result(self):
        self.begin()
        self.ctrl.on_tool_result(SID, {"name": "storyboard_project", "result": "{}", "data": {"projectId": "p9"}})
        self.assertEqual(self.state().project_id, "p9")
        self.ctrl.on_tool_result(SID, {"name": "storyboard_project", "result": "{}", "data": {"projectId": "p10"}})
        self.assertEqual(self.state().project_id, "p9")  # bound once; the turn end may still move it

    def test_an_error_or_an_armed_run_binds_nothing(self):
        self.ctrl.start(SID, app_request=object())
        self.ctrl.on_tool_result(SID, {"name": "storyboard_project", "data": {"projectId": "p9"}})
        self.assertIsNone(self.state().project_id)
        self.begin()
        self.ctrl.on_tool_result(SID, tool("storyboard_project", "bad", data={"projectId": "p9"}))
        self.assertIsNone(self.state().project_id)


class StallReasonTests(Base):
    def test_unreadable_calls_name_the_cause(self):
        self.begin()
        for _ in range(4):
            self.finish(text=REPLIES[0].replace("}}\n</tool_call>", ", }}\n</tool_call>"))
        run = self.state()
        self.assertEqual(run.stop_code, "calls_unreadable")
        self.assertIn("could not be read", run.stop_reason)
        self.assertIn("storyboard_audio", run.stop_reason)

    def test_the_loop_s_unreadable_errors_name_the_cause(self):
        self.begin()
        for rev in range(3):
            self.film = film(revision=10 + rev)  # progress each turn: only the same-error rule can stop it
            self.finish(text="", calls=[{"name": "storyboard_audio", "result": UNREADABLE, "data": None}] * 2)
        run = self.state()
        self.assertEqual(run.stop_code, "calls_unreadable")
        self.assertIn("could not be read", run.stop_reason)

    def test_the_same_refused_call_names_the_cause(self):
        self.begin()
        refused = {"name": "storyboard_audio", "arguments": {"action": "add", "revision": 17}, "data": {},
                   "result": "Error: The project is at revision 20 now."}
        for n in range(4):  # the error text moves, the call does not
            self.finish(calls=[dict(refused, result=f"Error: The project is at revision {20 + n} now.")])
        run = self.state()
        self.assertEqual(run.stop_code, "call_refused")
        self.assertIn("The same call was refused", run.stop_reason)
        self.assertIn("storyboard_audio", run.stop_reason)

    def test_mixed_causes_stay_plain_no_progress(self):
        self.begin()
        self.finish()
        self.finish(text=REPLIES[0].replace("}}\n</tool_call>", ", }}\n</tool_call>"))
        self.finish()
        self.finish()
        self.assertEqual(self.state().stop_code, "no_progress")


class ScriptedRecoveryRuns(Harness):
    """The rehearsal's stall, replayed: before FU-196 the broken call never ran and the run stopped on no progress."""

    def setUp(self):
        super().setUp()
        self.src = self.root / "outputs"
        self.track = fake_wav(self.src / "1794902867_ce532def.wav")
        roots = mock.patch.object(files, "output_roots", lambda: [p.resolve() for p in (self.src, *files._storyboard_exports())])
        roots.start()
        self.addCleanup(roots.stop)
        self.jobs = []
        chat_jobs = mock.patch.object(AppHooks, "chat_jobs", lambda _self: list(self.jobs))
        chat_jobs.start()
        self.addCleanup(chat_jobs.stop)

    def store(self):
        from backend_service.storyboard.store import default_store

        return default_store()

    def project(self):
        rows = self.store().list()
        return self.store().load(rows[0]["id"]) if rows else None

    def broken_call(self, fixable):
        """The rehearsal's reply for this project; ``fixable`` False adds a trailing comma no repair may touch."""
        p = self.project()
        text = REPLIES[1].replace("e818ea0a241c48d6", p.id).replace('"revision": 17', f'"revision": {p.revision}')
        text = text.replace("/Users/dan/.chaosengine/music/library/1794902867_ce532def.flac", str(self.track))
        return text if fixable else text.replace('"Background score"}}', '"Background score",}}')

    def run_film(self, script):
        self.state.runtime = StubModel(script)
        real_reader = kg_adapters.read_film_state

        def reader(pid):  # the film counts as finished once the music is in (shots and render are out of scope)
            state = real_reader(pid)
            if state is not None and state.music:
                return FilmState(project_id=state.project_id, name=state.name, revision=state.revision, shots=1,
                                 takes=1, music=True, render="done", render_finished_at=time.time())
            return state

        with mock.patch.object(kg_adapters, "read_film_state", reader):
            self.client.post(f"/api/chat/keepgoing/{self.sid}/start", json={"cap": 6, "grant": "auto", "hours": 1})
            self.send("Make a 12 second ad with music.")
            return self.wait_stopped()

    def brief_turn(self, model, history):
        """The brief: make the project, make the music (a finished chat job), then stop."""
        if not getattr(model, "created", False):
            model.created = True
            return result("", [call("storyboard_project", {"action": "create", "name": "Somnus"}, "c0")])
        self.jobs[:] = [made("music", self.track)]
        model.brief_done = True
        return result("The music is made.")

    def test_the_rehearsal_s_broken_call_is_mended_and_the_run_finishes(self):
        def script(model, history):
            if not getattr(model, "brief_done", False):
                return self.brief_turn(model, history)
            last = history[-1] if history else {}
            if last.get("role") == "tool":
                return result("Attached. The film is ready.")
            return result(self.broken_call(fixable=True))

        run = self.run_film(script)
        self.assertEqual(run["stopCode"], "complete", run)
        self.assertEqual([a.kind for a in self.project().audio], ["music"])
        nudge = next(m["text"] for m in self.messages() if m.get("keepGoing"))
        self.assertIn(f"music {self.track}", nudge)
        self.assertIn("do not make it again", nudge)

    def test_a_model_told_its_call_could_not_be_read_fixes_it(self):
        def script(model, history):
            if not getattr(model, "brief_done", False):
                return self.brief_turn(model, history)
            last = history[-1] if history else {}
            if last.get("role") == "tool":
                return result("Attached. The film is ready.")
            if last.get("role") == "user" and "could not be read" in str(last.get("text")):
                p = self.project()
                return result("", [call("storyboard_audio", {"action": "add", "project_id": p.id, "revision": p.revision,
                                                             "kind": "music", "path": str(self.track)}, "fix")])
            return result(self.broken_call(fixable=False))

        run = self.run_film(script)
        self.assertEqual(run["stopCode"], "complete", run)
        self.assertEqual([a.kind for a in self.project().audio], ["music"])

    def test_a_model_that_never_fixes_its_call_stops_with_that_reason(self):
        def script(model, history):
            if not getattr(model, "brief_done", False):
                return self.brief_turn(model, history)
            return result(self.broken_call(fixable=False))

        run = self.run_film(script)
        self.assertEqual(run["stopCode"], "calls_unreadable", run)
        self.assertIn("could not be read", run["stopReason"])
        self.assertEqual(self.project().audio, [])
        nudges = [m["text"] for m in self.messages() if m.get("keepGoing")]
        self.assertTrue(any("Your last tool call (storyboard_audio) could not be read" in t for t in nudges[1:]), nudges)


if __name__ == "__main__":
    unittest.main()
