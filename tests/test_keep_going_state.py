"""Keep going (FU-191): reading a Storyboard project, the nudge text, the adapters and the gate's grant end."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from backend_service.keep_going.adapters import GenericAdapter, StoryboardAdapter, stated_next_step
from backend_service.keep_going.nudge import (
    TurnOutcome,
    claims_done,
    nudge_text,
    plan_outcome,
    project_id_from,
    unfixed_errors,
)
from backend_service.keep_going.project_state import FilmState, film_state, target_seconds, wants_captions, wants_voice
from backend_service.storyboard.models import AudioClip, AudioSource, Caption, Project, Shot, Take
from backend_service.tools.studio_gate import StudioGate


def shot(sid, order, take=None, **kw):
    takes = [take] if take else []
    return Shot(id=sid, order=order, takes=takes, activeTake=take.id if take else None, **kw)


def take(tid, still=True, clip=True, at=100.0):
    return Take(id=tid, still=f"media/{tid}.png" if still else None, clip=f"media/{tid}.mp4" if clip else None,
                createdAt=at)


class FilmStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        (self.folder / "exports").mkdir()
        (self.folder / "cache" / "render").mkdir(parents=True)
        (self.folder / "project.json").write_text("{}")
        old = time.time() - 600
        os.utime(self.folder / "project.json", (old, old))

    def tearDown(self):
        self.tmp.cleanup()

    def project(self, shots, audio=(), captions=()):
        return Project(id="0123456789abcdef", name="Bakery ad", revision=7, shots=list(shots), audio=list(audio),
                       captions=list(captions))

    def render(self, state="done", finished=None, out="film.mp4", preview=False, write=True):
        name = "manifest-preview.json" if preview else "manifest-export.json"
        (self.folder / "cache" / "render" / name).write_text(json.dumps(
            {"state": state, "finishedAt": finished if finished is not None else time.time(), "out": out}))
        if write and state == "done" and not preview:
            (self.folder / "exports" / out).write_bytes(b"mp4")

    def test_missing_pieces_in_film_order(self):
        p = self.project([
            shot("s3", 2, take("t3", clip=False)),
            shot("s1", 0, take("t1")),
            shot("s2", 1),
            shot("s4", 3, take("t4"), status="stale"),
            shot("s5", 4, startFrom="text"),
        ])
        f = film_state(p, self.folder)
        self.assertEqual((f.needs_still, f.needs_clip, f.needs_redo), ([2], [3, 5], [4]))
        self.assertEqual(f.render, "none")
        self.assertFalse(f.complete(""))
        self.assertIn('what "still" (shots 2)', f.next_step(""))
        summary = f.summary("an ad with a voiceover and captions")
        for piece in ("still for shot 2", "clip for shot 3, 5", "music", "voice", "captions", "the final render"):
            self.assertIn(piece, summary)

    def test_complete_needs_clips_music_and_a_current_final_render(self):
        music = AudioClip(id="a1", kind="music", source=AudioSource(path="media/audio/m.wav"))
        p = self.project([shot("s1", 0, take("t1")), shot("s2", 1, take("t2"))], audio=[music])
        self.render(preview=True)
        self.assertEqual(film_state(p, self.folder).render, "preview")
        self.render()
        f = film_state(p, self.folder)
        self.assertEqual(f.render, "done")
        self.assertTrue(f.complete("an ad"))
        self.assertFalse(f.complete("an ad with narration"))  # the brief asks for a voice lane
        self.assertIn("speak the voiceover", f.next_step("an ad with narration"))

    def test_a_render_older_than_the_last_change_is_stale(self):
        music = AudioClip(id="a1", kind="music", source=AudioSource(path="media/audio/m.wav"))
        p = self.project([shot("s1", 0, take("t1", at=time.time()))], audio=[music])
        self.render(finished=time.time() - 300)
        f = film_state(p, self.folder)
        self.assertEqual(f.render, "stale")
        self.assertIn("preview false", f.next_step(""))

    def test_a_failed_or_missing_render_file(self):
        p = self.project([shot("s1", 0, take("t1"))])
        self.render(state="failed")
        self.assertEqual(film_state(p, self.folder).render, "failed")
        self.render(write=False)
        self.assertEqual(film_state(p, self.folder).render, "none")

    def test_captions_and_snapshot(self):
        p = self.project([shot("s1", 0, take("t1"))], captions=[Caption(id="c1", start=0, end=1, text="Hi")])
        f = film_state(p, self.folder)
        self.assertEqual(f.captions, 1)
        self.assertEqual(f.snapshot()["revision"], 7)

    def test_brief_words(self):
        self.assertTrue(wants_voice("with a warm voice-over"))
        self.assertTrue(wants_captions("burn in subtitles"))
        self.assertFalse(wants_voice("music only"))

    def test_no_shots(self):
        f = FilmState(project_id="p", name="n", revision=1, shots=0)
        self.assertIn("storyboard_shot add", f.next_step(""))
        self.assertFalse(f.complete(""))

    def test_batches(self):
        f = FilmState(project_id="p", name="n", revision=1, shots=2, needs_clip=[1], batch_paused="b1")
        self.assertIn("storyboard_batch action resume", f.next_step(""))


def outcome(text="", calls=None, plans=None, **kw):
    assistant = {"text": text, "toolCalls": calls or []}
    if plans:
        assistant["studioPlans"] = plans
    return TurnOutcome(session_id="s", assistant=assistant, **kw)


class NudgeTests(unittest.TestCase):
    def test_errors_and_fixes(self):
        calls = [{"name": "a", "result": "Error: one"}, {"name": "b", "data": {"error": "two"}},
                 {"name": "a", "result": "{}"}, {"name": "c", "data": {"declined": True, "error": "x"}}]
        self.assertEqual(unfixed_errors(outcome(calls=calls)), [("b", "two")])

    def test_plans(self):
        self.assertEqual(plan_outcome(outcome(plans=[{"status": "decided", "decisions": {"x": "deny"}}])), "declined")
        self.assertIsNone(plan_outcome(outcome(plans=[{"status": "decided", "decisions": {"x": "approve"}}])))
        self.assertEqual(plan_outcome(outcome(plans=[{"status": "unanswered"}])), "unanswered")

    def test_project_id(self):
        calls = [{"name": "storyboard_project", "data": {"projectId": "p_one"}},
                 {"name": "storyboard_shot", "arguments": {"project_id": "p_two"}, "data": {}},
                 {"name": "storyboard_shot", "arguments": {"project_id": "p_bad"}, "data": {"error": "nope"}}]
        self.assertEqual(project_id_from(outcome(calls=calls)), "p_two")

    def test_done_words(self):
        self.assertTrue(claims_done("The film is ready."))
        self.assertTrue(claims_done("All done!"))
        self.assertFalse(claims_done("I will render it next."))

    def test_text_is_short_and_concrete(self):
        text = nudge_text(number=3, cap=20, goal="x" * 5000, include_goal=False, summary="Project p: 4 shots.",
                          next_step="Do A.", last_error=("t", "e" * 900), finished_jobs=["image job j1 done"],
                          why="stopped")
        self.assertTrue(text.startswith("Keep going (automatic message 3 of 20"))
        self.assertLess(len(text), 900)
        self.assertIn("Next step: Fix that error first, then: do A.", text)
        full = nudge_text(number=1, cap=5, goal="brief " * 1000, include_goal=True, summary="", next_step="",
                          last_error=None, finished_jobs=[], why="resume")
        self.assertIn("The brief, as the user gave it", full)
        self.assertLess(len(full), 4600)


class AdapterTests(unittest.TestCase):
    def test_storyboard_adapter(self):
        film = FilmState(project_id="p", name="n", revision=2, shots=1, music=True, render="done")
        view = StoryboardAdapter(reader=lambda pid: film).read(goal="", project_id="p", outcome=None, last_reply="")
        self.assertEqual(view.complete, "complete")
        view = StoryboardAdapter(reader=lambda pid: None).read(goal="", project_id=None, outcome=outcome("All done."),
                                                              last_reply="")
        self.assertEqual(view.complete, "complete_no_project")
        self.assertIn("storyboard_project create", view.next_step)

    def test_generic_adapter(self):
        a = GenericAdapter()
        busy = a.read(goal="g", project_id=None, outcome=outcome("Next I will write the tests.",
                                                                  calls=[{"name": "code", "result": "1"}]),
                      last_reply="")
        same = a.read(goal="g", project_id=None, outcome=outcome("", calls=[{"name": "code", "result": "1"}]),
                      last_reply="")
        other = a.read(goal="g", project_id=None, outcome=outcome("", calls=[{"name": "code", "result": "2"}]),
                       last_reply="")
        self.assertEqual(busy.signature, same.signature)
        self.assertNotEqual(busy.signature, other.signature)
        self.assertIn("Next I will write the tests", busy.next_step)
        self.assertIsNone(busy.complete)
        done = a.read(goal="g", project_id=None, outcome=outcome("Everything is done."), last_reply="")
        self.assertEqual(done.complete, "complete_no_project")
        tool_turn = a.read(goal="g", project_id=None, outcome=outcome("All done.", calls=[{"name": "x"}]), last_reply="")
        self.assertIsNone(tool_turn.complete)  # a tool call in the turn: not done yet

    def test_stated_next_step(self):
        self.assertEqual(stated_next_step("Made 3 images. Next, I will add music."), "Next, I will add music.")
        self.assertIsNone(stated_next_step("Here you go."))


class GateGrantTests(unittest.TestCase):
    def call(self):
        return {"id": "c1", "type": "function", "function": {"name": "generate_image", "arguments": '{"prompt": "rose"}'}}

    def run_review(self, gate):
        review = gate.review([self.call()])
        try:
            first = next(review)
        except StopIteration as done:
            return "auto", done.value
        review.close()
        return "card", first

    def test_auto_until_in_the_future_approves(self):
        from unittest import mock

        with mock.patch("backend_service.tools.studio.ready_answer", return_value=None):
            self.assertEqual(self.run_review(StudioGate("s", auto_approve=True, auto_until=time.time() + 60))[0], "auto")
            # Expired mid-turn: the next render shows a card again.
            self.assertEqual(self.run_review(StudioGate("s", auto_approve=True, auto_until=time.time() - 1))[0], "card")
            self.assertEqual(self.run_review(StudioGate("s", auto_approve=True))[0], "auto")


class FilmLengthTests(unittest.TestCase):
    """The brief's length is checked against the shots (the 80 s film made for a 120 s brief, 2026-11-17)."""

    def test_the_length_a_brief_asks_for(self):
        self.assertEqual(target_seconds("a 2-minute film (~120 s), 4 s shots, 1024x576 at 24 fps, 30 shots"), 120.0)
        self.assertEqual(target_seconds("Make it two minutes long"), 120.0)
        self.assertEqual(target_seconds("about 90 seconds"), 90.0)
        self.assertIsNone(target_seconds("4 s shots at 24 fps, 8 h of rendering, 1024x576"))
        self.assertIsNone(target_seconds(""))

    def film(self, shots, seconds):
        return FilmState(project_id="p1", name="n", revision=1, shots=shots, seconds=seconds, music=True, render="done")

    def test_a_short_film_is_not_complete_and_says_how_many_shots_to_add(self):
        goal = "a 2-minute film"
        state = self.film(20, 80.0)
        self.assertFalse(state.complete(goal))
        self.assertIn("film length (80 s of the 120 s the brief asks for: about 10 more shots)", state.lacks(goal)[0])
        self.assertIn("Add about 10 more shots", state.next_step(goal))

    def test_a_film_near_the_length_is_complete(self):
        state = self.film(28, 112.0)
        self.assertTrue(state.complete("a 2-minute film"))
        self.assertIn("112 s long", state.next_step("a 2-minute film"))
        self.assertIn("Do not render again", state.next_step("a 2-minute film"))

    def test_no_length_in_the_brief_means_no_check(self):
        self.assertTrue(self.film(3, 12.0).complete("a short film"))


if __name__ == "__main__":
    unittest.main()
