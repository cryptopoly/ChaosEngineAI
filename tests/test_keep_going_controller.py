"""Keep going (FU-191): the run controller's state machine with fake app hooks. No models, no threads unless a
test starts them on purpose."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend_service.keep_going.adapters import StoryboardAdapter
from backend_service.keep_going.controller import KeepGoing
from backend_service.keep_going.nudge import TurnOutcome
from backend_service.keep_going.project_state import FilmState
from backend_service.keep_going.run import TurnStartError

SID = "chat-1"


class FakeHooks:
    def __init__(self):
        self.turns = []
        self.running = set()
        self.plans = 0
        self.jobs = []
        self.cancelled = []
        self.logs = []
        self.goal_in = True
        self.start_error = None
        self.on_start = None
        self.messages = 3
        self.last_user = "the brief"

    def start_turn(self, run, prompt, auto_until):
        if self.on_start:
            self.on_start(run)
        if self.start_error:
            raise self.start_error
        self.turns.append(SimpleNamespace(run_id=run.id, prompt=prompt, auto_until=auto_until, abort=run.abort))

    def turn_running(self, session_id):
        return session_id in self.running

    def pending_plans(self, session_id):
        return self.plans

    def chat_jobs(self):
        return list(self.jobs)

    def goal_in_history(self, run):
        return self.goal_in

    def last_user_text(self, session_id):
        return self.last_user

    def message_count(self, session_id):
        return self.messages

    def cancel_turn(self, session_id):
        self.cancelled.append(session_id)

    def log(self, level, text):
        self.logs.append((level, text))


def film(**over):
    base = dict(project_id="p1", name="Ad", revision=3, shots=4, needs_still=[], needs_clip=[3, 4], takes=2,
                music=False, render="none")
    base.update(over)
    return FilmState(**base)


def done_film(revision=9):
    return FilmState(project_id="p1", name="Ad", revision=revision, shots=4, takes=4, music=True, render="done",
                     render_finished_at=100.0)


def job(job_id, state="done", kind="image", error=None):
    return SimpleNamespace(id=job_id, state=state, kind=kind, error=error)


def tool(name, error=None, data=None):
    payload = dict(data or {})
    if error:
        payload["error"] = error
    return {"name": name, "result": f"Error: {error}" if error else "{}", "data": payload, "renderAs": "studio"}


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "keep_going_runs.json"
        self.hooks = FakeHooks()
        self.clock = Clock()
        self.film = film()
        self.ctrl = self.make()

    def tearDown(self):
        self.tmp.cleanup()

    def make(self):
        return KeepGoing(self.path, self.hooks, clock=self.clock, threaded=False,
                         adapters=lambda name: StoryboardAdapter(reader=lambda pid: self.film))

    def begin(self, cap=20, grant="manual", hours=6, brief="Make a 2 minute ad film with music."):
        self.ctrl.start(SID, cap=cap, grant_mode=grant, hours=hours, app_request=object())
        return self.ctrl.on_user_turn(SID, {"studioTools": True, "maxTokens": 4096, "prompt": brief}, brief, object())

    def finish(self, text="", calls=None, plans=None, **kw):
        run = self.ctrl.get(SID)
        assistant = {"role": "assistant", "text": text, "toolCalls": calls or []}
        if plans is not None:
            assistant["studioPlans"] = plans
        run_id = kw.pop("run_id", run.id if run.nudges else None)
        self.ctrl.on_turn_finished(TurnOutcome(session_id=SID, assistant=assistant, extra={"runId": run_id}, **kw))

    def state(self):
        return self.ctrl.get(SID)


class StartAndNudgeTests(Base):
    def test_off_by_default_and_armed_until_the_brief(self):
        self.assertIsNone(self.ctrl.status(SID))
        self.ctrl.start(SID, app_request=object())
        self.assertEqual(self.state().state, "armed")
        self.assertEqual(self.hooks.turns, [])

    def test_the_next_message_is_the_brief_and_the_turn_end_nudges(self):
        self.begin()
        self.assertEqual(self.state().goal, "Make a 2 minute ad film with music.")
        self.assertEqual(self.state().state, "running")
        self.finish(text="I made the shots.")
        self.assertEqual(len(self.hooks.turns), 1)
        prompt = self.hooks.turns[0].prompt
        self.assertTrue(prompt.startswith("Keep going (automatic message 1 of 20 from the app"))
        self.assertIn("clip for shot 3-4", prompt)
        self.assertIn("music", prompt)
        self.assertIn('storyboard_generate with project_id p1, what "clip" (shots 3-4)', prompt)
        self.assertNotIn("continue", prompt.lower().split("next step")[0][-20:])
        self.assertLess(len(prompt), 1200)  # short: the brief is not repeated while it is in the history
        self.assertEqual(self.state().nudges, 1)

    def test_start_is_idempotent(self):
        first = self.ctrl.start(SID, cap=5, app_request=object())
        second = self.ctrl.start(SID, cap=50, grant_mode="auto", app_request=object())
        self.assertIs(first, second)
        self.assertEqual(second.cap, 5)

    def test_start_while_the_brief_turn_runs_takes_that_message(self):
        self.hooks.running.add(SID)
        self.hooks.last_user = "the spec I just sent"
        run = self.ctrl.start(SID, app_request=object())
        self.assertEqual((run.state, run.goal, run.turn_pending), ("running", "the spec I just sent", True))

    def test_a_non_studio_message_does_not_become_the_brief(self):
        self.ctrl.start(SID, app_request=object())
        self.ctrl.on_user_turn(SID, {"studioTools": False}, "hello", object())
        self.assertEqual(self.state().state, "armed")

    def test_cap_is_clamped(self):
        self.assertEqual(self.ctrl.start(SID, cap=500, app_request=object()).cap, 100)
        self.ctrl.stop(SID)
        self.ctrl.forget(SID)
        self.assertEqual(self.ctrl.start(SID, cap=0, app_request=object()).cap, 1)

    def test_the_brief_is_repeated_when_the_history_lost_it(self):
        self.begin(brief="A long brief " * 10)
        self.hooks.goal_in = False
        self.finish()
        self.assertIn("The brief, as the user gave it:", self.hooks.turns[0].prompt)

    def test_a_turn_of_an_older_run_is_ignored(self):
        self.begin()
        self.finish()
        self.finish(run_id="kg-old")
        self.assertEqual(len(self.hooks.turns), 1)
        self.assertTrue(self.state().turn_pending)

    def test_cap_and_tool_limit_wording(self):
        self.begin()
        self.finish(max_iterations=True)
        self.assertIn("Your last turn stopped at the step limit.", self.hooks.turns[0].prompt)


class StopConditionTests(Base):
    def stopped(self, code):
        run = self.state()
        self.assertEqual(run.state, "stopped")
        self.assertEqual(run.stop_code, code)
        self.assertTrue(run.stop_reason)
        self.assertIsNone(run.grant)  # the grant ends with the run
        self.assertTrue(any(code in text or run.stop_reason in text for _, text in self.hooks.logs))

    def test_goal_complete_from_project_state(self):
        self.begin()
        self.film = done_film()
        self.finish(text="Here it is.")
        self.stopped("complete")
        self.assertEqual(self.hooks.turns, [])

    def test_done_claim_without_project_state_agreeing_nudges_instead(self):
        self.begin()
        self.finish(text="The film is finished.")
        self.assertIn("The project is not finished yet.", self.hooks.turns[0].prompt)

    def test_done_claim_with_no_project_stops(self):
        self.film = None
        self.begin()
        self.finish(text="All done, the images are ready.")
        self.stopped("complete_no_project")

    def test_complete_waits_for_running_jobs(self):
        self.begin()
        self.film = done_film()
        self.hooks.jobs = [job("j1", "running")]
        self.finish()
        self.assertEqual(self.state().state, "waiting")

    def test_nudge_cap(self):
        self.begin(cap=2)
        for rev in (4, 5):
            self.film = film(revision=rev)
            self.finish()
        self.assertEqual(len(self.hooks.turns), 2)
        self.film = film(revision=6)
        self.finish()
        self.stopped("cap_reached")
        self.assertIn("2", self.state().stop_reason)

    def test_no_progress_after_three_nudges(self):
        self.begin()
        for _ in range(4):
            self.finish()
        self.stopped("no_progress")
        self.assertEqual(len(self.hooks.turns), 3)

    def test_a_finished_job_counts_as_progress(self):
        self.begin()
        self.finish()
        for n in range(5):
            self.hooks.jobs.append(job(f"j{n}"))
            self.finish()
        self.assertEqual(self.state().state, "running")

    def test_same_error_three_times_stops(self):
        self.begin()
        err = "storyboard_generate failed: no video model installed"
        for i in range(2):
            self.film = film(revision=10 + i)
            self.finish(calls=[tool("storyboard_generate", err)])
            self.assertEqual(self.state().state, "running")
        self.film = film(revision=20)
        self.finish(calls=[tool("storyboard_generate", err)])
        self.stopped("repeated_error")
        self.assertIn("no video model installed", self.state().stop_reason)

    def test_one_error_gets_one_nudge_with_its_exact_text(self):
        self.begin()
        err = "The project is at revision 7 now. Call storyboard_project get and retry."
        self.finish(calls=[tool("storyboard_shot", err)])
        prompt = self.hooks.turns[0].prompt
        self.assertIn(f"Last error (storyboard_shot): {err}", prompt)
        self.assertIn("Fix that error first", prompt)
        self.assertIn("Your last turn ended after a tool error.", prompt)

    def test_an_error_fixed_later_in_the_turn_is_not_reported(self):
        self.begin()
        self.finish(calls=[tool("storyboard_shot", "bad"), tool("storyboard_shot", data={"projectId": "p1"})])
        self.assertNotIn("Last error", self.hooks.turns[0].prompt)

    def test_context_nearly_full(self):
        self.begin()
        self.finish(prompt_tokens=7000, context_tokens=8192)
        self.stopped("context_full")
        self.assertIn("7000 of 8192", self.state().stop_reason)

    def test_context_error_text_stops(self):
        self.begin()
        self.finish(error="the request exceeds the context window (n_ctx 8192)")
        self.stopped("context_full")

    def test_plan_declined_edit_and_unanswered(self):
        for status, decisions, code in (("decided", {"c1": "deny"}, "plan_declined"), ("edit", {"c1": "deny"}, "plan_edit"),
                                        ("unanswered", {"c1": "deny"}, "plan_unanswered")):
            self.ctrl.forget(SID)
            self.begin()
            self.finish(plans=[{"planId": "p", "status": status, "decisions": decisions}])
            self.stopped(code)

    def test_a_partly_approved_plan_goes_on(self):
        self.begin()
        self.finish(plans=[{"planId": "p", "status": "decided", "decisions": {"a": "approve", "b": "deny"}}])
        self.assertEqual(self.state().state, "running")

    def test_grant_expiry_pauses_the_run(self):
        until = self.begin(grant="auto", hours=1)
        self.assertEqual(until, self.clock.now + 3600)
        self.clock.now += 3601
        self.finish()
        self.stopped("approval_expired")
        self.assertTrue(self.ctrl.status(SID)["resumable"])

    def test_auto_grant_reaches_the_nudge_turn_and_manual_never_does(self):
        self.begin(grant="auto", hours=2)
        self.finish()
        self.assertEqual(self.hooks.turns[0].auto_until, 1000.0 + 7200)
        self.ctrl.forget(SID)
        self.hooks.turns.clear()
        self.begin(grant="manual")
        self.clock.now += 99 * 3600  # manual never expires
        self.finish()
        self.assertIsNone(self.hooks.turns[0].auto_until)

    def test_grant_hours_are_capped_at_twelve(self):
        self.begin(grant="auto", hours=48)
        self.assertEqual(self.state().grant.until, self.clock.now + 12 * 3600)

    def test_user_stop_cancels_the_turn(self):
        self.begin()
        self.finish()
        self.hooks.running.add(SID)
        self.ctrl.stop(SID)
        self.stopped("user_stopped")
        self.assertEqual(self.hooks.cancelled, [SID])
        self.assertTrue(self.hooks.turns[0].abort.is_set())

    def test_stop_in_the_chat_ends_the_run(self):
        self.begin()
        self.ctrl.on_chat_cancel(SID)
        self.stopped("user_stopped")

    def test_a_cancelled_turn_stops(self):
        self.begin()
        self.finish(cancelled=True)
        self.stopped("user_stopped")

    def test_a_new_user_message_cancels_the_run_and_wins(self):
        self.begin()
        self.finish()
        nudge_abort = self.hooks.turns[0].abort
        self.assertIsNone(self.ctrl.on_user_turn(SID, {"studioTools": True}, "actually make it blue", object()))
        self.stopped("user_message")
        self.assertTrue(nudge_abort.is_set())  # the nudge turn ends through its abort event
        self.assertEqual(self.hooks.cancelled, [])  # not the chat flag: the user's own turn clears that
        self.finish()  # the aborted nudge turn ending later changes nothing
        self.assertEqual(self.state().stop_code, "user_message")

    def test_model_failures(self):
        self.begin()
        self.finish(error="worker crashed")
        self.assertIn("Last error (chat): worker crashed", self.hooks.turns[0].prompt)
        self.finish(error="worker crashed")
        self.stopped("turn_failed")

    def test_model_missing_after_a_failure(self):
        self.begin()
        self.ctrl.on_turn_finished(TurnOutcome(session_id=SID, error="engine died",
                                               extra={"runId": None, "modelMissing": True}))
        self.stopped("model_unavailable")

    def test_a_nudge_that_cannot_start_stops_with_its_reason(self):
        self.begin()
        self.hooks.start_error = TurnStartError("model_unavailable", "Load a model before sending prompts.")
        self.finish()
        self.stopped("model_unavailable")
        self.assertEqual(self.state().nudges, 0)
        self.assertIn("Load a model", self.state().stop_reason)

    def test_a_user_message_during_the_start_keeps_its_reason(self):
        self.begin()

        def user_types(run):
            self.ctrl.on_user_turn(SID, {"studioTools": True}, "stop that", object())
            self.hooks.start_error = TurnStartError("user_message", "")

        self.hooks.on_start = user_types
        self.finish()
        self.stopped("user_message")


class WaitingTests(Base):
    def test_never_nudges_while_a_job_runs(self):
        self.begin()
        self.hooks.jobs = [job("j1", "running", "video")]
        self.finish()
        self.assertEqual(self.state().state, "waiting")
        self.assertEqual(self.hooks.turns, [])
        self.assertEqual(self.ctrl.status(SID)["phase"], "waiting_job")
        self.ctrl.poll_waiting()
        self.assertEqual(self.hooks.turns, [])
        self.hooks.jobs = [job("j1", "failed", "video", "out of memory")]
        self.ctrl.poll_waiting()
        self.assertEqual(len(self.hooks.turns), 1)
        self.assertIn("Since then: video job j1 failed: out of memory", self.hooks.turns[0].prompt)

    def test_jobs_finished_before_the_run_are_not_reported(self):
        self.hooks.jobs = [job("old")]
        self.begin()
        self.finish()
        self.assertNotIn("old", self.hooks.turns[0].prompt)

    def test_never_nudges_during_a_turn_or_a_plan_card(self):
        self.begin()
        self.hooks.plans = 1
        self.finish()
        self.assertEqual((self.state().state, self.hooks.turns), ("waiting", []))
        self.hooks.plans = 0
        self.hooks.running.add(SID)
        self.ctrl.poll_waiting()
        self.assertEqual(self.hooks.turns, [])
        self.hooks.running.clear()
        self.ctrl.poll_waiting()
        self.assertEqual(len(self.hooks.turns), 1)

    def test_grant_expiring_while_waiting(self):
        self.begin(grant="auto", hours=1)
        self.hooks.jobs = [job("j1", "running")]
        self.finish()
        self.clock.now += 7200
        self.hooks.jobs = [job("j1")]
        self.ctrl.poll_waiting()
        self.assertEqual(self.state().stop_code, "approval_expired")
        self.assertEqual(self.hooks.turns, [])

    def test_status_phases(self):
        self.begin()
        self.hooks.running.add(SID)
        self.assertEqual(self.ctrl.status(SID)["phase"], "turn")
        self.hooks.plans = 1
        self.assertEqual(self.ctrl.status(SID)["phase"], "awaiting_approval")


class RaceTests(Base):
    def test_two_dispatches_send_one_nudge(self):
        self.begin()
        gate = threading.Event()
        entered = threading.Event()

        def slow(run):
            entered.set()
            gate.wait(2)

        self.hooks.on_start = slow
        run = self.state()
        run.turn_pending = False
        threads = [threading.Thread(target=self.ctrl._dispatch, args=(SID,)) for _ in range(4)]
        for t in threads:
            t.start()
        entered.wait(2)
        time.sleep(0.05)
        gate.set()
        for t in threads:
            t.join(2)
        self.assertEqual(len(self.hooks.turns), 1)
        self.assertEqual(self.state().nudges, 1)

    def test_a_second_turn_end_for_the_same_nudge_is_ignored(self):
        self.begin()
        self.finish()
        run = self.state()
        run.turn_pending = False  # the turn already reported its end
        self.finish()
        self.assertEqual((run.nudges, len(self.hooks.turns)), (1, 1))


class PersistenceTests(Base):
    def test_saved_atomically_and_restored_as_interrupted(self):
        self.begin(grant="auto")
        self.finish()
        data = json.loads(self.path.read_text())
        record = data["runs"][SID]
        self.assertEqual((record["state"], record["nudges"], record["goal"]), ("running", 1, "Make a 2 minute ad film with music."))
        self.assertNotIn("grant", record)  # never saved: the grant dies with the app
        restarted = self.make()
        run = restarted.get(SID)
        self.assertEqual((run.state, run.stop_code), ("stopped", "app_restart"))
        self.assertIsNone(run.grant)
        status = restarted.status(SID)
        self.assertTrue(status["resumable"])
        self.assertIsNone(status["grant"])
        self.assertEqual(json.loads(self.path.read_text())["runs"][SID]["stop_code"], "app_restart")

    def test_resume_rebuilds_from_the_brief_and_asks_for_the_grant_again(self):
        self.begin(cap=3)
        self.finish()
        restarted = self.make()
        self.ctrl = restarted
        self.hooks.turns.clear()
        restarted.resume(SID, grant_mode="auto", hours=2, app_request=object())
        run = restarted.get(SID)
        self.assertEqual((run.state, run.nudges, run.resumes), ("running", 1, 1))
        prompt = self.hooks.turns[0].prompt
        self.assertIn("Resuming after a pause", prompt)
        self.assertIn("The brief, as the user gave it:", prompt)
        self.assertEqual(self.hooks.turns[0].auto_until, self.clock.now + 7200)

    def test_resume_twice_and_unresumable_runs(self):
        self.begin()
        self.ctrl.stop(SID)
        first = self.ctrl.resume(SID, app_request=object())
        again = self.ctrl.resume(SID, app_request=object())
        self.assertIs(first, again)
        self.assertEqual(len(self.hooks.turns), 1)
        self.ctrl.on_user_turn(SID, {"studioTools": True}, "new idea", object())
        with self.assertRaises(ValueError):
            self.ctrl.resume(SID, app_request=object())
        with self.assertRaises(LookupError):
            self.ctrl.resume("other", app_request=object())

    def test_resume_waits_for_a_running_job(self):
        self.begin()
        self.ctrl.stop(SID)
        self.hooks.jobs = [job("j9", "running")]
        self.ctrl.resume(SID, app_request=object())
        self.assertEqual((self.state().state, self.hooks.turns), ("waiting", []))

    def test_a_broken_file_is_ignored(self):
        self.path.write_text("{not json")
        self.assertIsNone(self.make().get(SID))

    def test_forget_drops_the_record(self):
        self.begin()
        self.ctrl.forget(SID)
        self.assertNotIn(SID, json.loads(self.path.read_text())["runs"])


if __name__ == "__main__":
    unittest.main()
