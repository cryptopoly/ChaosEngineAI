"""Keep going (FU-191) through the real chat path: the routes, then whole runs driven by a scripted stub model
with the real controller, the real studio tools and a real Storyboard project on a throwaway folder. The stub
stops early, hits a tool error, runs into the step limit, claims it is done too soon and finally finishes."""

import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_service import keep_going
from backend_service.keep_going import adapters as kg_adapters
from backend_service.keep_going.project_state import FilmState
from backend_service.state import ChaosEngineState
from backend_service.tools import studio_gate
from tests.test_backend_service import fake_library, fake_system_snapshot, make_test_client
from tests.test_chat_studio_tools import ScriptedRuntime, call, result


class StubModel(ScriptedRuntime):
    """Answers each agent step from ``script(step, history)``; the history holds this turn's tool results."""

    def __init__(self, script):
        super().__init__()
        self.script = script
        self.steps = 0
        self.prompts = []

    def generate(self, **kwargs):
        self.steps += 1
        if kwargs.get("prompt"):
            self.prompts.append(kwargs["prompt"])
        return self.script(self, kwargs.get("history") or [])


def last_tool(history):
    for message in reversed(history):
        if message.get("role") == "tool":
            return str(message.get("text") or "")
    return None


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"CHAOSENGINE_STORYBOARDS_DIR": str(root / "boards")})
        self.env.start()
        self.root = root
        self.state = self.make_state()
        self.client = make_test_client(self.state)
        self.client.post("/api/models/load", json={"modelRef": "google/gemma-4-E4B-it", "modelName": "Gemma",
                                                   "source": "catalog", "backend": "mock"})
        self.session = self.client.post("/api/chat/sessions", json={"title": "t"}).json()["session"]
        self.sid = self.session["id"]

    def tearDown(self):
        run = keep_going.for_state(self.state).get(self.sid)
        if run is not None and run.state in ("armed", "running", "waiting"):
            keep_going.for_state(self.state).stop(self.sid)
        deadline = time.monotonic() + 5
        while studio_gate.turn_running(self.sid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.env.stop()
        self.tmp.cleanup()

    def make_state(self):
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot, library_provider=fake_library,
            settings_path=self.root / "settings.json", benchmarks_path=self.root / "benchmarks.json",
            chat_sessions_path=self.root / "chat-sessions.json")
        state.runtime = StubModel(lambda model, history: result("ok"))
        return state

    def status(self):
        return self.client.get(f"/api/chat/keepgoing/{self.sid}").json()["run"]

    def wait_stopped(self, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.status()
            if run and run["state"] == "stopped" and not run["turnRunning"]:
                return run
            time.sleep(0.05)
        self.fail(f"run did not stop: {self.status()}")

    def send(self, prompt, **body):
        response = self.client.post("/api/chat/generate/stream",
                                    json={"sessionId": self.sid, "prompt": prompt, "studioTools": True,
                                          "maxTokens": 1024, **body})
        return response.text

    def messages(self):
        sessions = self.client.get("/api/workspace").json()["chatSessions"]
        return next(s for s in sessions if s["id"] == self.sid)["messages"]


class RouteTests(Harness):
    def test_start_status_stop_and_idempotence(self):
        self.assertIsNone(self.status())
        first = self.client.post(f"/api/chat/keepgoing/{self.sid}/start", json={"cap": 7, "grant": "auto", "hours": 3}).json()["run"]
        self.assertEqual((first["state"], first["cap"], first["grant"]["mode"]), ("armed", 7, "auto"))
        self.assertLessEqual(first["grant"]["secondsLeft"], 3 * 3600)
        again = self.client.post(f"/api/chat/keepgoing/{self.sid}/start", json={"cap": 50}).json()["run"]
        self.assertEqual((again["runId"], again["cap"]), (first["runId"], 7))
        stopped = self.client.post(f"/api/chat/keepgoing/{self.sid}/stop").json()["run"]
        self.assertEqual((stopped["state"], stopped["stopCode"], stopped["grant"]), ("stopped", "user_stopped", None))
        self.assertFalse(stopped["resumable"])  # never got a brief, so nothing to resume

    def test_bad_requests(self):
        self.assertEqual(self.client.post("/api/chat/keepgoing/no%20such/start", json={}).status_code, 404)
        # A new chat the app has not sent yet (a draft id) is created, so the run can be armed before the brief.
        draft = self.client.post("/api/chat/keepgoing/draft-123/start", json={}).json()["run"]
        self.assertEqual(draft["state"], "armed")
        self.assertEqual(self.client.post(f"/api/chat/keepgoing/{self.sid}/start", json={"cap": 101}).status_code, 422)
        self.assertEqual(self.client.post(f"/api/chat/keepgoing/{self.sid}/start", json={"hours": 13}).status_code, 422)
        self.assertEqual(self.client.post(f"/api/chat/keepgoing/{self.sid}/resume", json={}).status_code, 404)

    def test_deleting_the_chat_drops_the_run(self):
        self.client.post(f"/api/chat/keepgoing/{self.sid}/start", json={})
        self.client.delete(f"/api/chat/sessions/{self.sid}")
        self.assertIsNone(keep_going.for_state(self.state).get(self.sid))

    def test_studio_off_ends_the_run(self):
        self.client.post(f"/api/chat/keepgoing/{self.sid}/start", json={})
        run = self.client.post(f"/api/chat/keepgoing/{self.sid}/studio-off").json()["run"]
        self.assertEqual(run["stopCode"], "studio_off")


class ScriptedRunTests(Harness):
    def project_id(self):
        from backend_service.storyboard.store import default_store

        rows = default_store().list()
        return rows[0]["id"] if rows else None

    def revision(self):
        from backend_service.storyboard.store import default_store

        return default_store().load(self.project_id()).revision

    def test_a_whole_run_stops_early_errors_hits_the_step_limit_and_finishes(self):
        turn = {"n": 0}

        def script(model, history):
            tool_text = last_tool(history)
            n = turn["n"]
            if n == 0:  # the brief: make the project, then stop early
                if tool_text is None:
                    return result("", [call("storyboard_project", {"action": "create", "name": "Bakery ad"}, "c0")])
                turn["n"] = 1
                return result("I made the project.")
            if n == 1:  # nudge 1: a stale revision, then give up
                if tool_text is None:
                    return result("", [call("storyboard_shot", {"action": "add", "project_id": self.project_id(),
                                                                 "revision": 99, "fields": {"title": "Open"}}, "c1")])
                turn["n"] = 2
                return result("That failed.")
            if n == 2:  # nudge 2: reads the project until the step limit ends the turn
                model.reads = getattr(model, "reads", 0) + 1
                if model.reads >= 30:
                    turn["n"] = 3
                return result("", [call("storyboard_project", {"action": "get", "project_id": self.project_id()},
                                        f"g{model.reads}")])
            if n == 3:  # nudge 3: real progress, then claims it is done
                if tool_text is None:
                    return result("", [call("storyboard_shot", {"action": "add", "project_id": self.project_id(),
                                                                 "revision": self.revision(), "fields": {"title": "Open"}}, "c3")])
                turn["n"] = 4
                return result("The film is finished.")
            turn["n"] = n + 1
            return result("Rendered. The film is ready.")

        self.state.runtime = StubModel(script)
        real_reader = kg_adapters.read_film_state

        def reader(pid):
            film = real_reader(pid)
            if film is not None and turn["n"] >= 5:  # pretend the clips, music and render came in on the last turn
                return FilmState(project_id=film.project_id, name=film.name, revision=film.revision, shots=film.shots,
                                 takes=film.shots, music=True, render="done", render_finished_at=time.time())
            return film

        with mock.patch.object(kg_adapters, "read_film_state", reader):
            self.client.post(f"/api/chat/keepgoing/{self.sid}/start", json={"cap": 10, "grant": "auto", "hours": 6})
            self.send("Make a 30 second bakery ad film with music.")
            run = self.wait_stopped()
        self.assertEqual(run["stopCode"], "complete", run)
        self.assertEqual(run["nudges"], 4)
        self.assertEqual(run["projectId"], self.project_id())
        nudges = [m for m in self.messages() if m["role"] == "user" and m.get("keepGoing")]
        self.assertEqual([m["keepGoing"]["nudge"] for m in nudges], [1, 2, 3, 4])
        texts = [m["text"] for m in nudges]
        self.assertTrue(all(t.startswith("Keep going (automatic message") for t in texts))
        self.assertIn('Project "Bakery ad"', texts[0])
        self.assertIn("Last error (storyboard_shot): The project is at revision 1 now.", texts[1])
        self.assertIn("reached the tool-step limit", texts[2])
        self.assertIn("You said the work is finished, but the project disagrees.", texts[3])
        # The user's own brief is not marked; every nudge got a saved reply.
        self.assertNotIn("keepGoing", self.messages()[0])
        self.assertEqual(self.messages()[-1]["role"], "assistant")

    def test_cap_reached_and_no_progress(self):
        self.state.runtime = StubModel(lambda model, history: result("Working on it."))
        self.client.post(f"/api/chat/keepgoing/{self.sid}/start", json={"cap": 2})
        self.send("Make a film.")
        run = self.wait_stopped()
        self.assertEqual((run["stopCode"], run["nudges"]), ("cap_reached", 2))
        self.client.post(f"/api/chat/keepgoing/{self.sid}/resume", json={"cap": 10})
        run = self.wait_stopped()
        self.assertEqual(run["stopCode"], "no_progress")

    def test_a_user_message_cancels_and_its_turn_runs(self):
        hold = {"wait": True}

        def script(model, history):
            if "automatic message" in (model.prompts[-1] if model.prompts else "") and hold["wait"]:
                time.sleep(0.3)
            return result("ok")

        self.state.runtime = StubModel(script)
        self.client.post(f"/api/chat/keepgoing/{self.sid}/start", json={})
        self.send("Make a film.")
        time.sleep(0.1)
        self.send("Stop, make it a poster instead.")
        run = self.wait_stopped()
        self.assertEqual(run["stopCode"], "user_message")
        texts = [m["text"] for m in self.messages() if m["role"] == "user"]
        self.assertIn("Stop, make it a poster instead.", texts)
        deadline = time.monotonic() + 5
        while studio_gate.turn_running(self.sid) and time.monotonic() < deadline:
            time.sleep(0.05)
        messages = self.messages()
        for index, message in enumerate(messages):  # every nudge keeps its own reply right after it
            if message.get("keepGoing"):
                self.assertEqual(messages[index + 1]["role"], "assistant", [m["role"] for m in messages])
        self.assertEqual(messages[-1]["role"], "assistant")

    def test_restart_shows_resume_and_never_restarts_by_itself(self):
        started = {"n": 0}

        def script(model, history):
            started["n"] += 1
            return result("ok")

        self.state.runtime = StubModel(script)
        controller = keep_going.for_state(self.state)
        controller.start(self.sid, cap=5, app_request=None)
        controller.on_user_turn(self.sid, {"studioTools": True, "maxTokens": 512}, "Make a film.", None)
        # A second app on the same data folder (the restart): the run comes back stopped, with Resume.
        state2 = self.make_state()
        client2 = make_test_client(state2)
        run = client2.get(f"/api/chat/keepgoing/{self.sid}").json()["run"]
        self.assertEqual((run["state"], run["stopCode"], run["resumable"], run["grant"]), ("stopped", "app_restart", True, None))
        time.sleep(0.2)
        self.assertEqual(started["n"], 0)
        state2.runtime = StubModel(script)
        client2.post("/api/models/load", json={"modelRef": "google/gemma-4-E4B-it", "modelName": "Gemma",
                                               "source": "catalog", "backend": "mock"})
        resumed = client2.post(f"/api/chat/keepgoing/{self.sid}/resume", json={"grant": "auto", "hours": 1}).json()["run"]
        self.assertIn(resumed["state"], ("running", "stopped"))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            r = client2.get(f"/api/chat/keepgoing/{self.sid}").json()["run"]
            if r["state"] == "stopped" and not r["turnRunning"]:
                break
            time.sleep(0.05)
        sessions = client2.get("/api/workspace").json()["chatSessions"]
        texts = [m["text"] for m in next(s for s in sessions if s["id"] == self.sid)["messages"] if m.get("keepGoing")]
        self.assertTrue(texts and "Resuming after a pause" in texts[0] and "The brief, as the user gave it" in texts[0])
        keep_going.for_state(state2).stop(self.sid)


class PlanCardRunTests(Harness):
    def test_manual_grant_waits_at_the_card_and_a_decline_stops(self):
        def script(model, history):
            if last_tool(history) is None and not getattr(model, "asked", False):
                model.asked = True
                return result("", [call("generate_image", {"prompt": "loaf of bread"}, "img1")])
            return result("Waiting for you.")

        self.state.runtime = StubModel(script)
        with mock.patch("backend_service.tools.studio.ready_answer", return_value=None):
            self.client.post(f"/api/chat/keepgoing/{self.sid}/start", json={"grant": "manual"})
            import threading

            sender = threading.Thread(target=self.send, args=("Make a poster.",))
            sender.start()
            deadline = time.monotonic() + 10
            plans = []
            while time.monotonic() < deadline and not plans:
                plans = studio_gate.pending_for_session(self.sid)
                time.sleep(0.05)
            self.assertTrue(plans)
            self.assertEqual(self.status()["phase"], "awaiting_approval")
            studio_gate.decide(plans[0]["planId"], {"img1": "deny"})
            sender.join(10)
            run = self.wait_stopped()
        self.assertEqual(run["stopCode"], "plan_declined")
        self.assertEqual(run["nudges"], 0)


if __name__ == "__main__":
    unittest.main()
