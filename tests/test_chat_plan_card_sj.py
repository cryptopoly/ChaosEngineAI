"""S-J (v0.10.0 item 1): Chat's plan card. A running thread is named by its first message, a refused studio call
is kept in the history with its reason, and two chats can each have a turn running. No real models."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend_service.state import ChaosEngineState
from backend_service.state._helpers import _build_history_with_reasoning
from backend_service.tools import studio_gate
from tests.test_backend_service import fake_library, fake_system_snapshot, make_test_client
from tests.test_chat_studio_tools import ScriptedRuntime, call, result


def refused(name, reason, call_id="c1"):
    return {"id": call_id, "name": name, "renderAs": "studio", "result": f"Error: {reason}",
            "data": {"tool": name, "error": reason}}


class RefusalHistoryTests(unittest.TestCase):
    def test_a_refused_call_stays_in_the_history_with_its_reason(self):
        reason = "Invalid input for storyboard_generate: 'what' must be one of still, clip, both."
        history = _build_history_with_reasoning([
            {"role": "user", "text": "make the stills"},
            {"role": "assistant", "text": "", "toolCalls": [refused("storyboard_generate", reason)]},
            {"role": "user", "text": "try again"},
        ], preserve_reasoning=False)
        self.assertEqual(history[1]["text"], f"[storyboard_generate was not run: {reason}]")

    def test_the_note_follows_the_reply_text(self):
        history = _build_history_with_reasoning([
            {"role": "assistant", "text": "I could not start it.", "toolCalls": [refused("generate_image", "No image model.")]},
        ], preserve_reasoning=False)
        self.assertEqual(history[0]["text"], "I could not start it.\n\n[generate_image was not run: No image model.]")

    def test_a_refusal_fixed_in_the_same_turn_is_left_out(self):
        ok = {"id": "c2", "name": "storyboard_generate", "renderAs": "studio", "result": "{}", "data": {"jobId": "job-1"}}
        history = _build_history_with_reasoning([
            {"role": "assistant", "text": "Started.", "toolCalls": [refused("storyboard_generate", "bad what"), ok]},
        ], preserve_reasoning=False)
        self.assertEqual(history[0]["text"], "Started.")

    def test_other_tool_results_are_not_replayed(self):
        other = {"id": "c1", "name": "calculator", "renderAs": "json", "result": "4", "data": {"error": "x"}}
        history = _build_history_with_reasoning([
            {"role": "assistant", "text": "4", "toolCalls": [other]},
        ], preserve_reasoning=False)
        self.assertEqual(history[0]["text"], "4")


class PlanCardStreamTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.sessions_path = root / "chat-sessions.json"
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=root / "settings.json",
            benchmarks_path=root / "benchmark-history.json",
            chat_sessions_path=self.sessions_path,
        )
        self.runtime = ScriptedRuntime()
        state.runtime = self.runtime
        self.state = state
        self.client = make_test_client(state)
        self.client.post("/api/models/load", json={"modelRef": "google/gemma-4-E4B-it", "modelName": "Gemma",
                                                   "source": "catalog", "backend": "mock"})

    def tearDown(self):
        self.tempdir.cleanup()

    def new_session(self):
        return self.client.post("/api/chat/sessions", json={"title": "New chat"}).json()["session"]

    def open_stream(self, session_id, prompt):
        from backend_service.state import generation

        captured = {}
        with mock.patch.object(generation, "StreamingResponse", lambda body, **kw: captured.setdefault("body", body)):
            self.state.generate_stream(generation.GenerateRequest(sessionId=session_id, prompt=prompt, studioTools=True),
                                       app_request=SimpleNamespace(app=self.client.app))
        body = captured["body"]
        for chunk in body:
            if "studioPlan" in chunk:
                return body
        self.fail("no plan card")

    def saved_titles(self):
        return {s["id"]: s["title"] for s in json.loads(self.sessions_path.read_text())}

    def test_two_running_threads_are_listed_by_their_first_message(self):
        first, second = self.new_session(), self.new_session()
        self.runtime.replies = [result("", [call("generate_image", {"prompt": "rose"}, "a1")]),
                                result("", [call("generate_image", {"prompt": "tulip"}, "b1")])]
        bodies = [self.open_stream(first["id"], "Paint a red rose poster"),
                  self.open_stream(second["id"], "Draw a tulip field at dawn")]
        try:
            listed = {s["id"]: s for s in self.client.get("/api/workspace").json()["chatSessions"]}
            self.assertEqual(listed[first["id"]]["title"], "Paint a red rose poster")
            self.assertEqual(listed[second["id"]]["title"], "Draw a tulip field at dawn")
            # Saved to disk already, not only when the replies end.
            saved = self.saved_titles()
            self.assertEqual((saved[first["id"]], saved[second["id"]]),
                             ("Paint a red rose poster", "Draw a tulip field at dawn"))
            for session in (first, second):
                turn = self.client.get(f"/api/chat/studio/sessions/{session['id']}/turn").json()
                self.assertTrue(turn["running"])
                self.assertEqual(len(turn["plans"]), 1)
                self.assertEqual(turn["plans"][0]["sessionId"], session["id"])
        finally:
            for session, body in zip((first, second), bodies):
                self.state.request_cancel_chat(session["id"])
                "".join(body)
        for session in (first, second):
            self.assertFalse(studio_gate.turn_running(session["id"]))

    def test_a_refused_stills_call_shows_its_reason_and_the_next_turn_reads_it(self):
        session = self.new_session()
        seen_histories = []
        original = self.runtime.generate

        def generate(**kwargs):
            seen_histories.append([m.get("text") for m in kwargs.get("history") or []])
            return original(**kwargs)

        self.runtime.generate = generate
        bad = {"project_id": "p1", "revision": 1, "what": "stills"}
        self.runtime.replies = [result("", [call("storyboard_generate", bad, "c1")]), result("")]
        response = self.client.post("/api/chat/generate/stream",
                                    json={"sessionId": session["id"], "prompt": "make the stills", "studioTools": True})
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertFalse(any("studioPlan" in e for e in events))  # never a card for a call that cannot run
        refusal = next(e["toolCallResult"] for e in events if "toolCallResult" in e)
        self.assertEqual(refusal["renderAs"], "studio")
        self.assertIn("'what' must be one of still", refusal["data"]["error"])
        # Same turn: the model's next reply sees the tool answer. Next turn: the history carries the reason.
        self.runtime.replies = [result("Fixed.")]
        self.client.post("/api/chat/generate/stream",
                         json={"sessionId": session["id"], "prompt": "try again", "studioTools": True})
        last_history = "\n".join(t or "" for t in seen_histories[-1])
        self.assertIn("storyboard_generate was not run: Invalid input for storyboard_generate", last_history)

    def test_the_reloaded_window_answers_the_waiting_card(self):
        from backend_service.mcp_bridge.tools import HANDLERS

        session = self.new_session()
        self.runtime.replies = [result("", [call("generate_image", {"prompt": "rose"}, "c1")]), result("Queued.")]
        handlers = mock.patch.dict(HANDLERS, {"generate_image": lambda ctx, args: {"jobId": "job-r", "state": "queued"}})
        handlers.start()
        self.addCleanup(handlers.stop)
        body = self.open_stream(session["id"], "Paint a rose")
        body.close()  # reload
        plans = self.client.get(f"/api/chat/studio/sessions/{session['id']}/turn").json()["plans"]
        done = threading.Event()

        def wait_for_reply():
            deadline = time.monotonic() + 5
            while studio_gate.turn_running(session["id"]) and time.monotonic() < deadline:
                time.sleep(0.02)
            done.set()

        self.assertEqual(self.client.post(f"/api/chat/studio/plans/{plans[0]['planId']}/decision",
                                          json={"decisions": {"c1": "approve"}}).status_code, 200)
        threading.Thread(target=wait_for_reply, daemon=True).start()
        self.assertTrue(done.wait(6))
        listed = next(s for s in self.client.get("/api/workspace").json()["chatSessions"] if s["id"] == session["id"])
        self.assertEqual(listed["messages"][-1]["text"], "Queued.")
        self.assertEqual(listed["title"], "Paint a rose")


if __name__ == "__main__":
    unittest.main()
