"""FU-208: the open chat sees a Studio turn it does not stream (a Keep going nudge) as it happens: the detached-turn
poll carries the reply so far. No real models."""

import json
import unittest

from backend_service.tools import studio_live
from tests import test_chat_plan_card_sj as sj
from tests.test_chat_studio_tools import call, result


def frame(event):
    return f"data: {json.dumps(event)}\n\n"


class LiveTurnTests(unittest.TestCase):
    def test_frames_fold_into_one_partial_reply(self):
        live = studio_live.LiveTurn()
        for event in ({"token": "Making "}, {"token": "the film."}, {"reasoning": "plan"},
                      {"toolCallStart": {"id": "c1", "name": "wait_for_job", "arguments": {"job_id": "job-3"},
                                         "startedAt": 5}},
                      {"toolCallProgress": {"id": "c1", "name": "wait_for_job",
                                            "data": {"state": "running", "progress": {"step": 2}}}}):
            live.feed(frame(event))
        live.feed(": keepalive\n\n")
        live.feed("data: not json\n\n")
        snap = live.snapshot()
        self.assertEqual(snap["text"], "Making the film.")
        self.assertEqual(snap["reasoning"], "plan")
        self.assertEqual(snap["toolCalls"], [{
            "id": "c1", "name": "wait_for_job", "arguments": {"job_id": "job-3"}, "result": "", "elapsed": 0,
            "startedAt": 5, "running": True, "renderAs": "studio",
            "data": {"state": "running", "progress": {"step": 2}}}])
        live.feed(frame({"toolCallResult": {"id": "c1", "name": "wait_for_job", "arguments": {"job_id": "job-3"},
                                            "result": "{}", "elapsed": 61.5, "renderAs": "studio",
                                            "data": {"state": "done"}}}))
        done = live.snapshot()["toolCalls"][0]
        self.assertNotIn("running", done)
        self.assertEqual((done["startedAt"], done["elapsed"], done["data"]), (5, 61.5, {"state": "done"}))

    def test_arguments_sent_as_json_text_are_read(self):
        live = studio_live.LiveTurn()
        live.feed(frame({"toolCallStart": {"id": "c1", "name": "wait_for_job", "arguments": '{"job_id": "job-9"}'}}))
        live.feed(frame({"toolCallStart": {"id": "c2", "name": "speak", "arguments": "not json"}}))
        calls = live.snapshot()["toolCalls"]
        self.assertEqual([c["arguments"] for c in calls], [{"job_id": "job-9"}, {}])

    def test_partial_is_kept_only_while_the_turn_runs(self):
        live = studio_live.begin("s-live")
        live.feed(frame({"token": "hi"}))
        self.assertEqual(studio_live.partial("s-live")["text"], "hi")
        newer = studio_live.begin("s-live")
        studio_live.end("s-live", live)  # an older turn ending never drops the newer one
        self.assertIsNotNone(studio_live.partial("s-live"))
        studio_live.end("s-live", newer)
        self.assertIsNone(studio_live.partial("s-live"))


class TurnRouteTests(unittest.TestCase):
    """The real chat path: the poll returns the running turn's text and tool cards."""

    setUp = sj.PlanCardStreamTests.setUp
    tearDown = sj.PlanCardStreamTests.tearDown
    new_session = sj.PlanCardStreamTests.new_session
    open_stream = sj.PlanCardStreamTests.open_stream

    def test_turn_poll_carries_the_reply_so_far(self):
        session = self.new_session()
        self.runtime.replies = [result("", [call("list_models", {}, "a0")]),
                                result("", [call("generate_image", {"prompt": "rose"}, "a1")])]
        body = self.open_stream(session["id"], "Paint a red rose poster")
        try:
            turn = self.client.get(f"/api/chat/studio/sessions/{session['id']}/turn").json()
            self.assertTrue(turn["running"])
            self.assertEqual(set(turn["partial"]), {"text", "reasoning", "toolCalls"})
            first = turn["partial"]["toolCalls"][0]
            self.assertEqual((first["id"], first["name"], first["renderAs"]), ("a0", "list_models", "studio"))
            self.assertNotIn("running", first)  # answered: the same card as the streamed turn shows
        finally:
            self.state.request_cancel_chat(session["id"])
            "".join(body)
        after = self.client.get(f"/api/chat/studio/sessions/{session['id']}/turn").json()
        self.assertEqual((after["running"], after["partial"]), (False, None))


if __name__ == "__main__":
    unittest.main()
