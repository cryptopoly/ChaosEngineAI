"""S1 (FU-179): one render approval per storyboard render in Chat.

On the Mac a Chat run asked for ``storyboard_generate`` twice; the first approval rendered nothing. The first call
could not run (a stale revision, or an argument the tool refuses), but the gate showed it anyway; after the user
approved it the call failed, and the model asked again with a second card. These tests drive the real studio
registry, the real gate and a real project store (temporary folder); only the still and clip makers are fakes.
"""

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend_service.agent import run_agent_loop_streaming
from backend_service.inference import GenerationResult
from backend_service.tools import registry as base_registry
from backend_service.tools import studio, studio_gate
from backend_service.tools.studio import studio_registry
from backend_service.tools.studio_gate import StudioGate, decide
from tests.storyboard_fixtures import FakeGenerators
from tests.test_storyboard_routes import fake_wiring

APP = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(chaosengine=None)))


def call(name, args, call_id):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(tool_calls=None, text=""):
    return GenerationResult(text=text, finishReason="stop", promptTokens=1, completionTokens=1, totalTokens=2,
                            tokS=1.0, responseSeconds=0.1, tool_calls=tool_calls)


class BlockingGenerators(FakeGenerators):
    """Stills wait until the test lets them go, so a batch stays running."""

    def __init__(self, src):
        super().__init__(src)
        self.release = threading.Event()

    def make_still(self, shot, project):
        self.release.wait(10)
        return super().make_still(shot, project)


class OneApprovalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        (base / "src").mkdir()
        self.gens = BlockingGenerators(base / "src")
        self.addCleanup(self.finish_batches)
        patches = [
            mock.patch.dict(os.environ, {"CHAOSENGINE_STORYBOARDS_DIR": str(base / "storyboards")}),
            mock.patch("backend_service.storyboard.generators.studio_generators", fake_wiring(self.gens)),
            mock.patch("backend_service.routes.storyboard._project_variant", lambda request, kind, model: (None, "none")),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(studio._ACTIVE_RENDERS.clear)
        self.addCleanup(lambda: studio_gate._PLANS.clear())
        self.registry = studio_registry(APP, base_registry, [])
        info = self.tool("storyboard_project", action="create", name="Tea Shop Ad")
        for i in range(4):
            info = self.tool("storyboard_shot", action="add", project_id=info["projectId"], revision=info["revision"],
                             fields={"title": f"shot {i}", "stillPrompt": "tea shop", "seconds": 2})
        self.pid, self.rev = info["projectId"], info["revision"]

    def finish_batches(self):
        from backend_service.storyboard.batch import BATCHES

        self.gens.release.set()
        for batch in list(BATCHES.values()):
            batch.done.wait(10)

    def tool(self, tool_name, **args):
        out = self.registry.get(tool_name).execute_structured(**args)
        self.assertNotIn("error", out.data, out.text)
        return out.data

    def run_chat(self, replies):
        """Run one chat turn; every plan card is approved in full as soon as it shows. Returns (events, plans)."""
        plans, stop = [], threading.Event()

        def clicker():
            while not stop.is_set():
                with studio_gate._LOCK:
                    waiting = [p for p in studio_gate._PLANS.values() if not p.event.is_set()]
                for plan in waiting:
                    plans.append(plan)
                    decide(plan.id, {i["callId"]: "approve" for i in plan.items})
                time.sleep(0.01)

        thread = threading.Thread(target=clicker, daemon=True)
        thread.start()
        script = iter(replies)
        try:
            events = list(run_agent_loop_streaming(
                generate_fn=lambda **kw: next(script), stream_generate_fn=None, prompt="make the stills", history=[],
                system_prompt=None, max_tokens=64, temperature=0.1, tool_registry=self.registry,
                gate=StudioGate("s", poll_seconds=0.01),
            ))
        finally:
            stop.set()
            thread.join()
        return events, plans

    @staticmethod
    def results(events):
        return {e["tool_call_result"]["id"]: e["tool_call_result"] for e in events if "tool_call_result" in e}

    def test_stale_revision_gets_no_card_and_the_retry_gets_one(self):
        """The FU-179 sequence: generate with an old revision, then again with the right one. One card, and it renders."""
        events, plans = self.run_chat([
            reply([call("storyboard_generate", {"project_id": self.pid, "revision": 1, "what": "still"}, "g1")]),
            reply([call("storyboard_generate", {"project_id": self.pid, "revision": self.rev, "what": "still"}, "g2")]),
            reply(text="The stills are being made."),
        ])
        self.assertEqual(len(plans), 1)
        self.assertEqual([i["callId"] for i in plans[0].items], ["g2"])
        results = self.results(events)
        self.assertIn(f"revision {self.rev}", results["g1"]["result"])
        self.assertIn("error", results["g1"]["data"])
        self.assertNotIn("declined", results["g1"]["data"])
        self.assertTrue(results["g2"]["data"].get("jobId"), results["g2"]["result"])

    def test_refused_argument_gets_no_card(self):
        events, plans = self.run_chat([
            reply([call("storyboard_generate", {"project_id": self.pid, "revision": self.rev, "what": "stills"}, "g1")]),
            reply(text="ok"),
        ])
        self.assertEqual(plans, [])
        self.assertIn("what", self.results(events)["g1"]["data"]["error"])

    def test_nothing_to_generate_gets_no_card(self):
        events, plans = self.run_chat([
            reply([call("storyboard_generate", {"project_id": self.pid, "revision": self.rev, "what": "still",
                                                "shot_ids": ["s_nope"]}, "g1")]),
            reply(text="ok"),
        ])
        self.assertEqual(plans, [])
        self.assertIn("s_nope", self.results(events)["g1"]["data"]["error"])

    def test_edit_before_generate_in_the_same_reply_still_renders(self):
        """The model adds a shot and asks for the stills in one reply, both with the revision it last saw."""
        events, plans = self.run_chat([
            reply([call("storyboard_shot", {"action": "add", "project_id": self.pid, "revision": self.rev,
                                            "fields": {"title": "end card", "stillPrompt": "logo"}}, "e1"),
                   call("storyboard_generate", {"project_id": self.pid, "revision": self.rev, "what": "still"}, "g1")]),
            reply(text="ok"),
        ])
        self.assertEqual(len(plans), 1)
        self.assertTrue(self.results(events)["g1"]["data"].get("jobId"), self.results(events)["g1"]["result"])

    def test_a_repeat_while_the_job_runs_returns_that_job(self):
        events, plans = self.run_chat([
            reply([call("storyboard_generate", {"project_id": self.pid, "revision": self.rev, "what": "still"}, "g1")]),
            reply([call("storyboard_generate", {"project_id": self.pid, "revision": self.rev, "what": "still"}, "g2")]),
            reply(text="ok"),
        ])
        self.assertEqual(len(plans), 1)
        results = self.results(events)
        self.assertEqual(results["g2"]["data"]["jobId"], results["g1"]["data"]["jobId"])
        self.assertIn("already", results["g2"]["result"])

    def test_two_identical_calls_in_one_reply_make_one_row_and_one_job(self):
        args = {"project_id": self.pid, "revision": self.rev, "what": "still"}
        events, plans = self.run_chat([
            reply([call("storyboard_generate", args, "g1"), call("storyboard_generate", args, "g2")]),
            reply(text="ok"),
        ])
        self.assertEqual(len(plans), 1)
        self.assertEqual([i["callId"] for i in plans[0].items], ["g1"])
        results = self.results(events)
        self.assertEqual(results["g2"]["data"]["jobId"], results["g1"]["data"]["jobId"])

    def test_a_declined_first_copy_declines_the_repeat(self):
        args = {"project_id": self.pid, "revision": self.rev, "what": "still"}
        gate = StudioGate("s", poll_seconds=0.01)

        def deny():
            while True:
                with studio_gate._LOCK:
                    waiting = list(studio_gate._PLANS.values())
                if waiting:
                    decide(waiting[0].id, {})
                    return
                time.sleep(0.01)

        thread = threading.Thread(target=deny, daemon=True)
        thread.start()
        gen = gate.review([call("storyboard_generate", args, "g1"), call("storyboard_generate", args, "g2")])
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            denials = stop.value
        thread.join()
        self.assertEqual(set(denials), {"g1", "g2"})


IMAGE = {"name": "Iris-3B Turbo", "estimatedGenerationSeconds": 15}
VIDEO = {"name": "LTX-2.5", "estimatedGenerationSeconds": 60, "defaultDurationSeconds": 2}


class PlanEstimateTests(OneApprovalTests):
    """S7: the plan card's time per shot and total, from the Storyboard estimate."""

    def setUp(self):
        super().setUp()
        variants = {"image": IMAGE, "video": VIDEO}
        patcher = mock.patch("backend_service.routes.storyboard._project_variant",
                             lambda request, kind, model: (variants[kind], None))
        patcher.start()
        self.addCleanup(patcher.stop)

    def item(self, **args):
        return studio_gate.plan_item(call("storyboard_generate", {"project_id": self.pid, "revision": self.rev, **args}, "g1"),
                                     SimpleNamespace(runtime=None))

    def test_both_lists_each_shot_and_the_total(self):
        board = self.item()["estimate"]
        self.assertEqual(board["seconds"], 300)
        self.assertEqual(board["basis"], "estimated")
        self.assertEqual(len(board["shots"]), 4)
        self.assertEqual({(r["still"], r["clip"]) for r in board["shots"]}, {(15, 60)})
        self.assertEqual(board["shots"][0]["title"], "shot 0")
        self.assertEqual((board["imageModel"], board["videoModel"], board["missing"]), ("Iris-3B Turbo", "LTX-2.5", []))

    def test_stills_only_have_no_clip_time(self):
        item = self.item(what="still")
        board = item["estimate"]
        self.assertEqual(board["seconds"], 60)
        self.assertNotIn("clip", board["shots"][0])
        self.assertEqual(item["model"], "Iris-3B Turbo")

    def test_a_missing_model_is_named_not_guessed(self):
        with mock.patch("backend_service.routes.storyboard._project_variant",
                        lambda request, kind, model: (None, "No video model is installed.") if kind == "video" else (IMAGE, None)):
            board = self.item()["estimate"]
        self.assertIsNone(board["seconds"])
        self.assertEqual(board["missing"], ["No video model is installed."])
        self.assertEqual(board["shots"][0]["still"], 15)
        self.assertIsNone(board["shots"][0]["clip"])

    def test_no_project_no_estimate(self):
        item = studio_gate.plan_item(call("storyboard_generate", {"project_id": "nope", "revision": 1}, "g1"), None)
        self.assertNotIn("estimate", item)


if __name__ == "__main__":
    unittest.main()
