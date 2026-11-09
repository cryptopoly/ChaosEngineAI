"""Chat waits for a render (wait_for_job blocks up to 30 min, stop ends it, no iteration cost), job progress
fields, storyboard batch progress, and the timestamps new chat messages and tool calls carry. No real models."""

import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from backend_service.agent import run_agent_loop_streaming
from backend_service.inference import GenerationResult
from backend_service.mcp_bridge.jobs import CURRENT_OWNER, STORE, Job
from backend_service.tools import BaseTool, StructuredToolOutput, ToolRegistry
from backend_service.tools import studio as studio_module
from backend_service.tools.studio import CHAT_OWNER, CHAT_WAIT_CAP_SECONDS, studio_tools

APP = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(chaosengine=None)))


def call(name, args, call_id):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(text="", tool_calls=None):
    return GenerationResult(text=text, finishReason="stop", promptTokens=1, completionTokens=1, totalTokens=2,
                            tokS=1.0, responseSeconds=0.1, tool_calls=tool_calls)


def chat_job(runner):
    token = CURRENT_OWNER.set(CHAT_OWNER)
    try:
        return STORE.submit("voice", "test job", runner)
    finally:
        CURRENT_OWNER.reset(token)


class WaitTool(BaseTool):
    """Stands in for the studio wait_for_job: answers at once, keyed on the job id like the real one."""

    @property
    def name(self):
        return "wait_for_job"

    @property
    def description(self):
        return "wait"

    def parameters_schema(self):
        return {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}

    def execute(self, **kwargs):
        return "{}"

    def execute_structured(self, **kwargs):
        return StructuredToolOutput(text="{}", render_as="studio", data={"tool": "wait_for_job"})

    def budget_key(self, arguments):
        return str(arguments.get("job_id"))


class ChatWaitTests(unittest.TestCase):
    def setUp(self):
        self.tools = {t.name: t for t in studio_tools(APP)}

    def test_the_cap_is_thirty_minutes_and_mcp_stays_at_sixty(self):
        from backend_service.mcp_bridge.tools import TOOLS

        self.assertEqual(CHAT_WAIT_CAP_SECONDS, 30 * 60)
        spec = next(t for t in TOOLS if t["name"] == "wait_for_job")
        self.assertEqual(spec["inputSchema"]["properties"]["timeout_seconds"]["maximum"], 60)
        self.assertNotIn("timeout_seconds", self.tools["wait_for_job"].parameters_schema()["properties"])

    def test_blocks_until_the_job_is_done_even_past_sixty_seconds_of_polling(self):
        gate = threading.Event()
        job = chat_job(lambda: (gate.wait(5) and ({"ok": True}, [])) or ({"ok": True}, []))
        threading.Timer(0.6, gate.set).start()
        started = time.monotonic()
        out = self.tools["wait_for_job"].execute_structured(job_id=job.id)
        self.assertGreaterEqual(time.monotonic() - started, 0.5)  # it waited; it did not answer "not finished"
        self.assertEqual(out.data["state"], "done")
        self.assertNotIn("next", out.data)
        self.assertIn("waitedSeconds", out.data)

    def test_a_model_timeout_argument_is_ignored(self):
        job = chat_job(lambda: ({}, []))
        job.done.wait(5)
        out = self.tools["wait_for_job"].execute_structured(job_id=job.id, timeout_seconds=1)
        self.assertEqual(out.data["state"], "done")

    def test_stop_in_the_chat_ends_the_wait_promptly_and_leaves_the_job_running(self):
        release = threading.Event()
        job = chat_job(lambda: (release.wait(10) and None) or ({}, []))
        stopped = threading.Event()
        tool = next(t for t in studio_tools(APP, is_cancelled=stopped.is_set) if t.name == "wait_for_job")
        threading.Timer(0.3, stopped.set).start()
        started = time.monotonic()
        try:
            out = tool.execute_structured(job_id=job.id)
            elapsed = time.monotonic() - started
        finally:
            release.set()
        self.assertLess(elapsed, 3.0)
        self.assertIn("keeps running", out.data["next"])
        self.assertIn(out.data["state"], ("queued", "running"))

    def test_cap_passes_with_a_still_running_answer(self):
        release = threading.Event()
        job = chat_job(lambda: (release.wait(10) and None) or ({}, []))
        try:
            with mock.patch.object(studio_module, "CHAT_WAIT_CAP_SECONDS", 0.6):
                out = self.tools["wait_for_job"].execute_structured(job_id=job.id)
        finally:
            release.set()
        self.assertIn("Still running", out.data["next"])

    def test_live_progress_reports_the_job_while_the_wait_blocks(self):
        release = threading.Event()
        job = chat_job(lambda: (release.wait(10) and None) or ({}, []))
        time.sleep(0.2)
        try:
            tool = self.tools["wait_for_job"]
            self.assertTrue(tool.long_running({"job_id": job.id}))
            snap = tool.live_progress({"job_id": job.id})
        finally:
            release.set()
        self.assertEqual(snap["jobId"], job.id)
        self.assertIn("elapsedSeconds", snap["progress"])

    def test_only_wait_for_job_is_long_running(self):
        self.assertFalse(self.tools["job_status"].long_running({}))
        self.assertIsNone(self.tools["job_status"].budget_key({"job_id": "job-1"}))
        self.assertEqual(self.tools["wait_for_job"].budget_key({"job_id": "job-1"}), "job-1")


class IterationBudgetTests(unittest.TestCase):
    def run_loop(self, replies, max_iterations, reg=None):
        reg = reg or ToolRegistry()
        if reg.get("wait_for_job") is None:
            reg.register(WaitTool())
        it = iter(replies)
        return list(run_agent_loop_streaming(
            generate_fn=lambda **kw: next(it), stream_generate_fn=None, prompt="go", history=[],
            system_prompt=None, max_tokens=64, temperature=0.1, tool_registry=reg, max_iterations=max_iterations,
        ))

    def test_repeat_waits_on_one_job_do_not_use_the_budget(self):
        waits = [reply("", [call("wait_for_job", {"job_id": "job-11"}, f"c{i}")]) for i in range(24)]
        events = self.run_loop(waits + [reply("The film is done.")], max_iterations=3)
        self.assertNotIn("(Reached maximum tool-use iterations)", "".join(e.get("token", "") for e in events))
        self.assertEqual("".join(e.get("token", "") for e in events), "The film is done.")
        self.assertEqual(events[-1]["iterations"], 25)

    def test_the_first_wait_on_each_job_counts_once(self):
        replies = [reply("", [call("wait_for_job", {"job_id": "job-1"}, "a")]),
                   reply("", [call("wait_for_job", {"job_id": "job-2"}, "b")]),
                   reply("", [call("wait_for_job", {"job_id": "job-3"}, "c")]),
                   reply("never reached")]
        events = self.run_loop(replies, max_iterations=3)
        self.assertIn("(Reached maximum tool-use iterations)", "".join(e.get("token", "") for e in events))

    def test_other_tools_still_hit_the_limit(self):
        class Echo(WaitTool):
            @property
            def name(self):
                return "echo"

            def budget_key(self, arguments):
                return None

        reg = ToolRegistry()
        reg.register(Echo())
        events = self.run_loop([reply("", [call("echo", {"job_id": "x"}, f"e{i}")]) for i in range(10)], 4, reg)
        self.assertIn("(Reached maximum tool-use iterations)", "".join(e.get("token", "") for e in events))

    def test_a_model_that_polls_forever_still_ends(self):
        from backend_service import agent

        waits = [reply("", [call("wait_for_job", {"job_id": "job-1"}, f"c{i}")]) for i in range(agent._FREE_WAIT_LIMIT + 10)]
        events = self.run_loop(waits, max_iterations=3)
        self.assertIn("(Reached maximum tool-use iterations)", "".join(e.get("token", "") for e in events))


class LongToolProgressTests(unittest.TestCase):
    def test_progress_events_stream_while_the_wait_runs_and_calls_carry_a_start_time(self):
        class Slow(WaitTool):
            def long_running(self, arguments):
                return True

            def live_progress(self, arguments):
                return {"state": "running", "progress": {"elapsedSeconds": 3}}

            def execute_structured(self, **kwargs):
                time.sleep(0.5)
                return StructuredToolOutput(text="{}", render_as="studio", data={"tool": "wait_for_job", "state": "done"})

        reg = ToolRegistry()
        reg.register(Slow())
        replies = iter([reply("", [call("wait_for_job", {"job_id": "job-1"}, "c1")]), reply("done")])
        with mock.patch("backend_service.agent._PROGRESS_SECONDS", 0.1):
            events = list(run_agent_loop_streaming(
                generate_fn=lambda **kw: next(replies), stream_generate_fn=None, prompt="go", history=[],
                system_prompt=None, max_tokens=64, temperature=0.1, tool_registry=reg,
            ))
        progress = [e["tool_call_progress"] for e in events if "tool_call_progress" in e]
        self.assertGreaterEqual(len(progress), 2)
        self.assertEqual(progress[0]["id"], "c1")
        start = next(e["tool_call_start"] for e in events if "tool_call_start" in e)
        res = next(e["tool_call_result"] for e in events if "tool_call_result" in e)
        self.assertAlmostEqual(start["startedAt"] / 1000, time.time(), delta=30)
        self.assertEqual(res["startedAt"], start["startedAt"])
        self.assertEqual(events[-1]["tool_calls"][0]["startedAt"], start["startedAt"])


class JobProgressTests(unittest.TestCase):
    def job(self, **kw):
        job = Job("job-x", kw.pop("kind", "voice"), "t", None, **kw)
        job.state, job.started_at = "running", time.time() - 30
        return job

    def test_elapsed_only_without_a_basis(self):
        info = self.job().describe()["progress"]
        self.assertGreaterEqual(info["elapsedSeconds"], 30)
        self.assertNotIn("etaSeconds", info)

    def test_catalog_estimate_is_labelled(self):
        info = self.job(estimate_seconds=100).describe()["progress"]
        self.assertEqual(info["etaBasis"], "catalog")
        self.assertAlmostEqual(info["etaSeconds"], 70, delta=3)

    def test_no_eta_once_the_estimate_is_overrun(self):
        self.assertNotIn("etaSeconds", self.job(estimate_seconds=10).describe()["progress"])

    def test_steps_pace_gives_an_eta(self):
        snap = {"phase": "diffusing", "step": 4, "totalSteps": 8}
        with mock.patch("backend_service.mcp_bridge.jobs.IMAGE_PROGRESS") as tracker, \
                mock.patch.dict("backend_service.mcp_bridge.jobs._TRACKERS", {"image": tracker}):
            tracker.snapshot.return_value = snap
            tracker.phase_elapsed.return_value = 20.0
            info = self.job(kind="image").describe()["progress"]
        self.assertEqual((info["phase"], info["step"], info["totalSteps"]), ("diffusing", 4, 8))
        self.assertEqual((info["etaSeconds"], info["etaBasis"]), (20, "steps"))

    def test_own_progress_function_wins_and_a_broken_one_is_ignored(self):
        info = self.job(progress_fn=lambda: {"phase": "clip 2 of 4", "step": 2, "totalSteps": 4, "etaSeconds": 500,
                                             "etaBasis": "clips"}).describe()["progress"]
        self.assertEqual((info["phase"], info["etaSeconds"]), ("clip 2 of 4", 500))

        def broken():
            raise RuntimeError("x")

        self.assertIn("elapsedSeconds", self.job(progress_fn=broken).describe()["progress"])

    def test_finished_job_reports_its_duration(self):
        job = self.job()
        job.state, job.finished_at = "done", job.started_at + 42
        self.assertEqual(job.describe()["durationSeconds"], 42.0)


class BatchProgressTests(unittest.TestCase):
    def batch(self, steps, finished, current, processed):
        from backend_service.storyboard.batch import Batch

        b = Batch.__new__(Batch)
        b.steps, b._finished_steps, b._current, b._processed = steps, finished, current, processed
        b._began = time.monotonic() - 400
        return b

    def test_first_clip_running_has_no_eta_then_later_clips_use_the_measured_time(self):
        steps = [(f"s{i}", "clip") for i in range(1, 5)]
        now = time.monotonic()
        first = self.batch(steps, [], ("s1", "clip", now - 100), 0).progress()
        self.assertEqual((first["phase"], first["step"], first["totalSteps"]), ("clip 1 of 4", 1, 4))
        self.assertNotIn("etaSeconds", first)  # nothing measured yet: no invented time
        second = self.batch(steps, [("clip", 369.0)], ("s2", "clip", now - 69), 1).progress()
        self.assertEqual(second["phase"], "clip 2 of 4")
        self.assertEqual(second["etaBasis"], "clips")
        self.assertAlmostEqual(second["etaSeconds"], 300 + 2 * 369, delta=3)  # rest of clip 2 plus clips 3 and 4
        self.assertEqual(second["finishedSeconds"], {"clip": 369.0})

    def test_a_kind_without_a_measurement_blocks_the_eta(self):
        steps = [("s1", "still"), ("s1", "clip")]
        info = self.batch(steps, [("still", 20.0)], ("s1", "clip", time.monotonic()), 1).progress()
        self.assertNotIn("etaSeconds", info)  # a clip is not priced by a still

    def test_storyboard_batch_runs_report_progress_through_the_job(self):
        from backend_service.mcp_bridge import storyboard_tools

        src = open(storyboard_tools.__file__, encoding="utf-8").read()
        self.assertIn("progress=batch.progress", src)


class ChatTurnTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        from backend_service.state import ChaosEngineState
        from tests.test_chat_studio_tools import ScriptedRuntime
        from tests.test_backend_service import fake_library, fake_system_snapshot, make_test_client

        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.sessions_path = root / "chat-sessions.json"
        # A session saved by an older build: no createdAt on messages, no startedAt on tool calls.
        self.sessions_path.write_text(json.dumps([{
            "id": "old", "title": "Old chat", "updatedAt": "2026-01-01", "model": "m", "modelRef": "m",
            "messages": [{"role": "user", "text": "hi", "metrics": None},
                         {"role": "assistant", "text": "yo", "metrics": None,
                          "toolCalls": [{"id": "c", "name": "n", "arguments": {}, "result": "", "elapsed": 1}]}],
        }]), encoding="utf-8")
        state = ChaosEngineState(system_snapshot_provider=fake_system_snapshot, library_provider=fake_library,
                                 settings_path=root / "settings.json", benchmarks_path=root / "bench.json",
                                 chat_sessions_path=self.sessions_path)
        self.runtime = ScriptedRuntime()
        state.runtime = self.runtime
        self.client = make_test_client(state)
        self.client.post("/api/models/load", json={"modelRef": "google/gemma-4-E4B-it", "modelName": "Gemma",
                                                   "source": "catalog", "backend": "mock"})
        self.session = self.client.post("/api/chat/sessions", json={"title": "t"}).json()["session"]

    def tearDown(self):
        self.tempdir.cleanup()

    def stream(self, **body):
        response = self.client.post("/api/chat/generate/stream", json={"sessionId": self.session["id"], "prompt": "go", **body})
        return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]

    def test_new_messages_are_stamped_and_old_sessions_load_unchanged(self):
        before = int(time.time() * 1000)
        self.runtime.replies = [reply("hello")]
        self.stream()
        sessions = {s["id"]: s for s in self.client.get("/api/workspace").json()["chatSessions"]}
        messages = sessions[self.session["id"]]["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        for message in messages:
            self.assertGreaterEqual(message["createdAt"], before)
            self.assertLessEqual(message["createdAt"], int(time.time() * 1000))
        old = sessions["old"]["messages"]
        self.assertTrue(all("createdAt" not in m for m in old))
        self.assertNotIn("startedAt", old[1]["toolCalls"][0])

    def test_a_studio_wait_streams_progress_and_one_wait_ends_the_turn_with_a_summary(self):
        from backend_service.mcp_bridge.tools import HANDLERS

        def fake_image(ctx, args):
            job = chat_job(lambda: (time.sleep(0.6) or ({"path": None}, [])))
            return {"jobId": job.id, "state": "queued", "label": "poster"}

        self.runtime.replies = [
            reply("", [call("generate_image", {"prompt": "rose"}, "c1")]),
            reply("", [call("wait_for_job", {"job_id": "PLACEHOLDER"}, "c2")]),
            reply("The poster is done."),
        ]
        original = self.runtime.generate

        def generate(**kwargs):
            out = original(**kwargs)
            for tc in out.tool_calls or []:  # the job id exists only once generate_image ran
                if tc["function"]["name"] == "wait_for_job":
                    jid = max((j.id for j in STORE.list(None, 100) if j.owner == CHAT_OWNER), key=lambda i: int(i.split("-")[1]))
                    tc["function"]["arguments"] = json.dumps({"job_id": jid})
            return out

        self.runtime.generate = generate
        with mock.patch.dict(HANDLERS, {"generate_image": fake_image}), \
                mock.patch("backend_service.agent._PROGRESS_SECONDS", 0.1):
            events = self.stream(studioTools=True, studioAutoApprove=True)
        progress = [e["toolCallProgress"] for e in events if "toolCallProgress" in e]
        self.assertTrue(progress)
        self.assertEqual(progress[0]["name"], "wait_for_job")
        self.assertIn("elapsedSeconds", progress[0]["data"]["progress"])
        starts = [e["toolCallStart"] for e in events if "toolCallStart" in e]
        self.assertTrue(all(isinstance(s["startedAt"], int) for s in starts))
        wait_result = next(e["toolCallResult"] for e in events if "toolCallResult" in e and e["toolCallResult"]["name"] == "wait_for_job")
        self.assertEqual(wait_result["data"]["state"], "done")
        self.assertIn("The poster is done.", "".join(e.get("token", "") for e in events))

    def test_stop_in_the_chat_ends_a_waiting_turn_quickly(self):
        release = threading.Event()
        job = chat_job(lambda: (release.wait(15) and None) or ({}, []))
        self.runtime.replies = [reply("", [call("wait_for_job", {"job_id": job.id}, "c1")]), reply("unreachable")]
        done = {}

        def run():
            done["events"] = self.stream(studioTools=True)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        time.sleep(1.0)
        started = time.monotonic()
        self.client.post(f"/api/chat/generate/{self.session['id']}/cancel")
        thread.join(8)
        release.set()
        self.assertFalse(thread.is_alive(), "the stream must end after Stop")
        self.assertLess(time.monotonic() - started, 6)
        self.assertNotIn("unreachable", json.dumps(done["events"]))
