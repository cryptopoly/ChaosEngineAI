"""D-60: studio tools in Chat (switch, scopes, approve-the-plan gate, result trimming). No real models."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.agent import run_agent_loop_streaming
from backend_service.inference import GenerationResult
from backend_service.state import ChaosEngineState
from backend_service.tools import BaseTool, StructuredToolOutput, ToolRegistry, registry as base_registry
from backend_service.tools import studio_gate
from backend_service.tools.studio import (
    CHAT_OWNER,
    STUDIO_HINT,
    STUDIO_TOOL_NAMES,
    needs_approval,
    studio_registry,
    studio_tools,
    trim_result,
)
from backend_service.tools.studio_gate import DECLINED_TEXT, EDIT_TEXT, NO_ANSWER_TEXT, StudioGate, decide
from tests.test_backend_service import FakeRuntime, fake_library, fake_system_snapshot, make_test_client

APP = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(chaosengine=None)))


def call(name, args, call_id):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def result(text="", tool_calls=None):
    return GenerationResult(text=text, finishReason="stop", promptTokens=1, completionTokens=1, totalTokens=2,
                            tokS=1.0, responseSeconds=0.1, tool_calls=tool_calls)


def answer_when_pending(decisions_for, edit=False, always_allow=False, timeout=5.0):
    """Answer the next plan from another thread, as the app's click would."""
    seen = {}

    def run():
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with studio_gate._LOCK:
                plans = list(studio_gate._PLANS.values())
            if plans:
                plan = plans[0]
                seen["plan"] = plan
                decide(plan.id, decisions_for(plan), edit=edit, always_allow=always_allow)
                return
            time.sleep(0.01)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, seen


class ToolListTests(unittest.TestCase):
    def test_switch_off_keeps_the_chat_tools_only(self):
        names = set(base_registry.available_names())
        self.assertFalse(names & set(STUDIO_TOOL_NAMES))

    def test_switch_on_adds_the_studio_tools(self):
        reg = studio_registry(APP, base_registry, None)
        names = reg.available_names()
        for name in ("generate_image", "generate_video", "speak", "make_music", "make_sound", "storyboard_generate",
                     "storyboard_render", "music_library", "list_models", "system_status", "wait_for_job"):
            self.assertIn(name, names)
        self.assertIn("calculator", names)  # the usual chat tools stay
        only_studio = studio_registry(APP, base_registry, [])
        self.assertNotIn("calculator", only_studio.available_names())

    def test_no_cloned_voice_download_or_file_tools(self):
        names = {t.name for t in studio_tools(APP)}
        for banned in ("speak_cloned", "list_cloned_voices", "recommend_models", "upload_file", "upload_image",
                       "storyboard_export_otio", "transcribe"):
            self.assertNotIn(banned, names)
        self.assertTrue(all(t.provenance == "studio" for t in studio_tools(APP)))

    def test_schemas_drop_file_and_delete_actions_and_base64(self):
        tools = {t.name: t for t in studio_tools(APP)}
        project_actions = tools["storyboard_project"].parameters_schema()["properties"]["action"]["enum"]
        for dropped in ("export", "import", "delete"):
            self.assertNotIn(dropped, project_actions)
        self.assertIn("get", project_actions)
        self.assertNotIn("export_srt", tools["storyboard_captions"].parameters_schema()["properties"]["action"]["enum"])
        self.assertNotIn("delete", tools["music_library"].parameters_schema()["properties"]["action"]["enum"])
        self.assertNotIn("include_preview", tools["generate_image"].parameters_schema()["properties"])
        # The bridge's own schemas are untouched.
        from backend_service.mcp_bridge.tools import TOOLS

        bridge = next(t for t in TOOLS if t["name"] == "storyboard_project")
        self.assertIn("export", bridge["inputSchema"]["properties"]["action"]["enum"])

    def test_descriptions_stay_compact(self):
        for tool in studio_tools(APP):
            self.assertLess(len(tool.description), 320, tool.name)
        self.assertLess(len(STUDIO_HINT), 1000)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.tools = {t.name: t for t in studio_tools(APP)}

    def test_dropped_actions_are_refused(self):
        out = self.tools["storyboard_project"].execute_structured(action="export", project_id="p1")
        self.assertIn("must be one of", out.text)
        out = self.tools["music_library"].execute_structured(action="delete", track_id="t1")
        self.assertIn("must be one of", out.text)

    def test_files_scope_is_refused_even_past_the_schema(self):
        tool = self.tools["storyboard_project"]
        with mock.patch.object(tool, "_schema", {"type": "object", "properties": {
                "action": {"type": "string"}, "project_id": {"type": "string"}}}):
            out = tool.execute_structured(action="export", project_id="p1")
        self.assertIn("Chat cannot use storyboard_project", out.text)

    def test_jobs_are_owned_by_chat(self):
        seen = {}

        def handler(ctx, args):
            from backend_service.mcp_bridge.jobs import CURRENT_OWNER

            seen["owner"] = CURRENT_OWNER.get()
            return {"jobId": "job-9", "state": "queued", "previewBase64": "AAAA"}

        tool = self.tools["generate_image"]
        with mock.patch.object(tool, "_handler", handler):
            out = tool.execute_structured(prompt="a poster")
        self.assertEqual(seen["owner"], CHAT_OWNER)
        self.assertNotIn("previewBase64", out.text)
        self.assertEqual(out.render_as, "studio")
        self.assertEqual(out.data["jobId"], "job-9")

    def test_input_images_must_be_app_outputs_or_uploads(self):
        tool = self.tools["generate_image"]
        ran = []
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as other:
            inside, outside = Path(out) / "a.png", Path(other) / "id.png"
            inside.write_bytes(b"x")
            outside.write_bytes(b"x")
            with mock.patch("backend_service.mcp_bridge.files.output_roots", return_value=[Path(out).resolve()]), \
                    mock.patch.object(tool, "_handler", lambda ctx, args: ran.append(args) or {"jobId": "job-1"}):
                refused = tool.execute_structured(prompt="x", reference_images=[str(outside)])
                escaped = tool.execute_structured(prompt="x", reference_images=[f"{out}/../{Path(other).name}/id.png"])
                tool.execute_structured(prompt="x", reference_images=[str(inside), "upload:abc"])
        self.assertIn("not allowed in Chat", refused.text)
        self.assertIn("not allowed in Chat", escaped.text)
        self.assertEqual(len(ran), 1)
        video = self.tools["generate_video"]
        with mock.patch.object(video, "_handler", lambda ctx, args: ran.append(args) or {}):
            self.assertIn("not allowed in Chat", video.execute_structured(prompt="x", end_image="/etc/hosts").text)
        self.assertIn("output folders", tool.description)

    def test_job_tools_only_see_chat_jobs(self):
        from backend_service.mcp_bridge.jobs import CURRENT_OWNER, STORE

        mine_token = CURRENT_OWNER.set(CHAT_OWNER)
        try:
            mine = STORE.submit("image", "chat job", lambda: ({}, []))
        finally:
            CURRENT_OWNER.reset(mine_token)
        theirs = STORE.submit("image", "agent job", lambda: ({}, []))
        mine.done.wait(5)
        theirs.done.wait(5)
        for name in ("job_status", "wait_for_job", "cancel_job"):
            with self.subTest(name=name):
                out = self.tools[name].execute_structured(job_id=theirs.id)
                self.assertIn("Chat only sees the jobs it started", out.text)
        self.assertNotIn("Error", self.tools["job_status"].execute_structured(job_id=mine.id).text)
        listed = {j["jobId"] for j in self.tools["list_jobs"].execute_structured(limit=100).data["jobs"]}
        self.assertIn(mine.id, listed)
        self.assertNotIn(theirs.id, listed)

    def test_placeholder_fallback_fails_the_job(self):
        from backend_service.mcp_bridge.tools import _refuse_placeholder

        with self.assertRaisesRegex(RuntimeError, r"\(placeholder fallback\): no snapshot"):
            _refuse_placeholder({"activeEngine": "placeholder", "message": "no snapshot"})
        _refuse_placeholder({"activeEngine": "mflux"})

    def test_wait_for_job_defaults_to_a_long_wait(self):
        seen = {}
        tool = self.tools["wait_for_job"]
        with mock.patch.object(tool, "_handler", lambda ctx, args: seen.update(args) or {"state": "done"}), \
                mock.patch("backend_service.tools.studio._foreign_job_problem", return_value=None):
            tool.execute_structured(job_id="job-1")
        self.assertEqual(seen["timeout_seconds"], 60)

    def test_which_calls_need_approval(self):
        for name, args, expected in (
            ("generate_image", {"prompt": "x"}, True),
            ("generate_video", {"prompt": "x"}, True),
            ("speak", {"text": "x"}, True),
            ("make_music", {"caption": "x"}, True),
            ("make_sound", {"caption": "x"}, True),
            ("storyboard_generate", {"project_id": "p", "revision": 1}, True),
            ("storyboard_render", {"project_id": "p", "revision": 1}, True),
            ("storyboard_captions", {"action": "generate", "project_id": "p"}, True),
            ("storyboard_captions", {"action": "add", "project_id": "p"}, False),
            ("storyboard_project", {"action": "get", "project_id": "p"}, False),
            ("system_status", {}, False),
            ("list_models", {"kind": "image"}, False),
            ("list_jobs", {}, False),
            ("wait_for_job", {"job_id": "j"}, False),
        ):
            with self.subTest(name=name, args=args):
                self.assertEqual(needs_approval(name, args), expected)


class TrimTests(unittest.TestCase):
    def test_trim_drops_base64_and_downloads_and_cuts_lists(self):
        payload = {
            "jobId": "job-1",
            "output": {"path": "/x/a.png", "previewBase64": "A" * 5000, "download": {"url": "http://t"}},
            "notes": ["a", "b", "c", "d"],
            "models": [{"id": i} for i in range(30)],
            "prompt": "p" * 1000,
            "missing": None,
        }
        out = trim_result(payload)
        self.assertEqual(out["output"], {"path": "/x/a.png"})
        self.assertEqual(out["notes"], ["a", "b", "c"])
        self.assertEqual(len(out["models"]), 13)
        self.assertTrue(out["models"][-1].startswith("... 18 more"))
        self.assertLess(len(out["prompt"]), 410)
        self.assertNotIn("missing", out)


class GateTests(unittest.TestCase):
    def tearDown(self):
        with studio_gate._LOCK:
            studio_gate._PLANS.clear()

    def run_review(self, gate, calls):
        events = []
        gen = gate.review(calls)
        try:
            while True:
                events.append(next(gen))
        except StopIteration as stop:
            return events, stop.value or {}

    def test_read_only_calls_skip_the_gate(self):
        events, denials = self.run_review(StudioGate("s"), [call("system_status", {}, "c1"), call("list_models", {"kind": "image"}, "c2")])
        self.assertEqual((events, denials), ([], {}))

    def test_gate_blocks_until_approved(self):
        gate = StudioGate("s", poll_seconds=0.01)
        calls = [call("generate_image", {"prompt": "poster", "width": 1024, "height": 1536}, "c1"),
                 call("speak", {"text": "Fresh flowers daily"}, "c2")]
        thread, seen = answer_when_pending(lambda plan: {"c1": "approve", "c2": "approve"})
        events, denials = self.run_review(gate, calls)
        thread.join()
        self.assertEqual(denials, {})
        plan = events[0]["studio_plan"]
        self.assertEqual([i["kind"] for i in plan["items"]], ["image", "speech"])
        self.assertEqual(plan["items"][0]["size"], "1024x1536")
        self.assertEqual(plan["items"][0]["prompt"], "poster")
        self.assertEqual(events[-1]["studio_plan_resolved"]["decisions"], {"c1": "approve", "c2": "approve"})
        self.assertEqual(seen["plan"].id, plan["planId"])
        self.assertIsNone(studio_gate.pending(plan["planId"]))

    def test_deny_and_approve_this(self):
        gate = StudioGate("s", poll_seconds=0.01)
        calls = [call("generate_image", {"prompt": "a"}, "c1"), call("generate_image", {"prompt": "b"}, "c2"),
                 call("system_status", {}, "c3")]
        thread, _ = answer_when_pending(lambda plan: {"c2": "approve"})
        events, denials = self.run_review(gate, calls)
        thread.join()
        self.assertEqual(denials, {"c1": DECLINED_TEXT})
        self.assertEqual(len(events[0]["studio_plan"]["items"]), 2)  # the read-only call is not in the plan

    def test_edit_by_chat_declines_everything(self):
        gate = StudioGate("s", poll_seconds=0.01)
        thread, _ = answer_when_pending(lambda plan: {"c1": "approve"}, edit=True)
        _, denials = self.run_review(gate, [call("make_music", {"caption": "jingle"}, "c1")])
        thread.join()
        self.assertEqual(denials, {"c1": EDIT_TEXT})

    def test_always_allow_covers_the_rest_of_the_turn(self):
        gate = StudioGate("s", poll_seconds=0.01)
        thread, _ = answer_when_pending(lambda plan: {"c1": "approve"}, always_allow=True)
        self.run_review(gate, [call("generate_image", {"prompt": "a"}, "c1")])
        thread.join()
        events, denials = self.run_review(gate, [call("generate_image", {"prompt": "b"}, "c2")])
        self.assertEqual((events, denials), ([], {}))

    def test_auto_approve_from_the_chat_setting(self):
        events, denials = self.run_review(StudioGate("s", auto_approve=True), [call("generate_image", {"prompt": "a"}, "c1")])
        self.assertEqual((events, denials), ([], {}))

    def test_cancel_while_waiting_declines(self):
        gate = StudioGate("s", poll_seconds=0.01, is_cancelled=lambda: True)
        events, denials = self.run_review(gate, [call("generate_image", {"prompt": "a"}, "c1")])
        self.assertEqual(denials, {"c1": NO_ANSWER_TEXT})
        self.assertEqual(events[-1]["studio_plan_resolved"]["status"], "unanswered")

    def test_keepalive_while_waiting(self):
        gate = StudioGate("s", poll_seconds=0.01, keepalive_seconds=0.02, wait_timeout=0.15)
        events, _ = self.run_review(gate, [call("generate_image", {"prompt": "a"}, "c1")])
        self.assertTrue(any(e.get("keepalive") for e in events))

    def test_decide_unknown_or_twice(self):
        self.assertIsNone(decide("plan-nope", {}))

    def test_plan_estimates_and_memory_note(self):
        state = SimpleNamespace(runtime=SimpleNamespace(loaded_model=SimpleNamespace(name="Qwen3.8-27B")))
        row = {"id": "flux", "name": "FLUX.1-schnell", "sizeGb": 12.0, "estimatedSeconds": 40, "recommendedResolution": "1024x1024"}
        with mock.patch("backend_service.mcp_bridge.models.installed_models", return_value=[row]), \
                mock.patch("backend_service.mcp_bridge.models.default_model", return_value=row):
            item = studio_gate.plan_item(call("generate_image", {"prompt": "poster"}, "c1"), state)
        self.assertEqual((item["model"], item["estimatedSeconds"], item["peakGb"], item["size"]),
                         ("FLUX.1-schnell", 40, 12.0, "1024x1024"))
        note = studio_gate.memory_note([item], state)
        self.assertEqual(note["chatModel"], "Qwen3.8-27B")
        self.assertIsInstance(note["freeGb"], float)
        self.assertIn(note["level"], ("ok", "slower", "risky"))

    def test_plan_shows_arguments_that_change_the_render(self):
        item = studio_gate.plan_item(call("generate_image", {
            "prompt": "p" * 300, "reference_images": ["/Users/x/Desktop/id.jpg", "upload:abc"], "steps": 30}, "c1"), None)
        self.assertEqual(item["inputs"], ["id.jpg", "upload:abc"])
        self.assertEqual(item["steps"], 30)
        self.assertEqual(len(item["promptFull"]), 300)
        video = studio_gate.plan_item(call("generate_video", {"prompt": "x", "frames": 121, "start_image": "/o/a.png"}, "c2"), None)
        self.assertEqual((video["frames"], video["inputs"]), (121, ["a.png"]))
        music = studio_gate.plan_item(call("make_music", {"caption": "x", "variations": 4}, "c3"), None)
        self.assertEqual(music["variations"], 4)
        project = SimpleNamespace(shots=[SimpleNamespace(locked=False, takes=[], status="empty"),
                                         SimpleNamespace(locked=True, takes=[], status="empty")])
        store = SimpleNamespace(load=lambda pid: project)
        with mock.patch("backend_service.routes.storyboard.get_store", return_value=store), \
                mock.patch("backend_service.storyboard.batch.needs_work", return_value=True):
            board = studio_gate.plan_item(call("storyboard_generate", {"project_id": "p", "revision": 1}, "c4"), None)
            forced = studio_gate.plan_item(call("storyboard_generate", {"project_id": "p", "revision": 1, "force": True}, "c5"), None)
        self.assertEqual((board["shots"], forced["shots"], forced["force"]), (1, 2, True))

    def test_closed_stream_removes_the_waiting_plan(self):
        gate = StudioGate("s", poll_seconds=0.01, keepalive_seconds=0.01)
        gen = gate.review([call("generate_image", {"prompt": "a"}, "c1")])
        plan_id = next(gen)["studio_plan"]["planId"]
        self.assertIsNotNone(studio_gate.pending(plan_id))
        self.assertTrue(next(gen).get("keepalive"))
        gen.close()  # what a window reload does to the stream
        self.assertIsNone(studio_gate.pending(plan_id))
        self.assertIsNone(decide(plan_id, {"c1": "approve"}))


class FakeTool(BaseTool):
    def __init__(self, name):
        self._name = name
        self.calls = []

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._name

    def parameters_schema(self):
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return "ran"

    def execute_structured(self, **kwargs):
        self.calls.append(kwargs)
        return StructuredToolOutput(text=json.dumps({"jobId": f"job-{len(self.calls)}", "state": "queued"}), render_as="studio",
                                    data={"jobId": f"job-{len(self.calls)}"})


class AgentLoopTests(unittest.TestCase):
    def test_repeated_call_ids_get_their_own_approval(self):
        image = FakeTool("generate_image")
        reg = ToolRegistry()
        reg.register(image)
        replies = iter([
            result("", [call("generate_image", {"prompt": "a"}, "dup"), call("generate_image", {"prompt": "SECRET"}, "dup")]),
            result("done"),
        ])
        thread, seen = answer_when_pending(lambda plan: {plan.items[0]["callId"]: "approve"})
        events = list(run_agent_loop_streaming(
            generate_fn=lambda **kw: next(replies), stream_generate_fn=None, prompt="go", history=[], system_prompt=None,
            max_tokens=64, temperature=0.1, tool_registry=reg, gate=StudioGate("s", poll_seconds=0.01),
        ))
        thread.join()
        ids = [i["callId"] for i in seen["plan"].items]
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(image.calls, [{"prompt": "a"}])
        declined = [e["tool_call_result"] for e in events if "tool_call_result" in e and e["tool_call_result"]["result"] == DECLINED_TEXT]
        self.assertEqual(declined[0]["arguments"], {"prompt": "SECRET"})

    def test_declined_calls_do_not_run_and_the_model_reads_why(self):
        image, status = FakeTool("generate_image"), FakeTool("system_status")
        reg = ToolRegistry()
        reg.register(image)
        reg.register(status)
        replies = iter([
            result("Plan: two posters.", [call("generate_image", {"prompt": "a"}, "c1"), call("generate_image", {"prompt": "b"}, "c2"),
                                          call("system_status", {}, "c3")]),
            result("One poster is queued; you declined the other."),
        ])
        seen_history = []

        def generate_fn(**kwargs):
            seen_history.append(list(kwargs["history"]))
            return next(replies)

        thread, _ = answer_when_pending(lambda plan: {"c1": "approve", "c2": "deny"})
        events = list(run_agent_loop_streaming(
            generate_fn=generate_fn, stream_generate_fn=None, prompt="make posters", history=[], system_prompt=None,
            max_tokens=64, temperature=0.1, tool_registry=reg, gate=StudioGate("s", poll_seconds=0.01),
        ))
        thread.join()
        self.assertEqual(len(image.calls), 1)
        self.assertEqual(len(status.calls), 1)
        results = {e["tool_call_result"]["id"]: e["tool_call_result"] for e in events if "tool_call_result" in e}
        self.assertEqual(results["c2"]["result"], DECLINED_TEXT)
        self.assertTrue(results["c2"]["data"]["declined"])
        self.assertEqual(results["c2"]["arguments"], {"prompt": "b"})
        tool_msgs = [m for m in seen_history[-1] if m.get("role") == "tool"]
        self.assertIn(DECLINED_TEXT, [m["text"] for m in tool_msgs])
        kinds = [next(iter(e)) for e in events]
        self.assertLess(kinds.index("studio_plan"), kinds.index("tool_call_start"))
        self.assertIn("studio_plan_resolved", kinds)


class ScriptedRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.replies = []
        self.tools_seen = []
        self.system_prompts = []

    def generate(self, **kwargs):
        self.tools_seen.append([t["function"]["name"] for t in kwargs.get("tools") or []])
        self.system_prompts.append(kwargs.get("system_prompt") or "")
        return self.replies.pop(0) if self.replies else result("done")


class ChatStreamTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=root / "settings.json",
            benchmarks_path=root / "benchmark-history.json",
            chat_sessions_path=root / "chat-sessions.json",
        )
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

    def test_switch_off_offers_no_studio_tools(self):
        self.stream(enableTools=True)
        self.assertNotIn("generate_image", self.runtime.tools_seen[0])
        self.assertNotIn("Studio tools are on", self.runtime.system_prompts[0])

    def test_switch_on_offers_studio_tools_and_the_hint(self):
        self.stream(studioTools=True)
        self.assertIn("generate_image", self.runtime.tools_seen[0])
        self.assertNotIn("speak_cloned", self.runtime.tools_seen[0])
        self.assertNotIn("calculator", self.runtime.tools_seen[0])  # chat tools only when Tools is on too
        self.assertIn("Studio tools are on", self.runtime.system_prompts[0])
        both = self.runtime.tools_seen
        self.stream(studioTools=True, enableTools=True)
        self.assertIn("calculator", both[-1])

    def test_plan_card_approve_then_job_runs(self):
        from backend_service.mcp_bridge.tools import HANDLERS

        ran = []

        def fake_image(ctx, args):
            ran.append(args["prompt"])
            return {"jobId": f"job-{len(ran)}", "state": "queued", "label": "poster"}

        self.runtime.replies = [
            result("", [call("generate_image", {"prompt": "rose poster"}, "c1"), call("generate_image", {"prompt": "tulip poster"}, "c2")]),
            result("Both posters are queued."),
        ]

        def click():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with studio_gate._LOCK:
                    plans = list(studio_gate._PLANS)
                if plans:
                    self.decision = self.client.post(f"/api/chat/studio/plans/{plans[0]}/decision",
                                                     json={"decisions": {"c1": "approve", "c2": "deny"}})
                    return
                time.sleep(0.02)

        thread = threading.Thread(target=click, daemon=True)
        thread.start()
        with mock.patch.dict(HANDLERS, {"generate_image": fake_image}):
            events = self.stream(studioTools=True)
        thread.join()
        self.assertEqual(self.decision.status_code, 200)
        self.assertEqual(ran, ["rose poster"])
        plan = next(e["studioPlan"] for e in events if "studioPlan" in e)
        self.assertEqual(len(plan["items"]), 2)
        # The prefill phase ends before the plan, so the app's 60 s prompt watchdog cannot cancel it.
        plan_at = next(i for i, e in enumerate(events) if "studioPlan" in e)
        self.assertTrue(any(e.get("phase") == "generating" for e in events[:plan_at]))
        resolved = next(e["studioPlanResolved"] for e in events if "studioPlanResolved" in e)
        self.assertEqual(resolved["decisions"], {"c1": "approve", "c2": "deny"})
        results = [e["toolCallResult"] for e in events if "toolCallResult" in e]
        self.assertEqual(results[1]["result"], DECLINED_TEXT)
        done = next(e for e in events if e.get("done"))
        assistant = done["session"]["messages"][-1]
        self.assertEqual(assistant["studioPlans"][0]["planId"], plan["planId"])
        self.assertEqual(len(assistant["toolCalls"]), 2)
        # A late click finds nothing waiting.
        self.assertEqual(self.client.post(f"/api/chat/studio/plans/{plan['planId']}/decision", json={}).status_code, 404)

    def test_decision_route_needs_the_app_token(self):
        from backend_service.mcp_bridge import access

        anonymous = TestClient(self.client.app)
        self.assertEqual(anonymous.post("/api/chat/studio/plans/plan-x/decision", json={}).status_code, 401)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(access, "_path", lambda: Path(tmp) / "agent_access.json"):
            access.set_enabled(True)
            agent = {"Authorization": f"Bearer {access.get_token()}"}
            self.assertEqual(anonymous.post("/api/chat/studio/plans/plan-x/decision", json={}, headers=agent).status_code, 401)
            self.assertEqual(anonymous.get("/api/chat/studio/jobs/job-1", headers=agent).status_code, 401)
        self.assertEqual(self.client.post("/api/chat/studio/plans/plan-x/decision", json={}).status_code, 404)

    def test_closing_the_stream_cancels_a_waiting_plan(self):
        from backend_service.state import generation

        self.runtime.replies = [result("", [call("generate_image", {"prompt": "rose"}, "c1")]), result("ok")]
        captured = {}
        with mock.patch.object(generation, "StreamingResponse", lambda body, **kw: captured.setdefault("body", body)):
            self.client.app.state.chaosengine.generate_stream(
                generation.GenerateRequest(sessionId=self.session["id"], prompt="go", studioTools=True),
                app_request=SimpleNamespace(app=self.client.app))
        body = captured["body"]
        for chunk in body:
            if "studioPlan" in chunk:
                break
        with studio_gate._LOCK:
            self.assertEqual(len(studio_gate._PLANS), 1)
        body.close()  # the client went away
        with studio_gate._LOCK:
            self.assertEqual(studio_gate._PLANS, {})
        self.assertTrue(self.client.app.state.chaosengine.is_chat_cancel_requested(self.session["id"]))

    def test_auto_approve_runs_without_a_plan(self):
        from backend_service.mcp_bridge.tools import HANDLERS

        self.runtime.replies = [result("", [call("generate_image", {"prompt": "rose"}, "c1")]), result("ok")]
        with mock.patch.dict(HANDLERS, {"generate_image": lambda ctx, args: {"jobId": "job-1", "state": "queued"}}):
            events = self.stream(studioTools=True, studioAutoApprove=True)
        self.assertFalse(any("studioPlan" in e for e in events))
        self.assertEqual(json.loads(next(e["toolCallResult"] for e in events if "toolCallResult" in e)["result"])["jobId"], "job-1")

    def test_job_routes_only_show_chat_jobs(self):
        from backend_service.mcp_bridge.jobs import CURRENT_OWNER, STORE

        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "a.png"
            png.write_bytes(b"\x89PNG fake")
            token = CURRENT_OWNER.set(CHAT_OWNER)
            try:
                job = STORE.submit("image", "poster", lambda: ({"path": str(png), "previewBase64": "AAAA"}, []))
            finally:
                CURRENT_OWNER.reset(token)
            job.done.wait(5)
            other = STORE.submit("image", "agent job", lambda: ({"path": str(png)}, []))
            other.done.wait(5)
            roots = mock.patch("backend_service.mcp_bridge.files.output_roots", return_value=[Path(tmp).resolve()])
            roots.start()
            self.addCleanup(roots.stop)
            status = self.client.get(f"/api/chat/studio/jobs/{job.id}").json()["job"]
            self.assertEqual(status["state"], "done")
            self.assertNotIn("previewBase64", status["output"])
            self.assertNotIn("download", status["output"])
            self.assertEqual(self.client.get(f"/api/chat/studio/jobs/{job.id}/media").content, b"\x89PNG fake")
            self.assertEqual(self.client.get(f"/api/chat/studio/jobs/{other.id}").status_code, 404)
            self.assertEqual(self.client.get(f"/api/chat/studio/jobs/{job.id}/media?track=3").status_code, 404)
            self.assertFalse(self.client.post(f"/api/chat/studio/jobs/{job.id}/cancel").json()["cancelRequested"])


if __name__ == "__main__":
    unittest.main()
