"""Structured model events (D-88): the log, its limits, redaction, the controller hooks and the routes."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from backend_service import model_events
from backend_service.app import create_app
from backend_service.inference import BackendCapabilities, RuntimeController
from backend_service.inference.base import GenerationResult, LoadedModelInfo, StreamChunk
from backend_service.model_events import ModelEventLog, engine_environment, redact
from backend_service.state import ChaosEngineState
from tests.test_route_contracts import _FakeRuntime
from tests.test_setup_routes import TEST_API_TOKEN, _fake_system_snapshot


class EventLogTests(unittest.TestCase):
    def test_emit_shape_and_increasing_ids(self):
        log = ModelEventLog()
        first = log.emit("requested", model="Qwen", model_ref="qwen/q", request_id="r1", engine="mlx",
                         cache_strategy="native", message="Load requested.", data={"backend": "auto"})
        second = log.emit("loaded", level="info", model="Qwen", data={"seconds": 1.5})
        for key in ("id", "ts", "time", "kind", "level", "model", "modelRef", "requestId", "engine",
                    "cacheStrategy", "speculative", "message", "data", "detail"):
            self.assertIn(key, first)
        self.assertEqual(first["kind"], "requested")
        self.assertEqual(second["id"], first["id"] + 1)
        self.assertEqual([e["kind"] for e in log.events()], ["requested", "loaded"])

    def test_ring_is_bounded(self):
        log = ModelEventLog(capacity=5)
        for index in range(12):
            log.emit("generation", data={"n": index})
        events = log.events()
        self.assertEqual(len(events), 5)
        self.assertEqual([e["data"]["n"] for e in events], [7, 8, 9, 10, 11])

    def test_filters_and_limit(self):
        log = ModelEventLog()
        log.emit("loaded", model="A", request_id="x")
        log.emit("load_failed", level="error", model="B", request_id="y")
        log.emit("generation", model="A", request_id="z")
        self.assertEqual([e["kind"] for e in log.events(model="A")], ["loaded", "generation"])
        self.assertEqual([e["kind"] for e in log.events(level="error")], ["load_failed"])
        self.assertEqual([e["kind"] for e in log.events(request_id="z")], ["generation"])
        self.assertEqual([e["kind"] for e in log.events(kind="loaded")], ["loaded"])
        self.assertEqual([e["kind"] for e in log.events(limit=1)], ["generation"])
        self.assertEqual(log.events(limit=0), [])
        first_id = log.events()[0]["id"]
        self.assertEqual(len(log.events(since=first_id)), 2)

    def test_disk_rotation_keeps_a_bounded_number_of_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logs" / "model-events.jsonl"
            log = ModelEventLog(path, max_bytes=2048, backups=2)
            for index in range(400):
                log.emit("generation", model="M", message="x" * 40, data={"n": index})
            self.assertTrue(log.flush(5.0))
            names = {p.name for p in path.parent.iterdir()}
            # The live file is missing for a moment right after a rotation; never more than two backups.
            self.assertTrue({"model-events.jsonl.1", "model-events.jsonl.2"} <= names, names)
            self.assertTrue(names <= {"model-events.jsonl", "model-events.jsonl.1", "model-events.jsonl.2"}, names)
            total = sum(p.stat().st_size for p in path.parent.iterdir())
            self.assertLess(total, 3 * (2048 + 1024))
            self.assertEqual(log.dropped, 0)

    def test_full_disk_queue_drops_the_disk_copy_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = ModelEventLog(Path(tmp) / "e.jsonl", queue_size=3)
            with mock.patch.object(log, "_ensure_writer"):
                for _ in range(10):
                    log.emit("generation")
            self.assertEqual(len(log.events()), 10)
            self.assertEqual(log.dropped, 7)

    def test_history_survives_a_restart_and_ids_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.jsonl"
            log = ModelEventLog(path)
            for kind in ("requested", "loading", "loaded"):
                log.emit(kind, model="M")
            self.assertTrue(log.flush())
            again = ModelEventLog(path)
            self.assertEqual([e["kind"] for e in again.events()], ["requested", "loading", "loaded"])
            self.assertEqual(again.emit("idle")["id"], 4)

    def test_subscribers_get_new_events_and_full_queues_do_not_block(self):
        log = ModelEventLog()
        q = log.subscribe(maxsize=1)
        log.emit("loaded")
        log.emit("idle")  # queue full: dropped for this subscriber, still in the ring
        self.assertEqual(q.get_nowait()["kind"], "loaded")
        self.assertEqual(len(log.events()), 2)
        log.unsubscribe(q)
        log.unsubscribe(q)  # twice is harmless


class RedactionTests(unittest.TestCase):
    def test_tokens_and_secret_paths_are_hidden(self):
        home = str(Path.home())
        text = (
            "token hf_abcdefghijklmnopqrstuv and Authorization: Bearer abc.def-123 "
            "key sk-ABCDEFGH12345678 --api-key s3cr3t --hf-token=hf_zzzzzzzzzzzz "
            f"url ?token=abc123&x=1 file {home}/.cache/huggingface/token and {home}/.ssh/id_ed25519 "
            f"model {home}/models/qwen.gguf"
        )
        out = redact(text)
        for secret in ("hf_abcdefghijklmnopqrstuv", "abc.def-123", "sk-ABCDEFGH12345678", "s3cr3t",
                       "hf_zzzzzzzzzzzz", "abc123", "id_ed25519", "huggingface/token"):
            self.assertNotIn(secret, out)
        self.assertNotIn(home, out)
        self.assertIn("~/models/qwen.gguf", out)
        self.assertIn("<secret path>", out)

    def test_emit_redacts_message_model_and_detail(self):
        log = ModelEventLog()
        event = log.emit(
            "load_failed",
            model=f"{Path.home()}/m",
            message="failed with Bearer abcdef123",
            detail={"command": "llama-server --api-key topsecret -m x", "stderr": ["hf_abcdefghijklmnop leaked"]},
        )
        self.assertEqual(event["model"], "~/m")
        self.assertNotIn("abcdef123", event["message"])
        self.assertNotIn("topsecret", event["detail"]["command"])
        self.assertNotIn("hf_abcdefghijklmnop", event["detail"]["stderr"][0])

    def test_environment_is_allow_listed_and_secret_names_hidden(self):
        lines = engine_environment({
            "CHAOSENGINE_PORT": "8904",
            "HF_TOKEN": "hf_abcdefghijklmnop",
            "CHAOSENGINE_API_KEY": "abc",
            "HOME": "/Users/x",
            "AWS_SECRET_ACCESS_KEY": "nope",
        })
        self.assertIn("CHAOSENGINE_PORT=8904", lines)
        self.assertIn("HF_TOKEN=***", lines)
        self.assertIn("CHAOSENGINE_API_KEY=***", lines)
        self.assertFalse(any(line.startswith(("HOME=", "AWS_")) for line in lines))


def _info(ref: str = "model-a", *, engine: str = "dummy", speculative: bool = False, tree: int = 0) -> LoadedModelInfo:
    return LoadedModelInfo(
        ref=ref, name=ref, backend="mlx", source="catalog", engine=engine, cacheStrategy="native", cacheBits=0,
        fp16Layers=0, fusedAttention=False, fitModelInMemory=True, contextTokens=8192, loadedAt="now",
        speculativeDecoding=speculative, treeBudget=tree,
    )


class _Engine:
    engine_name = "dummy"
    engine_label = "Dummy"

    def __init__(self, *, fail_load: bool = False, fail_generate: bool = False) -> None:
        self.fail_load = fail_load
        self.fail_generate = fail_generate
        self.last_command = ["/bin/llama-server", "-m", "/x.gguf", "--api-key", "secretvalue"]
        self.unload_calls = 0

    def load_model(self, **kwargs):
        if self.fail_load:
            raise RuntimeError("worker exited: out of memory\nline two")
        return _info(str(kwargs["model_ref"]))

    def unload_model(self):
        self.unload_calls += 1

    def process_pid(self):
        return None

    def generate(self, **kwargs):
        if self.fail_generate:
            raise RuntimeError("engine died")
        return GenerationResult(text="hi", finishReason="stop", promptTokens=5, completionTokens=7, totalTokens=12,
                                tokS=42.0, responseSeconds=0.2)

    def stream_generate(self, **kwargs):
        yield StreamChunk(text="he")
        yield StreamChunk(text="llo")
        yield StreamChunk(done=True, finish_reason="stop", prompt_tokens=5, completion_tokens=2, tok_s=30.0)


def _controller(log: ModelEventLog) -> RuntimeController:
    caps = BackendCapabilities(pythonExecutable="/usr/bin/python3", mlxAvailable=False, mlxLmAvailable=False,
                               mlxUsable=False, ggufAvailable=False)
    with mock.patch("backend_service.inference.get_backend_capabilities", return_value=caps):
        controller = RuntimeController()
    controller.refresh_capabilities = mock.Mock(return_value=caps)
    controller.events = log
    return controller


def _load(controller: RuntimeController, ref: str, **extra):
    return controller.load_model(
        model_ref=ref, model_name=ref, source="catalog", backend="mlx", path=None, runtime_target=ref,
        cache_strategy="native", cache_bits=0, fp16_layers=0, fused_attention=False, fit_model_in_memory=True,
        context_tokens=8192, **extra,
    )


class ControllerEventTests(unittest.TestCase):
    def test_load_emits_requested_resolving_loading_loaded_with_one_request_id(self):
        log = ModelEventLog()
        controller = _controller(log)
        controller._select_engine = mock.Mock(return_value=_Engine())
        _load(controller, "model-a")
        events = log.events()
        self.assertEqual([e["kind"] for e in events], ["requested", "resolving", "loading", "loaded"])
        self.assertEqual(len({e["requestId"] for e in events}), 1)
        loaded = events[-1]
        self.assertEqual(loaded["engine"], "dummy")
        self.assertIn("seconds", loaded["data"])
        self.assertIn("memoryBytes", loaded["data"])
        self.assertFalse(loaded["data"]["warm"])
        self.assertIn("--api-key ***", loaded["detail"]["command"])

    def test_load_failure_emits_load_failed_with_error_detail(self):
        log = ModelEventLog()
        controller = _controller(log)
        controller._select_engine = mock.Mock(return_value=_Engine(fail_load=True))
        with self.assertRaises(RuntimeError):
            _load(controller, "model-a")
        failed = log.events()[-1]
        self.assertEqual(failed["kind"], "load_failed")
        self.assertEqual(failed["level"], "error")
        self.assertEqual(failed["message"], "worker exited: out of memory")
        self.assertIn("out of memory", failed["detail"]["error"])
        self.assertEqual(failed["detail"]["stderr"], ["worker exited: out of memory", "line two"])

    def test_swap_and_user_unload_carry_a_reason(self):
        log = ModelEventLog()
        controller = _controller(log)
        controller._select_engine = mock.Mock(side_effect=[_Engine(), _Engine()])
        _load(controller, "model-a")
        _load(controller, "model-b", keep_warm_previous=False)
        swap = [e for e in log.events(kind="unloaded")]
        self.assertEqual([(e["model"], e["data"]["reason"]) for e in swap], [("model-a", "swap")])
        controller.unload_model()
        last = log.events(kind="unloaded")[-1]
        self.assertEqual((last["model"], last["data"]["reason"]), ("model-b", "user"))

    def test_parking_a_model_warm_is_an_event(self):
        log = ModelEventLog()
        controller = _controller(log)
        controller._select_engine = mock.Mock(side_effect=[_Engine(), _Engine()])
        controller._memory_budget_bytes = mock.Mock(return_value=None)
        _load(controller, "model-a")
        _load(controller, "model-b")
        self.assertEqual([e["model"] for e in log.events(kind="parked")], ["model-a"])
        # Switching back to the warm model is a load with no engine start.
        _load(controller, "model-a")
        self.assertTrue(log.events(kind="loaded")[-1]["data"]["warm"])

    def test_generate_emits_generation_then_idle(self):
        log = ModelEventLog()
        controller = _controller(log)
        controller.engine = _Engine()
        controller.loaded_model = _info("model-a", speculative=True, tree=4)
        controller.generate(prompt="hi", history=[], system_prompt=None, max_tokens=16, temperature=0.2)
        kinds = [e["kind"] for e in log.events()]
        self.assertEqual(kinds, ["generation", "idle"])
        generation = log.events(kind="generation")[0]
        self.assertEqual(generation["data"]["tokS"], 42.0)
        self.assertEqual(generation["data"]["completionTokens"], 7)
        self.assertEqual(generation["speculative"], "ddtree")

    def test_stream_emits_first_token_with_prefill_then_generation_then_idle(self):
        log = ModelEventLog()
        controller = _controller(log)
        controller.engine = _Engine()
        controller.loaded_model = _info("model-a")
        chunks = list(controller.stream_generate(prompt="hi", history=[], system_prompt=None, max_tokens=16,
                                                 temperature=0.2))
        self.assertEqual("".join(c.text or "" for c in chunks), "hello")
        events = log.events()
        self.assertEqual([e["kind"] for e in events], ["first_token", "generation", "idle"])
        self.assertIn("prefillSeconds", events[0]["data"])
        self.assertIn("prefillSeconds", events[1]["data"])
        self.assertEqual(len({e["requestId"] for e in events}), 1)

    def test_stream_stopped_early_is_reported_cancelled(self):
        log = ModelEventLog()
        controller = _controller(log)
        controller.engine = _Engine()
        controller.loaded_model = _info("model-a")
        stream = controller.stream_generate(prompt="hi", history=[], system_prompt=None, max_tokens=16, temperature=0.2)
        next(stream)
        stream.close()
        generation = log.events(kind="generation")[-1]
        self.assertEqual(generation["data"]["finishReason"], "cancelled")
        self.assertEqual(log.events()[-1]["kind"], "idle")

    def test_generation_failure_is_an_error_event(self):
        log = ModelEventLog()
        controller = _controller(log)
        controller.engine = _Engine(fail_generate=True)
        controller.loaded_model = _info("model-a")
        with self.assertRaises(RuntimeError):
            controller.generate(prompt="hi", history=[], system_prompt=None, max_tokens=16, temperature=0.2)
        self.assertEqual([e["kind"] for e in log.events()], ["generation_failed", "idle"])
        self.assertEqual(log.events()[0]["level"], "error")

    def test_overlapping_generations_go_idle_once(self):
        log = ModelEventLog()
        controller = _controller(log)
        controller.engine = _Engine()
        controller.loaded_model = _info("model-a")
        first = controller.stream_generate(prompt="a", history=[], system_prompt=None, max_tokens=4, temperature=0)
        next(first)
        controller.generate(prompt="b", history=[], system_prompt=None, max_tokens=4, temperature=0)
        self.assertEqual(log.events(kind="idle"), [])
        list(first)
        self.assertEqual(len(log.events(kind="idle")), 1)

    def test_no_log_attached_is_a_no_op(self):
        controller = _controller(ModelEventLog())
        controller.events = None
        controller.engine = _Engine()
        controller.loaded_model = _info("model-a")
        result = controller.generate(prompt="hi", history=[], system_prompt=None, max_tokens=16, temperature=0.2)
        self.assertEqual(result.text, "hi")


class SpeculativeMethodTests(unittest.TestCase):
    def test_methods(self):
        self.assertIsNone(model_events.speculative_method(_info()))
        self.assertEqual(model_events.speculative_method(_info(speculative=True)), "dflash")
        self.assertEqual(model_events.speculative_method(_info(speculative=True, tree=8)), "ddtree")
        self.assertEqual(model_events.speculative_method(_info(engine="llama.cpp", speculative=True)), "draft")
        self.assertEqual(model_events.speculative_method(_info(engine="mtplx", speculative=True)), "mtp")


class EventRouteTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            library_provider=lambda: [],
            settings_path=Path(self.tempdir.name) / "settings.json",
            benchmarks_path=Path(self.tempdir.name) / "benchmarks.json",
            chat_sessions_path=Path(self.tempdir.name) / "chats.json",
            library_cache_path=Path(self.tempdir.name) / "library_cache.json",
        )
        state.runtime = _FakeRuntime()
        self.state = state
        self.app = create_app(state=state, api_token=TEST_API_TOKEN)
        self.client = TestClient(self.app)
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})

    def tearDown(self):
        self.state.model_events.flush()
        self.tempdir.cleanup()

    def test_event_log_lives_in_the_data_directory(self):
        self.state.model_events.emit("loaded")
        self.assertTrue(self.state.model_events.flush())
        self.assertTrue((Path(self.tempdir.name) / "logs" / "model-events.jsonl").exists())

    def test_events_route_shape_and_filters(self):
        self.state.model_events.emit("loaded", model="A", request_id="r1")
        self.state.model_events.emit("load_failed", level="error", model="B", request_id="r2")
        resp = self.client.get("/api/server/events")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(set(body), {"events", "capacity", "dropped"})
        self.assertEqual([e["kind"] for e in body["events"]], ["loaded", "load_failed"])
        self.assertEqual(len(self.client.get("/api/server/events", params={"level": "error"}).json()["events"]), 1)
        self.assertEqual(len(self.client.get("/api/server/events", params={"model": "A"}).json()["events"]), 1)
        self.assertEqual(len(self.client.get("/api/server/events", params={"requestId": "r2"}).json()["events"]), 1)
        self.assertEqual(len(self.client.get("/api/server/events", params={"limit": 1}).json()["events"]), 1)
        self.assertEqual(self.client.get("/api/server/events", params={"limit": 99999}).status_code, 422)

    def test_events_route_needs_the_api_token(self):
        bare = TestClient(self.app)
        self.assertEqual(bare.get("/api/server/events").status_code, 401)

    def _first_chunks(self, path: str, count: int) -> list[str]:
        route = next(r for r in self.app.routes if getattr(r, "path", None) == path)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(chaosengine=self.state)))

        async def collect():
            response = await route.endpoint(request)
            out: list[str] = []
            iterator = response.body_iterator
            async for chunk in iterator:
                out.append(chunk if isinstance(chunk, str) else chunk.decode())
                if len(out) >= count:
                    break
            await iterator.aclose()
            return out

        return asyncio.run(collect())

    def test_event_stream_sends_backlog_then_keepalive(self):
        self.state.model_events.emit("loaded", model="A")
        self.state.model_events.emit("idle", model="A")
        chunks = self._first_chunks("/api/server/events/stream", 3)
        self.assertEqual([json.loads(c[6:])["kind"] for c in chunks[:2]], ["loaded", "idle"])
        self.assertEqual(chunks[2], ": keepalive\n\n")

    def test_log_stream_backlog_is_the_newest_lines_oldest_first(self):
        for index in range(80):
            self.state.add_log("server", "info", f"line {index}")
        chunks = self._first_chunks("/api/server/logs/stream", 50)
        messages = [json.loads(c[6:])["message"] for c in chunks if c.startswith("data: ")]
        self.assertEqual(messages[0], "line 30")
        self.assertEqual(messages[-1], "line 79")


if __name__ == "__main__":
    unittest.main()
