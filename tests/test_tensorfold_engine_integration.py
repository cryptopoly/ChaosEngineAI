"""Integration tests for TensorFoldEngine using a stub ``tensorfold serve``.

Verifies spawn -> /health wait -> /v1/chat/completions round trips without
the real TensorFold install or any model weights. The stub
(``tests/fixtures/stub_tensorfold_server.py``) speaks the wire format of
TensorFold 0.5.0's own server, which was also checked directly by driving
this engine against ``tensorfold.server.http.make_handler``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.inference.base import BackendCapabilities
from backend_service.inference.tensorfold_engine import TensorFoldEngine

_STUB_SCRIPT = Path(__file__).parent / "fixtures" / "stub_tensorfold_server.py"
_QWEN_REPO = "Vontra/Qwen3.8-27B-MLX-4bit"
_GEMMA_REPO = "mlx-community/gemma-4-26b-a4b-it-4bit"
_DRAFTER_REPO = "z-lab/Qwen3.8-27B-DFlash2"


def _make_wrapper(directory: Path, *, fail_mode: str | None = None) -> Path:
    """An executable ``tensorfold`` that forwards argv to the python stub."""
    wrapper = directory / "tensorfold"
    extra = f" --fail-mode {fail_mode}" if fail_mode else ""
    wrapper.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{_STUB_SCRIPT}" "$@"{extra}\n')
    wrapper.chmod(0o755)
    return wrapper


def _write_checkpoint(directory: Path, *, model_type: str = "qwen3_5", window: int = 4096) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"model_type": model_type, "max_position_embeddings": window}))
    (directory / "model.safetensors").write_bytes(b"weights")


@unittest.skipIf(sys.platform == "win32", "TensorFold fixtures spawn a POSIX shell wrapper")
class TensorFoldEngineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.record = self.tmp / "requests.jsonl"
        self.wrapper = _make_wrapper(self.tmp)

        # A checkpoint laid out like a Hugging Face cache entry, so the repo
        # id is recoverable from its path the way library items are.
        self.repo_dir = self.tmp / "library" / f"models--{_QWEN_REPO.replace('/', '--')}"
        self.snapshot = self.repo_dir / "snapshots" / "rev1"
        _write_checkpoint(self.snapshot)

        # The hub cache the engine consults for the optional draft model.
        self.hub = self.tmp / "hub"
        self.hub.mkdir()
        env = mock.patch.dict(
            os.environ,
            {"HF_HUB_CACHE": str(self.hub), "STUB_TENSORFOLD_RECORD": str(self.record)},
        )
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)

        self.capabilities = BackendCapabilities(
            pythonExecutable=sys.executable,
            mlxAvailable=True,
            mlxLmAvailable=True,
            mlxUsable=True,
            tensorfoldAvailable=True,
            tensorfoldPythonPath=str(self.tmp / "python"),
        )
        self.engines: list[TensorFoldEngine] = []
        self.addCleanup(self._unload_all)

    def _unload_all(self) -> None:
        for engine in self.engines:
            engine.unload_model()
        self._tmp.cleanup()

    def _engine(self, wrapper: Path | None = None) -> TensorFoldEngine:
        engine = TensorFoldEngine(self.capabilities)
        target = str(wrapper or self.wrapper)
        engine._tensorfold_bin = lambda: target  # type: ignore[method-assign]
        self.engines.append(engine)
        return engine

    def _add_drafter(self, repo: str = _DRAFTER_REPO) -> Path:
        cache = self.hub / f"models--{repo.replace('/', '--')}"
        snapshot = cache / "snapshots" / "rev1"
        _write_checkpoint(snapshot)
        (cache / "refs").mkdir(parents=True)
        (cache / "refs" / "main").write_text("rev1")
        return snapshot

    def _load(self, engine: TensorFoldEngine, **overrides):
        kwargs = dict(
            model_ref=_QWEN_REPO,
            model_name="Qwen3.8 27B",
            canonical_repo=_QWEN_REPO,
            source="library",
            backend="tensorfold",
            path=str(self.repo_dir),
            runtime_target=_QWEN_REPO,
            cache_strategy="native",
            cache_bits=0,
            fp16_layers=0,
            fused_attention=False,
            fit_model_in_memory=True,
            context_tokens=8192,
            speculative_decoding=True,
        )
        kwargs.update(overrides)
        return engine.load_model(**kwargs)

    def _started(self) -> dict:
        return json.loads(Path(str(self.record) + ".start").read_text())

    def _requests(self) -> list[dict]:
        return [json.loads(line) for line in self.record.read_text().splitlines() if line.strip()]

    @staticmethod
    def _generate_kwargs(**overrides) -> dict:
        kwargs = dict(prompt="Hello", history=[], system_prompt=None, max_tokens=32, temperature=0.7)
        kwargs.update(overrides)
        return kwargs

    # ------------------------------------------------------------------
    # load_model
    # ------------------------------------------------------------------

    def test_load_starts_server_on_the_local_snapshot(self) -> None:
        drafter = self._add_drafter()
        engine = self._engine()
        info = self._load(engine)

        self.assertEqual(info.engine, "tensorfold")
        self.assertEqual(info.cacheStrategy, "native")
        self.assertEqual(info.cacheBits, 0)
        self.assertTrue(info.speculativeDecoding)
        self.assertEqual(info.dflashDraftModel, _DRAFTER_REPO)
        self.assertEqual(info.contextTokens, 8192)
        self.assertIn("exact speculative decoding", info.runtimeNote or "")
        self.assertIn(_DRAFTER_REPO, info.runtimeNote or "")
        self.assertIsNotNone(engine.process_pid())

        argv = self._started()["argv"]
        self.assertEqual(argv[0], "serve")
        # a directory, never a repo id: the server must not fetch weights itself
        self.assertEqual(argv[1], str(self.snapshot))
        self.assertIn("--no-update-check", argv)
        self.assertEqual(argv[argv.index("--snapshot-dir") + 1], "none")
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--name") + 1], _QWEN_REPO)
        self.assertEqual(argv[argv.index("--drafter") + 1], str(drafter))
        self.assertNotIn("--no-drafts", argv)

    def test_context_is_clamped_to_the_checkpoints_window(self) -> None:
        engine = self._engine()
        info = self._load(engine, context_tokens=32768)
        argv = self._started()["argv"]
        self.assertEqual(argv[argv.index("--context") + 1], "4096")
        self.assertEqual(info.contextTokens, 32768)  # the requested value keeps the warm-pool key stable
        self.assertIn("4,096", info.runtimeNote or "")

    def test_speculative_off_starts_serial(self) -> None:
        self._add_drafter()
        engine = self._engine()
        info = self._load(engine, speculative_decoding=False)
        argv = self._started()["argv"]
        self.assertIn("--no-drafts", argv)
        self.assertNotIn("--drafter", argv)
        self.assertFalse(info.speculativeDecoding)
        self.assertIsNone(info.dflashDraftModel)
        self.assertIn("serial", info.runtimeNote or "")

    def test_missing_drafter_is_not_fatal(self) -> None:
        engine = self._engine()
        with mock.patch("huggingface_hub.snapshot_download", side_effect=OSError("offline")):
            info = self._load(engine)
        argv = self._started()["argv"]
        self.assertEqual(argv[argv.index("--drafter") + 1], "none")
        self.assertIsNone(info.dflashDraftModel)
        self.assertIn("could not be downloaded", info.runtimeNote or "")
        self.assertIn("offline", info.runtimeNote or "")
        self.assertIsNotNone(engine.process_pid())

    def test_downloads_the_drafter_once_when_missing(self) -> None:
        engine = self._engine()
        fetched: list[str] = []

        def fake_download(*, repo_id: str, **_kwargs) -> str:
            fetched.append(repo_id)
            target = self.tmp / "downloaded-drafter"
            _write_checkpoint(target)
            return str(target)

        with mock.patch("huggingface_hub.snapshot_download", side_effect=fake_download):
            info = self._load(engine)
        self.assertEqual(fetched, [_DRAFTER_REPO])
        self.assertEqual(info.dflashDraftModel, _DRAFTER_REPO)
        argv = self._started()["argv"]
        self.assertEqual(argv[argv.index("--drafter") + 1], str(self.tmp / "downloaded-drafter"))

    def test_family_specific_drafter_flags_are_passed(self) -> None:
        gemma_repo = self.tmp / "library" / f"models--{_GEMMA_REPO.replace('/', '--')}"
        _write_checkpoint(gemma_repo / "snapshots" / "rev1", model_type="gemma4")
        self._add_drafter("z-lab/gemma-4-26B-A4B-it-DFlash")
        engine = self._engine()
        self._load(
            engine,
            model_ref=_GEMMA_REPO,
            canonical_repo=_GEMMA_REPO,
            runtime_target=_GEMMA_REPO,
            path=str(gemma_repo),
        )
        argv = self._started()["argv"]
        self.assertEqual(argv[argv.index("--drafter-bits") + 1], "8")

    def test_child_environment_is_scrubbed_but_keeps_user_settings(self) -> None:
        engine = self._engine()
        with mock.patch.dict(
            os.environ,
            {
                "PYTHONHOME": "/somewhere/embedded",
                "PYTHONPATH": "/somewhere/extras",
                "DYLD_LIBRARY_PATH": "/somewhere/lib",
                "VIRTUAL_ENV": "/somewhere/venv",
                "CHAOSENGINE_KEEP_ME": "yes",
            },
        ):
            self._load(engine)
        env = self._started()["env"]
        self.assertIsNone(env["PYTHONHOME"])
        self.assertIsNone(env["PYTHONPATH"])
        self.assertIsNone(env["DYLD_LIBRARY_PATH"])
        self.assertIsNone(env["VIRTUAL_ENV"])
        self.assertEqual(env["CHAOSENGINE_KEEP_ME"], "yes")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TENSORFOLD_NO_UPDATE_CHECK"], "1")
        self.assertEqual(Path(self._started()["cwd"]).name, ".chaosengine")

    def test_load_reports_server_progress(self) -> None:
        engine = self._engine(_make_wrapper(self.tmp, fail_mode="delay"))
        progress: list[dict] = []
        self._load(engine, progress_callback=progress.append)
        messages = [item["message"] for item in progress]
        self.assertTrue(any("loading stub weights" in message for message in messages), messages)
        self.assertEqual(progress[-1]["phase"], "ready")
        self.assertEqual(progress[-1]["percent"], 100.0)

    def test_load_refuses_when_not_installed(self) -> None:
        engine = TensorFoldEngine(
            BackendCapabilities(pythonExecutable=sys.executable, mlxAvailable=True, mlxLmAvailable=True, mlxUsable=True)
        )
        self.engines.append(engine)
        with self.assertRaises(RuntimeError) as ctx:
            self._load(engine)
        self.assertIn("not installed", str(ctx.exception).lower())

    def test_load_refuses_a_model_that_is_not_downloaded(self) -> None:
        engine = self._engine()
        with self.assertRaises(RuntimeError) as ctx:
            self._load(engine, path=None, runtime_target="Vontra/Not-Downloaded-4bit", model_ref="Vontra/Not-Downloaded-4bit", canonical_repo=None)
        self.assertIn("not downloaded", str(ctx.exception))
        self.assertIsNone(engine.process_pid())

    def test_load_refuses_an_incomplete_download(self) -> None:
        (self.snapshot / "model.safetensors").unlink()
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})
        )
        (self.snapshot / "model-00001-of-00002.safetensors").write_bytes(b"x")
        engine = self._engine()
        with self.assertRaises(RuntimeError) as ctx:
            self._load(engine)
        self.assertIn("incomplete", str(ctx.exception))
        self.assertIsNone(engine.process_pid())

    def test_startup_failure_reports_the_reason_and_cleans_up(self) -> None:
        engine = self._engine(_make_wrapper(self.tmp, fail_mode="crash-before-ready"))
        with self.assertRaises(RuntimeError) as ctx:
            self._load(engine)
        message = str(ctx.exception)
        self.assertIn("do not fit", message)
        self.assertNotIn("tensorfold:", message)  # the CLI's own prefix is dropped
        self.assertIsNone(engine.process_pid())
        self.assertIsNone(engine.log_path)

    # ------------------------------------------------------------------
    # generate / stream_generate
    # ------------------------------------------------------------------

    def test_generate_round_trip(self) -> None:
        engine = self._engine()
        self._load(engine)
        result = engine.generate(**self._generate_kwargs(system_prompt="You are a stub."))
        self.assertEqual(result.text, "stub-tensorfold says hi")
        self.assertEqual(result.finishReason, "stop")
        self.assertEqual(result.completionTokens, 4)
        self.assertEqual(result.tokS, 42.5)  # the server's own decode rate
        self.assertEqual(result.dflashAcceptanceRate, 2.0)  # 8 accepted over 4 rounds
        self.assertTrue(result.speculative_decoding)
        self.assertIsNone(result.tool_calls)
        self.assertIn("exact speculative decoding", result.runtimeNote or "")
        sent = self._requests()[-1]
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "You are a stub."})
        # non-streaming turns have nowhere to show reasoning
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": False})

    def test_stream_orders_reasoning_before_text_and_reports_stats(self) -> None:
        engine = self._engine()
        self._load(engine)
        chunks = list(engine.stream_generate(**self._generate_kwargs(thinking_mode="auto")))
        reasoning = "".join(chunk.reasoning for chunk in chunks if chunk.reasoning)
        text = "".join(chunk.text for chunk in chunks if chunk.text)
        self.assertEqual(reasoning, "planning a reply")
        self.assertEqual(text, "stub-tensorfold says hi")
        markers = [i for i, chunk in enumerate(chunks) if chunk.reasoning_done]
        self.assertEqual(len(markers), 1)
        self.assertLess(markers[0], next(i for i, chunk in enumerate(chunks) if chunk.text))
        done = chunks[-1]
        self.assertTrue(done.done)
        self.assertEqual(done.finish_reason, "stop")
        self.assertEqual(done.completion_tokens, 4)
        self.assertEqual(done.tok_s, 42.5)
        self.assertEqual(done.dflash_acceptance_rate, 2.0)
        self.assertTrue(done.speculative_decoding)
        self.assertNotIn("enable_thinking", (self._requests()[-1].get("chat_template_kwargs") or {}))

    def test_stream_thinking_off_disables_template_thinking(self) -> None:
        engine = self._engine()
        self._load(engine)
        chunks = list(engine.stream_generate(**self._generate_kwargs(thinking_mode="off")))
        self.assertFalse(any(chunk.reasoning for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks if chunk.text), "stub-tensorfold says hi")
        self.assertEqual(self._requests()[-1]["chat_template_kwargs"], {"enable_thinking": False})

    def test_serial_replies_report_no_drafting(self) -> None:
        engine = self._engine()
        self._load(engine, speculative_decoding=False)
        chunks = list(engine.stream_generate(**self._generate_kwargs()))
        done = chunks[-1]
        self.assertFalse(done.speculative_decoding)
        self.assertIsNone(done.dflash_acceptance_rate)

    def test_sampler_settings_are_forwarded_and_unsupported_ones_named(self) -> None:
        engine = self._engine()
        self._load(engine)
        result = engine.generate(
            **self._generate_kwargs(
                samplers={
                    "top_p": 0.9,
                    "top_k": 40,
                    "min_p": 0.05,
                    "seed": 7,
                    "stop": ["END"],
                    "repeat_penalty": 1.15,
                    "mirostat": 2,
                },
                reasoning_effort="low",
            )
        )
        sent = self._requests()[-1]
        self.assertEqual((sent["top_p"], sent["top_k"], sent["min_p"], sent["seed"]), (0.9, 40, 0.05, 7))
        self.assertEqual(sent["stop"], ["END"])
        self.assertNotIn("repeat_penalty", sent)
        self.assertNotIn("mirostat", sent)
        self.assertIn("repeat_penalty", result.runtimeNote or "")
        self.assertIn("mirostat", result.runtimeNote or "")

    def test_reasoning_effort_reaches_the_server_when_thinking(self) -> None:
        engine = self._engine()
        self._load(engine)
        list(engine.stream_generate(**self._generate_kwargs(thinking_mode="auto", reasoning_effort="medium")))
        self.assertEqual(self._requests()[-1]["reasoning_effort"], "medium")

    def test_images_are_ignored_with_a_note(self) -> None:
        engine = self._engine()
        self._load(engine)
        chunks = list(engine.stream_generate(**self._generate_kwargs(images=["AAAA"])))
        self.assertIn("image was ignored", chunks[-1].runtime_note or "")
        self.assertNotIn("image_url", json.dumps(self._requests()[-1]))

    # ------------------------------------------------------------------
    # tool calls
    # ------------------------------------------------------------------

    @staticmethod
    def _tools() -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ]

    def test_generate_returns_structured_tool_calls_and_the_text_form(self) -> None:
        engine = self._engine()
        self._load(engine)
        result = engine.generate(**self._generate_kwargs(prompt="USE_TOOL now", tools=self._tools()))
        self.assertEqual(result.finishReason, "tool_calls")
        self.assertEqual(len(result.tool_calls or []), 1)
        call = (result.tool_calls or [])[0]
        self.assertEqual(call["function"]["name"], "web_search")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"query": "cats"})
        # the agent loop's text fallback reads this form
        from backend_service.agent import _parse_tool_calls_from_response

        parsed = _parse_tool_calls_from_response(result.text) or []
        self.assertEqual([item["function"]["name"] for item in parsed], ["web_search"])

    def test_stream_renders_tool_calls_as_agent_readable_text(self) -> None:
        engine = self._engine()
        self._load(engine)
        chunks = list(engine.stream_generate(**self._generate_kwargs(prompt="USE_TOOL now", tools=self._tools())))
        text = "".join(chunk.text for chunk in chunks if chunk.text)
        self.assertIn("<tool_call>", text)
        self.assertIn('"web_search"', text)
        self.assertEqual(chunks[-1].finish_reason, "tool_calls")

    def test_tool_history_stays_structured_and_never_duplicates_a_call(self) -> None:
        engine = self._engine()
        self._load(engine)
        call = {"id": "call_9", "type": "function", "function": {"name": "web_search", "arguments": '{"query": "cats"}'}}
        history = [
            {"role": "user", "text": "find cats"},
            {
                "role": "assistant",
                "text": '<tool_call>{"name": "web_search", "arguments": {"query": "cats"}}</tool_call>',
                "tool_calls": [call],
            },
            {"role": "tool", "text": "cats are great", "tool_call_id": "call_9", "name": "web_search"},
        ]
        # the agent loop's follow-up iteration sends an empty prompt
        engine.generate(**self._generate_kwargs(prompt="", history=history, tools=self._tools()))
        messages = self._requests()[-1]["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool"])
        self.assertEqual(messages[1]["content"], "")
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "web_search")
        self.assertEqual(messages[2]["tool_call_id"], "call_9")

    # ------------------------------------------------------------------
    # refusals
    # ------------------------------------------------------------------

    def test_stream_retries_once_with_the_reply_room_the_server_names(self) -> None:
        engine = self._engine()
        self._load(engine)
        chunks = list(engine.stream_generate(**self._generate_kwargs(max_tokens=5000)))
        self.assertEqual("".join(chunk.text for chunk in chunks if chunk.text), "stub-tensorfold says hi")
        self.assertIn("990", chunks[-1].runtime_note or "")
        self.assertEqual([r["max_tokens"] for r in self._requests()], [5000, 990])

    def test_generate_retries_once_with_the_reply_room_the_server_names(self) -> None:
        engine = self._engine()
        self._load(engine)
        result = engine.generate(**self._generate_kwargs(max_tokens=5000))
        self.assertEqual(result.text, "stub-tensorfold says hi")
        self.assertIn("990", result.runtimeNote or "")
        self.assertEqual([r["max_tokens"] for r in self._requests()], [5000, 990])

    def test_structured_output_refusal_reaches_the_caller(self) -> None:
        engine = self._engine()
        self._load(engine)
        with self.assertRaises(RuntimeError) as ctx:
            list(engine.stream_generate(**self._generate_kwargs(json_schema={"type": "object"})))
        self.assertIn("xgrammar", str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            engine.generate(**self._generate_kwargs(json_schema={"type": "object"}))
        self.assertIn("xgrammar", str(ctx.exception))

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def test_generate_after_unload_raises(self) -> None:
        engine = self._engine()
        self._load(engine)
        engine.unload_model()
        self.assertIsNone(engine.process_pid())
        with self.assertRaises(RuntimeError):
            engine.generate(**self._generate_kwargs())
        with self.assertRaises(RuntimeError):
            list(engine.stream_generate(**self._generate_kwargs()))

    def test_reload_replaces_the_previous_server(self) -> None:
        engine = self._engine()
        self._load(engine)
        first = engine.process_pid()
        self._load(engine)
        second = engine.process_pid()
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)

    def test_unload_is_idempotent(self) -> None:
        engine = self._engine()
        engine.unload_model()
        engine.unload_model()
        self.assertIsNone(engine.process_pid())


if __name__ == "__main__":
    unittest.main()
