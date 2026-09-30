"""Unit tests for the TensorFold registry, wire helpers and child environment.

Pure-python: no server, no subprocess. The registry facts under test are the
ones recorded from TensorFold 0.5.0's own source (``families/*/__init__.py``),
so a failing assertion here usually means the table needs the same re-check
as the install pin does.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.inference import _tensorfold as tf
from backend_service.inference import _tensorfold_wire as wire
from backend_service.inference._utils import _isolated_child_env


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class RegistryInvariantTests(unittest.TestCase):
    def test_every_family_names_model_types_and_tested_repos(self) -> None:
        for family in tf.TENSORFOLD_FAMILIES:
            self.assertTrue(family.model_types, family.key)
            self.assertTrue(family.tested_repos, family.key)
            self.assertTrue(family.title, family.key)

    def test_keys_and_tested_repos_are_unique(self) -> None:
        keys = [family.key for family in tf.TENSORFOLD_FAMILIES]
        self.assertEqual(len(keys), len(set(keys)))
        repos = [repo.lower() for family in tf.TENSORFOLD_FAMILIES for repo in family.tested_repos]
        self.assertEqual(len(repos), len(set(repos)))

    def test_exclusive_model_types_identify_exactly_one_family(self) -> None:
        seen: dict[str, str] = {}
        for family in tf.TENSORFOLD_FAMILIES:
            if not family.exclusive:
                continue
            for model_type in family.model_types:
                self.assertNotIn(model_type, seen, f"{model_type} claimed twice")
                seen[model_type] = family.key
        # model types stock mlx-lm shares must never be exclusive
        self.assertNotIn("qwen3_5", seen)
        self.assertNotIn("qwen3_5_moe", seen)
        self.assertNotIn("nemotron_h", seen)

    def test_drafter_flags_only_accompany_a_drafter(self) -> None:
        for family in tf.TENSORFOLD_FAMILIES:
            if family.drafter_args:
                self.assertTrue(family.drafter, family.key)

    def test_listings(self) -> None:
        self.assertIn("mlx-community/gemma-4-26b-a4b-it-4bit", tf.supported_repos())
        self.assertIn("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP", tf.exclusive_repos())
        self.assertNotIn("mlx-community/gemma-4-26b-a4b-it-4bit", tf.exclusive_repos())
        self.assertLessEqual(set(tf.exclusive_repos()), set(tf.supported_repos()))


class MatchModelTests(unittest.TestCase):
    def test_repo_id_match_is_case_insensitive(self) -> None:
        match = tf.match_model(canonical_repo="vontra/qwen3.8-27b-mlx-4bit")
        assert match is not None
        self.assertEqual(match.family.key, "qwen3_8_dense")
        self.assertEqual(match.matched_by, "repo")

    def test_any_of_the_three_identifiers_can_carry_the_repo(self) -> None:
        for kwargs in (
            {"canonical_repo": "Vontra/Qwen3.8-27B-MLX-4bit"},
            {"model_ref": "Vontra/Qwen3.8-27B-MLX-4bit"},
            {"runtime_target": "Vontra/Qwen3.8-27B-MLX-4bit"},
        ):
            match = tf.match_model(**kwargs)
            self.assertIsNotNone(match, kwargs)

    def test_tiers(self) -> None:
        dense = tf.match_model(canonical_repo="Vontra/Qwen3.8-27B-MLX-4bit")
        gemma = tf.match_model(canonical_repo="mlx-community/gemma-4-26b-a4b-it-4bit")
        flash = tf.match_model(canonical_repo="Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
        deepseek = tf.match_model(canonical_repo="mlx-community/DeepSeek-V4-Flash-4bit")
        assert dense and gemma and flash and deepseek
        # stock MLX also loads these, so they are only "tested"
        self.assertEqual(dense.tier, "tested")
        self.assertEqual(gemma.tier, "tested")
        self.assertEqual(flash.tier, "exclusive")
        # the publisher namespace does not decide the tier: the family does
        self.assertEqual(deepseek.tier, "exclusive")

    def test_repo_is_recovered_from_a_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "models--Vontra--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit"
            (repo_dir / "snapshots" / "rev").mkdir(parents=True)
            match = tf.match_model(path=str(repo_dir))
            assert match is not None
            self.assertEqual(match.family.key, "nemotron_h")

    def test_unknown_model_does_not_match(self) -> None:
        self.assertIsNone(tf.match_model(canonical_repo="mlx-community/Llama-3.2-3B-Instruct-4bit"))
        self.assertIsNone(tf.match_model())

    def test_cuda_only_families_are_not_listed(self) -> None:
        # Qwen3.6-35B-A3B has a CUDA engine only; routing it to TensorFold on a Mac
        # would fail at serve. Its model_type is shared with stock mlx-lm anyway.
        self.assertIsNone(tf.match_model(canonical_repo="Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP"))
        self.assertNotIn("qwen3_5_moe", {t for f in tf.TENSORFOLD_FAMILIES for t in f.model_types})

    def test_config_identifies_exclusive_families_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            flash = Path(tmp) / "flash"
            _write(flash / "config.json", json.dumps({"model_type": "qwen4_exp"}))
            match = tf.match_model(model_dir=flash)
            assert match is not None
            self.assertEqual((match.family.key, match.tier, match.matched_by), ("qwen3_8_flash_next", "exclusive", "config"))

            # ``qwen3_5`` is shared by every Qwen3.5/3.6/3.8 dense checkpoint, and
            # TensorFold's own per-family check refuses most of them: not evidence.
            dense = Path(tmp) / "dense"
            _write(dense / "config.json", json.dumps({"model_type": "qwen3_5"}))
            self.assertIsNone(tf.match_model(model_dir=dense))

    def test_config_in_a_cache_snapshot_is_found_through_the_repo_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "custom-name"
            _write(repo_dir / "snapshots" / "rev1" / "config.json", json.dumps({"model_type": "deepseek_v4"}))
            match = tf.match_model(path=str(repo_dir))
            assert match is not None
            self.assertEqual(match.family.key, "deepseek_v4_flash")

    def test_read_model_type_falls_back_to_text_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write(directory / "config.json", json.dumps({"text_config": {"model_type": "gemma4_text"}}))
            self.assertEqual(tf.read_model_type(directory), "gemma4_text")
            _write(directory / "config.json", "not json")
            self.assertIsNone(tf.read_model_type(directory))
            self.assertIsNone(tf.read_model_type(None))


class WeightsCompleteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_single_file(self) -> None:
        self.assertFalse(tf.weights_complete(self.dir))
        _write(self.dir / "model.safetensors")
        self.assertTrue(tf.weights_complete(self.dir))

    def test_index_needs_every_shard_it_names(self) -> None:
        index = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
        _write(self.dir / "model.safetensors.index.json", json.dumps(index))
        _write(self.dir / "model-00001-of-00002.safetensors")
        self.assertFalse(tf.weights_complete(self.dir))
        _write(self.dir / "model-00002-of-00002.safetensors")
        self.assertTrue(tf.weights_complete(self.dir))

    def test_index_rejects_escaping_and_empty_maps(self) -> None:
        for weight_map in ({}, {"a": "../outside.safetensors"}, {"a": "/abs/model.safetensors"}, {"a": 3}):
            _write(self.dir / "model.safetensors.index.json", json.dumps({"weight_map": weight_map}))
            self.assertFalse(tf.weights_complete(self.dir), weight_map)
        _write(self.dir / "model.safetensors.index.json", "{not json")
        self.assertFalse(tf.weights_complete(self.dir))
        _write(self.dir / "model.safetensors.index.json", json.dumps({"other": 1}))
        self.assertFalse(tf.weights_complete(self.dir))

    def test_shard_set_without_index(self) -> None:
        _write(self.dir / "model-00001-of-00003.safetensors")
        _write(self.dir / "model-00003-of-00003.safetensors")
        self.assertFalse(tf.weights_complete(self.dir))
        _write(self.dir / "model-00002-of-00003.safetensors")
        self.assertTrue(tf.weights_complete(self.dir))

    def test_mixed_shard_totals_are_incomplete(self) -> None:
        _write(self.dir / "model-00001-of-00002.safetensors")
        _write(self.dir / "model-00001-of-00003.safetensors")
        self.assertFalse(tf.weights_complete(self.dir))


class ResolveModelDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        patch = mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.tmp / "hub")})
        patch.start()
        self.addCleanup(patch.stop)
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)

    def test_local_directory_wins(self) -> None:
        model = self.tmp / "local-model"
        _write(model / "config.json", "{}")
        self.assertEqual(tf.resolve_model_dir(path=str(model), runtime_target=None, repos=("a/b",)), model)

    def test_hf_cache_entry_resolves_to_its_snapshot(self) -> None:
        repo_dir = self.tmp / "hub" / "models--Vontra--Qwen3.8-27B-MLX-4bit"
        snapshot = repo_dir / "snapshots" / "abc"
        _write(snapshot / "config.json", "{}")
        _write(repo_dir / "refs" / "main", "abc")
        found = tf.resolve_model_dir(path=None, runtime_target="Vontra/Qwen3.8-27B-MLX-4bit", repos=("Vontra/Qwen3.8-27B-MLX-4bit",))
        self.assertEqual(found, snapshot)

    def test_a_repo_id_is_never_resolved_against_the_working_directory(self) -> None:
        relative = self.tmp / "owner" / "name"
        _write(relative / "config.json", "{}")
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            found = tf.resolve_model_dir(path=None, runtime_target="owner/name", repos=("owner/name",))
        finally:
            os.chdir(cwd)
        self.assertIsNone(found)

    def test_nothing_local_returns_none(self) -> None:
        self.assertIsNone(tf.resolve_model_dir(path=None, runtime_target="a/b", repos=("a/b", None)))

    def test_native_context_window(self) -> None:
        directory = self.tmp / "m"
        _write(directory / "config.json", json.dumps({"max_position_embeddings": 32768}))
        self.assertEqual(tf.native_context_window(directory), 32768)
        _write(directory / "config.json", json.dumps({"max_position_embeddings": 1, "text_config": {"max_position_embeddings": 262144}}))
        self.assertEqual(tf.native_context_window(directory), 262144)
        _write(directory / "config.json", json.dumps({"max_position_embeddings": True}))
        self.assertIsNone(tf.native_context_window(directory))
        _write(directory / "config.json", "{}")
        self.assertIsNone(tf.native_context_window(directory))


class ChildEnvironmentTests(unittest.TestCase):
    def test_import_redirecting_variables_are_dropped(self) -> None:
        polluted = {
            "PYTHONHOME": "/a", "PYTHONPATH": "/b", "PYTHONSTARTUP": "/c", "VIRTUAL_ENV": "/d",
            "DYLD_LIBRARY_PATH": "/e", "DYLD_FALLBACK_LIBRARY_PATH": "/f", "DYLD_INSERT_LIBRARIES": "/g",
            "LD_LIBRARY_PATH": "/h", "CHAOSENGINE_EMBEDDED_RUNTIME": "1",
            "PATH": "/usr/bin", "HF_HOME": "/hf", "TENSORFOLD_MEMORY_LIMIT_GB": "90",
        }
        with mock.patch.dict(os.environ, polluted):
            env = _isolated_child_env({"HF_HUB_OFFLINE": "1"})
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "VIRTUAL_ENV", "DYLD_LIBRARY_PATH",
                     "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "LD_LIBRARY_PATH",
                     "CHAOSENGINE_EMBEDDED_RUNTIME"):
            self.assertNotIn(name, env)
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertEqual(env["HF_HOME"], "/hf")
        self.assertEqual(env["TENSORFOLD_MEMORY_LIMIT_GB"], "90")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_does_not_mutate_the_parent_environment(self) -> None:
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/keep"}):
            _isolated_child_env()
            self.assertEqual(os.environ["PYTHONPATH"], "/keep")


class BuildMessagesTests(unittest.TestCase):
    def test_plain_conversation(self) -> None:
        messages = wire.build_messages(
            "be brief",
            [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}, {"role": "weird", "text": "x"}],
            "again",
        )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ],
        )

    def test_empty_prompt_adds_no_user_turn(self) -> None:
        self.assertEqual(wire.build_messages(None, [{"role": "tool", "text": "r", "tool_call_id": "c1"}], ""),
                         [{"role": "tool", "content": "r", "tool_call_id": "c1"}])

    def test_assistant_tool_calls_stay_structured_and_lose_the_text_form(self) -> None:
        calls = [
            {"id": "c1", "function": {"name": "a", "arguments": {"x": 1}}},
            {"id": "c2", "function": {"name": "b", "arguments": None}},
            {"function": {"name": ""}},
            "junk",
        ]
        text = 'thinking <tool_call>{"name": "a"}</tool_call>\n<TOOL_CALL>{"name": "b"}</TOOL_CALL>'
        [assistant] = wire.build_messages(None, [{"role": "assistant", "text": text, "tool_calls": calls}], "")
        self.assertEqual(assistant["content"], "thinking")
        self.assertEqual(
            assistant["tool_calls"],
            [
                {"id": "c1", "type": "function", "function": {"name": "a", "arguments": '{"x": 1}'}},
                {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ],
        )

    def test_content_parts_are_flattened(self) -> None:
        [message] = wire.build_messages(None, [{"role": "user", "text": [{"type": "text", "text": "a"}, {"text": "b"}]}], "")
        self.assertEqual(message["content"], "a b")


class BuildPayloadTests(unittest.TestCase):
    def _payload(self, **kwargs):
        base = dict(model="m", messages=[{"role": "user", "content": "x"}], max_tokens=8, temperature=0.5, stream=True)
        base.update(kwargs)
        return wire.build_payload(**base)

    def test_thinking_off_is_the_default(self) -> None:
        self.assertEqual(self._payload()["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(self._payload(thinking_mode="off")["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("chat_template_kwargs", self._payload(thinking_mode="auto"))

    def test_reasoning_effort_only_with_thinking_and_only_valid_values(self) -> None:
        self.assertEqual(self._payload(thinking_mode="auto", reasoning_effort="high")["reasoning_effort"], "high")
        self.assertNotIn("reasoning_effort", self._payload(thinking_mode="auto", reasoning_effort="ludicrous"))
        self.assertNotIn("reasoning_effort", self._payload(thinking_mode="off", reasoning_effort="high"))

    def test_samplers(self) -> None:
        payload = self._payload(samplers={"top_p": 0.8, "top_k": 0, "min_p": True, "seed": -1, "stop": "END", "logprobs": 5})
        self.assertEqual((payload["top_p"], payload["top_k"]), (0.8, 0))
        self.assertNotIn("min_p", payload)  # booleans are not numbers to the server
        self.assertNotIn("seed", payload)  # negative means "random" in the UI; the server wants >= 0
        self.assertEqual(payload["stop"], "END")
        self.assertNotIn("logprobs", payload)
        self.assertEqual(self._payload(samplers={"seed": 7, "stop": ["a", "b"]})["stop"], ["a", "b"])
        self.assertNotIn("stop", self._payload(samplers={"stop": [1, 2]}))

    def test_tools_and_schema(self) -> None:
        payload = self._payload(tools=[{"type": "function"}], json_schema={"type": "object"})
        self.assertEqual(payload["tools"], [{"type": "function"}])
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], {"type": "object"})
        self.assertNotIn("tools", self._payload(tools=[]))

    def test_unsupported_sampler_note(self) -> None:
        self.assertIsNone(wire.unsupported_sampler_note(None))
        self.assertIsNone(wire.unsupported_sampler_note({"repeat_penalty": 1.0, "mirostat": 0, "top_n_sigma": -1, "top_p": 0.9}))
        note = wire.unsupported_sampler_note(
            {"repeat_penalty": 1.1, "frequency_penalty": -0.2, "presence_penalty": 0, "dry_multiplier": 0.8, "xtc_probability": 0.1}
        )
        assert note is not None
        for name in ("repeat_penalty", "frequency_penalty", "dry_multiplier", "xtc_probability"):
            self.assertIn(name, note)
        self.assertNotIn("presence_penalty", note)


class ToolCallTests(unittest.TestCase):
    def test_assembler_joins_pieces_by_index(self) -> None:
        assembler = wire.ToolCallAssembler()
        assembler.feed([{"index": 0, "id": "c0", "type": "function", "function": {"name": "search", "arguments": ""}}])
        assembler.feed([{"index": 0, "function": {"arguments": '{"q": '}}])
        assembler.feed([{"index": 0, "function": {"arguments": '"cats"}'}}, {"index": 1, "id": "c1", "function": {"name": "open"}}])
        assembler.feed("garbage")
        assembler.feed([None])
        calls = assembler.calls()
        self.assertEqual([call["id"] for call in calls], ["c0", "c1"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"q": "cats"})

    def test_nameless_fragments_are_dropped(self) -> None:
        assembler = wire.ToolCallAssembler()
        assembler.feed([{"index": 0, "function": {"arguments": "{}"}}])
        self.assertEqual(assembler.calls(), [])

    def test_text_form(self) -> None:
        text = wire.tool_calls_to_text(
            [
                {"function": {"name": "a", "arguments": '{"x": 1}'}},
                {"function": {"name": "b", "arguments": "not json"}},
                {"function": {"name": "c", "arguments": "[1]"}},
                {"function": {"name": "d", "arguments": None}},
                {"function": {"name": ""}},
            ]
        )
        blocks = [json.loads(part.removeprefix("<tool_call>").removesuffix("</tool_call>")) for part in text.split("\n")]
        self.assertEqual(
            blocks,
            [
                {"name": "a", "arguments": {"x": 1}},
                {"name": "b", "arguments": {"raw": "not json"}},
                {"name": "c", "arguments": {"raw": "[1]"}},
                {"name": "d", "arguments": {}},
            ],
        )


class ErrorParsingTests(unittest.TestCase):
    def test_server_error_message(self) -> None:
        self.assertEqual(wire.server_error_message({"error": {"message": "bad", "type": "x"}}), "bad")
        self.assertEqual(wire.server_error_message({"error": "plain"}), "plain")
        self.assertEqual(wire.server_error_message("raw"), "raw")

    def test_reply_budget(self) -> None:
        message = "Reduce the prompt to at most 10 prompt tokens or request at most 990 reply tokens, including chat template"
        self.assertEqual(wire.parse_reply_budget(message), 990)
        self.assertIsNone(wire.parse_reply_budget("nothing to see"))
        self.assertIsNone(wire.parse_reply_budget("request at most 0 reply tokens"))

    def test_startup_failure_prefers_the_cli_reason(self) -> None:
        log = "[tensorfold] memory budget 100 GiB\ntensorfold: weights (30 GiB) do not fit\n"
        self.assertEqual(wire.summarize_startup_failure(log, 1), "weights (30 GiB) do not fit")

    def test_startup_failure_reads_the_traceback_tail(self) -> None:
        log = 'Traceback (most recent call last):\n  File "x.py", line 1, in <module>\n    boom()\nRuntimeError: Metal exploded\n'
        self.assertEqual(wire.summarize_startup_failure(log, 1), "RuntimeError: Metal exploded")

    def test_startup_failure_fallbacks(self) -> None:
        self.assertEqual(wire.summarize_startup_failure("just a line", 1), "just a line")
        self.assertIn("code 3", wire.summarize_startup_failure("", 3))
        self.assertEqual(wire.summarize_startup_failure("", None), "TensorFold exited during startup.")


if __name__ == "__main__":
    unittest.main()
