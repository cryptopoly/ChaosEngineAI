"""FU-178: llama.cpp 1.0 features on the GGUF lane.

Native q8_0 / q4_0 KV cache (with the memory estimate), the FP8 KV refusal
off CUDA, draft acceptance surfaced from llama-server, the Qwen3.8 MTP draft
count and the DFlash 2 spec type.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend_service.inference import BackendCapabilities, LlamaCppEngine, LoadedModelInfo
from backend_service.inference.llama_cpp_engine import _draft_acceptance, _refused_cache_type
from cache_compression import registry
from cache_compression.native import native_llama_cache_type

# Captured from llama-server b12441 (Qwen3.8-27B UD-Q4_K_M, draft-mtp N=3, M4 Max).
CAPTURED_LOG = """\
0.23.152.346 I slot print_timing: id  0 | task 0 | prompt eval time =     866.66 ms /    54 tokens (   16.05 ms per token,    62.31 tokens per second)
0.23.152.349 I slot print_timing: id  0 | task 0 |        eval time =   19272.54 ms /   400 tokens (   48.30 ms per token,    20.70 tokens per second)
0.23.152.355 I slot print_timing: id  0 | task 0 | draft acceptance = 0.92722 (  293 accepted /   316 generated), mean len =  3.76
"""


def _capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        pythonExecutable="/usr/bin/python3",
        mlxAvailable=False,
        mlxLmAvailable=False,
        mlxUsable=False,
        ggufAvailable=True,
        llamaServerPath="/usr/local/bin/llama-server",
    )


_ALL_TYPES = frozenset({"f32", "f16", "bf16", "f8_e4m3", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"})


class NativeQuantizedKvTests(unittest.TestCase):
    def test_bits_map_to_standard_cache_types(self):
        native = registry.get("native")
        self.assertEqual(native.llama_cpp_cache_flags(0), ["--cache-type-k", "f16", "--cache-type-v", "f16"])
        self.assertEqual(native.llama_cpp_cache_flags(8), ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])
        self.assertEqual(native.llama_cpp_cache_flags(4), ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"])
        # Anything else stays f16; FP8 is never emitted.
        for bits in (1, 2, 3, 5, 16):
            self.assertEqual(native_llama_cache_type(bits), "f16")
        self.assertEqual(native.label(8, 0), "Native q8_0")
        self.assertEqual(native.label(4, 0), "Native q4_0")
        self.assertEqual(native.label(0, 0), "Native f16")

    def test_estimate_halves_and_quarters(self):
        native = registry.get("native")
        base, same = native.estimate_cache_bytes(32, 32, 4096, 8192, 0, 0, 8)
        self.assertEqual(base, same)
        _, q8 = native.estimate_cache_bytes(32, 32, 4096, 8192, 8, 0, 8)
        _, q4 = native.estimate_cache_bytes(32, 32, 4096, 8192, 4, 0, 8)
        self.assertEqual(q8, int(base * 8.5 / 16))
        self.assertEqual(q4, int(base * 4.5 / 16))

    def test_cache_preview_shrinks_only_on_the_llama_cpp_lane(self):
        from backend_service.helpers.cache import compute_cache_preview

        args = dict(fp16_layers=0, num_layers=36, num_heads=32, num_kv_heads=8, hidden_size=4096,
                    context_tokens=32768, params_b=8.0, strategy="native", weights_gb=5.0, measured_tok_s=40.0)
        f16 = compute_cache_preview(bits=0, lane="llama.cpp", **args)
        q8 = compute_cache_preview(bits=8, lane="llama.cpp", **args)
        q4 = compute_cache_preview(bits=4, lane="llama.cpp", **args)
        mlx = compute_cache_preview(bits=8, lane=None, **args)
        self.assertAlmostEqual(q8["optimizedCacheGb"], f16["baselineCacheGb"] * 8.5 / 16, delta=0.1)
        self.assertAlmostEqual(q4["optimizedCacheGb"], f16["baselineCacheGb"] * 4.5 / 16, delta=0.1)
        self.assertEqual(q8["bits"], 8)
        self.assertIn("Native q8_0", q8["summary"])
        self.assertLessEqual(q8["estimatedTokS"], f16["estimatedTokS"])
        self.assertLess(q4["estimatedTokS"], f16["estimatedTokS"])
        # MLX keeps its own f16 cache whatever cacheBits says.
        self.assertEqual(mlx["optimizedCacheGb"], mlx["baselineCacheGb"])
        self.assertEqual(mlx["bits"], 0)

    def _build(self, bits: int, types=_ALL_TYPES):
        engine = LlamaCppEngine(_capabilities())
        with (
            mock.patch("backend_service.inference._find_open_port", return_value=9999),
            mock.patch("backend_service.inference.llama_cpp_engine._llama_server_supports", return_value=True),
            mock.patch("backend_service.inference.llama_cpp_engine._llama_server_cache_types", return_value=types),
        ):
            return engine._build_command(
                path="/tmp/model.gguf", runtime_target=None, cache_strategy="native", cache_bits=bits,
                context_tokens=8192, fit_enabled=True, is_fallback=False, fused_attention=True,
            )

    def test_command_runs_q8_0_and_q4_0(self):
        for bits, cache_type in ((8, "q8_0"), (4, "q4_0"), (0, "f16")):
            command, note, fell_back, _ = self._build(bits)
            self.assertEqual(command[command.index("--cache-type-k") + 1], cache_type)
            self.assertEqual(command[command.index("--cache-type-v") + 1], cache_type)
            self.assertFalse(fell_back)
            self.assertIsNone(note)

    def test_binary_without_q8_0_falls_back_to_f16(self):
        command, note, fell_back, _ = self._build(8, types=frozenset({"f16", "f32"}))
        self.assertEqual(command[command.index("--cache-type-k") + 1], "f16")
        self.assertTrue(fell_back)
        self.assertIn("q8_0", note)

    def _load(self, bits: int, wait_side_effect):
        engine = LlamaCppEngine(_capabilities())
        process = mock.Mock()
        process.poll.return_value = None
        with (
            mock.patch("backend_service.inference._find_open_port", return_value=9999),
            mock.patch("backend_service.inference.llama_cpp_engine._llama_server_supports", return_value=True),
            mock.patch("backend_service.inference.llama_cpp_engine._llama_server_cache_types", return_value=_ALL_TYPES),
            mock.patch.object(engine, "_wait_for_server", side_effect=wait_side_effect),
            mock.patch.object(engine, "_cleanup_process"),
            mock.patch("backend_service.inference.subprocess.Popen", return_value=process) as popen,
        ):
            loaded = engine.load_model(
                model_ref="m", model_name="M", canonical_repo=None, source="library", backend="llama.cpp",
                path=None, runtime_target="org/m-GGUF", cache_strategy="native", cache_bits=bits, fp16_layers=0,
                fused_attention=True, fit_model_in_memory=True, context_tokens=8192,
            )
        return loaded, popen

    def test_load_reports_the_quantized_cache(self):
        loaded, popen = self._load(8, [None])
        self.assertEqual(loaded.cacheStrategy, "native")
        self.assertEqual(loaded.cacheBits, 8)
        self.assertIn("Native q8_0", loaded.runtimeNote)
        self.assertEqual(popen.call_count, 1)

    def test_failed_quantized_start_retries_at_f16(self):
        loaded, popen = self._load(4, [RuntimeError("boom"), None])
        self.assertEqual(popen.call_count, 2)
        retry = popen.call_args_list[1].args[0]
        self.assertEqual(retry[retry.index("--cache-type-k") + 1], "f16")
        self.assertEqual(loaded.cacheBits, 0)
        self.assertIn("Native q4_0", loaded.runtimeNote)
        self.assertIn("f16", loaded.runtimeNote)


class Fp8KvRefusalTests(unittest.TestCase):
    def _host(self, kind: str):
        return mock.patch(
            "backend_service.helpers.hardware_class.host_hardware_class",
            return_value=SimpleNamespace(kind=kind),
        )

    def test_refused_off_cuda(self):
        for kind in ("apple", "cpu"):
            with self._host(kind):
                reason = _refused_cache_type("f8_e4m3")
            self.assertIsNotNone(reason)
            self.assertIn("NVIDIA", reason)

    def test_allowed_on_cuda_and_other_types_untouched(self):
        with self._host("cuda"):
            self.assertIsNone(_refused_cache_type("f8_e4m3"))
        with self._host("apple"):
            self.assertIsNone(_refused_cache_type("q8_0"))

    def test_build_command_swaps_fp8_for_f16_on_metal(self):
        engine = LlamaCppEngine(_capabilities())
        fp8 = ["--cache-type-k", "f8_e4m3", "--cache-type-v", "f8_e4m3"]
        with (
            self._host("apple"),
            mock.patch("backend_service.inference._find_open_port", return_value=9999),
            mock.patch("backend_service.inference.llama_cpp_engine._llama_server_supports", return_value=True),
            mock.patch("backend_service.inference.llama_cpp_engine._llama_server_cache_types", return_value=_ALL_TYPES),
            mock.patch("cache_compression.native.NativeStrategy.llama_cpp_cache_flags", return_value=fp8),
        ):
            command, note, fell_back, _ = engine._build_command(
                path="/tmp/model.gguf", runtime_target=None, cache_strategy="native", cache_bits=8,
                context_tokens=8192, fit_enabled=True, is_fallback=False,
            )
        self.assertNotIn("f8_e4m3", command)
        self.assertEqual(command[command.index("--cache-type-k") + 1], "f16")
        self.assertTrue(fell_back)
        self.assertIn("FP8", note)


class DraftAcceptanceTests(unittest.TestCase):
    def test_captured_log_line(self):
        timings = {"predicted_n": 400, "draft_n": 316, "draft_n_accepted": 293}
        self.assertEqual(_draft_acceptance(timings, CAPTURED_LOG), 2.76)

    def test_line_from_another_request_is_not_used(self):
        timings = {"draft_n": 100, "draft_n_accepted": 50}
        self.assertIsNone(_draft_acceptance(timings, CAPTURED_LOG))

    def test_no_drafting_means_no_rate(self):
        self.assertIsNone(_draft_acceptance({"predicted_n": 400}, CAPTURED_LOG))
        self.assertIsNone(_draft_acceptance(None, CAPTURED_LOG))
        self.assertIsNone(_draft_acceptance({"draft_n": 316, "draft_n_accepted": 293}, ""))

    def _engine(self, log_text: str) -> LlamaCppEngine:
        engine = LlamaCppEngine(_capabilities())
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        engine.log_path = Path(tmp.name, "llama.log")
        engine.log_path.write_text(log_text)
        engine.port = 9999
        engine.process = mock.Mock()
        engine.process.poll.return_value = None
        engine.loaded_model = LoadedModelInfo(
            ref="m", name="M", canonicalRepo=None, backend="llama.cpp", source="library", engine="llama.cpp",
            cacheStrategy="native", cacheBits=0, fp16Layers=0, fusedAttention=True, fitModelInMemory=True,
            contextTokens=8192, loadedAt="now", path=None, runtimeTarget=None, runtimeNote=None,
        )
        return engine

    def test_stream_reports_acceptance_from_final_timings(self):
        engine = self._engine(CAPTURED_LOG)
        events = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 54, "completion_tokens": 400},
             "timings": {"predicted_n": 400, "draft_n": 316, "draft_n_accepted": 293}},
        ]
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        response = io.BytesIO(body.encode())
        with mock.patch("urllib.request.urlopen", return_value=response):
            chunks = list(engine.stream_generate(
                prompt="hi", history=[], system_prompt=None, max_tokens=16, temperature=0.0,
            ))
        self.assertEqual(chunks[-1].dflash_acceptance_rate, 2.76)

    def test_generate_reports_acceptance(self):
        engine = self._engine(CAPTURED_LOG)
        reply = {
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 54, "completion_tokens": 400},
            "timings": {"draft_n": 316, "draft_n_accepted": 293},
        }
        with mock.patch("backend_service.inference.llama_cpp_engine._http_json", return_value=reply):
            result = engine.generate(prompt="hi", history=[], system_prompt=None, max_tokens=16, temperature=0.0)
        self.assertEqual(result.dflashAcceptanceRate, 2.76)
        self.assertEqual(result.to_metrics()["dflashAcceptanceRate"], 2.76)

    def test_generate_sends_the_thinking_switch_per_request(self):
        # FU-608: the non-streaming route used to leave it out, so only the server-wide --reasoning applied.
        reply = {"choices": [{"message": {"content": "x"}}], "usage": {"completion_tokens": 1}}
        for mode, expected in (("off", False), ("auto", True)):
            engine = self._engine("")
            with mock.patch("backend_service.inference.llama_cpp_engine._http_json", return_value=reply) as http:
                engine.generate(prompt="hi", history=[], system_prompt=None, max_tokens=4, temperature=0.0,
                                thinking_mode=mode)
            self.assertEqual(http.call_args.kwargs["payload"]["chat_template_kwargs"], {"enable_thinking": expected})

    def test_generate_without_a_thinking_mode_leaves_the_template_alone(self):
        engine = self._engine("")
        reply = {"choices": [{"message": {"content": "x"}}], "usage": {"completion_tokens": 1}}
        with mock.patch("backend_service.inference.llama_cpp_engine._http_json", return_value=reply) as http:
            engine.generate(prompt="hi", history=[], system_prompt=None, max_tokens=4, temperature=0.0)
        self.assertNotIn("chat_template_kwargs", http.call_args.kwargs["payload"])

    def test_plain_decode_reports_none(self):
        engine = self._engine("")
        reply = {"choices": [{"message": {"content": "x"}}], "usage": {"completion_tokens": 1},
                 "timings": {"predicted_n": 1}}
        with mock.patch("backend_service.inference.llama_cpp_engine._http_json", return_value=reply):
            result = engine.generate(prompt="hi", history=[], system_prompt=None, max_tokens=4, temperature=0.0)
        self.assertIsNone(result.dflashAcceptanceRate)


class Qwen38SpeculativeTests(unittest.TestCase):
    def test_unsloth_gguf_gets_the_qwen38_draft_count(self):
        from backend_service.inference._mtp import get_mtp_draft_n

        self.assertEqual(get_mtp_draft_n("unsloth/Qwen3.8-27B-GGUF"), 3)
        self.assertEqual(get_mtp_draft_n("Qwen/Qwen3.8-27B"), 3)

    def test_baked_in_mtp_head_uses_the_mapped_draft_count(self):
        from backend_service.inference.llama_cpp_engine import _speculative_args

        with (
            mock.patch("backend_service.inference.llama_cpp_engine._llama_server_supports", return_value=True),
            mock.patch("backend_service.inference._mtp.model_has_mtp_tensors", return_value=True),
        ):
            args, note = _speculative_args(
                "/bin/llama-server", resolved_gguf="/m/Qwen3.8-27B-UD-Q4_K_M.gguf",
                path="/m/Qwen3.8-27B-UD-Q4_K_M.gguf", repo="unsloth/Qwen3.8-27B-GGUF",
            )
        self.assertEqual(args, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"])


class DFlash2PreferenceTests(unittest.TestCase):
    """llama.cpp's ``draft-dflash`` runs every DFlash-family drafter and picks
    the DFlash 2 code from the drafter's metadata (b12441 logs ``adding
    speculative implementation 'draft-dflash2'``), so the app keeps emitting
    ``draft-dflash``; a block drafter now outranks a baked-in MTP head."""

    def _args(self, *, drafter, mtp: bool):
        from backend_service.inference.llama_cpp_engine import _speculative_args

        with (
            mock.patch("backend_service.inference.llama_cpp_engine._llama_server_supports", return_value=True),
            mock.patch("backend_service.inference._mtp.model_has_mtp_tensors", return_value=mtp),
            mock.patch("backend_service.helpers.text_gguf.resolve_pinned_drafter", return_value=drafter),
            mock.patch("backend_service.helpers.text_gguf.resolve_spec_sidecar", return_value=None),
        ):
            return _speculative_args(
                "/bin/llama-server", resolved_gguf="/m/Qwen3.8-27B-Q4_K_M.gguf",
                path="/m/Qwen3.8-27B-Q4_K_M.gguf", repo="lmstudio-community/Qwen3.8-27B-GGUF",
            )

    def test_catalog_qwen38_gguf_is_paired_with_the_dflash2_drafter(self):
        from backend_service.helpers.text_gguf import pinned_drafter

        self.assertEqual(
            pinned_drafter("lmstudio-community/Qwen3.8-27B-GGUF"),
            ("z-lab/Qwen3.8-27B-DFlash2-GGUF", "Qwen3.8-27B-DFlash2-Q8_0.gguf", "draft-dflash"),
        )

    def test_block_drafter_outranks_a_baked_in_mtp_head(self):
        args, note = self._args(drafter=("draft-dflash", "/d/Qwen3.8-27B-DFlash2-Q8_0.gguf"), mtp=True)
        self.assertEqual(
            args,
            ["--model-draft", "/d/Qwen3.8-27B-DFlash2-Q8_0.gguf", "--spec-type", "draft-dflash", "--spec-draft-n-max", "15"],
        )
        self.assertIn("Qwen3.8-27B-DFlash2-Q8_0.gguf", note)

    def test_explicit_dflash2_type_also_asks_for_a_full_block(self):
        # Without the full block llama.cpp drafts 3 tokens and the drafter
        # lost a third of its speed (26 against 37 tok/s on the M4 Max).
        args, _ = self._args(drafter=("draft-dflash2", "/d/x.gguf"), mtp=False)
        self.assertEqual(args[-2:], ["--spec-draft-n-max", "15"])

    def test_mtp_head_still_used_without_a_drafter(self):
        args, _ = self._args(drafter=None, mtp=True)
        self.assertEqual(args[:2], ["--spec-type", "draft-mtp"])
