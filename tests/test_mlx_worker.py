import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend_service.mlx_worker import (
    RunawayGuard,
    ThinkingTokenFilter,
    TranscriptLoopFilter,
    WorkerState,
    _build_prompt_text,
    _should_retry_cache_failure,
    _strip_thinking_tokens,
    _trim_transcript_continuation,
)


class _MissingTemplateTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        raise ValueError(
            "Cannot use chat template functions because tokenizer.chat_template is not set "
            "and no template argument was passed!"
        )


class _WorkingTemplateTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return "<templated>"


class MlxWorkerPromptTests(unittest.TestCase):
    def test_uses_chat_template_when_available(self):
        prompt, note = _build_prompt_text(
            _WorkingTemplateTokenizer(),
            history=[],
            prompt="Hello",
            system_prompt="Be concise.",
        )

        self.assertEqual(prompt, "<templated>")
        self.assertIsNone(note)

    def test_falls_back_when_chat_template_is_missing(self):
        prompt, note = _build_prompt_text(
            _MissingTemplateTokenizer(),
            history=[{"role": "assistant", "text": "Earlier reply"}],
            prompt="Hello",
            system_prompt="Be concise.",
        )

        self.assertIn("SYSTEM: Be concise.", prompt)
        self.assertIn("ASSISTANT: Earlier reply", prompt)
        self.assertIn("USER: Hello", prompt)
        self.assertTrue(prompt.endswith("ASSISTANT:"))
        self.assertIsNotNone(note)
        self.assertIn("chat template was unavailable", note.lower())

    def test_trim_transcript_continuation_keeps_first_answer(self):
        trimmed, stopped = _trim_transcript_continuation(
            "Hello!\n\nUSER: Can you check the repo?\n\nASSISTANT: Hello!"
        )

        self.assertTrue(stopped)
        self.assertEqual(trimmed, "Hello!")

    def test_trim_transcript_continuation_strips_leading_assistant_label(self):
        trimmed, stopped = _trim_transcript_continuation(
            "ASSISTANT: Hello!\n\nUSER: Can you check the repo?"
        )

        self.assertTrue(stopped)
        self.assertEqual(trimmed, "Hello!")

    def test_transcript_loop_filter_handles_split_stream_chunks(self):
        filter_ = TranscriptLoopFilter()

        first = filter_.feed("Hello!\n\nUSE")
        second = filter_.feed("R: Can you check the repo?\n\nASSISTANT: Hello!")
        flushed = filter_.flush()

        self.assertEqual(first, "Hello!\n")
        self.assertEqual(second, "")
        self.assertEqual(flushed, "")
        self.assertTrue(filter_.stopped)


class RunawayGuardTests(unittest.TestCase):
    def test_allows_normal_output(self):
        guard = RunawayGuard()
        guard.feed("Hello! How can I help you today?\n")
        guard.feed("I'm an AI assistant.\n")
        guard.flush()  # Should not raise

    def test_detects_repeated_lines(self):
        guard = RunawayGuard(min_line_length=20, max_repeats=3)
        with self.assertRaises(RuntimeError) as ctx:
            for _ in range(5):
                guard.feed("Wait, I will write 'Qwen3.5'. Let me try again.\n")
        self.assertIn("repeating itself", str(ctx.exception))

    def test_ignores_short_repeated_lines(self):
        guard = RunawayGuard(min_line_length=30, max_repeats=3)
        for _ in range(10):
            guard.feed("OK.\n")
        guard.flush()  # Should not raise — lines are too short

    def test_resets_on_different_line(self):
        guard = RunawayGuard(min_line_length=20, max_repeats=3)
        guard.feed("Wait, I will write 'Qwen3.5'. Let me try again.\n")
        guard.feed("Wait, I will write 'Qwen3.5'. Let me try again.\n")
        guard.feed("Actually, here is a completely different line of text.\n")
        guard.feed("Wait, I will write 'Qwen3.5'. Let me try again.\n")
        guard.flush()  # Should not raise — count reset on different line

    def test_detects_thinking_heading(self):
        guard = RunawayGuard()
        guard.feed("Thinking Process:\n")
        self.assertTrue(guard.saw_thinking_heading)

    def test_no_thinking_heading_in_normal_text(self):
        guard = RunawayGuard()
        guard.feed("Hello! I can help you with that.\n")
        self.assertFalse(guard.saw_thinking_heading)

    def test_detects_reasoning_loop(self):
        guard = RunawayGuard(max_reasoning_lines=10)
        with self.assertRaises(RuntimeError) as ctx:
            guard.feed("Wait, I should check the constraint again and verify.\n")
            guard.feed("Okay, I will just say 'Hello! How can I help?'\n")
            guard.feed("Actually, looking closer at the instruction again.\n")
            guard.feed("Wait, I need to check if I should explain more.\n")
            guard.feed("Let me re-read the constraint one more time now.\n")
            guard.feed("Wait, I should check the constraint once more time.\n")
        self.assertIn("reasoning loop", str(ctx.exception))

    def test_detects_alternating_reasoning_with_calculations(self):
        """Catches 'Wait, I need to confirm' / '31536000 seconds' loops."""
        guard = RunawayGuard(max_reasoning_lines=20)
        with self.assertRaises(RuntimeError):
            for _ in range(25):
                guard.feed("Wait, I need to confirm the calculation result.\n")
                guard.feed("31536000 seconds.\n")


class CacheProfileTests(unittest.TestCase):
    def test_native_profile_zeros_quantized_knobs(self):
        worker = WorkerState()

        note = worker._apply_cache_profile(
            cache_strategy="native",
            cache_bits=4,
            fp16_layers=8,
            fused_attention=False,
        )

        self.assertIsNone(note)
        self.assertEqual(worker.cache_strategy, "native")
        self.assertEqual(worker.cache_bits, 0)
        self.assertEqual(worker.fp16_layers, 0)

    def test_retryable_cache_failures_include_swapaxes_attribute_errors(self):
        self.assertTrue(_should_retry_cache_failure(AttributeError("'tuple' object has no attribute 'swapaxes'")))
        self.assertTrue(_should_retry_cache_failure(RuntimeError("[broadcast_shapes] Shapes (1,1,117,48) and (1,1,117,51) cannot be broadcast.")))
        self.assertFalse(_should_retry_cache_failure(RuntimeError("Tokenizer chat template missing.")))


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, tokens):
        return "decoded output"


class DFlashCompatibilityTests(unittest.TestCase):
    def test_load_model_uses_dflash_runtime_bundle_api(self):
        worker = WorkerState()
        fake_model = SimpleNamespace(layers=[object(), object()])
        fake_tokenizer = _FakeTokenizer()
        fake_runtime = SimpleNamespace(
            configure_full_attention_split=Mock(),
            load_draft_bundle=Mock(return_value=("draft_bundle", {"config": True})),
        )
        fake_pkg = SimpleNamespace(runtime=fake_runtime)
        fake_mlx_lm = SimpleNamespace(
            load=lambda *args, **kwargs: (fake_model, fake_tokenizer, {"num_hidden_layers": 2, "num_attention_heads": 2, "hidden_size": 8}),
        )

        with patch.dict("sys.modules", {"mlx_lm": fake_mlx_lm, "dflash_mlx": fake_pkg, "dflash_mlx.runtime": fake_runtime}):
            result = worker.load_model(
                {
                    "target": "/tmp/model",
                    "speculativeDecoding": True,
                    "dflashDraftModel": "z-lab/Qwen3.5-35B-A3B-DFlash",
                    "treeBudget": 64,
                }
            )

        self.assertTrue(result["speculativeDecoding"])
        self.assertEqual(result["treeBudget"], 64)
        self.assertEqual(result["dflashDraftModel"], "z-lab/Qwen3.5-35B-A3B-DFlash")
        self.assertIs(worker._dflash_target, fake_model)
        self.assertEqual(worker._dflash_generator, "draft_bundle")
        self.assertIs(worker._ddtree_target, fake_model)
        self.assertEqual(worker._ddtree_draft, "draft_bundle")
        fake_runtime.configure_full_attention_split.assert_called_once_with(fake_model, enabled=True)
        fake_runtime.load_draft_bundle.assert_called_once_with(
            "z-lab/Qwen3.5-35B-A3B-DFlash",
            lazy=True,
        )

    def test_generate_dflash_uses_runtime_summary_shape(self):
        worker = WorkerState()
        worker.model = "standard_target"
        worker.tokenizer = _FakeTokenizer()
        worker._dflash_target = "dflash_target"
        worker._dflash_generator = "draft_bundle"

        def _fake_stream(**_kwargs):
            # Emit a couple of per-token events followed by the summary event,
            # matching the v0.1.4 ``stream_dflash_generate`` protocol.
            yield {"event": "token", "token_id": 10, "generated_tokens": 1}
            yield {"event": "token", "token_id": 11, "generated_tokens": 2}
            yield {"event": "token", "token_id": 12, "generated_tokens": 3}
            yield {
                "event": "summary",
                "generated_token_ids": [10, 11, 12],
                "generation_tokens": 3,
                "prompt_token_count": 3,
                "elapsed_us": 2_000_000,
                "accepted_from_draft": 8,
                "cycles_completed": 4,
                "phase_timings_us": {"prefill": 500_000},
                "peak_memory_gb": 12.34,
            }

        fake_runtime = SimpleNamespace(stream_dflash_generate=_fake_stream)
        fake_pkg = SimpleNamespace(runtime=fake_runtime)

        with patch.dict("sys.modules", {"dflash_mlx": fake_pkg, "dflash_mlx.runtime": fake_runtime}):
            result = worker._generate_dflash({"prompt": "hello", "maxTokens": 16})

        self.assertEqual(result["text"], "decoded output")
        self.assertEqual(result["promptTokens"], 3)
        self.assertEqual(result["completionTokens"], 3)
        self.assertEqual(result["totalTokens"], 6)
        self.assertEqual(result["dflashAcceptanceRate"], 2.0)
        self.assertGreater(result["tokS"], 0)
        self.assertIn("DFLASH speculative decoding", result["runtimeNote"])

    def test_stream_generate_notes_speculative_fallback_when_dflash_stream_path_fails(self):
        worker = WorkerState()
        worker.model = object()
        worker.tokenizer = _WorkingTemplateTokenizer()
        worker.speculative_decoding = True
        worker._dflash_generator = object()

        fake_response = SimpleNamespace(
            text="fallback answer",
            finish_reason="stop",
            prompt_tokens=3,
            generation_tokens=2,
            generation_tps=11.0,
            prompt_tps=0.0,
            peak_memory=1.25,
        )
        fake_mlx_lm = SimpleNamespace(
            stream_generate=lambda *args, **kwargs: [fake_response],
        )
        fake_sample_utils = SimpleNamespace(
            make_sampler=lambda **kwargs: object(),
        )
        emitted: list[dict[str, object]] = []

        with (
            patch.object(worker, "_generate_dflash", side_effect=RuntimeError("draft verifier failed")),
            patch("backend_service.mlx_worker._emit", side_effect=emitted.append),
            patch.dict("sys.modules", {"mlx_lm": fake_mlx_lm, "mlx_lm.sample_utils": fake_sample_utils}),
        ):
            worker.stream_generate({"prompt": "hello", "maxTokens": 8})

        done_payload = next(payload for payload in emitted if payload.get("done"))
        result = done_payload["result"]
        self.assertEqual(result["finishReason"], "stop")
        self.assertEqual(result["speculativeDecoding"], False)
        self.assertEqual(result["treeBudget"], 0)
        self.assertIn("Speculative decoding stream path failed", result["runtimeNote"])
        self.assertIn("Fell back to standard generation.", result["runtimeNote"])


class ThinkingTokenFilterTests(unittest.TestCase):
    @staticmethod
    def _collect(*parts):
        text = "".join(part.text for part in parts)
        reasoning = "".join(part.reasoning for part in parts)
        reasoning_done = any(part.reasoning_done for part in parts)
        return text, reasoning, reasoning_done

    def test_strips_xml_think_tags_streaming(self):
        f = ThinkingTokenFilter()
        out1 = f.feed("Hello <think>internal")
        out2 = f.feed(" reasoning</think> world")
        out3 = f.flush()
        text, reasoning, reasoning_done = self._collect(out1, out2, out3)
        self.assertEqual(text, "Hello  world")
        self.assertEqual(reasoning, "internal reasoning")
        self.assertTrue(reasoning_done)

    def test_suppresses_raw_thinking_heading(self):
        f = ThinkingTokenFilter()
        out1 = f.feed("Thinking Process:\n")
        out2 = f.feed("1. Analyze the request\n")
        out3 = f.feed("Wait, I should check constraints\n")
        # Now feed actual content
        out4 = f.feed("Hello! I'm here to help.\n")
        out5 = f.flush()
        text, reasoning, reasoning_done = self._collect(out1, out2, out3, out4, out5)
        self.assertNotIn("Thinking Process", text)
        self.assertNotIn("Analyze", text)
        self.assertIn("Thinking Process", reasoning)
        self.assertIn("Analyze", reasoning)
        self.assertIn("Hello", text)
        self.assertTrue(reasoning_done)

    def test_passes_normal_text_through(self):
        f = ThinkingTokenFilter()
        out = f.feed("Hello! How can I help you today?")
        flushed = f.flush()
        text, reasoning, reasoning_done = self._collect(out, flushed)
        self.assertIn("Hello", text)
        self.assertEqual(reasoning, "")
        self.assertFalse(reasoning_done)

    def test_suppresses_token_by_token_thinking_heading(self):
        """Simulate real streaming where tokens arrive character-by-character."""
        f = ThinkingTokenFilter()
        # Model outputs "Thinking Process:\n..." one token at a time
        raw = "Thinking Process:\nAnalyze the request\nWait, I should check\nHello! I'm here to help.\n"
        parts = []
        for char in raw:
            parts.append(f.feed(char))
        parts.append(f.flush())
        combined, reasoning, reasoning_done = self._collect(*parts)
        # Should suppress thinking and only show the actual answer
        self.assertNotIn("Thinking Process", combined)
        self.assertNotIn("Analyze", combined)
        self.assertIn("Thinking Process", reasoning)
        self.assertIn("Hello", combined)
        self.assertTrue(reasoning_done)

    def test_releases_normal_text_after_first_line(self):
        """Normal text should stream through after the first line proves safe."""
        f = ThinkingTokenFilter()
        parts = []
        for char in "Hello! I'm here to help.\nHow can I assist?":
            parts.append(f.feed(char))
        parts.append(f.flush())
        combined, reasoning, reasoning_done = self._collect(*parts)
        self.assertIn("Hello", combined)
        self.assertIn("assist", combined)
        self.assertEqual(reasoning, "")
        self.assertFalse(reasoning_done)

    def test_handles_unclosed_xml_think(self):
        f = ThinkingTokenFilter()
        out = f.feed("<think>reasoning that never closes")
        flushed = f.flush()
        text, reasoning, reasoning_done = self._collect(out, flushed)
        self.assertEqual(text, "")
        self.assertEqual(reasoning, "reasoning that never closes")
        self.assertTrue(reasoning_done)

    def test_keeps_bullets_and_meta_sections_in_reasoning(self):
        f = ThinkingTokenFilter()
        parts = [
            f.feed("Thinking Process:\n"),
            f.feed("- Check constraints\n"),
            f.feed("  - Verify cache mode\n"),
            f.feed("Confidence Score: 0.78\n"),
            f.feed("Hello! Final answer.\n"),
            f.flush(),
        ]
        text, reasoning, reasoning_done = self._collect(*parts)
        self.assertEqual(text, "Hello! Final answer.\n")
        self.assertIn("- Check constraints", reasoning)
        self.assertIn("Confidence Score: 0.78", reasoning)
        self.assertTrue(reasoning_done)

    def test_flushes_reasoning_only_when_no_final_answer_arrives(self):
        f = ThinkingTokenFilter()
        parts = [
            f.feed("Thinking Process:\n"),
            f.feed("1. Check the requirements\n"),
            f.feed("- Verify the constraint\n"),
            f.feed("Mental Sandbox: compare two options"),
            f.flush(),
        ]
        text, reasoning, reasoning_done = self._collect(*parts)
        self.assertEqual(text, "")
        self.assertIn("1. Check the requirements", reasoning)
        self.assertIn("Mental Sandbox", reasoning)
        self.assertTrue(reasoning_done)

    def test_keeps_draft_and_verification_sections_inside_reasoning(self):
        f = ThinkingTokenFilter()
        parts = [
            f.feed("Thinking Process:\n"),
            f.feed("An LLM works by predicting the next word in a sequence.\n"),
            f.feed("4. Refining for Constraints: Ensure no internal reasoning tags.\n"),
            f.feed("*Current Draft:*\n"),
            f.feed("An LLM works by predicting the next word in a sequence.\n"),
            f.feed("*Word Count:* 168 words.\n"),
            f.feed("6. Final Verification:\n"),
            f.feed("- Under 200 words? Yes.\n"),
            f.flush(),
        ]
        text, reasoning, reasoning_done = self._collect(*parts)
        self.assertEqual(text, "")
        self.assertIn("An LLM works by predicting the next word", reasoning)
        self.assertIn("Current Draft", reasoning)
        self.assertIn("Word Count", reasoning)
        self.assertIn("Final Verification", reasoning)
        self.assertTrue(reasoning_done)


class StripThinkingTokensTests(unittest.TestCase):
    def test_strips_xml_think_tags(self):
        text = "<think>internal reasoning here</think>Hello!"
        self.assertEqual(_strip_thinking_tokens(text), "Hello!")

    def test_strips_unclosed_think_tag(self):
        text = "<think>reasoning that never closes"
        self.assertEqual(_strip_thinking_tokens(text), "")

    def test_strips_raw_thinking_heading(self):
        text = "Thinking Process:\nStep 1: analyze\nStep 2: decide\nFinal Check:\nHello!"
        result = _strip_thinking_tokens(text)
        self.assertEqual(result, "Hello!")

    def test_preserves_normal_text(self):
        text = "Hello! I'm an AI assistant."
        self.assertEqual(_strip_thinking_tokens(text), text)


if __name__ == "__main__":
    unittest.main()
