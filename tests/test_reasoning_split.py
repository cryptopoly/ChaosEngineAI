"""Tests for the reasoning-split layer (Bug 2: Gemma 4 channel-token leak)."""

from __future__ import annotations

import unittest

from backend_service.reasoning_split import (
    ThinkingTokenFilter,
    reasoning_delimiters_for,
    strip_harmony_boilerplate,
)


class ReasoningDelimitersForTests(unittest.TestCase):
    """``reasoning_delimiters_for`` must return Harmony tags for Gemma 4
    + gpt-oss families, and the default ``<think>...</think>`` for
    everything else."""

    def test_default_for_unknown_model(self):
        self.assertEqual(reasoning_delimiters_for(None), ("<think>", "</think>"))
        self.assertEqual(reasoning_delimiters_for(""), ("<think>", "</think>"))
        self.assertEqual(
            reasoning_delimiters_for("Qwen/Qwen3-7B"),
            ("<think>", "</think>"),
        )
        self.assertEqual(
            reasoning_delimiters_for("deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
            ("<think>", "</think>"),
        )

    def test_gemma_4_canonical_uses_harmony(self):
        self.assertEqual(
            reasoning_delimiters_for("google/gemma-4-26B-A4B-it"),
            ("<|channel|>thought", "<|end|>"),
        )
        self.assertEqual(
            reasoning_delimiters_for("google/gemma-4-E4B-it"),
            ("<|channel|>thought", "<|end|>"),
        )

    def test_gemma_4_community_mirrors_use_harmony(self):
        self.assertEqual(
            reasoning_delimiters_for("mlx-community/gemma-4-26b-a4b-it-5bit"),
            ("<|channel|>thought", "<|end|>"),
        )
        self.assertEqual(
            reasoning_delimiters_for("lmstudio-community/gemma-4-12B-it"),
            ("<|channel|>thought", "<|end|>"),
        )

    def test_gemma_3_falls_through_to_default(self):
        # Gemma 3 emits plain text (no Harmony channels). Defaults apply.
        self.assertEqual(
            reasoning_delimiters_for("google/gemma-3-12b-it"),
            ("<think>", "</think>"),
        )
        self.assertEqual(
            reasoning_delimiters_for("mlx-community/gemma-3-9b-it-8bit"),
            ("<think>", "</think>"),
        )

    def test_gpt_oss_uses_harmony(self):
        self.assertEqual(
            reasoning_delimiters_for("openai/gpt-oss-20b"),
            ("<|channel|>thought", "<|end|>"),
        )

    def test_case_insensitive_match(self):
        self.assertEqual(
            reasoning_delimiters_for("GOOGLE/GEMMA-4-26B-A4B-IT"),
            ("<|channel|>thought", "<|end|>"),
        )


class StripHarmonyBoilerplateTests(unittest.TestCase):
    """Harmony channel boilerplate (``<|start|>``, ``<|channel|>``,
    ``<|message|>``, ``<|end|>``, ``<|return|>``) must be removed from
    user-visible text after the ThinkingTokenFilter pass."""

    def test_idempotent_on_plain_text(self):
        self.assertEqual(strip_harmony_boilerplate("Hello world."), "Hello world.")
        self.assertEqual(strip_harmony_boilerplate(""), "")

    def test_idempotent_on_qwen_xml_thinking(self):
        # Qwen3 / DeepSeek output uses <think>...</think> XML tags. The
        # Harmony stripper must not touch those.
        text = "Some text <think>reasoning</think> answer."
        self.assertEqual(strip_harmony_boilerplate(text), text)

    def test_strips_start_assistant(self):
        text = "<|start|>assistant Hello there"
        self.assertEqual(strip_harmony_boilerplate(text), "Hello there")

    def test_strips_channel_final_message(self):
        text = "<|channel|>final<|message|>The answer is 42."
        self.assertEqual(strip_harmony_boilerplate(text), "The answer is 42.")

    def test_strips_end_token(self):
        text = "Final answer.<|end|>"
        self.assertEqual(strip_harmony_boilerplate(text), "Final answer.")

    def test_strips_return_token(self):
        text = "Bye!<|return|>"
        self.assertEqual(strip_harmony_boilerplate(text), "Bye!")

    def test_strips_full_harmony_response(self):
        text = (
            "<|start|>assistant<|channel|>final<|message|>"
            "The capital of France is Paris.<|end|>"
        )
        self.assertEqual(
            strip_harmony_boilerplate(text),
            "The capital of France is Paris.",
        )

    def test_collapses_excess_blank_lines(self):
        text = "Para 1.\n\n\n\n\nPara 2."
        self.assertEqual(strip_harmony_boilerplate(text), "Para 1.\n\nPara 2.")


class GemmaThinkFilterIntegrationTests(unittest.TestCase):
    """End-to-end: feed a Gemma-4-shaped Harmony stream through
    ThinkingTokenFilter with the registered delimiters, then post-strip
    boilerplate. The user-visible text should be the final answer only."""

    def test_extracts_thought_channel_into_reasoning(self):
        open_tag, close_tag = reasoning_delimiters_for("google/gemma-4-26B-A4B-it")
        filt = ThinkingTokenFilter(
            detect_raw_reasoning=True,
            open_tag=open_tag,
            close_tag=close_tag,
        )
        # Simulate Gemma 4 Harmony output.
        stream = (
            "<|start|>assistant"
            "<|channel|>thought"
            "<|message|>The user asks about caching. I should explain LRU.<|end|>"
            "<|start|>assistant"
            "<|channel|>final"
            "<|message|>LRU caches evict least-recently-used entries first.<|end|>"
        )
        result = filt.feed(stream)
        flushed = filt.flush()
        text = strip_harmony_boilerplate(
            f"{result.text}{flushed.text}".strip()
        )
        self.assertEqual(
            text,
            "LRU caches evict least-recently-used entries first.",
        )

    def test_default_filter_path_still_works_for_qwen(self):
        # Regression check: Qwen3-style <think>...</think> still splits.
        open_tag, close_tag = reasoning_delimiters_for("Qwen/Qwen3-8B")
        filt = ThinkingTokenFilter(
            detect_raw_reasoning=True,
            open_tag=open_tag,
            close_tag=close_tag,
        )
        result = filt.feed("<think>hidden reasoning</think>The answer is 42.")
        flushed = filt.flush()
        text = strip_harmony_boilerplate(
            f"{result.text}{flushed.text}".strip()
        )
        self.assertEqual(text, "The answer is 42.")


if __name__ == "__main__":
    unittest.main()
