"""Qwen3-TTS speed-ups in the Qwen worker: the plain code-predictor loop and the per-clip clone prompt cache.
Fakes only; no GPU needed."""

from __future__ import annotations

import os
import types
import unittest
from unittest import mock

from backend_service import qwen_tts_fast as fast
from backend_service import voice_worker_torch as wt

try:
    import torch
except ImportError:  # pragma: no cover - the CI image has torch
    torch = None


class FakePredictor:
    """Records each forward call; logits pick token ``generation_steps`` (greedy) so the loop's order is visible."""

    def __init__(self, vocab: int = 8) -> None:
        self.calls: list[dict] = []
        self.vocab = vocab

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        step = 0 if "inputs_embeds" in kwargs else kwargs["generation_steps"]
        logits = torch.full((1, 1, self.vocab), -1.0)
        logits[0, -1, step % self.vocab] = 5.0
        return types.SimpleNamespace(logits=logits, generation_steps=step + 1)


@unittest.skipUnless(torch, "torch is not installed here")
class LoopTests(unittest.TestCase):
    def test_the_loop_prefills_once_then_one_pass_per_code_with_the_library_step_numbers(self) -> None:
        predictor = FakePredictor()
        original = mock.Mock()
        generate = fast.make_fast_generate(predictor, original)
        out = generate(inputs_embeds=torch.zeros(1, 2, 4), max_new_tokens=5, do_sample=False, top_k=50, top_p=1.0,
                       temperature=0.9, output_hidden_states=True, return_dict_in_generate=True)
        self.assertEqual(out.sequences.tolist(), [[0, 1, 2, 3, 4]])
        original.assert_not_called()
        self.assertEqual(len(predictor.calls), 5)  # prefill + 4 (no pass after the last code)
        self.assertEqual(predictor.calls[0]["cache_position"].tolist(), [0, 1])
        self.assertEqual([c["cache_position"].tolist() for c in predictor.calls[1:]], [[2], [3], [4], [5]])
        self.assertEqual([c["generation_steps"] for c in predictor.calls[1:]], [1, 2, 3, 4])
        caches = {id(c["past_key_values"]) for c in predictor.calls}
        self.assertEqual(len(caches), 1)  # one KV cache for the whole talker step

    def test_any_other_call_goes_to_the_library(self) -> None:
        original = mock.Mock(return_value="library")
        generate = fast.make_fast_generate(FakePredictor(), original)
        self.assertEqual(generate(inputs_embeds=torch.zeros(1, 2, 4), max_new_tokens=3, streamer=object()), "library")
        self.assertEqual(generate(input_ids=torch.zeros(1, 1), max_new_tokens=3), "library")

    def test_a_failing_loop_falls_back_to_the_library_with_a_note_and_stays_off(self) -> None:
        predictor = mock.Mock(side_effect=RuntimeError("boom"))
        original = mock.Mock(return_value="library")
        notes: list[str] = []
        generate = fast.make_fast_generate(predictor, original, notes=notes)
        for _ in range(2):
            self.assertEqual(generate(inputs_embeds=torch.zeros(1, 2, 4), max_new_tokens=3, do_sample=True), "library")
        self.assertEqual(predictor.call_count, 1)
        self.assertEqual(len(notes), 1)
        self.assertIn("boom", notes[0])

    def test_sampling_matches_temperature_then_top_k_then_one_multinomial_draw(self) -> None:
        logits = torch.tensor([[0.0, 3.0, 1.0, 2.0]])
        self.assertEqual(fast.sample_next(logits, False, 50, 1.0, 0.9).tolist(), [1])
        with mock.patch.object(torch, "multinomial", wraps=torch.multinomial) as draw:
            token = fast.sample_next(logits, True, 2, 1.0, 0.5)
        probs = draw.call_args.args[0]
        self.assertEqual(probs[0, 0].item(), 0.0)  # outside the top 2
        self.assertEqual(probs[0, 2].item(), 0.0)
        expected = torch.softmax(torch.tensor([3.0, 2.0]) / 0.5, dim=-1)
        self.assertAlmostEqual(probs[0, 1].item(), expected[0].item(), places=5)
        self.assertIn(token.item(), (1, 3))

    def test_installing_twice_keeps_one_wrapper_and_an_unknown_layout_changes_nothing(self) -> None:
        predictor = types.SimpleNamespace(generate=mock.Mock())
        model = types.SimpleNamespace(model=types.SimpleNamespace(talker=types.SimpleNamespace(code_predictor=predictor)))
        self.assertTrue(fast.fast_code_predictor(model))
        first = predictor.generate
        self.assertTrue(fast.fast_code_predictor(model))
        self.assertIs(predictor.generate, first)
        self.assertFalse(fast.fast_code_predictor(types.SimpleNamespace()))


class PromptCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        fast.clear_prompts()

    def test_the_reference_is_encoded_once_per_clip_transcript_and_mode(self) -> None:
        model = mock.Mock()
        model.create_voice_clone_prompt.side_effect = lambda **kw: [kw["ref_text"]]
        stat = mock.Mock(return_value=types.SimpleNamespace(st_mtime=1.0))
        first = fast.cached_clone_prompt(model, "/r.wav", "Hello.", False, stat=stat)
        self.assertIs(fast.cached_clone_prompt(model, "/r.wav", "Hello.", False, stat=stat), first)
        self.assertEqual(model.create_voice_clone_prompt.call_count, 1)
        fast.cached_clone_prompt(model, "/r.wav", None, True, stat=stat)  # embedding-only mode is its own entry
        stat.return_value = types.SimpleNamespace(st_mtime=2.0)  # the clip was replaced
        fast.cached_clone_prompt(model, "/r.wav", "Hello.", False, stat=stat)
        self.assertEqual(model.create_voice_clone_prompt.call_count, 3)
        self.assertEqual(model.create_voice_clone_prompt.call_args.kwargs,
                         {"ref_audio": "/r.wav", "ref_text": "Hello.", "x_vector_only_mode": False})


class WorkerWiringTests(unittest.TestCase):
    def test_a_clone_reuses_the_cached_prompt(self) -> None:
        import numpy as np

        model = mock.Mock()
        model.generate_voice_clone.return_value = ([np.zeros(10, dtype=np.float32)], 24_000)
        with mock.patch.object(wt, "readable_audio", return_value="/tmp/ref.wav"), \
                mock.patch.object(fast, "cached_clone_prompt", return_value=["prompt"]) as cached:
            wt.qwen_call(model, {"text": "Hi.", "refAudioPath": "/v/ref.wav", "refText": " Hello. "})
        cached.assert_called_once_with(model, "/tmp/ref.wav", "Hello.", False)
        self.assertEqual(model.generate_voice_clone.call_args.kwargs,
                         {"text": "Hi.", "language": "auto", "voice_clone_prompt": ["prompt"]})

    def test_the_speed_up_can_be_switched_off_and_never_blocks_loading(self) -> None:
        with mock.patch.object(fast, "fast_code_predictor") as install, \
                mock.patch.dict(os.environ, {"CHAOSENGINE_QWEN_FAST": "0"}):
            wt.speed_up_qwen(object())
        install.assert_not_called()
        with mock.patch.object(fast, "fast_code_predictor", side_effect=RuntimeError("x")), \
                mock.patch.dict(os.environ, {"CHAOSENGINE_QWEN_FAST": "1"}):
            wt.speed_up_qwen(object())  # no exception


if __name__ == "__main__":
    unittest.main()
