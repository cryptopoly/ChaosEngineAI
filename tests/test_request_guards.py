import json
import unittest

from backend_service.helpers.generation_limits import (
    align_video_frames,
    distilled_steps_note,
    normalize_image_size,
    normalize_video_size,
    snap_to_multiple,
)
from backend_service.helpers.pipeline_kwargs import filter_pipeline_kwargs
from backend_service.helpers.request_guards import (
    empty_reply_note,
    friendly_llama_error,
    guard_chat_request,
    merge_notes,
)
from backend_service.helpers.images import _image_variant_inputs
from backend_service.reasoning_split import strip_thinking_tokens


def _guard(**overrides):
    args = dict(engine="llama.cpp", vision_enabled=True, images=None, samplers=None, json_schema=None)
    args.update(overrides)
    return guard_chat_request(**args)


class ChatGuardTests(unittest.TestCase):
    def test_zero_repeat_penalty_is_dropped_with_a_note(self):
        guard = _guard(samplers={"repeat_penalty": 0.0, "top_k": 40})
        self.assertEqual(guard.samplers, {"top_k": 40})
        self.assertTrue(any("Repeat penalty" in note for note in guard.notes))

    def test_valid_repeat_penalty_is_kept(self):
        guard = _guard(samplers={"repeat_penalty": 1.1})
        self.assertEqual(guard.samplers, {"repeat_penalty": 1.1})
        self.assertEqual(guard.notes, [])

    def test_xtc_with_zero_threshold_is_dropped(self):
        guard = _guard(samplers={"xtc_probability": 1.0, "xtc_threshold": 0.0, "top_k": 40})
        self.assertEqual(guard.samplers, {"top_k": 40})
        self.assertTrue(any("XTC" in note for note in guard.notes))

    def test_xtc_without_threshold_gets_the_llama_default(self):
        guard = _guard(engine="mlx", samplers={"xtc_probability": 0.5})
        self.assertEqual(guard.samplers, {"xtc_probability": 0.5, "xtc_threshold": 0.1})

    def test_xtc_off_leaves_nothing_behind(self):
        guard = _guard(samplers={"xtc_probability": 0.0, "xtc_threshold": 0.2})
        self.assertIsNone(guard.samplers)

    def test_images_are_dropped_for_a_model_without_vision(self):
        for engine in ("llama.cpp", "mlx"):
            guard = _guard(engine=engine, vision_enabled=False, images=["abc"])
            self.assertIsNone(guard.images)
            self.assertTrue(any("cannot read images" in note for note in guard.notes))

    def test_images_are_kept_when_vision_is_on(self):
        guard = _guard(vision_enabled=True, images=["abc"])
        self.assertEqual(guard.images, ["abc"])

    def test_llama_only_samplers_are_reported_on_mlx(self):
        guard = _guard(engine="mlx", samplers={"mirostat": 2, "dry_multiplier": 0.8, "top_k": 40})
        note = " ".join(guard.notes)
        self.assertIn("Mirostat", note)
        self.assertIn("DRY", note)

    def test_llama_only_samplers_are_quiet_on_llama(self):
        guard = _guard(engine="llama.cpp", samplers={"mirostat": 2, "dry_multiplier": 0.8})
        self.assertEqual(guard.notes, [])

    def test_mirostat_off_is_not_reported(self):
        guard = _guard(engine="mlx", samplers={"mirostat": 0})
        self.assertEqual(guard.notes, [])

    def test_json_schema_is_flagged_on_mlx_only(self):
        self.assertTrue(_guard(engine="mlx", json_schema={"type": "object"}).notes)
        self.assertEqual(_guard(engine="llama.cpp", json_schema={"type": "object"}).notes, [])


class EmptyReplyTests(unittest.TestCase):
    def test_no_tokens_no_note(self):
        self.assertIsNone(empty_reply_note(completion_tokens=0, finish_reason="stop", had_reasoning=False, json_schema=None))

    def test_budget_spent_on_thinking(self):
        note = empty_reply_note(completion_tokens=200, finish_reason="length", had_reasoning=True, json_schema=None)
        self.assertIn("thinking", note)
        self.assertIn("Max tokens", note)

    def test_schema_that_produced_nothing(self):
        note = empty_reply_note(completion_tokens=200, finish_reason="length", had_reasoning=False, json_schema={"type": "object"})
        self.assertIn("JSON schema", note)

    def test_merge_notes_skips_empty_parts(self):
        self.assertEqual(merge_notes(None, ["a", "b"]), "a · b")
        self.assertEqual(merge_notes("x", []), "x")
        self.assertIsNone(merge_notes(None, []))


class LlamaErrorTests(unittest.TestCase):
    def test_context_overflow_names_both_sizes(self):
        body = json.dumps({"error": {"type": "exceed_context_size_error", "message": "request (18027 tokens) exceeds", "n_prompt_tokens": 18027, "n_ctx": 8192}})
        message = friendly_llama_error(body)
        self.assertIn("18027", message)
        self.assertIn("8192", message)
        self.assertIn("Context", message)

    def test_image_error_is_plain(self):
        body = json.dumps({"error": {"code": 500, "message": "image input is not supported - hint: provide the mmproj", "type": "server_error"}})
        self.assertIn("cannot read images", friendly_llama_error(body))

    def test_other_errors_keep_the_server_message(self):
        body = json.dumps({"error": {"message": "something specific", "type": "invalid_request_error"}})
        self.assertEqual(friendly_llama_error(body), "something specific")

    def test_non_json_passes_through(self):
        self.assertEqual(friendly_llama_error("plain text"), "plain text")


class UnopenedThinkTagTests(unittest.TestCase):
    def test_reasoning_before_a_lone_close_tag_is_dropped(self):
        self.assertEqual(strip_thinking_tokens("17 * 23 = 391.</think>The answer is 391."), "The answer is 391.")

    def test_a_normal_block_is_unchanged(self):
        self.assertEqual(strip_thinking_tokens("<think>work</think>done"), "done")


class FakePipeline:
    def __call__(self, prompt, width, height, num_inference_steps, guidance_scale=None, generator=None):
        return None


class FakeOpenPipeline:
    def __call__(self, prompt, **kwargs):
        return None


class PipelineKwargsTests(unittest.TestCase):
    def test_unsupported_arguments_are_dropped_and_named(self):
        kwargs = {"prompt": "x", "width": 512, "height": 512, "num_inference_steps": 4, "negative_prompt": "bad", "num_images_per_prompt": 2}
        accepted, notes = filter_pipeline_kwargs(FakePipeline(), kwargs)
        self.assertNotIn("negative_prompt", accepted)
        self.assertNotIn("num_images_per_prompt", accepted)
        self.assertEqual(accepted["width"], 512)
        self.assertEqual(len(notes), 1)
        self.assertIn("negative prompt", notes[0])
        self.assertIn("batch size", notes[0])

    def test_supported_arguments_pass_untouched(self):
        kwargs = {"prompt": "x", "width": 1, "height": 1, "num_inference_steps": 1, "guidance_scale": 3.0}
        accepted, notes = filter_pipeline_kwargs(FakePipeline(), kwargs)
        self.assertEqual(accepted, kwargs)
        self.assertEqual(notes, [])

    def test_a_pipeline_taking_kwargs_is_left_alone(self):
        kwargs = {"prompt": "x", "anything": 1}
        accepted, notes = filter_pipeline_kwargs(FakeOpenPipeline(), kwargs)
        self.assertEqual(accepted, kwargs)
        self.assertEqual(notes, [])

    def test_progress_callback_is_dropped_quietly(self):
        accepted, notes = filter_pipeline_kwargs(FakePipeline(), {"prompt": "x", "callback_on_step_end": object()})
        self.assertNotIn("callback_on_step_end", accepted)
        self.assertEqual(notes, [])

    def test_private_flags_survive_for_the_runtime_to_strip(self):
        accepted, _ = filter_pipeline_kwargs(FakePipeline(), {"prompt": "x", "__cfg_decay": True})
        self.assertIn("__cfg_decay", accepted)

    def test_an_unreadable_signature_changes_nothing(self):
        kwargs = {"prompt": "x", "weird": 1}
        accepted, notes = filter_pipeline_kwargs(object(), kwargs)
        self.assertEqual(accepted, kwargs)
        self.assertEqual(notes, [])


class SizeAndFrameTests(unittest.TestCase):
    def test_image_size_snaps_to_sixteen(self):
        width, height, note = normalize_image_size(1001, 777)
        self.assertEqual((width, height), (1008, 784))
        self.assertIn("multiples of 16", note)

    def test_a_valid_image_size_is_untouched(self):
        self.assertEqual(normalize_image_size(1024, 768), (1024, 768, None))

    def test_image_size_respects_the_model_minimum(self):
        width, height, _ = normalize_image_size(300, 300, min_side=768)
        self.assertEqual((width, height), (768, 768))

    def test_image_size_never_exceeds_the_cap(self):
        width, height, _ = normalize_image_size(2047, 2047)
        self.assertLessEqual(max(width, height), 2048)

    def test_snap_ties_round_up_and_stay_in_range(self):
        self.assertEqual(snap_to_multiple(24, 16, lo=16, hi=2048), 32)
        self.assertEqual(snap_to_multiple(5000, 16, lo=16, hi=2048), 2048)
        self.assertEqual(snap_to_multiple(1, 16, lo=16, hi=2048), 16)

    def test_ltx_video_size_snaps_to_thirty_two(self):
        width, height, note = normalize_video_size(770, 500, "Lightricks/LTX-Video")
        self.assertEqual((width, height), (768, 512))
        self.assertIn("multiples of 32", note)

    def test_wan_video_size_snaps_to_sixteen(self):
        width, height, _ = normalize_video_size(833, 481, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        self.assertEqual((width, height), (832, 480))

    def test_wan_frames_align_down_to_four_k_plus_one(self):
        self.assertEqual(align_video_frames("Wan-AI/Wan2.2-TI2V-5B-Diffusers", 50)[0], 49)
        self.assertEqual(align_video_frames("Wan-AI/Wan2.2-TI2V-5B-Diffusers", 49), (49, None))

    def test_ltx_frames_align_to_eight_k_plus_one(self):
        frames, note = align_video_frames("prince-canuma/LTX-2.3-distilled", 100)
        self.assertEqual(frames, 97)
        self.assertIn("8k+1", note)

    def test_short_clips_round_up_to_the_smallest_valid_count(self):
        self.assertEqual(align_video_frames("Lightricks/LTX-Video", 3)[0], 9)

    def test_unknown_models_keep_their_frame_count(self):
        self.assertEqual(align_video_frames("some/other-model", 50), (50, None))


class DistilledStepsTests(unittest.TestCase):
    def test_many_steps_on_a_turbo_model_are_explained_not_changed(self):
        note = distilled_steps_note("Qwen-Image-2.1 Turbo", 40, 8)
        self.assertIn("tuned for 8 steps", note)
        self.assertIn("5 times", note)

    def test_the_tuned_count_and_a_small_overshoot_are_quiet(self):
        self.assertIsNone(distilled_steps_note("Iris-3B", 4, 4))
        self.assertIsNone(distilled_steps_note("Iris-3B", 8, 4))

    def test_full_step_models_and_unknown_defaults_are_quiet(self):
        self.assertIsNone(distilled_steps_note("FLUX.1 dev", 60, 28))
        self.assertIsNone(distilled_steps_note("Model", 60, None))


class ImageInputsTests(unittest.TestCase):
    def test_a_distilled_mflux_model_takes_neither_guidance_nor_a_negative_prompt(self):
        self.assertEqual(
            _image_variant_inputs({"engine": "mflux", "repo": "speridlabs/iris-3b"}),
            {"acceptsGuidance": False, "acceptsNegativePrompt": False},
        )

    def test_qwen_image_takes_a_negative_prompt_but_no_guidance(self):
        self.assertEqual(
            _image_variant_inputs({"engine": "mflux", "repo": "Qwen/Qwen-Image-2.1"}),
            {"acceptsGuidance": False, "acceptsNegativePrompt": True},
        )

    def test_other_engines_take_both(self):
        self.assertEqual(
            _image_variant_inputs({"repo": "stabilityai/stable-diffusion-xl-base-1.0"}),
            {"acceptsGuidance": True, "acceptsNegativePrompt": True},
        )


if __name__ == "__main__":
    unittest.main()


class GatedTokenMessageTests(unittest.TestCase):
    def test_a_fine_grained_token_without_gated_access_gets_exact_steps(self) -> None:
        from backend_service.helpers.hf_errors import _friendly_hf_download_error

        error = (
            "403 Forbidden: Please enable access to public gated repositories in your "
            "fine-grained token settings to view this repository."
        )
        message = _friendly_hf_download_error("dgrauet/ltx-2.5-mlx-q4", error)
        self.assertIn("settings/tokens", message)
        self.assertIn("gated", message)
        self.assertIn("licence", message)

    def test_a_plain_403_keeps_the_generic_message(self) -> None:
        from backend_service.helpers.hf_errors import _friendly_hf_download_error

        message = _friendly_hf_download_error("org/model", "HTTP 403 refused")
        self.assertIn("refused access", message)


class ContextOverflowStatusTests(unittest.TestCase):
    def test_a_context_overflow_is_a_not_usable_request(self) -> None:
        from backend_service.helpers.request_guards import RequestNotUsable, llama_error

        body = json.dumps({"error": {"type": "exceed_context_size_error", "n_prompt_tokens": 9000, "n_ctx": 4096, "message": "x"}})
        error = llama_error(body, "fallback")
        self.assertIsInstance(error, RequestNotUsable)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("9000", str(error))

    def test_other_errors_stay_runtime_errors(self) -> None:
        from backend_service.helpers.request_guards import RequestNotUsable, llama_error

        error = llama_error(json.dumps({"error": {"type": "server_error", "message": "boom"}}), "fallback")
        self.assertNotIsInstance(error, RequestNotUsable)
        self.assertEqual(str(error), "boom")
        self.assertEqual(str(llama_error("", "fallback")), "fallback")


class JsonSchemaFallbackTests(unittest.TestCase):
    """A reasoning model under a JSON grammar can answer with nothing; the controller asks in words."""

    SCHEMA = {"type": "object", "properties": {"answer": {"type": "number"}}}

    def _controller(self, calls, *, engine_name="llama.cpp"):
        from types import SimpleNamespace

        from backend_service.inference.base import GenerationResult, StreamChunk
        from backend_service.inference.controller import RuntimeController

        def result(text):
            return GenerationResult(text=text, finishReason="stop", promptTokens=5, completionTokens=7,
                                    totalTokens=12, tokS=10.0, responseSeconds=0.5)

        class Engine:
            def generate(self, **kwargs):
                calls.append(("generate", kwargs.get("json_schema"), kwargs.get("system_prompt")))
                return result("" if kwargs.get("json_schema") else '```json\n{"answer": 391}\n```')

            def stream_generate(self, **kwargs):
                calls.append(("stream", kwargs.get("json_schema"), kwargs.get("system_prompt")))
                yield StreamChunk(reasoning="thinking")
                yield StreamChunk(done=True, finish_reason="length", completion_tokens=50, total_tokens=55)

        controller = RuntimeController.__new__(RuntimeController)
        controller.loaded_model = SimpleNamespace(engine=engine_name, visionEnabled=False)
        controller.engine = Engine()
        controller.runtime_note = None
        return controller

    def test_an_empty_reply_under_a_grammar_is_retried_as_plain_json(self):
        calls = []
        controller = self._controller(calls)
        result = controller.generate(prompt="17*23?", history=[], system_prompt=None, max_tokens=64,
                                     temperature=0.0, json_schema=self.SCHEMA)
        self.assertEqual(result.text, '{"answer": 391}')
        self.assertIn("plain JSON", result.runtimeNote)
        self.assertEqual([c[0] for c in calls], ["generate", "generate"])
        self.assertIsNone(calls[1][1])
        self.assertIn('"answer"', calls[1][2])

    def test_the_streamed_reply_gets_the_same_retry(self):
        calls = []
        controller = self._controller(calls)
        chunks = list(controller.stream_generate(prompt="17*23?", history=[], system_prompt=None, max_tokens=64,
                                                 temperature=0.0, json_schema=self.SCHEMA))
        text = "".join(c.text or "" for c in chunks)
        self.assertEqual(text, '{"answer": 391}')
        self.assertTrue(chunks[-1].done)
        self.assertIn("plain JSON", chunks[-1].runtime_note)

    def test_other_engines_are_left_alone(self):
        calls = []
        controller = self._controller(calls, engine_name="mlx")
        controller.generate(prompt="x", history=[], system_prompt=None, max_tokens=8, temperature=0.0)
        self.assertEqual(len(calls), 1)

    def test_a_fenced_reply_is_unwrapped_only_when_the_whole_reply_is_fenced(self):
        from backend_service.helpers.request_guards import strip_json_fence

        self.assertEqual(strip_json_fence('```json\n{"a": 1}\n```'), '{"a": 1}')
        self.assertEqual(strip_json_fence('```\n[1, 2]\n```'), "[1, 2]")
        self.assertEqual(strip_json_fence('text ```json\n{"a": 1}\n``` more'), 'text ```json\n{"a": 1}\n``` more')
        self.assertEqual(strip_json_fence('{"a": 1}'), '{"a": 1}')
