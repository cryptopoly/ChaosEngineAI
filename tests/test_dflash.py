"""Tests for the DFLASH speculative decoding integration module."""

import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from dflash import (
    DRAFT_MODEL_MAP,
    get_draft_model,
    is_mlx_available,
    is_vllm_available,
    is_available,
    supported_models,
    availability_info,
)


class DraftModelLookupTests(unittest.TestCase):
    """Tests for the draft model registry and fuzzy matching."""

    def test_exact_match(self):
        self.assertEqual(
            get_draft_model("Qwen/Qwen3-4B"),
            "z-lab/Qwen3-4B-DFlash-b16",
        )

    def test_exact_match_llama(self):
        self.assertEqual(
            get_draft_model("meta-llama/Llama-3.1-8B-Instruct"),
            "z-lab/Llama-3.1-8B-Instruct-DFlash",
        )

    def test_explicit_alias_mlx_community(self):
        self.assertEqual(
            get_draft_model("mlx-community/Qwen3-4B-bf16"),
            "z-lab/Qwen3-4B-DFlash-b16",
        )

    def test_explicit_alias_4bit(self):
        self.assertEqual(
            get_draft_model("mlx-community/Qwen3-4B-4bit"),
            "z-lab/Qwen3-4B-DFlash-b16",
        )

    def test_fuzzy_match_strips_quant_suffix(self):
        # Unknown quant variant but model name matches after stripping
        result = get_draft_model("mlx-community/Qwen3-8B-bf16")
        self.assertEqual(result, "z-lab/Qwen3-8B-DFlash-b16")

    def test_fuzzy_match_community_prefix_strip(self):
        # Community prefix stripped, then model name matched
        result = get_draft_model("mlx-community/Qwen3.5-7B-bf16")
        self.assertEqual(result, "z-lab/Qwen3.5-7B-DFlash")

    def test_unknown_model_returns_none(self):
        self.assertIsNone(get_draft_model("some-org/UnknownModel-7B"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(get_draft_model(""))

    def test_all_map_entries_return_values(self):
        for target, expected_draft in DRAFT_MODEL_MAP.items():
            result = get_draft_model(target)
            self.assertEqual(result, expected_draft, f"Failed for target: {target}")


class AvailabilityDetectionTests(unittest.TestCase):
    """Tests for DFLASH backend availability checks."""

    @patch("dflash.importlib.util.find_spec")
    def test_mlx_available_when_installed(self, mock_find_spec):
        mock_find_spec.return_value = SimpleNamespace(name="dflash_mlx")
        self.assertTrue(is_mlx_available())
        mock_find_spec.assert_called_with("dflash_mlx")

    @patch("dflash.importlib.util.find_spec", return_value=None)
    def test_mlx_unavailable_when_missing(self, mock_find_spec):
        self.assertFalse(is_mlx_available())

    @patch("dflash.importlib.util.find_spec")
    def test_vllm_available_when_dflash_model_exists(self, mock_find_spec):
        def find_spec_side_effect(name):
            if name == "dflash.model":
                return SimpleNamespace(name="dflash.model")
            return None
        mock_find_spec.side_effect = find_spec_side_effect
        self.assertTrue(is_vllm_available())

    @patch("dflash.importlib.util.find_spec", return_value=None)
    def test_vllm_unavailable_when_missing(self, mock_find_spec):
        self.assertFalse(is_vllm_available())

    @patch("dflash.is_mlx_available", return_value=True)
    @patch("dflash.is_vllm_available", return_value=False)
    def test_is_available_true_with_mlx_only(self, *_):
        self.assertTrue(is_available())

    @patch("dflash.is_mlx_available", return_value=False)
    @patch("dflash.is_vllm_available", return_value=True)
    def test_is_available_true_with_vllm_only(self, *_):
        self.assertTrue(is_available())

    @patch("dflash.is_mlx_available", return_value=False)
    @patch("dflash.is_vllm_available", return_value=False)
    def test_is_available_false_when_nothing_installed(self, *_):
        self.assertFalse(is_available())


class SupportedModelsTests(unittest.TestCase):
    def test_supported_models_returns_sorted_list(self):
        models = supported_models()
        self.assertIsInstance(models, list)
        self.assertEqual(models, sorted(models))
        self.assertGreater(len(models), 0)
        self.assertIn("Qwen/Qwen3-4B", models)


class AvailabilityInfoTests(unittest.TestCase):
    @patch("dflash.is_available", return_value=True)
    @patch("dflash.is_mlx_available", return_value=True)
    @patch("dflash.is_vllm_available", return_value=False)
    def test_availability_info_structure(self, *_):
        info = availability_info()
        self.assertIn("available", info)
        self.assertIn("mlxAvailable", info)
        self.assertIn("vllmAvailable", info)
        self.assertIn("supportedModels", info)
        self.assertTrue(info["available"])
        self.assertTrue(info["mlxAvailable"])
        self.assertFalse(info["vllmAvailable"])
        self.assertIsInstance(info["supportedModels"], list)


class WarmPoolKeyTests(unittest.TestCase):
    """Verify that speculative_decoding differentiates warm pool keys."""

    def test_warm_pool_key_distinguishes_speculative(self):
        from backend_service.inference import RuntimeController

        key_off = RuntimeController._warm_pool_key(
            model_ref="Qwen/Qwen3-4B",
            runtime_target=None,
            path=None,
            cache_strategy="native",
            cache_bits=0,
            fp16_layers=0,
            fused_attention=False,
            fit_model_in_memory=True,
            context_tokens=8192,
            speculative_decoding=False,
        )
        key_on = RuntimeController._warm_pool_key(
            model_ref="Qwen/Qwen3-4B",
            runtime_target=None,
            path=None,
            cache_strategy="native",
            cache_bits=0,
            fp16_layers=0,
            fused_attention=False,
            fit_model_in_memory=True,
            context_tokens=8192,
            speculative_decoding=True,
        )
        self.assertNotEqual(key_off, key_on)


class LoadModelRequestTests(unittest.TestCase):
    """Verify speculativeDecoding field in Pydantic models."""

    def test_load_model_request_default_false(self):
        from backend_service.models import LoadModelRequest
        req = LoadModelRequest(modelRef="test/model")
        self.assertFalse(req.speculativeDecoding)

    def test_load_model_request_can_be_true(self):
        from backend_service.models import LoadModelRequest
        req = LoadModelRequest(modelRef="test/model", speculativeDecoding=True)
        self.assertTrue(req.speculativeDecoding)

    def test_launch_preferences_request_default_false(self):
        from backend_service.models import LaunchPreferencesRequest
        req = LaunchPreferencesRequest()
        self.assertFalse(req.speculativeDecoding)

    def test_benchmark_run_request_default_false(self):
        from backend_service.models import BenchmarkRunRequest
        req = BenchmarkRunRequest()
        self.assertFalse(req.speculativeDecoding)


class LoadedModelInfoTests(unittest.TestCase):
    """Verify speculativeDecoding and dflashDraftModel in LoadedModelInfo."""

    def test_loaded_model_info_defaults(self):
        from backend_service.inference import LoadedModelInfo
        info = LoadedModelInfo(
            ref="test", name="test", backend="mlx", source="catalog",
            engine="mlx", cacheStrategy="native", cacheBits=0, fp16Layers=0,
            fusedAttention=False, fitModelInMemory=True, contextTokens=8192,
            loadedAt="2025-01-01 00:00:00",
        )
        self.assertFalse(info.speculativeDecoding)
        self.assertIsNone(info.dflashDraftModel)

    def test_loaded_model_info_to_dict_includes_dflash(self):
        from backend_service.inference import LoadedModelInfo
        info = LoadedModelInfo(
            ref="test", name="test", backend="mlx", source="catalog",
            engine="mlx", cacheStrategy="native", cacheBits=0, fp16Layers=0,
            fusedAttention=False, fitModelInMemory=True, contextTokens=8192,
            loadedAt="2025-01-01 00:00:00",
            speculativeDecoding=True,
            dflashDraftModel="z-lab/Qwen3-4B-DFlash-b16",
        )
        d = info.to_dict()
        self.assertTrue(d["speculativeDecoding"])
        self.assertEqual(d["dflashDraftModel"], "z-lab/Qwen3-4B-DFlash-b16")


class GenerationResultTests(unittest.TestCase):
    """Verify dflashAcceptanceRate in GenerationResult."""

    def test_generation_result_default_none(self):
        from backend_service.inference import GenerationResult
        result = GenerationResult(
            text="hello", finishReason="stop", promptTokens=10,
            completionTokens=5, totalTokens=15, tokS=42.0, responseSeconds=1.0,
        )
        self.assertIsNone(result.dflashAcceptanceRate)

    def test_generation_result_to_metrics_includes_acceptance_rate(self):
        from backend_service.inference import GenerationResult
        result = GenerationResult(
            text="hello", finishReason="stop", promptTokens=10,
            completionTokens=5, totalTokens=15, tokS=42.0, responseSeconds=1.0,
            dflashAcceptanceRate=8.5,
        )
        metrics = result.to_metrics()
        self.assertEqual(metrics["dflashAcceptanceRate"], 8.5)

    def test_generation_result_to_metrics_excludes_none_acceptance_rate(self):
        from backend_service.inference import GenerationResult
        result = GenerationResult(
            text="hello", finishReason="stop", promptTokens=10,
            completionTokens=5, totalTokens=15, tokS=42.0, responseSeconds=1.0,
        )
        metrics = result.to_metrics()
        self.assertNotIn("dflashAcceptanceRate", metrics)


if __name__ == "__main__":
    unittest.main()
