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
    is_ddtree_available,
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
            "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
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
        result = get_draft_model("mlx-community/Qwen3.5-9B-bf16")
        self.assertEqual(result, "z-lab/Qwen3.5-9B-DFlash")

    def test_lmstudio_community_prefix_fuzzy_match(self):
        """lmstudio-community GGUF repos should match via prefix stripping."""
        result = get_draft_model("lmstudio-community/Qwen3-8B-GGUF")
        self.assertEqual(result, "z-lab/Qwen3-8B-DFlash-b16")

    def test_lmstudio_community_qwen35_match(self):
        result = get_draft_model("lmstudio-community/Qwen3.5-35B-A3B-GGUF")
        self.assertEqual(result, "z-lab/Qwen3.5-35B-A3B-DFlash")

    def test_thebloke_prefix_fuzzy_match(self):
        result = get_draft_model("TheBloke/Qwen3-8B-GGUF")
        self.assertEqual(result, "z-lab/Qwen3-8B-DFlash-b16")

    def test_bartowski_prefix_fuzzy_match(self):
        result = get_draft_model("bartowski/Qwen3-4B-GGUF")
        self.assertEqual(result, "z-lab/Qwen3-4B-DFlash-b16")

    def test_stacked_suffixes_stripped(self):
        """Multiple quant suffixes like -GGUF-q4_k_m should all be stripped."""
        result = get_draft_model("lmstudio-community/Qwen3-8B-GGUF")
        self.assertEqual(result, "z-lab/Qwen3-8B-DFlash-b16")

    def test_unknown_model_returns_none(self):
        self.assertIsNone(get_draft_model("some-org/UnknownModel-7B"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(get_draft_model(""))

    def test_all_map_entries_return_values(self):
        for target, expected_draft in DRAFT_MODEL_MAP.items():
            result = get_draft_model(target)
            self.assertEqual(result, expected_draft, f"Failed for target: {target}")

    # ------------------------------------------------------------------
    # 2026-05-10: dflash-mlx 0.1.5 added Gemma4 backend (commit 05cc456)
    # and z-lab published draft checkpoints for Gemma-4, Qwen3.5-122B,
    # MiniMax M2.5/M2.7, and Kimi K2.6. Pin those mappings so future
    # regressions surface here rather than at first-use.
    # ------------------------------------------------------------------

    def test_gemma4_31b_drafter(self):
        self.assertEqual(
            get_draft_model("google/gemma-4-31B-it"),
            "z-lab/gemma-4-31B-it-DFlash",
        )

    def test_gemma4_26b_a4b_drafter(self):
        self.assertEqual(
            get_draft_model("google/gemma-4-26B-A4B-it"),
            "z-lab/gemma-4-26B-A4B-it-DFlash",
        )

    def test_gemma4_mlx_community_alias(self):
        # Apple Silicon users pull the mlx-community quants; the alias
        # map has to canonicalise back to the google/ key so the drafter
        # is still reachable.
        for variant in (
            "mlx-community/gemma-4-31B-it-bf16",
            "mlx-community/gemma-4-31B-it-4bit",
            "mlx-community/gemma-4-31B-it-8bit",
        ):
            self.assertEqual(
                get_draft_model(variant),
                "z-lab/gemma-4-31B-it-DFlash",
                f"alias mismatch for {variant}",
            )

    def test_qwen35_122b_a10b_drafter(self):
        self.assertEqual(
            get_draft_model("Qwen/Qwen3.5-122B-A10B"),
            "z-lab/Qwen3.5-122B-A10B-DFlash",
        )

    def test_minimax_m25_drafter(self):
        self.assertEqual(
            get_draft_model("MiniMaxAI/MiniMax-M2.5"),
            "z-lab/MiniMax-M2.5-DFlash",
        )

    def test_minimax_m27_drafter(self):
        self.assertEqual(
            get_draft_model("MiniMaxAI/MiniMax-M2.7"),
            "z-lab/MiniMax-M2.7-DFlash",
        )

    def test_kimi_k26_drafter(self):
        self.assertEqual(
            get_draft_model("moonshotai/Kimi-K2.6"),
            "z-lab/Kimi-K2.6-DFlash",
        )

    def test_kimi_k26_mlx_community_alias(self):
        self.assertEqual(
            get_draft_model("mlx-community/Kimi-K2.6-4bit"),
            "z-lab/Kimi-K2.6-DFlash",
        )

    def test_coder_next_mlx_4bit_alias_resolves(self):
        """FU-041: ``lmstudio-community/Qwen3-Coder-Next-MLX-4bit`` is
        the Qwen3-Next MoE coder (qwen3_next architecture, 512 experts,
        hidden_size=2048). Confirmed by inspecting the local config.json
        — it is NOT the same checkpoint as ``mlx-community/Qwen3.6-27B-4bit``
        (which is the dense Qwen3.6-27B). The alias routes to the
        Coder-Next drafter; the dense 27B-4bit has no drafter."""
        self.assertEqual(
            get_draft_model("lmstudio-community/Qwen3-Coder-Next-MLX-4bit"),
            "z-lab/Qwen3-Coder-Next-DFlash",
        )

    def test_qwen36_27b_4bit_is_dense_not_coder_next(self):
        """Regression test for the FU-038 bug we reverted in FU-041:
        ``mlx-community/Qwen3.6-27B-4bit`` is the DENSE 27B Coder
        (qwen3 architecture, hidden_size=5120). It must NOT alias to
        the Qwen3-Next MoE drafter."""
        self.assertNotEqual(
            get_draft_model("mlx-community/Qwen3.6-27B-4bit"),
            "z-lab/Qwen3-Coder-Next-DFlash",
        )

    def test_qwen36_27b_dense_drafter(self):
        """The dense 27B got its own drafter (2026-04); the FU-041 note
        that it "has no drafter" is superseded."""
        for ref in ("Qwen/Qwen3.6-27B", "mlx-community/Qwen3.6-27B-4bit"):
            self.assertEqual(get_draft_model(ref), "z-lab/Qwen3.6-27B-DFlash", ref)

    def test_gpt_oss_uses_openai_org_and_lowercase_drafters(self):
        self.assertEqual(get_draft_model("openai/gpt-oss-20b"), "z-lab/gpt-oss-20b-DFlash")
        self.assertEqual(get_draft_model("mlx-community/gpt-oss-120B-4bit"), "z-lab/gpt-oss-120b-DFlash")

    def test_gemma4_12b_drafter(self):
        self.assertEqual(get_draft_model("google/gemma-4-12B-it"), "z-lab/gemma4-12B-it-DFlash")

    def test_every_drafter_is_a_z_lab_repo(self):
        from dflash import DRAFT_MODEL_MAP, _ALIASES

        for target, draft in DRAFT_MODEL_MAP.items():
            self.assertTrue(draft.startswith("z-lab/"), target)
        # Every alias must land on a real map key, never dangle.
        for alias, canonical in _ALIASES.items():
            self.assertIn(canonical, DRAFT_MODEL_MAP, alias)


class ModelResolutionTests(unittest.TestCase):
    def test_resolve_dflash_target_prefers_canonical_repo(self):
        from backend_service.model_resolution import resolve_dflash_target_ref

        resolved = resolve_dflash_target_ref(
            canonical_repo="mlx-community/Qwen3.5-9B-4bit",
            path="/tmp/models--someone--else/snapshots/1234",
            model_ref="Qwen 3.5 local copy",
        )

        self.assertEqual(resolved, "mlx-community/Qwen3.5-9B-4bit")

    def test_infer_hf_repo_from_local_path_reads_snapshot_layout(self):
        from backend_service.model_resolution import infer_hf_repo_from_local_path

        resolved = infer_hf_repo_from_local_path(
            "/Users/test/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/abcd"
        )

        self.assertEqual(resolved, "mlx-community/Qwen3.5-9B-4bit")


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

    # FU-091: the CUDA lane is vLLM's built-in DFlash method, gated on
    # the installed vLLM version — not the PyPI ``dflash`` package.
    def test_vllm_available_at_min_version(self):
        for raw in ("0.28.0", "0.30.0", "0.28.1.dev12+gabc", "1.0.0"):
            with patch("dflash.vllm_version", return_value=raw):
                self.assertTrue(is_vllm_available(), raw)

    def test_vllm_unavailable_below_min_or_missing(self):
        for raw in ("0.27.9", "0.24.0", None, "garbage"):
            with patch("dflash.vllm_version", return_value=raw):
                self.assertFalse(is_vllm_available(), raw)

    def test_vllm_probe_ignores_pypi_dflash_package(self):
        # Even if something named ``dflash.model`` resolved, only vLLM's
        # version decides — the old probe keyed on that submodule.
        with patch("dflash.importlib.util.find_spec", return_value=SimpleNamespace(name="dflash.model")), \
             patch("dflash.vllm_version", return_value=None):
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


class DDTreeAvailabilityProbeTests(unittest.TestCase):
    """FU-071: the probe must key off the symbols our code actually imports
    from ``dflash_mlx.runtime`` (new ``target_ops`` adapter API), not the
    pre-0.1.5 ``target_forward_with_hidden_states`` that was renamed away."""

    def _probe_with_source(self, source: str) -> bool:
        spec = SimpleNamespace(origin="/fake/dflash_mlx/runtime.py")
        with patch("dflash.importlib.util.find_spec", return_value=spec), patch(
            "dflash.Path"
        ) as mock_path:
            mock_path.return_value.read_text.return_value = source
            return is_ddtree_available()

    def test_modern_runtime_with_target_ops_api_is_available(self):
        # Mirrors the installed dflash-mlx 0.1.5+ surface.
        source = "def resolve_target_ops(): ...\nload_draft_bundle\nstream_dflash_generate\n"
        self.assertTrue(self._probe_with_source(source))

    def test_legacy_only_symbol_is_not_enough(self):
        # The obsolete pre-0.1.5 symbol on its own must NOT satisfy the probe.
        source = "target_forward_with_hidden_states\nload_target_bundle\n"
        self.assertFalse(self._probe_with_source(source))

    def test_missing_resolve_target_ops_is_unavailable(self):
        source = "load_draft_bundle\nstream_dflash_generate\n"
        self.assertFalse(self._probe_with_source(source))

    def test_unimportable_runtime_is_unavailable(self):
        with patch("dflash.importlib.util.find_spec", return_value=None):
            self.assertFalse(is_ddtree_available())


if __name__ == "__main__":
    unittest.main()


class DFlash2DraftMapTests(unittest.TestCase):
    """FU-089: DFlash 2 drafters are vLLM-only (pinned dflash-mlx predates them)."""

    def test_vllm_opt_in_resolves_dflash2_draft(self):
        from dflash import get_draft_model

        self.assertEqual(
            get_draft_model("Qwen/Qwen3.8-27B", allow_dflash2=True),
            "z-lab/Qwen3.8-27B-DFlash2",
        )

    def test_mlx_default_never_gets_a_dflash2_draft(self):
        from dflash import get_draft_model

        self.assertIsNone(get_draft_model("Qwen/Qwen3.8-27B"))
        self.assertIsNone(get_draft_model("mlx-community/Qwen3.8-27B-4bit"))

    def test_availability_lists_dflash2_targets_only_with_vllm(self):
        import dflash

        with patch("dflash.is_vllm_available", return_value=False), \
             patch("dflash.is_mlx_available", return_value=True):
            self.assertNotIn("Qwen/Qwen3.8-27B", dflash.availability_info()["supportedModels"])
        with patch("dflash.is_vllm_available", return_value=True), \
             patch("dflash.is_mlx_available", return_value=False):
            self.assertIn("Qwen/Qwen3.8-27B", dflash.availability_info()["supportedModels"])
