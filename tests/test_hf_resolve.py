"""Tests for the arbitrary-HF-repo resolver (#5).

``resolve_hf_model`` is pure — these exercise classification, GGUF file
selection, context inference, and capability inference with no network.
"""

import unittest

from backend_service.helpers.hf_resolve import resolve_hf_model


def _f(path, size=1_000_000):
    return {"path": path, "sizeBytes": size, "kind": "weight"}


class ResolveHfModelTests(unittest.TestCase):
    def test_gguf_repo_picks_q4_k_m_and_defaults_context(self):
        files = [
            _f("model.Q8_0.gguf", 8_000_000),
            _f("model.Q4_K_M.gguf", 4_000_000),
            _f("model.Q2_K.gguf", 2_000_000),
        ]
        d = resolve_hf_model("bartowski/Some-Model-GGUF", files=files)
        self.assertEqual(d["backend"], "llama.cpp")
        self.assertEqual(d["ggufFile"], "model.Q4_K_M.gguf")
        self.assertEqual(d["contextTokens"], 8192)
        self.assertEqual(d["sizeBytes"], 4_000_000)
        self.assertTrue(any("Context length" in w for w in d["warnings"]))
        self.assertEqual(d["repo"], "bartowski/Some-Model-GGUF")
        self.assertTrue(d["custom"])

    def test_requested_gguf_file_is_honored(self):
        files = [_f("model.Q4_K_M.gguf"), _f("model.Q8_0.gguf")]
        d = resolve_hf_model("x/y-GGUF", files=files, requested_file="model.Q8_0.gguf")
        self.assertEqual(d["ggufFile"], "model.Q8_0.gguf")

    def test_sharded_gguf_picks_first_shard(self):
        files = [
            _f("model-00002-of-00003.gguf"),
            _f("model-00001-of-00003.gguf"),
            _f("model-00003-of-00003.gguf"),
        ]
        d = resolve_hf_model("x/Big-GGUF", files=files)
        self.assertEqual(d["ggufFile"], "model-00001-of-00003.gguf")

    def test_context_read_from_config(self):
        files = [_f("model.Q4_K_M.gguf")]
        d = resolve_hf_model("x/y-GGUF", files=files, config={"max_position_embeddings": 32768})
        self.assertEqual(d["contextTokens"], 32768)
        self.assertFalse(any("Context length" in w for w in d["warnings"]))

    def test_context_clamped_to_ceiling(self):
        files = [_f("model.Q4_K_M.gguf")]
        d = resolve_hf_model("x/y-GGUF", files=files, config={"max_position_embeddings": 1_000_000})
        self.assertEqual(d["contextTokens"], 131072)

    def test_mlx_community_repo_is_mlx_backend(self):
        files = [_f("model.safetensors", 4_000_000)]
        config = {"quantization": {"group_size": 64, "bits": 4}, "max_position_embeddings": 40960}
        d = resolve_hf_model("mlx-community/Qwen3-8B-4bit", files=files, config=config)
        self.assertEqual(d["backend"], "mlx")
        self.assertIsNone(d["ggufFile"])
        self.assertEqual(d["contextTokens"], 40960)

    def test_mlx_detected_from_quantization_stanza_without_namespace(self):
        files = [_f("model.safetensors")]
        config = {"quantization": {"bits": 4}}
        d = resolve_hf_model("someone/custom-mlx-conv", files=files, config=config)
        self.assertEqual(d["backend"], "mlx")

    def test_raw_safetensors_is_vllm_with_warning(self):
        files = [_f("model.safetensors", 16_000_000)]
        d = resolve_hf_model("meta-llama/Some-Raw-Model", files=files, config={"max_position_embeddings": 8192})
        self.assertEqual(d["backend"], "vllm")
        self.assertTrue(any("CUDA" in w or "convert" in w for w in d["warnings"]))

    def test_vision_capability_from_config_and_mmproj(self):
        files = [_f("model.Q4_K_M.gguf"), _f("mmproj-model.gguf")]
        d = resolve_hf_model("x/VL-GGUF", files=files)
        self.assertTrue(d["capabilities"]["vision"])

        d2 = resolve_hf_model(
            "mlx-community/VL-4bit",
            files=[_f("model.safetensors")],
            config={"quantization": {"bits": 4}, "vision_config": {"x": 1}},
        )
        self.assertTrue(d2["capabilities"]["vision"])

    def test_empty_repo_is_unknown_with_warning(self):
        d = resolve_hf_model("x/empty", files=[])
        self.assertEqual(d["backend"], "unknown")
        self.assertTrue(d["warnings"])


if __name__ == "__main__":
    unittest.main()
