"""KV-cache sizing from a model's real attention layout (``helpers.model_arch``).

The configs below are trimmed copies of real ones (Kolibri 1, Qwen3.8-27B,
Qwen3-1.7B, DeepSeek-V3, Nemotron-H); the expected sizes were checked by hand.
"""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend_service.helpers.model_arch import (
    architecture_from_config,
    architecture_from_gguf_metadata,
    read_kv_architecture,
)

K32 = 32768

KOLIBRI = {
    "model_type": "kolibri1", "num_hidden_layers": 50, "num_attention_heads": 48,
    "num_key_value_heads": 4, "head_dim": 128, "hidden_size": 2560, "sliding_window": 513,
    "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 10,
}
QWEN38 = {
    "model_type": "qwen3_5", "text_config": {
        "model_type": "qwen3_5_text", "num_hidden_layers": 64, "num_attention_heads": 24,
        "num_key_value_heads": 4, "head_dim": 256, "hidden_size": 5120, "full_attention_interval": 4,
    },
}
QWEN3_1_7B = {
    "model_type": "qwen3", "num_hidden_layers": 28, "num_attention_heads": 16,
    "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 2048,
    "sliding_window": 32768, "use_sliding_window": False,
}
DEEPSEEK_V3 = {
    "model_type": "deepseek_v3", "num_hidden_layers": 61, "num_attention_heads": 128,
    "num_key_value_heads": 128, "hidden_size": 7168, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
}
NEMOTRON_H = {
    "model_type": "nemotron_h", "num_hidden_layers": 8, "num_attention_heads": 32,
    "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 4096, "hybrid_override_pattern": "M-M*M-M*",
}


class ConfigTests(unittest.TestCase):
    def test_kolibri_counts_only_full_layers_at_full_context(self):
        arch = architecture_from_config(KOLIBRI)
        self.assertEqual((arch.full_layers, arch.sliding_layers, arch.sliding_window), (10, 40, 513))
        per_layer = 4 * 256 * 2
        self.assertEqual(arch.cache_bytes(K32), per_layer * (10 * K32 + 40 * 513))

    def test_hybrid_interval_counts_one_layer_in_four(self):
        arch = architecture_from_config(QWEN38)
        self.assertEqual(arch.full_layers, 16)
        self.assertEqual(arch.cache_bytes(K32), 16 * 4 * 512 * 2 * K32)

    def test_unused_sliding_window_is_ignored(self):
        arch = architecture_from_config(QWEN3_1_7B)
        self.assertEqual((arch.full_layers, arch.sliding_layers), (28, 0))
        self.assertAlmostEqual(arch.cache_bytes(K32) / 1e9, 3.76, places=2)

    def test_mla_caches_one_latent_per_token(self):
        arch = architecture_from_config(DEEPSEEK_V3)
        self.assertEqual(arch.latent_dim, 576)
        self.assertEqual(arch.cache_bytes(K32), 61 * 576 * 2 * K32)

    def test_nemotron_h_counts_attention_marks(self):
        self.assertEqual(architecture_from_config(NEMOTRON_H).full_layers, 2)

    def test_mamba_has_no_kv_cache(self):
        arch = architecture_from_config({"model_type": "mamba", "num_hidden_layers": 64})
        self.assertEqual(arch.cache_bytes(K32), 0)

    def test_missing_fields_give_none(self):
        self.assertIsNone(architecture_from_config({"model_type": "x"}))


def _gguf_bytes(meta: dict) -> bytes:
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(meta) + 1)
    # A tokenizer-like string array first, which the reader must skip.
    out += s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 2) + s("a") + s("bc")
    for key, value in meta.items():
        out += s(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + s(value)
        elif isinstance(value, list):
            out += struct.pack("<I", 9) + struct.pack("<IQ", 4, len(value))
            out += b"".join(struct.pack("<I", v) for v in value)
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", value)
    return out


class GgufTests(unittest.TestCase):
    def _read(self, meta: dict, name: str = "model-Q4_K_M.gguf"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(_gguf_bytes(meta))
            return read_kv_architecture(path), read_kv_architecture(tmp)

    def test_plain_gqa_model(self):
        arch, from_dir = self._read({
            "general.architecture": "qwen3", "qwen3.block_count": 28, "qwen3.attention.head_count": 16,
            "qwen3.attention.head_count_kv": 8, "qwen3.attention.key_length": 128,
            "qwen3.attention.value_length": 128, "qwen3.embedding_length": 2048,
        })
        self.assertEqual(arch, from_dir)
        self.assertEqual(arch.source, "gguf")
        self.assertAlmostEqual(arch.cache_bytes(K32) / 1e9, 3.76, places=2)

    def test_per_layer_kv_heads_mark_recurrent_layers(self):
        arch, _ = self._read({
            "general.architecture": "nemotron_h", "nemotron_h.block_count": 4,
            "nemotron_h.attention.head_count": 32, "nemotron_h.attention.head_count_kv": [0, 8, 0, 0],
            "nemotron_h.attention.key_length": 128, "nemotron_h.embedding_length": 4096,
        })
        self.assertEqual(arch.full_layers, 1)

    def test_gemma3_window_layout_is_known(self):
        arch = architecture_from_gguf_metadata({
            "general.architecture": "gemma3", "gemma3.block_count": 12, "gemma3.attention.head_count": 8,
            "gemma3.attention.head_count_kv": 4, "gemma3.attention.key_length": 256,
            "gemma3.attention.sliding_window": 1024,
        })
        self.assertEqual((arch.full_layers, arch.sliding_layers), (2, 10))

    def test_not_a_gguf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.gguf"
            path.write_bytes(b"nope")
            self.assertIsNone(read_kv_architecture(path))


class PreviewRouteTests(unittest.TestCase):
    def setUp(self):
        from backend_service.app import create_app
        from backend_service.state import ChaosEngineState
        from tests.test_diagnostics_routes import FakeRuntime
        from tests.test_setup_routes import TEST_API_TOKEN, _fake_system_snapshot

        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.models = base / "models"
        (self.models / "kolibri").mkdir(parents=True)
        (self.models / "kolibri" / "config.json").write_text(json.dumps(KOLIBRI))
        self.outside = base / "outside"
        self.outside.mkdir()
        (self.outside / "config.json").write_text(json.dumps(KOLIBRI))
        state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            library_provider=lambda: [],
            settings_path=base / "settings.json",
            benchmarks_path=base / "benchmarks.json",
            chat_sessions_path=base / "chats.json",
            library_cache_path=base / "library_cache.json",
        )
        state.runtime = FakeRuntime()
        state.settings["modelDirectories"] = [{"id": "m", "path": str(self.models), "enabled": True}]
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})
        from unittest import mock

        self._snapshot = mock.patch(
            "backend_service.routes.cache._build_system_snapshot", return_value={"totalMemoryGb": 64.0},
        )
        self._snapshot.start()

    def tearDown(self):
        self._snapshot.stop()
        self.tmp.cleanup()

    def _preview(self, **params):
        base = {"bits": 0, "num_layers": 80, "num_heads": 64, "num_kv_heads": 8, "hidden_size": 8192,
                "context_tokens": K32, "params_b": 78.1, "strategy": "native"}
        resp = self.client.get("/api/cache/preview", params={**base, **params})
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def test_model_path_sizes_the_cache_from_the_config(self):
        guessed = self._preview()
        real = self._preview(model_path=str(self.models / "kolibri"))
        self.assertEqual(guessed["archSource"], "estimate")
        self.assertEqual(real["archSource"], "config")
        self.assertGreater(guessed["baselineCacheGb"], 9)
        self.assertLess(real["baselineCacheGb"], 0.8)

    def test_paths_outside_model_directories_are_not_read(self):
        self.assertEqual(self._preview(model_path=str(self.outside))["archSource"], "estimate")
        self.assertEqual(
            self._preview(model_path=str(self.models / ".." / "outside"))["archSource"], "estimate",
        )

    def test_compressed_strategy_keeps_its_ratio(self):
        real = self._preview(model_path=str(self.models / "kolibri"), bits=3, fp16_layers=0, strategy="turboquant")
        self.assertGreater(real["compressionRatio"], 4)
        self.assertLess(real["optimizedCacheGb"], real["baselineCacheGb"] / 4)

    def test_contexts_over_256k_are_accepted(self):
        self._preview(context_tokens=1_048_576)


if __name__ == "__main__":
    unittest.main()
