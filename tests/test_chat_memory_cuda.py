"""Chat memory and speed on a discrete NVIDIA GPU (RTX 4090 audit, 2026-11-11).

The figures were measured with llama-server b11981 (CUDA) and the TurboQuant fork, the app's own launch flags:
Nemotron 3.5 Bolt E4B Q4_K_M put 3536 MiB of embeddings in system memory and 3023 MiB on the card, and its f16
cache was 1040 MiB at 64K tokens; Qwen3-0.6B Q8_0 used 158 / 604 MiB. The Apple Silicon estimates are checked
unchanged at the end.
"""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers.cache import compute_cache_preview, estimate_decode_tok_s
from backend_service.helpers.chat_memory import llama_gpu_need, turbo_llama_bits
from backend_service.helpers.fit import variant_fit
from backend_service.helpers.gguf_layout import gguf_layout
from backend_service.helpers.hardware_class import HardwareClass, hardware_class
from backend_service.helpers.model_arch import KvArchitecture, architecture_from_config, architecture_from_gguf_metadata

MIB = 1024 ** 2
GIB = 1024 ** 3


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path: Path, tensors: list[tuple[str, list[int], int]], meta: dict | None = None) -> None:
    """A GGUF with the given ``(name, dims, data bytes)`` tensors; data is zeros, aligned to 32."""
    meta = {"general.architecture": "llama", **(meta or {})}
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), len(meta) + 1)
    out += _s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 2) + _s("a") + _s("bc")
    for key, value in meta.items():
        out += _s(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + _s(value)
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", value)
    offset = 0
    for name, dims, size in tensors:
        out += _s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", 0, offset)
        offset += (size + 31) // 32 * 32
    out += b"\0" * ((32 - len(out) % 32) % 32)
    for _name, _dims, size in tensors:
        out += b"\0" * ((size + 31) // 32 * 32)
    path.write_bytes(out)


def bolt_like(path: Path) -> None:
    """Bolt E4B's split in miniature (1/1024 scale): tied output, a large per-layer embedding table."""
    write_gguf(path, [
        ("token_embd.weight", [2560, 262144], 681 * 1024),
        ("per_layer_token_embd.weight", [10752, 262144], 2855 * 1024),
        ("blk.0.attn_q.weight", [2560, 4096], 2342 * 1024),
    ])


class GgufLayoutTests(unittest.TestCase):
    def test_bolt_keeps_its_embeddings_in_system_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bolt.gguf"
            bolt_like(path)
            layout = gguf_layout(path)
        self.assertTrue(layout.tied_output)
        self.assertEqual(layout.host_bytes, 3536 * 1024)
        # file - embeddings + a GPU copy of token_embd for the tied output head: 3023 (measured 3023.05 MiB x 1/1024)
        self.assertEqual(layout.gpu_bytes, 3023 * 1024)
        self.assertEqual(layout.vocab, 262144)
        self.assertAlmostEqual(layout.compute_buffer_bytes() / MIB, 563, delta=1)  # measured 582

    def test_untied_model_puts_everything_but_the_input_table_on_the_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen.gguf"
            write_gguf(path, [
                ("token_embd.weight", [1024, 151936], 158 * 1024),
                ("output.weight", [1024, 151936], 158 * 1024),
                ("blk.0.ffn_up.weight", [1024, 3072], 446 * 1024),
            ])
            layout = gguf_layout(path)
        self.assertFalse(layout.tied_output)
        self.assertEqual((layout.host_bytes, layout.gpu_bytes), (158 * 1024, 604 * 1024))

    def test_split_files_are_summed(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "m-00001-of-00002.gguf"
            write_gguf(first, [("token_embd.weight", [8, 1000], 4096), ("output.weight", [8, 1000], 4096)])
            write_gguf(Path(tmp) / "m-00002-of-00002.gguf", [("blk.0.attn_q.weight", [8, 8], 8192)])
            layout = gguf_layout(first)
        self.assertEqual((layout.total_bytes, layout.gpu_bytes, layout.host_bytes), (16384, 12288, 4096))

    def test_unreadable_files_give_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.gguf"
            bad.write_bytes(b"nope")
            self.assertIsNone(gguf_layout(bad))
            self.assertIsNone(gguf_layout(Path(tmp) / "missing.gguf"))
            self.assertIsNone(gguf_layout(tmp))


BOLT_META = {
    "general.architecture": "gemma4", "gemma4.block_count": 42, "gemma4.attention.head_count": 8,
    "gemma4.attention.head_count_kv": 2, "gemma4.attention.key_length": 512, "gemma4.attention.value_length": 512,
    "gemma4.attention.key_length_swa": 256, "gemma4.attention.value_length_swa": 256,
    "gemma4.attention.sliding_window": 512, "gemma4.attention.shared_kv_layers": 18,
    "gemma4.embedding_length": 2560,
    "gemma4.attention.sliding_window_pattern": ([True] * 5 + [False]) * 7,
}


class SharedKvLayoutTests(unittest.TestCase):
    def test_bolt_caches_four_global_and_twenty_sliding_layers(self):
        arch = architecture_from_gguf_metadata(BOLT_META)
        self.assertEqual((arch.full_layers, arch.sliding_layers), (4, 20))
        self.assertEqual((arch.sliding_key_dim, arch.sliding_value_dim), (256, 256))
        # llama.cpp: 1040 MiB of f16 cache at 66K tokens (W4); the old reading (7 + 35 layers, 512-wide) said 1.82 GB.
        self.assertAlmostEqual(arch.cache_bytes(65536) / MIB, 1044, delta=10)

    def test_config_shared_layers_are_left_out(self):
        cfg = {
            "model_type": "gemma3n_text", "num_hidden_layers": 6, "num_attention_heads": 8, "num_key_value_heads": 2,
            "head_dim": 256, "hidden_size": 2048, "sliding_window": 512, "num_kv_shared_layers": 2,
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention"] * 2,
        }
        arch = architecture_from_config(cfg)
        self.assertEqual((arch.full_layers, arch.sliding_layers), (1, 3))

    def test_models_without_shared_layers_are_unchanged(self):
        arch = architecture_from_gguf_metadata({
            "general.architecture": "qwen3", "qwen3.block_count": 28, "qwen3.attention.head_count": 16,
            "qwen3.attention.head_count_kv": 8, "qwen3.attention.key_length": 128, "qwen3.embedding_length": 1024,
        })
        self.assertEqual((arch.full_layers, arch.sliding_layers, arch.sliding_key_dim), (28, 0, 0))
        self.assertEqual(arch.cache_bytes(32768) / MIB, 3584)  # llama.cpp: 3584 MiB at 32K


QWEN06 = KvArchitecture(28, 28, 0, 0, 8, 128, 128, source="gguf")


class TurboQuantLlamaLaneTests(unittest.TestCase):
    def _preview(self, bits: int, **extra):
        with mock.patch("backend_service.helpers.bandwidth.memory_bandwidth", return_value=(1008, "RTX 4090", True)):
            return compute_cache_preview(
                bits=bits, fp16_layers=4, context_tokens=32768, params_b=0.6, strategy="turboquant",
                arch=QWEN06, weights_gb=0.75, **extra,
            )

    def test_every_layer_is_compressed_to_the_measured_bits(self):
        # Measured KV buffers at 32K: turbo3 700 MiB, turbo4 952 MiB (f16 3584).
        for bits, measured_mib in ((3, 700), (4, 952)):
            body = self._preview(bits, lane="llama.cpp", cuda=True)
            self.assertAlmostEqual(body["optimizedCacheGb"] * 1024, measured_mib, delta=measured_mib * 0.02)
        self.assertEqual(turbo_llama_bits(1), turbo_llama_bits(2))  # 1 bit runs as turbo2

    def test_cuda_speed_is_the_measured_share_of_f16(self):
        body = self._preview(3, lane="llama.cpp", cuda=True)
        self.assertEqual(body["speedRatio"], 0.85)  # measured 0.82 to 0.87
        self.assertNotIn("M4 Max", body["summary"])

    def test_the_mlx_lane_is_unchanged(self):
        body = self._preview(3)
        self.assertEqual(body["speedRatio"], 0.18)
        self.assertAlmostEqual(body["optimizedCacheGb"], 1.37, places=2)


class CudaDecodeSpeedTests(unittest.TestCase):
    """Measured on an RTX 4090 (1008 GB/s), app launch flags, 256 tokens after a short prompt."""

    POINTS = [
        # GPU weights (bytes), layers, KV read per token at 2K, measured tok/s
        (604.15 * MIB, 28, 2048 * 114688, 565),   # Qwen3-0.6B Q8_0
        (3023.05 * MIB, 42, 2048 * 16384, 174),   # Nemotron 3.5 Bolt E4B Q4_K_M
    ]

    def test_within_a_fifth_of_measured(self):
        for weights, layers, kv, measured in self.POINTS:
            est = estimate_decode_tok_s(
                weights_gb=weights / 1e9, active_fraction=1.0, kv_bytes_per_token_read=kv,
                bandwidth_gbps=1008, cuda_layers=layers,
            )
            self.assertLess(abs(est / measured - 1), 0.2, (measured, est))

    def test_the_mac_fit_is_unchanged_without_layers(self):
        est = estimate_decode_tok_s(weights_gb=0.3, active_fraction=1.0, kv_bytes_per_token_read=0, bandwidth_gbps=546)
        self.assertAlmostEqual(est, 1 / (0.3e9 / (0.84 * 546e9) + 0.002), places=6)


def _bolt_layout():
    from backend_service.helpers.gguf_layout import GgufLayout

    return GgufLayout(total_bytes=int(5879 * MIB), gpu_bytes=int(3023 * MIB), host_bytes=int(3536 * MIB),
                      vocab=262144, tied_output=True)


class CudaChatFitTableTests(unittest.TestCase):
    """Bolt E4B at 8K: needs about 3.9 GB of VRAM (measured 3.7 to 3.9 at 4K to 16K), not the 6.5 GB of
    'weights + 10 % + cache'. Cards from 8 GB to 96 GB, and a 128 GB machine with shared memory."""

    VARIANT = {"id": "nvidia/Bolt", "sizeGb": 6.9, "paramsB": 8.0, "backend": "llama.cpp", "quantization": "Q4_K_M"}

    def test_vram_table(self):
        arch = architecture_from_gguf_metadata(BOLT_META)
        expected = {4: "wont-fit", 5: "tight", 6: "fits", 8: "fits", 12: "fits", 16: "fits", 24: "fits", 96: "fits"}
        for gb, status in expected.items():
            hw = HardwareClass("cuda", f"cuda-{gb}", "", 64.0, float(gb))
            fit = variant_fit(dict(self.VARIANT), "chat", hw, arch=arch, layout=_bolt_layout())
            self.assertEqual(fit["status"], status, (gb, fit))
            self.assertAlmostEqual(fit["needGb"], 3.9, delta=0.15)
            self.assertIn("system RAM", fit["reason"])

    def test_without_the_file_or_on_shared_memory_the_old_figure_stands(self):
        arch = architecture_from_gguf_metadata(BOLT_META)
        spark = hardware_class({"totalMemoryGb": 128.0, "gpuVramTotalGb": 128.0, "gpuUnifiedMemory": True})
        on_spark = variant_fit(dict(self.VARIANT), "chat", spark, arch=arch, layout=_bolt_layout())
        hw = HardwareClass("cuda", "cuda-8", "", 64.0, 8.0)
        no_file = variant_fit(dict(self.VARIANT), "chat", hw, arch=arch)
        for fit in (on_spark, no_file):
            self.assertAlmostEqual(fit["needGb"], 6.9 * 1.1 + arch.cache_bytes(8192) / GIB, delta=0.06)

    def test_turbo_fork_counts_its_extra_buffer(self):
        plain = llama_gpu_need(_bolt_layout(), 0)["needGb"]
        turbo = llama_gpu_need(_bolt_layout(), 0, turbo=True)["needGb"]
        self.assertAlmostEqual(turbo - plain, 262144 * 512 * 4 * 1.1 / GIB, places=3)


class UnifiedMemoryHardwareTests(unittest.TestCase):
    def test_shared_memory_keeps_room_for_the_system(self):
        for total, usable in ((128.0, 115.2), (64.0, 56.0), (32.0, 24.0)):
            hw = hardware_class({"totalMemoryGb": total, "gpuVramTotalGb": total, "gpuUnifiedMemory": True})
            self.assertEqual((hw.kind, hw.unified, hw.usable_gb), ("cuda", True, usable))

    def test_discrete_cards_use_their_whole_vram(self):
        for vram in (8.0, 12.0, 16.0, 24.0, 32.0, 96.0):
            hw = hardware_class({"totalMemoryGb": 64.0, "gpuVramTotalGb": vram})
            self.assertEqual((hw.usable_gb, hw.unified), (vram, False))



class TurboBinaryLookupTests(unittest.TestCase):
    """The TurboQuant lane needs ``llama-server-turbo``; on Windows the file is ``llama-server-turbo.exe``."""

    def _lookup(self, platform: str, filename: str):
        from backend_service.inference import binaries

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / filename
            target.write_bytes(b"")
            target.chmod(0o755)
            with mock.patch.object(binaries, "_LLAMA_FALLBACK_DIRS", [tmp]),                     mock.patch.object(binaries.sys, "platform", platform),                     mock.patch.object(binaries.shutil, "which", return_value=None):
                found = binaries._which_with_fallbacks("llama-server-turbo")
            return Path(found).name if found else None

    def test_windows_finds_the_exe(self):
        self.assertEqual(self._lookup("win32", "llama-server-turbo.exe"), "llama-server-turbo.exe")

    def test_other_systems_do_not_add_a_suffix(self):
        self.assertIsNone(self._lookup("linux", "llama-server-turbo.exe"))


class MimallocPurgeTests(unittest.TestCase):
    """After an Unload the Windows backend held 16 GB (FLUX.2 Klein) to 26 GB of system RAM until these were set."""

    def _env_after_import(self, extra: dict[str, str]) -> str:
        import os
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_")}
        env.update(extra)
        code = "import os, backend_service; print(os.environ.get('MIMALLOC_PURGE_DELAY'), os.environ.get('MIMALLOC_RESET_DELAY'))"
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout.strip()

    def test_set_on_windows_only_and_a_user_value_wins(self):
        import sys

        expected = "0 0" if sys.platform == "win32" else "None None"
        self.assertEqual(self._env_after_import({}), expected)
        self.assertTrue(self._env_after_import({"MIMALLOC_PURGE_DELAY": "250"}).startswith("250"))

if __name__ == "__main__":
    unittest.main()
