"""Mixture-of-experts expert offload in the llama.cpp engine (FU-550).

Reference figures: Qwen3.6-35B-A3B Q4_K_M (ggml-org), RTX 4090, llama.cpp b12649, 8K context, f16 cache, flash
attention on. VRAM above the idle card: no offload 20,048 MiB, ``--n-cpu-moe 20`` 11,606 MiB, ``--cpu-moe``
2,966 MiB. The layout below is that file's real tensor table summary.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import backend_service.app  # noqa: F401 - loads the routes package before routes.cache (circular import)
from backend_service.helpers.fit import variant_fit
from backend_service.helpers.gguf_layout import GgufLayout, gguf_layout
from backend_service.helpers.hardware_class import HardwareClass
from backend_service.helpers.moe_offload import auto_cpu_layers, estimate_split, split
from backend_service.inference import llama_cpp_engine as engine_mod
from tests.test_chat_memory_cuda import write_gguf

MIB = 1024 ** 2
GIB = 1024 ** 3
QWEN_CACHE_8K = 160 * MIB  # 10 full-attention layers x 8,192 tokens, f16 (llama.cpp printed 160 MiB)


def qwen36_layout() -> GgufLayout:
    return GgufLayout(
        total_bytes=20531743232, gpu_bytes=20245678592, host_bytes=286064640, vocab=248320, tied_output=False,
        expert_bytes_by_layer=(452984832,) * 40,
    )


class LayoutExpertTests(unittest.TestCase):
    def test_routed_experts_are_counted_per_block_and_shared_experts_are_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "moe.gguf"
            write_gguf(path, [
                ("token_embd.weight", [8, 1000], 4096),
                ("output.weight", [8, 1000], 4096),
                ("blk.0.ffn_up.weight", [8, 8], 1024),           # a dense leading block
                ("blk.1.ffn_up_exps.weight", [8, 8, 4], 2048),
                ("blk.1.ffn_down_exps.weight", [8, 8, 4], 2048),
                ("blk.1.ffn_gate_exps.weight", [8, 8, 4], 2048),
                ("blk.1.ffn_up_shexp.weight", [8, 8], 512),      # shared expert stays on the GPU
                ("blk.1.ffn_gate_inp.weight", [8, 4], 256),      # router stays on the GPU
                ("blk.2.ffn_gate_up_exps.weight", [8, 16, 4], 4096),
            ])
            layout = gguf_layout(path)
        self.assertEqual(layout.expert_bytes_by_layer, (0, 6144, 4096))
        self.assertEqual(layout.expert_bytes, 10240)

    def test_a_dense_file_has_no_experts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dense.gguf"
            write_gguf(path, [("token_embd.weight", [8, 1000], 4096), ("blk.0.ffn_up.weight", [8, 8], 1024)])
            layout = gguf_layout(path)
        self.assertEqual(layout.expert_bytes_by_layer, ())


class SplitMathTests(unittest.TestCase):
    def test_split_matches_the_measured_vram_within_three_percent(self):
        layout = qwen36_layout()
        for layers, measured_mib in ((0, 20048), (20, 11606), (40, 2966)):
            parts = split(layout, QWEN_CACHE_8K, layers)
            self.assertLess(abs(parts["gpuGb"] * 1024 / measured_mib - 1), 0.035, (layers, parts))
        self.assertAlmostEqual(split(layout, QWEN_CACHE_8K, 40)["ramGb"], 17.1, delta=0.1)
        self.assertEqual(split(layout, QWEN_CACHE_8K, 99)["cpuMoeLayers"], 40)

    def test_auto_picks_the_fewest_blocks_that_fit(self):
        layout = qwen36_layout()
        self.assertEqual(auto_cpu_layers(layout, QWEN_CACHE_8K, 22.0), 0)   # a free 24 GB card: no offload
        self.assertEqual(auto_cpu_layers(layout, QWEN_CACHE_8K, 11.0), 21)  # about a 12 GB card
        self.assertEqual(auto_cpu_layers(layout, QWEN_CACHE_8K, 7.0), 31)   # about an 8 GB card
        self.assertIsNone(auto_cpu_layers(layout, QWEN_CACHE_8K, 2.0))      # even the dense part does not fit
        for budget in (11.0, 7.0):
            layers = auto_cpu_layers(layout, QWEN_CACHE_8K, budget)
            self.assertLessEqual(split(layout, QWEN_CACHE_8K, layers)["gpuGb"], budget)
            self.assertGreater(split(layout, QWEN_CACHE_8K, layers - 1)["gpuGb"], budget)

    def test_dense_models_are_never_offloaded(self):
        dense = GgufLayout(total_bytes=GIB, gpu_bytes=GIB, host_bytes=0, vocab=1000, tied_output=False)
        self.assertIsNone(auto_cpu_layers(dense, 0, 0.5))
        self.assertIsNone(auto_cpu_layers(None, 0, 0.5))

    def test_estimate_for_a_file_not_on_disk(self):
        # A 125B-A10B-class 2-bit file of 67 GB with 48 blocks, 22 GB of card.
        est = estimate_split(need_gb=75.0, weights_gb=67.0, active_fraction=0.08, gpu_budget_gb=22.0, num_layers=48)
        self.assertAlmostEqual(est["ramGb"], 53.0, delta=0.01)
        self.assertAlmostEqual(est["gpuGb"], 22.0, delta=0.01)
        self.assertEqual(est["cpuMoeLayers"], 44)
        self.assertIsNone(
            estimate_split(need_gb=75.0, weights_gb=67.0, active_fraction=0.5, gpu_budget_gb=22.0, num_layers=48)
        )


class FitBadgeTests(unittest.TestCase):
    VARIANT = {
        "id": "ggml-org/Qwen3.6-35B-A3B-GGUF", "name": "Qwen3.6 35B A3B", "repo": "ggml-org/Qwen3.6-35B-A3B-GGUF",
        "sizeGb": 20.5, "paramsB": 35.0, "backend": "llama.cpp", "quantization": "Q4_K_M",
    }

    def test_a_moe_that_does_not_fit_the_card_fits_with_offload(self):
        hw = HardwareClass("cuda", "cuda-12", "", 64.0, 12.0)
        fit = variant_fit(dict(self.VARIANT), "chat", hw, layout=qwen36_layout())
        self.assertEqual(fit["status"], "tight")
        self.assertEqual(fit["offload"]["moeLayers"], 40)
        self.assertLessEqual(fit["needGb"], 12.0 * 0.92)
        self.assertGreater(fit["offload"]["cpuMoeLayers"], 0)
        self.assertIn("expert offload", fit["reason"])
        self.assertNotIn("would run far slower", fit["reason"])

    def test_it_stays_wont_fit_without_the_ram(self):
        hw = HardwareClass("cuda", "cuda-8", "", 16.0, 8.0)  # 16 GB RAM cannot hold ~13 GB of experts at 80 %
        fit = variant_fit(dict(self.VARIANT), "chat", hw, layout=qwen36_layout())
        self.assertEqual(fit["status"], "wont-fit")
        self.assertNotIn("offload", fit)

    def test_a_24_gb_card_fits_it_whole(self):
        hw = HardwareClass("cuda", "cuda-24", "", 64.0, 24.0)
        fit = variant_fit(dict(self.VARIANT), "chat", hw, layout=qwen36_layout())
        self.assertNotIn("offload", fit)

    def test_a_large_moe_row_not_downloaded_is_rated_from_its_name(self):
        row = {
            "id": "x/Big-125B-A10B-GGUF", "name": "Big 125B A10B", "repo": "x/Big-125B-A10B-GGUF", "sizeGb": 67.0,
            "paramsB": 125.0, "backend": "llama.cpp", "format": "GGUF", "quantization": "UD-IQ2_XXS",
        }
        on_64 = variant_fit(dict(row), "chat", HardwareClass("cuda", "cuda-24", "", 63.7, 24.0))
        self.assertEqual(on_64["status"], "tight")
        self.assertLess(on_64["offload"]["ramGb"], 63.7 * 0.8)
        self.assertAlmostEqual(on_64["needGb"], 24.0 * 0.92, delta=0.05)
        on_32 = variant_fit(dict(row), "chat", HardwareClass("cuda", "cuda-24", "", 32.0, 24.0))
        self.assertEqual(on_32["status"], "wont-fit")

    def test_macs_and_dense_models_are_unchanged(self):
        mac = HardwareClass("apple", "apple-16", "", 16.0, 11.0)
        self.assertNotIn("offload", variant_fit(dict(self.VARIANT), "chat", mac))
        dense = {"id": "d", "name": "Dense 70B", "repo": "d/Dense-70B", "sizeGb": 40.0, "paramsB": 70.0,
                 "backend": "llama.cpp", "format": "GGUF", "quantization": "Q4_K_M"}
        fit = variant_fit(dense, "chat", HardwareClass("cuda", "cuda-24", "", 128.0, 24.0))
        self.assertEqual(fit["status"], "wont-fit")
        self.assertNotIn("offload", fit)


HELP_WITH_MOE = "-cmoe, --cpu-moe\n-ncmoe, --n-cpu-moe N\n--load-mode"


class EngineArgumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gguf = str(Path(self.tmp.name) / "moe.gguf")
        write_gguf(Path(self.gguf), [
            ("token_embd.weight", [8, 1000], 4096),
            ("output.weight", [8, 1000], 4096),
            *[(f"blk.{i}.ffn_up_exps.weight", [8, 8, 4], 2048) for i in range(4)],
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def _args(self, requested, *, fit=True, help_text=HELP_WITH_MOE, auto=None, cuda=True):
        with mock.patch.object(engine_mod, "_llama_server_help_text", return_value=help_text.lower()), \
                mock.patch.object(engine_mod, "_discrete_cuda_host", return_value=cuda), \
                mock.patch.object(engine_mod, "_auto_cpu_moe_layers", return_value=auto):
            return engine_mod._expert_offload_args(
                "llama-server", requested=requested, fit_enabled=fit, gguf=self.gguf, context_tokens=8192,
                cache_flags=["--cache-type-k", "f16", "--cache-type-v", "f16"],
            )

    def test_a_count_maps_to_n_cpu_moe(self):
        args, moved, note = self._args(2)
        self.assertEqual(args, ["--n-cpu-moe", "2"])
        self.assertEqual(moved, 2)
        self.assertIn("2 of 4 layers", note)

    def test_every_block_maps_to_cpu_moe(self):
        self.assertEqual(self._args(4)[0], ["--cpu-moe"])
        self.assertEqual(self._args(9)[:2], (["--cpu-moe"], 4))

    def test_zero_keeps_everything_on_the_gpu(self):
        self.assertEqual(self._args(0), ([], None, None))

    def test_auto_with_fit_on_leaves_it_to_llama_cpp(self):
        self.assertEqual(self._args(None, fit=True, auto=3), ([], None, None))

    def test_auto_with_fit_off_uses_the_app_choice_on_a_discrete_card(self):
        self.assertEqual(self._args(None, fit=False, auto=3)[0], ["--n-cpu-moe", "3"])
        self.assertEqual(self._args(None, fit=False, auto=None), ([], None, None))
        self.assertEqual(self._args(None, fit=False, auto=3, cuda=False), ([], None, None))

    def test_a_binary_without_the_option_is_never_given_it(self):
        args, moved, note = self._args(2, help_text="--ctx-size N")
        self.assertEqual((args, moved), ([], None))
        self.assertIn("ignored", note)

    def test_a_dense_file_ignores_the_setting(self):
        dense = Path(self.tmp.name) / "dense.gguf"
        write_gguf(dense, [("token_embd.weight", [8, 1000], 4096), ("blk.0.ffn_up.weight", [8, 8], 1024)])
        with mock.patch.object(engine_mod, "_llama_server_help_text", return_value=HELP_WITH_MOE.lower()):
            args, moved, note = engine_mod._expert_offload_args(
                "llama-server", requested=2, fit_enabled=True, gguf=str(dense), context_tokens=8192, cache_flags=[],
            )
        self.assertEqual((args, moved), ([], None))
        self.assertIn("not a mixture-of-experts", note)

    def test_auto_count_from_free_vram(self):
        with mock.patch("backend_service.helpers.memory_gate.cuda_free_vram_gb", return_value=1.0 + 2.0e-6), \
                mock.patch("backend_service.helpers.moe_offload.auto_cpu_layers", return_value=3) as auto:
            self.assertEqual(engine_mod._auto_cpu_moe_layers(self.gguf, 8192, ["--cache-type-k", "q8_0"]), 3)
        self.assertAlmostEqual(auto.call_args.args[2], 2.0e-6)

    def test_build_command_adds_the_flags_and_reads_the_file_in(self):
        from backend_service.inference.base import BackendCapabilities

        caps = mock.MagicMock(spec=BackendCapabilities)
        caps.llamaServerPath = "llama-server"
        caps.llamaServerTurboPath = None
        engine = engine_mod.LlamaCppEngine(caps)
        with mock.patch.object(engine_mod, "_llama_server_help_text", return_value=HELP_WITH_MOE.lower()), \
                mock.patch.object(engine_mod, "_llama_server_cache_types", return_value=frozenset({"f16"})), \
                mock.patch.object(engine_mod, "nvidia_shared_memory", return_value=False), \
                mock.patch.dict("os.environ", {}, clear=False):
            command, _note, _fell_back, _mmproj = engine._build_command(
                path=self.gguf, runtime_target=None, cache_strategy="native", cache_bits=0, context_tokens=8192,
                fit_enabled=True, is_fallback=False, moe_cpu_layers=2,
            )
        self.assertIn("--n-cpu-moe", command)
        self.assertEqual(command[command.index("--n-cpu-moe") + 1], "2")
        self.assertEqual(command[command.index("--load-mode") + 1], "none")
        self.assertEqual(engine.applied_moe_cpu_layers, 2)


class MemoryCheckTests(unittest.TestCase):
    def _verdict(self, *, free_gpu, free_ram):
        from backend_service.routes.cache import _memory_verdict

        return _memory_verdict(
            weights_gb=20.5, cache_gb=QWEN_CACHE_8K / GIB, context_tokens=8192, free_gb=free_ram, reclaimable_gb=0.0,
            free_gpu_gb=free_gpu, layout=qwen36_layout(), arch_source="gguf",
        )

    def test_a_moe_that_does_not_fit_the_card_loads_with_offload_instead_of_a_warning(self):
        result = self._verdict(free_gpu=12.0, free_ram=40.0)
        self.assertEqual(result["verdict"], "tight")
        self.assertIsNone(result["message"])
        self.assertGreater(result["moeOffload"]["cpuMoeLayers"], 0)
        self.assertLessEqual(result["neededGb"], 10.0)

    def test_without_the_ram_it_still_warns(self):
        result = self._verdict(free_gpu=12.0, free_ram=6.0)
        self.assertEqual(result["verdict"], "may_not_fit")
        self.assertNotIn("moeOffload", result)

    def test_a_card_with_room_needs_nothing(self):
        result = self._verdict(free_gpu=23.0, free_ram=40.0)
        self.assertIn(result["verdict"], {"fits", "tight"})
        self.assertNotIn("moeOffload", result)

    def test_preview_reports_the_layer_count_and_automatic_choice(self):
        from backend_service.routes.cache import _moe_offload_preview

        with mock.patch("backend_service.helpers.memory_gate.cuda_free_vram_gb", return_value=12.0):
            preview = _moe_offload_preview({"cuda": True, "layout": qwen36_layout()}, QWEN_CACHE_8K / GIB)
        self.assertEqual((preview["blocks"], preview["autoCpuLayers"]), (40, 21))
        self.assertIsNone(_moe_offload_preview({"cuda": False, "layout": qwen36_layout()}, 0.1))
        dense = GgufLayout(total_bytes=GIB, gpu_bytes=GIB, host_bytes=0, vocab=1000, tied_output=False)
        self.assertIsNone(_moe_offload_preview({"cuda": True, "layout": dense}, 0.1))

    def test_only_gguf_rows_are_offered_offload(self):
        row = {"id": "n/Lightning-30B-A3B-BF16", "name": "Lightning 30B A3B", "repo": "n/Lightning-30B-A3B-BF16",
               "sizeGb": 65.8, "paramsB": 30.0, "backend": "vllm", "format": "Transformers", "quantization": "BF16"}
        fit = variant_fit(row, "chat", HardwareClass("cuda", "cuda-24", "", 128.0, 24.0))
        self.assertNotIn("offload", fit)


if __name__ == "__main__":
    unittest.main()
