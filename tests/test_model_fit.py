"""Hardware fit for the chat, image and video catalogs (``helpers/fit.py``, ``helpers/hardware_class.py``).

Covers the 70 % / 92 % thresholds at their boundaries, ``unknown`` for variants that cannot run on
this kind of machine, the best-fit tie-breaks, measured versus estimated numbers, the Metal limit
read, and the payload shape of the three real catalogs on a 16 GB Mac, a 64 GB Mac and a 24 GB-VRAM
PC. Nothing loads a model: the Metal read is patched and Hub lookups are never made.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from backend_service.catalog import IMAGE_MODEL_FAMILIES, MODEL_FAMILIES, VIDEO_MODEL_FAMILIES
from backend_service.helpers import fit as fit_mod
from backend_service.helpers import hardware_class as hc
from backend_service.helpers.fit import (
    annotate_chat_families,
    best_fit_variant_id,
    classify_share,
    estimate_chat_tok_s,
    local_kv_architecture,
    platform_gate,
    variant_fit,
)
from backend_service.helpers.hardware_class import HardwareClass, hardware_class
from backend_service.helpers.model_arch import KvArchitecture

APPLE_64 = HardwareClass("apple", "apple-64", "Apple Silicon, 64 GB", 64.0, 51.8)
APPLE_16 = HardwareClass("apple", "apple-16", "Apple Silicon, 16 GB", 16.0, 10.7)
CUDA_24 = HardwareClass("cuda", "cuda-24", "NVIDIA GPU, 24 GB VRAM", 32.0, 24.0)
CPU_32 = HardwareClass("cpu", "cpu", "CPU only, 32 GB RAM", 32.0, 19.2)

SYSTEM_MAC_16 = {"platform": "Darwin", "arch": "arm64", "totalMemoryGb": 16.0}
SYSTEM_MAC_64 = {"platform": "Darwin", "arch": "arm64", "totalMemoryGb": 64.0}
SYSTEM_PC_24GB_VRAM = {"platform": "Windows", "arch": "AMD64", "totalMemoryGb": 32.0, "gpuVramTotalGb": 24.0}

FIT_KEYS = {"status", "needGb", "usableGb", "headroomPct", "reason", "basis"}
STATUSES = {"fits", "tight", "wont-fit", "unknown"}


def chat_variant(**overrides):
    variant = {
        "id": "org/model-8b",
        "name": "Model 8B",
        "repo": "org/model-8b",
        "paramsB": 8.0,
        "sizeGb": 10.0,
        "quantization": "4-bit",
        "backend": "llama.cpp",
        "releaseDate": "2026-01",
    }
    variant.update(overrides)
    return variant


def media_variant(**overrides):
    variant = {
        "id": "org/image-model",
        "repo": "org/image-model",
        "runtime": "diffusers Flux2Pipeline",
        "styleTags": [],
        "sizeGb": 10.0,
        "releaseDate": "2026-01",
        "estimatedGenerationSeconds": 20.0,
    }
    variant.update(overrides)
    return variant


class ThresholdTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(classify_share(70.0, 100.0), "fits")
        self.assertEqual(classify_share(70.1, 100.0), "tight")
        self.assertEqual(classify_share(92.0, 100.0), "tight")
        self.assertEqual(classify_share(92.1, 100.0), "wont-fit")

    def test_boundaries_with_awkward_floats(self):
        # 51.8 * 0.7 and 51.8 * 0.92 are not exact in binary floating point.
        self.assertEqual(classify_share(51.8 * 0.7, 51.8), "fits")
        self.assertEqual(classify_share(51.8 * 0.92, 51.8), "tight")

    def test_variant_fit_applies_the_thresholds_to_the_estimate(self):
        # media footprint 36.26 GB on 51.8 GB usable is exactly 70 %
        at_limit = variant_fit(media_variant(runtimeFootprintMpsGb=36.26), "image", APPLE_64)
        self.assertEqual(at_limit["status"], "fits")
        over = variant_fit(media_variant(runtimeFootprintMpsGb=36.4), "image", APPLE_64)
        self.assertEqual(over["status"], "tight")
        wont = variant_fit(media_variant(runtimeFootprintMpsGb=48.0), "image", APPLE_64)
        self.assertEqual(wont["status"], "wont-fit")
        self.assertEqual(wont["headroomPct"], 7)  # over the 92 % line but still inside the memory
        beyond = variant_fit(media_variant(runtimeFootprintMpsGb=60.0), "image", APPLE_64)
        self.assertLess(beyond["headroomPct"], 0)

    def test_fit_dict_shape(self):
        result = variant_fit(media_variant(runtimeFootprintMpsGb=25.9), "image", APPLE_64)
        self.assertEqual(set(result), FIT_KEYS)
        self.assertEqual(result["needGb"], 25.9)
        self.assertEqual(result["usableGb"], 51.8)
        self.assertEqual(result["headroomPct"], 50)  # (51.8 - 25.9) / 51.8


class ChatNeedTests(unittest.TestCase):
    def test_weights_runtime_share_and_cache_from_the_parameter_table(self):
        result = variant_fit(chat_variant(), "chat", APPLE_64)
        # 10 GB weights + 10 % runtime + 8K cache of an 8B-class model (36 layers x 8 KV heads x 128)
        self.assertAlmostEqual(result["needGb"], 12.125, delta=0.06)
        self.assertEqual(result["status"], "fits")
        self.assertEqual(result["basis"], "estimated")
        self.assertIn("Weights 10.0 GB", result["reason"])
        self.assertIn("8K context", result["reason"])

    def test_real_layout_replaces_the_table(self):
        arch = KvArchitecture(32, 32, 0, 0, 8, 128, 128)  # 128 KiB per token, 1 GiB at 8K
        result = variant_fit(chat_variant(), "chat", APPLE_64, arch=arch)
        self.assertAlmostEqual(result["needGb"], 12.0, delta=0.06)

    def test_context_is_capped_by_the_model_window(self):
        arch = KvArchitecture(32, 32, 0, 0, 8, 128, 128)
        short = variant_fit(chat_variant(maxContext=2048), "chat", APPLE_64, arch=arch)
        self.assertAlmostEqual(short["needGb"], 11.25, delta=0.06)
        self.assertIn("2K context", short["reason"])

    def test_row_with_its_own_memory_figure_replaces_weights_plus_runtime(self):
        result = variant_fit(chat_variant(), "chat", APPLE_64, explicit_memory_gb=20.0)
        self.assertAlmostEqual(result["needGb"], 21.125, delta=0.06)
        self.assertIn("catalog estimate", result["reason"])

    def test_no_file_size_falls_back_to_parameters_and_quantisation(self):
        variant = chat_variant(sizeGb=None, paramsB=8.0, quantization="BF16")
        result = variant_fit(variant, "chat", APPLE_64)
        self.assertAlmostEqual(result["needGb"], 8 * 2.0 * 1.1 + 1.125, delta=0.06)

    def test_no_size_at_all_is_unknown(self):
        result = variant_fit(chat_variant(sizeGb=None, paramsB=None), "chat", APPLE_64)
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["needGb"])

    def test_cuda_overflow_says_it_is_about_gpu_memory(self):
        result = variant_fit(chat_variant(sizeGb=30.0, paramsB=30.0), "chat", CUDA_24)
        self.assertEqual(result["status"], "wont-fit")
        self.assertIn("GPU memory", result["reason"])


class PlatformGateTests(unittest.TestCase):
    def test_cuda_only_variant_on_a_mac_is_unknown_not_wont_fit(self):
        result = variant_fit(chat_variant(backend="vllm", sizeGb=400.0), "chat", APPLE_64)
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["needGb"])
        self.assertIsNone(result["headroomPct"])
        self.assertIn("NVIDIA", result["reason"])
        self.assertIn("Mac", result["reason"])

    def test_mlx_variant_off_apple_silicon_is_unknown(self):
        for hardware in (CUDA_24, CPU_32):
            result = variant_fit(chat_variant(backend="mlx"), "chat", hardware)
            self.assertEqual(result["status"], "unknown", hardware.kind)
            self.assertIn("Apple Silicon", result["reason"])

    def test_every_chat_backend_that_needs_apple_silicon_is_gated(self):
        for backend in ("mlx", "mtplx", "tensorfold"):
            self.assertEqual(platform_gate(chat_variant(backend=backend), "chat"), "apple-silicon", backend)
        self.assertEqual(platform_gate(chat_variant(backend="llama.cpp"), "chat"), "any")
        self.assertEqual(platform_gate(chat_variant(backend="vllm"), "chat"), "cuda")

    def test_image_and_video_runtime_gates(self):
        self.assertEqual(platform_gate(media_variant(runtime="mflux (MLX native)"), "image"), "apple-silicon")
        self.assertEqual(platform_gate(media_variant(engine="mflux", runtime=""), "image"), "apple-silicon")
        self.assertEqual(platform_gate(media_variant(runtime="ltx-2-mlx (MLX native)"), "video"), "apple-silicon")
        self.assertEqual(platform_gate(media_variant(repo="prince-canuma/LTX-2-dev", runtime=""), "video"), "apple-silicon")
        self.assertEqual(platform_gate(media_variant(runtime="diffusers + nunchaku SVDQuant (CUDA)"), "image"), "cuda")
        self.assertEqual(platform_gate(media_variant(runtime="stable-diffusion.cpp (subprocess)"), "image"), "any")

    def test_media_variants_are_gated_both_ways(self):
        mflux = media_variant(runtime="mflux (MLX native)", sizeGb=20.0)
        nunchaku = media_variant(runtime="diffusers + nunchaku SVDQuant (CUDA)", sizeGb=7.0)
        self.assertEqual(variant_fit(mflux, "image", CUDA_24)["status"], "unknown")
        self.assertEqual(variant_fit(nunchaku, "image", APPLE_64)["status"], "unknown")
        self.assertEqual(variant_fit(nunchaku, "image", CUDA_24)["status"], "fits")
        self.assertEqual(variant_fit(mflux, "image", APPLE_64)["status"], "fits")

    def test_unreadable_usable_memory_is_unknown(self):
        broken = HardwareClass("cpu", "cpu", "CPU only, 0 GB RAM", 0.0, 0.0)
        self.assertEqual(variant_fit(media_variant(), "image", broken)["status"], "unknown")


class MediaFootprintTests(unittest.TestCase):
    def test_footprint_for_the_hardware_wins_over_the_generic_one(self):
        variant = media_variant(runtimeFootprintGb=14.0, runtimeFootprintMpsGb=20.0, runtimeFootprintCudaGb=11.0)
        self.assertEqual(variant_fit(variant, "image", APPLE_64)["needGb"], 20.0)
        self.assertEqual(variant_fit(variant, "image", CUDA_24)["needGb"], 11.0)
        self.assertEqual(variant_fit(variant, "image", CPU_32)["needGb"], 14.0)

    def test_missing_footprint_falls_back_to_download_size_times_1_3(self):
        result = variant_fit(media_variant(sizeGb=10.0), "image", APPLE_64)
        self.assertEqual(result["needGb"], 13.0)
        self.assertEqual(result["basis"], "estimated")
        self.assertIn("download", result["reason"])

    def test_no_size_information_is_unknown(self):
        result = variant_fit(media_variant(sizeGb=None), "video", APPLE_64)
        self.assertEqual(result["status"], "unknown")

    def test_measured_footprint_is_marked_measured_on_apple_silicon(self):
        qwen = IMAGE_MODEL_FAMILIES_BY_ID["Qwen/Qwen-Image-2.1-mflux"]
        for hardware in (APPLE_64, APPLE_16):
            result = variant_fit(qwen, "image", hardware)
            self.assertEqual(result["basis"], "measured", hardware.label)
            self.assertEqual(result["needGb"], 50.0)
        self.assertEqual(variant_fit(qwen, "image", APPLE_16)["status"], "wont-fit")
        self.assertEqual(variant_fit(qwen, "image", APPLE_64)["status"], "tight")  # a measured peak that ran is never won't-fit
        ltx = VIDEO_VARIANTS_BY_ID["dgrauet/ltx-2.3-mlx-q4"]
        self.assertEqual(variant_fit(ltx, "video", APPLE_64)["basis"], "measured")

    def test_copied_or_scaled_footprints_stay_estimated(self):
        # The 8-bit row scales the bf16 figure; only the bf16 and Turbo rows were run.
        for variant_id in ("Qwen/Qwen-Image-2.1-mflux-q8",):
            result = variant_fit(IMAGE_MODEL_FAMILIES_BY_ID[variant_id], "image", APPLE_64)
            self.assertEqual(result["basis"], "estimated", variant_id)
        self.assertEqual(variant_fit(VIDEO_VARIANTS_BY_ID["prince-canuma/LTX-2.3-dev"], "video", APPLE_64)["basis"], "estimated")

    def test_a_measured_number_is_not_claimed_for_another_kind_of_machine(self):
        # Same catalog id and footprint key as the LTX-2.3 row measured on a Mac, but the machine here is a PC.
        variant = media_variant(id="prince-canuma/LTX-2.3-distilled", runtime="diffusers", runtimeFootprintGb=40.0)
        self.assertEqual(variant_fit(variant, "video", APPLE_64)["basis"], "measured")
        self.assertEqual(variant_fit(variant, "video", CUDA_24)["basis"], "estimated")

    def test_catalog_row_can_declare_its_footprint_measured(self):
        variant = media_variant(runtimeFootprintMpsGb=12.0, footprintBasis="measured")
        self.assertEqual(variant_fit(variant, "image", APPLE_64)["basis"], "measured")
        self.assertEqual(variant_fit(variant, "image", CUDA_24)["basis"], "estimated")

    def test_every_measured_footprint_id_exists_in_a_catalog(self):
        known = set(IMAGE_MODEL_FAMILIES_BY_ID) | set(VIDEO_VARIANTS_BY_ID)
        for variant_id, key in fit_mod.MEASURED_FOOTPRINTS.items():
            self.assertIn(variant_id, known)
            row = IMAGE_MODEL_FAMILIES_BY_ID.get(variant_id) or VIDEO_VARIANTS_BY_ID[variant_id]
            self.assertIn(key, row, f"{variant_id} has no {key}")


IMAGE_MODEL_FAMILIES_BY_ID = {v["id"]: v for f in IMAGE_MODEL_FAMILIES for v in f["variants"]}
VIDEO_VARIANTS_BY_ID = {v["id"]: v for f in VIDEO_MODEL_FAMILIES for v in f["variants"]}


def fitted(kind, hardware, **overrides):
    """A variant with its ``fit`` attached, as the payload builders produce it."""
    variant = (chat_variant if kind == "chat" else media_variant)(**overrides)
    variant["fit"] = variant_fit(variant, kind, hardware)
    return variant


class BestFitTests(unittest.TestCase):
    def test_newest_release_that_fits_wins_over_a_faster_older_one(self):
        old_fast = fitted("image", APPLE_64, id="old", releaseDate="2025-01", estimatedGenerationSeconds=2.0)
        new_slow = fitted("image", APPLE_64, id="new", releaseDate="2026-03", estimatedGenerationSeconds=90.0)
        self.assertEqual(best_fit_variant_id([old_fast, new_slow], "image"), "new")

    def test_a_newer_release_that_does_not_fit_is_skipped(self):
        fits = fitted("image", APPLE_64, id="fits", releaseDate="2025-06", runtimeFootprintMpsGb=20.0)
        tight = fitted("image", APPLE_64, id="tight", releaseDate="2026-06", runtimeFootprintMpsGb=40.0)
        wont = fitted("image", APPLE_64, id="wont", releaseDate="2026-09", runtimeFootprintMpsGb=80.0)
        unknown = fitted("image", APPLE_64, id="unknown", releaseDate="2026-10", runtime="nunchaku (CUDA)")
        self.assertEqual(tight["fit"]["status"], "tight")
        self.assertEqual(unknown["fit"]["status"], "unknown")
        self.assertEqual(best_fit_variant_id([fits, tight, wont, unknown], "image"), "fits")

    def test_same_release_goes_to_the_faster_one(self):
        slow = fitted("video", APPLE_64, id="slow", estimatedGenerationSeconds=300.0)
        quick = fitted("video", APPLE_64, id="quick", estimatedGenerationSeconds=60.0)
        self.assertEqual(best_fit_variant_id([slow, quick], "video"), "quick")

    def test_a_variant_with_no_time_estimate_loses_the_speed_tie(self):
        unknown_time = fitted("image", APPLE_64, id="untimed", estimatedGenerationSeconds=None)
        timed = fitted("image", APPLE_64, id="timed", estimatedGenerationSeconds=500.0)
        self.assertEqual(best_fit_variant_id([unknown_time, timed], "image"), "timed")

    def test_equal_speed_goes_to_the_smaller(self):
        big = fitted("image", APPLE_64, id="big", sizeGb=20.0, runtimeFootprintMpsGb=26.0)
        small = fitted("image", APPLE_64, id="small", sizeGb=8.0, runtimeFootprintMpsGb=10.0)
        self.assertEqual(best_fit_variant_id([big, small], "image"), "small")

    def test_a_row_that_runs_as_downloaded_beats_a_larger_convert_row(self):
        direct = fitted("chat", APPLE_64, id="direct", releaseDate="2026-02", paramsB=9.0, sizeGb=5.0, launchMode="direct")
        convert = fitted("chat", APPLE_64, id="convert", releaseDate="2026-02", paramsB=35.0, sizeGb=20.0, launchMode="convert")
        self.assertEqual(best_fit_variant_id([convert, direct], "chat"), "direct")
        # With nothing else that fits, the convert row is still the pick.
        self.assertEqual(best_fit_variant_id([convert], "chat"), "convert")

    def test_none_when_nothing_fits(self):
        tight = fitted("image", APPLE_64, id="tight", runtimeFootprintMpsGb=40.0)
        self.assertIsNone(best_fit_variant_id([tight], "image"))
        self.assertIsNone(best_fit_variant_id([], "image"))

    def test_missing_release_date_counts_as_oldest(self):
        undated = fitted("image", APPLE_64, id="undated", releaseDate=None, estimatedGenerationSeconds=1.0)
        dated = fitted("image", APPLE_64, id="dated", releaseDate="2024-08", estimatedGenerationSeconds=100.0)
        self.assertEqual(best_fit_variant_id([undated, dated], "image"), "dated")

    def test_chat_ties_use_the_speed_callback_only_for_tied_variants(self):
        a = fitted("chat", APPLE_64, id="a", releaseDate="2026-02")
        b = fitted("chat", APPLE_64, id="b", releaseDate="2026-02")
        older = fitted("chat", APPLE_64, id="older", releaseDate="2025-02")
        asked: list[str] = []

        def speed_for(variant):
            asked.append(variant["id"])
            return {"a": 40.0, "b": 90.0}[variant["id"]]

        self.assertEqual(best_fit_variant_id([a, b, older], "chat", speed_for=speed_for), "b")
        self.assertEqual(sorted(asked), ["a", "b"])
        asked.clear()
        self.assertEqual(best_fit_variant_id([a, older], "chat", speed_for=speed_for), "a")
        self.assertEqual(asked, [])  # a single newest candidate needs no speed

    def test_chat_with_unknown_speeds_and_equal_parameters_falls_through_to_the_smaller(self):
        # Two builds of one model (say 8-bit and bf16): same parameter count, no speed data.
        big = fitted("chat", APPLE_64, id="big", sizeGb=20.0)
        small = fitted("chat", APPLE_64, id="small", sizeGb=6.0)
        self.assertEqual(best_fit_variant_id([big, small], "chat", speed_for=lambda v: None), "small")

    def test_chat_prefers_the_larger_model_among_those_released_together(self):
        # A family that lists several sizes dated alike must recommend the most capable one that fits
        # comfortably, not the smallest (every Gemma 4 size shares one release date).
        tiny = fitted("chat", APPLE_64, id="tiny", sizeGb=2.0, paramsB=2.0, releaseDate="2025-05")
        mid = fitted("chat", APPLE_64, id="mid", sizeGb=18.0, paramsB=31.0, releaseDate="2025-05")
        self.assertEqual(best_fit_variant_id([tiny, mid], "chat", speed_for=lambda v: {"tiny": 200.0, "mid": 30.0}[v["id"]]), "mid")

    def test_a_larger_model_that_does_not_fit_is_never_chosen(self):
        tiny = fitted("chat", APPLE_64, id="tiny", sizeGb=2.0, paramsB=2.0, releaseDate="2025-05")
        huge = fitted("chat", APPLE_64, id="huge", sizeGb=120.0, paramsB=120.0, releaseDate="2025-05")
        self.assertEqual(best_fit_variant_id([tiny, huge], "chat"), "tiny")


class ChatAnnotationTests(unittest.TestCase):
    def family(self, *variants):
        return {"id": "fam", "variants": list(variants)}

    def test_measured_speed_beats_the_bandwidth_estimate(self):
        # Two builds of one model (same parameter count): the bandwidth estimate favours the smaller file.
        small = chat_variant(id="small", repo="o/small", sizeGb=4.0, paramsB=7.0)
        big = chat_variant(id="big", repo="o/big", sizeGb=12.0, paramsB=7.0)
        families = [self.family(small, big)]
        with mock.patch("backend_service.helpers.bandwidth.memory_bandwidth", return_value=(400.0, "test", True)):
            annotate_chat_families(families, APPLE_64, arch_for=lambda repo: None)
            self.assertEqual(families[0]["bestFitVariantId"], "small")
            measured = {"small": 20.0, "big": 60.0}
            annotate_chat_families(families, APPLE_64, arch_for=lambda repo: None, measured_speed_for=lambda v: measured[v["id"]])
        self.assertEqual(families[0]["bestFitVariantId"], "big")

    def test_every_variant_gets_fit_and_the_family_gets_a_pick(self):
        families = [self.family(chat_variant(id="x", repo="o/x"), chat_variant(id="y", repo="o/y", backend="vllm"))]
        annotate_chat_families(families, APPLE_64, arch_for=lambda repo: None)
        self.assertEqual(set(families[0]["variants"][0]["fit"]), FIT_KEYS)
        self.assertEqual(families[0]["variants"][1]["fit"]["status"], "unknown")
        self.assertEqual(families[0]["bestFitVariantId"], "x")

    def test_layout_lookup_is_skipped_for_variants_that_cannot_run_here(self):
        asked: list[str] = []

        def arch_for(repo):
            asked.append(repo)
            return None

        families = [self.family(chat_variant(id="x", repo="o/x"), chat_variant(id="y", repo="o/y", backend="vllm"))]
        annotate_chat_families(families, APPLE_64, arch_for=arch_for)
        self.assertEqual(asked, ["o/x"])

    def test_a_failing_layout_lookup_falls_back_to_the_table(self):
        def boom(repo):
            raise OSError("disk gone")

        families = [self.family(chat_variant(id="x", repo="o/x"))]
        annotate_chat_families(families, APPLE_64, arch_for=boom)
        self.assertEqual(families[0]["variants"][0]["fit"]["status"], "fits")

    def test_moe_name_marks_the_active_share(self):
        dense = chat_variant(id="d", name="Model 26B", repo="o/model-26b", paramsB=26.0, sizeGb=15.0)
        moe = chat_variant(id="m", name="Model 26B A4B", repo="o/model-26b-a4b-it", paramsB=26.0, sizeGb=15.0)
        self.assertGreater(
            estimate_chat_tok_s(moe, arch=None, bandwidth_gbps=400.0),
            estimate_chat_tok_s(dense, arch=None, bandwidth_gbps=400.0) * 2,
        )

    def test_local_layout_is_read_from_the_downloaded_snapshot_only(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({
                "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
                "hidden_size": 4096, "head_dim": 128,
            }))
            with mock.patch("backend_service.helpers.hf_cache_paths._hf_repo_snapshot_dir", return_value=Path(tmp)):
                arch = local_kv_architecture("org/model")
            self.assertIsNotNone(arch)
            self.assertEqual(arch.kv_heads, 8)
        with mock.patch("backend_service.helpers.hf_cache_paths._hf_repo_snapshot_dir", return_value=None):
            self.assertIsNone(local_kv_architecture("org/model"))
        self.assertIsNone(local_kv_architecture(""))


class HardwareClassTests(unittest.TestCase):
    def setUp(self):
        hc.reset_metal_cache()
        self.addCleanup(hc.reset_metal_cache)

    def classify(self, system, *, on_this_machine=False, metal=None, wired=None):
        with mock.patch.object(hc, "_on_macos", return_value=True), \
             mock.patch.object(hc, "_is_this_machine", return_value=on_this_machine), \
             mock.patch.object(hc, "_metal_info", return_value=metal), \
             mock.patch.object(hc, "_wired_limit_gb", return_value=wired):
            return hardware_class(system)

    def test_apple_tiers_and_heuristic_shares(self):
        cases = {16.0: ("apple-16", 10.7), 36.0: ("apple-36", 24.1), 48.0: ("apple-48", 36.0), 64.0: ("apple-64", 48.0)}
        for total, (bucket, usable) in cases.items():
            result = self.classify({"platform": "Darwin", "arch": "arm64", "totalMemoryGb": total})
            self.assertEqual((result.kind, result.bucket, result.usable_gb), ("apple", bucket, usable), total)

    def test_metal_limit_replaces_the_heuristic_on_this_machine(self):
        result = self.classify(SYSTEM_MAC_64, on_this_machine=True, metal=(64.0, 51.84))
        self.assertEqual(result.usable_gb, 51.8)

    def test_metal_limit_of_another_memory_size_is_ignored(self):
        result = self.classify(SYSTEM_MAC_64, on_this_machine=True, metal=(32.0, 25.0))
        self.assertEqual(result.usable_gb, 48.0)

    def test_a_system_dict_for_another_machine_ignores_this_machines_limits(self):
        result = self.classify(SYSTEM_MAC_16, on_this_machine=False, metal=(64.0, 51.84), wired=40.0)
        self.assertEqual(result.usable_gb, 10.7)

    def test_wired_limit_is_used_when_the_metal_read_fails(self):
        result = self.classify(SYSTEM_MAC_64, on_this_machine=True, metal=None, wired=56.0)
        self.assertEqual(result.usable_gb, 56.0)
        result = self.classify(SYSTEM_MAC_64, on_this_machine=True, metal=None, wired=None)
        self.assertEqual(result.usable_gb, 48.0)

    def test_cuda_uses_the_cards_memory(self):
        result = hardware_class(SYSTEM_PC_24GB_VRAM)
        self.assertEqual((result.kind, result.bucket, result.usable_gb, result.total_gb), ("cuda", "cuda-24", 24.0, 32.0))

    def test_cpu_only_host_gets_sixty_percent_of_ram(self):
        result = hardware_class({"platform": "Linux", "arch": "x86_64", "totalMemoryGb": 32.0, "gpuVramTotalGb": None})
        self.assertEqual((result.kind, result.bucket, result.usable_gb), ("cpu", "cpu", 19.2))

    def test_intel_mac_is_cpu_only(self):
        result = hardware_class({"platform": "Darwin", "arch": "x86_64", "totalMemoryGb": 16.0})
        self.assertEqual(result.kind, "cpu")

    def test_snapshot_without_an_arch_is_a_current_mac(self):
        result = self.classify({"platform": "Darwin", "totalMemoryGb": 24.0})
        self.assertEqual(result.kind, "apple")


class MetalReadTests(unittest.TestCase):
    def setUp(self):
        hc.reset_metal_cache()
        self.addCleanup(hc.reset_metal_cache)

    def run_read(self, result=None, raises=None):
        patches = [mock.patch("backend_service.inference.binaries._resolve_mlx_python", return_value="python")]
        if raises is not None:
            patches.append(mock.patch("backend_service.inference.binaries._json_subprocess", side_effect=raises))
        else:
            patches.append(mock.patch("backend_service.inference.binaries._json_subprocess", return_value=result))
        with patches[0], patches[1] as subprocess_mock:
            return hc._read_metal_working_set_gb(), subprocess_mock

    def test_parses_the_device_info(self):
        value, call = self.run_read((0, {"workingSetBytes": 55662788608, "memoryBytes": 68719476736}, ""))
        self.assertEqual(value, (64.0, 55662788608 / 1024 ** 3))
        command = call.call_args[0][0]
        self.assertEqual(command[:2], ["python", "-c"])
        self.assertIn("device_info", command[2])

    def test_failures_return_none_and_never_raise(self):
        self.assertIsNone(self.run_read((1, None, "boom"))[0])
        self.assertIsNone(self.run_read((0, {"workingSetBytes": 0, "memoryBytes": 1}, ""))[0])
        self.assertIsNone(self.run_read((0, {"workingSetBytes": 9, "memoryBytes": 4}, ""))[0])
        self.assertIsNone(self.run_read((0, ["not", "a", "dict"], ""))[0])
        self.assertIsNone(self.run_read(raises=RuntimeError("no python"))[0])

    def test_a_good_read_is_kept_and_a_failed_one_is_not_retried_at_once(self):
        with mock.patch.object(hc, "_read_metal_working_set_gb", return_value=(64.0, 51.84)) as read:
            self.assertEqual(hc._metal_info(), (64.0, 51.84))
            self.assertEqual(hc._metal_info(), (64.0, 51.84))
            self.assertEqual(read.call_count, 1)
        hc.reset_metal_cache()
        with mock.patch.object(hc, "_read_metal_working_set_gb", return_value=None) as read:
            self.assertIsNone(hc._metal_info())
            self.assertIsNone(hc._metal_info())
            self.assertEqual(read.call_count, 1)
            with mock.patch.object(hc.time, "monotonic", return_value=hc.time.monotonic() + 10_000):
                self.assertIsNone(hc._metal_info())
            self.assertEqual(read.call_count, 2)

    def test_a_slow_read_does_not_hold_the_caller_and_lands_later(self):
        import threading
        import time

        gate = threading.Event()

        def slow_read():
            gate.wait(5)
            return (64.0, 51.84)

        hc._host_cache["value"] = (time.monotonic(), "classified with the heuristic")
        with mock.patch.object(hc, "_read_metal_working_set_gb", side_effect=slow_read), \
             mock.patch.object(hc, "_METAL_WAIT_SECONDS", 0.05):
            started = time.monotonic()
            self.assertIsNone(hc._metal_info())
            self.assertLess(time.monotonic() - started, 1.0)
            gate.set()
            value = None
            for _ in range(100):
                value = hc._metal_info()
                if value is not None:
                    break
            self.assertEqual(value, (64.0, 51.84))
        self.assertNotIn("value", hc._host_cache, "the host class made without the read must be recomputed")

    def test_host_hardware_class_is_kept_for_a_minute(self):
        hc.reset_metal_cache()
        fake = {"platform": "Linux", "arch": "x86_64", "totalMemoryGb": 8.0, "gpuVramTotalGb": None}
        with mock.patch.object(hc, "host_system_dict", return_value=fake) as probe:
            first = hc.host_hardware_class()
            second = hc.host_hardware_class()
        self.assertIs(first, second)
        self.assertEqual(probe.call_count, 1)


def fake_live_metadata(_repo):
    return {}


@mock.patch("backend_service.helpers.video._image_repo_live_metadata", fake_live_metadata)
@mock.patch("backend_service.helpers.images._image_repo_live_metadata", fake_live_metadata)
@mock.patch("backend_service.helpers.hardware_class._is_this_machine", return_value=False)
@mock.patch("backend_service.helpers.fit.local_kv_architecture", return_value=None)
class CatalogPayloadTests(unittest.TestCase):
    """The real catalogs on a 16 GB Mac, a 64 GB Mac and a 24 GB-VRAM PC."""

    SYSTEMS = (("mac16", SYSTEM_MAC_16), ("mac64", SYSTEM_MAC_64), ("pc24", SYSTEM_PC_24GB_VRAM))

    def chat(self, system):
        from backend_service.helpers.discovery import _model_family_payloads

        return _model_family_payloads(system, [])

    def image(self, system):
        from backend_service.helpers.images import _image_model_payloads

        return _image_model_payloads([], hardware_class(system))

    def video(self, system):
        from backend_service.helpers.video import _video_model_payloads

        return _video_model_payloads([], hardware_class(system))

    def check_shape(self, families, label):
        self.assertTrue(families, label)
        for family in families:
            self.assertIn("bestFitVariantId", family, f"{label} {family['id']}")
            by_id = {variant["id"]: variant for variant in family["variants"]}
            for variant in family["variants"]:
                fit = variant["fit"]
                self.assertEqual(set(fit), FIT_KEYS, f"{label} {variant['id']}")
                self.assertIn(fit["status"], STATUSES)
                self.assertIn(fit["basis"], {"measured", "estimated"})
                self.assertTrue(fit["reason"])
                if fit["status"] == "unknown":
                    self.assertIsNone(fit["needGb"])
                else:
                    self.assertGreater(fit["needGb"], 0)
                    self.assertIsNotNone(fit["headroomPct"])
            best = family["bestFitVariantId"]
            if best is None:
                self.assertFalse(
                    [v for v in family["variants"] if v["fit"]["status"] == "fits"],
                    f"{label} {family['id']} has fitting variants but no pick",
                )
            else:
                self.assertEqual(by_id[best]["fit"]["status"], "fits", f"{label} {family['id']}")

    def test_chat_shape_on_every_system(self, *_):
        for label, system in self.SYSTEMS:
            self.check_shape(self.chat(system), f"chat {label}")

    def test_image_shape_on_every_system(self, *_):
        for label, system in self.SYSTEMS:
            self.check_shape(self.image(system), f"image {label}")

    def test_video_shape_on_every_system(self, *_):
        for label, system in self.SYSTEMS:
            self.check_shape(self.video(system), f"video {label}")

    def count(self, families, status):
        return sum(1 for f in families for v in f["variants"] if v["fit"]["status"] == status)

    def test_more_memory_fits_more_models(self, *_):
        for build in (self.chat, self.image, self.video):
            small, large = build(SYSTEM_MAC_16), build(SYSTEM_MAC_64)
            self.assertGreater(self.count(large, "fits"), self.count(small, "fits"), build.__name__)
            self.assertGreater(self.count(small, "wont-fit"), self.count(large, "wont-fit"), build.__name__)

    def test_chat_platform_split(self, *_):
        mac, pc = self.chat(SYSTEM_MAC_64), self.chat(SYSTEM_PC_24GB_VRAM)
        for families, blocked_backend in ((mac, "vllm"), (pc, "mlx")):
            variants = [v for f in families for v in f["variants"] if v["backend"] == blocked_backend]
            self.assertTrue(variants, blocked_backend)
            self.assertTrue(all(v["fit"]["status"] == "unknown" for v in variants), blocked_backend)
        mac_vllm = [v for f in mac for v in f["variants"] if v["backend"] == "vllm"]
        pc_vllm = [v for f in pc for v in f["variants"] if v["backend"] == "vllm"]
        self.assertTrue(any(v["fit"]["status"] != "unknown" for v in pc_vllm))
        self.assertTrue(all(v["fit"]["status"] == "unknown" for v in mac_vllm))

    def test_chat_budget_is_the_heuristic_usable_memory_for_the_fake_system(self, *_):
        usable = {v["fit"]["usableGb"] for f in self.chat(SYSTEM_MAC_64) for v in f["variants"]}
        self.assertEqual(usable, {48.0})
        usable = {v["fit"]["usableGb"] for f in self.chat(SYSTEM_PC_24GB_VRAM) for v in f["variants"]}
        self.assertEqual(usable, {24.0})

    def test_known_chat_rows_on_a_small_mac(self, *_):
        families = {f["id"]: f for f in self.chat(SYSTEM_MAC_16)}
        small = next(v for v in families["gemma-4"]["variants"] if v["id"] == "google/gemma-4-E2B-it-qat-q4_0-gguf")
        self.assertEqual(small["fit"]["status"], "fits")
        huge = next(v for f in families.values() for v in f["variants"] if v["id"] == "mlx-community/DeepSeek-V4-Flash-4bit")
        self.assertEqual(huge["fit"]["status"], "wont-fit")

    def test_image_catalog_pick_is_a_variant_of_the_family(self, *_):
        for family in self.image(SYSTEM_MAC_64):
            if family["bestFitVariantId"] is not None:
                self.assertIn(family["bestFitVariantId"], {v["id"] for v in family["variants"]})

    def test_video_pc_catalog_gates_mlx_rows_out_of_the_pick(self, *_):
        families = self.video(SYSTEM_PC_24GB_VRAM)
        for family in families:
            for variant in family["variants"]:
                if "mlx" in variant["runtime"].lower():
                    self.assertEqual(variant["fit"]["status"], "unknown", variant["id"])
        # Apple-only rows are dropped from the payload on a non-Apple host; on a Mac they stay but are unknown.
        ltx2 = next((f for f in families if f["id"] == "ltx-2"), None)
        if ltx2 is not None:
            self.assertIsNone(ltx2["bestFitVariantId"])


class RoutePayloadTests(unittest.TestCase):
    """The three routes carry ``fit`` and ``bestFitVariantId`` through the real handlers."""

    def setUp(self):
        from tests.test_video_routes import make_client

        self.client, self.tempdir, self.env_snapshot = make_client()
        self.addCleanup(self.tempdir.cleanup)

        def restore():
            from tests.test_video_routes import restore_env

            restore_env(self.env_snapshot)

        self.addCleanup(restore)
        for target in (
            "backend_service.helpers.video._image_repo_live_metadata",
            "backend_service.helpers.images._image_repo_live_metadata",
        ):
            patcher = mock.patch(target, fake_live_metadata)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch("backend_service.helpers.fit.host_hardware_class", return_value=CUDA_24)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch("backend_service.helpers.fit.local_kv_architecture", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch("backend_service.helpers.hardware_class._is_this_machine", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_video_catalog_route(self):
        payload = self.client.get("/api/video/catalog").json()
        for family in payload["families"]:
            self.assertIn("bestFitVariantId", family)
            for variant in family["variants"]:
                self.assertEqual(set(variant["fit"]), FIT_KEYS)
                self.assertEqual(variant["fit"]["usableGb"], 24.0)
        wan = next(f for f in payload["families"] if f["id"] == "wan-2-1")
        small = next(v for v in wan["variants"] if v["id"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.assertEqual(small["fit"]["status"], "fits")  # 14 GB footprint on a 24 GB card

    def test_image_catalog_route_covers_families_and_latest_rows(self):
        latest_rows = [
            {"id": "org/tracked", "repo": "org/tracked", "sizeGb": 8.0, "runtime": "Tracked diffusers candidate"},
            {"id": "org/unsized", "repo": "org/unsized", "sizeGb": None, "runtime": "Tracked diffusers candidate"},
        ]
        with mock.patch("backend_service.routes.images._latest_image_model_payloads", return_value=latest_rows):
            payload = self.client.get("/api/images/catalog").json()
        for family in payload["families"]:
            self.assertIn("bestFitVariantId", family)
            for variant in family["variants"]:
                self.assertEqual(set(variant["fit"]), FIT_KEYS)
        tracked, unsized = payload["latest"]
        self.assertEqual(tracked["fit"]["status"], "fits")
        self.assertEqual(unsized["fit"]["status"], "unknown")
        self.assertNotIn("fit", latest_rows[0], "the cached latest rows must not be mutated")

    def test_workspace_route_attaches_fit_to_the_featured_models(self):
        payload = self.client.get("/api/workspace").json()
        families = payload["featuredModels"]
        self.assertTrue(families)
        for family in families:
            self.assertIn("bestFitVariantId", family)
            for variant in family["variants"]:
                self.assertEqual(set(variant["fit"]), FIT_KEYS)
        # the fake system is a 48 GB Mac: 36 GB usable, so mlx rows are sized and vllm rows are not
        mlx = next(v for f in families for v in f["variants"] if v["backend"] == "mlx")
        vllm = next(v for f in families for v in f["variants"] if v["backend"] == "vllm")
        self.assertEqual(mlx["fit"]["usableGb"], 36.0)
        self.assertEqual(vllm["fit"]["status"], "unknown")

    def test_search_route_carries_fit_too(self):
        with mock.patch("backend_service.routes.models._search_huggingface_hub", return_value=[]):
            payload = self.client.get("/api/models/search", params={"q": "gemma"}).json()
        self.assertTrue(payload["results"])
        self.assertTrue(all("fit" in v for f in payload["results"] for v in f["variants"]))


if __name__ == "__main__":
    unittest.main()


class MeasuredPeakTests(unittest.TestCase):
    def test_a_measured_peak_that_ran_is_tight_not_wont_fit(self):
        # Qwen-Image-2.1 peaks at 46-50 GB of 51.8 on a 64 GB Mac and runs there.
        variant = {"id": "Qwen/Qwen-Image-2.1-mflux", "repo": "Qwen/Qwen-Image-2.1", "runtime": "mflux (MLX native)",
                   "runtimeFootprintMpsGb": 50.0, "sizeGb": 33.1, "releaseDate": "2026-09"}
        fit = variant_fit(variant, "image", APPLE_64)
        self.assertEqual(fit["basis"], "measured")
        self.assertEqual(fit["status"], "tight")

    def test_an_estimated_need_that_close_to_the_limit_is_still_wont_fit(self):
        variant = {"id": "x/y", "repo": "x/y", "runtime": "diffusers (MPS / CUDA)", "runtimeFootprintGb": 50.0,
                   "sizeGb": 30.0, "releaseDate": "2026-09"}
        self.assertEqual(variant_fit(variant, "image", APPLE_64)["status"], "wont-fit")
