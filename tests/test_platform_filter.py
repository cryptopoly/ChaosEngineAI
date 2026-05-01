"""Tests for the MLX-only catalog filter.

Validates that ``filter_mlx_only_families`` strips Apple-only variants on
non-Apple hosts and leaves them visible on Apple Silicon. The detector
covers explicit ``mlxOnly`` flags, ``engine`` markers, and the runtime
strings used by the live catalog.
"""

from __future__ import annotations

import unittest

from backend_service.helpers.platform_filter import (
    filter_mlx_only_families,
    is_apple_silicon,
    is_mlx_only_variant,
)


def _flux_dev_gguf() -> dict[str, object]:
    return {
        "id": "black-forest-labs/FLUX.1-dev-gguf-q8",
        "name": "FLUX.1 Dev · GGUF Q8_0",
        "engine": None,
        "runtime": "Stub diffusion pipeline",
        "styleTags": ["general", "detailed", "gguf"],
    }


def _flux_dev_mflux() -> dict[str, object]:
    return {
        "id": "black-forest-labs/FLUX.1-dev-mflux",
        "name": "FLUX.1 Dev · mflux (MLX)",
        "engine": "mflux",
        "runtime": "mflux (MLX native)",
        "styleTags": ["general", "detailed", "apple-silicon"],
    }


def _ltx2_distilled_mlx() -> dict[str, object]:
    return {
        "id": "prince-canuma/LTX-2-distilled",
        "name": "LTX-2 · distilled (MLX)",
        "runtime": "mlx-video (MLX native)",
        "styleTags": ["general", "fast", "motion", "mlx"],
    }


def _wan_diffusers() -> dict[str, object]:
    return {
        "id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "name": "Wan 2.1 T2V 1.3B",
        "runtime": "diffusers (MPS / CUDA)",
        "styleTags": ["general", "motion"],
    }


class IsAppleSiliconTests(unittest.TestCase):
    def test_darwin_arm64_is_apple_silicon(self) -> None:
        self.assertTrue(is_apple_silicon(system="Darwin", machine="arm64"))

    def test_darwin_x86_64_is_not_apple_silicon(self) -> None:
        self.assertFalse(is_apple_silicon(system="Darwin", machine="x86_64"))

    def test_windows_is_not_apple_silicon(self) -> None:
        self.assertFalse(is_apple_silicon(system="Windows", machine="AMD64"))

    def test_linux_is_not_apple_silicon(self) -> None:
        self.assertFalse(is_apple_silicon(system="Linux", machine="x86_64"))


class IsMlxOnlyVariantTests(unittest.TestCase):
    def test_mflux_engine_marker(self) -> None:
        self.assertTrue(is_mlx_only_variant(_flux_dev_mflux()))

    def test_mlx_video_runtime_marker(self) -> None:
        self.assertTrue(is_mlx_only_variant(_ltx2_distilled_mlx()))

    def test_explicit_mlx_only_flag(self) -> None:
        variant = {"id": "x", "name": "x", "mlxOnly": True}
        self.assertTrue(is_mlx_only_variant(variant))

    def test_diffusers_runtime_is_not_mlx_only(self) -> None:
        self.assertFalse(is_mlx_only_variant(_wan_diffusers()))

    def test_gguf_variant_is_not_mlx_only(self) -> None:
        self.assertFalse(is_mlx_only_variant(_flux_dev_gguf()))

    def test_engine_field_case_insensitive(self) -> None:
        variant = {"id": "x", "engine": "MFlux"}
        self.assertTrue(is_mlx_only_variant(variant))


class FilterMlxOnlyFamiliesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.flux_family = {
            "id": "flux-dev",
            "name": "FLUX.1 Dev",
            "variants": [_flux_dev_gguf(), _flux_dev_mflux()],
        }
        self.ltx_only_family = {
            "id": "ltx-2",
            "name": "LTX-2 (MLX)",
            "variants": [_ltx2_distilled_mlx()],
        }
        self.wan_family = {
            "id": "wan-2-1",
            "name": "Wan 2.1",
            "variants": [_wan_diffusers()],
        }

    def test_apple_silicon_passes_everything_through(self) -> None:
        families = [self.flux_family, self.ltx_only_family, self.wan_family]
        result = filter_mlx_only_families(families, on_apple_silicon=True)
        self.assertEqual(len(result), 3)
        self.assertEqual([f["id"] for f in result], ["flux-dev", "ltx-2", "wan-2-1"])

    def test_non_apple_drops_mlx_variants(self) -> None:
        families = [self.flux_family]
        result = filter_mlx_only_families(families, on_apple_silicon=False)
        self.assertEqual(len(result), 1)
        ids = [v["id"] for v in result[0]["variants"]]
        self.assertEqual(ids, ["black-forest-labs/FLUX.1-dev-gguf-q8"])

    def test_non_apple_drops_mlx_only_families(self) -> None:
        """A family whose only variant is MLX-only disappears entirely."""
        families = [self.flux_family, self.ltx_only_family, self.wan_family]
        result = filter_mlx_only_families(families, on_apple_silicon=False)
        ids = [f["id"] for f in result]
        self.assertEqual(ids, ["flux-dev", "wan-2-1"])

    def test_does_not_mutate_input(self) -> None:
        families = [self.flux_family]
        original_variant_count = len(families[0]["variants"])
        _ = filter_mlx_only_families(families, on_apple_silicon=False)
        self.assertEqual(len(families[0]["variants"]), original_variant_count)

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(filter_mlx_only_families([], on_apple_silicon=True), [])
        self.assertEqual(filter_mlx_only_families([], on_apple_silicon=False), [])


if __name__ == "__main__":
    unittest.main()
