"""Catalog LoRAs are fetched on first use, then fused from the local cache."""
from __future__ import annotations

import unittest
from unittest import mock

from backend_service.helpers.lora import fuse_catalog_lora


class _FakePipeline:
    def __init__(self, calls: list[tuple], *, unload_raises: bool = False) -> None:
        self.calls = calls
        self.unload_raises = unload_raises

    def load_lora_weights(self, repo, **kwargs):
        self.calls.append(("load", repo, kwargs))

    def fuse_lora(self, **kwargs):
        self.calls.append(("fuse", kwargs))

    def unload_lora_weights(self):
        self.calls.append(("unload",))
        if self.unload_raises:
            raise RuntimeError("not supported")


class FuseCatalogLoraTests(unittest.TestCase):
    def _fuse(self, pipeline, scale=0.8):
        with mock.patch(
            "backend_service.helpers.hf_local.fetch_file",
            side_effect=lambda repo, name: pipeline.calls.append(("fetch", repo, name)) or "/cache/x",
        ):
            return fuse_catalog_lora(pipeline, "Kijai/WanVideo_comfy", "lora.safetensors", scale)

    def test_the_file_is_fetched_before_the_cache_only_load(self):
        calls: list[tuple] = []
        note = self._fuse(_FakePipeline(calls))

        self.assertEqual([call[0] for call in calls], ["fetch", "load", "fuse", "unload"])
        self.assertEqual(calls[0], ("fetch", "Kijai/WanVideo_comfy", "lora.safetensors"))
        self.assertEqual(
            calls[1],
            ("load", "Kijai/WanVideo_comfy", {"weight_name": "lora.safetensors", "local_files_only": True}),
        )
        self.assertEqual(calls[2], ("fuse", {"lora_scale": 0.8}))
        self.assertEqual(note, "LoRA: Kijai/WanVideo_comfy/lora.safetensors @ scale 0.800")

    def test_a_missing_scale_fuses_at_full_strength(self):
        calls: list[tuple] = []
        self._fuse(_FakePipeline(calls), scale=None)
        self.assertEqual(calls[2], ("fuse", {"lora_scale": 1.0}))

    def test_a_pipeline_that_cannot_unload_still_counts_as_fused(self):
        calls: list[tuple] = []
        note = self._fuse(_FakePipeline(calls, unload_raises=True))
        self.assertTrue(note.startswith("LoRA: "))

    def test_a_failed_fetch_raises_before_touching_the_pipeline(self):
        calls: list[tuple] = []
        pipeline = _FakePipeline(calls)
        with mock.patch("backend_service.helpers.hf_local.fetch_file", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                fuse_catalog_lora(pipeline, "a/b", "c.safetensors", 1.0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
