"""FLUX on CUDA: CPU offload only when the card cannot hold the whole pipeline.

No GPU needed: VRAM is passed in, or read from a fake ``torch``.
"""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

from backend_service.catalog.image_models import IMAGE_MODEL_FAMILIES
from backend_service.image_runtime import DiffusersTextToImageEngine
from backend_service.image_runtime.transformer_loaders import (
    cpu_offload_note,
    cuda_total_vram_gb,
    full_gpu_need_gb,
    should_use_model_cpu_offload,
)

SCHNELL = "black-forest-labs/FLUX.1-schnell"
DEV = "black-forest-labs/FLUX.1-dev"
NUNCHAKU_GB = 21.0
BF16_DEV_GB = 38.0
# Totals as torch reports them (GiB), for the common cards.
CARD_24 = 23.99   # RTX 4090
CARD_32 = 31.84   # RTX 5090
CARD_40 = 39.39   # A100 40 GB
CARD_48 = 47.99   # RTX 6000 Ada / A6000
CARD_80 = 79.25   # H100 / A100 80 GB


def _variant(variant_id: str) -> dict:
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["id"] == variant_id:
                return variant
    raise KeyError(variant_id)


class ShouldUseModelCpuOffloadTests(unittest.TestCase):
    def test_unknown_vram_or_footprint_keeps_offload(self):
        self.assertTrue(should_use_model_cpu_offload(SCHNELL, "cuda"))
        self.assertTrue(should_use_model_cpu_offload(SCHNELL, "cuda", None, NUNCHAKU_GB))
        self.assertTrue(should_use_model_cpu_offload(SCHNELL, "cuda", CARD_80, None))
        self.assertTrue(should_use_model_cpu_offload(SCHNELL, "cuda", 0.0, NUNCHAKU_GB))
        self.assertTrue(should_use_model_cpu_offload(SCHNELL, "cuda", CARD_80, 0.0))

    def test_24_gb_card_keeps_offload_for_the_nunchaku_footprint(self):
        self.assertTrue(should_use_model_cpu_offload(SCHNELL, "cuda", CARD_24, NUNCHAKU_GB))

    def test_32_48_80_gb_cards_run_nunchaku_fully_on_the_gpu(self):
        for card in (CARD_32, CARD_48, CARD_80):
            with self.subTest(card=card):
                self.assertFalse(
                    should_use_model_cpu_offload(SCHNELL, "cuda", card, NUNCHAKU_GB)
                )

    def test_bf16_flux_dev_needs_a_48_gb_card(self):
        # 38 GB x 1.15 + 2 GB = 45.7 GB.
        self.assertAlmostEqual(full_gpu_need_gb(BF16_DEV_GB), 45.7)
        self.assertTrue(should_use_model_cpu_offload(DEV, "cuda", CARD_24, BF16_DEV_GB))
        self.assertTrue(should_use_model_cpu_offload(DEV, "cuda", CARD_32, BF16_DEV_GB))
        self.assertTrue(should_use_model_cpu_offload(DEV, "cuda", CARD_40, BF16_DEV_GB))
        self.assertFalse(should_use_model_cpu_offload(DEV, "cuda", CARD_48, BF16_DEV_GB))
        self.assertFalse(should_use_model_cpu_offload(DEV, "cuda", CARD_80, BF16_DEV_GB))

    def test_non_flux_repos_and_non_cuda_devices_are_unaffected(self):
        for repo in ("stabilityai/stable-diffusion-xl-base-1.0", "Qwen/Qwen-Image"):
            for vram, footprint in ((None, None), (CARD_24, 60.0), (CARD_80, 10.0)):
                with self.subTest(repo=repo, vram=vram):
                    self.assertFalse(should_use_model_cpu_offload(repo, "cuda", vram, footprint))
        for device in ("mps", "cpu"):
            self.assertFalse(should_use_model_cpu_offload(SCHNELL, device))
            self.assertFalse(should_use_model_cpu_offload(SCHNELL, device, 8.0, NUNCHAKU_GB))


class CudaTotalVramTests(unittest.TestCase):
    def test_reads_total_memory_of_the_current_device(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            current_device=lambda: 0,
            get_device_properties=lambda _i: SimpleNamespace(total_memory=24 * 1024 ** 3),
        ))
        self.assertAlmostEqual(cuda_total_vram_gb(fake_torch), 24.0)

    def test_returns_none_when_cuda_cannot_be_read(self):
        def _boom(_i):
            raise RuntimeError("no device")

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            current_device=lambda: 0, get_device_properties=_boom,
        ))
        self.assertIsNone(cuda_total_vram_gb(fake_torch))
        self.assertIsNone(cuda_total_vram_gb(SimpleNamespace()))


class CpuOffloadNoteTests(unittest.TestCase):
    def test_notes(self):
        self.assertIsNone(cpu_offload_note(None, NUNCHAKU_GB, True))
        self.assertIsNone(cpu_offload_note(CARD_80, None, True))
        self.assertEqual(
            cpu_offload_note(CARD_24, NUNCHAKU_GB, True),
            "Using CPU offload: this card has 24 GB, the pipeline needs about 26 GB.",
        )
        self.assertEqual(
            cpu_offload_note(CARD_32, NUNCHAKU_GB, False),
            "Kept the whole pipeline on the GPU (32 GB card, about 21 GB needed).",
        )


class CatalogFootprintTests(unittest.TestCase):
    def test_nunchaku_rows_carry_the_measured_footprint(self):
        for family in IMAGE_MODEL_FAMILIES:
            for variant in family["variants"]:
                if variant.get("nunchakuRepo"):
                    with self.subTest(variant=variant["id"]):
                        self.assertEqual(variant.get("fullGpuFootprintGb"), NUNCHAKU_GB)

    def test_flux_rows_have_plausible_footprints(self):
        self.assertEqual(_variant(DEV)["fullGpuFootprintGb"], BF16_DEV_GB)
        self.assertEqual(_variant(SCHNELL)["fullGpuFootprintGb"], BF16_DEV_GB)
        for family in IMAGE_MODEL_FAMILIES:
            for variant in family["variants"]:
                footprint = variant.get("fullGpuFootprintGb")
                if footprint is None:
                    continue
                with self.subTest(variant=variant["id"]):
                    # Never below the download: the whole pipeline is on the card.
                    self.assertGreaterEqual(footprint, float(variant["sizeGb"]))
                    self.assertIsNone(variant.get("engine"))


class _FakePipeline:
    def __init__(self):
        self.offloaded = False
        self.moved_to = None
        self.transformer_kwarg = None

    def enable_model_cpu_offload(self):
        self.offloaded = True

    def to(self, device):
        self.moved_to = device
        return self


def _fake_torch(total_gb: float | None) -> ModuleType:
    torch = ModuleType("torch")

    def _props(_index):
        if total_gb is None:
            raise RuntimeError("no device")
        return SimpleNamespace(total_memory=int(total_gb * 1024 ** 3))

    torch.cuda = SimpleNamespace(
        is_available=lambda: False,  # keeps the SageAttention helper a no-op
        current_device=lambda: 0,
        get_device_properties=_props,
        empty_cache=lambda: None,
    )
    torch.bfloat16 = "bf16"
    torch.float16 = "fp16"
    torch.float32 = "fp32"
    return torch


class EnsurePipelineOffloadTests(unittest.TestCase):
    """``_ensure_pipeline`` with diffusers and torch faked out."""

    def _load(self, *, total_gb, footprint_gb, nunchaku_result):
        engine = DiffusersTextToImageEngine()
        pipeline = _FakePipeline()

        def _from_pretrained(_path, **kwargs):
            pipeline.transformer_kwarg = kwargs.get("transformer")
            return pipeline

        diffusers = ModuleType("diffusers")
        diffusers.AutoPipelineForText2Image = SimpleNamespace(from_pretrained=_from_pretrained)
        nf4 = mock.Mock(return_value=(None, None))
        with mock.patch.dict(sys.modules, {"torch": _fake_torch(total_gb), "diffusers": diffusers}), \
                mock.patch("backend_service.helpers.hf_local.local_snapshot_path", return_value="/snap"), \
                mock.patch("backend_service.image_runtime.validate_local_diffusers_snapshot", return_value=None), \
                mock.patch.object(engine, "_detect_device", return_value="cuda"), \
                mock.patch.object(engine, "_try_load_nunchaku_transformer", return_value=nunchaku_result), \
                mock.patch.object(engine, "_try_load_nf4_flux_transformer", nf4):
            engine._ensure_pipeline(
                SCHNELL,
                nunchaku_repo="mit-han-lab/svdq-int4-flux.1-schnell",
                full_gpu_footprint_gb=footprint_gb,
            )
        return engine, pipeline, nf4

    def test_32_gb_card_keeps_nunchaku_on_the_gpu_and_says_so(self):
        loaded = (object(), "Nunchaku SVDQuant transformer loaded from mit-han-lab/svdq-int4-flux.1-schnell.")
        engine, pipeline, nf4 = self._load(total_gb=32.0, footprint_gb=NUNCHAKU_GB, nunchaku_result=loaded)
        self.assertFalse(pipeline.offloaded)
        self.assertEqual(pipeline.moved_to, "cuda")
        nf4.assert_not_called()
        self.assertIn(loaded[1], engine._load_notes)
        self.assertIn(
            "Kept the whole pipeline on the GPU (32 GB card, about 21 GB needed).",
            engine._load_notes,
        )

    def test_24_gb_card_offloads_and_says_so(self):
        loaded = (object(), "Nunchaku SVDQuant transformer loaded.")
        engine, pipeline, _nf4 = self._load(total_gb=24.0, footprint_gb=NUNCHAKU_GB, nunchaku_result=loaded)
        self.assertTrue(pipeline.offloaded)
        self.assertIsNone(pipeline.moved_to)
        self.assertIn("Nunchaku SVDQuant transformer loaded.", engine._load_notes)
        self.assertIn(
            "Using CPU offload: this card has 24 GB, the pipeline needs about 26 GB.",
            engine._load_notes,
        )

    def test_unknown_footprint_or_vram_keeps_offload(self):
        loaded = (object(), "Nunchaku SVDQuant transformer loaded.")
        for total_gb, footprint_gb in ((80.0, None), (None, NUNCHAKU_GB)):
            with self.subTest(total_gb=total_gb, footprint_gb=footprint_gb):
                engine, pipeline, _nf4 = self._load(
                    total_gb=total_gb, footprint_gb=footprint_gb, nunchaku_result=loaded,
                )
                self.assertTrue(pipeline.offloaded)
                self.assertEqual(engine._load_notes, ["Nunchaku SVDQuant transformer loaded."])

    def test_nunchaku_fallback_goes_back_to_offload_and_reports_why(self):
        # The 21 GB footprint was for the INT4 transformer; without it the
        # bf16 one loads, so even an 80 GB card goes back to the old path.
        missing = (None, "Nunchaku (SVDQuant) not installed. Falling back to the standard transformer.")
        engine, pipeline, nf4 = self._load(total_gb=80.0, footprint_gb=NUNCHAKU_GB, nunchaku_result=missing)
        self.assertTrue(pipeline.offloaded)
        nf4.assert_called_once()
        self.assertEqual(engine._load_notes, [missing[1]])


if __name__ == "__main__":
    unittest.main()
