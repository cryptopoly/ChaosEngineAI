"""Image and video fit badges on a discrete NVIDIA card follow the runtime's model offload (RTX 4090 audit).

Measured on an RTX 4090 through the app: FLUX.2 Klein 4B peaked at 9.1 GB at 1024x1024 (the badge said 20.8 GB and
"tight"); Wan 2.2 TI2V 5B at 17.3 GB at its recommended 1280x704 x 121 frames (badge 22 GB, "tight"). A Mac, a
DGX Spark (shared memory) and rows the runtime keeps whole on the card are unchanged.
"""
from __future__ import annotations

import unittest

from backend_service.catalog.image_models import IMAGE_MODEL_FAMILIES
from backend_service.catalog.video_models import VIDEO_MODEL_FAMILIES
from backend_service.helpers.fit import variant_fit
from backend_service.helpers.hardware_class import HardwareClass, hardware_class


def _row(families, variant_id):
    for family in families:
        for variant in family["variants"]:
            if variant["id"] == variant_id:
                return dict(variant)
    raise KeyError(variant_id)


KLEIN = _row(IMAGE_MODEL_FAMILIES, "black-forest-labs/FLUX.2-klein-4B")
WAN_TI2V = _row(VIDEO_MODEL_FAMILIES, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
LTX_4BIT = _row(VIDEO_MODEL_FAMILIES, "OzzyGT/LTX-2.3-Distilled-bnb-nf4")


def _card(gb: float) -> HardwareClass:
    return HardwareClass("cuda", f"cuda-{gb:g}", "", 64.0, gb)


class CudaOffloadFitTableTests(unittest.TestCase):
    def test_flux2_klein(self):
        expected = {8: "wont-fit", 12: "wont-fit", 16: "tight", 24: "fits", 32: "fits", 96: "fits"}
        for gb, status in expected.items():
            fit = variant_fit(dict(KLEIN), "image", _card(gb))
            self.assertEqual(fit["status"], status, (gb, fit))
        on_24 = variant_fit(dict(KLEIN), "image", _card(24))
        self.assertAlmostEqual(on_24["needGb"], 16.0 * 1.3 * 0.55, delta=0.1)  # 11.4 against 9.1 measured
        self.assertIn("model offload", on_24["reason"])

    def test_wan_ti2v(self):
        expected = {12: "wont-fit", 16: "wont-fit", 24: "tight", 32: "fits", 96: "fits"}
        for gb, status in expected.items():
            self.assertEqual(variant_fit(dict(WAN_TI2V), "video", _card(gb))["status"], status, gb)
        self.assertAlmostEqual(variant_fit(dict(WAN_TI2V), "video", _card(24))["needGb"], 17.6, delta=0.1)

    def test_measured_rows_keep_their_measurement(self):
        fit = variant_fit(dict(LTX_4BIT), "video", _card(24))
        self.assertEqual((fit["status"], fit["needGb"]), ("tight", 22.5))
        self.assertNotIn("model offload", fit["reason"])

    def test_rows_the_runtime_keeps_whole_are_not_offloaded(self):
        sd = {"id": "x/sd35", "repo": "stabilityai/stable-diffusion-3.5-large", "runtime": "diffusers StableDiffusion3Pipeline",
              "sizeGb": 28.0}
        self.assertEqual(variant_fit(sd, "image", _card(24))["status"], "wont-fit")

    def test_shared_memory_and_macs_are_unchanged(self):
        spark = hardware_class({"totalMemoryGb": 128.0, "gpuVramTotalGb": 128.0, "gpuUnifiedMemory": True})
        fit = variant_fit(dict(KLEIN), "image", spark)
        self.assertEqual((fit["status"], fit["needGb"]), ("fits", 20.8))
        mac = HardwareClass("apple", "apple-24", "", 24.0, 16.0)
        self.assertEqual(variant_fit(dict(KLEIN), "image", mac)["needGb"], 20.8)


if __name__ == "__main__":
    unittest.main()
