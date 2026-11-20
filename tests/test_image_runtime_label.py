"""FU-476 / FU-472: display label for the runtime string, and the measured Nunchaku INT4 schnell figures."""

import unittest

from backend_service.catalog.image_models import IMAGE_MODEL_FAMILIES
from backend_service.helpers.images import image_runtime_label


def _variant(variant_id: str) -> dict:
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["id"] == variant_id:
                return variant
    raise KeyError(variant_id)


class RuntimeLabelTests(unittest.TestCase):
    def test_stub_rows_show_diffusers_and_keep_the_gate_string(self) -> None:
        schnell = _variant("black-forest-labs/FLUX.1-schnell")
        self.assertEqual(schnell["runtime"], "Stub diffusion pipeline", "the platform gate reads this string")
        self.assertEqual(image_runtime_label(schnell), "diffusers")

    def test_other_runtimes_are_shown_as_they_are(self) -> None:
        self.assertEqual(image_runtime_label({"runtime": "mflux (MLX native)"}), "mflux (MLX native)")
        self.assertIsNone(image_runtime_label({}))


class NunchakuSchnellFiguresTests(unittest.TestCase):
    def test_studio_figures_are_the_ones_measured_through_the_app_on_the_4090(self) -> None:
        row = _variant("black-forest-labs/FLUX.1-schnell-nunchaku-int4")
        # 5.9 s warm, about 18 GB peak on the card (2026-11-19, docs/progress/rtx4090-progress.md).
        self.assertEqual(row["estimatedGenerationSeconds"], 5.9)
        self.assertEqual(row["runtimeFootprintCudaGb"], 16.5)
        self.assertEqual(row["footprintBasis"], "measured")
        self.assertEqual(row["footprintMeasuredOn"], "cuda")
        self.assertNotIn("sub-second", row["note"])


if __name__ == "__main__":
    unittest.main()
