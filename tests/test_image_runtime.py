import unittest

from backend_service.image_runtime import (
    DiffusersTextToImageEngine,
    ImageGenerationConfig,
    PlaceholderImageEngine,
)


class PlaceholderImageEngineTests(unittest.TestCase):
    def test_placeholder_generates_svg_without_pillow(self):
        engine = PlaceholderImageEngine()
        config = ImageGenerationConfig(
            modelId="placeholder",
            modelName="Placeholder",
            repo="placeholder/repo",
            prompt="A skyline at dusk",
            negativePrompt="",
            width=512,
            height=512,
            steps=16,
            guidance=5.0,
            batchSize=1,
            seed=123,
        )

        result = engine.generate(config)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].extension, "svg")
        self.assertEqual(result[0].mimeType, "image/svg+xml")
        self.assertIn(b"<svg", result[0].bytes)


class DiffusersTextToImageEngineTests(unittest.TestCase):
    def test_qwen_image_prefers_cpu_over_mps(self):
        engine = DiffusersTextToImageEngine()

        device = engine._preferred_execution_device("Qwen/Qwen-Image", "mps")

        self.assertEqual(device, "cpu")

    def test_qwen_image_uses_true_cfg_scale_and_blank_negative_prompt(self):
        engine = DiffusersTextToImageEngine()
        config = ImageGenerationConfig(
            modelId="Qwen/Qwen-Image",
            modelName="Qwen-Image",
            repo="Qwen/Qwen-Image",
            prompt="A neon cafe sign",
            negativePrompt="",
            width=1024,
            height=1024,
            steps=30,
            guidance=4.0,
            batchSize=1,
        )

        kwargs = engine._build_pipeline_kwargs(config, generator="GEN")

        self.assertNotIn("guidance_scale", kwargs)
        self.assertEqual(kwargs["true_cfg_scale"], 4.0)
        self.assertEqual(kwargs["negative_prompt"], " ")
        self.assertEqual(kwargs["generator"], "GEN")

    def test_non_qwen_image_keeps_guidance_scale(self):
        engine = DiffusersTextToImageEngine()
        config = ImageGenerationConfig(
            modelId="stabilityai/stable-diffusion-xl-base-1.0",
            modelName="Stable Diffusion XL Base 1.0",
            repo="stabilityai/stable-diffusion-xl-base-1.0",
            prompt="A landscape",
            negativePrompt="blurry",
            width=1024,
            height=1024,
            steps=30,
            guidance=6.5,
            batchSize=1,
        )

        kwargs = engine._build_pipeline_kwargs(config, generator="GEN")

        self.assertEqual(kwargs["guidance_scale"], 6.5)
        self.assertEqual(kwargs["negative_prompt"], "blurry")


if __name__ == "__main__":
    unittest.main()
