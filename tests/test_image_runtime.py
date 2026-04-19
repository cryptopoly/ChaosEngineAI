import json
import tempfile
import unittest
from pathlib import Path

from backend_service.image_runtime import (
    DiffusersTextToImageEngine,
    ImageGenerationConfig,
    PlaceholderImageEngine,
    validate_local_diffusers_snapshot,
)


def _ltx_model_index() -> dict:
    """The LTX-Video pipeline contract — five required components."""
    return {
        "_class_name": "LTXPipeline",
        "_diffusers_version": "0.32.0.dev0",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5Tokenizer"],
        "transformer": ["diffusers", "LTXVideoTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLLTXVideo"],
    }


def _seed_component(root: Path, name: str, config_filename: str = "config.json") -> None:
    component_dir = root / name
    component_dir.mkdir(parents=True, exist_ok=True)
    (component_dir / config_filename).write_text("{}", encoding="utf-8")


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


class ValidateLocalDiffusersSnapshotTests(unittest.TestCase):
    """Regression coverage for the LTX-Video corruption case.

    Real bug: a user pulled LTX-Video before the allow_patterns scoping landed,
    HF queued the legacy root-level safetensors first, and the snapshot ended
    up with model_index.json + scheduler/ + text_encoder/ but no transformer/
    or vae/. Diffusers then raised a generic "no file named config.json" error
    pointing at the snapshot root, which is unhelpful. The validator should
    catch this BEFORE we hand the directory to ``from_pretrained``.
    """

    def test_returns_none_when_all_components_present(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "model_index.json").write_text(
                json.dumps(_ltx_model_index()), encoding="utf-8",
            )
            _seed_component(root, "scheduler", "scheduler_config.json")
            _seed_component(root, "text_encoder")
            _seed_component(root, "tokenizer", "tokenizer_config.json")
            _seed_component(root, "transformer")
            _seed_component(root, "vae")

            self.assertIsNone(validate_local_diffusers_snapshot(root, "Lightricks/LTX-Video"))

    def test_flags_missing_component_subfolders(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "model_index.json").write_text(
                json.dumps(_ltx_model_index()), encoding="utf-8",
            )
            _seed_component(root, "scheduler", "scheduler_config.json")
            _seed_component(root, "text_encoder")
            # transformer/, tokenizer/, vae/ deliberately missing — exactly the
            # corrupt-LTX shape we hit in the wild.

            error = validate_local_diffusers_snapshot(root, "Lightricks/LTX-Video")

            self.assertIsNotNone(error)
            self.assertIn("missing components", error)
            self.assertIn("transformer", error)
            self.assertIn("tokenizer", error)
            self.assertIn("vae", error)
            self.assertNotIn("scheduler", error)
            self.assertNotIn("text_encoder", error)

    def test_flags_component_folder_present_but_empty(self):
        """An empty subfolder is the same failure mode as a missing one."""
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "model_index.json").write_text(
                json.dumps(_ltx_model_index()), encoding="utf-8",
            )
            _seed_component(root, "scheduler", "scheduler_config.json")
            _seed_component(root, "text_encoder")
            _seed_component(root, "tokenizer", "tokenizer_config.json")
            _seed_component(root, "vae")
            (root / "transformer").mkdir()  # exists but no config.json

            error = validate_local_diffusers_snapshot(root, "Lightricks/LTX-Video")

            self.assertIsNotNone(error)
            self.assertIn("transformer", error)

    def test_skips_optional_null_components(self):
        """Pipelines list ``[null, null]`` for opted-out components.

        The validator must not flag them — they're deliberately absent on
        community checkpoints (e.g. SDXL without safety_checker).
        """
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            index = {
                "_class_name": "StableDiffusionPipeline",
                "scheduler": ["diffusers", "DDIMScheduler"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "unet": ["diffusers", "UNet2DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "safety_checker": [None, None],
                "feature_extractor": [None, None],
            }
            (root / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
            _seed_component(root, "scheduler", "scheduler_config.json")
            _seed_component(root, "text_encoder")
            _seed_component(root, "tokenizer", "tokenizer_config.json")
            _seed_component(root, "unet")
            _seed_component(root, "vae")

            self.assertIsNone(validate_local_diffusers_snapshot(root, "runwayml/stable-diffusion-v1-5"))

    def test_still_flags_missing_model_index(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            error = validate_local_diffusers_snapshot(root, "Lightricks/LTX-Video")
            self.assertIsNotNone(error)
            self.assertIn("model_index.json", error)

    def test_handles_malformed_model_index(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "model_index.json").write_text("{not valid json", encoding="utf-8")
            error = validate_local_diffusers_snapshot(root, "Lightricks/LTX-Video")
            self.assertIsNotNone(error)
            self.assertIn("model_index.json", error)


if __name__ == "__main__":
    unittest.main()
