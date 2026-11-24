"""FU-273: FLUX 3 Image Klein on diffusers and sd.cpp, Qwen-Image-2.1 Turbo on sd.cpp, Kandinsky 6.0 Lite video.

No network, no GPU: catalogue rows, engine routing, the modular snapshot check, the FLUX.3 loader and call against
stand-ins, and the sound sample rate Kandinsky states on its pipeline.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from backend_service.catalog.image_models import IMAGE_MODEL_FAMILIES
from backend_service.catalog.video_models import VIDEO_MODEL_FAMILIES


def _variant(families: list[dict], variant_id: str) -> dict:
    return next(v for f in families for v in f["variants"] if v["id"] == variant_id)


FLUX3 = "black-forest-labs/flux-3-image-klein"
FLUX3_SDCPP = "leejet/flux-3-image-klein-GGUF-q8"
QWEN_TURBO = "unsloth/Qwen-Image-2.1-Turbo-GGUF-q4km"
KANDINSKY = "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers"


class CatalogRowTests(unittest.TestCase):
    def test_flux3_diffusers_row(self) -> None:
        row = _variant(IMAGE_MODEL_FAMILIES, FLUX3)
        self.assertEqual(row["familyId"], "flux-3")
        self.assertEqual(row["repo"], FLUX3)
        self.assertNotIn("engine", row)
        self.assertTrue(row["cudaOnly"])
        self.assertIn("FLUX Kommunity", row["note"])
        self.assertIn("US$5M", row["note"])
        self.assertEqual(row["footprintMeasuredOn"], "cuda")
        self.assertEqual(row["taskSupport"], ["txt2img"])  # editing did not follow the prompt here (FU-273)
        self.assertIsInstance(row["runtimeFootprintCudaGb"], float)

    def test_flux3_sdcpp_row_pins_its_files(self) -> None:
        row = _variant(IMAGE_MODEL_FAMILIES, FLUX3_SDCPP)
        self.assertEqual(row["engine"], "sdcpp")
        self.assertEqual(row["repo"], row["ggufRepo"])
        self.assertEqual(row["ggufFile"], "flux3-image-klein-Q8_0.gguf")
        self.assertEqual(row["sdcppAux"]["--llm"]["repo"], "Qwen/Qwen3-VL-4B-Instruct-GGUF")
        self.assertEqual(row["sdcppAux"]["--vae"], {"repo": FLUX3, "file": "ae.safetensors"})
        self.assertIn("FLUX Kommunity", row["note"])

    def test_qwen_turbo_row_runs_its_own_schedule(self) -> None:
        row = _variant(IMAGE_MODEL_FAMILIES, QWEN_TURBO)
        self.assertEqual(row["engine"], "sdcpp")
        self.assertEqual(row["repo"], row["ggufRepo"])
        args = row["sdcppArgs"]
        sigmas = args[args.index("--sigmas") + 1].split(",")
        self.assertEqual(len(sigmas), 9)  # the 8 shipped sigmas plus the terminal 0
        self.assertEqual(row["defaultSteps"], 8)
        self.assertTrue(row["estimateRef"]["ignoresSteps"])
        self.assertIn("non-commercial", row["note"])

    def test_kandinsky_row(self) -> None:
        row = _variant(VIDEO_MODEL_FAMILIES, KANDINSKY)
        self.assertEqual(row["repo"], KANDINSKY)
        self.assertTrue(row["cudaOnly"])
        self.assertIn("MIT", row["note"])
        self.assertIn("img2video", row["taskSupport"])
        self.assertEqual(row["estimateRef"]["steps"], 10)


class RoutingTests(unittest.TestCase):
    def test_flux3_goes_to_the_modular_loader(self) -> None:
        from backend_service.image_runtime.flux3_modular import is_flux3_modular_repo

        self.assertTrue(is_flux3_modular_repo(FLUX3))
        self.assertFalse(is_flux3_modular_repo("black-forest-labs/FLUX.2-klein-4B"))
        self.assertFalse(is_flux3_modular_repo(None))

    def test_sdcpp_accepts_the_new_gguf_repos_and_never_falls_back(self) -> None:
        from backend_service.image_runtime import _NO_DIFFUSERS_PIPELINE
        from backend_service.sdcpp_image_runtime import supported_repos

        for repo in ("leejet/flux-3-image-klein-GGUF", "unsloth/Qwen-Image-2.1-Turbo-GGUF"):
            self.assertIn(repo, supported_repos())
            self.assertIn(repo, _NO_DIFFUSERS_PIPELINE)

    def test_kandinsky_pipeline_and_defaults(self) -> None:
        from backend_service.video_runtime.repos import (
            IMAGE_TO_VIDEO_REGISTRY,
            PIPELINE_REGISTRY,
            _VIDEO_PIPELINE_DEFAULTS,
        )

        self.assertEqual(PIPELINE_REGISTRY[KANDINSKY]["class_name"], "Kandinsky6TI2VAPipeline")
        self.assertEqual(IMAGE_TO_VIDEO_REGISTRY[KANDINSKY], {"class_name": "Kandinsky6TI2VAPipeline", "endImage": False})
        self.assertEqual(_VIDEO_PIPELINE_DEFAULTS[KANDINSKY]["steps"], 10)
        self.assertEqual(_VIDEO_PIPELINE_DEFAULTS[KANDINSKY]["guidance"], 1.0)

    def test_flux3_studio_inputs(self) -> None:
        from backend_service.helpers.images import _image_variant_inputs

        inputs = _image_variant_inputs(_variant(IMAGE_MODEL_FAMILIES, FLUX3))
        self.assertEqual(inputs, {"acceptsGuidance": False, "acceptsNegativePrompt": False, "maxReferenceImages": 0})
        # B1309: the sd.cpp row edits from up to 4 reference pictures.
        self.assertEqual(_image_variant_inputs(_variant(IMAGE_MODEL_FAMILIES, FLUX3_SDCPP))["maxReferenceImages"], 4)

    def test_flux3_diffusers_row_refuses_reference_images(self) -> None:
        # FU-273: the diffusers row returned the reference nearly unchanged; only the sd.cpp row takes them (B1309).
        from backend_service.helpers.media_input import InputImageError, checked_reference_images

        with self.assertRaisesRegex(InputImageError, "cannot take a reference image"):
            checked_reference_images(["a"], _variant(IMAGE_MODEL_FAMILIES, FLUX3))
        with self.assertRaisesRegex(InputImageError, "not found"):
            checked_reference_images(["a"], _variant(IMAGE_MODEL_FAMILIES, FLUX3_SDCPP))

    def test_downloads_fetch_the_modular_index(self) -> None:
        from backend_service.helpers.images import _image_repo_allow_patterns

        patterns = _image_repo_allow_patterns(FLUX3)
        self.assertIn("modular_model_index.json", patterns)
        self.assertNotIn("*.safetensors", patterns)  # never the root reference-code weights or the fp8/nvfp4 variants


class ModularSnapshotTests(unittest.TestCase):
    def _snapshot(self, root: Path, *, transformer: bool = True) -> None:
        spec = lambda sub: {"pretrained_model_name_or_path": FLUX3, "subfolder": sub}  # noqa: E731
        index = {
            "_class_name": "Flux3DistilledModularPipeline",
            "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration", spec("text_encoder")],
            "tokenizer": ["transformers", "Qwen3VLProcessor", spec("text_encoder")],
            "transformer": ["diffusers", "Flux3Transformer2DModel", spec("transformer")],
            "vae": ["diffusers", "AutoencoderKLFlux2", spec("vae")],
            "prompt_upsampler": ["transformers", "Qwen3_5ForConditionalGeneration",
                                 {"pretrained_model_name_or_path": "black-forest-labs/flux-3-image-klein-pup", "subfolder": ""}],
        }
        (root / "modular_model_index.json").write_text(json.dumps(index), encoding="utf-8")
        for name in ("text_encoder", "vae") + (("transformer",) if transformer else ()):
            (root / name).mkdir()
            (root / name / "config.json").write_text("{}", encoding="utf-8")

    def test_complete_modular_snapshot_passes_without_the_upsampler(self) -> None:
        from backend_service.image_runtime.snapshot import validate_local_diffusers_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            self._snapshot(Path(tmp))
            self.assertIsNone(validate_local_diffusers_snapshot(Path(tmp), FLUX3))

    def test_missing_transformer_is_reported(self) -> None:
        from backend_service.image_runtime.snapshot import validate_local_diffusers_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            self._snapshot(Path(tmp), transformer=False)
            self.assertIn("transformer", validate_local_diffusers_snapshot(Path(tmp), FLUX3) or "")


class _FakeTransformer:
    def __init__(self) -> None:
        self.hooks: list = []

    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        handle = mock.Mock()
        handle.remove.side_effect = lambda: self.hooks.remove(hook)
        return handle


class Flux3LoaderTests(unittest.TestCase):
    def test_load_drops_the_upsampler_and_loads_from_the_snapshot(self) -> None:
        from backend_service.image_runtime import flux3_modular

        sub_blocks = {"prompt_upsampler": object(), "text_encoder": object()}
        pipeline = mock.Mock()
        pipeline.pretrained_component_names = ["text_encoder", "tokenizer", "transformer", "vae",
                                               "prompt_upsampler", "prompt_upsampler_processor"]
        blocks = mock.Mock(sub_blocks=sub_blocks)
        blocks.init_pipeline.return_value = pipeline
        fake = types.SimpleNamespace(
            ModularPipeline=mock.Mock(from_pretrained=mock.Mock(return_value=mock.Mock(blocks=blocks))),
        )
        torch = types.SimpleNamespace(float32="f32", bfloat16="bf16")
        with mock.patch.dict(sys.modules, {"diffusers": fake}),                 mock.patch.object(flux3_modular, "place_on_cuda", return_value="encoder note") as place:
            loaded, notes = flux3_modular.load_pipeline("C:/snap", torch, "cuda")
        self.assertIs(loaded, pipeline)
        self.assertNotIn("prompt_upsampler", sub_blocks)
        kwargs = pipeline.load_components.call_args.kwargs
        self.assertEqual(kwargs["names"], ["text_encoder", "tokenizer", "transformer", "vae"])
        self.assertEqual(kwargs["pretrained_model_name_or_path"], "C:/snap")
        self.assertTrue(kwargs["local_files_only"])
        self.assertEqual(kwargs["dtype"], {"vae": "f32", "default": "bf16"})
        place.assert_called_once_with(pipeline, torch)
        pipeline.vae.enable_tiling.assert_called_once_with()
        self.assertTrue(any("upsampler" in note for note in notes))
        self.assertIn("encoder note", notes)

    def _placement(self, total_gb: float):
        from backend_service.image_runtime import flux3_modular

        gb = 1024**3
        pipeline = mock.Mock()
        pipeline.text_encoder.get_memory_footprint.return_value = 8.9 * gb
        pipeline.transformer.get_memory_footprint.return_value = 11.1 * gb
        pipeline.vae.get_memory_footprint.return_value = 0.3 * gb
        pipeline.text_encoder.parameters.side_effect = lambda: iter([types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))])
        torch = mock.Mock()
        torch.cuda.get_device_properties.return_value = types.SimpleNamespace(total_memory=total_gb * gb)
        hook = mock.Mock()
        accelerate = types.SimpleNamespace(cpu_offload_with_hook=mock.Mock(return_value=(None, hook)))
        with mock.patch.dict(sys.modules, {"accelerate": accelerate}):
            note = flux3_modular.place_on_cuda(pipeline, torch)
        return pipeline, note, hook, torch

    def test_a_24_gb_card_keeps_the_text_encoder_in_ram_between_prompts(self) -> None:
        pipeline, note, hook, torch = self._placement(24.0)
        self.assertIn("system RAM", note)
        pipeline.to.assert_not_called()
        pipeline.transformer.to.assert_called_once_with("cuda")
        pre_hook = pipeline.transformer.register_forward_pre_hook.call_args.args[0]
        pre_hook(None, ())
        hook.offload.assert_called_once_with()
        torch.cuda.empty_cache.assert_called_once_with()

    def test_a_big_card_holds_everything(self) -> None:
        pipeline, note, _hook, _torch = self._placement(48.0)
        self.assertIsNone(note)
        pipeline.to.assert_called_once_with("cuda")

    def test_generate_reports_steps(self) -> None:
        from backend_service.image_runtime import flux3_modular

        transformer = _FakeTransformer()
        seen: dict = {}

        def run(output, **kwargs):
            seen.update(kwargs, output=output)
            for _ in range(4):
                for hook in list(transformer.hooks):
                    hook(transformer, ())
            return ["img"]

        pipeline = mock.Mock(side_effect=run)
        pipeline.transformer = transformer
        steps: list[tuple[int, int]] = []
        images, seconds = flux3_modular.generate(
            pipeline, repo=FLUX3, prompt="p", width=512, height=768, steps=4, generator="g",
            batch_size=1, on_step=lambda d, t: steps.append((d, t)),
        )
        self.assertEqual(images, ["img"])
        self.assertGreater(seconds, 0)
        self.assertEqual(steps, [(0, 4), (1, 4), (2, 4), (3, 4)])
        self.assertEqual(seen["output"], "images")
        self.assertEqual((seen["width"], seen["height"], seen["num_inference_steps"]), (512, 768, 4))
        self.assertNotIn("image", seen)
        self.assertEqual(transformer.hooks, [])  # the hook is removed after the run


class MossVendoredCodeTests(unittest.TestCase):
    def test_the_vendored_pipeline_is_the_pinned_upstream_file(self) -> None:
        import hashlib

        from backend_service.image_runtime import moss_image

        path = Path(moss_image.__file__).with_name("moss_image_pipeline.py")
        lines = path.read_bytes().replace(b"\r\n", b"\n").split(b"\n")  # a Windows checkout may add CRs
        header = [line for line in lines if line.startswith(b"#")][:7]
        body = b"\n".join(lines[len(header):])
        self.assertIn(moss_image.MOSS_REVISION.encode(), b"\n".join(header))
        self.assertEqual(
            hashlib.sha256(body).hexdigest(),
            "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31",
        )

    def test_the_vendored_pipeline_reaches_no_network_or_processes(self) -> None:
        from backend_service.image_runtime import moss_image

        import re

        text = Path(moss_image.__file__).with_name("moss_image_pipeline.py").read_text(encoding="utf-8")
        code = "\n".join(line for line in text.splitlines() if not line.startswith("#"))  # the header names them
        for marker in ("subprocess", "os.system", "requests", "urllib", "socket", "trust_remote_code", "open("):
            self.assertFalse(marker in code, marker)
        self.assertIsNone(re.search(r"(?<![.\w])(exec|eval|__import__)\(", code))  # module.eval() is fine

    def test_catalog_row(self) -> None:
        row = _variant(IMAGE_MODEL_FAMILIES, "OpenMOSS-Team/MOSS-Image-SFT")
        self.assertNotIn("engine", row)
        self.assertTrue(row["cudaOnly"])
        self.assertIn("Apache 2.0", row["note"])
        self.assertEqual((row["defaultSteps"], row["cfgOverride"]), (50, 3.0))

    def test_downloads_skip_the_repo_code_at_the_root(self) -> None:
        from backend_service.helpers.images import _image_repo_allow_patterns

        patterns = _image_repo_allow_patterns("OpenMOSS-Team/MOSS-Image-SFT")
        self.assertNotIn("*.py", patterns)
        self.assertNotIn("pipeline_moss_image.py", patterns)

    def test_routing(self) -> None:
        from backend_service.image_runtime.moss_image import is_moss_repo

        self.assertTrue(is_moss_repo("OpenMOSS-Team/MOSS-Image-SFT"))
        self.assertFalse(is_moss_repo("OpenMOSS-Team/MOSS-Image-Base"))


class KandinskyAudioRateTests(unittest.TestCase):
    def test_rate_comes_from_the_pipeline_when_the_vocoder_has_none(self) -> None:
        from backend_service.video_runtime.pipeline_helpers import invoke_pipeline

        result = types.SimpleNamespace(frames=[["f1", "f2"]], audio="wave")

        class Pipe:
            audio_sample_rate = 44100
            vocoder = types.SimpleNamespace(config=types.SimpleNamespace())

            def __call__(self, **kwargs):
                return result

        frames = invoke_pipeline(Pipe(), {"num_inference_steps": 10, "guidance_scale": 1.0})
        self.assertEqual(list(frames), ["f1", "f2"])
        self.assertEqual(frames.audio_sample_rate, 44100)


if __name__ == "__main__":
    unittest.main()
