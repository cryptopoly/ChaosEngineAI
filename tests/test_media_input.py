import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from backend_service.helpers.media_input import InputImageError, image_size, resolve_input_image
from backend_service.ltx2_mlx_runtime import Ltx2MlxVideoEngine
from backend_service.models import VideoGenerationRequest
from backend_service.video_runtime.types import VideoGenerationConfig


def _png(path: Path, size=(64, 32)) -> Path:
    from PIL import Image

    Image.new("RGB", size, "red").save(path)
    return path


class ResolveInputImageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_accepts_a_png_and_reads_its_size(self):
        path = _png(self.dir / "a.png")
        self.assertEqual(resolve_input_image(str(path)), path.resolve())
        self.assertEqual(image_size(path), (64, 32))

    def test_rejects_bad_input_with_readable_messages(self):
        (self.dir / "note.txt").write_text("x")
        (self.dir / "empty.png").write_bytes(b"")
        for raw, needle in (
            ("", "empty"),
            (str(self.dir / "missing.png"), "not found"),
            (str(self.dir), "not a file"),
            (str(self.dir / "note.txt"), "PNG, JPEG or WebP"),
            (str(self.dir / "empty.png"), "empty"),
        ):
            with self.assertRaises(InputImageError, msg=raw) as ctx:
                resolve_input_image(raw)
            self.assertIn(needle, str(ctx.exception))

    def test_size_cap(self):
        path = _png(self.dir / "big.png")
        with mock.patch("backend_service.helpers.media_input.MAX_INPUT_IMAGE_BYTES", 10):
            with self.assertRaises(InputImageError):
                resolve_input_image(str(path))


class LtxCommandTests(unittest.TestCase):
    def config(self, **extra):
        return VideoGenerationConfig(
            modelId="m", modelName="LTX", repo="dgrauet/ltx-2.5-mlx-q4", prompt="p", negativePrompt="",
            width=768, height=512, numFrames=65, fps=24, guidance=1.0, **extra,
        )

    def build(self, config):
        return Ltx2MlxVideoEngine._build_command(config, Path("/pack"), None, 768, 512, 7, Path("/o.mp4"))

    def test_an_end_image_alone_anchors_the_last_frame(self):
        args = self.build(self.config(endImagePath="/b.png"))
        pairs = [args[i + 1 : i + 4] for i, a in enumerate(args) if a == "--image"]
        self.assertEqual(pairs, [["/b.png", "64", "1"]])

    def test_no_images_no_flag(self):
        self.assertNotIn("--image", self.build(self.config()))

    def test_start_and_end_images_anchor_first_and_last_frame(self):
        args = self.build(self.config(startImagePath="/a.png", endImagePath="/b.png", imageStrength=0.8))
        pairs = [args[i + 1 : i + 4] for i, a in enumerate(args) if a == "--image"]
        self.assertEqual(pairs, [["/a.png", "0", "0.8"], ["/b.png", "64", "0.8"]])
        self.assertEqual(args[-2:], ["--output", str(Path("/o.mp4"))])


class RouteChecksTests(unittest.TestCase):
    def check(self, body, repo="dgrauet/ltx-2.5-mlx-q4"):
        from backend_service.routes.video import _checked_input_images

        return _checked_input_images(body, {"name": "Model", "repo": repo})

    def request(self, **kw):
        return VideoGenerationRequest(modelId="m", prompt="p", **kw)

    def test_no_image_passes_through(self):
        body = self.request()
        self.assertIs(self.check(body, repo="Wan-AI/other"), body)

    def test_other_models_refuse_with_a_reason(self):
        with self.assertRaises(HTTPException) as ctx:
            self.check(self.request(startImagePath="/x.png"), repo="Wan-AI/other")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("LTX-2", ctx.exception.detail)

    def test_missing_file_is_refused(self):
        for kw in ({"startImagePath": "/nope.png"}, {"endImagePath": "/nope.png"}):
            with self.assertRaises(HTTPException) as ctx:
                self.check(self.request(**kw))
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("not found", ctx.exception.detail)

    def test_an_end_image_alone_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _png(Path(tmp) / "e.png")
            out = self.check(self.request(endImagePath=str(path)))
            self.assertIsNone(out.startImagePath)
            self.assertEqual(out.endImagePath, str(path.resolve()))

    def test_valid_paths_are_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _png(Path(tmp) / "a.png")
            out = self.check(self.request(startImagePath=str(path)))
            self.assertEqual(out.startImagePath, str(path.resolve()))


class ReferenceImageTests(unittest.TestCase):
    """Slice B: reference images for image models."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.png = _png(Path(self.tmp.name) / "r.png")

    @staticmethod
    def variant(repo, engine="mflux"):
        return {"name": "Model", "repo": repo, "engine": engine}

    def cmd(self, repo, refs, strength=None):
        from backend_service.image_runtime.mflux_engine import MfluxImageEngine, mflux_family_for_repo
        from backend_service.image_runtime.types import ImageGenerationConfig

        config = ImageGenerationConfig(
            modelId="x", modelName="M", repo=repo, prompt="p", negativePrompt="", width=1024, height=768,
            steps=4, guidance=3.5, batchSize=1, runtime="mflux", referenceImagePaths=refs, imageStrength=strength,
        )
        return MfluxImageEngine()._build_command(mflux_family_for_repo(repo), config, seed=1, output_path=Path("/o.png"))

    def test_limits_per_family(self):
        from backend_service.image_runtime.mflux_engine import reference_image_limit as limit

        self.assertEqual(limit("black-forest-labs/FLUX.1-dev"), 1)
        self.assertEqual(limit("black-forest-labs/FLUX.2-klein-4B"), 1)
        self.assertEqual(limit("Qwen/Qwen-Image-2.1"), 1)
        self.assertEqual(limit("black-forest-labs/flux-3-image-klein"), 4)
        self.assertEqual(limit("speridlabs/iris-3b"), 0)
        self.assertEqual(limit("stabilityai/stable-diffusion-3.5-medium"), 0)

    def test_init_image_command(self):
        args = self.cmd("black-forest-labs/FLUX.1-dev", ["/a.png"], 0.6)
        self.assertEqual(Path(args[0]).stem, "mflux-generate")
        self.assertEqual(args[args.index("--image") : args.index("--image") + 3], ["--image", "/a.png", "0.6"])
        no_strength = self.cmd("black-forest-labs/FLUX.1-dev", ["/a.png"])
        self.assertEqual(no_strength[no_strength.index("--image") + 1 :][:1], ["/a.png"])
        self.assertNotIn("0.6", no_strength)

    def test_flux3_uses_the_edit_command_with_all_images(self):
        args = self.cmd("black-forest-labs/flux-3-image-klein", ["/a.png", "/b.png"])
        self.assertEqual(Path(args[0]).stem, "mflux-generate-flux3-edit")
        i = args.index("--image-paths")
        self.assertEqual(args[i + 1 : i + 3], ["/a.png", "/b.png"])

    def test_no_reference_keeps_the_normal_command(self):
        args = self.cmd("black-forest-labs/flux-3-image-klein", None)
        self.assertEqual(Path(args[0]).stem, "mflux-generate-flux3")
        self.assertNotIn("--image", args)
        self.assertNotIn("--image-paths", args)

    def test_model_check(self):
        from backend_service.helpers.media_input import checked_reference_images as check

        self.assertEqual(check(None, self.variant("x")), [])
        out = check([str(self.png)], self.variant("black-forest-labs/FLUX.1-dev"))
        self.assertEqual(out, [str(self.png.resolve())])
        for variant, refs, needle in (
            (self.variant("speridlabs/iris-3b"), [str(self.png)], "cannot take a reference image"),
            (self.variant("black-forest-labs/FLUX.1-dev", engine="diffusers"), [str(self.png)], "cannot take a reference image"),
            (self.variant("black-forest-labs/FLUX.1-dev"), [str(self.png)] * 2, "takes 1 reference image"),
            (self.variant("black-forest-labs/FLUX.1-dev"), ["/nope.png"], "not found"),
        ):
            with self.assertRaises(InputImageError) as ctx:
                check(refs, variant)
            self.assertIn(needle, str(ctx.exception))

    def test_route_returns_400(self):
        import backend_service.app  # noqa: F401  (loads the routes in their usual order)
        from backend_service.models import ImageGenerationRequest
        from backend_service.routes.images import _checked_reference_images

        body = ImageGenerationRequest(modelId="m", prompt="p", referenceImagePaths=[str(self.png)])
        with self.assertRaises(HTTPException) as ctx:
            _checked_reference_images(body, self.variant("speridlabs/iris-3b"))
        self.assertEqual(ctx.exception.status_code, 400)
        out = _checked_reference_images(body, self.variant("Qwen/Qwen-Image-2.1"))
        self.assertEqual(out.referenceImagePaths, [str(self.png.resolve())])

    def test_request_limits(self):
        from pydantic import ValidationError

        from backend_service.models import ImageGenerationRequest

        with self.assertRaises(ValidationError):
            ImageGenerationRequest(modelId="m", prompt="p", referenceImagePaths=["/a.png"] * 5)
        with self.assertRaises(ValidationError):
            ImageGenerationRequest(modelId="m", prompt="p", imageStrength=0.0)


class CatalogFlagsTests(unittest.TestCase):
    """Slice C: the catalogs tell the Studio which variants take an input image."""

    def test_image_variant_inputs_report_reference_limit(self):
        from backend_service.helpers.images import _image_variant_inputs as inputs

        self.assertEqual(inputs({"engine": "mflux", "repo": "black-forest-labs/flux-3-image-klein"})["maxReferenceImages"], 4)
        self.assertEqual(inputs({"engine": "mflux", "repo": "Qwen/Qwen-Image-2.1"})["maxReferenceImages"], 1)
        self.assertEqual(inputs({"engine": "mflux", "repo": "speridlabs/iris-3b"})["maxReferenceImages"], 0)
        self.assertEqual(inputs({"repo": "black-forest-labs/FLUX.1-dev"})["maxReferenceImages"], 0)

    def test_video_payloads_flag_who_takes_a_start_and_an_end_picture(self):
        from backend_service.helpers.video import _video_model_payloads

        # Diffusers-only rows are hidden on Apple Silicon; pin the host so this runs the same everywhere.
        with mock.patch("backend_service.helpers.video.is_apple_silicon", return_value=False):
            families = _video_model_payloads([])
        rows = {v["repo"]: v for family in families for v in family["variants"]}
        # Wan 2.2 TI2V-5B (diffusers) follows the first frame only; LTX-2 (a Mac engine, so absent from the list on
        # other computers) takes both; the rest take none.
        ti2v = rows["Wan-AI/Wan2.2-TI2V-5B-Diffusers"]
        self.assertTrue(ti2v["acceptsImageInput"])
        self.assertFalse(ti2v["acceptsEndImage"])
        if "dgrauet/ltx-2.5-mlx-q4" in rows:
            self.assertTrue(rows["dgrauet/ltx-2.5-mlx-q4"]["acceptsImageInput"])
            self.assertTrue(rows["dgrauet/ltx-2.5-mlx-q4"]["acceptsEndImage"])
        self.assertFalse(rows["Lightricks/LTX-Video"]["acceptsImageInput"])
        self.assertFalse(rows["Lightricks/LTX-Video"]["acceptsEndImage"])

if __name__ == "__main__":
    unittest.main()
