import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service import ltx2_mlx_runtime as ltx2
from backend_service.catalog import VIDEO_MODEL_FAMILIES
from backend_service.helpers import video as video_helpers
from backend_service.helpers.companions import companion_downloads
from backend_service.video_runtime.types import VideoGenerationConfig

REPO = "dgrauet/ltx-2.3-mlx-q4"


def _variant() -> dict:
    return next(v for f in VIDEO_MODEL_FAMILIES for v in f["variants"] if v["repo"] == REPO)


def _config(**overrides) -> VideoGenerationConfig:
    values = dict(
        modelId=REPO, modelName="LTX-2.3 · distilled q4", repo=REPO, prompt="a red fox", negativePrompt="",
        width=768, height=512, numFrames=97, fps=24, guidance=3.0, steps=50, seed=7,
    )
    values.update(overrides)
    return VideoGenerationConfig(**values)


class CatalogTests(unittest.TestCase):
    def test_the_variant_downloads_only_the_distilled_files(self) -> None:
        patterns = video_helpers._video_repo_allow_patterns(REPO)
        for name in ltx2.REQUIRED_FILES:
            self.assertIn(name, patterns)
        self.assertNotIn("transformer-dev.safetensors", patterns)
        self.assertFalse(any("lora" in p for p in patterns))

    def test_the_text_encoder_comes_with_the_download(self) -> None:
        self.assertEqual(companion_downloads(REPO), [(ltx2.GEMMA_REPO, ["*"])])

    def test_the_required_files_match_the_catalog(self) -> None:
        pinned = set(_variant()["downloadFiles"])
        self.assertTrue(set(ltx2.REQUIRED_FILES) <= pinned)

    def test_the_repo_is_routed_to_this_engine(self) -> None:
        self.assertTrue(ltx2._is_ltx2_mlx_repo(REPO))
        self.assertFalse(ltx2._is_ltx2_mlx_repo("prince-canuma/LTX-2.3-distilled"))
        self.assertFalse(ltx2._is_ltx2_mlx_repo(None))

    def test_a_missing_pack_is_not_installed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(video_helpers, "_hf_repo_snapshot_dir", return_value=tmp):
            error = video_helpers._video_variant_validation_error(_variant())
        self.assertIn("missing", error)

    def test_a_complete_pack_with_its_text_encoder_is_installed(self) -> None:
        with tempfile.TemporaryDirectory() as pack, tempfile.TemporaryDirectory() as gemma:
            for name in ltx2.REQUIRED_FILES:
                (Path(pack) / name).write_bytes(b"x")
            (Path(gemma) / "model.safetensors.index.json").write_text("{}")
            roots = {REPO: pack, ltx2.GEMMA_REPO: gemma}
            with mock.patch.object(video_helpers, "_hf_repo_snapshot_dir", side_effect=lambda repo: roots.get(repo)):
                self.assertIsNone(video_helpers._video_variant_validation_error(_variant()))
                roots.pop(ltx2.GEMMA_REPO)
                self.assertIn("text encoder", video_helpers._video_variant_validation_error(_variant()))


class EngineTests(unittest.TestCase):
    def test_sizes_snap_to_sixty_four(self) -> None:
        self.assertEqual(ltx2.align_size(768, 512), (768, 512))
        self.assertEqual(ltx2.align_size(770, 500), (768, 512))
        self.assertEqual(ltx2.align_size(10, 10), (64, 64))

    def test_the_command_runs_the_distilled_pipeline_with_local_paths(self) -> None:
        args = ltx2.Ltx2MlxVideoEngine._build_command(
            _config(), Path("/pack"), Path("/gemma"), 768, 512, 7, Path("/tmp/out.mp4")
        )
        self.assertEqual(args[1:3], ["generate", "--distilled"])
        self.assertEqual(args[args.index("--model") + 1], "/pack")
        self.assertEqual(args[args.index("--gemma") + 1], "/gemma")
        self.assertEqual(args[args.index("--frames") + 1], "97")
        self.assertEqual(args[args.index("--frame-rate") + 1], "24")
        self.assertEqual(args[args.index("--seed") + 1], "7")
        self.assertNotIn("--negative-prompt", args)
        self.assertNotIn("--quiet", args)

    def test_generation_refuses_when_the_engine_is_not_installed(self) -> None:
        with mock.patch.object(ltx2, "is_installed", return_value=False), mock.patch.object(ltx2, "_apple_silicon", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                ltx2.Ltx2MlxVideoEngine().generate(_config())

    def test_generation_refuses_an_unknown_repo(self) -> None:
        with mock.patch.object(ltx2, "is_installed", return_value=True), mock.patch.object(ltx2, "_apple_silicon", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "does not support"):
                ltx2.Ltx2MlxVideoEngine().generate(_config(repo="some/other"))

    def test_generation_names_a_missing_download(self) -> None:
        with (
            mock.patch.object(ltx2, "is_installed", return_value=True),
            mock.patch.object(ltx2, "_apple_silicon", return_value=True),
            mock.patch.object(ltx2, "_local_snapshot", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "not downloaded"):
                ltx2.Ltx2MlxVideoEngine().generate(_config())

    def test_the_probe_reports_each_reason_it_cannot_run(self) -> None:
        engine = ltx2.Ltx2MlxVideoEngine()
        with mock.patch.object(ltx2, "_apple_silicon", return_value=False):
            self.assertIn("Apple Silicon", engine.probe().message)
        with mock.patch.object(ltx2, "_apple_silicon", return_value=True), mock.patch.object(ltx2, "is_installed", return_value=False):
            self.assertFalse(engine.probe().realGenerationAvailable)
        with mock.patch.object(ltx2, "_apple_silicon", return_value=True), mock.patch.object(ltx2, "is_installed", return_value=True):
            self.assertTrue(engine.probe().realGenerationAvailable)

    def test_a_system_ffmpeg_needs_no_shim(self) -> None:
        with mock.patch("shutil.which", return_value="/opt/homebrew/bin/ffmpeg"):
            self.assertIsNone(ltx2._ffmpeg_path_dir())

    def test_the_child_cannot_download(self) -> None:
        self.assertEqual(ltx2.Ltx2MlxVideoEngine._child_env()["HF_HUB_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
