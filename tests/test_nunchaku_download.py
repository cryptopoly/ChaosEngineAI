"""The FLUX.1 Nunchaku rows download their 4-bit pack plus only the base-repo parts they read.

Both Nunchaku rows share their ``repo`` with the ordinary bf16 FLUX.1 rows, and downloads were keyed by
that repo: choosing the Nunchaku row fetched the whole base repo (24 GB of bf16 transformer shards it never
loads) and never the pack, so generation fell back silently. The row now downloads under its pack repo and
brings the base repo's text encoders, tokenizers, VAE, scheduler and transformer config as a companion.
"""

from __future__ import annotations

import fnmatch
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import backend_service.app  # noqa: F401 — loads the routes in their usual order (routes.images imports app)
from backend_service.catalog import IMAGE_MODEL_FAMILIES
from backend_service.helpers import image_validation
from backend_service.helpers.companions import NUNCHAKU_BASE_ALLOW_PATTERNS, companion_downloads
from backend_service.helpers.images import _image_repo_allow_patterns
from backend_service.models import DownloadModelRequest
from backend_service.routes import images as image_routes
from backend_service.state import ChaosEngineState

GB = 1024 ** 3
SCHNELL = "black-forest-labs/FLUX.1-schnell"
DEV = "black-forest-labs/FLUX.1-dev"
SCHNELL_PACK = "mit-han-lab/svdq-int4-flux.1-schnell"
DEV_PACK = "mit-han-lab/svdq-int4-flux.1-dev"

# The FLUX.1 base repo layout (file names from the Hub listing).
_BASE_FILES = [
    "model_index.json",
    "README.md",
    "LICENSE.md",
    "flux1-schnell.safetensors",
    "ae.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model-00001-of-00002.safetensors",
    "text_encoder_2/model-00002-of-00002.safetensors",
    "text_encoder_2/model.safetensors.index.json",
    "tokenizer/vocab.json",
    "tokenizer_2/spiece.model",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]


def _variant(variant_id: str) -> dict:
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["id"] == variant_id:
                return variant
    raise AssertionError(variant_id)


def _fetched(patterns: list[str]) -> set[str]:
    return {name for name in _BASE_FILES if any(fnmatch.fnmatch(name, p) for p in patterns)}


class NunchakuCompanionTests(unittest.TestCase):
    def test_the_pack_brings_the_base_repo_without_its_transformer_shards(self) -> None:
        for pack, base in ((SCHNELL_PACK, SCHNELL), (DEV_PACK, DEV)):
            self.assertEqual(companion_downloads(pack), [(base, NUNCHAKU_BASE_ALLOW_PATTERNS)])
        fetched = _fetched(NUNCHAKU_BASE_ALLOW_PATTERNS)
        self.assertIn("transformer/config.json", fetched)
        self.assertIn("text_encoder_2/model-00002-of-00002.safetensors", fetched)
        self.assertIn("text_encoder_2/model.safetensors.index.json", fetched)
        self.assertIn("vae/diffusion_pytorch_model.safetensors", fetched)
        self.assertIn("scheduler/scheduler_config.json", fetched)
        self.assertIn("tokenizer_2/spiece.model", fetched)
        self.assertFalse(any(name.startswith("transformer/diffusion_pytorch_model") for name in fetched))
        self.assertNotIn("flux1-schnell.safetensors", fetched)
        self.assertNotIn("ae.safetensors", fetched)

    def test_the_ordinary_base_repo_download_is_unchanged(self) -> None:
        self.assertEqual(companion_downloads(SCHNELL), [])
        self.assertEqual(companion_downloads(DEV), [])
        patterns = _image_repo_allow_patterns(SCHNELL)
        self.assertIn("transformer/**", patterns)
        self.assertIsNone(_image_repo_allow_patterns(SCHNELL_PACK))

    def test_rows_size_the_pack_plus_the_base_parts(self) -> None:
        for variant_id in ("black-forest-labs/FLUX.1-schnell-nunchaku-int4", "black-forest-labs/FLUX.1-dev-nunchaku-int4"):
            self.assertEqual(_variant(variant_id)["sizeGb"], 16.7)


class NunchakuProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.state = ChaosEngineState(
            settings_path=tmp / "settings.json",
            benchmarks_path=tmp / "benchmarks.json",
            chat_sessions_path=tmp / "chats.json",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_total_covers_pack_and_base_parts_and_progress_counts_both(self) -> None:
        class NoThread:
            def __init__(self, *, target=None, daemon=None):
                pass

            def start(self):
                pass

        calls: list[tuple[str, object]] = []

        def cached(repo, patterns=None):
            calls.append((repo, patterns))
            if repo == SCHNELL_PACK:
                return 2 * GB
            if repo == SCHNELL and patterns == NUNCHAKU_BASE_ALLOW_PATTERNS:
                return 3 * GB
            return 0

        with mock.patch("backend_service.helpers.huggingface._hf_repo_downloaded_bytes", side_effect=cached), \
             mock.patch("backend_service.helpers.huggingface._hf_repo_preflight_size_gb", return_value=6.29), \
             mock.patch("backend_service.helpers.huggingface._hub_repo_files", return_value={"files": []}), \
             mock.patch("backend_service.state.downloads.threading.Thread", side_effect=NoThread):
            row = dict(self.state.start_download(SCHNELL_PACK))
        # 16.7 decimal GB in the GiB the bar counts, not the 6.29 GiB pack alone.
        self.assertAlmostEqual(row["totalGb"], 16.7e9 / GB, places=1)
        self.assertAlmostEqual(row["downloadedGb"], 5.0, places=1)
        self.assertIn((SCHNELL, NUNCHAKU_BASE_ALLOW_PATTERNS), calls)


def _write_base_snapshot(root: Path, *, with_transformer_weights: bool) -> None:
    index = {
        "_class_name": "FluxPipeline",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    (root / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
    configs = {
        "scheduler": "scheduler_config.json",
        "tokenizer": "tokenizer_config.json",
        "tokenizer_2": "tokenizer_config.json",
    }
    for component in ("scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "transformer", "vae"):
        (root / component).mkdir()
        (root / component / configs.get(component, "config.json")).write_text("{}", encoding="utf-8")
    if with_transformer_weights:
        (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"0")


class NunchakuInstalledTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.base = tmp / "base"
        self.pack = tmp / "pack"
        self.base.mkdir()
        self.snapshots: dict[str, Path] = {}
        patcher = mock.patch.object(image_validation, "_hf_repo_snapshot_dir", side_effect=self.snapshots.get)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self.nunchaku = _variant("black-forest-labs/FLUX.1-schnell-nunchaku-int4")
        self.bf16 = _variant(SCHNELL)

    def test_the_nunchaku_row_needs_the_base_parts_and_the_pack(self) -> None:
        _write_base_snapshot(self.base, with_transformer_weights=False)
        self.snapshots[SCHNELL] = self.base
        self.assertFalse(image_validation._image_variant_available_locally(self.nunchaku, []))
        self.assertIsNotNone(image_validation._nunchaku_download_validation_error(SCHNELL_PACK))
        self.pack.mkdir()
        (self.pack / "transformer_blocks.safetensors").write_bytes(b"0")
        self.snapshots[SCHNELL_PACK] = self.pack
        self.assertTrue(image_validation._image_variant_available_locally(self.nunchaku, []))
        self.assertIsNone(image_validation._nunchaku_download_validation_error(SCHNELL_PACK))

    def test_the_pack_alone_is_not_installed(self) -> None:
        self.pack.mkdir()
        (self.pack / "transformer_blocks.safetensors").write_bytes(b"0")
        self.snapshots[SCHNELL_PACK] = self.pack
        self.assertFalse(image_validation._image_variant_available_locally(self.nunchaku, []))

    def test_the_bf16_row_is_not_installed_by_the_nunchaku_download(self) -> None:
        _write_base_snapshot(self.base, with_transformer_weights=False)
        self.snapshots[SCHNELL] = self.base
        library = [{"name": SCHNELL, "path": str(self.base)}]
        self.assertFalse(image_validation._image_variant_available_locally(self.bf16, library))
        (self.base / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"0")
        self.assertTrue(image_validation._image_variant_available_locally(self.bf16, library))


    def test_generation_is_refused_until_pack_and_base_parts_are_on_disk(self) -> None:
        """It answered HTTP 200 with a placeholder picture and a raw Hub error (RTX 4090, 2026-11-20)."""
        block = image_validation._nunchaku_generation_block
        self.assertIn("is not downloaded", block(self.nunchaku, None))
        self.assertIn("still downloading", block(self.nunchaku, "downloading"))
        _write_base_snapshot(self.base, with_transformer_weights=False)
        self.snapshots[SCHNELL] = self.base
        self.assertIn("incomplete", block(self.nunchaku, "failed"))
        self.pack.mkdir()
        self.snapshots[SCHNELL_PACK] = self.pack
        (self.pack / "transformer_blocks.safetensors").write_bytes(b"")
        self.assertIn("incomplete", block(self.nunchaku, "completed"), "an empty pack file is not a pack")
        (self.pack / "transformer_blocks.safetensors").write_bytes(b"0")
        with mock.patch("backend_service.inference.accelerators.nunchaku_available", return_value=True):
            self.assertIsNone(block(self.nunchaku, "completed"))
        self.assertIsNone(block(self.bf16, None), "only Nunchaku rows are checked here")

    def test_files_on_disk_without_the_package_are_refused_with_the_way_to_install(self) -> None:
        """FU-478: the load failed after it started, saying it fell back and then that it could not."""
        block = image_validation._nunchaku_generation_block
        _write_base_snapshot(self.base, with_transformer_weights=False)
        self.snapshots[SCHNELL] = self.base
        self.pack.mkdir()
        self.snapshots[SCHNELL_PACK] = self.pack
        (self.pack / "transformer_blocks.safetensors").write_bytes(b"0")
        with mock.patch("backend_service.inference.accelerators.nunchaku_available", return_value=False):
            reason = block(self.nunchaku, "completed")
        self.assertIn("needs the Nunchaku package, which is not installed", reason)
        self.assertIn("Performance boosters", reason)
        self.assertNotIn("Falling back", reason)

    def test_a_missing_download_points_to_image_models(self) -> None:
        self.assertIn("Download it on Image Models first", image_validation._nunchaku_generation_block(self.nunchaku, None))

class NunchakuRouteTests(unittest.TestCase):
    def _request(self, state) -> SimpleNamespace:
        return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(chaosengine=state)))

    def test_download_of_the_pack_checks_both_parts_and_the_base_download_is_plain(self) -> None:
        state = mock.Mock()
        state.start_download.return_value = {"repo": SCHNELL_PACK}
        image_routes.download_image_model(self._request(state), DownloadModelRequest(repo=SCHNELL_PACK))
        state.start_download.assert_called_once_with(
            SCHNELL_PACK, validation_error_fn=image_validation._nunchaku_download_validation_error,
        )
        state.start_download.reset_mock()
        image_routes.download_image_model(self._request(state), DownloadModelRequest(repo=SCHNELL))
        state.start_download.assert_called_once_with(SCHNELL)

    def test_status_lists_the_pack_download(self) -> None:
        state = mock.Mock()
        state.download_status.return_value = [{"repo": SCHNELL_PACK}, {"repo": "someone/else"}]
        downloads = image_routes.image_download_status(self._request(state))["downloads"]
        self.assertEqual([item["repo"] for item in downloads], [SCHNELL_PACK])

    def _delete(self, *, base_has_transformer: bool, base_busy: bool = False) -> list[str]:
        state = mock.Mock()
        state._downloads = {SCHNELL: {"state": "downloading"}} if base_busy else {}
        state.delete_download.return_value = {"repo": SCHNELL_PACK, "state": "deleted"}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "transformer").mkdir()
            if base_has_transformer:
                (base / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"0")
            with mock.patch.object(image_routes, "_hf_repo_snapshot_dir", return_value=base):
                image_routes.delete_image_download(self._request(state), DownloadModelRequest(repo=SCHNELL_PACK))
        return [call.args[0] for call in state.delete_download.call_args_list]

    def test_delete_removes_the_base_parts_the_variant_brought(self) -> None:
        self.assertEqual(self._delete(base_has_transformer=False), [SCHNELL_PACK, SCHNELL])

    def test_delete_keeps_a_base_repo_the_bf16_row_uses(self) -> None:
        self.assertEqual(self._delete(base_has_transformer=True), [SCHNELL_PACK])
        self.assertEqual(self._delete(base_has_transformer=False, base_busy=True), [SCHNELL_PACK])


    def test_generate_answers_409_without_running_when_the_pack_is_missing(self) -> None:
        from fastapi import HTTPException

        from backend_service.models import ImageGenerationRequest

        state = mock.Mock()
        state._downloads = {}
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(chaosengine=state)), state=SimpleNamespace())
        body = ImageGenerationRequest(modelId="black-forest-labs/FLUX.1-schnell-nunchaku-int4", prompt="an apple")
        with mock.patch.object(image_validation, "_hf_repo_snapshot_dir", return_value=None),              mock.patch.object(image_routes, "_generate_image_artifacts") as run:
            with self.assertRaises(HTTPException) as caught:
                image_routes._generate_image_locked(request, body)
        self.assertEqual(caught.exception.status_code, 409)
        self.assertIn("is not downloaded", caught.exception.detail["message"])
        run.assert_not_called()


class NunchakuNoPlaceholderTests(unittest.TestCase):
    def test_a_failed_nunchaku_load_raises_instead_of_a_placeholder_picture(self) -> None:
        from backend_service.image_runtime import ImageGenerationConfig, ImageRuntimeManager

        manager = ImageRuntimeManager()
        config = ImageGenerationConfig(
            modelId="black-forest-labs/FLUX.1-schnell-nunchaku-int4", modelName="FLUX.1 Schnell Nunchaku",
            repo=SCHNELL, prompt="an apple", negativePrompt="", width=512, height=512, steps=2, guidance=1.0,
            batchSize=1, seed=1, nunchakuRepo=SCHNELL_PACK,
        )
        ready = SimpleNamespace(realGenerationAvailable=True, to_dict=lambda: {}, device="cuda",
                                pythonExecutable=None, loadedModelRepo=None)
        missing = RuntimeError("This Nunchaku model has no standard transformer to fall back to.")
        with mock.patch.object(manager._diffusers, "probe", return_value=ready),              mock.patch("backend_service.image_runtime._MODEL_THREAD.run", side_effect=missing),              mock.patch.object(manager._placeholder, "generate") as placeholder:
            with self.assertRaises(RuntimeError):
                manager.generate(config)
            placeholder.assert_not_called()
        not_ready = SimpleNamespace(realGenerationAvailable=False, message="torch is not installed", to_dict=lambda: {})
        with mock.patch.object(manager._diffusers, "probe", return_value=not_ready),              mock.patch.object(manager._placeholder, "generate") as placeholder:
            with self.assertRaisesRegex(RuntimeError, "torch is not installed"):
                manager.generate(config)
            placeholder.assert_not_called()

class NunchakuFallbackNoteTests(unittest.TestCase):
    def test_a_missing_pack_is_said_in_plain_words(self) -> None:
        from huggingface_hub.errors import LocalEntryNotFoundError

        from backend_service.image_runtime import transformer_loaders

        fake = types.ModuleType("nunchaku")
        fake.NunchakuFluxTransformer2dModel = object
        with mock.patch("backend_service.inference.accelerators.nunchaku_available", return_value=True), \
             mock.patch.dict(sys.modules, {"nunchaku": fake}), \
             mock.patch("huggingface_hub.snapshot_download", side_effect=LocalEntryNotFoundError("pass local_files_only=False")):
            transformer, note = transformer_loaders.try_load_nunchaku_transformer(SCHNELL, SCHNELL_PACK, None, None)
        self.assertIsNone(transformer)
        self.assertIn("not downloaded", note)
        self.assertNotIn("local_files_only", note)


if __name__ == "__main__":
    unittest.main()
