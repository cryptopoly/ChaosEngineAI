"""LTX-2.5 on diffusers (NVIDIA): catalog rows, access gate, 4-bit text-encoder branch, tier choice. Unit only (fakes)."""
from __future__ import annotations

import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from backend_service.catalog import VIDEO_MODEL_FAMILIES
from backend_service.catalog.video_ltx25 import LTX25_BF16_ID, LTX25_NF4_ID, LTX25_REPO
from backend_service.helpers.video_gating import gated_access_reason
from backend_service.video_runtime import ltx25_loader as loader
from backend_service.video_runtime import transformer_loaders
from backend_service.video_runtime.repos import (
    LTX2_DIFFUSERS_REPOS,
    PIPELINE_REGISTRY,
    _bnb_nf4_transformer_class_for_repo,
    is_ltx2_distilled_repo,
)


def _rows() -> dict[str, dict]:
    family = next(f for f in VIDEO_MODEL_FAMILIES if f["id"] == "ltx-2-5-diffusers")
    return {v["id"]: v for v in family["variants"]}


class _Torch:
    bfloat16 = "bf16"

    class cuda:  # noqa: N801
        @staticmethod
        def current_device() -> int:
            return 0


class CatalogTests(unittest.TestCase):
    def test_two_rows_share_the_gated_repo(self) -> None:
        rows = _rows()
        self.assertEqual(set(rows), {LTX25_NF4_ID, LTX25_BF16_ID})
        for row in rows.values():
            self.assertEqual(row["repo"], LTX25_REPO)
            self.assertIs(row["cudaOnly"], True)
            self.assertIs(row["gated"], True)
            self.assertEqual(row["footprintBasis"], "estimated")
            self.assertEqual(row["estimatedGenerationSecondsBasis"], "estimated")
            self.assertNotIn("footprintMeasuredOn", row)
            self.assertIn("US$10M", row["note"])
            self.assertIn("affiliates", row["note"])
            self.assertIn("Commercial Use Agreement", row["note"])
            self.assertIn("fine-tunes", row["note"])
        self.assertEqual(rows[LTX25_NF4_ID]["loadQuantization"], "nf4")
        self.assertNotIn("loadQuantization", rows[LTX25_BF16_ID])

    def test_footprints_are_the_estimates_asked_for(self) -> None:
        rows = _rows()
        self.assertEqual(rows[LTX25_NF4_ID]["runtimeFootprintGb"], 22.5)
        self.assertEqual(rows[LTX25_BF16_ID]["runtimeFootprintGb"], 76.0)
        self.assertIn("not measured", rows[LTX25_NF4_ID]["note"])

    def test_allow_patterns_are_the_distilled_pipeline(self) -> None:
        files = _rows()[LTX25_NF4_ID]["downloadFiles"]
        self.assertIn("transformer/*-of-00004.safetensors", files)
        self.assertIn("connectors/*-of-00002.safetensors", files)
        self.assertIn("text_encoder/*.safetensors", files)
        self.assertFalse([f for f in files if f.startswith(("transformer_full", "prompt_enhancer"))])

    def test_default_is_the_four_bit_row_and_the_mac_row_is_untouched(self) -> None:
        family = next(f for f in VIDEO_MODEL_FAMILIES if f["id"] == "ltx-2-5-diffusers")
        self.assertEqual(family["defaultVariantId"], LTX25_NF4_ID)
        mac = next(v for f in VIDEO_MODEL_FAMILIES for v in f["variants"] if v["id"] == "dgrauet/ltx-2.5-mlx-q4")
        self.assertEqual(mac["familyId"], "ltx-2")
        self.assertNotIn("cudaOnly", mac)

    def test_no_card_names_or_card_sizes_in_the_rows(self) -> None:
        text = json.dumps(_rows())
        self.assertIsNone(re.search(r"RTX|4090|3090|5090|GeForce|\bH100\b|\bA100\b", text))

    def test_runtime_registries_know_the_repo(self) -> None:
        self.assertIn(LTX25_REPO, LTX2_DIFFUSERS_REPOS)
        self.assertTrue(is_ltx2_distilled_repo(LTX25_REPO))
        self.assertEqual(PIPELINE_REGISTRY[LTX25_REPO]["class_name"], "LTX2Pipeline")
        for repo in (LTX25_REPO, "diffusers/LTX-2.3-Distilled-Diffusers", "diffusers/LTX-2.3-Diffusers"):
            self.assertEqual(_bnb_nf4_transformer_class_for_repo(repo), "LTX2VideoTransformer3DModel")
        self.assertIsNone(_bnb_nf4_transformer_class_for_repo("OzzyGT/LTX-2.3-Distilled-bnb-nf4"))  # already 4-bit


class GatingTests(unittest.TestCase):
    def test_blocked_without_weights_or_token(self) -> None:
        reason = gated_access_reason(_rows()[LTX25_NF4_ID], available_locally=False, token="")
        self.assertIn("gated", reason)
        self.assertIn("token", reason)

    def test_open_with_a_token_or_with_the_weights(self) -> None:
        row = _rows()[LTX25_NF4_ID]
        self.assertIsNone(gated_access_reason(row, available_locally=False, token="hf_x"))
        self.assertIsNone(gated_access_reason(row, available_locally=True, token=""))
        self.assertIsNone(gated_access_reason({"repo": "a/b"}, available_locally=False, token=""))

    def test_a_blocked_row_is_never_a_fit(self) -> None:
        from backend_service.helpers.fit import variant_fit
        from backend_service.helpers.hardware_class import HardwareClass

        hardware = HardwareClass(kind="cuda", bucket="cuda-24", label="NVIDIA, 24 GB", total_gb=24.0, usable_gb=23.0)
        row = dict(_rows()[LTX25_NF4_ID], accessBlockedReason="gated")
        self.assertEqual(variant_fit(row, "video", hardware)["status"], "unknown")
        self.assertEqual(variant_fit(_rows()[LTX25_NF4_ID], "video", hardware)["basis"], "estimated")

    def test_download_route_refuses_without_a_token(self) -> None:
        from fastapi import HTTPException

        from backend_service.routes import video as video_routes

        request = mock.MagicMock()
        with mock.patch("backend_service.helpers.huggingface._hf_token_value", return_value=""):
            with self.assertRaises(HTTPException) as caught:
                video_routes.download_video_model(request, mock.MagicMock(modelId=LTX25_NF4_ID))
        self.assertEqual(caught.exception.status_code, 403)

    def test_download_route_passes_the_gate_with_a_token(self) -> None:
        from backend_service.routes import video as video_routes

        request = mock.MagicMock()
        request.app.state.chaosengine.start_download.return_value = {"id": "d1"}
        with mock.patch("backend_service.helpers.huggingface._hf_token_value", return_value="hf_x"):
            video_routes.download_video_model(request, mock.MagicMock(modelId=LTX25_NF4_ID, repo=LTX25_REPO))
        request.app.state.chaosengine.start_download.assert_called()

    def test_catalog_row_reason_follows_the_token(self) -> None:
        row = _rows()[LTX25_NF4_ID]
        with mock.patch("backend_service.helpers.huggingface._hf_token_value", return_value=""):
            from backend_service.helpers.huggingface import _hf_token_value
            self.assertTrue(gated_access_reason(row, available_locally=False, token=_hf_token_value()))
        with mock.patch("backend_service.helpers.huggingface._hf_token_value", return_value="hf_x"):
            from backend_service.helpers.huggingface import _hf_token_value
            self.assertIsNone(gated_access_reason(row, available_locally=False, token=_hf_token_value()))


class TierTests(unittest.TestCase):
    def test_precision_by_free_vram(self) -> None:
        self.assertEqual(loader.text_encoder_precision(12.0), "nf4")
        self.assertEqual(loader.text_encoder_precision(24.0), "nf4")
        self.assertEqual(loader.text_encoder_precision(47.9), "nf4")
        self.assertEqual(loader.text_encoder_precision(48.0), "bf16")
        self.assertEqual(loader.text_encoder_precision(96.0), "bf16")

    def test_unified_memory_and_unreadable_free_memory(self) -> None:
        self.assertEqual(loader.text_encoder_precision(20.0, unified_memory=True), "bf16")
        self.assertEqual(loader.text_encoder_precision(None), "nf4")

    def test_threshold_can_be_configured(self) -> None:
        with mock.patch.dict("os.environ", {"CHAOSENGINE_LTX25_TE_BF16_MIN_GB": "30"}):
            self.assertEqual(loader.text_encoder_precision(32.0), "bf16")
        with mock.patch.dict("os.environ", {"CHAOSENGINE_LTX25_TE_BF16_MIN_GB": "junk"}):
            self.assertEqual(loader.text_encoder_precision(32.0), "nf4")


class _FakeBnbConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeEncoder:
    loaded: dict = {}

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.loaded = {"path": path, **kwargs}
        return cls()


def _snapshot(architecture: str | None = "Gemma4UnifiedForConditionalGeneration") -> str:
    root = Path(tempfile.mkdtemp())
    (root / "text_encoder").mkdir()
    config = {"architectures": [architecture]} if architecture else {}
    (root / "text_encoder" / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(root)


def _transformers(**attrs) -> types.ModuleType:
    module = types.ModuleType("transformers")
    module.__version__ = attrs.pop("version", "5.22.2")
    module.BitsAndBytesConfig = _FakeBnbConfig
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class QuantisationTests(unittest.TestCase):
    def test_config_skips_towers_embeddings_and_head_and_computes_in_bf16(self) -> None:
        with mock.patch.dict(sys.modules, {"transformers": _transformers()}), \
                mock.patch("importlib.util.find_spec", return_value=object()):
            config, note = loader.build_text_encoder_quantization_config(_Torch)
        self.assertIsNone(note)
        self.assertIs(config.kwargs["load_in_4bit"], True)
        self.assertEqual(config.kwargs["bnb_4bit_quant_type"], "nf4")
        self.assertEqual(config.kwargs["bnb_4bit_compute_dtype"], "bf16")
        skip = set(config.kwargs["llm_int8_skip_modules"])
        self.assertTrue({"lm_head", "embed_tokens", "vision_tower", "audio_tower", "embed_vision", "embed_audio"} <= skip)

    def test_missing_bitsandbytes_is_a_note(self) -> None:
        with mock.patch("importlib.util.find_spec", return_value=None):
            config, note = loader.build_text_encoder_quantization_config(_Torch)
        self.assertIsNone(config)
        self.assertIn("bitsandbytes", note)

    def test_loads_with_the_class_the_config_names(self) -> None:
        mods = {"transformers": _transformers(Gemma4UnifiedForConditionalGeneration=_FakeEncoder)}
        with mock.patch.dict(sys.modules, mods), mock.patch("importlib.util.find_spec", return_value=object()):
            encoder, note = loader.try_load_ltx25_text_encoder_4bit(_snapshot(), _Torch, "cuda")
        self.assertIsInstance(encoder, _FakeEncoder)
        self.assertIn("NF4", note)
        self.assertTrue(_FakeEncoder.loaded["path"].endswith("text_encoder"))
        self.assertEqual(_FakeEncoder.loaded["dtype"], "bf16")  # transformers 5 spelling
        self.assertIs(_FakeEncoder.loaded["local_files_only"], True)
        self.assertIsInstance(_FakeEncoder.loaded["quantization_config"], _FakeBnbConfig)

    def test_older_transformers_gets_torch_dtype(self) -> None:
        mods = {"transformers": _transformers(version="4.57.1", Gemma4UnifiedForConditionalGeneration=_FakeEncoder)}
        with mock.patch.dict(sys.modules, mods), mock.patch("importlib.util.find_spec", return_value=object()):
            loader.try_load_ltx25_text_encoder_4bit(_snapshot(), _Torch, "cuda")
        self.assertEqual(_FakeEncoder.loaded["torch_dtype"], "bf16")
        self.assertNotIn("dtype", _FakeEncoder.loaded)

    def test_class_missing_from_transformers_falls_back_with_a_note(self) -> None:
        with mock.patch.dict(sys.modules, {"transformers": _transformers()}), \
                mock.patch("importlib.util.find_spec", return_value=object()):
            encoder, note = loader.try_load_ltx25_text_encoder_4bit(_snapshot(), _Torch, "cuda")
        self.assertIsNone(encoder)
        self.assertIn("Gemma4UnifiedForConditionalGeneration", note)
        self.assertIn("stays bf16", note)

    def test_no_architecture_non_cuda_and_failed_load_fall_back(self) -> None:
        mods = {"transformers": _transformers(Gemma4UnifiedForConditionalGeneration=_FakeEncoder)}
        with mock.patch.dict(sys.modules, mods), mock.patch("importlib.util.find_spec", return_value=object()):
            self.assertIn("no architecture", loader.try_load_ltx25_text_encoder_4bit(_snapshot(None), _Torch, "cuda")[1])
            self.assertIn("needs CUDA", loader.try_load_ltx25_text_encoder_4bit(_snapshot(), _Torch, "cpu")[1])
            with mock.patch.object(_FakeEncoder, "from_pretrained", side_effect=RuntimeError("boom")):
                encoder, note = loader.try_load_ltx25_text_encoder_4bit(_snapshot(), _Torch, "cuda")
        self.assertIsNone(encoder)
        self.assertIn("boom", note)


class ComponentTests(unittest.TestCase):
    def _load(self, free_gb, unified=False, transformer=("T", "nf4 ok"), encoder=("E", "enc ok")):
        with mock.patch.object(loader, "try_load_ltx25_text_encoder_4bit", return_value=encoder) as te:
            result = loader.load_ltx25_components(
                LTX25_REPO, "/snap", _Torch, "cuda", free_gb=free_gb, unified_memory=unified,
                load_transformer=lambda *a: transformer,
            )
        return result, te

    def test_small_cards_get_both_in_4bit_and_no_prompt_enhancer(self) -> None:
        for free in (12.0, 24.0):
            (kwargs, notes), te = self._load(free)
            self.assertEqual(kwargs, {"prompt_enhancer": None, "transformer": "T", "text_encoder": "E"})
            te.assert_called_once()
            self.assertEqual(notes, ["nf4 ok", "enc ok"])

    def test_big_cards_and_unified_memory_keep_the_text_encoder_bf16(self) -> None:
        for free, unified in ((48.0, False), (96.0, False), (10.0, True)):
            (kwargs, notes), te = self._load(free, unified)
            self.assertNotIn("text_encoder", kwargs)
            self.assertEqual(kwargs["transformer"], "T")
            te.assert_not_called()
            self.assertTrue(any("kept bf16" in n for n in notes))

    def test_text_encoder_fallback_keeps_the_note_and_the_transformer(self) -> None:
        (kwargs, notes), _ = self._load(24.0, encoder=(None, "stays bf16 because x"))
        self.assertNotIn("text_encoder", kwargs)
        self.assertIn("stays bf16 because x", notes)

    def test_a_transformer_that_will_not_quantise_stops_the_load(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            self._load(24.0, transformer=(None, "bitsandbytes package missing"))
        self.assertIn("bitsandbytes", str(caught.exception))


class WindowsKwargsTests(unittest.TestCase):
    def test_nf4_transformer_load_passes_the_platform_kwargs(self) -> None:
        captured: dict = {}

        class _Model:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                captured.update(kwargs)
                return cls()

        diffusers = types.ModuleType("diffusers")
        diffusers.BitsAndBytesConfig = _FakeBnbConfig
        diffusers.LTX2VideoTransformer3DModel = _Model
        for platform, expected in (("win32", True), ("linux", None)):
            captured.clear()
            with mock.patch.dict(sys.modules, {"diffusers": diffusers}), \
                    mock.patch("importlib.util.find_spec", return_value=object()), \
                    mock.patch("backend_service.helpers.malloc_release.sys.platform", platform):
                model, _ = transformer_loaders.try_load_bnb_nf4_transformer(LTX25_REPO, "/snap", _Torch, "cuda")
            self.assertIsInstance(model, _Model)
            self.assertEqual(captured.get("disable_mmap"), expected)
            self.assertEqual(captured["subfolder"], "transformer")


class SnapshotTests(unittest.TestCase):
    def test_a_snapshot_without_the_prompt_enhancer_is_complete(self) -> None:
        from backend_service.image_runtime import validate_local_diffusers_snapshot

        root = Path(tempfile.mkdtemp())
        index = {
            "_class_name": "LTX2Pipeline",
            "transformer": ["diffusers", "LTX2VideoTransformer3DModel"],
            "prompt_enhancer": ["transformers", "Gemma4ForConditionalGeneration"],
        }
        (root / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
        (root / "transformer").mkdir()
        (root / "transformer" / "config.json").write_text("{}", encoding="utf-8")
        self.assertIsNone(validate_local_diffusers_snapshot(root, LTX25_REPO))
        self.assertIn("prompt_enhancer", validate_local_diffusers_snapshot(root, "someone/else") or "")


if __name__ == "__main__":
    unittest.main()
