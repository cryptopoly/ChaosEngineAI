"""Library discovery marks checkpoints that cannot chat as modelType "other".

The Chat tab's My Models list showed Trellis / TripoSG components, DINO and text
encoders, Kokoro and bare .safetensors files as chat models, because anything not
recognised as image / video / draft was "text".
"""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from backend_service.helpers.discovery import _discover_local_models
from backend_service.helpers.model_classifier import _gguf_architecture


def _gguf(path: Path, architecture: str) -> None:
    key = b"general.architecture"
    value = architecture.encode()
    path.write_bytes(
        b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
        + struct.pack("<Q", len(key)) + key + struct.pack("<I", 8)
        + struct.pack("<Q", len(value)) + value
    )


class NonChatClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="ce-nonchat-"))

    def _model(self, relative: str, config: dict | None = None, files: tuple[str, ...] = (), chat_template: bool = False) -> None:
        folder = self.root / relative
        folder.mkdir(parents=True, exist_ok=True)
        if config is not None:
            (folder / "config.json").write_text(json.dumps(config))
        (folder / "tokenizer.json").write_text("{}")
        if chat_template:
            (folder / "chat_template.jinja").write_text("{{ messages }}")
        for name in files:
            (folder / name).write_bytes(b"x" * 16)

    def _types(self) -> dict[str, str]:
        items = _discover_local_models([{"id": "t", "label": "t", "path": str(self.root), "enabled": True}])
        return {item["name"]: item["modelType"] for item in items}

    def test_language_models_stay_text(self) -> None:
        self._model("llama", {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}, ("model.safetensors",))
        self._model("vl", {"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}, ("model.safetensors",))
        self._model("kolibri", {"model_type": "kolibri1", "architectures": ["Kolibri1ForCausalLM"]}, ("model.safetensors",))
        types = self._types()
        self.assertEqual({types["llama"], types["vl"], types["kolibri"]}, {"text"})

    def test_an_unusual_architecture_with_a_chat_template_stays_text(self) -> None:
        self._model("odd", {"model_type": "odd", "architectures": ["WeirdModel"]}, ("model.safetensors",), chat_template=True)
        self.assertEqual(self._types()["odd"], "text")

    def test_encoders_and_voice_models_are_other(self) -> None:
        self._model("dino", {"model_type": "dinov2", "architectures": ["Dinov2Model"]}, ("model.safetensors",))
        self._model("speech", {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}, ("model.safetensors",))
        self._model("tts", {"istftnet": {"x": 1}, "dim_in": 64}, ("kokoro.safetensors",))
        self._model("diffusers-part", {"_class_name": "AutoencoderKL"}, ("model.safetensors",))
        types = self._types()
        self.assertEqual({types[n] for n in ("dino", "speech", "tts", "diffusers-part")}, {"other"})

    def test_a_bare_safetensors_file_is_other(self) -> None:
        (self.root / "ss_dec_conv3d_16l8_fp16.safetensors").write_bytes(b"x" * 16)
        self.assertEqual(self._types()["ss_dec_conv3d_16l8_fp16"], "other")

    def test_a_three_d_tool_tree_is_other_even_without_a_config(self) -> None:
        folder = self.root / "trellis2" / "shape"
        folder.mkdir(parents=True)
        (folder / "dit.gguf").write_bytes(b"x" * 16)
        self.assertEqual(self._types()["shape"], "other")

    def test_a_gguf_decides_by_its_architecture(self) -> None:
        for name, arch in (("chat-gguf", "qwen3"), ("image-gguf", "flux"), ("embed-gguf", "bert")):
            folder = self.root / name
            folder.mkdir()
            _gguf(folder / "m.gguf", arch)
        types = self._types()
        self.assertEqual(types["chat-gguf"], "text")
        self.assertEqual(types["image-gguf"], "other")
        self.assertEqual(types["embed-gguf"], "other")

    def test_the_gguf_header_reader(self) -> None:
        path = self.root / "a.gguf"
        _gguf(path, "llama")
        self.assertEqual(_gguf_architecture(path), "llama")
        path.write_bytes(b"not a gguf at all")
        self.assertIsNone(_gguf_architecture(path))
        self.assertIsNone(_gguf_architecture(self.root / "missing.gguf"))


if __name__ == "__main__":
    unittest.main()


class Fp8CompressedTensorsOnMacTests(unittest.TestCase):
    def test_fp8_checkpoint_is_unsupported_on_a_mac_only(self) -> None:
        from unittest import mock

        from backend_service.helpers import quantization

        config = {"quantization_config": {"quant_method": "compressed-tensors", "format": "float-quantized"}}
        with mock.patch.object(quantization.sys, "platform", "darwin"):
            self.assertIn("NVIDIA", quantization._unsupported_mlx_quantization_reason(config) or "")
        with mock.patch.object(quantization.sys, "platform", "linux"):
            self.assertIsNone(quantization._unsupported_mlx_quantization_reason(config))
        with mock.patch.object(quantization.sys, "platform", "darwin"):
            self.assertIsNone(quantization._unsupported_mlx_quantization_reason(
                {"quantization_config": {"quant_method": "compressed-tensors", "format": "pack-quantized"}}
            ))
