"""GGUF routing reads the files, never "gguf" in a folder or repo name.

An MLX folder a user calls ``my-gguf-notes-mlx`` used to load on llama.cpp
and fail with "No .gguf weights found". These tests pin the content rules:
a ``.gguf`` file is GGUF wherever it lives, a folder with ``config.json``
and safetensors is a checkpoint whatever its name, an HF-cache GGUF repo
still loads on llama.cpp, a mixed folder is a checkpoint unless llama.cpp
is asked for, and a folder with no weights names what it found.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers.discovery import _discover_local_models
from backend_service.helpers.weights_format import has_gguf_magic, looks_like_gguf, weights_format
from backend_service.inference import (
    BackendCapabilities,
    LlamaCppEngine,
    MLXWorkerEngine,
    RuntimeController,
    _looks_like_gguf,
    _resolve_gguf_path,
)
from backend_service.state import ChaosEngineState
from tests.test_backend_service import fake_system_snapshot

GGUF_BYTES = b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * 56


def _mlx_folder(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "quantization": {"bits": 4, "group_size": 64}}), encoding="utf-8"
    )
    (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"x" * 64)
    return folder


def _hf_cache_repo(hub: Path, repo: str, files: dict[str, bytes]) -> Path:
    """An HF cache folder: blobs under blobs/, named links under snapshots/<rev>/."""
    repo_dir = hub / f"models--{repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo_dir / "blobs").mkdir()
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    for index, (name, payload) in enumerate(files.items()):
        blob = repo_dir / "blobs" / f"blob{index}"
        blob.write_bytes(payload)
        (snapshot / name).symlink_to(blob)
    return repo_dir


def _capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        pythonExecutable=sys.executable,
        mlxAvailable=True,
        mlxLmAvailable=True,
        mlxUsable=True,
        ggufAvailable=True,
        llamaServerPath="/usr/bin/true",
    )


def _controller() -> RuntimeController:
    with mock.patch("backend_service.inference.get_backend_capabilities", return_value=_capabilities()):
        controller = RuntimeController()
    controller.capabilities = _capabilities()
    return controller


class _TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.hub = self.root / "hub"
        self.hub.mkdir()
        patcher = mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.hub)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _state(self, library: list[dict]) -> ChaosEngineState:
        return ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=lambda: library,
            settings_path=self.root / "settings.json",
            benchmarks_path=self.root / "benchmarks.json",
            chat_sessions_path=self.root / "sessions.json",
        )

    def _library(self, models_root: Path) -> list[dict]:
        return _discover_local_models([{"label": "Models", "path": str(models_root), "enabled": True, "source": "user"}])


class WeightsFormatTests(_TempCase):
    def test_mlx_folder_with_gguf_in_its_name_is_safetensors(self) -> None:
        folder = _mlx_folder(self.root / "my-gguf-notes-mlx")
        self.assertEqual(weights_format(folder), "safetensors")
        self.assertFalse(looks_like_gguf(str(folder)))
        self.assertFalse(_looks_like_gguf(str(folder)))

    def test_gguf_file_in_a_folder_named_mlx_is_gguf(self) -> None:
        folder = self.root / "mlx"
        folder.mkdir()
        model = folder / "Qwen3-0.6B-Q4_K_M.gguf"
        model.write_bytes(GGUF_BYTES)
        self.assertEqual(weights_format(model), "gguf")
        self.assertEqual(weights_format(folder), "gguf")
        self.assertTrue(looks_like_gguf(str(model)))
        self.assertTrue(looks_like_gguf(str(folder)))
        self.assertTrue(has_gguf_magic(model))

    def test_hf_cache_gguf_repo_is_gguf_by_path_and_by_repo_id(self) -> None:
        repo = "lmstudio-community/Qwen3-0.6B-GGUF"
        repo_dir = _hf_cache_repo(self.hub, repo, {"Qwen3-0.6B-Q4_K_M.gguf": GGUF_BYTES})
        self.assertEqual(weights_format(repo_dir), "gguf")
        self.assertTrue(looks_like_gguf(str(repo_dir)))
        self.assertTrue(looks_like_gguf(repo))
        self.assertTrue(_resolve_gguf_path(str(repo_dir), None).endswith("Qwen3-0.6B-Q4_K_M.gguf"))

    def test_hf_cache_mlx_repo_with_gguf_in_its_name_is_not_gguf(self) -> None:
        repo = "someone/gguf-free-Qwen3-0.6B-mlx-4bit"
        _hf_cache_repo(self.hub, repo, {"config.json": b"{}", "model.safetensors": b"x" * 64})
        self.assertFalse(looks_like_gguf(repo))

    def test_mixed_folder_is_a_checkpoint_but_llama_cpp_still_finds_its_gguf(self) -> None:
        folder = _mlx_folder(self.root / "mixed")
        (folder / "extra-Q4_K_M.gguf").write_bytes(GGUF_BYTES)
        self.assertEqual(weights_format(folder), "safetensors")
        self.assertFalse(looks_like_gguf(str(folder)))
        self.assertTrue(_resolve_gguf_path(str(folder), None).endswith("extra-Q4_K_M.gguf"))

    def test_gguf_only_folder_with_safetensors_sidecar_but_no_config_stays_gguf(self) -> None:
        folder = self.root / "Some-GGUF"
        folder.mkdir()
        (folder / "model-Q4_K_M.gguf").write_bytes(GGUF_BYTES)
        (folder / "adapter.safetensors").write_bytes(b"x")
        self.assertEqual(weights_format(folder), "gguf")

    def test_remote_repo_id_with_nothing_on_disk_falls_back_to_its_name(self) -> None:
        self.assertTrue(looks_like_gguf("ggml-org/Qwen3-1.7B-GGUF"))
        self.assertFalse(looks_like_gguf("mlx-community/Qwen3-1.7B-4bit"))

    def test_missing_local_path_with_gguf_in_name_is_not_gguf(self) -> None:
        self.assertFalse(looks_like_gguf(str(self.root / "nope-gguf-folder")))
        self.assertTrue(looks_like_gguf(str(self.root / "nope" / "model.gguf")))

    def test_magic_check(self) -> None:
        pointer = self.root / "pointer.gguf"
        pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        self.assertFalse(has_gguf_magic(pointer))
        self.assertIsNone(has_gguf_magic(self.root / "absent.gguf"))


class DiscoveryTests(_TempCase):
    def test_library_formats_come_from_files(self) -> None:
        models = self.root / "Models"
        _mlx_folder(models / "my-gguf-notes-mlx")
        (models / "mlx").mkdir(parents=True)
        (models / "mlx" / "tiny-Q4_K_M.gguf").write_bytes(GGUF_BYTES)
        mixed = _mlx_folder(models / "mixed-model")
        (mixed / "extra-Q4_K_M.gguf").write_bytes(GGUF_BYTES)

        by_name = {entry["name"]: entry for entry in self._library(models)}
        self.assertEqual(by_name["my-gguf-notes-mlx"]["format"], "MLX")
        self.assertEqual(by_name["my-gguf-notes-mlx"]["backend"], "mlx")
        self.assertEqual(by_name["mixed-model"]["format"], "MLX")
        gguf_entries = [e for e in by_name.values() if e["format"] == "GGUF"]
        self.assertTrue(gguf_entries, by_name)
        self.assertTrue(all(e["backend"] == "llama.cpp" for e in gguf_entries))

    def test_hf_cache_gguf_repo_is_listed_as_gguf(self) -> None:
        _hf_cache_repo(self.hub, "lmstudio-community/Qwen3-0.6B-GGUF", {"Qwen3-0.6B-Q4_K_M.gguf": GGUF_BYTES})
        entries = self._library(self.hub)
        entry = next(e for e in entries if e["name"] == "lmstudio-community/Qwen3-0.6B-GGUF")
        self.assertEqual(entry["format"], "GGUF")
        self.assertEqual(entry["backend"], "llama.cpp")


class ResolveModelTargetTests(_TempCase):
    def test_mlx_folder_named_gguf_stays_on_mlx_for_auto_and_explicit_mlx(self) -> None:
        models = self.root / "Models"
        folder = _mlx_folder(models / "my-gguf-notes-mlx")
        state = self._state(self._library(models))
        for backend in ("auto", "mlx"):
            target, resolved = state._resolve_model_target(model_ref="my-gguf-notes-mlx", path=str(folder), backend=backend)
            self.assertEqual(resolved, "mlx", backend)
            self.assertEqual(target, str(folder))

    def test_gguf_file_in_mlx_folder_goes_to_llama_cpp(self) -> None:
        folder = self.root / "mlx"
        folder.mkdir()
        model = folder / "tiny-Q4_K_M.gguf"
        model.write_bytes(GGUF_BYTES)
        state = self._state([])
        target, resolved = state._resolve_model_target(model_ref="tiny", path=str(model), backend="auto")
        self.assertEqual((target, resolved), (str(model), "llama.cpp"))

    def test_hf_cache_gguf_library_entry_goes_to_llama_cpp_from_its_folder(self) -> None:
        repo = "lmstudio-community/Qwen3-0.6B-GGUF"
        repo_dir = _hf_cache_repo(self.hub, repo, {"Qwen3-0.6B-Q4_K_M.gguf": GGUF_BYTES})
        state = self._state(self._library(self.hub))
        target, resolved = state._resolve_model_target(model_ref=repo, path=str(repo_dir), backend="auto")
        self.assertEqual((target, resolved), (str(repo_dir), "llama.cpp"))

    def test_unlisted_folder_is_judged_by_its_files(self) -> None:
        mlx = _mlx_folder(self.root / "loose-gguf-mlx")
        gguf_dir = self.root / "loose-mlx"
        gguf_dir.mkdir()
        (gguf_dir / "m.gguf").write_bytes(GGUF_BYTES)
        state = self._state([])
        self.assertEqual(state._resolve_model_target(model_ref=None, path=str(mlx), backend="auto")[1], "auto")
        self.assertEqual(state._resolve_model_target(model_ref=None, path=str(gguf_dir), backend="auto")[1], "llama.cpp")


class SelectEngineTests(_TempCase):
    def test_auto_picks_engine_from_files(self) -> None:
        controller = _controller()
        mlx = _mlx_folder(self.root / "my-gguf-notes-mlx")
        engine = controller._select_engine(backend="auto", runtime_target=str(mlx), path=str(mlx), model_ref="my-gguf-notes-mlx")
        self.assertIsInstance(engine, MLXWorkerEngine)
        gguf_dir = self.root / "mlx"
        gguf_dir.mkdir()
        (gguf_dir / "m-Q4_K_M.gguf").write_bytes(GGUF_BYTES)
        engine = controller._select_engine(backend="auto", runtime_target=str(gguf_dir), path=str(gguf_dir), model_ref="m")
        self.assertIsInstance(engine, LlamaCppEngine)

    def test_folder_without_weights_names_what_it_found(self) -> None:
        folder = self.root / "half-downloaded-gguf"
        folder.mkdir()
        (folder / "config.json").write_text("{}", encoding="utf-8")
        (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(RuntimeError) as caught:
            _controller()._select_engine(backend="auto", runtime_target=str(folder), path=str(folder), model_ref="x")
        message = str(caught.exception)
        self.assertIn("No model weights found", message)
        self.assertIn("config.json", message)
        self.assertIn("tokenizer.json", message)

    def test_llama_cpp_on_a_safetensors_folder_says_so(self) -> None:
        folder = _mlx_folder(self.root / "some-gguf-named-mlx")
        engine = LlamaCppEngine(_capabilities())
        with self.assertRaises(RuntimeError) as caught:
            engine.load_model(
                model_ref="x", model_name="x", canonical_repo=None, source="library", backend="llama.cpp", path=str(folder), runtime_target=str(folder),
                cache_strategy="native", cache_bits=0, fp16_layers=0, fused_attention=False,
                fit_model_in_memory=False, context_tokens=2048,
            )
        message = str(caught.exception)
        self.assertIn("No .gguf weights found", message)
        self.assertIn("model.safetensors", message)
        self.assertIn("MLX backend", message)

    def test_llama_cpp_refuses_a_gguf_without_the_header(self) -> None:
        pointer = self.root / "model-Q4_K_M.gguf"
        pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        engine = LlamaCppEngine(_capabilities())
        with self.assertRaises(RuntimeError) as caught:
            engine.load_model(
                model_ref="x", model_name="x", canonical_repo=None, source="library", backend="llama.cpp", path=str(pointer), runtime_target=str(pointer),
                cache_strategy="native", cache_bits=0, fp16_layers=0, fused_attention=False,
                fit_model_in_memory=False, context_tokens=2048,
            )
        self.assertIn("not a GGUF file", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
