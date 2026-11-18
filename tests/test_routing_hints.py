"""A model's own location decides which model it is, not the repo hints a client sends (L-16).

Before the fix the backend took ``canonicalRepo`` (and a ``modelRef`` naming a
catalogue row) on the client's word. Loading the local
``mlx-community/Qwen2.5-0.5B-Instruct-4bit`` with another row's repo answered
500 "Qwen3.8 Flash Next runs only on TensorFold" (D-13). These tests send
wrong, stale and hostile hints the way the app, the CLI, an agent or an older
window can, and check the request reaches engine selection as the model the
path holds. Engine selection runs for real (a controller with MLX usable and
TensorFold not installed); only the engine start is faked.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from backend_service.helpers.discovery import _discover_local_models
from backend_service.inference import MLXWorkerEngine
from backend_service.inference.base import LoadedModelInfo
from backend_service.state import ChaosEngineState
from tests.test_backend_service import FakeRuntime, fake_system_snapshot, make_test_client
from tests.test_cli_smoke import cli
from tests.test_gguf_routing import GGUF_BYTES, _controller, _hf_cache_repo

QWEN25 = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
FLASH_NEXT = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"  # a TensorFold-only catalogue row
FLASH_NEXT_NAME = "Qwen3.8 Flash Next MLX 4-bit (TensorFold)"
BOLT = "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB"
BOLT_FILE = "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"  # the file that catalogue row pins

_MLX_FILES = {
    "config.json": json.dumps({"model_type": "qwen2", "quantization": {"bits": 4, "group_size": 64}}).encode(),
    "tokenizer.json": b"{}",
    "tokenizer_config.json": json.dumps({"chat_template": "{{ messages }}"}).encode(),
    "model.safetensors": b"x" * 64,
}


def _mlx_folder(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, payload in _MLX_FILES.items():
        (folder / name).write_bytes(payload)
    return folder


class RecordingRuntime(FakeRuntime):
    """Records what reaches the runtime and runs the real engine choice on it."""

    def __init__(self) -> None:
        super().__init__()
        self.controller = _controller()
        self.calls: list[dict[str, Any]] = []

    def load_model(self, **kwargs: Any) -> LoadedModelInfo:
        call = dict(kwargs)
        call.pop("progress_callback", None)
        engine = self.controller._select_engine(
            backend=kwargs.get("backend", "auto"),
            runtime_target=kwargs.get("runtime_target"),
            path=kwargs.get("path"),
            model_ref=kwargs.get("model_ref", ""),
            canonical_repo=kwargs.get("canonical_repo"),
        )
        call["engine"] = type(engine).__name__
        self.calls.append(call)
        info = super().load_model(**kwargs)
        info.canonicalRepo = kwargs.get("canonical_repo")
        info.path = kwargs.get("path")
        return info


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.hub = self.root / "hub"
        self.hub.mkdir()
        self.models = self.root / "Models"
        self.models.mkdir()
        env = mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.hub)})
        env.start()
        self.addCleanup(env.stop)
        memory = mock.patch("backend_service.helpers.memory_gate.snapshot_memory_signals", return_value=(64.0, 10.0))
        memory.start()
        self.addCleanup(memory.stop)
        self.qwen = _hf_cache_repo(self.hub, QWEN25, _MLX_FILES)

    def _client(self):
        library = _discover_local_models([
            {"label": "HF", "path": str(self.hub), "enabled": True, "source": "user"},
            {"label": "Models", "path": str(self.models), "enabled": True, "source": "user"},
        ])
        self.state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=lambda: library,
            settings_path=self.root / "settings.json",
            benchmarks_path=self.root / "benchmarks.json",
            chat_sessions_path=self.root / "sessions.json",
        )
        self.runtime = RecordingRuntime()
        self.state.runtime = self.runtime
        return make_test_client(self.state)

    def _load(self, client, **body: Any):
        payload = {"source": "library", "backend": "auto", "contextTokens": 2048, **body}
        return client.post("/api/models/load", json=payload)


class HfCachePathTests(_Case):
    def test_wrong_canonical_repo_for_hf_cache_path_loads_the_model_at_the_path(self) -> None:
        client = self._client()
        response = self._load(client, modelRef=QWEN25, modelName=QWEN25, canonicalRepo=FLASH_NEXT, path=str(self.qwen))
        self.assertEqual(response.status_code, 200, response.text)
        call = self.runtime.calls[-1]
        self.assertEqual(call["canonical_repo"], QWEN25)
        self.assertEqual(call["model_ref"], QWEN25)
        self.assertEqual(call["engine"], MLXWorkerEngine.__name__)

    def test_the_old_hint_still_routes_to_tensorfold_when_it_reaches_the_engine(self) -> None:
        """The failure the fix prevents: the wrong hint alone sends this folder to TensorFold."""
        with self.assertRaisesRegex(RuntimeError, "runs only on TensorFold"):
            _controller()._select_engine(
                backend="mlx", runtime_target=QWEN25, path=str(self.qwen), model_ref=QWEN25, canonical_repo=FLASH_NEXT,
            )

    def test_model_ref_naming_another_catalogue_row_is_replaced_by_the_path_repo(self) -> None:
        client = self._client()
        response = self._load(client, modelRef=FLASH_NEXT, modelName=FLASH_NEXT_NAME, canonicalRepo=FLASH_NEXT,
                              path=str(self.qwen))
        self.assertEqual(response.status_code, 200, response.text)
        call = self.runtime.calls[-1]
        self.assertEqual((call["model_ref"], call["canonical_repo"]), (QWEN25, QWEN25))
        self.assertNotEqual(call["model_name"], FLASH_NEXT_NAME)
        self.assertEqual(call["engine"], MLXWorkerEngine.__name__)

    def test_snapshot_subfolder_is_the_same_library_entry(self) -> None:
        client = self._client()
        snapshot = self.qwen / "snapshots" / "abc123"
        response = self._load(client, modelRef=QWEN25, canonicalRepo=FLASH_NEXT, path=str(snapshot))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.runtime.calls[-1]["canonical_repo"], QWEN25)
        self.assertEqual(self.state._find_library_entry(str(snapshot), None)["name"], QWEN25)

    def test_path_wins_over_a_model_ref_that_names_another_library_entry(self) -> None:
        other = _hf_cache_repo(self.hub, "mlx-community/Qwen3-0.6B-4bit", _MLX_FILES)
        client = self._client()
        entry = self.state._find_library_entry(str(self.qwen), "mlx-community/Qwen3-0.6B-4bit")
        self.assertEqual(entry["name"], QWEN25)
        response = self._load(client, modelRef="mlx-community/Qwen3-0.6B-4bit", path=str(self.qwen))
        self.assertEqual(response.status_code, 200, response.text)
        call = self.runtime.calls[-1]
        self.assertEqual(call["model_ref"], QWEN25)
        self.assertNotIn(str(other), str(call["runtime_target"]))


class LocalFolderTests(_Case):
    def test_non_catalogue_local_folder_ignores_a_catalogue_hint(self) -> None:
        folder = _mlx_folder(self.models / "my-tiny-chat")
        client = self._client()
        response = self._load(client, modelRef="my-tiny-chat", canonicalRepo=FLASH_NEXT, path=str(folder))
        self.assertEqual(response.status_code, 200, response.text)
        call = self.runtime.calls[-1]
        self.assertIsNone(call["canonical_repo"])
        self.assertEqual(call["model_ref"], "my-tiny-chat")
        self.assertEqual(call["engine"], MLXWorkerEngine.__name__)

    def test_unknown_path_with_hostile_hints_loads_what_is_there(self) -> None:
        folder = _mlx_folder(self.root / "elsewhere" / "downloaded-thing")  # not in any model folder
        client = self._client()
        response = self._load(client, modelRef=FLASH_NEXT, modelName=FLASH_NEXT_NAME, canonicalRepo=FLASH_NEXT,
                              path=str(folder))
        self.assertEqual(response.status_code, 200, response.text)
        call = self.runtime.calls[-1]
        self.assertEqual(call["model_ref"], "downloaded-thing")
        self.assertIsNone(call["canonical_repo"])
        self.assertEqual(call["runtime_target"], str(folder))
        self.assertEqual(call["engine"], MLXWorkerEngine.__name__)

    def test_shared_words_are_not_a_match(self) -> None:
        """L-19: "qwen", "4bit", "mlx" in common are not the same model."""
        folder = _mlx_folder(self.models / "Qwen3.8-Flash-MLX-4bit")
        self._client()
        self.assertIsNone(self.state._resolve_canonical_repo(model_ref=None, path=str(folder), canonical_repo=FLASH_NEXT))

    def test_missing_path_with_a_hint_names_nothing(self) -> None:
        self._client()
        missing = str(self.root / "gone" / "model")
        self.assertIsNone(self.state._resolve_canonical_repo(model_ref="x", path=missing, canonical_repo=FLASH_NEXT))


class ExactCatalogueMatchTests(_Case):
    def test_hf_cache_copy_of_the_row_keeps_its_repo_and_its_engine(self) -> None:
        flash = _hf_cache_repo(self.hub, FLASH_NEXT, _MLX_FILES)
        client = self._client()
        response = self._load(client, modelRef=FLASH_NEXT, canonicalRepo=FLASH_NEXT, path=str(flash))
        # TensorFold is not installed in this controller, so the real row is still sent there.
        self.assertEqual(response.status_code, 500)
        self.assertIn("runs only on TensorFold", response.text)
        self.assertEqual(self.state._resolve_canonical_repo(model_ref=FLASH_NEXT, path=str(flash), canonical_repo=None),
                         FLASH_NEXT)

    def test_folder_named_exactly_as_the_repo_keeps_the_hint(self) -> None:
        self._client()
        for folder in (self.models / "Qwen3.8-Flash-Next-MLX-4bit-MTP", self.models / "Vontra" / "Qwen3.8-Flash-Next-MLX-4bit-MTP",
                       self.models / "Vontra--Qwen3.8-Flash-Next-MLX-4bit-MTP"):
            _mlx_folder(folder)
            self.assertEqual(
                self.state._resolve_canonical_repo(model_ref=None, path=str(folder), canonical_repo=FLASH_NEXT), FLASH_NEXT,
                str(folder),
            )

    def test_gguf_file_with_the_pinned_name_keeps_the_hint_and_another_name_does_not(self) -> None:
        self._client()
        pinned = self.models / "bolt" / BOLT_FILE
        other = self.models / "bolt" / "Nemotron-Bolt-Q4.gguf"
        pinned.parent.mkdir()
        pinned.write_bytes(GGUF_BYTES)
        other.write_bytes(GGUF_BYTES)
        self.assertEqual(self.state._resolve_canonical_repo(model_ref=None, path=str(pinned), canonical_repo=BOLT), BOLT)
        self.assertIsNone(self.state._resolve_canonical_repo(model_ref=None, path=str(other), canonical_repo=BOLT))

    def test_catalogue_id_that_differs_from_its_repo_is_kept_for_its_cache_copy(self) -> None:
        row_id, row_repo = "meta-models/Muse-Glimmer-30B-GGUF-q4km", "meta-models/Muse-Glimmer-30B-GGUF"
        cache = _hf_cache_repo(self.hub, row_repo, {"Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf": GGUF_BYTES})
        client = self._client()
        response = self._load(client, modelRef=row_id, canonicalRepo=row_repo, path=str(cache), backend="llama.cpp")
        self.assertEqual(response.status_code, 200, response.text)
        call = self.runtime.calls[-1]
        self.assertEqual((call["model_ref"], call["canonical_repo"]), (row_id, row_repo))

    def test_catalogue_load_without_a_path_uses_the_row_repo_over_a_hint(self) -> None:
        self._client()
        self.assertEqual(self.state._resolve_canonical_repo(model_ref=BOLT, path=None, canonical_repo=FLASH_NEXT), BOLT)
        self.assertEqual(self.state._resolve_canonical_repo(model_ref=BOLT, path=None, canonical_repo=None), BOLT)


class OtherCallerTests(_Case):
    def test_cli_load_with_a_wrong_canonical_repo(self) -> None:
        client = self._client()

        def forward(method: str, url: str, *, body: Any = None, timeout: float = 30.0, **_: Any):
            response = client.request(method, "/" + url.split("/", 3)[3], json=body)
            return response.status_code, response.json()

        with mock.patch.object(cli, "_request", side_effect=forward), redirect_stdout(io.StringIO()):
            code = cli.main(["--port", "1", "load", QWEN25, "--path", str(self.qwen),
                             "--canonical-repo", FLASH_NEXT, "--context", "2048"])
        self.assertEqual(code, 0)
        call = self.runtime.calls[-1]
        self.assertEqual((call["canonical_repo"], call["engine"]), (QWEN25, MLXWorkerEngine.__name__))

    def test_chat_generate_with_stale_hints_keeps_the_path_repo(self) -> None:
        """What an agent, the bridge's keep-going runs or an older window send with each message."""
        client = self._client()
        response = client.post("/api/chat/generate", json={
            "prompt": "hi", "modelRef": FLASH_NEXT, "modelName": FLASH_NEXT_NAME, "canonicalRepo": FLASH_NEXT,
            "source": "library", "backend": "auto", "path": str(self.qwen), "maxTokens": 8,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.runtime.calls[-1]["canonical_repo"], QWEN25)
        self.assertEqual(self.state.runtime.loaded_model.canonicalRepo, QWEN25)
        self.assertEqual(response.json()["session"]["canonicalRepo"], QWEN25)

    def test_in_process_callers_with_a_wrong_hint(self) -> None:
        """Compare, HTML Challenge and the keep-going model swap build a LoadModelRequest themselves."""
        from backend_service.models import LoadModelRequest

        self._client()
        self.state.load_model(LoadModelRequest(
            modelRef=QWEN25, canonicalRepo=FLASH_NEXT, source="library", backend="auto", path=str(self.qwen),
        ), keep_warm_previous=False)
        call = self.runtime.calls[-1]
        self.assertEqual((call["canonical_repo"], call["engine"]), (QWEN25, MLXWorkerEngine.__name__))

if __name__ == "__main__":
    unittest.main()
