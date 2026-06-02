"""Tests for the out-of-box RAG embedding-model installer + status.

Covers ``backend_service/routes/setup/embedding_model.py``:
* ``GET /api/rag/status`` reports vector vs lexical correctly.
* ``POST /api/setup/install-embedding-model`` runs the download worker
  and reports a clean ``done`` outcome.

The ``llama-embedding`` binary, the model resolution, and the actual HF
download are all mocked so the test never touches the network or the
real data dir.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.state import ChaosEngineState

TEST_API_TOKEN = "test-api-token"

EMBED_MOD = "backend_service.routes.setup.embedding_model"


def _fake_system_snapshot():
    return {
        "platform": "Darwin",
        "arch": "arm64",
        "hardwareSummary": "Test Machine",
        "backendLabel": "test",
        "appVersion": "0.0.0-test",
        "availableCacheStrategies": [],
        "dflash": {"available": False, "mlxAvailable": False, "vllmAvailable": False, "supportedModels": []},
        "vllmAvailable": False,
        "mlxAvailable": False,
        "mlxLmAvailable": False,
        "mlxUsable": False,
        "ggufAvailable": True,
        "converterAvailable": False,
        "nativePython": "/usr/bin/python3",
        "llamaServerPath": "/usr/local/bin/llama-server",
        "llamaServerTurboPath": None,
        "llamaCliPath": None,
        "nativeRuntimeMessage": None,
        "totalMemoryGb": 64,
        "availableMemoryGb": 32,
        "usedMemoryGb": 32,
        "swapUsedGb": 0,
        "swapTotalGb": 0,
        "compressedMemoryGb": 0,
        "memoryPressurePercent": 50.0,
        "cpuUtilizationPercent": 10.0,
        "gpuUtilizationPercent": None,
        "spareHeadroomGb": 26.0,
        "battery": None,
        "runningLlmProcesses": [],
        "uptimeMinutes": 1.0,
    }


class _FakeThread:
    """Runs the target synchronously on ``start()`` for deterministic tests."""

    def __init__(self, target=None, name=None, daemon=None, **_kwargs):
        self._target = target

    def start(self):
        if self._target is not None:
            self._target()


class EmbeddingSetupTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            settings_path=Path(self.tempdir.name) / "settings.json",
            benchmarks_path=Path(self.tempdir.name) / "benchmarks.json",
            chat_sessions_path=Path(self.tempdir.name) / "chats.json",
        )
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})

    def test_rag_status_vector_when_binary_and_model_present(self):
        with mock.patch(
            "backend_service.rag.embedding_client._resolve_binary",
            return_value="/opt/homebrew/bin/llama-embedding",
        ), mock.patch(
            "backend_service.rag.embedding_client._resolve_model",
            return_value="/data/embeddings/nomic.gguf",
        ):
            resp = self.client.get("/api/rag/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["mode"], "vector")
        self.assertTrue(body["binaryAvailable"])
        self.assertTrue(body["modelAvailable"])
        self.assertTrue(body["installed"])
        self.assertEqual(body["recommended"]["repo"], "nomic-ai/nomic-embed-text-v1.5-GGUF")
        self.assertTrue(body["recommended"]["file"].endswith(".gguf"))

    def test_rag_status_lexical_when_model_missing(self):
        with mock.patch(
            "backend_service.rag.embedding_client._resolve_binary",
            return_value="/opt/homebrew/bin/llama-embedding",
        ), mock.patch(
            "backend_service.rag.embedding_client._resolve_model",
            return_value=None,
        ):
            resp = self.client.get("/api/rag/status")
        body = resp.json()
        self.assertEqual(body["mode"], "lexical")
        self.assertTrue(body["binaryAvailable"])
        self.assertFalse(body["modelAvailable"])

    def test_install_embedding_model_downloads_and_reports_done(self):
        # Fake download writes a >1 MB file so the verify step passes.
        fake_gguf = Path(self.tempdir.name) / "nomic.gguf"
        fake_gguf.write_bytes(b"\0" * 2_000_000)

        with mock.patch(f"{EMBED_MOD}.threading.Thread", _FakeThread), mock.patch(
            f"{EMBED_MOD}._download_embedding_model", return_value=fake_gguf
        ):
            resp = self.client.post("/api/setup/install-embedding-model")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        # _FakeThread ran synchronously, so the job is already done.
        self.assertEqual(body["phase"], "done")
        self.assertTrue(body["done"])
        self.assertEqual(body["targetPath"], str(fake_gguf))

        status = self.client.get("/api/setup/install-embedding-model/status").json()
        self.assertEqual(status["phase"], "done")
        self.assertEqual(status["percent"], 100.0)

    def test_install_embedding_model_reports_error_on_truncated_download(self):
        tiny = Path(self.tempdir.name) / "tiny.gguf"
        tiny.write_bytes(b"\0" * 10)  # below the 1 MB sanity floor

        with mock.patch(f"{EMBED_MOD}.threading.Thread", _FakeThread), mock.patch(
            f"{EMBED_MOD}._download_embedding_model", return_value=tiny
        ):
            resp = self.client.post("/api/setup/install-embedding-model")

        body = resp.json()
        self.assertEqual(body["phase"], "error")
        self.assertIsNotNone(body["error"])


if __name__ == "__main__":
    unittest.main()
