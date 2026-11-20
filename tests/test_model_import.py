"""Tests for Ollama / LM Studio model import (#4).

Pure scanners + symlink import run against a fixture tree in a tempdir;
the two routes run through a TestClient with the store dirs (env) and the
app data dir (DOCUMENTS_DIR) patched to the tempdir so nothing touches the
real filesystem.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.helpers import model_import as mi
from backend_service.state import ChaosEngineState

TEST_API_TOKEN = "test-api-token"
HEX = "a" * 64


def _fake_system_snapshot():
    return {
        "platform": "Darwin", "arch": "arm64", "hardwareSummary": "Test", "backendLabel": "test",
        "appVersion": "0.0.0-test", "availableCacheStrategies": [],
        "dflash": {"available": False, "mlxAvailable": False, "vllmAvailable": False, "supportedModels": []},
        "vllmAvailable": False, "mlxAvailable": False, "mlxLmAvailable": False, "mlxUsable": False,
        "ggufAvailable": True, "converterAvailable": False, "nativePython": "/usr/bin/python3",
        "llamaServerPath": "/usr/local/bin/llama-server", "llamaServerTurboPath": None, "llamaCliPath": None,
        "nativeRuntimeMessage": None, "totalMemoryGb": 64, "availableMemoryGb": 32, "usedMemoryGb": 32,
        "swapUsedGb": 0, "swapTotalGb": 0, "compressedMemoryGb": 0, "memoryPressurePercent": 50.0,
        "cpuUtilizationPercent": 10.0, "gpuUtilizationPercent": None, "spareHeadroomGb": 26.0,
        "battery": None, "runningLlmProcesses": [], "uptimeMinutes": 1.0,
    }


def _build_ollama_store(root: Path) -> Path:
    models = root / ".ollama" / "models"
    (models / "blobs").mkdir(parents=True)
    (models / "manifests" / "registry.ollama.ai" / "library" / "llama3.2").mkdir(parents=True)
    blob = models / "blobs" / f"sha256-{HEX}"
    blob.write_bytes(b"\0" * 2_000_000)
    manifest = {
        "schemaVersion": 2,
        "layers": [
            {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:" + "b" * 64, "size": 10},
            {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{HEX}", "size": 2_000_000},
        ],
    }
    (models / "manifests" / "registry.ollama.ai" / "library" / "llama3.2" / "latest").write_text(json.dumps(manifest))
    return models


def _build_lmstudio_store(root: Path) -> Path:
    models = root / "lmstudio"
    repo_dir = models / "bartowski" / "Qwen3-8B-GGUF"
    repo_dir.mkdir(parents=True)
    (repo_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"\0" * 1_500_000)
    return models


class OllamaManifestParseTests(unittest.TestCase):
    def test_parses_model_layer(self):
        hex_part, size = mi.parse_ollama_manifest(
            {"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{HEX}", "size": 99}]}
        )
        self.assertEqual(hex_part, HEX)
        self.assertEqual(size, 99)

    def test_no_model_layer_returns_none(self):
        hex_part, _ = mi.parse_ollama_manifest(
            {"layers": [{"mediaType": "application/vnd.ollama.image.license", "digest": f"sha256:{HEX}"}]}
        )
        self.assertIsNone(hex_part)

    def test_malformed_digest_rejected(self):
        hex_part, _ = mi.parse_ollama_manifest(
            {"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:NOTHEX"}]}
        )
        self.assertIsNone(hex_part)


class ScannerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_scan_ollama_finds_model(self):
        models = _build_ollama_store(self.root)
        found = mi.scan_ollama(models)
        self.assertEqual(len(found), 1)
        c = found[0]
        self.assertEqual(c.name, "llama3.2:latest")
        self.assertEqual(c.repo, "llama3.2")
        self.assertEqual(c.source, "ollama")
        self.assertTrue(c.path.endswith(f"sha256-{HEX}"))
        self.assertEqual(c.size_bytes, 2_000_000)

    def test_scan_ollama_missing_dir_is_empty(self):
        self.assertEqual(mi.scan_ollama(self.root / "nope" / "models"), [])

    def test_scan_lmstudio_finds_gguf(self):
        models = _build_lmstudio_store(self.root)
        found = mi.scan_lmstudio([models])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].source, "lmstudio")
        self.assertEqual(found[0].repo, "bartowski/Qwen3-8B-GGUF")
        self.assertTrue(found[0].path.endswith(".gguf"))


class ShareWithLmStudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        snap = self.root / "hub" / "models--ggml-org--Qwen3.8-27B-GGUF" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        self.gguf = snap / "Qwen3.8-27B-Q4_K_M.gguf"
        self.gguf.write_bytes(b"GGUF" + b"0" * 64)
        self.lm = self.root / "lmstudio"

    def test_layout_from_hf_cache_path(self):
        out = mi.share_with_lmstudio(path=str(self.gguf), lmstudio_dir=self.lm)
        dest = Path(out["path"])
        self.assertEqual(dest, self.lm / "ggml-org" / "Qwen3.8-27B-GGUF" / "Qwen3.8-27B-Q4_K_M.gguf")
        self.assertEqual(dest.read_bytes(), self.gguf.read_bytes())
        self.assertEqual(out["linkType"], "hardlink")
        self.assertFalse(out["alreadyShared"])

    def test_repo_overrides_path_and_second_share_is_idempotent(self):
        out = mi.share_with_lmstudio(path=str(self.gguf), lmstudio_dir=self.lm, repo="acme/Thing")
        self.assertEqual((out["publisher"], out["model"]), ("acme", "Thing"))
        again = mi.share_with_lmstudio(path=str(self.gguf), lmstudio_dir=self.lm, repo="acme/Thing")
        self.assertTrue(again["alreadyShared"])

    def test_plain_path_falls_back_to_chaosengine_publisher(self):
        plain = self.root / "x.gguf"
        plain.write_bytes(b"GGUF")
        out = mi.share_with_lmstudio(path=str(plain), lmstudio_dir=self.lm)
        self.assertEqual((out["publisher"], out["model"]), ("chaosengine", "x"))

    def test_traversal_in_repo_is_neutralised(self):
        out = mi.share_with_lmstudio(path=str(self.gguf), lmstudio_dir=self.lm, repo="../../evil/..")
        self.assertTrue(Path(out["path"]).resolve().is_relative_to(self.lm.resolve()))

    def test_non_gguf_rejected(self):
        other = self.root / "a.bin"
        other.write_bytes(b"x")
        with self.assertRaises(FileNotFoundError):
            mi.share_with_lmstudio(path=str(other), lmstudio_dir=self.lm)


class ImportByReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_symlink_created_with_gguf_extension(self):
        models = _build_ollama_store(self.root)
        blob = models / "blobs" / f"sha256-{HEX}"
        data_dir = self.root / "data"
        result = mi.import_by_reference(source="ollama", path=str(blob), name="llama3.2:latest", data_dir=data_dir)
        dest = Path(result["importedPath"])
        self.assertFalse(result["alreadyImported"])
        self.assertTrue(dest.is_file())
        self.assertEqual(dest.suffix, ".gguf")
        self.assertTrue(dest.samefile(blob))

    def test_hard_link_when_symlink_refused(self):
        models = _build_ollama_store(self.root)
        blob = models / "blobs" / f"sha256-{HEX}"
        with mock.patch("os.symlink", side_effect=OSError(1314, "privilege not held")):
            result = mi.import_by_reference(source="ollama", path=str(blob), name="m", data_dir=self.root / "data")
        dest = Path(result["importedPath"])
        self.assertFalse(dest.is_symlink())
        self.assertTrue(dest.samefile(blob))

    def test_copy_when_no_link_is_possible(self):
        models = _build_ollama_store(self.root)
        blob = models / "blobs" / f"sha256-{HEX}"
        with mock.patch("os.symlink", side_effect=OSError("no")), mock.patch("os.link", side_effect=OSError("no")):
            result = mi.import_by_reference(source="ollama", path=str(blob), name="m", data_dir=self.root / "data")
        dest = Path(result["importedPath"])
        self.assertFalse(dest.samefile(blob))
        self.assertEqual(dest.read_bytes(), blob.read_bytes())
        self.assertEqual(list(dest.parent.glob("*.partial")), [])

    def test_second_import_is_idempotent(self):
        models = _build_ollama_store(self.root)
        blob = models / "blobs" / f"sha256-{HEX}"
        data_dir = self.root / "data"
        mi.import_by_reference(source="ollama", path=str(blob), name="llama3.2:latest", data_dir=data_dir)
        second = mi.import_by_reference(source="ollama", path=str(blob), name="llama3.2:latest", data_dir=data_dir)
        self.assertTrue(second["alreadyImported"])

    def test_missing_source_raises(self):
        with self.assertRaises(FileNotFoundError):
            mi.import_by_reference(source="ollama", path=str(self.root / "ghost"), name="x", data_dir=self.root / "d")


@unittest.skipIf(sys.platform == "win32", "symlink import requires privilege on Windows")
class ImportRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            settings_path=self.root / "settings.json",
            benchmarks_path=self.root / "benchmarks.json",
            chat_sessions_path=self.root / "chats.json",
        )
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})

    def test_scan_route_lists_both_sources(self):
        _build_ollama_store(self.root)
        _build_lmstudio_store(self.root)
        env = {
            "CHAOSENGINE_OLLAMA_DIR": str(self.root / ".ollama"),
            "CHAOSENGINE_LMSTUDIO_DIR": str(self.root / "lmstudio"),
        }
        with mock.patch.dict(os.environ, env):
            resp = self.client.get("/api/models/import/scan")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ollama"]["available"])
        self.assertEqual(len(body["ollama"]["models"]), 1)
        self.assertTrue(body["lmstudio"]["available"])
        self.assertEqual(len(body["lmstudio"]["models"]), 1)

    def test_import_route_symlinks_and_registers_directory(self):
        models = _build_ollama_store(self.root)
        blob = models / "blobs" / f"sha256-{HEX}"
        data_dir = self.root / "appdata"
        documents = data_dir / "documents"
        documents.mkdir(parents=True)
        with mock.patch("backend_service.app.DOCUMENTS_DIR", documents):
            resp = self.client.post(
                "/api/models/import",
                json={"source": "ollama", "path": str(blob), "name": "llama3.2:latest", "repo": "llama3.2"},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["repo"], "llama3.2")
        dest = Path(body["imported"]["importedPath"])
        self.assertTrue(dest.is_symlink())
        # Imported dir registered in settings for library discovery.
        state = self.client.app.state.chaosengine
        paths = [d.get("path") for d in state.settings["modelDirectories"]]
        self.assertIn(str(data_dir / "imported-models"), paths)


class ShareRouteTests(unittest.TestCase):
    """Share with LM Studio over HTTP, using the saved ``lmstudioDir`` setting (runs on Windows too)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            settings_path=self.root / "settings.json",
            benchmarks_path=self.root / "benchmarks.json",
            chat_sessions_path=self.root / "chats.json",
        )
        self.state = state
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})
        snap = self.root / "hub" / "models--ggml-org--Qwen3.8-27B-GGUF" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        self.gguf = snap / "m.gguf"
        self.gguf.write_bytes(b"GGUF")

    def test_unset_folder_is_a_clear_error(self):
        with mock.patch.dict(os.environ, {"CHAOSENGINE_LMSTUDIO_DIR": str(self.root / "missing")}):
            resp = self.client.post("/api/models/share-lmstudio", json={"path": str(self.gguf)})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("shared models folder is not set", resp.json()["detail"]["message"])

    def test_saved_folder_is_used_and_persisted(self):
        lm = self.root / "lmstudio"
        saved = self.client.post("/api/settings/lmstudio-dir", json={"path": str(lm)})
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(self.client.get("/api/settings/storage").json()["lmstudioDir"], str(lm.resolve()))
        resp = self.client.post("/api/models/share-lmstudio", json={"path": str(self.gguf)})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue((lm / "ggml-org" / "Qwen3.8-27B-GGUF" / "m.gguf").is_file())

    def test_relative_folder_rejected(self):
        resp = self.client.post("/api/settings/lmstudio-dir", json={"path": "relative/dir"})
        self.assertEqual(resp.status_code, 400)

    def test_missing_gguf_is_404(self):
        resp = self.client.post(
            "/api/models/share-lmstudio", json={"path": str(self.root / "no.gguf"), "lmstudioDir": str(self.root / "lm")}
        )
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
