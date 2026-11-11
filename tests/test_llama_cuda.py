"""CUDA llama-server for Linux (x86_64 and ARM64): asset choice, safe install, selection."""
from __future__ import annotations

import io
import json
import os
import re
import stat
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers import llama_cuda

ROOT = Path(__file__).resolve().parents[1]


def _fake_server(path: Path, devices: str) -> None:
    path.write_text(f'#!/bin/sh\necho "Available devices:"\necho "{devices}"\n', encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _tarball(path: Path, files: dict[str, bytes], top: str) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))


class AssetChoiceTests(unittest.TestCase):
    def test_arm64_always_uses_cuda_13(self) -> None:
        self.assertEqual(llama_cuda.cuda_version_for("arm64", None), "13.5")
        self.assertEqual(llama_cuda.cuda_version_for("arm64", 550), "13.5")

    def test_x64_falls_back_to_cuda_12_on_an_older_driver(self) -> None:
        self.assertEqual(llama_cuda.cuda_version_for("x64", 575), "12.8")
        self.assertEqual(llama_cuda.cuda_version_for("x64", 580), "13.5")

    def test_asset_names_match_the_published_release(self) -> None:
        # Names as published with llama.cpp b12292.
        self.assertEqual(
            llama_cuda.asset_names("arm64", "13.5", "b12292", "linux"),
            ("llama-b12292-bin-ubuntu-cuda-13.5-arm64.tar.gz",
             "cudart-llama-b12292-bin-ubuntu-cuda-13.5-arm64.tar.gz"),
        )

    def test_tag_matches_the_bundled_llama_cpp(self) -> None:
        stage = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        staged = re.search(r'LLAMA_CPP_TAG = process\.env\.CHAOSENGINE_LLAMA_CPP_TAG \|\| "(b\d+)"', stage)
        self.assertIsNotNone(staged)
        self.assertEqual(llama_cuda.LLAMA_CPP_TAG, staged.group(1))

    def test_only_linux_hosts_qualify(self) -> None:
        with mock.patch.object(llama_cuda.sys, "platform", "darwin"):
            self.assertIsNone(llama_cuda.host_arch())
        with mock.patch.object(llama_cuda.sys, "platform", "linux"), \
                mock.patch.object(llama_cuda.platform, "machine", return_value="aarch64"):
            self.assertEqual(llama_cuda.host_arch(), "arm64")


class _TempRoot(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="ce-llama-cuda-"))
        patcher = mock.patch.object(llama_cuda, "INSTALL_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        llama_cuda.clear_cache()
        self.addCleanup(llama_cuda.clear_cache)

    def _install_fake(self, devices: str) -> Path:
        folder = self.root / llama_cuda.LLAMA_CPP_TAG
        folder.mkdir(parents=True)
        (folder / "installed.json").write_text("{}", encoding="utf-8")
        _fake_server(folder / "llama-server", devices)
        return folder / "llama-server"


@unittest.skipIf(sys.platform == "win32", "the fake llama-server is a shell script; the feature is Linux-only")
class SelectionTests(_TempRoot):
    def test_used_when_it_lists_a_cuda_device(self) -> None:
        binary = self._install_fake("  CUDA0: NVIDIA GB10 (122000 MiB, 120000 MiB free)")
        self.assertEqual(llama_cuda.usable_binary(), str(binary))

    def test_not_used_without_a_cuda_device(self) -> None:
        self._install_fake("  (none)")
        self.assertIsNone(llama_cuda.usable_binary())

    def test_not_used_when_switched_off(self) -> None:
        self._install_fake("  CUDA0: NVIDIA GB10")
        with mock.patch.dict(os.environ, {"CHAOSENGINE_LLAMA_CUDA": "0"}):
            self.assertIsNone(llama_cuda.usable_binary())

    def test_not_used_without_the_install_marker(self) -> None:
        binary = self._install_fake("  CUDA0: NVIDIA GB10")
        (binary.parent / "installed.json").unlink()
        self.assertIsNone(llama_cuda.usable_binary())

    def test_the_backend_lookup_prefers_it_over_the_bundled_build(self) -> None:
        from backend_service.inference import binaries

        binary = self._install_fake("  CUDA0: NVIDIA GB10")
        with mock.patch.dict(os.environ, {"CHAOSENGINE_LLAMA_SERVER": "/bundled/llama-server"}):
            self.assertEqual(binaries._resolve_llama_server(), str(binary))
        llama_cuda.clear_cache()
        (binary.parent / "installed.json").unlink()
        with mock.patch.dict(os.environ, {"CHAOSENGINE_LLAMA_SERVER": "/bundled/llama-server"}):
            self.assertEqual(binaries._resolve_llama_server(), "/bundled/llama-server")


class InstallTests(_TempRoot):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.object(llama_cuda, "host_os", return_value="linux")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_install_unpacks_both_archives_into_one_folder(self) -> None:
        sources = Path(tempfile.mkdtemp(prefix="ce-llama-src-"))
        build, cudart = llama_cuda.asset_names("arm64", "13.5", llama_cuda.LLAMA_CPP_TAG, "linux")
        _tarball(sources / build, {"llama-server": b"#!/bin/sh\n", "libggml-cuda.so": b"x"}, "llama-b12292")
        _tarball(sources / cudart, {"libcudart.so.13": b"x", "libcublas.so.13": b"x"}, "cudart")

        def fake_download(url: str, dest: Path, progress) -> None:
            data = (sources / url.rsplit("/", 1)[1]).read_bytes()
            dest.write_bytes(data)
            progress(len(data), len(data))

        with mock.patch.object(llama_cuda, "host_arch", return_value="arm64"), \
                mock.patch.object(llama_cuda, "nvidia_driver_major", return_value=580), \
                mock.patch.object(llama_cuda, "nvidia_driver_cuda", return_value=(13, 1)), \
                mock.patch.object(llama_cuda, "expected_sha256", return_value=None), \
                mock.patch.object(llama_cuda, "_download", side_effect=fake_download):
            binary = llama_cuda.install()

        folder = binary.parent
        self.assertEqual(
            sorted(p.name for p in folder.iterdir()),
            ["installed.json", "libcublas.so.13", "libcudart.so.13", "libggml-cuda.so", "llama-server"],
        )
        self.assertTrue(os.access(binary, os.X_OK))
        self.assertEqual(json.loads((folder / "installed.json").read_text())["cuda"], "13.5")
        self.assertEqual(llama_cuda.installed_binary(), binary)
        self.assertFalse(any(p.name.startswith(".staging-") for p in self.root.iterdir()))

    def test_install_refuses_non_linux_hosts(self) -> None:
        with mock.patch.object(llama_cuda, "host_arch", return_value=None):
            with self.assertRaises(RuntimeError):
                llama_cuda.install()

    def test_extract_refuses_a_member_outside_the_folder(self) -> None:
        archive = self.root / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
        dest = self.root / "out"
        dest.mkdir()
        with self.assertRaises(ValueError):
            llama_cuda._safe_extract(archive, dest)
        self.assertFalse((self.root / "escape").exists())


class RouteTests(_TempRoot):
    def test_status_shape_on_a_host_without_nvidia(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from backend_service.routes.setup.llama_cuda import router

        app = FastAPI()
        app.include_router(router)
        with mock.patch.object(llama_cuda, "host_os", return_value="linux"), \
                mock.patch.object(llama_cuda, "host_arch", return_value="arm64"), \
                mock.patch.object(llama_cuda, "nvidia_driver_major", return_value=None):
            body = TestClient(app).get("/api/setup/llama-cuda-status").json()
        self.assertFalse(body["supported"])
        self.assertFalse(body["installed"])
        self.assertFalse(body["inUse"])
        self.assertEqual(body["state"], "no-driver")
        self.assertIsNone(body["cudaVersion"])


if __name__ == "__main__":
    unittest.main()
