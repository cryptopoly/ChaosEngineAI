"""CUDA llama-server on Windows (and the shared driver gate): assets, checksums, install, selection, status.

The Linux paths stay in ``test_llama_cuda.py``. Everything here fakes the host, the driver and the
download, so it runs on any OS.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from backend_service.helpers import llama_cuda


def _zipfile(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def _host(os_name, arch, driver, cuda):
    return [
        mock.patch.object(llama_cuda, "host_os", return_value=os_name),
        mock.patch.object(llama_cuda, "host_arch", return_value=arch),
        mock.patch.object(llama_cuda, "nvidia_driver_major", return_value=driver),
        mock.patch.object(llama_cuda, "nvidia_driver_cuda", return_value=cuda),
    ]


class _Base(unittest.TestCase):
    def start_host(self, *host) -> None:
        for patcher in _host(*host):
            patcher.start()
            self.addCleanup(patcher.stop)


class _TempRoot(_Base):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="ce-llama-cuda-win-"))
        patcher = mock.patch.object(llama_cuda, "INSTALL_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        llama_cuda.clear_cache()
        self.addCleanup(llama_cuda.clear_cache)


class WindowsAssetTests(unittest.TestCase):
    def test_cuda_build_follows_the_driver_cuda_version(self) -> None:
        pick = llama_cuda.windows_cuda_version
        self.assertEqual(pick("x64", (13, 5)), "13.5")
        self.assertEqual(pick("x64", (13, 0)), "13.5")
        self.assertEqual(pick("x64", (12, 9)), "12.4")
        self.assertEqual(pick("x64", (12, 4)), "12.4")
        self.assertIsNone(pick("x64", (12, 3)))
        self.assertIsNone(pick("x64", (11, 8)))
        self.assertIsNone(pick("x64", None))
        # ARM64 has a 13.5 build only.
        self.assertEqual(pick("arm64", (13, 1)), "13.5")
        self.assertIsNone(pick("arm64", (12, 8)))

    def test_driver_major_decides_when_nvidia_smi_gives_no_cuda_line(self) -> None:
        pick = llama_cuda.windows_cuda_version
        self.assertEqual(pick("x64", None, 621), "13.5")
        self.assertEqual(pick("x64", None, 580), "13.5")
        self.assertEqual(pick("x64", None, 552), "12.4")
        self.assertIsNone(pick("x64", None, 546))

    def test_asset_names_match_the_published_release(self) -> None:
        # Names as published with llama.cpp b12317: the cudart zip carries no tag.
        self.assertEqual(
            llama_cuda.asset_names("x64", "13.5", "b12317", "windows"),
            ("llama-b12317-bin-win-cuda-13.5-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip"),
        )
        self.assertEqual(
            llama_cuda.asset_names("x64", "12.4", "b12317", "windows"),
            ("llama-b12317-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"),
        )

    def test_pinned_checksums_cover_every_asset_we_can_pick(self) -> None:
        for arch, cuda in (("x64", "13.5"), ("x64", "12.4"), ("arm64", "13.5")):
            for name in llama_cuda.asset_names(arch, cuda, llama_cuda.LLAMA_CPP_TAG, "windows"):
                self.assertRegex(llama_cuda._PINNED_SHA256.get(name, ""), r"^[0-9a-f]{64}$", name)

    def test_expected_sha256_prefers_the_pin_then_the_release(self) -> None:
        name = "llama-b12317-bin-win-cuda-13.5-x64.zip"
        never = mock.Mock(side_effect=AssertionError("pin should win"))
        self.assertEqual(llama_cuda.expected_sha256(name, "b12317", never), llama_cuda._PINNED_SHA256[name])
        self.assertEqual(llama_cuda.expected_sha256("x.zip", "b99999", lambda _t: {"x.zip": "ab"}), "ab")
        self.assertIsNone(llama_cuda.expected_sha256("x.zip", "b99999", lambda _t: {}))

    def test_release_digests_reads_the_published_sha256(self) -> None:
        payload = json.dumps({"assets": [
            {"name": "a.zip", "digest": "sha256:ABC123"}, {"name": "b.zip", "digest": None}, {"name": "c.zip"},
        ]}).encode()

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        with mock.patch.object(llama_cuda.urllib.request, "urlopen", return_value=_Resp(payload)):
            self.assertEqual(llama_cuda.release_digests("b1"), {"a.zip": "abc123"})
        with mock.patch.object(llama_cuda.urllib.request, "urlopen", side_effect=OSError("offline")):
            self.assertEqual(llama_cuda.release_digests("b1"), {})

    def test_host_os_and_server_name(self) -> None:
        with mock.patch.object(llama_cuda.sys, "platform", "win32"):
            self.assertEqual(llama_cuda.host_os(), "windows")
            self.assertEqual(llama_cuda.server_name(), "llama-server.exe")
        with mock.patch.object(llama_cuda.sys, "platform", "linux"):
            self.assertEqual(llama_cuda.host_os(), "linux")
            self.assertEqual(llama_cuda.server_name(), "llama-server")
        with mock.patch.object(llama_cuda.sys, "platform", "darwin"):
            self.assertIsNone(llama_cuda.host_os())
            self.assertIsNone(llama_cuda.host_arch())


class PlanTests(_Base):
    def _plan(self, *host) -> dict:
        self.start_host(*host)
        return llama_cuda.plan()

    def test_windows_any_recent_card_is_ready(self) -> None:
        # The gate reads the driver only; a 12 GB card and a 5090 take the same path.
        got = self._plan("windows", "x64", 621, (13, 5))
        self.assertTrue(got["ok"])
        self.assertEqual((got["state"], got["cuda"]), ("ready", "13.5"))

    def test_windows_cuda_12_driver_gets_the_12_4_build(self) -> None:
        got = self._plan("windows", "x64", 552, (12, 4))
        self.assertEqual((got["ok"], got["cuda"]), (True, "12.4"))

    def test_windows_old_driver_is_refused_with_a_clear_message(self) -> None:
        got = self._plan("windows", "x64", 531, (12, 1))
        self.assertFalse(got["ok"])
        self.assertEqual(got["state"], "driver-too-old")
        self.assertIn("too old", got["message"])
        self.assertIn("12.1", got["message"])

    def test_no_nvidia_driver(self) -> None:
        got = self._plan("windows", "x64", None, None)
        self.assertEqual((got["ok"], got["state"]), (False, "no-driver"))

    def test_unsupported_host(self) -> None:
        got = self._plan(None, None, None, None)
        self.assertEqual(got["state"], "unsupported")

    def test_linux_keeps_its_choice(self) -> None:
        self.assertEqual(self._plan("linux", "x64", 575, (12, 9))["cuda"], "12.8")
        self.assertEqual(self._plan("linux", "arm64", 580, (13, 0))["cuda"], "13.5")


class WindowsInstallTests(_TempRoot):
    def setUp(self) -> None:
        super().setUp()
        self.start_host("windows", "x64", 621, (13, 5))
        self.sources = Path(tempfile.mkdtemp(prefix="ce-llama-win-src-"))
        self.build, self.cudart = llama_cuda.asset_names("x64", "13.5", llama_cuda.LLAMA_CPP_TAG)
        _zipfile(self.sources / self.build, {"llama-server.exe": b"MZ", "ggml-cuda.dll": b"x", "llama-bench.exe": b"MZ"})
        _zipfile(self.sources / self.cudart, {"cudart64_13.dll": b"x", "cublas64_13.dll": b"x"})

    def _download(self, url: str, dest: Path, progress) -> None:
        data = (self.sources / url.rsplit("/", 1)[1]).read_bytes()
        dest.write_bytes(data)
        progress(len(data), len(data))

    def _digests(self) -> dict[str, str]:
        return {n: hashlib.sha256((self.sources / n).read_bytes()).hexdigest() for n in (self.build, self.cudart)}

    @unittest.skipUnless(sys.platform == "win32", "builds Windows paths; runs on the Windows box")
    def test_installs_server_and_runtime_dlls_next_to_each_other_and_records_it(self) -> None:
        digests = self._digests()
        with mock.patch.object(llama_cuda, "_download", side_effect=self._download), \
                mock.patch.object(llama_cuda, "expected_sha256", side_effect=lambda n, _t: digests[n]):
            binary = llama_cuda.install()
        self.assertEqual(binary.name, "llama-server.exe")
        self.assertEqual(
            sorted(p.name for p in binary.parent.iterdir()),
            ["cublas64_13.dll", "cudart64_13.dll", "ggml-cuda.dll", "installed.json", "llama-bench.exe", "llama-server.exe"],
        )
        info = json.loads((binary.parent / "installed.json").read_text())
        self.assertEqual(
            (info["os"], info["arch"], info["cuda"], info["tag"]),
            ("windows", "x64", "13.5", llama_cuda.LLAMA_CPP_TAG),
        )
        self.assertEqual(info["sha256"], digests)
        self.assertEqual(info["verified"], {self.build: True, self.cudart: True})
        self.assertEqual(info["driverCuda"], "13.5")
        self.assertEqual(llama_cuda.installed_binary(), binary)
        self.assertFalse(any(p.name.startswith(".staging-") for p in self.root.iterdir()))

    def test_a_checksum_mismatch_installs_nothing(self) -> None:
        with mock.patch.object(llama_cuda, "_download", side_effect=self._download), \
                mock.patch.object(llama_cuda, "expected_sha256", return_value="0" * 64):
            with self.assertRaisesRegex(RuntimeError, "sha256"):
                llama_cuda.install()
        self.assertIsNone(llama_cuda.installed_binary())
        self.assertEqual([p.name for p in self.root.iterdir()], [])

    def test_unpublished_checksum_is_recorded_as_unverified(self) -> None:
        with mock.patch.object(llama_cuda, "_download", side_effect=self._download), \
                mock.patch.object(llama_cuda, "expected_sha256", return_value=None):
            binary = llama_cuda.install()
        info = json.loads((binary.parent / "installed.json").read_text())
        self.assertEqual(info["verified"], {self.build: False, self.cudart: False})

    def test_old_driver_downloads_nothing(self) -> None:
        self.start_host("windows", "x64", 531, (12, 1))
        with mock.patch.object(llama_cuda, "_download") as download:
            with self.assertRaisesRegex(RuntimeError, "too old"):
                llama_cuda.install()
        download.assert_not_called()

    def test_a_full_disk_downloads_nothing(self) -> None:
        with mock.patch.object(llama_cuda, "_download") as download, \
                mock.patch.object(llama_cuda.shutil, "disk_usage", return_value=mock.Mock(free=1 << 30)):
            with self.assertRaisesRegex(RuntimeError, "disk space"):
                llama_cuda.install()
        download.assert_not_called()

    def test_reinstall_replaces_the_old_folder(self) -> None:
        digests = self._digests()
        with mock.patch.object(llama_cuda, "_download", side_effect=self._download), \
                mock.patch.object(llama_cuda, "expected_sha256", side_effect=lambda n, _t: digests[n]):
            first = llama_cuda.install()
            (first.parent / "stale.dll").write_bytes(b"x")
            second = llama_cuda.install()
        self.assertEqual(first, second)
        self.assertFalse((second.parent / "stale.dll").exists())

    def test_zip_member_outside_the_folder_is_refused(self) -> None:
        archive = self.root / "bad.zip"
        _zipfile(archive, {"../escape.dll": b"x"})
        dest = self.root / "out"
        dest.mkdir()
        with self.assertRaises(ValueError):
            llama_cuda._safe_extract_zip(archive, dest)
        self.assertFalse((self.root / "escape.dll").exists())


class WindowsSelectionTests(_TempRoot):
    """The CUDA server is chosen on a GPU box when installed, else the standard one stays."""

    def setUp(self) -> None:
        super().setUp()
        self.start_host("windows", "x64", 621, (13, 5))
        folder = self.root / llama_cuda.LLAMA_CPP_TAG
        folder.mkdir(parents=True)
        (folder / "installed.json").write_text("{}", encoding="utf-8")
        (folder / "llama-server.exe").write_bytes(b"MZ")
        self.binary = folder / "llama-server.exe"

    @unittest.skipUnless(sys.platform == "win32", "builds Windows paths; runs on the Windows box")
    def test_used_when_it_lists_a_cuda_device(self) -> None:
        with mock.patch.object(llama_cuda, "lists_cuda_device", return_value=True):
            self.assertEqual(llama_cuda.usable_binary(), str(self.binary))

    def test_not_used_without_a_cuda_device(self) -> None:
        with mock.patch.object(llama_cuda, "lists_cuda_device", return_value=False):
            self.assertIsNone(llama_cuda.usable_binary())

    @unittest.skipUnless(sys.platform == "win32", "builds Windows paths; runs on the Windows box")
    def test_backend_lookup_prefers_cuda_then_falls_back(self) -> None:
        from backend_service.inference import binaries

        bundled = {"CHAOSENGINE_LLAMA_SERVER": "C:/bundled/llama-server.exe"}
        with mock.patch.dict(os.environ, bundled), \
                mock.patch.object(llama_cuda, "lists_cuda_device", return_value=True):
            self.assertEqual(binaries._resolve_llama_server(), str(self.binary))
        llama_cuda.clear_cache()
        with mock.patch.dict(os.environ, bundled), \
                mock.patch.object(llama_cuda, "lists_cuda_device", return_value=False):
            self.assertEqual(binaries._resolve_llama_server(), "C:/bundled/llama-server.exe")
        llama_cuda.clear_cache()
        self.binary.unlink()
        with mock.patch.dict(os.environ, bundled):
            self.assertEqual(binaries._resolve_llama_server(), "C:/bundled/llama-server.exe")

    def test_switch_off_keeps_the_standard_server(self) -> None:
        with mock.patch.dict(os.environ, {"CHAOSENGINE_LLAMA_CUDA": "0"}), \
                mock.patch.object(llama_cuda, "lists_cuda_device", return_value=True):
            self.assertIsNone(llama_cuda.usable_binary())

    def test_list_devices_parser_reads_the_cuda_line(self) -> None:
        out = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 4090 (24563 MiB, 22000 MiB free)\n"
        done = mock.Mock(returncode=0, stdout=out, stderr="")
        with mock.patch.object(llama_cuda.subprocess, "run", return_value=done):
            self.assertTrue(llama_cuda.lists_cuda_device(self.binary))
        done = mock.Mock(returncode=0, stdout="Available devices:\n  Vulkan0: AMD Radeon\n", stderr="")
        with mock.patch.object(llama_cuda.subprocess, "run", return_value=done):
            self.assertFalse(llama_cuda.lists_cuda_device(self.binary))


class CapabilityRefreshTests(_TempRoot):
    @unittest.skipUnless(sys.platform == "win32", "builds Windows paths; runs on the Windows box")
    def test_install_detection_repoints_the_cached_server_path(self) -> None:
        from backend_service.inference import capabilities as caps
        from backend_service.inference.base import BackendCapabilities

        self.start_host("windows", "x64", 621, (13, 5))
        folder = self.root / llama_cuda.LLAMA_CPP_TAG
        folder.mkdir(parents=True)
        (folder / "installed.json").write_text("{}", encoding="utf-8")
        (folder / "llama-server.exe").write_bytes(b"MZ")
        cached = BackendCapabilities(
            pythonExecutable="python", mlxAvailable=False, mlxLmAvailable=False, mlxUsable=False,
            llamaServerPath="C:/bundled/llama-server.exe",
        )
        with mock.patch.object(caps, "_capability_cache", (0.0, cached)), \
                mock.patch.object(llama_cuda, "lists_cuda_device", return_value=True):
            caps.refresh_install_detection()
        self.assertEqual(cached.llamaServerPath, str(folder / "llama-server.exe"))


class WindowsRouteTests(_TempRoot):
    def _client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from backend_service.routes.setup.llama_cuda import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def _status(self, *host) -> dict:
        self.start_host(*host)
        return self._client().get("/api/setup/llama-cuda-status").json()

    def test_not_installed_and_installable(self) -> None:
        body = self._status("windows", "x64", 621, (13, 5))
        self.assertEqual((body["supported"], body["canInstall"], body["state"]), (True, True, "not-installed"))
        self.assertEqual((body["os"], body["cudaVersion"], body["driverCuda"]), ("windows", "13.5", "13.5"))
        self.assertFalse(body["installed"])

    def test_driver_too_old_is_supported_but_not_installable(self) -> None:
        body = self._status("windows", "x64", 531, (12, 1))
        self.assertEqual((body["supported"], body["canInstall"], body["driverTooOld"]), (True, False, True))
        self.assertEqual(body["state"], "driver-too-old")
        self.assertIn("too old", body["message"])

    def test_no_driver_is_not_supported(self) -> None:
        body = self._status("windows", "x64", None, None)
        self.assertEqual((body["supported"], body["state"]), (False, "no-driver"))

    @unittest.skipUnless(sys.platform == "win32", "builds Windows paths; runs on the Windows box")
    def test_installed_and_in_use_or_without_a_device(self) -> None:
        folder = self.root / llama_cuda.LLAMA_CPP_TAG
        folder.mkdir(parents=True)
        (folder / "installed.json").write_text(
            json.dumps({"cuda": "13.5", "verified": {"a": True, "b": True}}), encoding="utf-8")
        (folder / "llama-server.exe").write_bytes(b"MZ")
        self.start_host("windows", "x64", 621, (13, 5))
        with mock.patch.object(llama_cuda, "lists_cuda_device", return_value=True):
            body = self._client().get("/api/setup/llama-cuda-status").json()
        self.assertEqual(
            (body["state"], body["inUse"], body["verified"], body["installedCuda"]),
            ("in-use", True, True, "13.5"),
        )
        llama_cuda.clear_cache()
        with mock.patch.object(llama_cuda, "lists_cuda_device", return_value=False):
            body = self._client().get("/api/setup/llama-cuda-status").json()
        self.assertEqual((body["state"], body["inUse"]), ("installed-no-device", False))

    def test_install_endpoint_reports_a_too_old_driver_as_an_error(self) -> None:
        import time

        from backend_service.routes.setup import llama_cuda as route

        self.start_host("windows", "x64", 531, (12, 1))
        client = self._client()
        route._JOB.__init__()  # type: ignore[misc]
        client.post("/api/setup/install-llama-cuda")
        for _ in range(100):
            body = client.get("/api/setup/install-llama-cuda/status").json()
            if body["done"]:
                break
            time.sleep(0.05)
        self.assertEqual(body["phase"], "error")
        self.assertIn("too old", body["error"])
        route._JOB.__init__()  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
