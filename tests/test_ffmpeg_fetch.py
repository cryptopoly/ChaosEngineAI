"""Pinned ffmpeg download: pin table, hash check before extract, safe extraction, atomic install, routes."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import tempfile
import threading
import time
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.ffmpeg_fetch import install, pins
from backend_service.ffmpeg_fetch.pins import Pin

GOOD_VERSION = "ffmpeg version 9.0.2-test Copyright\nconfiguration: --disable-gpl --enable-libass\n"
LGPL = "GNU Lesser General Public License\nversion 2.1 or later\n"
FILTERS = "\n".join(f" TSC {name:<12} A->A  x" for name in sorted(install.REQUIRED_FILTERS))


def fake_runner(version=GOOD_VERSION, licence=LGPL, buildconf="--disable-gpl", filters=FILTERS):
    table = {"-version": version, "-L": licence, "-buildconf": buildconf, "-filters": filters}
    return lambda exe, *args: table[args[0]]


def make_zip(path: Path, members: dict[str, bytes]) -> None:
    # Store member names verbatim. On Windows ``ZipInfo`` rewrites "\\" -> "/" at write time
    # (because ``os.sep == "\\"``), which would erase a backslash member before it is stored; a
    # malicious archive built off-Windows keeps the backslash, so we make ``os.sep`` look like "/"
    # here to reproduce that archive faithfully on every platform.
    with mock.patch.object(zipfile.os, "sep", "/"):
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in members.items():
                zf.writestr(name, data)


def make_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:xz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


class PinTable(unittest.TestCase):
    def test_every_row_is_complete(self):
        self.assertTrue(pins.PINS)
        for key, pin in pins.PINS.items():
            self.assertEqual(pin.key, key)
            self.assertTrue(pin.url.startswith("https://"), key)
            self.assertRegex(pin.sha256, r"^[0-9a-f]{64}$")
            self.assertGreater(pin.size, 10_000_000)
            self.assertIn(pin.archive, ("zip", "tar.xz"))
            self.assertTrue(pin.asset.endswith(pin.archive), key)
            self.assertIn("lgpl", pin.asset)
            self.assertNotIn("-gpl-", pin.asset)
            self.assertTrue(pin.version and pin.licence_url and pin.source_url)
            self.assertIn(pins.TAG, pin.url)

    def test_host_keys(self):
        self.assertEqual(pins.host_key("win32", "AMD64"), "windows-x86_64")
        self.assertEqual(pins.host_key("win32", "ARM64"), "windows-arm64")
        self.assertEqual(pins.host_key("linux", "x86_64"), "linux-x86_64")
        self.assertEqual(pins.host_key("linux", "aarch64"), "linux-aarch64")
        self.assertIsNone(pins.host_key("darwin", "arm64"))
        self.assertIsNone(pins.host_key("linux", "riscv64"))

    def test_path_is_off_unless_pins_are_verified(self):
        self.assertIsNotNone(pins.pin_for_host("linux", "x86_64"))
        with mock.patch.object(pins, "PINS_VERIFIED", False):
            self.assertIsNone(pins.pin_for_host("linux", "x86_64"))
        self.assertIsNone(pins.pin_for_host("darwin", "arm64"))


class InstallBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.root = self.dir / "ffmpeg"
        self.progress = mock.Mock()

    def pin_for(self, archive: Path, kind: str = "zip") -> Pin:
        data = archive.read_bytes()
        return Pin("windows-x86_64", f"ffmpeg-test.{kind}", hashlib.sha256(data).hexdigest(), len(data), kind,
                   version="9.0.2-test")

    def zip_archive(self, members=None) -> Path:
        path = self.dir / "a.zip"
        make_zip(path, members or {"ffmpeg-x/bin/ffmpeg.exe": b"MZ-ffmpeg", "ffmpeg-x/bin/ffprobe.exe": b"MZ-probe",
                                   "ffmpeg-x/LICENSE.txt": b"lic", "ffmpeg-x/doc/readme": b"r"})
        return path

    @staticmethod
    def opener_for(path: Path):
        class Response(io.BytesIO):
            length = None

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        return lambda url, timeout=60: Response(path.read_bytes())


class Verify(InstallBase):
    def test_happy_path_installs_and_points_current(self):
        archive = self.zip_archive()
        pin = self.pin_for(archive)
        exe = install.install_pin(pin, self.root, self.progress, opener=self.opener_for(archive), runner=fake_runner())
        self.assertEqual(exe, self.root / "9.0.2-test" / "ffmpeg.exe")
        self.assertEqual(exe.read_bytes(), b"MZ-ffmpeg")
        self.assertEqual((self.root / "current").read_text(), "9.0.2-test")
        self.assertEqual(sorted(p.name for p in (self.root / "9.0.2-test").iterdir()), ["ffmpeg.exe", "ffprobe.exe"])
        self.assertEqual([p.name for p in self.root.glob(".partial-*")], [])
        phases = [call.args[0] for call in self.progress.call_args_list]
        self.assertEqual(phases[0], "download")
        self.assertEqual(phases[-1], "install")

    def test_tar_xz_installs(self):
        archive = self.dir / "a.tar.xz"
        make_tar(archive, {"ffmpeg-x/bin/ffmpeg": b"ELF", "ffmpeg-x/bin/ffprobe": b"ELF2"})
        pin = self.pin_for(archive, "tar.xz")
        exe = install.install_pin(pin, self.root, self.progress, opener=self.opener_for(archive), runner=fake_runner())
        self.assertEqual(exe.name, "ffmpeg")
        self.assertTrue(os.access(exe, os.X_OK) or os.name == "nt")

    def test_hash_mismatch_installs_nothing_and_runs_nothing(self):
        archive = self.zip_archive()
        pin = replace(self.pin_for(archive), sha256="0" * 64)
        runner = mock.Mock()
        with self.assertRaisesRegex(install.FetchError, "corrupted or changed upstream"):
            install.install_pin(pin, self.root, self.progress, opener=self.opener_for(archive), runner=runner)
        runner.assert_not_called()
        self.assertFalse((self.root / "current").exists())
        self.assertEqual([p.name for p in self.root.iterdir()], [])

    def test_hash_checked_before_extract(self):
        archive = self.zip_archive()
        pin = replace(self.pin_for(archive), sha256="0" * 64)
        with mock.patch.object(install, "extract_members") as extract:
            with self.assertRaises(install.FetchError):
                install.install_pin(pin, self.root, self.progress, opener=self.opener_for(archive), runner=fake_runner())
        extract.assert_not_called()

    def test_gpl_build_is_refused(self):
        archive = self.zip_archive()
        pin = self.pin_for(archive)
        for runner in (fake_runner(licence="GNU General Public License"),
                       fake_runner(buildconf="--enable-gpl --enable-libx264"),
                       fake_runner(version=GOOD_VERSION + "--enable-nonfree")):
            with self.assertRaisesRegex(install.FetchError, "not an LGPL build"):
                install.install_pin(pin, self.root, self.progress, opener=self.opener_for(archive), runner=runner)
        self.assertFalse((self.root / "current").exists())

    def test_missing_filter_is_refused(self):
        archive = self.zip_archive()
        pin = self.pin_for(archive)
        with self.assertRaisesRegex(install.FetchError, "subtitles"):
            install.install_pin(pin, self.root, self.progress, opener=self.opener_for(archive),
                                runner=fake_runner(filters=FILTERS.replace("subtitles", "other")))

    def test_not_ffmpeg_is_refused(self):
        archive = self.zip_archive()
        with self.assertRaisesRegex(install.FetchError, "does not look like ffmpeg"):
            install.install_pin(self.pin_for(archive), self.root, self.progress,
                                opener=self.opener_for(archive), runner=fake_runner(version="hello"))

    def test_low_disk_space(self):
        archive = self.zip_archive()
        with self.assertRaisesRegex(install.FetchError, "free disk space"):
            install.install_pin(self.pin_for(archive), self.root, self.progress, free_bytes=lambda p: 1,
                                opener=self.opener_for(archive), runner=fake_runner())

    def test_network_failure_names_url_and_hash(self):
        archive = self.zip_archive()
        pin = self.pin_for(archive)

        def broken(url, timeout=60):
            raise OSError("no route")

        with self.assertRaises(install.FetchError) as ctx:
            install.install_pin(pin, self.root, self.progress, opener=broken, runner=fake_runner())
        self.assertIn(pin.url, str(ctx.exception))
        self.assertIn(pin.sha256, str(ctx.exception))

    def test_cancel(self):
        archive = self.zip_archive()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(install.Cancelled):
            install.install_pin(self.pin_for(archive), self.root, self.progress, cancel,
                                opener=self.opener_for(archive), runner=fake_runner())
        self.assertEqual([p.name for p in self.root.iterdir()], [])

    def test_http_url_is_refused(self):
        archive = self.zip_archive()
        pin = self.pin_for(archive)

        with mock.patch.object(Pin, "url", new_callable=mock.PropertyMock, return_value="http://x/y"):
            with self.assertRaisesRegex(install.FetchError, "not https"):
                install.install_pin(pin, self.root, self.progress, opener=self.opener_for(archive), runner=fake_runner())


class Extract(InstallBase):
    def test_path_traversal_is_rejected(self):
        for name in ("../evil/bin/ffmpeg", "/abs/bin/ffmpeg", "x/../../bin/ffmpeg", "x\\bin\\ffmpeg"):
            archive = self.dir / "t.zip"
            make_zip(archive, {name: b"x"})
            with self.assertRaisesRegex(install.FetchError, "unsafe path"):
                install.extract_members(archive, self.pin_for(archive), self.dir / "out")

    def test_drive_letter_member_is_rejected(self):
        archive = self.dir / "d.zip"
        make_zip(archive, {"C:/x/bin/ffmpeg": b"x"})
        with self.assertRaisesRegex(install.FetchError, "unsafe path"):
            install.extract_members(archive, self.pin_for(archive), self.dir / "out")

    def test_backslash_member_is_rejected_as_windows_reads_it(self):
        # On Windows the zip reader turns "\\" into "/" in ``filename``; the check must use the stored name.
        archive = self.dir / "w.zip"
        make_zip(archive, {"x\\bin\\ffmpeg": b"x"})
        with mock.patch.object(zipfile.os, "sep", "\\"):
            with self.assertRaisesRegex(install.FetchError, "unsafe path"):
                install.extract_members(archive, self.pin_for(archive), self.dir / "out")

    def test_tar_link_member_is_rejected(self):
        archive = self.dir / "l.tar.xz"
        with tarfile.open(archive, "w:xz") as tf:
            info = tarfile.TarInfo("x/bin/ffmpeg")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tf.addfile(info)
        with self.assertRaisesRegex(install.FetchError, "link"):
            install.extract_members(archive, self.pin_for(archive, "tar.xz"), self.dir / "out")

    def test_zip_symlink_member_is_rejected(self):
        archive = self.dir / "l.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            info = zipfile.ZipInfo("x/bin/ffmpeg")
            info.external_attr = (0o120777 << 16)
            zf.writestr(info, "/etc/passwd")
        with self.assertRaisesRegex(install.FetchError, "link"):
            install.extract_members(archive, self.pin_for(archive), self.dir / "out")

    def test_nothing_else_is_extracted(self):
        archive = self.zip_archive({"x/bin/ffmpeg.exe": b"a", "x/bin/other.dll": b"b", "x/bin/sub/ffmpeg.exe": b"c"})
        written = install.extract_members(archive, self.pin_for(archive), self.dir / "out")
        self.assertEqual([p.name for p in written], ["ffmpeg.exe"])

    def test_archive_without_ffmpeg(self):
        archive = self.zip_archive({"x/readme": b"a"})
        with self.assertRaisesRegex(install.FetchError, "no ffmpeg"):
            install.extract_members(archive, self.pin_for(archive), self.dir / "out")

    def test_corrupt_archive(self):
        archive = self.dir / "bad.zip"
        archive.write_bytes(b"not a zip")
        with self.assertRaisesRegex(install.FetchError, "could not be read"):
            install.extract_members(archive, self.pin_for(archive), self.dir / "out")


class Atomic(InstallBase):
    def installed(self, version="1.0", payload=b"old"):
        folder = self.root / version
        folder.mkdir(parents=True)
        (folder / "ffmpeg.exe").write_bytes(payload)
        (self.root / "current").write_text(version)

    def test_failure_after_move_keeps_old_binary_current(self):
        self.installed("1.0")
        staged = self.dir / "staged"
        staged.mkdir()
        (staged / "ffmpeg.exe").write_bytes(b"new")
        with mock.patch.object(install, "_write_atomic", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                install.switch_to(self.root, "2.0", staged)
        self.assertEqual((self.root / "current").read_text(), "1.0")
        self.assertEqual((self.root / "1.0" / "ffmpeg.exe").read_bytes(), b"old")

    def test_reinstall_of_same_version_restores_on_failure(self):
        self.installed("1.0")
        staged = self.dir / "staged"
        staged.mkdir()
        (staged / "ffmpeg.exe").write_bytes(b"new")
        with mock.patch.object(install, "_write_atomic", side_effect=OSError("x")):
            with self.assertRaises(OSError):
                install.switch_to(self.root, "1.0", staged)
        self.assertEqual((self.root / "1.0" / "ffmpeg.exe").read_bytes(), b"old")

    def test_success_removes_old_versions(self):
        self.installed("1.0")
        staged = self.dir / "staged"
        staged.mkdir()
        (staged / "ffmpeg.exe").write_bytes(b"new")
        install.switch_to(self.root, "2.0", staged)
        self.assertEqual((self.root / "current").read_text(), "2.0")
        self.assertFalse((self.root / "1.0").exists())

    def test_partials_are_cleaned_at_next_install(self):
        (self.root / ".partial-dead").mkdir(parents=True)
        (self.root / ".partial-dead" / "x").write_text("x")
        archive = self.zip_archive()
        install.install_pin(self.pin_for(archive), self.root, self.progress, opener=self.opener_for(archive),
                            runner=fake_runner())
        self.assertFalse((self.root / ".partial-dead").exists())

    def test_failed_install_keeps_previous_current(self):
        self.installed("1.0")
        archive = self.zip_archive()
        pin = replace(self.pin_for(archive), sha256="1" * 64)
        with self.assertRaises(install.FetchError):
            install.install_pin(pin, self.root, self.progress, opener=self.opener_for(archive), runner=fake_runner())
        self.assertEqual((self.root / "current").read_text(), "1.0")
        self.assertTrue((self.root / "1.0" / "ffmpeg.exe").is_file())


class ChooseFile(InstallBase):
    def test_pinned_archive_chosen_offline_gets_the_same_checks(self):
        archive = self.zip_archive()
        pin = self.pin_for(archive)
        exe = install.install_from_file(pin, archive, self.root, self.progress, runner=fake_runner())
        self.assertTrue(exe.is_file())
        bad = replace(pin, sha256="2" * 64)
        with self.assertRaisesRegex(install.FetchError, "corrupted or changed"):
            install.install_from_file(bad, archive, self.dir / "other", self.progress, runner=fake_runner())

    def test_missing_file(self):
        with self.assertRaisesRegex(install.FetchError, "not found"):
            install.install_from_file(self.pin_for(self.zip_archive()), self.dir / "no.zip", self.root, self.progress)

    def test_custom_validation(self):
        exe = self.dir / "mine"
        exe.write_text("x")
        self.assertEqual(install.validate_custom(exe, fake_runner()), "9.0.2-test")
        with self.assertRaises(install.FetchError):
            install.validate_custom(exe, fake_runner(version="nope"))
        with self.assertRaises(install.FetchError):
            install.validate_custom(self.dir / "gone", fake_runner())


class Routes(InstallBase):
    def setUp(self):
        super().setUp()
        from backend_service.app import create_app
        from backend_service.routes.setup import ffmpeg as route

        self.route = route
        patcher = mock.patch.dict(os.environ, {"CHAOSENGINE_FFMPEG_DIR": str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("CHAOSENGINE_FFMPEG", None)
        self.client = TestClient(create_app(api_token="test-api-token"))
        self.client.headers.update({"Authorization": "Bearer test-api-token"})
        self.headers = {}

    def get_status(self):
        with mock.patch.object(self.route.ffmpeg_bin, "_imageio_exe", return_value=None):
            return self.client.get("/api/setup/ffmpeg", headers=self.headers)

    def test_status_shape_when_nothing_installed(self):
        response = self.get_status()
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for key in ("installed", "path", "source", "version", "licence", "encoder", "pin", "canDownload", "job"):
            self.assertIn(key, data)
        self.assertFalse(data["installed"])

    def test_install_without_a_pin_is_a_400(self):
        with mock.patch.object(self.route, "pin_for_host", return_value=None):
            response = self.client.post("/api/setup/ffmpeg/install", json={"source": "download"}, headers=self.headers)
        self.assertEqual(response.status_code, 400)

    def test_install_job_with_fake_download_then_status_done(self):
        archive = self.zip_archive()
        pin = self.pin_for(archive)
        with mock.patch.object(self.route, "pin_for_host", return_value=pin), \
                mock.patch.object(self.route.fetch, "install_pin") as fake:
            fake.side_effect = lambda p, root, progress, cancel: progress("download", 5, 10)
            self.route._JOB.thread = None
            response = self.client.post("/api/setup/ffmpeg/install", json={"source": "download"}, headers=self.headers)
            self.assertEqual(response.status_code, 200)
            for _ in range(100):
                job = self.client.get("/api/setup/ffmpeg/install/status", headers=self.headers).json()["job"]
                if job["done"] or job["error"]:
                    break
                time.sleep(0.02)
        self.assertTrue(job["done"], job)
        self.assertEqual(job["percent"], 100.0)

    def test_install_error_is_reported(self):
        pin = self.pin_for(self.zip_archive())
        with mock.patch.object(self.route, "pin_for_host", return_value=pin), \
                mock.patch.object(self.route.fetch, "install_pin", side_effect=install.FetchError("boom")):
            self.route._JOB.thread = None
            response = self.client.post("/api/setup/ffmpeg/install", json={"source": "download"}, headers=self.headers)
            for _ in range(100):
                job = self.client.get("/api/setup/ffmpeg/install/status", headers=self.headers).json()["job"]
                if job["error"]:
                    break
                time.sleep(0.02)
        self.assertEqual(job["error"], "boom")

    def test_custom_file_is_validated_then_used(self):
        exe = self.dir / "myffmpeg"
        exe.write_text("x")
        with mock.patch.object(self.route.fetch, "validate_custom", return_value="7.0"):
            response = self.client.post("/api/setup/ffmpeg/custom", json={"path": str(exe)}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual((self.root / "custom").read_text(), str(exe.resolve()))
        bad = self.client.post("/api/setup/ffmpeg/custom", json={"path": str(self.dir / "nope")}, headers=self.headers)
        self.assertEqual(bad.status_code, 400)


if __name__ == "__main__":
    unittest.main()
