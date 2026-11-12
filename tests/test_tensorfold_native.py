"""TensorFold 1.x native engine: platform keys, release assets, capability gate,
VRAM estimates, serve command, safe unpack, installer, engine refusals and routing."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from backend_service.inference import _tensorfold_native as native
from backend_service.inference.base import BackendCapabilities
from backend_service.inference.tensorfold_native_engine import (
    TensorFoldNativeEngine,
    build_native_serve_command,
    grammar_library_present,
)
from backend_service.routes.setup import tensorfold_native as installer

CAPS_JSON = json.dumps(
    {
        "schema": 1,
        "version": "1.2.7",
        "chip": "nvidia-sm89",
        "backends": ["cuda"],
        "families": {"qwen3_5": ["mlx-q4g64", "exl3"], "nemotron_h": ["mlx-q4g64", "modelopt-nvfp4"]},
        "serve": {"flags": {"--vision": {}, "--drafter": {}, "--context": {}, "--parallel": {}, "--snapshot-dir": {}}},
    }
)


def _bc(**kw) -> BackendCapabilities:
    base = dict(pythonExecutable=None, mlxAvailable=False, mlxLmAvailable=False, mlxUsable=False)
    base.update(kw)
    return BackendCapabilities(**base)


def gpu(name="Test GPU", cc="8.9", driver="621.14", total=24564, free=22000) -> native.GpuInfo:
    return native.GpuInfo(name, cc, driver, total, free)


class PlatformAndAssets(unittest.TestCase):
    def test_platform_keys(self):
        self.assertEqual(native.platform_key("win32", "AMD64"), "windows-x86_64")
        self.assertEqual(native.platform_key("linux", "x86_64"), "linux-x86_64")
        self.assertEqual(native.platform_key("linux", "aarch64"), "linux-aarch64")
        self.assertEqual(native.platform_key("darwin", "arm64"), "macos-arm64")
        self.assertIsNone(native.platform_key("darwin", "x86_64"))
        self.assertIsNone(native.platform_key("win32", "ARM64"))

    def test_release_asset_names_and_urls(self):
        asset = native.release_asset("linux-x86_64", version="1.2.7")
        self.assertEqual(asset.name, "tensorfold-1.2.7-linux-x86_64.tar.gz")
        self.assertTrue(asset.url.endswith("/v1.2.7/tensorfold-1.2.7-linux-x86_64.tar.gz"))
        self.assertEqual(asset.sha256_url, asset.url + ".sha256")
        self.assertEqual(native.release_asset("linux-aarch64").name, "tensorfold-1.2.7-linux-aarch64.tar.gz")

    def test_windows_has_no_asset_unless_self_hosted(self):
        self.assertIsNone(native.release_asset("windows-x86_64", environ={}))
        digest = "a" * 64
        env = {native.WINDOWS_ARCHIVE_URL_ENV: "https://example.test/tf-win.zip", native.WINDOWS_ARCHIVE_SHA_ENV: digest}
        asset = native.release_asset("windows-x86_64", environ=env)
        self.assertEqual((asset.name, asset.sha256), ("tf-win.zip", digest))
        # http or a malformed digest is refused
        self.assertIsNone(native.release_asset("windows-x86_64", environ={**env, native.WINDOWS_ARCHIVE_URL_ENV: "http://x/y.zip"}))
        self.assertIsNone(native.release_asset("windows-x86_64", environ={**env, native.WINDOWS_ARCHIVE_SHA_ENV: "zz"}))

    def test_parse_sha256_file(self):
        digest = "ab" * 32
        self.assertEqual(native.parse_sha256_file(f"{digest}  tensorfold-1.2.7-linux-x86_64.tar.gz\n"), digest)
        self.assertEqual(native.parse_sha256_file(digest), digest)
        self.assertIsNone(native.parse_sha256_file("not a digest"))


class Probes(unittest.TestCase):
    def test_parse_nvidia_smi(self):
        info = native.parse_nvidia_smi("NVIDIA GeForce RTX 4090, 8.9, 621.14, 24564, 22363\n", "| CUDA Version: 13.2 |")
        self.assertEqual((info.name, info.compute_capability, info.driver_major, info.free_mb), ("NVIDIA GeForce RTX 4090", "8.9", 621, 22363))
        self.assertEqual(info.cuda_version, "13.2")
        self.assertEqual(info.cc_tuple, (8, 9))
        self.assertIsNone(native.parse_nvidia_smi("garbage"))

    def test_parse_capabilities(self):
        caps = native.parse_capabilities(CAPS_JSON)
        self.assertEqual((caps.chip, caps.backends, caps.version), ("nvidia-sm89", ("cuda",), "1.2.7"))
        self.assertTrue(caps.supports_flag("--vision"))
        self.assertFalse(caps.supports_flag("--kv-cache"))
        self.assertIn("mlx-q4g64", caps.families["qwen3_5"])
        self.assertIsNone(native.parse_capabilities("nope"))


class Gate(unittest.TestCase):
    def gate(self, g, caps=None, key="linux-x86_64", asset=True, present=False, smallest=3.1):
        return native.evaluate_gate(g, caps, key=key, asset_available=asset, binary_present=present, smallest_model_gb=smallest)

    def test_no_gpu_is_refused(self):
        result = self.gate(None)
        self.assertEqual(result.status, "refused")
        self.assertIn("NVIDIA", result.message)

    def test_old_driver_is_refused(self):
        result = self.gate(gpu(driver="570.10"))
        self.assertEqual(result.status, "refused")
        self.assertIn("R580", result.message)

    def test_pre_ampere_and_unknown_arch_are_refused(self):
        self.assertEqual(self.gate(gpu(cc="7.5")).status, "refused")
        self.assertEqual(self.gate(gpu(cc="[N/A]")).status, "refused")

    def test_card_too_small_for_any_model_is_refused(self):
        result = self.gate(gpu(total=2000, free=1800))
        self.assertEqual(result.status, "refused")
        self.assertIn("VRAM", result.message)

    def test_12gb_card_is_offered_with_a_free_vram_note(self):
        result = self.gate(gpu(cc="8.6", total=12288, free=2000))
        self.assertTrue(result.offered)
        self.assertTrue(any("free right now" in reason for reason in result.reasons))

    def test_release_image_arches_are_ready_with_an_archive(self):
        for cc in ("8.0", "8.6", "12.0", "12.1"):  # A100, 3090, 5090, DGX Spark
            self.assertEqual(self.gate(gpu(cc=cc), present=True).status, "ready", cc)
        self.assertEqual(self.gate(gpu(cc="12.1")).status, "needs-binary")

    def test_sm89_without_release_image_is_unqualified_until_the_binary_reports(self):
        result = self.gate(gpu(cc="8.9"), present=True)
        self.assertEqual(result.status, "unqualified")
        self.assertTrue(result.offered)
        ready = self.gate(gpu(cc="8.9"), native.parse_capabilities(CAPS_JSON), present=True)
        self.assertEqual(ready.status, "ready")

    def test_windows_without_binary_says_to_build_from_source(self):
        result = self.gate(gpu(), key="windows-x86_64", asset=False)
        self.assertEqual(result.status, "needs-binary")
        self.assertIn("source", result.message)

    def test_binary_reporting_another_chip_is_refused(self):
        other = native.parse_capabilities(CAPS_JSON.replace("nvidia-sm89", "cpu").replace('["cuda"]', '[]'))
        result = self.gate(gpu(), other, present=True)
        self.assertEqual(result.status, "refused")

    def test_unsupported_platform_is_refused(self):
        self.assertEqual(self.gate(gpu(), key=None).status, "refused")

    def test_mac_uses_metal(self):
        self.assertEqual(self.gate(None, key="macos-arm64").status, "needs-binary")


class Registry(unittest.TestCase):
    def test_27b_estimate_matches_the_measurement(self):
        model = native.match_native_model("TensorFold/Qwen3.8-27B-MLX-4bit")
        self.assertEqual(native.estimate_vram_gb(model, 8192), 18.8)
        self.assertEqual(native.match_native_model("vontra/qwen3.8-27b-mlx-4bit"), model)
        self.assertLess(native.estimate_vram_gb(model, 8192, drafts=False), 18.8)
        self.assertGreater(native.estimate_vram_gb(model, 32768), 18.8)

    def test_unknown_repo_matches_nothing(self):
        self.assertIsNone(native.match_native_model("someone/else", None))

    def test_supported_models_follow_the_binarys_families(self):
        caps = native.parse_capabilities(CAPS_JSON)
        rows = {row["key"]: row for row in native.supported_models(caps, free_gb=22.0)}
        self.assertTrue(rows["qwen3_8_27b"]["servable"] and rows["qwen3_8_27b"]["fits"])
        self.assertTrue(rows["nemotron_lightning"]["servable"])
        bare = native.parse_capabilities(CAPS_JSON.replace(', "modelopt-nvfp4"', ""))
        self.assertFalse({r["key"]: r for r in native.supported_models(bare)}["nemotron_bolt"]["servable"])
        self.assertFalse(native.supported_models(None)[0]["servable"])

    def test_lfm2_5_is_listed_for_cuda_with_an_unmeasured_floor(self):
        # FU-168: LFM2.5-24B-A2B on CUDA (1.2.5, GB10), from the 4-bit export.
        model = native.match_native_model("liquidai/lfm2.5-24b-a2b-mlx-4bit")
        assert model is not None
        self.assertEqual((model.family, model.weight_format, model.drafter, model.vision), ("lfm2_moe", "mlx-q4g64", None, False))
        self.assertFalse(model.measured)
        self.assertIn("not documented or measured", model.note)
        self.assertEqual(native.estimate_vram_gb(model, 8192), 15.7)
        self.assertIsNone(native.match_native_model("LiquidAI/LFM2.5-24B-A2B-MLX-8bit"))  # CUDA runs the 4-bit only
        caps = native.parse_capabilities(CAPS_JSON)
        self.assertFalse({r["key"]: r for r in native.supported_models(caps)}["lfm2_5_24b_a2b"]["servable"])
        with_lfm = native.parse_capabilities(CAPS_JSON.replace('"nemotron_h"', '"lfm2_moe": ["mlx-q4g64"], "nemotron_h"'))
        rows = {r["key"]: r for r in native.supported_models(with_lfm, free_gb=11.0)}
        self.assertTrue(rows["lfm2_5_24b_a2b"]["servable"])
        self.assertFalse(rows["lfm2_5_24b_a2b"]["fits"])  # the memory gate: 15.7 GB > 11 GB free
        self.assertTrue({r["key"]: r for r in native.supported_models(with_lfm, free_gb=22.0)}["lfm2_5_24b_a2b"]["fits"])

    def test_small_card_marks_big_models_as_not_fitting(self):
        rows = {row["key"]: row for row in native.supported_models(native.parse_capabilities(CAPS_JSON), free_gb=11.0)}
        self.assertFalse(rows["qwen3_8_27b"]["fits"])
        self.assertTrue(rows["qwen3_5_2b"]["fits"])


class ServeCommand(unittest.TestCase):
    def test_flags_follow_capabilities(self):
        caps = native.parse_capabilities(CAPS_JSON)
        command = build_native_serve_command(
            "tf", Path("model"), port=9000, name="m", context=8192, caps=caps, drafter_dir=Path("drafter"), vision=True
        )
        self.assertEqual(command[:3], ["tf", "serve", str(Path("model"))])
        for flag in ("--context", "--parallel", "--vision", "--drafter", "--snapshot-dir"):
            self.assertIn(flag, command)
        self.assertNotIn("--no-update-check", command)  # not listed by this build
        self.assertEqual(command[command.index("--drafter") + 1], str(Path("drafter")))

    def test_unlisted_flags_are_dropped_and_no_drafts_is_explicit(self):
        caps = native.parse_capabilities(CAPS_JSON.replace('"--vision": {}, ', ""))
        command = build_native_serve_command("tf", Path("m"), port=1, name="m", context=None, caps=caps, vision=True, drafts=False)
        self.assertNotIn("--vision", command)
        self.assertNotIn("--context", command)
        self.assertNotIn("--drafter", command)

    def test_grammar_library_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "bin" / "tensorfold-native"
            binary.parent.mkdir()
            binary.write_bytes(b"x")
            self.assertFalse(grammar_library_present(binary))
            (Path(tmp) / "lib").mkdir()
            (Path(tmp) / "lib" / "libtfgrammar.so").write_bytes(b"x")
            self.assertTrue(grammar_library_present(binary))


def _targz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class InstallerBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "tensorfold-native"
        patcher = mock.patch.object(native, "install_root", lambda: self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)


class SafeExtract(InstallerBase):
    def test_traversal_is_refused_in_tar_and_zip(self):
        bad_tar = Path(self.tmp.name) / "bad.tar.gz"
        bad_tar.write_bytes(_targz({"../evil": b"x"}))
        with self.assertRaises(installer.InstallError):
            installer.safe_extract(bad_tar, Path(self.tmp.name) / "out1")
        bad_zip = Path(self.tmp.name) / "bad.zip"
        with zipfile.ZipFile(bad_zip, "w") as bundle:
            bundle.writestr("../evil", b"x")
        with self.assertRaises(installer.InstallError):
            installer.safe_extract(bad_zip, Path(self.tmp.name) / "out2")
        self.assertFalse((Path(self.tmp.name) / "evil").exists())

    def test_good_archive_unpacks(self):
        archive = Path(self.tmp.name) / "ok.tar.gz"
        archive.write_bytes(_targz({"pkg/bin/tensorfold-native": b"bin"}))
        out = Path(self.tmp.name) / "out"
        installer.safe_extract(archive, out)
        self.assertEqual(native.locate_binary_in(out).read_bytes(), b"bin")


class InstallFromAsset(InstallerBase):
    def _asset(self, digest_override=None):
        data = _targz({"tensorfold/bin/tensorfold-native": b"#!fake"})
        digest = hashlib.sha256(data).hexdigest()
        asset = native.ReleaseAsset("linux-x86_64", "tf.tar.gz", "https://example.test/tf.tar.gz", "https://example.test/tf.tar.gz.sha256", None)

        def fake_download(url, target, progress):
            target.write_bytes(data)
            progress(1.0)

        return asset, digest_override or digest, fake_download

    def _run(self, asset, published, fake_download):
        caps = native.parse_capabilities(CAPS_JSON)
        with mock.patch.object(installer, "_download", fake_download), mock.patch.object(
            installer, "_fetch_text", lambda url: f"{published}  tf.tar.gz\n"
        ), mock.patch.object(native, "probe_capabilities", lambda binary, timeout=30.0: caps):
            return installer.install_from_asset(asset, lambda *a: None)

    def test_verified_archive_is_installed_and_recorded(self):
        asset, digest, fake = self._asset()
        binary, caps = self._run(asset, digest, fake)
        self.assertTrue(binary.is_file())
        record = native.read_install_record()
        self.assertEqual((record["version"], record["sha256"], record["platform"]), ("1.2.7", digest, "linux-x86_64"))
        self.assertEqual(native.find_binary({}), binary.resolve())
        self.assertEqual(list(self.root.glob("download-*")), [])  # temp download removed

    def test_sha_mismatch_installs_nothing(self):
        asset, _digest, fake = self._asset(digest_override="0" * 64)
        with self.assertRaises(installer.InstallError) as ctx:
            self._run(asset, "0" * 64, fake)
        self.assertIn("sha256 mismatch", str(ctx.exception))
        self.assertIsNone(native.read_install_record())

    def test_missing_published_digest_is_refused(self):
        asset, _digest, fake = self._asset()
        with self.assertRaises(installer.InstallError):
            self._run(asset, "no digest here", fake)

    def test_http_url_is_refused(self):
        asset = native.ReleaseAsset("linux-x86_64", "tf.tar.gz", "http://example.test/tf.tar.gz", None, None)
        with self.assertRaises(installer.InstallError):
            installer.install_from_asset(asset, lambda *a: None)

    def test_binary_that_does_not_answer_is_not_recorded(self):
        asset, digest, fake = self._asset()
        with mock.patch.object(installer, "_download", fake), mock.patch.object(
            installer, "_fetch_text", lambda url: digest
        ), mock.patch.object(native, "probe_capabilities", lambda binary, timeout=30.0: None):
            with self.assertRaises(installer.InstallError):
                installer.install_from_asset(asset, lambda *a: None)
        self.assertIsNone(native.read_install_record())


class Adopt(InstallerBase):
    def test_adopting_a_source_build_copies_hashes_and_records(self):
        source = Path(self.tmp.name) / "build" / "tensorfold-native.exe"
        source.parent.mkdir()
        source.write_bytes(b"MZ-fake")
        caps = native.parse_capabilities(CAPS_JSON)
        with mock.patch.object(native, "probe_capabilities", lambda binary, timeout=30.0: caps):
            dest, _ = installer.adopt_binary(source, platform="windows-x86_64")
        digest = hashlib.sha256(b"MZ-fake").hexdigest()
        self.assertEqual(dest.read_bytes(), b"MZ-fake")
        self.assertEqual(native.read_install_record()["sha256"], digest)
        self.assertTrue(native.read_install_record()["source"].startswith("adopted:"))

    def test_wrong_file_name_is_refused(self):
        other = Path(self.tmp.name) / "notepad.exe"
        other.write_bytes(b"x")
        with self.assertRaises(installer.InstallError):
            installer.adopt_binary(other, platform="windows-x86_64")

    def test_binary_override_env_wins_and_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "tensorfold-native"
            binary.write_bytes(b"x")
            self.assertEqual(native.find_binary({native.BINARY_ENV: str(binary)}), binary)
            self.assertIsNone(native.find_binary({native.BINARY_ENV: str(Path(tmp) / "missing")}))

    def test_install_record_cannot_point_outside_the_install_folder(self):
        self.root.mkdir(parents=True)
        outside = Path(self.tmp.name) / "outside"
        outside.write_bytes(b"x")
        native.install_record_path().write_text(json.dumps({"binary": "../outside"}), encoding="utf-8")
        self.assertIsNone(native.find_binary({}))


class StatusAndEngine(InstallerBase):
    def test_status_reports_gate_and_models(self):
        with mock.patch.object(native, "read_gpu", lambda timeout=10.0: gpu()), mock.patch.object(
            native, "platform_key", lambda *a, **k: "windows-x86_64"
        ):
            state = installer.native_status()
        self.assertFalse(state["installed"])
        self.assertTrue(state["needsBinaryPath"])
        self.assertEqual(state["gate"]["status"], "needs-binary")
        self.assertEqual(len(state["models"]), len(native.NATIVE_MODELS))

    def test_status_refuses_an_old_driver(self):
        with mock.patch.object(native, "read_gpu", lambda timeout=10.0: gpu(driver="550.1")), mock.patch.object(
            native, "platform_key", lambda *a, **k: "linux-x86_64"
        ):
            state = installer.native_status()
        self.assertFalse(state["gate"]["offered"])

    def _engine(self):
        return TensorFoldNativeEngine(_bc())

    def _load(self, engine, **overrides):
        args = dict(
            model_ref="TensorFold/Qwen3.8-27B-MLX-4bit", model_name="Qwen3.8-27B", canonical_repo="TensorFold/Qwen3.8-27B-MLX-4bit",
            source="library", backend="tensorfold", path=None, runtime_target=None, cache_strategy="native", cache_bits=0,
            fp16_layers=0, fused_attention=False, fit_model_in_memory=True, context_tokens=8192, speculative_decoding=True,
        )
        args.update(overrides)
        return engine.load_model(**args)

    def test_load_without_a_binary_says_install(self):
        with mock.patch.object(native, "find_binary", lambda environ=None: None):
            with self.assertRaises(RuntimeError) as ctx:
                self._load(self._engine())
        self.assertIn("not installed", str(ctx.exception))

    def test_load_is_refused_by_the_gate_on_an_old_driver(self):
        caps = native.parse_capabilities(CAPS_JSON)
        with mock.patch.object(native, "find_binary", lambda environ=None: Path("tf")), mock.patch.object(
            native, "probe_capabilities", lambda binary, timeout=30.0: caps
        ), mock.patch.object(native, "read_gpu", lambda timeout=10.0: gpu(driver="535.0")), mock.patch.object(
            native, "platform_key", lambda *a, **k: "windows-x86_64"
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self._load(self._engine())
        self.assertIn("R580", str(ctx.exception))

    def test_load_is_refused_when_free_vram_is_below_the_estimate(self):
        caps = native.parse_capabilities(CAPS_JSON)
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"x")
            with mock.patch.object(native, "find_binary", lambda environ=None: Path("tf")), mock.patch.object(
                native, "probe_capabilities", lambda binary, timeout=30.0: caps
            ), mock.patch.object(native, "read_gpu", lambda timeout=10.0: gpu(free=9000)), mock.patch.object(
                native, "platform_key", lambda *a, **k: "windows-x86_64"
            ), mock.patch.object(
                TensorFoldNativeEngine, "_ensure_drafter", lambda self, repo, cb: (None, None)
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    self._load(self._engine(), runtime_target=str(model_dir), speculative_decoding=False)
        self.assertIn("GB of VRAM", str(ctx.exception))
        self.assertIn("9.0 GB free", str(ctx.exception))

    def _mac_load(self, repo: str) -> list[str]:
        """Load ``repo`` on the Metal build with the server mocked; return the serve argv."""
        mac_caps = native.parse_capabilities(
            json.dumps({"version": "1.2.7", "chip": "apple-m4", "backends": ["metal"], "families": {},
                        "serve": {"flags": {"--vision": {}, "--drafter": {}, "--context": {}, "--parallel": {}}}})
        )
        commands: list[list[str]] = []

        class _Proc:
            def __init__(self, command, **_kw):
                commands.append(command)

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"x")
            drafter_dir = Path(tmp) / "drafter"
            with mock.patch.object(native, "find_binary", lambda environ=None: Path("tf")), mock.patch.object(
                native, "probe_capabilities", lambda binary, timeout=30.0: mac_caps
            ), mock.patch.object(native, "read_gpu", lambda timeout=10.0: None), mock.patch.object(
                native, "platform_key", lambda *a, **k: "macos-arm64"
            ), mock.patch.object(
                TensorFoldNativeEngine, "_ensure_drafter", lambda self, r, cb: (drafter_dir, None)
            ), mock.patch.object(TensorFoldNativeEngine, "_wait_for_server", lambda self, name, cb: None), mock.patch(
                "backend_service.inference.tensorfold_native_engine.subprocess.Popen", _Proc
            ):
                engine = self._engine()
                info = self._load(engine, model_ref=repo, canonical_repo=repo, runtime_target=str(model_dir))
                engine._cleanup_process = lambda: None
        self.info = info
        return commands[0]

    def test_mac_family_table_supplies_drafter_and_vision(self):
        # FU-168: checkpoints without a CUDA VRAM row take their drafter and
        # vision from the Mac family table.
        command = self._mac_load("Nanbeige/Nanbeige4.3-33B-INT4")
        self.assertIn("--drafter", command)
        self.assertNotIn("--vision", command)
        self.assertEqual(self.info.dflashDraftModel, "Nanbeige/Nanbeige4.1-3B")
        self.assertNotIn("not in the registry", self.info.runtimeNote)

        command = self._mac_load("BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34")
        self.assertIn("--vision", command)
        self.assertNotIn("--drafter", command)
        self.assertTrue(self.info.visionEnabled)

        command = self._mac_load("LiquidAI/LFM3-1.7B-MLX-4bit")
        self.assertNotIn("--drafter", command)
        self.assertNotIn("--vision", command)
        self.assertNotIn("not in the registry", self.info.runtimeNote)


class Routing(unittest.TestCase):
    def test_tensorfold_hint_picks_the_native_engine_off_the_mac(self):
        from backend_service.inference.controller import RuntimeController

        controller = RuntimeController.__new__(RuntimeController)
        controller.capabilities = _bc(tensorfoldNativeAvailable=True)
        engine = controller._select_tensorfold(
            hint="tensorfold", target="C:/m", model_ref="x/y", canonical_repo=None, runtime_target=None, path=None
        )
        self.assertIsInstance(engine, TensorFoldNativeEngine)

    def test_native_wins_over_the_python_install_when_both_are_present(self):
        from backend_service.inference.controller import RuntimeController

        controller = RuntimeController.__new__(RuntimeController)
        controller.capabilities = _bc(tensorfoldAvailable=True, tensorfoldNativeAvailable=True)
        engine = controller._select_tensorfold(
            hint="tensorfold", target="/m", model_ref="x/y", canonical_repo=None, runtime_target=None, path=None
        )
        self.assertIs(type(engine), TensorFoldNativeEngine)

    def test_a_family_the_native_binary_skips_uses_the_python_install(self):
        from backend_service.inference.controller import RuntimeController
        from backend_service.inference.tensorfold_engine import TensorFoldEngine

        controller = RuntimeController.__new__(RuntimeController)
        controller.capabilities = _bc(tensorfoldAvailable=True, tensorfoldNativeAvailable=True)
        gemma = dict(
            hint="tensorfold", target="/m", model_ref="mlx-community/gemma-4-26b-a4b-it-4bit",
            canonical_repo="mlx-community/gemma-4-26b-a4b-it-4bit", runtime_target=None, path=None,
        )
        self.assertIs(type(controller._select_tensorfold(**gemma)), TensorFoldEngine)
        controller.capabilities = _bc(tensorfoldNativeAvailable=True)
        with self.assertRaises(RuntimeError) as ctx:
            controller._select_tensorfold(**gemma)
        self.assertIn("not served by TensorFold 1.x", str(ctx.exception))

    def test_nothing_installed_asks_for_an_install(self):
        from backend_service.inference.controller import RuntimeController

        controller = RuntimeController.__new__(RuntimeController)
        controller.capabilities = _bc()
        with self.assertRaises(RuntimeError):
            controller._select_tensorfold(
                hint="tensorfold", target="/m", model_ref="x/y", canonical_repo=None, runtime_target=None, path=None
            )


if __name__ == "__main__":
    unittest.main()
