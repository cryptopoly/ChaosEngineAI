"""Tests for FU-056 Phase 1 accelerator capability probes.

Covers ``backend_service/inference/accelerators.py`` and its wiring
into ``BackendCapabilities.to_dict``. The probes are intentionally
boring — we're mostly pinning the "package present / package absent /
package broken" matrix so future regressions can't silently flip the
UI gating that downstream phases depend on.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from backend_service.inference import accelerators
from backend_service.inference.base import BackendCapabilities


class SpecExistsTests(unittest.TestCase):
    def test_returns_true_when_module_resolvable(self):
        # ``json`` is in the stdlib — always findable.
        self.assertTrue(accelerators._spec_exists("json"))

    def test_returns_false_when_module_absent(self):
        self.assertFalse(accelerators._spec_exists("nunchaku_fake_module_xyz"))

    def test_swallows_partial_install_raise(self):
        with patch(
            "backend_service.inference.accelerators.importlib.util.find_spec",
            side_effect=ValueError("broken __spec__"),
        ):
            self.assertFalse(accelerators._spec_exists("anything"))


class SafeVersionTests(unittest.TestCase):
    def test_returns_none_when_module_absent(self):
        self.assertIsNone(accelerators._safe_version("nunchaku_fake_module_xyz"))

    def test_returns_version_string_when_present(self):
        fake_module = MagicMock(__version__="1.2.3")
        with patch.object(accelerators, "_spec_exists", return_value=True):
            with patch.object(
                accelerators.importlib,
                "import_module",
                return_value=fake_module,
            ):
                self.assertEqual(accelerators._safe_version("anything"), "1.2.3")

    def test_returns_none_when_module_lacks_version(self):
        fake_module = MagicMock(spec=[])  # no __version__ attribute
        with patch.object(accelerators, "_spec_exists", return_value=True):
            with patch.object(
                accelerators.importlib,
                "import_module",
                return_value=fake_module,
            ):
                self.assertIsNone(accelerators._safe_version("anything"))

    def test_swallows_import_failure(self):
        with patch.object(accelerators, "_spec_exists", return_value=True):
            with patch.object(
                accelerators.importlib,
                "import_module",
                side_effect=ImportError("broken native ext"),
            ):
                self.assertIsNone(accelerators._safe_version("anything"))


class PerAcceleratorAvailabilityTests(unittest.TestCase):
    """Each accelerator's ``*_available()`` helper must flip cleanly on
    ``find_spec`` answers. Patching ``_spec_exists`` rather than
    ``find_spec`` keeps the test independent of how the real probes
    are implemented underneath."""

    def test_nunchaku_available_true(self):
        with patch.object(accelerators, "_spec_exists", return_value=True):
            self.assertTrue(accelerators.nunchaku_available())

    def test_nunchaku_available_false(self):
        with patch.object(accelerators, "_spec_exists", return_value=False):
            self.assertFalse(accelerators.nunchaku_available())

    def test_sageattention_available_true(self):
        with patch.object(accelerators, "_spec_exists", return_value=True):
            self.assertTrue(accelerators.sageattention_available())

    def test_triattention_available_true(self):
        with patch.object(accelerators, "_spec_exists", return_value=True):
            self.assertTrue(accelerators.triattention_available())

    def test_kvpress_available_true(self):
        with patch.object(accelerators, "_spec_exists", return_value=True):
            self.assertTrue(accelerators.kvpress_available())


class DflashAvailabilityTests(unittest.TestCase):
    """DFlash MLX / CUDA flags delegate to ``dflash.is_mlx_available`` and
    ``dflash.is_vllm_available``. Patch those to drive the branch matrix."""

    def test_mlx_available_when_helper_returns_true(self):
        with patch("dflash.is_mlx_available", return_value=True, create=True):
            self.assertTrue(accelerators.dflash_mlx_available())

    def test_mlx_unavailable_when_helper_returns_false(self):
        with patch("dflash.is_mlx_available", return_value=False, create=True):
            self.assertFalse(accelerators.dflash_mlx_available())

    def test_mlx_unavailable_when_helper_raises(self):
        with patch("dflash.is_mlx_available", side_effect=RuntimeError("boom"), create=True):
            self.assertFalse(accelerators.dflash_mlx_available())

    def test_cuda_available_when_helper_returns_true(self):
        with patch("dflash.is_vllm_available", return_value=True, create=True):
            self.assertTrue(accelerators.dflash_cuda_available())

    def test_cuda_unavailable_when_helper_returns_false(self):
        with patch("dflash.is_vllm_available", return_value=False, create=True):
            self.assertFalse(accelerators.dflash_cuda_available())

    def test_cuda_version_returns_none_when_unavailable(self):
        with patch("dflash.is_vllm_available", return_value=False, create=True):
            self.assertIsNone(accelerators.dflash_cuda_version())


class Wsl2AvailableTests(unittest.TestCase):
    def test_returns_false_off_windows(self):
        with patch.object(accelerators.sys, "platform", "linux"):
            self.assertFalse(accelerators.wsl2_available())
        with patch.object(accelerators.sys, "platform", "darwin"):
            self.assertFalse(accelerators.wsl2_available())

    def test_returns_true_when_wsl_status_succeeds(self):
        fake_result = MagicMock(returncode=0)
        with patch.object(accelerators.sys, "platform", "win32"):
            with patch.object(
                accelerators.subprocess,
                "run",
                return_value=fake_result,
            ) as run_mock:
                self.assertTrue(accelerators.wsl2_available())
                run_mock.assert_called_once()
                self.assertEqual(run_mock.call_args.args[0][0], "wsl")
                self.assertEqual(run_mock.call_args.args[0][1], "--status")

    def test_returns_false_when_wsl_status_fails(self):
        fake_result = MagicMock(returncode=1)
        with patch.object(accelerators.sys, "platform", "win32"):
            with patch.object(accelerators.subprocess, "run", return_value=fake_result):
                self.assertFalse(accelerators.wsl2_available())

    def test_returns_false_when_wsl_not_installed(self):
        with patch.object(accelerators.sys, "platform", "win32"):
            with patch.object(
                accelerators.subprocess,
                "run",
                side_effect=FileNotFoundError(),
            ):
                self.assertFalse(accelerators.wsl2_available())

    def test_returns_false_on_subprocess_timeout(self):
        with patch.object(accelerators.sys, "platform", "win32"):
            with patch.object(
                accelerators.subprocess,
                "run",
                side_effect=accelerators.subprocess.TimeoutExpired(cmd="wsl", timeout=2.0),
            ):
                self.assertFalse(accelerators.wsl2_available())


class WslDetailProbeTests(unittest.TestCase):
    """FU-056 Phase 8: WSL2 + vLLM-bridge detail probes. All four
    return safely-default values off Windows so the capability layer
    never throws on a macOS / Linux host."""

    def test_default_distro_off_windows_returns_none(self):
        with patch.object(accelerators.sys, "platform", "linux"):
            self.assertIsNone(accelerators.wsl_default_distro())

    def test_default_distro_parses_status_output(self):
        # ``wsl --status`` emits UTF-16 LE. Synthesize that shape so
        # the decoder is exercised.
        status_text = (
            "Default Distribution: Ubuntu-24.04\r\n"
            "Default Version: 2\r\n"
        )
        fake_result = MagicMock(
            returncode=0,
            stdout=status_text.encode("utf-16-le"),
        )
        with patch.object(accelerators.sys, "platform", "win32"):
            with patch.object(accelerators.subprocess, "run", return_value=fake_result):
                self.assertEqual(accelerators.wsl_default_distro(), "Ubuntu-24.04")

    def test_default_distro_returns_none_when_no_default_line(self):
        fake_result = MagicMock(
            returncode=0,
            stdout="Default Version: 2\r\n".encode("utf-16-le"),
        )
        with patch.object(accelerators.sys, "platform", "win32"):
            with patch.object(accelerators.subprocess, "run", return_value=fake_result):
                self.assertIsNone(accelerators.wsl_default_distro())

    def test_default_distro_returns_none_when_wsl_exits_nonzero(self):
        fake_result = MagicMock(returncode=1, stdout=b"")
        with patch.object(accelerators.sys, "platform", "win32"):
            with patch.object(accelerators.subprocess, "run", return_value=fake_result):
                self.assertIsNone(accelerators.wsl_default_distro())

    def test_cuda_available_off_windows_returns_false(self):
        with patch.object(accelerators.sys, "platform", "darwin"):
            self.assertFalse(accelerators.wsl_cuda_available())

    def test_cuda_available_true_when_nvidia_smi_lists_gpu(self):
        fake_result = MagicMock(
            returncode=0,
            stdout=b"GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-...)\n",
        )
        with patch.object(accelerators.sys, "platform", "win32"):
            with patch.object(accelerators.subprocess, "run", return_value=fake_result):
                self.assertTrue(accelerators.wsl_cuda_available())

    def test_cuda_available_false_when_nvidia_smi_returns_empty(self):
        fake_result = MagicMock(returncode=0, stdout=b"")
        with patch.object(accelerators.sys, "platform", "win32"):
            with patch.object(accelerators.subprocess, "run", return_value=fake_result):
                self.assertFalse(accelerators.wsl_cuda_available())

    def test_cuda_available_false_when_nvidia_smi_missing(self):
        fake_result = MagicMock(returncode=127, stdout=b"")
        with patch.object(accelerators.sys, "platform", "win32"):
            with patch.object(accelerators.subprocess, "run", return_value=fake_result):
                self.assertFalse(accelerators.wsl_cuda_available())

    def test_vllm_available_off_windows_returns_false(self):
        with patch.object(accelerators.sys, "platform", "linux"):
            self.assertFalse(accelerators.wsl_vllm_available())

    def test_vllm_available_true_when_import_returns_zero(self):
        fake_result = MagicMock(returncode=0, stdout=b"", stderr=b"")
        with patch.object(accelerators.sys, "platform", "win32"):
            with patch.object(accelerators.subprocess, "run", return_value=fake_result):
                self.assertTrue(accelerators.wsl_vllm_available())

    def test_vllm_available_false_when_import_fails(self):
        fake_result = MagicMock(returncode=1, stdout=b"", stderr=b"ModuleNotFoundError")
        with patch.object(accelerators.sys, "platform", "win32"):
            with patch.object(accelerators.subprocess, "run", return_value=fake_result):
                self.assertFalse(accelerators.wsl_vllm_available())

    def test_vllm_version_returns_none_when_unavailable(self):
        with patch.object(accelerators, "wsl_vllm_available", return_value=False):
            self.assertIsNone(accelerators.wsl_vllm_version())

    def test_vllm_version_reads_stdout_when_available(self):
        # Two-shot: ``wsl_vllm_available`` runs the import-check
        # subprocess, then ``wsl_vllm_version`` runs a second subprocess
        # to read ``__version__``. We stub the version-fetch result.
        fake_version_result = MagicMock(returncode=0, stdout=b"0.6.3\n", stderr=b"")
        with patch.object(accelerators, "wsl_vllm_available", return_value=True):
            with patch.object(accelerators.sys, "platform", "win32"):
                with patch.object(accelerators.subprocess, "run", return_value=fake_version_result):
                    self.assertEqual(accelerators.wsl_vllm_version(), "0.6.3")


class BackendCapabilitiesToDictTests(unittest.TestCase):
    """The frontend reads accelerator flags via ``/api/health``. Pin
    the serialized payload so a future field rename (or a forgetful
    ``to_dict`` update) gets caught here rather than in a vague UI bug."""

    def test_to_dict_includes_every_accelerator_field(self):
        caps = BackendCapabilities(
            pythonExecutable="/x/python",
            mlxAvailable=False,
            mlxLmAvailable=False,
            mlxUsable=False,
            nunchakuAvailable=True,
            nunchakuVersion="1.2.1",
            sageattentionAvailable=True,
            sageattentionVersion="2.2.0",
            dflashMlxAvailable=False,
            dflashMlxVersion=None,
            dflashCudaAvailable=True,
            dflashCudaVersion="0.1.0",
            triattentionAvailable=True,
            triattentionVersion="0.2.0",
            kvpressAvailable=False,
            kvpressVersion=None,
            wsl2Available=True,
        )
        payload = caps.to_dict()
        for key in (
            "nunchakuAvailable",
            "nunchakuVersion",
            "sageattentionAvailable",
            "sageattentionVersion",
            "dflashMlxAvailable",
            "dflashMlxVersion",
            "dflashCudaAvailable",
            "dflashCudaVersion",
            "triattentionAvailable",
            "triattentionVersion",
            "kvpressAvailable",
            "kvpressVersion",
            "wsl2Available",
        ):
            self.assertIn(key, payload, f"{key} missing from to_dict payload")
        self.assertTrue(payload["nunchakuAvailable"])
        self.assertEqual(payload["sageattentionVersion"], "2.2.0")
        self.assertFalse(payload["dflashMlxAvailable"])
        self.assertTrue(payload["wsl2Available"])

    def test_defaults_render_as_false_and_none(self):
        caps = BackendCapabilities(
            pythonExecutable="/x/python",
            mlxAvailable=False,
            mlxLmAvailable=False,
            mlxUsable=False,
        )
        payload = caps.to_dict()
        for flag in (
            "nunchakuAvailable",
            "sageattentionAvailable",
            "dflashMlxAvailable",
            "dflashCudaAvailable",
            "triattentionAvailable",
            "kvpressAvailable",
            "wsl2Available",
        ):
            self.assertFalse(payload[flag], f"{flag} should default False")
        for version in (
            "nunchakuVersion",
            "sageattentionVersion",
            "dflashMlxVersion",
            "dflashCudaVersion",
            "triattentionVersion",
            "kvpressVersion",
        ):
            self.assertIsNone(payload[version], f"{version} should default None")


if __name__ == "__main__":
    unittest.main()
