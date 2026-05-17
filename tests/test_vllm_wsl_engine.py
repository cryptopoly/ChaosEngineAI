"""Tests for the FU-056 Phase 8 follow-up: VllmWslEngine.

Pinned at the unit level — we don't actually shell out to ``wsl --``
or talk to a real vLLM server. ``subprocess.Popen`` + ``_http_json``
are mocked so the engine's lifecycle (spawn → wait_for_server →
generate → unload) gets exercised without a 90-second model load.

Two layers:
  - ``windows_path_to_wsl`` is a pure function and gets exhaustive
    coverage against the path-flavour matrix.
  - The engine class gets happy-path + Windows-only gate + missing-
    capability rejection tests.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from backend_service.inference.base import BackendCapabilities
from backend_service.inference.vllm_wsl_engine import (
    VllmWslEngine,
    windows_path_to_wsl,
)


class WindowsPathToWslTests(unittest.TestCase):
    """Pure path-translation matrix — no side effects."""

    def test_translates_backslash_drive_letter_path(self):
        self.assertEqual(
            windows_path_to_wsl(r"C:\Users\Dan\AI_Models\Qwen3-7B"),
            "/mnt/c/Users/Dan/AI_Models/Qwen3-7B",
        )

    def test_translates_forward_slash_drive_letter_path(self):
        self.assertEqual(
            windows_path_to_wsl("C:/Users/Dan/AI_Models/Qwen3-7B"),
            "/mnt/c/Users/Dan/AI_Models/Qwen3-7B",
        )

    def test_lowercases_drive_letter(self):
        # WSL expects ``/mnt/c/...`` not ``/mnt/C/...``.
        self.assertEqual(
            windows_path_to_wsl(r"D:\Models"),
            "/mnt/d/Models",
        )

    def test_passes_through_existing_wsl_path(self):
        self.assertEqual(
            windows_path_to_wsl("/home/dan/models/Qwen3-7B"),
            "/home/dan/models/Qwen3-7B",
        )

    def test_passes_through_hf_repo_id(self):
        # vLLM accepts ``org/name`` directly and downloads to its
        # HF cache. We mustn't mangle it into a path.
        self.assertEqual(
            windows_path_to_wsl("Qwen/Qwen3.5-7B"),
            "Qwen/Qwen3.5-7B",
        )

    def test_passes_through_empty_string(self):
        self.assertEqual(windows_path_to_wsl(""), "")

    def test_passes_through_unc_path(self):
        # UNC paths (\\\\server\\share) aren't translated — vLLM wouldn't
        # load from them inside WSL anyway. Just don't crash.
        unc = r"\\server\share\models"
        self.assertEqual(windows_path_to_wsl(unc), unc)

    def test_passes_through_relative_path(self):
        # Relative paths have no drive letter; leave alone.
        self.assertEqual(windows_path_to_wsl(r"models\Qwen3"), r"models\Qwen3")


def _make_caps(*, wsl_vllm: bool = True, distro: str | None = "Ubuntu-24.04") -> BackendCapabilities:
    """Build a capabilities snapshot with the WSL bridge in the
    requested state. Default = "ready"."""
    return BackendCapabilities(
        pythonExecutable="/x/python",
        mlxAvailable=False,
        mlxLmAvailable=False,
        mlxUsable=False,
        wsl2Available=wsl_vllm,
        wslDistroName=distro,
        wslCudaAvailable=wsl_vllm,
        wslVllmAvailable=wsl_vllm,
        wslVllmVersion="0.6.3" if wsl_vllm else None,
    )


class VllmWslEngineGatesTests(unittest.TestCase):
    """Pre-spawn validation: platform + capability checks should
    raise *before* any subprocess is touched."""

    def test_load_rejects_off_windows(self):
        engine = VllmWslEngine(_make_caps(wsl_vllm=True))
        with patch.object(sys, "platform", "linux"):
            with self.assertRaises(RuntimeError) as ctx:
                engine.load_model(
                    model_ref="Qwen/Qwen3.5-7B",
                    model_name="Qwen3.5-7B",
                    canonical_repo="Qwen/Qwen3.5-7B",
                    source="catalog",
                    backend="vllm",
                    path=None,
                    runtime_target=None,
                    cache_strategy="native",
                    cache_bits=0,
                    fp16_layers=0,
                    fused_attention=False,
                    fit_model_in_memory=True,
                    context_tokens=8192,
                )
        self.assertIn("Windows-only", str(ctx.exception))

    def test_load_rejects_when_wsl_vllm_missing(self):
        engine = VllmWslEngine(_make_caps(wsl_vllm=False))
        with patch.object(sys, "platform", "win32"):
            with self.assertRaises(RuntimeError) as ctx:
                engine.load_model(
                    model_ref="Qwen/Qwen3.5-7B",
                    model_name="Qwen3.5-7B",
                    canonical_repo="Qwen/Qwen3.5-7B",
                    source="catalog",
                    backend="vllm",
                    path=None,
                    runtime_target=None,
                    cache_strategy="native",
                    cache_bits=0,
                    fp16_layers=0,
                    fused_attention=False,
                    fit_model_in_memory=True,
                    context_tokens=8192,
                )
        # The error points the user at the install panel rather than
        # leaving them guessing what to do next.
        self.assertIn("WSL2 vLLM bridge", str(ctx.exception))


class VllmWslEngineCommandTests(unittest.TestCase):
    """Argv composition — checks each flag is present and ordered
    so the WSL command stays valid for the upstream parser."""

    def test_build_command_includes_required_flags(self):
        engine = VllmWslEngine(_make_caps())
        command = engine._build_wsl_command(
            model_arg="Qwen/Qwen3.5-7B",
            port=8000,
            max_model_len=8192,
        )

        # Prefix is the wsl entry-point + arg separator. Without ``--``
        # the wsl CLI tries to interpret the rest as wsl options.
        self.assertEqual(command[0], "wsl")
        self.assertEqual(command[1], "--")

        # The venv-bound Python invocation — relative to the WSL
        # user's $HOME via the leading ~. ``wsl --`` expands the ~.
        self.assertIn("~/.chaosengine/vllm-venv/bin/python", command)
        self.assertIn("-m", command)
        self.assertIn("vllm.entrypoints.openai.api_server", command)

        # User-driven flags.
        self.assertIn("--model", command)
        self.assertIn("Qwen/Qwen3.5-7B", command)
        self.assertIn("--port", command)
        self.assertIn("8000", command)
        self.assertIn("--max-model-len", command)
        self.assertIn("8192", command)

        # Safety: bound to loopback so the model isn't exposed to the
        # LAN, and ``--trust-remote-code`` covers repos like Qwen3-VL.
        self.assertIn("--host", command)
        self.assertIn("127.0.0.1", command)
        self.assertIn("--trust-remote-code", command)


class VllmWslEngineLifecycleTests(unittest.TestCase):
    """Happy-path lifecycle with the subprocess + HTTP probe mocked
    out. We never actually shell out to wsl.exe."""

    def test_load_spawns_subprocess_and_polls_health(self):
        engine = VllmWslEngine(_make_caps())

        # Fake the subprocess: poll() returns None (still running),
        # PID is a known int. ``terminate`` / ``wait`` are mocked so
        # ``unload_model`` doesn't hang.
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.pid = 4242
        fake_proc.wait.return_value = 0

        with patch.object(sys, "platform", "win32"):
            with patch(
                "backend_service.inference.vllm_wsl_engine.subprocess.Popen",
                return_value=fake_proc,
            ) as popen_mock:
                with patch(
                    "backend_service.inference.vllm_wsl_engine._http_json",
                    return_value={},  # ``/health`` returns OK immediately
                ) as http_mock:
                    with patch(
                        "backend_service.inference.vllm_wsl_engine._find_open_port",
                        return_value=8765,
                    ):
                        info = engine.load_model(
                            model_ref="Qwen/Qwen3.5-7B",
                            model_name="Qwen3.5-7B",
                            canonical_repo="Qwen/Qwen3.5-7B",
                            source="catalog",
                            backend="vllm",
                            path=None,
                            runtime_target=None,
                            cache_strategy="native",
                            cache_bits=0,
                            fp16_layers=0,
                            fused_attention=False,
                            fit_model_in_memory=True,
                            context_tokens=8192,
                        )

        # Subprocess was spawned exactly once with the WSL argv.
        popen_mock.assert_called_once()
        spawned_argv = popen_mock.call_args.args[0]
        self.assertEqual(spawned_argv[0], "wsl")

        # Health probe was hit.
        http_mock.assert_called()

        # Loaded info reflects the spawn.
        self.assertEqual(info.engine, "vllm-wsl")
        self.assertEqual(info.ref, "Qwen/Qwen3.5-7B")
        self.assertEqual(engine.port, 8765)
        self.assertEqual(engine.process_pid(), 4242)

    def test_load_translates_windows_path_in_runtime_target(self):
        engine = VllmWslEngine(_make_caps())

        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.pid = 7

        with patch.object(sys, "platform", "win32"):
            with patch(
                "backend_service.inference.vllm_wsl_engine.subprocess.Popen",
                return_value=fake_proc,
            ) as popen_mock:
                with patch(
                    "backend_service.inference.vllm_wsl_engine._http_json",
                    return_value={},
                ):
                    with patch(
                        "backend_service.inference.vllm_wsl_engine._find_open_port",
                        return_value=9000,
                    ):
                        engine.load_model(
                            model_ref="Qwen/Qwen3.5-7B",
                            model_name="Qwen3.5-7B",
                            canonical_repo=None,
                            source="local",
                            backend="vllm",
                            path=r"C:\Users\Dan\AI_Models\Qwen3-7B",
                            runtime_target=None,
                            cache_strategy="native",
                            cache_bits=0,
                            fp16_layers=0,
                            fused_attention=False,
                            fit_model_in_memory=True,
                            context_tokens=8192,
                        )

        spawned_argv = popen_mock.call_args.args[0]
        # The model arg should have been translated into the WSL
        # /mnt/c/... form so vLLM can find the weights from inside WSL.
        self.assertIn("/mnt/c/Users/Dan/AI_Models/Qwen3-7B", spawned_argv)


if __name__ == "__main__":
    unittest.main()
