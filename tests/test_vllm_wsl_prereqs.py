"""Preflight / verify build-prerequisite checks for the WSL vLLM installer,
the root-cause-first log helper, and the capability refresh on completion.

``wsl --`` is never run: ``_run_wsl_step`` is replaced by a fake that
answers by command text.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend_service.inference._utils import _read_log_with_root_cause
from backend_service.routes.setup import vllm_wsl as module


def _fake_wsl(*, compiler=True, header=True, venv=True, import_ok=True):
    calls: list[str] = []

    def run(cmd: str, timeout: float):
        calls.append(cmd)
        if cmd.startswith("nvidia-smi"):
            return 0, "GPU 0: RTX"
        if "ensurepip" in cmd:
            return (0, "") if venv else (1, "No module named ensurepip")
        if "command -v cc" in cmd:
            return (0, "") if compiler else (1, "")
        if "Python.h" in cmd:
            return 0, ("True\n" if header else "False\n")
        if "import vllm" in cmd:
            return (0, "0.34.0\n") if import_ok else (1, "ImportError")
        return 0, ""

    return run, calls


def _reset_job() -> None:
    module._JOB.__init__()


class PreflightPrereqTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_job()

    def _run(self, **kw):
        fake, calls = _fake_wsl(**kw)
        with patch.object(module, "_run_wsl_step", fake):
            module._run_install_steps()
        return module._JOB, calls

    def test_missing_compiler(self):
        job, calls = self._run(compiler=False)
        self.assertEqual(job.phase, "error")
        self.assertIn("a C compiler (build-essential)", job.error)
        self.assertIn("sudo apt update && sudo apt install -y build-essential\n", job.error)
        self.assertNotIn("python3-dev", job.error)
        self.assertFalse(any("venv " in c for c in calls))  # stopped at preflight

    def test_missing_python_h(self):
        job, _ = self._run(header=False)
        self.assertEqual(job.phase, "error")
        self.assertIn("Python.h", job.error)
        self.assertIn("apt install -y python3-dev\n", job.error)
        self.assertNotIn("build-essential", job.error)

    def test_missing_everything_gives_one_combined_message(self):
        job, _ = self._run(compiler=False, header=False, venv=False)
        self.assertEqual(
            job.error,
            "WSL is missing python3-venv, a C compiler (build-essential), "
            "the Python development headers (python3-dev, Python.h). "
            "Open a WSL shell and run:\n"
            "    sudo apt update && sudo apt install -y "
            "python3-venv build-essential python3-dev\n"
            "then retry this installer. (On a distro other than Debian or "
            "Ubuntu, install the equivalent packages.)",
        )

    def test_everything_present_passes_and_finishes_ready(self):
        job, _ = self._run()
        self.assertEqual(job.phase, "done")
        self.assertIn("vLLM ready", job.message)

    def test_verify_catches_missing_compiler_after_pip(self):
        # Preflight sees a compiler; it disappears before verify.
        fake, _ = _fake_wsl()
        seen = {"cc": 0}

        def run(cmd: str, timeout: float):
            if "command -v cc" in cmd:
                seen["cc"] += 1
                return (0, "") if seen["cc"] == 1 else (1, "")
            return fake(cmd, timeout)

        with patch.object(module, "_run_wsl_step", run):
            module._run_install_steps()
        job = module._JOB
        self.assertEqual(job.phase, "error")
        self.assertEqual(job.attempts[-1]["phase"], "verify")
        self.assertIn("a C compiler", job.error)
        self.assertNotIn("vLLM ready", job.message)

    def test_pip_step_message_states_time_and_size(self):
        messages: list[str] = []
        fake, _ = _fake_wsl()

        def run(cmd: str, timeout: float):
            if "pip install vllm" in cmd:
                messages.append(module._JOB.message)
            return fake(cmd, timeout)

        with patch.object(module, "_run_wsl_step", run):
            module._run_install_steps()
        self.assertEqual(len(messages), 1)
        self.assertIn("10 to 15 minutes", messages[0])
        self.assertIn("9 GB", messages[0])


class CapabilityRefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_job()

    def tearDown(self) -> None:
        module._STATE = None

    def test_refresh_called_on_success_and_on_failure(self):
        for compiler in (True, False):
            state = MagicMock()
            module._STATE = state
            fake, _ = _fake_wsl(compiler=compiler)
            with patch.object(module, "_run_wsl_step", fake):
                module._job_worker()
            state.runtime.refresh_capabilities.assert_called_once_with(force=True)


class RootCauseLogTests(unittest.TestCase):
    def _log(self, lines: list[str]) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "vllm.log"
        tmp.write_text("\n".join(lines), encoding="utf-8")
        return tmp

    def _traceback(self, cause: str) -> list[str]:
        head = ["INFO starting"] + [f"INFO step {i}" for i in range(5)]
        head += ["(EngineCore pid=1) ERROR Traceback (most recent call last):", cause]
        filler = [f"  File \"x{i}.py\", line {i}, in f" for i in range(45)]
        return head + filler + ["RuntimeError: Engine core initialization failed"]

    def test_c_compiler_cause_first(self):
        cause = "RuntimeError: Failed to find C compiler. Please specify via CC environment variable"
        text = _read_log_with_root_cause(self._log(self._traceback(cause)))
        self.assertTrue(text.startswith("Likely cause:\n" + cause))
        self.assertIn("Log tail:", text)
        self.assertTrue(text.endswith("Engine core initialization failed"))

    def test_python_h_cause_first(self):
        cause = "triton/backends/nvidia/driver.c:9:10: fatal error: Python.h: No such file or directory"
        text = _read_log_with_root_cause(self._log(self._traceback(cause)))
        self.assertTrue(text.startswith("Likely cause:\n" + cause))

    def test_at_most_three_causes(self):
        lines = [f"fatal error: thing{i}" for i in range(6)] + ["x"] * 50
        text = _read_log_with_root_cause(self._log(lines))
        head = text.split("\n\nLog tail:")[0]
        self.assertEqual(len(head.splitlines()), 4)  # header + 3

    def test_no_match_keeps_plain_tail(self):
        path = self._log([f"INFO line {i}" for i in range(60)])
        from backend_service.inference._utils import _read_text_tail

        self.assertEqual(_read_log_with_root_cause(path), _read_text_tail(path))

    def test_missing_file_is_empty(self):
        self.assertEqual(_read_log_with_root_cause(Path("nope.log")), "")
        self.assertEqual(_read_log_with_root_cause(None), "")


if __name__ == "__main__":
    unittest.main()
