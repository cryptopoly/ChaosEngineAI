"""A native-backend probe killed by a signal must not turn MLX off for the session.

The probe result is cached, so one SIGTERM at launch (seen as "probe exited with
code -15" in the v0.9.8 app) used to make every MLX model fail to load until the
app restarted.
"""
from __future__ import annotations

import unittest
from unittest import mock

from backend_service.inference import capabilities

_GOOD = {"mlxAvailable": True, "mlxLmAvailable": True, "mlxUsable": True, "message": None}


def _probe(results):
    calls = []

    def fake(command, **kwargs):
        calls.append(command)
        return results[min(len(calls), len(results)) - 1]

    with mock.patch.object(capabilities, "_json_subprocess", side_effect=fake), \
            mock.patch.object(capabilities, "_resolve_llama_server", return_value=None), \
            mock.patch.object(capabilities, "_resolve_llama_server_turbo", return_value=None), \
            mock.patch.object(capabilities, "_resolve_llama_cli", return_value=None):
        return capabilities._probe_native_backends(), calls


class ProbeRetryTests(unittest.TestCase):
    def test_a_probe_killed_by_sigterm_is_retried_once(self) -> None:
        caps, calls = _probe([(-15, None, ""), (0, _GOOD, "")])
        self.assertTrue(caps.mlxUsable)
        self.assertEqual(len(calls), 2)

    def test_a_probe_that_keeps_dying_reports_the_exit_code(self) -> None:
        caps, calls = _probe([(-15, None, "")])
        self.assertFalse(caps.mlxUsable)
        self.assertEqual(caps.mlxMessage, "probe exited with code -15")
        self.assertEqual(len(calls), 2)

    def test_a_timeout_is_not_retried(self) -> None:
        caps, calls = _probe([(-1, None, "timed out after 45.0 seconds")])
        self.assertFalse(caps.mlxUsable)
        self.assertEqual(len(calls), 1)

    def test_a_working_probe_runs_once(self) -> None:
        caps, calls = _probe([(0, _GOOD, "")])
        self.assertTrue(caps.mlxUsable)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()


class SelectEngineReprobeTests(unittest.TestCase):
    """Loading an MLX model re-runs a failed probe once before refusing."""

    def _controller(self, caps):
        from backend_service.inference import RuntimeController

        with mock.patch("backend_service.inference.get_backend_capabilities", return_value=caps):
            return RuntimeController()

    @staticmethod
    def _caps(usable: bool):
        from backend_service.inference.base import BackendCapabilities

        return BackendCapabilities(
            pythonExecutable="/usr/bin/python3",
            mlxAvailable=usable,
            mlxLmAvailable=usable,
            mlxUsable=usable,
            mlxMessage=None if usable else "probe exited with code -15",
            ggufAvailable=False,
        )

    def _select(self, controller):
        return controller._select_engine(backend="mlx", runtime_target=None, path=None, model_ref="m")

    def test_a_stale_failed_probe_is_refreshed_before_refusing(self) -> None:
        from backend_service.inference import MLXWorkerEngine

        controller = self._controller(self._caps(False))
        with mock.patch("backend_service.inference.controller.get_backend_capabilities", return_value=self._caps(True)) as probe:
            engine = self._select(controller)
        self.assertIsInstance(engine, MLXWorkerEngine)
        probe.assert_called_once_with(force=True)

    def test_mlx_still_refused_when_the_second_probe_also_fails(self) -> None:
        controller = self._controller(self._caps(False))
        with mock.patch("backend_service.inference.controller.get_backend_capabilities", return_value=self._caps(False)):
            with self.assertRaisesRegex(RuntimeError, "probe exited with code -15"):
                self._select(controller)
