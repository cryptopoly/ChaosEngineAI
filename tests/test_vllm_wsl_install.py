"""Tests for FU-056 Phase 8 install-vllm-wsl endpoints.

The install itself runs in a background thread + shells out to
``wsl --``, which we don't want to actually execute under pytest
(the wsl subprocess can take 5-15 min on a real host). These tests
pin the route contract + the platform gate so the endpoint can't
silently regress shape.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend_service.routes.setup.vllm_wsl import (
    _INSTALL_PHASES,
    _JOB,
    _VllmWslJobState,
    router as vllm_wsl_router,
)


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(vllm_wsl_router)
    return app


class VllmWslJobStateShapeTests(unittest.TestCase):
    def test_to_dict_exposes_install_panel_fields(self):
        # The shared InstallLogPanel reads these keys; pin them so a
        # backend refactor can't silently break the frontend renderer.
        state = _VllmWslJobState()
        payload = state.to_dict()
        for key in (
            "id",
            "phase",
            "message",
            "packageCurrent",
            "packageIndex",
            "packageTotal",
            "percent",
            "targetDir",
            "error",
            "startedAt",
            "finishedAt",
            "attempts",
            "done",
        ):
            self.assertIn(key, payload, f"{key} missing from to_dict()")

    def test_phases_match_documented_step_order(self):
        # Five user-visible steps: preflight (CUDA check), venv,
        # pip-upgrade, pip-vllm (the long one), verify (import works).
        self.assertEqual(
            _INSTALL_PHASES,
            ("preflight", "venv", "pip-upgrade", "pip-vllm", "verify"),
        )


class VllmWslEndpointTests(unittest.TestCase):
    """The POST endpoint rejects on non-Windows hosts with HTTP 400.
    Status GET is always allowed so polling works after a Windows
    user runs the install, even if their dev box accidentally swaps
    platforms mid-poll."""

    def setUp(self) -> None:
        self.client = TestClient(_make_app())
        # Reset the singleton job so test ordering doesn't matter.
        # The module-level state is the only contract the frontend
        # talks to — leaving "done" state from a previous test would
        # taint the next start.
        _JOB.id = ""
        _JOB.phase = "idle"
        _JOB.message = ""
        _JOB.package_current = None
        _JOB.package_index = 0
        _JOB.package_total = len(_INSTALL_PHASES)
        _JOB.percent = 0.0
        _JOB.target_dir = None
        _JOB.error = None
        _JOB.started_at = 0.0
        _JOB.finished_at = 0.0
        _JOB.attempts = []
        _JOB.done = False

    def test_post_rejects_off_windows(self):
        with patch.object(sys, "platform", "linux"):
            response = self.client.post("/api/setup/install-vllm-wsl")
        self.assertEqual(response.status_code, 400)
        body = response.json()
        # ``localized_detail`` wraps the message in {message, localized,
        # locale} — the user-facing string is in ``message``.
        detail = body.get("detail")
        self.assertIsInstance(detail, dict)
        self.assertIn("Windows", detail.get("message", ""))

    def test_status_returns_idle_when_no_install_started(self):
        response = self.client.get("/api/setup/install-vllm-wsl/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["phase"], "idle")
        self.assertFalse(payload["done"])

    def test_post_on_windows_returns_job_state(self):
        # Patch the background worker so the test doesn't actually shell
        # out to ``wsl --`` (which would hang or fail on CI). The thread
        # still starts but the worker function is a no-op.
        from backend_service.routes.setup import vllm_wsl as module

        with patch.object(sys, "platform", "win32"):
            with patch.object(module, "_job_worker", lambda: None):
                response = self.client.post("/api/setup/install-vllm-wsl")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["phase"], "preflight")
        self.assertEqual(payload["packageTotal"], len(_INSTALL_PHASES))
        self.assertEqual(payload["targetDir"], "~/.chaosengine/vllm-venv")
        self.assertTrue(payload["id"].startswith("vllm-wsl-"))


if __name__ == "__main__":
    unittest.main()
