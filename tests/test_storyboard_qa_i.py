"""Regression tests for defects the v0.10.0 Storyboard QA pass (package S-I) found by trying to break the merged build."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.storyboard.store import ProjectStore


class StoreCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        env = mock.patch.dict(os.environ, {"CHAOSENGINE_STORYBOARDS_DIR": str(self.base / "storyboards")})
        env.start()
        self.addCleanup(env.stop)
        self.store = ProjectStore(self.base / "storyboards")


class PrecheckDraftTests(StoreCase):
    def test_a_draft_final_pass_is_refused_before_a_plan_card(self):
        """A draft with ``what: final`` (or a redo) is refused by the batch; the gate must say so before the card."""
        from backend_service.storyboard.ops import run_op
        from backend_service.tools import studio

        project = self.store.create("Draft check")
        project = run_op(self.store, project.id, project.revision, "add_shot", {"title": "a", "stillPrompt": "tea"})
        shot = project.shots[0]
        project = run_op(self.store, project.id, project.revision, "set_kept", {"shotId": shot.id, "kept": True})
        base = {"project_id": project.id, "revision": project.revision}
        self.assertIsNone(studio.precheck("storyboard_generate", {**base, "what": "final"}))
        problem = studio.precheck("storyboard_generate", {**base, "what": "final", "draft": True})
        self.assertIsNotNone(problem)
        self.assertIn("draft", problem)


if __name__ == "__main__":
    unittest.main()
