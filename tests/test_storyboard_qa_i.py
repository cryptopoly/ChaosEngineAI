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


class MalformedPlanReplyTests(StoreCase):
    """A small chat model answers with valid JSON of the wrong shape; the plan routes must answer a readable
    error or a cleaned plan, never a 500."""

    def setUp(self):
        super().setUp()
        from backend_service.storyboard.ops import run_op

        project = self.store.create("Plan check")
        for title in ("one", "two"):
            project = run_op(self.store, project.id, project.revision, "add_shot", {"title": title, "stillPrompt": "tea"})
        self.project = project

    def ask(self, fn, text, *args):
        from tests.test_storyboard_plan import FakeRuntime, fake_state

        return fn(fake_state(FakeRuntime([text])), self.project, self.store.folder(self.project.id), *args)

    def test_direct_with_lists_of_the_wrong_type(self):
        from backend_service.storyboard import planner

        for text in ('{"reply": "ok", "edits": 5, "add": 3, "remove": 7}',
                     '{"reply": "ok", "edits": {"shot": 1}, "add": "x", "remove": "12"}',
                     '{"reply": "ok", "remove": [{"shot": 1}, null, [2]]}',
                     '{"reply": "ok", "remove": [1e999], "edits": [{"shot": 1e999, "title": "x"}]}'):
            out = self.ask(planner.direct, text, "shorter")
            self.assertEqual(out["changes"]["edits"], [])
            self.assertEqual(out["changes"]["add"], [])
            self.assertEqual(out["changes"]["remove"], [])

    def test_infinite_or_huge_seconds_are_held_to_the_clip_limit(self):
        from backend_service.storyboard import planner

        out = self.ask(planner.direct, '{"reply": "ok", "edits": [{"shot": 1, "seconds": 1e999}], '
                                       '"add": [{"after": 1, "stillPrompt": "cup", "seconds": -Infinity}]}', "x")
        self.assertEqual(out["changes"]["edits"][0]["seconds"], 10.0)
        self.assertEqual(out["changes"]["add"][0]["seconds"], 1.0)
        out = self.ask(planner.propose, '{"shots": [{"title": "a", "stillPrompt": "b", "seconds": Infinity}]}',
                       "a film", None, None)
        self.assertEqual(out["shots"][0]["seconds"], 10.0)

    def test_propose_with_shots_that_are_not_a_list(self):
        from backend_service.storyboard import planner
        from backend_service.storyboard.planner import PlanError

        for text in ('{"shots": 4}', '{"shots": true}', '{"shots": null, "note": 3}'):
            with self.assertRaises(PlanError) as caught:
                self.ask(planner.propose, text, "a film", None, None)
            self.assertEqual(caught.exception.code, "bad-reply")


class NonAsciiExportNameTests(StoreCase):
    def test_a_bundle_of_a_project_with_a_non_ascii_name_can_be_fetched(self):
        """A Japanese or accented project name (the sample film is named in the user's language) made a bundle
        file the files route refused with 404, so the export could not be downloaded."""
        from fastapi.testclient import TestClient

        from backend_service.app import create_app

        client = TestClient(create_app(api_token="app-token"))
        headers = {"Authorization": "Bearer app-token"}
        for name in ("灯台の夜", "Café crème", "Ein Fluss · Größe"):
            created = client.post("/api/storyboard/projects", json={"name": name}, headers=headers).json()
            pid = created["project"]["id"]
            bundle = client.post(f"/api/storyboard/projects/{pid}/bundle", headers=headers).json()
            got = client.get(f"/api/storyboard/projects/{pid}/files", params={"path": bundle["file"]}, headers=headers)
            self.assertEqual(got.status_code, 200, (name, bundle["file"], got.text))
            self.assertEqual(len(got.content), bundle["bytes"])
        for bad in ("exports/../project.json", "exports/.hidden", "exports/a/b.mp4", "exports/a\\b.mp4", "exports/"):
            got = client.get(f"/api/storyboard/projects/{pid}/files", params={"path": bad}, headers=headers)
            self.assertEqual(got.status_code, 404, bad)


if __name__ == "__main__":
    unittest.main()
