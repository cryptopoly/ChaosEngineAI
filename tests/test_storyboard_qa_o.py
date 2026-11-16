"""Regression tests for the S-O polish pass (v0.10.0)."""
from __future__ import annotations

import unittest

from tests.test_storyboard_routes import StoryboardRouteCase


class RenderStatusHasPreviewTests(StoryboardRouteCase):
    """The Mixer asked for exports/preview.mp4 on every project and got a 404 twice on one never rendered: the render
    status now says whether the preview file is there, so the Mixer asks only then."""

    def test_has_preview_follows_the_file(self):
        pid = self.post("/projects", {"name": "Never rendered"})["project"]["id"]
        status = self.get(f"/projects/{pid}/render")
        self.assertIsNone(status["render"])
        self.assertIs(status["hasPreview"], False)

        folder = next((self.base / "storyboards").rglob("project.json")).parent
        (folder / "exports").mkdir(exist_ok=True)
        (folder / "exports" / "preview.mp4").write_bytes(b"\x00")
        self.assertIs(self.get(f"/projects/{pid}/render")["hasPreview"], True)


if __name__ == "__main__":
    unittest.main()
