"""The hub's generated tracker pages must match CLAUDE.md, and the hub must build."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import hub_tracker  # noqa: E402


class HubTrackerTests(unittest.TestCase):
    def test_every_follow_up_row_is_parsed(self) -> None:
        rows = hub_tracker.parse((ROOT / "CLAUDE.md").read_text(encoding="utf-8"))
        text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        expected = sum(1 for line in text.splitlines() if hub_tracker._ROW_START.match(line))
        self.assertEqual(len(rows), expected)
        self.assertGreater(len(rows), 100)
        self.assertEqual(len({row.fu for row in rows}), len(rows), "duplicate FU ids in CLAUDE.md")

    def test_committed_tracker_pages_are_current(self) -> None:
        stale = [
            name for name, page in hub_tracker.build().items()
            if not (hub_tracker.OUT_DIR / name).exists()
            or (hub_tracker.OUT_DIR / name).read_text(encoding="utf-8") != page
        ]
        self.assertEqual(stale, [], "run: python3 tools/hub_tracker.py")

    def test_repo_links_in_rows_point_at_github_not_missing_hub_files(self) -> None:
        for name in ("open.md", "closed.md"):
            body = (hub_tracker.OUT_DIR / name).read_text(encoding="utf-8")
            self.assertNotIn("](backend_service/", body)
            self.assertNotIn("](src/", body)

    def test_a_row_is_closed_when_struck_or_shipped_without_open_points(self) -> None:
        rows = {
            "struck": hub_tracker.Row("FU-1", True, "x", "**Shipped.**", "done"),
            "shipped": hub_tracker.Row("FU-2", False, "x", "**Shipped 2026-10-01.**", "all done"),
            "open points": hub_tracker.Row("FU-3", False, "x", "**Shipped 2026-10-01.**", "**Open:** a thing"),
            "pending": hub_tracker.Row("FU-4", False, "x", "**Decision pending.**", "x"),
        }
        self.assertEqual({k: r.closed for k, r in rows.items()},
                         {"struck": True, "shipped": True, "open points": False, "pending": False})

    def test_the_hub_builds_without_errors(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "build_hub.py"), "--check", "--strict"],
            capture_output=True, text=True, cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])


if __name__ == "__main__":
    unittest.main()
