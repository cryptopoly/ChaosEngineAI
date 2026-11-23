"""Hugging Face likes and downloads behind the "Most popular" pick: file cache, refresh rules, offline behaviour."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend_service.helpers import hf_stats
from backend_service.helpers.chat_picks import chat_picks
from tests.test_chat_picks import CARD_24, _family, _variant


class StatsFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "hf-stats.json"

    def tearDown(self):
        self.dir.cleanup()

    def test_missing_or_broken_file_is_empty_not_an_error(self):
        self.assertEqual(hf_stats.load_stats(self.path), {"fetchedAt": None, "repos": {}})
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(hf_stats.load_stats(self.path)["repos"], {})

    def test_refresh_merges_and_dates_the_file(self):
        now = datetime(2026, 11, 23, tzinfo=timezone.utc)
        ok = hf_stats.refresh_stats(
            ["org/a", "org/b"], self.path,
            fetch=lambda repo: {"likes": 5, "downloads": 100, "downloadsAllTime": 900} if repo == "org/a" else None,
            now=now,
        )
        self.assertTrue(ok)
        stats = hf_stats.load_stats(self.path)
        self.assertEqual(stats["fetchedAt"], now.isoformat())
        self.assertEqual(list(stats["repos"]), ["org/a"])

    def test_offline_refresh_changes_nothing(self):
        self.path.write_text(json.dumps({"fetchedAt": "x", "repos": {"org/a": {"downloads": 1}}}), encoding="utf-8")
        self.assertFalse(hf_stats.refresh_stats(["org/a"], self.path, fetch=lambda repo: None))
        self.assertEqual(hf_stats.load_stats(self.path)["repos"], {"org/a": {"downloads": 1}})

    def test_fresh_file_with_every_repo_is_not_refreshed(self):
        now = datetime.now(timezone.utc)
        stats = {"fetchedAt": now.isoformat(), "repos": {"org/a": {}}}
        self.assertTrue(hf_stats._is_fresh(stats, ["org/a"], now))
        self.assertFalse(hf_stats._is_fresh(stats, ["org/a", "org/new"], now))
        self.assertFalse(hf_stats._is_fresh(stats, ["org/a"], now + timedelta(days=2)))
        self.assertFalse(hf_stats._is_fresh({"fetchedAt": None, "repos": {}}, ["org/a"], now))

    def test_background_refresh_never_starts_under_test_or_offline(self):
        self.assertFalse(hf_stats.refresh_in_background(["org/a"], self.path))


class PicksCarryStatsTests(unittest.TestCase):
    families = [_family("a", _variant("a"), best="a"), _family("b", _variant("b"), best="b")]

    def test_rows_carry_counters_when_known_and_none_offline(self):
        stats = {"fetchedAt": "2026-11-23T00:00:00+00:00", "repos": {"org/a": {"likes": 12, "downloads": 3400}}}
        rows = {r["variantId"]: r for r in chat_picks(self.families, CARD_24, bandwidth_gbps=lambda: 1000.0, hf_stats=stats)}
        self.assertEqual((rows["a"]["likes"], rows["a"]["downloads"]), (12, 3400))
        self.assertEqual(rows["a"]["statsAsOf"], "2026-11-23T00:00:00+00:00")
        self.assertEqual((rows["b"]["likes"], rows["b"]["downloads"], rows["b"]["statsAsOf"]), (None, None, None))
        offline = chat_picks(self.families, CARD_24, bandwidth_gbps=lambda: 1000.0)
        self.assertTrue(all(r["downloads"] is None and r["statsAsOf"] is None for r in offline))


if __name__ == "__main__":
    unittest.main()
