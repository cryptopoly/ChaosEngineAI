"""hub_dir_for_home: a chosen cache folder that already holds ``models--*`` repos is the hub itself."""

import tempfile
import unittest
from pathlib import Path

from backend_service.helpers.hf_cache_paths import hub_dir_for_home


class HubDirForHomeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)

    def test_empty_folder_uses_hub_subfolder(self):
        self.assertEqual(hub_dir_for_home(self.home), self.home / "hub")

    def test_folder_holding_repos_is_the_hub(self):
        (self.home / "models--org--name").mkdir()
        self.assertEqual(hub_dir_for_home(self.home), self.home)

    def test_existing_hub_folder_wins(self):
        (self.home / "models--org--name").mkdir()
        (self.home / "hub").mkdir()
        self.assertEqual(hub_dir_for_home(self.home), self.home / "hub")

    def test_missing_folder_uses_hub_subfolder(self):
        missing = self.home / "nope"
        self.assertEqual(hub_dir_for_home(missing), missing / "hub")


if __name__ == "__main__":
    unittest.main()
