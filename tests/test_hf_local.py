import unittest
from unittest import mock

from backend_service.helpers.hf_local import local_snapshot_path


class IncompleteSnapshotError(Exception):
    def __init__(self, message: str, snapshot_path: str) -> None:
        super().__init__(message)
        self.snapshot_path = snapshot_path


class LocalSnapshotPathTests(unittest.TestCase):
    def test_returns_the_snapshot_path_when_the_cache_is_complete(self):
        with mock.patch("huggingface_hub.snapshot_download", return_value="/cache/snap") as dl:
            self.assertEqual(local_snapshot_path("org/model", resume_download=True), "/cache/snap")
        dl.assert_called_once_with(repo_id="org/model", local_files_only=True, resume_download=True)

    def test_accepts_a_cache_filtered_by_the_download_allow_list(self):
        error = IncompleteSnapshotError("5 file(s) are missing", snapshot_path="/cache/snap")
        with mock.patch("huggingface_hub.snapshot_download", side_effect=error):
            self.assertEqual(local_snapshot_path("org/model"), "/cache/snap")

    def test_other_errors_still_propagate(self):
        with mock.patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                local_snapshot_path("org/model")


if __name__ == "__main__":
    unittest.main()
