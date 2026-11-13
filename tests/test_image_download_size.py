"""The download bar of a pinned image row counts the files it fetches, not the whole repo (FU-175)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

FLUX2_DEV = "black-forest-labs/FLUX.2-dev"


class _NoThread:
    def __init__(self, *, target=None, daemon=None):
        pass

    def start(self):
        pass


def _state(tmp: str):
    from backend_service.state import ChaosEngineState

    return ChaosEngineState(
        settings_path=Path(tmp) / "settings.json",
        benchmarks_path=Path(tmp) / "benchmarks.json",
        chat_sessions_path=Path(tmp) / "chats.json",
    )


class Flux2DevDownloadSizeTests(unittest.TestCase):
    def test_catalog_size_is_the_diffusers_folders(self) -> None:
        from backend_service.helpers.huggingface import _known_repo_size_gb

        # transformer 64.45 + text encoder 48.02 + VAE 0.34 GB; the repo's single-file copy is skipped.
        self.assertAlmostEqual(_known_repo_size_gb(FLUX2_DEV, catalog_only=True), 112.8)

    def test_cached_hub_total_does_not_size_the_bar(self) -> None:
        from backend_service.helpers import huggingface

        with tempfile.TemporaryDirectory() as tmp:
            state = _state(tmp)
            key = (FLUX2_DEV, huggingface._hf_token_cache_key())
            with mock.patch.dict(huggingface._HUB_FILE_CACHE, {key: {"totalSizeGb": 154.08}}), \
                 mock.patch("backend_service.helpers.huggingface._hf_repo_downloaded_bytes", return_value=0), \
                 mock.patch("backend_service.helpers.huggingface._hf_repo_preflight_size_gb", return_value=154.08), \
                 mock.patch("backend_service.state.downloads.threading.Thread", side_effect=_NoThread):
                row = dict(state.start_download(FLUX2_DEV))
        self.assertAlmostEqual(row["totalGb"], 105.05, places=1)

    def test_gated_failure_reports_the_catalog_size(self) -> None:
        from backend_service.helpers import huggingface

        with tempfile.TemporaryDirectory() as tmp:
            state = _state(tmp)
            key = (FLUX2_DEV, huggingface._hf_token_cache_key())
            with mock.patch.dict(huggingface._HUB_FILE_CACHE, {key: {"totalSizeGb": 154.08}}), \
                 mock.patch("backend_service.helpers.huggingface._hf_repo_downloaded_bytes", return_value=0), \
                 mock.patch("backend_service.helpers.huggingface._hf_repo_preflight_size_gb",
                            side_effect=RuntimeError("401 gated repo")):
                row = dict(state.start_download(FLUX2_DEV))
        self.assertEqual(row["state"], "failed")
        self.assertAlmostEqual(row["totalGb"], 105.05, places=1)


if __name__ == "__main__":
    unittest.main()
