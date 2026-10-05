"""Pinned text GGUF files — download allowlist, sizing, and loader pick.

Regression cover for multi-quant community repos: before ``ggufFile`` a
Q4_K_M catalog row downloaded every quant in the repo (68 GB for
Qwen3.6-27B, ~11 TB for the unsloth GLM-5.1 row) and the loader then ran
the largest file — the Q8_0 / BF16, not the quant the row advertised.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.catalog.text_models import MODEL_FAMILIES
from backend_service.helpers import text_gguf
from backend_service.inference._utils import _resolve_gguf_path
from backend_service.state import ChaosEngineState

_QWEN_REPO = "lmstudio-community/Qwen3.6-27B-GGUF"
_QWEN_PIN = "Qwen3.6-27B-Q4_K_M.gguf"


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


class HelperTests(unittest.TestCase):
    def test_pinned_file_lookup(self) -> None:
        self.assertEqual(text_gguf.pinned_gguf_file(_QWEN_REPO), _QWEN_PIN)
        self.assertIsNone(text_gguf.pinned_gguf_file("nobody/unknown-GGUF"))
        self.assertIsNone(text_gguf.pinned_gguf_file(None))

    def test_shard_pattern(self) -> None:
        self.assertEqual(text_gguf.gguf_file_pattern("a/M-Q4_K_M.gguf"), "a/M-Q4_K_M.gguf")
        self.assertEqual(
            text_gguf.gguf_file_pattern("UD-Q4_K_M/GLM-5.1-UD-Q4_K_M-00001-of-00011.gguf"),
            "UD-Q4_K_M/GLM-5.1-UD-Q4_K_M-*-of-00011.gguf",
        )

    def test_allow_patterns(self) -> None:
        self.assertIsNone(text_gguf.text_repo_allow_patterns("nobody/unknown-GGUF"))
        patterns = text_gguf.text_repo_allow_patterns(_QWEN_REPO)
        self.assertIn(_QWEN_PIN, patterns)
        self.assertIn("*mmproj*.gguf", patterns)

    def test_matched_size_counts_only_the_pinned_quant(self) -> None:
        files = [
            {"path": "Qwen3.6-27B-Q4_K_M.gguf", "sizeBytes": 16},
            {"path": "Qwen3.6-27B-Q8_0.gguf", "sizeBytes": 29},
            {"path": "mmproj-Qwen3.6-27B-BF16.gguf", "sizeBytes": 1},
            {"path": "README.md", "sizeBytes": 0},
        ]
        patterns = text_gguf.text_repo_allow_patterns(_QWEN_REPO)
        self.assertEqual(text_gguf.matched_size_bytes(files, patterns), 17)

    def test_allow_patterns_add_one_sidecar_per_kind(self) -> None:
        # FU-089: the drafter closest to the pinned quant rides along;
        # other precisions of the same drafter don't.
        files = [
            {"path": "Qwen3.6-27B-Q4_K_M.gguf", "sizeBytes": 16},
            {"path": "Qwen3.6-27B-Q8_0.gguf", "sizeBytes": 29},
            {"path": "dflash-Qwen3.6-27B-Q4_K_M.gguf", "sizeBytes": 2},
            {"path": "dflash-Qwen3.6-27B-BF16.gguf", "sizeBytes": 4},
            {"path": "mtp-Qwen3.6-27B-Q8_0.gguf", "sizeBytes": 1},
        ]
        patterns = text_gguf.text_repo_allow_patterns(_QWEN_REPO, files)
        self.assertIn("dflash-Qwen3.6-27B-Q4_K_M.gguf", patterns)
        self.assertIn("mtp-Qwen3.6-27B-Q8_0.gguf", patterns)
        self.assertNotIn("dflash-Qwen3.6-27B-BF16.gguf", patterns)
        self.assertEqual(text_gguf.matched_size_bytes(files, patterns), 16 + 2 + 1)

    def test_repo_id_from_path(self) -> None:
        self.assertEqual(
            text_gguf.repo_id_from_path(
                Path("/hub/models--lmstudio-community--Qwen3.6-27B-GGUF/snapshots/abc")
            ),
            _QWEN_REPO,
        )
        self.assertEqual(
            text_gguf.repo_id_from_path(Path("/AI_Models/prism-ml/Bonsai-8B-gguf")),
            "prism-ml/Bonsai-8B-gguf",
        )


class ResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_pinned_quant_beats_larger_siblings(self) -> None:
        snap = self.root / "models--lmstudio-community--Qwen3.6-27B-GGUF" / "snapshots" / "rev"
        pinned = _write(snap / _QWEN_PIN, 16)
        _write(snap / "Qwen3.6-27B-Q8_0.gguf", 29)
        _write(snap / "mmproj-Qwen3.6-27B-BF16.gguf", 64)
        self.assertEqual(_resolve_gguf_path(str(snap), None), str(pinned))

    def test_unpinned_repo_keeps_largest_file(self) -> None:
        snap = self.root / "models--someone--Model-GGUF" / "snapshots" / "rev"
        _write(snap / "Model-Q4_K_M.gguf", 16)
        largest = _write(snap / "Model-Q8_0.gguf", 29)
        self.assertEqual(_resolve_gguf_path(str(snap), None), str(largest))

    def test_split_gguf_resolves_to_first_shard(self) -> None:
        # Unpinned: the largest shard is rarely 00001, but llama.cpp must
        # be handed the first one.
        snap = self.root / "models--someone--Big-GGUF" / "snapshots" / "rev"
        first = _write(snap / "Q4" / "Big-Q4-00001-of-00003.gguf", 10)
        _write(snap / "Q4" / "Big-Q4-00002-of-00003.gguf", 30)
        _write(snap / "Q4" / "Big-Q4-00003-of-00003.gguf", 20)
        self.assertEqual(_resolve_gguf_path(str(snap), None), str(first))

    def test_pinned_split_quant_picks_its_first_shard(self) -> None:
        snap = self.root / "models--unsloth--GLM-5.1-GGUF" / "snapshots" / "rev"
        first = _write(snap / "UD-Q4_K_M" / "GLM-5.1-UD-Q4_K_M-00001-of-00011.gguf", 5)
        _write(snap / "UD-Q4_K_M" / "GLM-5.1-UD-Q4_K_M-00002-of-00011.gguf", 6)
        _write(snap / "BF16" / "GLM-5.1-BF16-00001-of-00033.gguf", 50)
        self.assertEqual(_resolve_gguf_path(str(snap), None), str(first))

    def test_pin_missing_on_disk_falls_back_to_largest(self) -> None:
        snap = self.root / "models--lmstudio-community--Qwen3.6-27B-GGUF" / "snapshots" / "rev"
        largest = _write(snap / "Qwen3.6-27B-Q8_0.gguf", 29)
        _write(snap / "Qwen3.6-27B-Q6_K.gguf", 22)
        self.assertEqual(_resolve_gguf_path(str(snap), None), str(largest))


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.state = ChaosEngineState(
            settings_path=tmp / "settings.json",
            benchmarks_path=tmp / "benchmarks.json",
            chat_sessions_path=tmp / "chats.json",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _start(self, repo: str, hub_files: list[dict]):
        process = mock.Mock()
        process.poll.return_value = 0
        process.returncode = 0
        process.wait.return_value = 0
        created: list[object] = []

        class ImmediateThread:
            def __init__(self, *, target=None, daemon=None):
                self.target = target
                self.run_target = not created
                created.append(self)

            def start(self):
                if self.run_target and self.target is not None:
                    self.target()

            def join(self, timeout=None):
                return None

        with mock.patch("backend_service.state.downloads._spawn_snapshot_download", return_value=process) as spawn, \
             mock.patch("backend_service.helpers.huggingface._hf_repo_downloaded_bytes", return_value=0), \
             mock.patch("backend_service.helpers.huggingface._hf_repo_preflight_size_gb", return_value=68.2), \
             mock.patch("backend_service.helpers.huggingface._hub_repo_files", return_value={"files": hub_files}), \
             mock.patch("backend_service.state.downloads.threading.Thread", side_effect=ImmediateThread):
            self.state.start_download(repo)
        return spawn

    def test_pinned_repo_downloads_one_quant_and_sizes_progress_to_it(self) -> None:
        gb = 1024 ** 3
        files = [
            {"path": _QWEN_PIN, "sizeBytes": 16 * gb},
            {"path": "Qwen3.6-27B-Q6_K.gguf", "sizeBytes": 22 * gb},
            {"path": "Qwen3.6-27B-Q8_0.gguf", "sizeBytes": 29 * gb},
        ]
        spawn = self._start(_QWEN_REPO, files)
        self.assertEqual(spawn.call_args.kwargs["allow_patterns"][0], _QWEN_PIN)
        self.assertEqual(self.state._downloads[_QWEN_REPO]["totalGb"], 16.0)

    def test_unpinned_text_repo_downloads_in_full(self) -> None:
        spawn = self._start("nobody/unknown-GGUF", [])
        self.assertIsNone(spawn.call_args.kwargs["allow_patterns"])


class CatalogPinInvariantTests(unittest.TestCase):
    def test_pins_are_main_weight_first_shards(self) -> None:
        pinned = [v for f in MODEL_FAMILIES for v in f["variants"] if v.get("ggufFile")]
        self.assertTrue(pinned)
        for variant in pinned:
            name = variant["ggufFile"]
            self.assertEqual(variant["format"], "GGUF", variant["id"])
            self.assertTrue(name.endswith(".gguf"), variant["id"])
            self.assertNotIn("mmproj", name.lower(), variant["id"])
            shard = re.search(r"-(\d{5})-of-\d{5}\.gguf$", name)
            if shard:
                self.assertEqual(shard.group(1), "00001", variant["id"])

    def test_one_pin_per_repo(self) -> None:
        # pinned_gguf_file() keys by repo, so two rows pinning different
        # files in the same repo would silently shadow each other.
        pins: dict[str, str] = {}
        for family in MODEL_FAMILIES:
            for variant in family["variants"]:
                if variant.get("ggufFile"):
                    previous = pins.setdefault(variant["repo"], variant["ggufFile"])
                    self.assertEqual(previous, variant["ggufFile"], variant["repo"])


if __name__ == "__main__":
    unittest.main()
