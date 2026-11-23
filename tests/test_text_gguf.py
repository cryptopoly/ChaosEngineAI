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

    def test_allow_patterns_one_mmproj_and_the_mtp_subfolder(self) -> None:
        # FU-259: unsloth ships three mmproj precisions and the sidecar in MTP/.
        files = [
            {"path": "Qwen3.6-27B-Q4_K_M.gguf", "sizeBytes": 16},
            {"path": "mmproj-BF16.gguf", "sizeBytes": 2},
            {"path": "mmproj-F16.gguf", "sizeBytes": 2},
            {"path": "mmproj-F32.gguf", "sizeBytes": 4},
            {"path": "MTP/mtp-Qwen3.6-27B-Q8_0.gguf", "sizeBytes": 1},
        ]
        patterns = text_gguf.text_repo_allow_patterns(_QWEN_REPO, files)
        self.assertIn("mmproj-F16.gguf", patterns)
        self.assertNotIn("*mmproj*.gguf", patterns)
        self.assertIn("MTP/mtp-Qwen3.6-27B-Q8_0.gguf", patterns)
        self.assertEqual(text_gguf.matched_size_bytes(files, patterns), 16 + 2 + 1)

    def test_a_dflash_drafter_that_measured_slower_is_not_downloaded_or_run(self) -> None:
        # FU-263: on ggml-org's Qwen3.6-35B-A3B the DFlash drafter ran at 120 tok/s against 177 plain.
        repo = "ggml-org/Qwen3.6-35B-A3B-GGUF"
        files = [
            {"path": "Qwen3.6-35B-A3B-Q4_K_M.gguf", "sizeBytes": 20},
            {"path": "dflash-Qwen3.6-35B-A3B-Q8_0.gguf", "sizeBytes": 1},
            {"path": "mtp-Qwen3.6-35B-A3B-Q4_0.gguf", "sizeBytes": 1},
            {"path": "mtp-Qwen3.6-35B-A3B-Q8_0.gguf", "sizeBytes": 2},
            {"path": "mmproj-Qwen3.6-35B-A3B-BF16.gguf", "sizeBytes": 1},
            {"path": "mmproj-Qwen3.6-35B-A3B-Q8_0.gguf", "sizeBytes": 1},
        ]
        patterns = text_gguf.text_repo_allow_patterns(repo, files)
        self.assertNotIn("dflash-Qwen3.6-35B-A3B-Q8_0.gguf", patterns)
        self.assertIn("mtp-Qwen3.6-35B-A3B-Q4_0.gguf", patterns)
        self.assertIn("mmproj-Qwen3.6-35B-A3B-Q8_0.gguf", patterns)
        self.assertEqual(text_gguf.matched_size_bytes(files, patterns), 20 + 1 + 1)

    def test_the_skipped_dflash_drafter_is_not_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "models--ggml-org--Qwen3.6-35B-A3B-GGUF" / "snapshots" / "rev"
            model = _write(snap / "Qwen3.6-35B-A3B-Q4_K_M.gguf", 16)
            _write(snap / "dflash-Qwen3.6-35B-A3B-Q8_0.gguf", 1)
            mtp = _write(snap / "mtp-Qwen3.6-35B-A3B-Q4_0.gguf", 1)
            self.assertEqual(text_gguf.resolve_spec_sidecar(str(model)), ("draft-mtp", str(mtp)))
            # Any other repo keeps the block drafter first (FU-178).
            other = Path(tmp) / "models--someone--Other-GGUF" / "snapshots" / "rev"
            m2 = _write(other / "Other-Q4_K_M.gguf", 16)
            df = _write(other / "dflash-Other-Q8_0.gguf", 1)
            _write(other / "mtp-Other-Q4_0.gguf", 1)
            self.assertEqual(text_gguf.resolve_spec_sidecar(str(m2)), ("draft-dflash", str(df)))

    def test_inkling_downloads_its_mainline_quant_and_mmproj_but_no_drafter(self) -> None:
        # File list of unsloth/Inkling-2-Mini-GGUF at 800c946 (FU-540), sizes in GB.
        repo = "unsloth/Inkling-2-Mini-GGUF"
        files = [
            {"path": "Inkling-2-Mini-UD-Q4_K_M.gguf", "sizeBytes": 16610},
            {"path": "Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file", "sizeBytes": 16611},
            {"path": "Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file", "sizeBytes": 22867},
            {"path": "Mainline_Support/dspark-Inkling-2-Mini-Q8_0.gguf_file", "sizeBytes": 504},
            {"path": "dspark-Inkling-2-Mini-Q8_0.gguf", "sizeBytes": 402},
            {"path": "mmproj-BF16.gguf", "sizeBytes": 642},
            {"path": "mmproj-F16.gguf", "sizeBytes": 639},
            {"path": "mmproj-F32.gguf", "sizeBytes": 1273},
        ]
        patterns = text_gguf.text_repo_allow_patterns(repo, files)
        self.assertEqual(patterns[:2], ["Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file", "mmproj-F16.gguf"])
        self.assertEqual(text_gguf.matched_size_bytes(files, patterns), 16611 + 639)

    def test_one_mmproj_choice_falls_back(self) -> None:
        self.assertEqual(text_gguf._best_mmproj(["mmproj-F32.gguf", "mmproj-BF16.gguf"]), "mmproj-BF16.gguf")
        # Bonsai 2 ships a BF16 (0.93 GB) and a Q8_0 (0.63 GB, the one run on the 4090): the smaller one.
        self.assertEqual(
            text_gguf._best_mmproj(["Ternary-Bonsai-2-27B-mmproj-BF16.gguf", "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"]),
            "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
        )
        self.assertEqual(text_gguf._best_mmproj(["mmproj-F32.gguf"]), "mmproj-F32.gguf")
        self.assertIsNone(text_gguf._best_mmproj(["a.gguf"]))

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

    def test_mtp_subfolder_sidecar_is_found(self) -> None:
        # FU-259: unsloth repos keep the drafter in MTP/, not beside the model.
        snap = self.root / "models--unsloth--X-GGUF" / "snapshots" / "rev"
        model = _write(snap / "X-Q8_0.gguf", 16)
        sidecar = _write(snap / "MTP" / "mtp-X-Q8_0.gguf", 1)
        _write(snap / "other" / "mtp-Y-Q8_0.gguf", 1)
        self.assertEqual(text_gguf.resolve_spec_sidecar(str(model)), ("draft-mtp", str(sidecar)))

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

    def test_renamed_gguf_pin_gets_a_gguf_alias_and_loads(self) -> None:
        # FU-540: Inkling 2 Mini's upstream-llama.cpp files are published as ``*.gguf_file``; the app links a
        # ``.gguf`` name beside the pinned one and loads that, never the root file in the other layout.
        snap = self.root / "models--unsloth--Inkling-2-Mini-GGUF" / "snapshots" / "rev"
        hub_file = _write(snap / "Mainline_Support" / "Inkling-2-Mini-UD-Q4_K_M.gguf_file", 16)
        _write(snap / "Inkling-2-Mini-UD-Q8_K_XL.gguf", 29)  # another quant in the other layout
        _write(snap / "mmproj-F16.gguf", 4)
        resolved = _resolve_gguf_path(str(snap), None)
        alias = snap / "Mainline_Support" / "Inkling-2-Mini-UD-Q4_K_M.gguf"
        self.assertEqual(resolved, str(alias))
        self.assertTrue(alias.is_file())
        self.assertEqual(alias.stat().st_size, hub_file.stat().st_size)
        # Idempotent: a second pass creates nothing.
        self.assertEqual(text_gguf.ensure_gguf_aliases(snap), [])

    def test_local_gguf_name(self) -> None:
        self.assertEqual(text_gguf.local_gguf_name("a/B-Q4.gguf_file"), "a/B-Q4.gguf")
        self.assertEqual(text_gguf.local_gguf_name("B-Q4.gguf"), "B-Q4.gguf")

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


    def test_worker_downloads_without_unused_shards(self) -> None:
        pruned = (["transformer/model-00001-of-00003.safetensors"], 4 * 1024 ** 3, 10 * 1024 ** 3)
        with mock.patch("backend_service.helpers.shard_index.pruned_allow_patterns", return_value=pruned):
            spawn = self._start("nobody/sharded-repo", [])
        self.assertEqual(spawn.call_args.kwargs["allow_patterns"], pruned[0])
        self.assertEqual(self.state._downloads["nobody/sharded-repo"]["totalGb"], 4.0)


    def test_bolt_download_also_fetches_its_separately_published_drafter(self) -> None:
        spawn = self._start(_BOLT_REPO, [{"path": "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf", "sizeBytes": 6 * 1024 ** 3}])
        fetched = [(c.args[0], c.kwargs["allow_patterns"]) for c in spawn.call_args_list]
        self.assertEqual(
            fetched,
            [
                (_BOLT_REPO, fetched[0][1]),
                (_BOLT_DRAFTER_REPO, [_BOLT_DRAFTER_FILE]),
            ],
        )
        self.assertEqual(self.state._downloads[_BOLT_REPO]["state"], "completed")

    def test_a_repo_without_a_pinned_drafter_fetches_nothing_extra(self) -> None:
        spawn = self._start(_QWEN_REPO, [{"path": _QWEN_PIN, "sizeBytes": 16 * 1024 ** 3}])
        self.assertEqual([c.args[0] for c in spawn.call_args_list], [_QWEN_REPO])

    def test_a_failed_drafter_fetch_does_not_fail_the_model(self) -> None:
        ok = mock.Mock(); ok.poll.return_value = 0; ok.returncode = 0; ok.wait.return_value = 0
        bad = mock.Mock(); bad.poll.return_value = 1; bad.returncode = 1; bad.wait.return_value = 1
        with mock.patch("backend_service.state.downloads._spawn_snapshot_download", side_effect=[ok, bad]), \
             mock.patch("backend_service.helpers.huggingface._hf_repo_downloaded_bytes", return_value=0), \
             mock.patch("backend_service.helpers.huggingface._hf_repo_preflight_size_gb", return_value=6.0), \
             mock.patch("backend_service.helpers.huggingface._hub_repo_files", return_value={"files": []}), \
             mock.patch("backend_service.state.downloads.threading.Thread") as thread:
            thread.side_effect = lambda *, target=None, daemon=None: mock.Mock(start=lambda: target() if target.__name__ == "_download_worker" else None)
            self.state.start_download(_BOLT_REPO)
        self.assertEqual(self.state._downloads[_BOLT_REPO]["state"], "completed")


class CatalogPinInvariantTests(unittest.TestCase):
    def test_pins_are_main_weight_first_shards(self) -> None:
        pinned = [v for f in MODEL_FAMILIES for v in f["variants"] if v.get("ggufFile")]
        self.assertTrue(pinned)
        for variant in pinned:
            name = variant["ggufFile"]
            self.assertEqual(variant["format"], "GGUF", variant["id"])
            self.assertTrue(name.endswith((".gguf", text_gguf.RENAMED_GGUF_SUFFIX)), variant["id"])
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


@unittest.skipIf(__import__("os").name == "nt", "needs symlinks")
class DownloadedBytesScopeTests(unittest.TestCase):
    """Progress counts only the files a pinned download fetches."""

    def test_other_quants_in_the_cache_do_not_inflate_progress(self) -> None:
        import tempfile
        from pathlib import Path
        from unittest import mock

        from backend_service.helpers import hf_cache_paths

        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "models--acme--Model-GGUF"
            snapshot = repo_dir / "snapshots" / "abc"
            blobs = repo_dir / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            for name, size in {"Model-Q8_0.gguf": 100, "Model-Q4_K_M.gguf": 400}.items():
                blob = blobs / f"sha-{name}"
                blob.write_bytes(b"x" * size)
                (snapshot / name).symlink_to(blob)
            (blobs / "partial.incomplete").write_bytes(b"x" * 7)
            with mock.patch.object(hf_cache_paths, "_hf_repo_cache_dir", return_value=repo_dir):
                whole = hf_cache_paths._hf_repo_downloaded_bytes("acme/Model-GGUF")
                scoped = hf_cache_paths._hf_repo_downloaded_bytes("acme/Model-GGUF", ["Model-Q8_0.gguf"])
        self.assertEqual(whole, 507)
        self.assertEqual(scoped, 107)  # the pinned file plus the in-progress blob


class LoadSizeTests(unittest.TestCase):
    """A repo folder with many quants is sized by the one file that loads."""

    def test_only_the_loaded_quant_counts(self) -> None:
        import tempfile
        from pathlib import Path

        from backend_service.helpers.text_gguf import gguf_load_bytes

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "models--acme--Model-GGUF" / "snapshots" / "abc"
            repo.mkdir(parents=True)
            (repo / "Model-BF16.gguf").write_bytes(b"x" * 480)
            (repo / "Model-Q8_0.gguf").write_bytes(b"x" * 255)
            (repo / "Model-Q4_K_M.gguf").write_bytes(b"x" * 150)
            (repo / "mmproj-Model.gguf").write_bytes(b"x" * 900)
            # no catalog pin for this repo: the largest model file loads
            self.assertEqual(gguf_load_bytes(repo), 480)

    def test_split_shards_are_summed(self) -> None:
        import tempfile
        from pathlib import Path

        from backend_service.helpers.text_gguf import gguf_load_bytes

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "models--acme--Big-GGUF" / "snapshots" / "abc"
            repo.mkdir(parents=True)
            (repo / "Big-00001-of-00002.gguf").write_bytes(b"x" * 100)
            (repo / "Big-00002-of-00002.gguf").write_bytes(b"x" * 300)
            self.assertEqual(gguf_load_bytes(repo), 400)


_BOLT_REPO = "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB"
_BOLT_DRAFTER_REPO = "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash"
_BOLT_DRAFTER_FILE = "Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf"


class PinnedDrafterTests(unittest.TestCase):
    """NVIDIA publishes the Bolt DFlash drafter in a repo of its own, under a
    name with no ``dflash-`` prefix, so the sidecar scan can never find it."""

    def test_bolt_is_paired_with_nvidias_drafter(self) -> None:
        self.assertEqual(
            text_gguf.pinned_drafter(_BOLT_REPO),
            (_BOLT_DRAFTER_REPO, _BOLT_DRAFTER_FILE, "draft-dflash"),
        )
        self.assertIsNone(text_gguf.pinned_drafter("someone/else-GGUF"))
        self.assertIsNone(text_gguf.pinned_drafter(None))

    def test_resolves_the_drafter_from_the_hub_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snap"
            drafter = _write(snapshot / _BOLT_DRAFTER_FILE, 8)
            with mock.patch(
                "backend_service.helpers.hf_cache_paths._hf_repo_snapshot_dir", return_value=snapshot
            ):
                self.assertEqual(
                    text_gguf.resolve_pinned_drafter(_BOLT_REPO), ("draft-dflash", str(drafter))
                )
                # A model launched from its library folder: only the path names the repo.
                model = Path(tmp) / "models--nvidia--NVIDIA-Nemotron-3.5-Bolt-E4B-4GB" / "snapshots" / "abc" / "m.gguf"
                self.assertEqual(
                    text_gguf.resolve_pinned_drafter("", str(model)), ("draft-dflash", str(drafter))
                )

    def test_missing_drafter_resolves_to_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "backend_service.helpers.hf_cache_paths._hf_repo_snapshot_dir", return_value=Path(tmp)
            ):
                self.assertIsNone(text_gguf.resolve_pinned_drafter(_BOLT_REPO))
            with mock.patch(
                "backend_service.helpers.hf_cache_paths._hf_repo_snapshot_dir", return_value=None
            ):
                self.assertIsNone(text_gguf.resolve_pinned_drafter(_BOLT_REPO))

    def test_every_pin_names_a_catalog_target_and_a_gguf(self) -> None:
        catalog_repos = {v["repo"] for f in MODEL_FAMILIES for v in f["variants"]}
        for target, (drafter_repo, drafter_file, spec_type) in text_gguf.PINNED_DRAFTERS.items():
            self.assertIn(target, catalog_repos)
            self.assertTrue(drafter_file.endswith(".gguf"))
            self.assertTrue(spec_type.startswith("draft-"))
            self.assertNotEqual(drafter_repo, target)

    def test_engine_loads_the_pinned_drafter_before_scanning_for_sidecars(self) -> None:
        from backend_service.inference import llama_cpp_engine

        with tempfile.TemporaryDirectory() as tmp:
            model = _write(Path(tmp) / "model.gguf", 8)
            drafter = Path(tmp) / "drafter.gguf"
            with (
                mock.patch.object(llama_cpp_engine, "_llama_server_supports", return_value=True),
                mock.patch("backend_service.inference._mtp.model_has_mtp_tensors", return_value=False),
                mock.patch(
                    "backend_service.helpers.text_gguf.resolve_pinned_drafter",
                    return_value=("draft-dflash", str(drafter)),
                ),
            ):
                args, note = llama_cpp_engine._speculative_args(
                    "llama-server", resolved_gguf=str(model), path=str(model), repo=_BOLT_REPO
                )
        self.assertEqual(
            args, ["--model-draft", str(drafter), "--spec-type", "draft-dflash", "--spec-draft-n-max", "15"]
        )
        self.assertIn("drafter.gguf", note or "")


_GEMMA12_REPO = "google/gemma-4-12B-it-qat-q4_0-gguf"


class GemmaMtpDrafterTests(unittest.TestCase):
    """Google ships no GGUF drafter for its QAT Gemma 4 (the ``-assistant`` repos are safetensors);
    Unsloth's ``mtp-`` GGUF of the 12B assistant ran against it on llama.cpp b12441 (2026-11-19)."""

    def test_the_12b_qat_row_is_paired_with_the_unsloth_mtp_gguf(self) -> None:
        self.assertEqual(
            text_gguf.pinned_drafter(_GEMMA12_REPO),
            ("unsloth/gemma-4-12B-it-GGUF", "mtp-gemma-4-12b-it.gguf", "draft-mtp"),
        )

    def test_the_drafter_downloads_beside_the_model(self) -> None:
        from backend_service.helpers.companions import companion_downloads

        self.assertEqual(
            companion_downloads(_GEMMA12_REPO),
            [("unsloth/gemma-4-12B-it-GGUF", ["mtp-gemma-4-12b-it.gguf"])],
        )

    def test_the_engine_runs_it_as_draft_mtp_with_two_tokens(self) -> None:
        from backend_service.inference import llama_cpp_engine

        with tempfile.TemporaryDirectory() as tmp:
            model = _write(Path(tmp) / "gemma-4-12b-it-qat-q4_0.gguf", 8)
            drafter = _write(Path(tmp) / "mtp-gemma-4-12b-it.gguf", 8)
            with (
                mock.patch("backend_service.inference.llama_cpp_engine._llama_server_supports", return_value=True),
                mock.patch("backend_service.inference._mtp.model_has_mtp_tensors", return_value=False),
                mock.patch(
                    "backend_service.helpers.text_gguf.resolve_pinned_drafter",
                    return_value=("draft-mtp", str(drafter)),
                ),
            ):
                args, _note = llama_cpp_engine._speculative_args(
                    "llama-server", resolved_gguf=str(model), path=str(model), repo=_GEMMA12_REPO
                )
        self.assertEqual(args, ["--model-draft", str(drafter), "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"])

    def test_the_31b_qat_row_is_paired_too(self) -> None:
        # FU-264: ran live on a 4090, 45 tok/s plain, about 60 with the drafter at two per round.
        from backend_service.helpers.companions import companion_downloads

        repo = "google/gemma-4-31B-it-qat-q4_0-gguf"
        self.assertEqual(
            text_gguf.pinned_drafter(repo), ("unsloth/gemma-4-31B-it-GGUF", "mtp-gemma-4-31B-it.gguf", "draft-mtp")
        )
        self.assertEqual(companion_downloads(repo), [("unsloth/gemma-4-31B-it-GGUF", ["mtp-gemma-4-31B-it.gguf"])])

    def test_the_26b_a4b_qat_row_is_paired_too(self) -> None:
        self.assertEqual(
            text_gguf.pinned_drafter("google/gemma-4-26B-A4B-it-qat-q4_0-gguf"),
            ("unsloth/gemma-4-26B-A4B-it-GGUF", "mtp-gemma-4-26B-A4B-it.gguf", "draft-mtp"),
        )
        # Only a pairing that ran live is pinned: the Q8_0 row has none (it needs 32 GB, not run here).
        self.assertIsNone(text_gguf.pinned_drafter("unsloth/gemma-4-26B-A4B-it-GGUF"))
