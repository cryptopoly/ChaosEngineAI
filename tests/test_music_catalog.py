"""Music catalog rows, fit for kind ``music``, availability, the download branches and the estimate table."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service import music_env
from backend_service.catalog import MUSIC_MODEL_FAMILIES
from backend_service.catalog.music_models import music_variants
from backend_service.helpers import music as music_helpers
from backend_service.helpers.fit import annotate_fit, variant_fit
from backend_service.helpers.hardware_class import HardwareClass

APPLE_64 = HardwareClass("apple", "apple-64", "Apple Silicon, 64 GB", 64.0, 51.8)
APPLE_36 = HardwareClass("apple", "apple-36", "Apple Silicon, 36 GB", 36.0, 24.0)
APPLE_16 = HardwareClass("apple", "apple-16", "Apple Silicon, 16 GB", 16.0, 10.7)
CUDA_24 = HardwareClass("cuda", "cuda-24", "NVIDIA, 24 GB", 24.0, 24.0)
CPU = HardwareClass("cpu", "cpu", "CPU", 32.0, 19.2)


def _variant(variant_id: str) -> dict:
    return next(v for v in music_variants() if v["id"] == variant_id)


TURBO = "ACE-Step/Ace-Step1.5"
XL = "ACE-Step/acestep-v15-xl-turbo"


class CatalogRowTests(unittest.TestCase):
    def test_only_integrated_models_with_the_ui_fields(self) -> None:
        music_rows = [v for v in music_variants() if v.get("kind") != "sound"]
        self.assertEqual([v["id"] for v in music_rows], [TURBO, XL])
        for variant in music_rows:
            licence = variant["licence"]
            self.assertEqual((licence["id"], licence["name"], licence["commercial"], licence["gated"]), ("mit", "MIT", "yes", False))
            self.assertTrue(licence["url"].startswith("https://github.com/ace-step/ACE-Step-1.5/"))
            self.assertIn("Weights are MIT-licensed. The authors state the model was trained on legally compliant data and "
                          "that output may be used commercially. You are responsible for checking that your output does not "
                          "copy existing works.", licence["summaryLines"])
            self.assertEqual(variant["makes"], ["songs", "instrumental"])
            self.assertEqual(variant["platforms"], ["apple-silicon", "cuda"])
            self.assertEqual(variant["taskSupport"], ["txt2music"])
            self.assertEqual(variant["engine"], "acestep")
            self.assertEqual(len(variant["revision"]), 40)
            for key in ("secondsPer10sAudio", "peakGb", "idleGb", "diskGb", "basis", "measuredOn", "wallSecondsAt"):
                self.assertIn(key, variant["estimates"])
            self.assertIn(variant["estimates"]["basis"], ("measured", "estimated"))
            self.assertEqual(variant["defaultSettings"]["sampleRate"], 48000)

    def test_spike_numbers(self) -> None:
        turbo, xl = _variant(TURBO), _variant(XL)
        self.assertEqual((turbo["sizeGb"], turbo["estimates"]["idleGb"]), (10.09, 15.2))
        self.assertLessEqual(turbo["estimates"]["peakGb"], 23.0)
        self.assertEqual((xl["sizeGb"], xl["estimates"]["peakGb"], xl["minMemoryGb"]), (19.95, 47.0, 64))
        self.assertTrue(xl["unloadsOthers"] and xl["large"])
        self.assertEqual(xl["requires"], [TURBO])

    def test_exported_from_the_catalog_package(self) -> None:
        self.assertEqual(MUSIC_MODEL_FAMILIES[0]["defaultVariantId"], TURBO)


class MusicFitTests(unittest.TestCase):
    def test_turbo_fits_a_64_gb_mac_on_the_measured_peak(self) -> None:
        fit = variant_fit(_variant(TURBO), "music", APPLE_64)
        self.assertEqual((fit["status"], fit["basis"], fit["needGb"]), ("fits", "measured", 22.9))
        self.assertIn("15.2 GB idle", fit["reason"])

    def test_xl_is_tight_on_64_and_refused_below(self) -> None:
        fit = variant_fit(_variant(XL), "music", APPLE_64)
        self.assertEqual(fit["status"], "tight")
        self.assertIn("unloaded first", fit["reason"])
        small = variant_fit(_variant(XL), "music", APPLE_36)
        self.assertEqual(small["status"], "wont-fit")
        self.assertIn("64 GB or more", small["reason"])

    def test_turbo_on_a_16_gb_mac_does_not_fit(self) -> None:
        self.assertEqual(variant_fit(_variant(TURBO), "music", APPLE_16)["status"], "wont-fit")

    def test_cuda_is_supported_per_the_card_but_unmeasured(self) -> None:
        xl = variant_fit(_variant(XL), "music", CUDA_24)
        self.assertEqual((xl["basis"], xl["needGb"]), ("estimated", 20.0))
        self.assertIn("not measured", xl["reason"])
        self.assertIn("not built yet", xl["reason"])
        turbo = variant_fit(_variant(TURBO), "music", CUDA_24)
        self.assertEqual(turbo["status"], "unknown")

    def test_platform_gate(self) -> None:
        self.assertEqual(variant_fit(_variant(TURBO), "music", CPU)["status"], "unknown")
        mac_only_absent = {**_variant(TURBO), "platforms": ["cuda"]}
        self.assertIn("Apple Silicon", variant_fit(mac_only_absent, "music", APPLE_64)["reason"])

    def test_annotate_fit_picks_turbo(self) -> None:
        families = annotate_fit([{**MUSIC_MODEL_FAMILIES[0], "variants": [dict(v) for v in music_variants()]}], "music", APPLE_64)
        self.assertEqual(families[0]["bestFitVariantId"], TURBO)


class AvailabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data = root / "music"
        self.cache = root / "hub"
        self.patches = [
            mock.patch.object(music_env, "MUSIC_DATA_DIR", self.data),
            mock.patch.dict("os.environ", {"HF_HUB_CACHE": str(self.cache)}),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    def _snapshot(self, variant: dict) -> Path:
        snap = self.cache / f"models--{variant['repo'].replace('/', '--')}" / "snapshots" / variant["revision"]
        subdir = variant.get("checkpointSubdir") or ""
        for folder in variant["weightFolders"]:
            inner = folder[len(subdir):].lstrip("/") if subdir and folder.startswith(subdir) else folder
            target = snap / inner
            target.mkdir(parents=True, exist_ok=True)
            (target / "model.safetensors").write_bytes(b"w")
            (target / "config.json").write_text("{}")
        return snap

    def test_nothing_downloaded(self) -> None:
        rows = music_helpers.music_model_payloads(APPLE_64)[0]["variants"]
        self.assertEqual([(r["availableLocally"], r["weightsSource"]) for r in rows], [(False, None), (False, None)])
        self.assertEqual(rows[1]["missingRequires"], [TURBO])

    def test_cache_snapshot_then_checkpoints_mirror(self) -> None:
        self._snapshot(_variant(TURBO))
        self.assertEqual(music_helpers.weights_source(_variant(TURBO)), "hf-cache")
        self.assertIsNone(music_helpers.music_download_validation_error(TURBO))
        music_env.prepare_checkpoints(music_variants())
        self.assertEqual(music_helpers.weights_source(_variant(TURBO)), "checkpoints")
        self.assertTrue((self.data / "checkpoints" / "vae" / "model.safetensors").exists())
        rows = music_helpers.music_model_payloads(APPLE_64)[0]["variants"]
        self.assertTrue(rows[0]["availableLocally"])
        self.assertFalse(rows[1]["availableLocally"])

    def test_xl_goes_into_its_own_subfolder(self) -> None:
        self._snapshot(_variant(XL))
        music_env.prepare_checkpoints(music_variants())
        self.assertTrue((self.data / "checkpoints" / "acestep-v15-xl-turbo" / "model.safetensors").exists())
        self.assertFalse((self.data / "checkpoints" / "model.safetensors").exists())

    def test_validation_names_a_missing_revision(self) -> None:
        self.assertIn("did not produce the pinned revision", music_helpers.music_download_validation_error(TURBO))
        self.assertIsNone(music_helpers.music_download_validation_error("someone/else"))

    def test_remove_checkpoints_stays_inside(self) -> None:
        self._snapshot(_variant(TURBO))
        music_env.prepare_checkpoints(music_variants())
        music_env.remove_checkpoints({"weightFolders": ["vae", "../../outside", ""]})
        self.assertFalse((self.data / "checkpoints" / "vae").exists())
        self.assertTrue((self.data / "checkpoints" / "acestep-v15-turbo").exists())
        self.assertTrue((self.data / "checkpoints").is_dir())


class DownloadBranchTests(unittest.TestCase):
    def test_revision_size_and_lookup(self) -> None:
        from backend_service.helpers.huggingface import _known_repo_size_gb

        self.assertEqual(music_helpers.music_repo_revision(TURBO), _variant(TURBO)["revision"])
        self.assertIsNone(music_helpers.music_repo_revision("black-forest-labs/FLUX.1-dev"))
        with mock.patch("backend_service.helpers.huggingface._hf_repo_cache_dir") as cache_dir:
            cache_dir.return_value = Path("/nonexistent")
            with mock.patch("backend_service.helpers.huggingface._load_hub_files_cache", return_value=None, create=True):
                self.assertEqual(_known_repo_size_gb(XL), 19.95)
        self.assertTrue(music_helpers.is_music_repo(XL))
        self.assertIsNone(music_helpers.find_music_variant("/etc/passwd"))

    def test_companions_have_an_explicit_music_branch(self) -> None:
        from backend_service.helpers.companions import companion_downloads

        self.assertEqual(companion_downloads(XL), [])

    def test_the_helper_passes_the_revision(self) -> None:
        from backend_service.app import HF_SNAPSHOT_DOWNLOAD_HELPER
        from backend_service.state import _helpers

        with mock.patch.object(_helpers.subprocess, "Popen") as popen:
            _helpers._spawn_snapshot_download("a/b", {}, None, allow_patterns=None, revision="f" * 40)
        self.assertEqual(popen.call_args.args[0][-1], "f" * 40)
        self.assertIn("kwargs['revision'] = sys.argv[3]", HF_SNAPSHOT_DOWNLOAD_HELPER)


class EstimateTests(unittest.TestCase):
    def test_reads_the_catalog_table(self) -> None:
        from backend_service.music_runtime import estimate_run

        warm60 = estimate_run(TURBO, 60, 1, warm=True)
        self.assertEqual((warm60["seconds"], warm60["basis"], warm60["peakGb"], warm60["idleGb"]), (5.9, "measured", 22.9, 15.2))
        self.assertEqual(estimate_run(TURBO, 600, 1)["seconds"], 82.4)
        self.assertAlmostEqual(estimate_run(TURBO, 120, 1)["seconds"], 5.9 + (18.1 - 5.9) * 60 / 120, places=1)
        self.assertAlmostEqual(estimate_run(TURBO, 10, 1)["seconds"], 1.2, places=1)

    def test_fixed_window_rows_cost_the_same_for_any_length(self) -> None:
        from backend_service.music_runtime import estimate_run, resolve_variant

        full, small = "stabilityai/stable-audio-open-1.0", "stabilityai/stable-audio-open-small"
        for seconds in (1, 2, 20, 47):
            self.assertAlmostEqual(estimate_run(full, seconds, 1)["seconds"], 208.0, delta=1.0)
            self.assertAlmostEqual(estimate_run(small, seconds, 1)["seconds"], 0.7, places=1)
        # Music rows stay proportional: no fixedWindow flag, and 10 s is still cheaper than 60 s.
        self.assertFalse(resolve_variant(TURBO)["estimates"].get("fixedWindow"))
        self.assertLess(estimate_run(TURBO, 10, 1)["seconds"], estimate_run(TURBO, 60, 1)["seconds"])
        cold = estimate_run(TURBO, 60, 2, warm=False)
        self.assertEqual(cold["seconds"], round(5.9 * 2 + 25.0, 1))
        self.assertEqual(cold["writesMb"], 23.0)
        self.assertEqual(estimate_run(XL, 60, 1)["seconds"], 12.4)
        with self.assertRaises(ValueError):
            estimate_run("nope/nope", 60)


class MusicGateTests(unittest.TestCase):
    def test_budgets_from_the_spike(self) -> None:
        from backend_service.helpers.memory_gate import gate_music_generation

        self.assertIsNone(gate_music_generation(30.0, 50.0))
        refusal = gate_music_generation(20.0, 50.0, model_name="ACE-Step 1.5 turbo")
        self.assertEqual((refusal["code"], refusal["needGb"], refusal["fix"]), ("memory_gate_music_low_available", 23.0, "unload-models"))
        self.assertIn("ACE-Step 1.5 turbo needs about 23 GB", refusal["message"])
        # A worker that already holds the model needs only the difference (23 - 15).
        self.assertIsNone(gate_music_generation(9.0, 50.0, resident=True))
        self.assertIsNotNone(gate_music_generation(7.0, 50.0, resident=True))
        xl = gate_music_generation(40.0, 50.0, peak_gb=47.0, idle_gb=40.2, large=True)
        self.assertEqual(xl["fix"], "use-turbo")
        self.assertIn("turbo", xl["message"])
        self.assertEqual(gate_music_generation(30.0, 99.0)["code"], "memory_gate_music_high_pressure")


class ClassifierTests(unittest.TestCase):
    def test_ace_step_repos_are_not_chat_models(self) -> None:
        from backend_service.helpers.model_classifier import _looks_like_non_chat_model

        for name in ("ACE-Step/Ace-Step1.5", "ACE-Step/acestep-v15-xl-turbo", "acestep-5Hz-lm-1.7B"):
            self.assertTrue(_looks_like_non_chat_model(Path("/x"), name, "Transformers", "HF cache", None, False), name)

    def test_process_markers(self) -> None:
        from backend_service.inference.controller import RuntimeController

        self.assertIn("music_worker.py", RuntimeController._OWNED_PROCESS_MARKERS)
        self.assertIn("/music-venv/", RuntimeController._OWNED_PROCESS_MARKERS)


if __name__ == "__main__":
    unittest.main()
