"""FU-168: the TensorFold family table built from the binary's own ``capabilities --json``.

Two canned reports drive these tests:

* ``tensorfold_caps_macos_1_2_8.json`` is the macOS 1.2.8 Metal build's real
  output, verbatim (M4 Max, 2026-11-14).
* ``tensorfold_caps_cuda_1_2_8.json`` is assembled from the v1.2.7 source (the
  CUDA registry in ``zig/src/native/cuda_families.zig``) and the formats of the
  GB10 receipts in ``docs/receipts``; it carries the four CUDA-only families the
  Windows 1.2.6 build reported (dory, kolibri1, lfm2_vl, nemotron_h_omni),
  plus the two CUDA families v1.2.8 added (qwen3, dots_ocr; bf16).
  Replace it with a verbatim dump from the 4090 when one is taken.

Also covers D-77: the three ``native=False`` rows keep routing to the 0.6.6
Python install while the binary does not list them, and move to the binary
once a release does.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.inference import RuntimeController
from backend_service.inference import _tensorfold as tf
from backend_service.inference import _tensorfold_native as native
from backend_service.inference.base import BackendCapabilities
from backend_service.inference.mlx_engine import MLXWorkerEngine
from backend_service.inference.tensorfold_engine import TensorFoldEngine
from backend_service.inference.tensorfold_native_engine import TensorFoldNativeEngine

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]
CUDA_ONLY = ("dory", "kolibri1", "lfm2_vl", "nemotron_h_omni", "qwen3", "dots_ocr")
LEGACY = ("deepseek_v4", "prism_hadamard_qwen35")

_DEEPSEEK = "mlx-community/DeepSeek-V4-Flash-4bit"
_BONSAI = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
_GEMMA_26B = "mlx-community/gemma-4-26b-a4b-it-4bit"


def _caps(name: str) -> native.NativeCapabilities:
    caps = native.parse_capabilities((FIXTURES / name).read_text(encoding="utf-8"))
    assert caps is not None
    return caps


def _with_families(caps: native.NativeCapabilities, **extra: tuple[str, ...]) -> native.NativeCapabilities:
    return native.NativeCapabilities(caps.version, caps.chip, caps.backends, {**caps.families, **extra}, caps.flags)


MAC = _caps("tensorfold_caps_macos_1_2_8.json")
CUDA = _caps("tensorfold_caps_cuda_1_2_8.json")


def _by_type(report: list[dict]) -> dict[str, dict]:
    return {entry["modelType"]: entry for entry in report}


class CuratedRowTests(unittest.TestCase):
    def test_cuda_only_families_are_listed_not_offered(self) -> None:
        rows = {mt: fam for fam in tf.TENSORFOLD_FAMILIES for mt in fam.model_types}
        for model_type in CUDA_ONLY:
            family = rows[model_type]
            self.assertEqual(family.tested_repos, (), model_type)
            self.assertIn("CUDA only", family.note, model_type)
            # Never routes by config: kolibri1 is also the app's own MLX port.
            self.assertFalse(family.exclusive, model_type)

    def test_the_cuda_only_checkpoints_match_no_family(self) -> None:
        for repo in (
            "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4",
            "Aleph-Alpha/Kolibri-1",
            "LiquidAI/d1-3B",
            "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
        ):
            self.assertIsNone(tf.match_model(canonical_repo=repo), repo)
            self.assertNotIn(repo, tf.supported_repos())

    def test_a_kolibri_mlx_build_is_never_captured_by_config(self) -> None:
        self.assertNotIn("kolibri1", tf._EXCLUSIVE_MODEL_TYPES)
        self.assertIn("kolibri1", tf._SHARED_MODEL_TYPES)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(tf, "_stock_mlx_loads", return_value=False):
            model = Path(tmp) / "Kolibri-1-MLX-4bit"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"model_type": "kolibri1"}))
            self.assertIsNone(tf.match_model(canonical_repo="velaia/Kolibri-1-MLX-4bit", path=str(model), model_dir=model))


class FamilyReportTests(unittest.TestCase):
    def test_mac_report_lists_exactly_the_binarys_families_all_curated(self) -> None:
        report = native.family_report(MAC)
        served = {entry["modelType"] for entry in report if entry["servedHere"]}
        self.assertEqual(served, set(MAC.families))
        self.assertEqual([entry["modelType"] for entry in report if not entry["curated"]], [])

    def test_mac_report_marks_rows_the_metal_build_does_not_serve(self) -> None:
        entries = _by_type(native.family_report(MAC))
        for model_type in (*LEGACY, *CUDA_ONLY):
            self.assertFalse(entries[model_type]["servedHere"], model_type)
            self.assertFalse(entries[model_type]["offered"], model_type)
        for model_type in LEGACY:
            self.assertTrue(entries[model_type]["legacy"], model_type)
        for model_type in CUDA_ONLY:
            self.assertFalse(entries[model_type]["legacy"], model_type)
        # Gemma 4: the type is listed (31B, 12B) but the 26B-A4B stays on Python.
        self.assertTrue(entries["gemma4"]["servedHere"])
        self.assertTrue(entries["gemma4"]["legacy"])
        self.assertNotIn(_GEMMA_26B, entries["gemma4"]["repos"])

    def test_mac_report_offers_the_metal_tables_checkpoints_only(self) -> None:
        entries = _by_type(native.family_report(MAC))
        self.assertIn("LiquidAI/LFM3-1.7B-MLX-4bit", entries["lfm3"]["repos"])
        self.assertIn("LiquidAI/LFM2.5-24B-A2B-MLX-4bit", entries["lfm2_moe"]["repos"])
        # The CUDA registry's Gemma 4 31B row is not a Mac offer.
        self.assertNotIn("mlx-community/gemma-4-31B-it-4bit", entries["gemma4"]["repos"])
        for model_type in ("kimi_k3", "fish_omni", "lfm3_vl", "step5_vl"):
            self.assertFalse(entries[model_type]["offered"], model_type)
        # 1.2.8 added Flint on Metal: listed and curated, not offered yet.
        self.assertTrue(entries["flint"]["servedHere"])
        self.assertTrue(entries["flint"]["curated"])
        self.assertFalse(entries["flint"]["offered"])
        self.assertFalse(entries["flint"]["legacy"])

    def test_cuda_report_lists_the_four_families_curated_and_not_offered(self) -> None:
        report = native.family_report(CUDA)
        entries = _by_type(report)
        self.assertEqual({e["modelType"] for e in report if e["servedHere"]}, set(CUDA.families))
        self.assertEqual([e["modelType"] for e in report if not e["curated"]], [])
        for model_type in CUDA_ONLY:
            entry = entries[model_type]
            self.assertTrue(entry["servedHere"] and entry["curated"], model_type)
            self.assertFalse(entry["offered"], model_type)
            self.assertEqual(entry["formats"], list(CUDA.families[model_type]))
        # The CUDA registry decides what is offered on a CUDA build.
        self.assertIn("mlx-community/gemma-4-31B-it-4bit", entries["gemma4"]["repos"])
        self.assertIn("LiquidAI/LFM2.5-24B-A2B-MLX-4bit", entries["lfm2_moe"]["repos"])
        # Metal-only families are absent from the CUDA build.
        for model_type in ("lfm3", "llama", "step5_vl", "glm_moe_dsa"):
            self.assertFalse(entries[model_type]["servedHere"], model_type)

    def test_a_family_with_no_row_is_listed_uncurated_and_never_offered(self) -> None:
        entries = _by_type(native.family_report(_with_families(MAC, brand_new=("mlx-q4g64", "bf16"))))
        entry = entries["brand_new"]
        self.assertEqual(
            (entry["curated"], entry["servedHere"], entry["offered"], entry["formats"]),
            (False, True, False, ["mlx-q4g64", "bf16"]),
        )

    def test_no_binary_no_report(self) -> None:
        self.assertEqual(native.family_report(None), [])


class ServesFamilyTests(unittest.TestCase):
    """D-77, both directions."""

    def _family(self, repo: str) -> tf.TensorFoldFamily:
        match = tf.match_model(canonical_repo=repo)
        assert match is not None
        return match.family

    def test_without_a_binary_report_the_native_flag_answers(self) -> None:
        for repo in (_DEEPSEEK, _BONSAI, _GEMMA_26B):
            self.assertTrue(native.prefers_python(self._family(repo), None), repo)
        self.assertFalse(native.prefers_python(self._family("LiquidAI/LFM3-1.7B-MLX-4bit"), None))

    def test_unlisted_legacy_rows_stay_on_python(self) -> None:
        for repo in (_DEEPSEEK, _BONSAI, _GEMMA_26B):
            self.assertTrue(native.prefers_python(self._family(repo), MAC), repo)
            self.assertTrue(native.prefers_python(self._family(repo), CUDA), repo)

    def test_a_binary_that_lists_a_legacy_family_wins(self) -> None:
        caps = _with_families(MAC, deepseek_v4=("mlx-q4g64",), prism_hadamard_qwen35=("mlx-q2g64",))
        for repo in (_DEEPSEEK, _BONSAI):
            self.assertTrue(native.serves_family(self._family(repo), caps), repo)
            self.assertFalse(native.prefers_python(self._family(repo), caps), repo)

    def test_gemma_26b_stays_on_python_although_gemma4_is_listed(self) -> None:
        self.assertIn("gemma4", MAC.families)
        self.assertFalse(native.serves_family(self._family(_GEMMA_26B), MAC))

    def test_native_rows_never_go_to_python(self) -> None:
        # Even where the binary does not list them (the CUDA build has no lfm3).
        for repo in ("LiquidAI/LFM3-1.7B-MLX-4bit", "Vontra/Qwen3.8-27B-MLX-4bit"):
            self.assertFalse(native.prefers_python(self._family(repo), CUDA), repo)


class CachedCapabilitiesTests(unittest.TestCase):
    def setUp(self) -> None:
        native._CAPS_CACHE.clear()
        self.addCleanup(native._CAPS_CACHE.clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.binary = Path(self.tmp.name) / "tensorfold-native"
        self.binary.write_bytes(b"v1")

    def test_probes_once_per_binary_and_again_when_it_changes(self) -> None:
        probe = mock.Mock(return_value=MAC)
        with mock.patch.object(native, "probe_capabilities", probe):
            self.assertIs(native.cached_capabilities(self.binary), MAC)
            self.assertIs(native.cached_capabilities(str(self.binary)), MAC)
            self.assertEqual(probe.call_count, 1)
            self.binary.write_bytes(b"version two")
            native.cached_capabilities(self.binary)
            self.assertEqual(probe.call_count, 2)

    def test_a_failed_probe_is_retried(self) -> None:
        probe = mock.Mock(side_effect=[None, MAC])
        with mock.patch.object(native, "probe_capabilities", probe):
            self.assertIsNone(native.cached_capabilities(self.binary))
            self.assertIs(native.cached_capabilities(self.binary), MAC)

    def test_missing_binary(self) -> None:
        self.assertIsNone(native.cached_capabilities(None))
        self.assertIsNone(native.cached_capabilities(Path(self.tmp.name) / "absent"))


def _mac_controller(*, python: bool, native_path: str | None = "/fake/tensorfold-native") -> RuntimeController:
    caps = BackendCapabilities(
        pythonExecutable=sys.executable,
        mlxAvailable=True,
        mlxLmAvailable=True,
        mlxUsable=True,
        tensorfoldAvailable=python,
        tensorfoldNativeAvailable=True,
        tensorfoldNativePath=native_path,
    )
    with mock.patch("backend_service.inference.get_backend_capabilities", return_value=caps):
        controller = RuntimeController()
    controller.capabilities = caps
    return controller


def _select(controller: RuntimeController, repo: str, *, speculative: bool = True, **extra):
    return controller._select_engine(
        backend=extra.pop("backend", "mlx"),
        runtime_target=extra.pop("runtime_target", repo),
        path=extra.pop("path", None),
        model_ref=repo,
        canonical_repo=repo,
        speculative_decoding=speculative,
    )


class LegacyRoutingTests(unittest.TestCase):
    """D-77 through the controller on a Mac with both installs."""

    def _route(self, repo: str, caps: native.NativeCapabilities | None, *, python: bool = True, **extra):
        with mock.patch("backend_service.inference.controller.sys.platform", "darwin"), mock.patch.object(
            native, "cached_capabilities", lambda binary: caps
        ):
            return _select(_mac_controller(python=python), repo, **extra)

    def test_unlisted_legacy_families_route_to_the_python_install(self) -> None:
        for repo in (_DEEPSEEK, _BONSAI, _GEMMA_26B):
            self.assertIsInstance(self._route(repo, MAC), TensorFoldEngine, repo)
            self.assertNotIsInstance(self._route(repo, MAC), TensorFoldNativeEngine, repo)

    def test_listed_legacy_families_route_to_the_binary(self) -> None:
        caps = _with_families(MAC, deepseek_v4=("mlx-q4g64",), prism_hadamard_qwen35=("mlx-q2g64",))
        for repo in (_DEEPSEEK, _BONSAI):
            self.assertIsInstance(self._route(repo, caps), TensorFoldNativeEngine, repo)
        # Gemma 4 26B-A4B is refused by name, so a listed gemma4 does not move it.
        self.assertIsInstance(self._route(_GEMMA_26B, caps), TensorFoldEngine)

    def test_without_a_binary_report_the_flag_still_routes_to_python(self) -> None:
        for repo in (_DEEPSEEK, _BONSAI):
            self.assertIsInstance(self._route(repo, None), TensorFoldEngine, repo)

    def test_missing_python_install_names_it_when_the_binary_does_not_list_the_family(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            self._route(_DEEPSEEK, MAC, python=False)
        self.assertIn("older Python", str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            self._route(_DEEPSEEK, MAC, python=False, backend="tensorfold")
        self.assertIn("older Python", str(ctx.exception))

    def test_listed_family_needs_no_python_install(self) -> None:
        caps = _with_families(MAC, deepseek_v4=("mlx-q4g64",))
        self.assertIsInstance(self._route(_DEEPSEEK, caps, python=False), TensorFoldNativeEngine)

    def test_native_families_never_probe_the_binary(self) -> None:
        probe = mock.Mock(side_effect=AssertionError("probed"))
        with mock.patch("backend_service.inference.controller.sys.platform", "darwin"), mock.patch.object(
            native, "cached_capabilities", probe
        ):
            engine = _select(_mac_controller(python=True), "LiquidAI/LFM3-1.7B-MLX-4bit")
        self.assertIsInstance(engine, TensorFoldNativeEngine)
        probe.assert_not_called()

    def test_kolibri_mlx_build_stays_on_the_bundled_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "Kolibri-1-MLX-4bit"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"model_type": "kolibri1"}))
            with mock.patch.object(tf, "_stock_mlx_loads", return_value=False):
                engine = self._route(
                    "velaia/Kolibri-1-MLX-4bit", CUDA, path=str(model), runtime_target=str(model)
                )
        self.assertIsInstance(engine, MLXWorkerEngine)


class StatusRouteTests(unittest.TestCase):
    def test_native_status_carries_the_family_report(self) -> None:
        from backend_service.routes.setup import tensorfold_native as installer

        with mock.patch.object(native, "find_binary", lambda environ=None: Path("/fake/tensorfold-native")), mock.patch.object(
            native, "cached_capabilities", lambda binary: MAC
        ), mock.patch.object(native, "platform_key", lambda *a, **k: "macos-arm64"), mock.patch.object(
            installer, "_grammar_present", lambda binary: True
        ):
            state = installer.native_status()
        self.assertEqual(state["families"], native.family_report(MAC))
        self.assertEqual({e["modelType"] for e in state["families"] if e["servedHere"]}, set(MAC.families))

    def test_native_status_without_a_binary_has_no_families(self) -> None:
        from backend_service.routes.setup import tensorfold_native as installer

        with mock.patch.object(native, "find_binary", lambda environ=None: None), mock.patch.object(
            native, "platform_key", lambda *a, **k: "macos-arm64"
        ):
            state = installer.native_status()
        self.assertEqual(state["families"], [])


class CheckUpstreamsTests(unittest.TestCase):
    def _module(self):
        spec = importlib.util.spec_from_file_location("check_upstreams_r9", ROOT / "scripts" / "check-upstreams.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(module)
        return module

    def test_unlisted_families_reports_only_uncurated_types(self) -> None:
        module = self._module()
        for caps, expected in ((MAC, []), (CUDA, []), (_with_families(CUDA, zz_new=("bf16",)), ["zz_new"])):
            with mock.patch.object(native, "find_binary", lambda environ=None: Path("/fake/tensorfold-native")), mock.patch.object(
                native, "probe_capabilities", lambda binary, timeout=30.0, caps=caps: caps
            ):
                self.assertEqual(module.unlisted_families(), ("1.2.8", expected))


if __name__ == "__main__":
    unittest.main()
