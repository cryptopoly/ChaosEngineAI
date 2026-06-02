"""Unit tests for ``scripts/cache-strategy-matrix.py``.

Covers the pure functions (``skip_reason``, ``write_csv``, ``write_markdown``,
``print_summary``) without standing up a live backend. The HTTP layer
(``_api`` / ``_stream_inference`` / ``run_cell``) is exercised end-to-end
by the matrix runner itself when invoked against a running sidecar.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock


def _load_runner_module():
    """Import the runner script as a module despite the dash in its name."""
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "cache-strategy-matrix.py"
    spec = importlib.util.spec_from_file_location(
        "cache_strategy_matrix_runner", script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["cache_strategy_matrix_runner"] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner_module()


def _caps(
    *,
    available: set[str] | None = None,
    dflash: bool = True,
    ddtree: bool = True,
    mtplx: bool = True,
    gguf_mtp: bool = True,
    vllm: bool = True,
    turbo: bool = True,
    library: set[str] | None = None,
) -> "runner.BackendCapabilities":
    return runner.BackendCapabilities(
        available_strategies=available or {"native", "turboquant", "triattention"},
        dflash_available=dflash,
        ddtree_available=ddtree,
        mtplx_available=mtplx,
        gguf_mtp_available=gguf_mtp,
        vllm_available=vllm,
        has_turbo_binary=turbo,
        library_refs=library or {
            runner.SMALL_MLX,
            runner.MID_MLX_DFLASH_CAPABLE,
            runner.MID_MLX_MTPLX_CAPABLE,
            runner.SMALL_GGUF,
            runner.LARGE_GGUF_MTP,
            runner.VLLM_SMALL,
            runner.VLLM_MID,
        },
    )


class SkipReasonTests(unittest.TestCase):
    def test_quick_skips_non_quick_cells(self):
        cell = runner.MatrixCell(
            "heavy", runner.MID_MLX_DFLASH_CAPABLE, "mlx", "native", 0, "dflash",
            quick=False,
        )
        self.assertEqual(
            runner.skip_reason(cell, _caps(), quick=True),
            "deferred to full run (drop --quick)",
        )

    def test_full_run_keeps_non_quick_cells(self):
        cell = runner.MatrixCell(
            "heavy", runner.MID_MLX_DFLASH_CAPABLE, "mlx", "native", 0, "dflash",
            quick=False,
        )
        self.assertIsNone(runner.skip_reason(cell, _caps(), quick=False))

    def test_skips_when_strategy_unavailable(self):
        cell = runner.MatrixCell(
            "tri off", runner.SMALL_MLX, "mlx", "triattention", 3, "none",
        )
        caps = _caps(available={"native", "turboquant"})
        self.assertEqual(
            runner.skip_reason(cell, caps, quick=False),
            "strategy 'triattention' unavailable in this runtime",
        )

    def test_native_never_blocked_by_availability(self):
        cell = runner.MatrixCell("nat", runner.SMALL_MLX, "mlx", "native", 0, "none")
        caps = _caps(available=set())
        self.assertIsNone(runner.skip_reason(cell, caps, quick=False))

    def test_skips_gguf_turboquant_without_turbo_binary(self):
        cell = runner.MatrixCell("tq gguf", runner.SMALL_GGUF, "gguf", "turboquant", 3, "none")
        caps = _caps(turbo=False)
        self.assertEqual(
            runner.skip_reason(cell, caps, quick=False),
            "llama-server-turbo binary missing",
        )

    def test_skips_dflash_on_gguf_backend(self):
        cell = runner.MatrixCell("dflash gguf", runner.SMALL_GGUF, "gguf", "native", 0, "dflash")
        self.assertEqual(
            runner.skip_reason(cell, _caps(), quick=False),
            "speculative decoding requires MLX/vLLM, not GGUF",
        )

    def test_skips_dflash_when_runtime_missing(self):
        cell = runner.MatrixCell(
            "dflash mlx", runner.MID_MLX_DFLASH_CAPABLE, "mlx", "native", 0, "dflash",
        )
        self.assertEqual(
            runner.skip_reason(cell, _caps(dflash=False), quick=False),
            "DFlash runtime not installed",
        )

    def test_skips_ddtree_when_runtime_missing(self):
        cell = runner.MatrixCell(
            "ddtree mlx", runner.MID_MLX_DFLASH_CAPABLE, "mlx", "native", 0, "ddtree", tree_budget=8,
        )
        self.assertEqual(
            runner.skip_reason(cell, _caps(ddtree=False), quick=False),
            "DDTree runtime not available",
        )

    def test_skips_when_model_not_in_library(self):
        cell = runner.MatrixCell(
            "missing", "made-up/unicorn-1B", "mlx", "native", 0, "none",
        )
        self.assertIn("model not in library", runner.skip_reason(cell, _caps(), quick=False))

    def test_legacy_chaosengine_uses_turboquant_availability(self):
        """FU-030: ``chaosengine`` must canonicalise to ``turboquant`` for
        the availability check; otherwise legacy persisted configs would
        always skip even when TurboQuant is installed."""
        cell = runner.MatrixCell(
            "legacy", runner.SMALL_MLX, "mlx", "chaosengine", 4, "none",
        )
        # turboquant present, chaosengine obviously not present in registry
        caps = _caps(available={"native", "turboquant"})
        self.assertIsNone(runner.skip_reason(cell, caps, quick=False))

    def test_legacy_chaosengine_skips_when_turboquant_unavailable(self):
        """The flip side of the previous test — if TurboQuant itself isn't
        installed, the legacy id should also skip with the canonical name
        in the message so users know what to install."""
        cell = runner.MatrixCell(
            "legacy", runner.SMALL_MLX, "mlx", "chaosengine", 4, "none",
        )
        caps = _caps(available={"native"})
        self.assertEqual(
            runner.skip_reason(cell, caps, quick=False),
            "strategy 'turboquant' unavailable in this runtime",
        )


class ClassifyLoadSkipTests(unittest.TestCase):
    """FU-070: a catalogued-but-undownloaded model fails at load time, not
    at the library check (``library_refs`` is catalog-derived). The runner
    classifies a 'no weights on disk' load error as a skip, not a fail."""

    def test_classifies_missing_gguf_weights_as_skip(self):
        msg = (
            "API POST /api/models/load -> 500: Cannot load "
            "'ggml-org/Qwen3.6-27B-MTP-GGUF': No .gguf, .safetensors, or "
            "pytorch weights found in HF cache entry."
        )
        self.assertEqual(runner.classify_load_skip(msg), "weights not downloaded")

    def test_classifies_hf_cache_entry_phrasing_as_skip(self):
        msg = "load failed (HTTP 500): no weights found in HF cache entry for repo"
        self.assertEqual(runner.classify_load_skip(msg), "weights not downloaded")

    def test_genuine_load_error_is_not_a_skip(self):
        msg = "API POST /api/models/load -> 500: CUDA out of memory"
        self.assertIsNone(runner.classify_load_skip(msg))

    def test_empty_message_is_not_a_skip(self):
        self.assertIsNone(runner.classify_load_skip(""))


class WriteCsvTests(unittest.TestCase):
    def test_writes_header_and_rows(self):
        results = [
            runner.CellResult(
                label="ok",
                model_ref="m/x",
                backend="mlx",
                strategy="native",
                bits=0,
                spec_decode="none",
                tree_budget=0,
                ok=True,
                tokens_per_sec=42.0,
                output_sha="deadbeef0000",
                output_chars=128,
                actual_strategy="native",
                runtime_note="ok",
                duration_seconds=1.5,
            ),
            runner.CellResult(
                label="skipped",
                model_ref="m/y",
                backend="gguf",
                strategy="turboquant",
                bits=3,
                spec_decode="none",
                tree_budget=0,
                skipped=True,
                skip_reason="missing binary",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = runner.write_csv(Path(tmp), results)
            text = csv_path.read_text(encoding="utf-8")

        self.assertIn("label,model_ref,backend", text)
        self.assertIn("ok,m/x,mlx,native,0,none,0,False,,True,,42.00,1.50", text)
        self.assertIn("skipped,m/y,gguf,turboquant,3,none,0,True,missing binary", text)


class WriteMarkdownTests(unittest.TestCase):
    def test_markdown_includes_legacy_alias_table_when_legacy_rows_present(self):
        results = [
            runner.CellResult(
                label="legacy chaosengine",
                model_ref=runner.SMALL_MLX,
                backend="mlx",
                strategy="chaosengine",
                bits=4,
                spec_decode="none",
                tree_budget=0,
                ok=True,
                actual_strategy="turboquant",
                runtime_note="coerced",
                tokens_per_sec=22.0,
                output_sha="cafebabe1234",
            ),
            runner.CellResult(
                label="native baseline",
                model_ref=runner.SMALL_MLX,
                backend="mlx",
                strategy="native",
                bits=0,
                spec_decode="none",
                tree_budget=0,
                ok=True,
                actual_strategy="native",
                tokens_per_sec=20.0,
                output_sha="aaaa11112222",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            md_path = runner.write_markdown(Path(tmp), results)
            text = md_path.read_text(encoding="utf-8")

        self.assertIn("FU-030 legacy alias coercion", text)
        self.assertIn("| chaosengine | turboquant | yes |", text)

    def test_markdown_flags_coercion_regression(self):
        results = [
            runner.CellResult(
                label="legacy rotorquant",
                model_ref=runner.SMALL_MLX,
                backend="mlx",
                strategy="rotorquant",
                bits=3,
                spec_decode="none",
                tree_budget=0,
                ok=True,
                actual_strategy="native",  # wrong — should be turboquant
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            md_path = runner.write_markdown(Path(tmp), results)
            text = md_path.read_text(encoding="utf-8")

        self.assertIn("| rotorquant | native | **NO** |", text)


class PrintSummaryTests(unittest.TestCase):
    def _stub_stdout(self):
        """Replace stdout for the duration of one test."""
        return mock.patch("sys.stdout", new_callable=StringIO)

    def test_returns_zero_on_all_pass(self):
        results = [
            runner.CellResult(
                label="a", model_ref="m", backend="mlx", strategy="native",
                bits=0, spec_decode="none", tree_budget=0, ok=True,
            ),
        ]
        with self._stub_stdout():
            self.assertEqual(runner.print_summary(results), 0)

    def test_returns_one_on_failure(self):
        results = [
            runner.CellResult(
                label="a", model_ref="m", backend="mlx", strategy="native",
                bits=0, spec_decode="none", tree_budget=0,
                ok=False, error="boom",
            ),
        ]
        with self._stub_stdout():
            self.assertEqual(runner.print_summary(results), 1)

    def test_returns_two_on_coercion_regression(self):
        results = [
            runner.CellResult(
                label="legacy", model_ref=runner.SMALL_MLX, backend="mlx",
                strategy="chaosengine", bits=4, spec_decode="none", tree_budget=0,
                ok=True, actual_strategy="native",  # wrong
            ),
        ]
        with self._stub_stdout():
            self.assertEqual(runner.print_summary(results), 2)


class MatrixDefinitionTests(unittest.TestCase):
    def test_matrix_includes_legacy_coercion_cells(self):
        labels = {cell.label for cell in runner.MATRIX}
        self.assertIn("legacy id chaosengine -> turboquant", labels)
        self.assertIn("legacy id rotorquant  -> turboquant", labels)

    def test_matrix_strategy_ids_use_active_or_legacy_set(self):
        active = {"native", "turboquant", "triattention"}
        legacy = {"chaosengine", "rotorquant"}
        for cell in runner.MATRIX:
            self.assertIn(
                cell.strategy, active | legacy,
                f"unknown strategy in matrix: {cell.strategy}",
            )

    def test_matrix_backends_are_supported(self):
        for cell in runner.MATRIX:
            self.assertIn(cell.backend, ("mlx", "gguf", "vllm"))


if __name__ == "__main__":
    unittest.main()
