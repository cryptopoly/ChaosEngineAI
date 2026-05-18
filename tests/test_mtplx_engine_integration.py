"""Integration tests for MtplxEngine using a stub mtplx server.

Verifies the full spawn → /health probe → /v1/chat/completions round-trip
without needing the real MTPLX install or any MTP-bearing model on disk.

Stub lives at ``tests/fixtures/stub_mtplx_server.py`` and implements the
minimal surface ``MtplxEngine`` talks to.
"""

from __future__ import annotations

import stat
import sys
import textwrap
import unittest
from pathlib import Path

from backend_service.inference.base import BackendCapabilities
from backend_service.inference.mtplx_engine import MtplxEngine

# The integration fixtures spawn a ``#!/usr/bin/env bash`` wrapper
# script — POSIX-only. Windows can't honour that shebang, so the whole
# integration class skips there. MTPLX itself is also macOS-Apple-Silicon
# in production, so this also covers the "MTPLX runtime unavailable" path.
_REQUIRES_POSIX = "MTPLX integration fixtures need a POSIX shell"


_FIXTURES = Path(__file__).parent / "fixtures"
_STUB_SCRIPT = _FIXTURES / "stub_mtplx_server.py"


def _make_mtplx_wrapper(tmp_path: Path, *, fail_mode: str | None = None) -> Path:
    """Write an executable wrapper that mimics ``mtplx`` CLI.

    MtplxEngine spawns ``[bin, "start", "--model", X, "--port", N]`` — so the
    wrapper just forwards argv into the python stub.
    """
    wrapper = tmp_path / "mtplx"
    extra = f' --fail-mode {fail_mode}' if fail_mode else ""
    wrapper.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        exec {sys.executable} {_STUB_SCRIPT} "$@"{extra}
        """))
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


def _make_capabilities(mtplx_python: str) -> BackendCapabilities:
    return BackendCapabilities(
        pythonExecutable=sys.executable,
        mlxAvailable=True,
        mlxLmAvailable=True,
        mlxUsable=True,
        mtplxAvailable=True,
        mtplxPythonPath=mtplx_python,
    )


@unittest.skipIf(sys.platform == "win32", _REQUIRES_POSIX)
class MtplxEngineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.wrapper = _make_mtplx_wrapper(self.tmp_path)
        # Place a fake python sibling so capabilities resolver is satisfied
        (self.tmp_path / "python").write_text("#!/usr/bin/env bash\n")
        (self.tmp_path / "python").chmod(0o755)
        self.capabilities = _make_capabilities(str(self.tmp_path / "python"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_engine(self) -> MtplxEngine:
        engine = MtplxEngine(self.capabilities)
        # Patch _mtplx_bin so the engine resolves to the test wrapper rather
        # than the production ~/.chaosengine/mtplx-venv path.
        engine._mtplx_bin = lambda: str(self.wrapper)  # type: ignore[method-assign]
        return engine

    def test_load_model_starts_server_and_returns_info(self) -> None:
        engine = self._make_engine()
        try:
            info = engine.load_model(
                model_ref="Qwen/Qwen3.5-7B",
                model_name="Qwen3.5-7B",
                canonical_repo="Qwen/Qwen3.5-7B",
                source="catalog",
                backend="mtplx",
                path=None,
                runtime_target=None,
                cache_strategy="native",
                cache_bits=0,
                fp16_layers=0,
                fused_attention=False,
                fit_model_in_memory=True,
                context_tokens=8192,
                speculative_decoding=True,
            )
            self.assertEqual(info.ref, "Qwen/Qwen3.5-7B")
            self.assertEqual(info.engine, "mtplx")
            self.assertTrue(info.speculativeDecoding)
            self.assertIsNotNone(info.runtimeNote)
            self.assertIn("MTPLX", info.runtimeNote or "")
            self.assertIn("draft tokens", info.runtimeNote or "")
            self.assertIsNotNone(engine.port)
            self.assertIsNotNone(engine.process_pid())
        finally:
            engine.unload_model()
        self.assertIsNone(engine.process_pid())

    def test_load_model_raises_when_capability_missing(self) -> None:
        caps = BackendCapabilities(pythonExecutable=sys.executable, mlxAvailable=True, mlxLmAvailable=True, mlxUsable=True, mtplxAvailable=False)
        engine = MtplxEngine(caps)
        engine._mtplx_bin = lambda: str(self.wrapper)  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError) as ctx:
            engine.load_model(
                model_ref="Qwen/Qwen3.5-7B",
                model_name="Qwen3.5-7B",
                canonical_repo="Qwen/Qwen3.5-7B",
                source="catalog",
                backend="mtplx",
                path=None,
                runtime_target=None,
                cache_strategy="native",
                cache_bits=0,
                fp16_layers=0,
                fused_attention=False,
                fit_model_in_memory=True,
                context_tokens=8192,
            )
        self.assertIn("not installed", str(ctx.exception).lower())

    def test_load_model_raises_when_server_exits_during_startup(self) -> None:
        crash_dir = self.tmp_path / "crash"
        crash_dir.mkdir()
        bad_wrapper = _make_mtplx_wrapper(crash_dir, fail_mode="crash-before-ready")
        engine = MtplxEngine(self.capabilities)
        engine._mtplx_bin = lambda: str(bad_wrapper)  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            engine.load_model(
                model_ref="Qwen/Qwen3.5-7B",
                model_name="Qwen3.5-7B",
                canonical_repo="Qwen/Qwen3.5-7B",
                source="catalog",
                backend="mtplx",
                path=None,
                runtime_target=None,
                cache_strategy="native",
                cache_bits=0,
                fp16_layers=0,
                fused_attention=False,
                fit_model_in_memory=True,
                context_tokens=8192,
            )
        self.assertIsNone(engine.process_pid())

    def test_generate_round_trip_returns_text_and_tokens(self) -> None:
        engine = self._make_engine()
        try:
            engine.load_model(
                model_ref="Qwen/Qwen3.5-7B",
                model_name="Qwen3.5-7B",
                canonical_repo="Qwen/Qwen3.5-7B",
                source="catalog",
                backend="mtplx",
                path=None,
                runtime_target=None,
                cache_strategy="native",
                cache_bits=0,
                fp16_layers=0,
                fused_attention=False,
                fit_model_in_memory=True,
                context_tokens=8192,
            )
            result = engine.generate(
                prompt="Hello",
                history=[],
                system_prompt="You are a stub.",
                max_tokens=32,
                temperature=0.7,
            )
            self.assertEqual(result.text, "stub-mtplx says hi")
            self.assertEqual(result.finishReason, "stop")
            self.assertGreater(result.completionTokens, 0)
            self.assertGreater(result.promptTokens, 0)
            self.assertGreater(result.tokS, 0.0)
            self.assertIn("MTPLX", result.runtimeNote or "")
        finally:
            engine.unload_model()

    def test_stream_generate_yields_text_then_done(self) -> None:
        engine = self._make_engine()
        try:
            engine.load_model(
                model_ref="Qwen/Qwen3.5-7B",
                model_name="Qwen3.5-7B",
                canonical_repo="Qwen/Qwen3.5-7B",
                source="catalog",
                backend="mtplx",
                path=None,
                runtime_target=None,
                cache_strategy="native",
                cache_bits=0,
                fp16_layers=0,
                fused_attention=False,
                fit_model_in_memory=True,
                context_tokens=8192,
            )
            text_chunks = []
            done_chunk = None
            for chunk in engine.stream_generate(
                prompt="Hello",
                history=[],
                system_prompt=None,
                max_tokens=32,
                temperature=0.7,
            ):
                if chunk.text:
                    text_chunks.append(chunk.text)
                if chunk.done:
                    done_chunk = chunk
            joined = "".join(text_chunks)
            self.assertIn("stub", joined)
            self.assertIn("hi", joined)
            self.assertIsNotNone(done_chunk)
            assert done_chunk is not None
            self.assertEqual(done_chunk.finish_reason, "stop")
            self.assertGreater(done_chunk.completion_tokens, 0)
        finally:
            engine.unload_model()

    def test_generate_after_unload_raises(self) -> None:
        engine = self._make_engine()
        try:
            engine.load_model(
                model_ref="Qwen/Qwen3.5-7B",
                model_name="Qwen3.5-7B",
                canonical_repo="Qwen/Qwen3.5-7B",
                source="catalog",
                backend="mtplx",
                path=None,
                runtime_target=None,
                cache_strategy="native",
                cache_bits=0,
                fp16_layers=0,
                fused_attention=False,
                fit_model_in_memory=True,
                context_tokens=8192,
            )
        finally:
            engine.unload_model()
        with self.assertRaises(RuntimeError):
            engine.generate(
                prompt="Hello",
                history=[],
                system_prompt=None,
                max_tokens=32,
                temperature=0.7,
            )

    def test_unload_idempotent(self) -> None:
        engine = self._make_engine()
        engine.unload_model()
        engine.unload_model()
        self.assertIsNone(engine.process_pid())


class MtplxEngineControllerFallbackTests(unittest.TestCase):
    """Verify the controller falls back to MLXWorkerEngine when MTPLX startup fails."""

    def test_controller_select_engine_picks_mtplx_when_model_supported(self) -> None:
        from backend_service.inference.controller import RuntimeController

        controller = RuntimeController()
        controller.capabilities = BackendCapabilities(
            pythonExecutable=sys.executable,
            mlxAvailable=True,
            mlxLmAvailable=True,
            mlxUsable=True,
            mtplxAvailable=True,
        )
        engine = controller._select_engine(
            backend="mlx",
            runtime_target=None,
            path=None,
            model_ref="Qwen/Qwen3.5-7B",
            canonical_repo="Qwen/Qwen3.5-7B",
            speculative_decoding=True,
        )
        self.assertEqual(engine.engine_name, "mtplx")

    def test_controller_select_engine_falls_through_when_model_not_supported(self) -> None:
        from backend_service.inference.controller import RuntimeController
        from backend_service.inference.mlx_engine import MLXWorkerEngine

        controller = RuntimeController()
        controller.capabilities = BackendCapabilities(
            pythonExecutable=sys.executable,
            mlxAvailable=True,
            mlxLmAvailable=True,
            mlxUsable=True,
            mtplxAvailable=True,
        )
        engine = controller._select_engine(
            backend="mlx",
            runtime_target=None,
            path=None,
            model_ref="some/random-model-without-mtp",
            canonical_repo="some/random-model-without-mtp",
            speculative_decoding=True,
        )
        self.assertIsInstance(engine, MLXWorkerEngine)

    def test_controller_select_engine_skips_mtplx_when_speculative_off(self) -> None:
        from backend_service.inference.controller import RuntimeController
        from backend_service.inference.mlx_engine import MLXWorkerEngine

        controller = RuntimeController()
        controller.capabilities = BackendCapabilities(
            pythonExecutable=sys.executable,
            mlxAvailable=True,
            mlxLmAvailable=True,
            mlxUsable=True,
            mtplxAvailable=True,
        )
        engine = controller._select_engine(
            backend="mlx",
            runtime_target=None,
            path=None,
            model_ref="Qwen/Qwen3.5-7B",
            canonical_repo="Qwen/Qwen3.5-7B",
            speculative_decoding=False,
        )
        self.assertIsInstance(engine, MLXWorkerEngine)


if __name__ == "__main__":
    unittest.main()
