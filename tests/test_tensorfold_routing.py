"""Controller tests for the TensorFold lane: engine selection tiers, startup
fallback, memory handling and orphan pruning.

Selection rules under test (see ``RuntimeController._select_tensorfold``):

* an explicit ``tensorfold`` backend always gets the engine;
* families only TensorFold can load ("exclusive") always get it, and an
  uninstalled TensorFold becomes an install prompt;
* checkpoints TensorFold is tested with ("tested") get it when speculative
  decoding is requested and it is installed — the MTPLX rule — else standard
  MLX;
* everything else stays on standard MLX unless asked for.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.inference import RuntimeController
from backend_service.inference.base import BackendCapabilities, LoadedModelInfo
from backend_service.inference.mlx_engine import MLXWorkerEngine
from backend_service.inference.tensorfold_engine import TensorFoldEngine

_FLASH = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"
_NATIVE = "Vontra/Qwen3.8-27B-MLX-4bit"
_COMMUNITY = "mlx-community/gemma-4-26b-a4b-it-4bit"
_UNRELATED = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def _capabilities(*, tensorfold: bool, mlx: bool = True) -> BackendCapabilities:
    return BackendCapabilities(
        pythonExecutable=sys.executable,
        mlxAvailable=mlx,
        mlxLmAvailable=mlx,
        mlxUsable=mlx,
        tensorfoldAvailable=tensorfold,
    )


def _controller(*, tensorfold: bool, mlx: bool = True) -> RuntimeController:
    with mock.patch(
        "backend_service.inference.get_backend_capabilities",
        return_value=_capabilities(tensorfold=tensorfold, mlx=mlx),
    ):
        controller = RuntimeController()
    controller.capabilities = _capabilities(tensorfold=tensorfold, mlx=mlx)
    return controller


def _select(controller: RuntimeController, repo: str, *, backend: str = "mlx", **extra):
    return controller._select_engine(
        backend=backend,
        runtime_target=extra.pop("runtime_target", repo),
        path=extra.pop("path", None),
        model_ref=repo,
        canonical_repo=repo,
        speculative_decoding=extra.pop("speculative_decoding", True),
    )


class SelectEngineTests(unittest.TestCase):
    def test_explicit_backend_gets_tensorfold_for_any_mlx_model(self) -> None:
        engine = _select(_controller(tensorfold=True), _COMMUNITY, backend="tensorfold")
        self.assertIsInstance(engine, TensorFoldEngine)
        engine = _select(_controller(tensorfold=True), _UNRELATED, backend="tensorfold")
        self.assertIsInstance(engine, TensorFoldEngine)

    def test_explicit_backend_needs_the_install(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            _select(_controller(tensorfold=False), _COMMUNITY, backend="tensorfold")
        self.assertIn("not installed", str(ctx.exception))

    def test_explicit_backend_refuses_gguf(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            _select(_controller(tensorfold=True), "unsloth/Some-Model-GGUF", backend="tensorfold")
        self.assertIn("GGUF", str(ctx.exception))

    def test_exclusive_family_routes_without_being_asked(self) -> None:
        for backend in ("mlx", "auto"):
            engine = _select(_controller(tensorfold=True), _FLASH, backend=backend, speculative_decoding=False)
            self.assertIsInstance(engine, TensorFoldEngine, backend)

    def test_exclusive_family_does_not_need_the_apps_own_mlx(self) -> None:
        engine = _select(_controller(tensorfold=True, mlx=False), _FLASH)
        self.assertIsInstance(engine, TensorFoldEngine)

    def test_exclusive_family_without_install_is_an_install_prompt(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            _select(_controller(tensorfold=False), _FLASH)
        message = str(ctx.exception)
        self.assertIn("Qwen3.8 Flash Next", message)
        self.assertIn("runs only on TensorFold", message)
        self.assertIn("128 GB", message)

    def test_exclusive_family_is_recognised_from_a_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "my-converted-model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
            engine = _select(_controller(tensorfold=True), "local/my-converted-model", path=str(model), runtime_target=str(model))
            self.assertIsInstance(engine, TensorFoldEngine)
            with self.assertRaises(RuntimeError):
                _select(_controller(tensorfold=False), "local/my-converted-model", path=str(model), runtime_target=str(model))

    def test_tested_checkpoints_follow_the_speculative_toggle(self) -> None:
        for repo in (_NATIVE, _COMMUNITY):
            self.assertIsInstance(_select(_controller(tensorfold=True), repo, speculative_decoding=True), TensorFoldEngine, repo)
            # toggle off: the standard MLX path, so the toggle is the A/B switch
            self.assertIsInstance(_select(_controller(tensorfold=True), repo, speculative_decoding=False), MLXWorkerEngine, repo)
            # not installed: standard MLX, never an error
            self.assertIsInstance(_select(_controller(tensorfold=False), repo, speculative_decoding=True), MLXWorkerEngine, repo)

    def test_tensorfold_takes_precedence_over_mtplx_for_its_checkpoints(self) -> None:
        controller = _controller(tensorfold=True)
        controller.capabilities.mtplxAvailable = True
        engine = _select(controller, _NATIVE, speculative_decoding=True)
        self.assertIsInstance(engine, TensorFoldEngine)

    def test_unrelated_models_stay_on_mlx_even_with_speculative_on(self) -> None:
        self.assertIsInstance(_select(_controller(tensorfold=True), _UNRELATED, speculative_decoding=True), MLXWorkerEngine)

    def test_gguf_target_never_routes_to_tensorfold(self) -> None:
        controller = _controller(tensorfold=True)
        controller.capabilities.ggufAvailable = True
        engine = controller._select_engine(
            backend="auto",
            runtime_target="/models/Vontra-Qwen3.8-27B-Q4_K_M.gguf",
            path="/models/Vontra-Qwen3.8-27B-Q4_K_M.gguf",
            model_ref=_NATIVE,
            canonical_repo=_NATIVE,
        )
        self.assertNotIsInstance(engine, TensorFoldEngine)


class _LoadedByStub:
    """Minimal engine double whose ``load_model`` records its call."""

    engine_name = "stub"

    def __init__(self, label: str, *, fail: Exception | None = None) -> None:
        self.label = label
        self.fail = fail
        self.loads: list[dict] = []
        self.unloads = 0

    def load_model(self, **kwargs):
        self.loads.append(kwargs)
        if self.fail is not None:
            raise self.fail
        return LoadedModelInfo(
            ref=kwargs["model_ref"], name=kwargs["model_name"], backend=kwargs["backend"], source=kwargs["source"],
            engine=self.engine_name, cacheStrategy=kwargs["cache_strategy"], cacheBits=kwargs["cache_bits"],
            fp16Layers=kwargs["fp16_layers"], fusedAttention=kwargs["fused_attention"],
            fitModelInMemory=kwargs["fit_model_in_memory"], contextTokens=kwargs["context_tokens"],
            loadedAt="now", path=kwargs.get("path"), runtimeTarget=kwargs.get("runtime_target"), runtimeNote=self.label,
        )

    def unload_model(self) -> None:
        self.unloads += 1

    def process_pid(self):
        return None


def _real_tensorfold_engine_that_fails(controller: RuntimeController, message: str) -> TensorFoldEngine:
    engine = TensorFoldEngine(controller.capabilities)
    engine.load_model = mock.Mock(side_effect=RuntimeError(message))  # type: ignore[method-assign]
    return engine


class StartupFallbackTests(unittest.TestCase):
    def _load(self, controller: RuntimeController, repo: str):
        return controller.load_model(
            model_ref=repo, model_name=repo, canonical_repo=repo, source="library", backend="mlx",
            path=None, runtime_target=repo, speculative_decoding=True,
        )

    def test_a_failed_tested_checkpoint_falls_back_to_standard_mlx(self) -> None:
        controller = _controller(tensorfold=True)
        controller._select_engine = mock.Mock(return_value=_real_tensorfold_engine_that_fails(controller, "Metal exploded"))
        fallback = _LoadedByStub("standard MLX")
        with mock.patch("backend_service.inference.controller.MLXWorkerEngine", return_value=fallback):
            loaded = self._load(controller, _NATIVE)
        self.assertIs(controller.engine, fallback)
        self.assertIn("TensorFold startup failed (Metal exploded); using standard MLX.", loaded.runtimeNote or "")

    def test_a_failed_exclusive_family_reports_the_error_instead(self) -> None:
        controller = _controller(tensorfold=True)
        controller._select_engine = mock.Mock(return_value=_real_tensorfold_engine_that_fails(controller, "weights do not fit"))
        with mock.patch("backend_service.inference.controller.MLXWorkerEngine") as standard:
            with self.assertRaises(RuntimeError) as ctx:
                self._load(controller, _FLASH)
        standard.assert_not_called()
        self.assertIn("weights do not fit", str(ctx.exception))
        self.assertIsNone(controller.loaded_model)

    def test_no_fallback_when_the_apps_mlx_is_unusable(self) -> None:
        controller = _controller(tensorfold=True, mlx=False)
        controller._select_engine = mock.Mock(return_value=_real_tensorfold_engine_that_fails(controller, "boom"))
        with self.assertRaises(RuntimeError):
            self._load(controller, _NATIVE)

    def test_an_explicit_request_for_an_unlisted_model_falls_back_too(self) -> None:
        controller = _controller(tensorfold=True)
        controller._select_engine = mock.Mock(return_value=_real_tensorfold_engine_that_fails(controller, "unsupported model_type"))
        fallback = _LoadedByStub("standard MLX")
        with mock.patch("backend_service.inference.controller.MLXWorkerEngine", return_value=fallback):
            loaded = self._load(controller, _COMMUNITY)
        self.assertIn("using standard MLX", loaded.runtimeNote or "")


class MemoryHandlingTests(unittest.TestCase):
    def test_loading_tensorfold_clears_warm_models_and_does_not_park_the_previous_one(self) -> None:
        controller = _controller(tensorfold=True)
        previous = _LoadedByStub("previous")
        previous.engine_name = "mlx"
        controller.engine = previous  # type: ignore[assignment]
        controller.loaded_model = previous.load_model(
            model_ref="other/model", model_name="other", backend="mlx", source="library", cache_strategy="native",
            cache_bits=0, fp16_layers=0, fused_attention=False, fit_model_in_memory=True, context_tokens=8192,
        )
        warm = _LoadedByStub("warm")
        controller._warm_pool["k"] = (warm, controller.loaded_model)  # type: ignore[assignment]

        tensorfold = TensorFoldEngine(controller.capabilities)
        tensorfold.load_model = mock.Mock(  # type: ignore[method-assign]
            return_value=LoadedModelInfo(
                ref=_NATIVE, name="n", backend="tensorfold", source="library", engine="tensorfold", cacheStrategy="native",
                cacheBits=0, fp16Layers=0, fusedAttention=False, fitModelInMemory=True, contextTokens=8192, loadedAt="now",
            )
        )
        controller._select_engine = mock.Mock(return_value=tensorfold)
        controller.load_model(
            model_ref=_NATIVE, model_name="n", canonical_repo=_NATIVE, source="library", backend="mlx",
            path=None, runtime_target=_NATIVE, keep_warm_previous=True,
        )
        self.assertEqual(warm.unloads, 1)
        self.assertEqual(previous.unloads, 1)
        self.assertEqual(controller._warm_pool, {})
        self.assertIs(controller.engine, tensorfold)


class OrphanPruningTests(unittest.TestCase):
    def test_an_untracked_tensorfold_server_is_pruned(self) -> None:
        controller = _controller(tensorfold=True)
        controller._tracked_process_pids = mock.Mock(return_value=set())
        orphan = mock.Mock()
        orphan.pid = 4242
        orphan.cmdline.return_value = [
            "/Users/x/.chaosengine/tensorfold-venv/bin/python",
            "/Users/x/.chaosengine/tensorfold-venv/bin/tensorfold",
            "serve",
            "/models/m",
        ]
        orphan.name.return_value = "python"
        orphan.create_time.return_value = 0.0
        parent = mock.Mock()
        parent.children.return_value = [orphan]
        with mock.patch("psutil.Process", return_value=parent):
            status = controller.status()
        orphan.terminate.assert_called_once()
        record = status["recentOrphanedWorkers"][0]
        self.assertEqual((record["pid"], record["kind"], record["label"]), (4242, "tensorfold", "TensorFold server"))

    def test_the_tracked_server_is_left_alone(self) -> None:
        controller = _controller(tensorfold=True)
        controller._tracked_process_pids = mock.Mock(return_value={4242})
        server = mock.Mock()
        server.pid = 4242
        server.cmdline.return_value = ["/x/.chaosengine/tensorfold-venv/bin/python", "tensorfold", "serve"]
        server.name.return_value = "python"
        server.create_time.return_value = 0.0
        parent = mock.Mock()
        parent.children.return_value = [server]
        with mock.patch("psutil.Process", return_value=parent):
            controller.prune_stale_backend_children()
        server.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
