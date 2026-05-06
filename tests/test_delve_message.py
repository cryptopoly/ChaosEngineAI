"""Phase 3.6 tests for delve_message."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from backend_service.inference import LoadedModelInfo
from backend_service.state import ChaosEngineState


def _fake_system_snapshot(capabilities=None):
    return {
        "platform": "Darwin",
        "arch": "arm64",
        "hardwareSummary": "test",
        "backendLabel": "test",
        "appVersion": "test",
        "mlxAvailable": False,
        "mlxLmAvailable": False,
        "mlxUsable": False,
        "ggufAvailable": False,
        "converterAvailable": False,
        "totalMemoryGb": 16.0,
        "availableMemoryGb": 8.0,
        "usedMemoryGb": 8.0,
        "swapUsedGb": 0.0,
        "cpuUtilizationPercent": 10.0,
        "gpuUtilizationPercent": None,
        "spareHeadroomGb": 4.0,
        "runningLlmProcesses": [],
    }


@dataclass
class _FakeResult:
    text: str = "Critique: Looks fine.\n\nRevised answer: Same as before."
    finishReason: str = "stop"
    promptTokens: int = 60
    completionTokens: int = 30
    totalTokens: int = 90
    tokS: float = 18.0
    responseSeconds: float = 1.2
    runtimeNote: str | None = None
    dflashAcceptanceRate: float | None = None
    cache_strategy: str | None = None
    cache_bits: int | None = None
    fp16_layers: int | None = None
    speculative_decoding: bool | None = None
    tree_budget: int | None = None


class _FakeEngine:
    engine_label = "fake"


class _FakeRuntime:
    def __init__(self, loaded_model: LoadedModelInfo | None):
        self.runtime_note = None
        self.loaded_model = loaded_model
        self.engine = _FakeEngine()
        self.last_call: dict | None = None

    def status(self, **_kwargs):
        return {"engineLabel": self.engine.engine_label}

    def generate(self, **kwargs):
        self.last_call = kwargs
        return _FakeResult()


def _make_loaded() -> LoadedModelInfo:
    return LoadedModelInfo(
        ref="critic/model-7b",
        name="Critic 7B",
        backend="auto",
        source="library",
        engine="llamacpp",
        cacheStrategy="native",
        cacheBits=8,
        fp16Layers=0,
        fusedAttention=False,
        fitModelInMemory=True,
        contextTokens=4096,
        loadedAt="2026-05-02T00:00:00Z",
        canonicalRepo=None,
        path=None,
    )


def _make_state(tmp_path: Path, runtime: _FakeRuntime) -> ChaosEngineState:
    state = ChaosEngineState(
        system_snapshot_provider=_fake_system_snapshot,
        library_provider=lambda: [],
        settings_path=tmp_path / "settings.json",
        benchmarks_path=tmp_path / "benchmarks.json",
        chat_sessions_path=tmp_path / "chat_sessions.json",
    )
    state.runtime = runtime
    return state


class DelveMessageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.runtime = _FakeRuntime(_make_loaded())
        self.state = _make_state(Path(self._tmp.name), self.runtime)
        self.session = self.state.create_session(title="Delve test")
        self.session["messages"] = [
            {"role": "user", "text": "Why is the sky blue?"},
            {
                "role": "assistant",
                "text": "Because of Rayleigh scattering of light.",
                "metrics": {"tokS": 30.0},
            },
        ]
        self.state._persist_sessions()

    def tearDown(self):
        self._tmp.cleanup()

    def test_attaches_critique_variant(self):
        updated = self.state.delve_message(
            session_id=self.session["id"],
            message_index=1,
        )
        variants = updated["messages"][1].get("variants")
        self.assertEqual(len(variants), 1)
        variant = variants[0]
        self.assertEqual(variant["modelName"], "Delve critique")
        self.assertIn("Critique:", variant["text"])

    def test_critique_system_prompt_passes_through(self):
        self.state.delve_message(
            session_id=self.session["id"],
            message_index=1,
        )
        self.assertIsNotNone(self.runtime.last_call)
        self.assertIn("critic", self.runtime.last_call["system_prompt"].lower())

    def test_history_contains_original_answer(self):
        self.state.delve_message(
            session_id=self.session["id"],
            message_index=1,
        )
        history = self.runtime.last_call["history"]
        # History ends with the assistant's original answer so the
        # critique pass has full context to react to.
        self.assertEqual(history[-1]["role"], "assistant")
        self.assertIn("Rayleigh", history[-1]["text"])

    def test_rejects_user_message(self):
        with self.assertRaises(ValueError):
            self.state.delve_message(
                session_id=self.session["id"],
                message_index=0,
            )

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            self.state.delve_message(
                session_id=self.session["id"],
                message_index=99,
            )

    def test_rejects_when_no_model_loaded(self):
        self.runtime.loaded_model = None
        with self.assertRaises(ValueError):
            self.state.delve_message(
                session_id=self.session["id"],
                message_index=1,
            )


if __name__ == "__main__":
    unittest.main()
