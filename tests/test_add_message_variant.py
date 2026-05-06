"""Tests for the Phase 2.5 in-thread compare `add_message_variant`.

Variant generation re-runs the user prompt through a different warm
model and attaches the result to the original assistant message's
``variants`` list — so the frontend can render side-by-side answers
under the primary bubble.
"""

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
    text: str = "Alt response"
    finishReason: str = "stop"
    promptTokens: int = 10
    completionTokens: int = 20
    totalTokens: int = 30
    tokS: float = 25.0
    responseSeconds: float = 0.8
    runtimeNote: str | None = None
    dflashAcceptanceRate: float | None = None
    cache_strategy: str | None = None
    cache_bits: int | None = None
    fp16_layers: int | None = None
    speculative_decoding: bool | None = None
    tree_budget: int | None = None


class _FakeEngine:
    engine_label = "fake-llamacpp"


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


def _make_loaded(ref: str, name: str = "Override Model") -> LoadedModelInfo:
    return LoadedModelInfo(
        ref=ref,
        name=name,
        backend="auto",
        source="library",
        engine="llamacpp",
        cacheStrategy="native",
        cacheBits=8,
        fp16Layers=0,
        fusedAttention=False,
        fitModelInMemory=True,
        contextTokens=4096,
        loadedAt="2026-05-01T00:00:00Z",
        canonicalRepo=None,
        path="/tmp/model.gguf",
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


class AddMessageVariantTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.loaded = _make_loaded("alt/model-7b", name="Alt 7B")
        self.runtime = _FakeRuntime(self.loaded)
        self.state = _make_state(Path(self._tmp.name), self.runtime)
        self.session = self.state.create_session(title="Compare test")
        self.session["messages"] = [
            {"role": "user", "text": "What's 2+2?"},
            {
                "role": "assistant",
                "text": "Four.",
                "metrics": {"tokS": 30.0, "model": "Primary", "modelRef": "primary/model"},
            },
        ]
        self.state._persist_sessions()

    def tearDown(self):
        self._tmp.cleanup()

    def test_attaches_variant_to_assistant_message(self):
        updated = self.state.add_message_variant(
            session_id=self.session["id"],
            message_index=1,
            model_ref="alt/model-7b",
            model_name="Alt 7B",
            canonical_repo=None,
            source="library",
            path="/tmp/alt.gguf",
            backend="auto",
            max_tokens=128,
            temperature=0.7,
        )
        variants = updated["messages"][1].get("variants")
        self.assertIsNotNone(variants)
        self.assertEqual(len(variants), 1)
        variant = variants[0]
        self.assertEqual(variant["modelRef"], "alt/model-7b")
        self.assertEqual(variant["modelName"], "Alt 7B")
        self.assertEqual(variant["text"], "Alt response")
        self.assertIn("metrics", variant)
        self.assertEqual(variant["metrics"]["model"], "Alt 7B")
        self.assertEqual(variant["metrics"]["tokS"], 25.0)

    def test_passes_user_prompt_to_runtime(self):
        self.state.add_message_variant(
            session_id=self.session["id"],
            message_index=1,
            model_ref="alt/model-7b",
            model_name="Alt 7B",
            canonical_repo=None,
            source="library",
            path=None,
            backend="auto",
            max_tokens=64,
            temperature=0.5,
        )
        self.assertIsNotNone(self.runtime.last_call)
        self.assertEqual(self.runtime.last_call["prompt"], "What's 2+2?")
        self.assertEqual(self.runtime.last_call["max_tokens"], 64)
        self.assertEqual(self.runtime.last_call["temperature"], 0.5)

    def test_appends_multiple_variants(self):
        for tag in ("alt-a", "alt-b"):
            self.state.add_message_variant(
                session_id=self.session["id"],
                message_index=1,
                model_ref="alt/model-7b",
                model_name=f"Alt {tag}",
                canonical_repo=None,
                source="library",
                path=None,
                backend="auto",
                max_tokens=32,
                temperature=0.7,
            )
        variants = self.state.chat_sessions[0]["messages"][1]["variants"]
        self.assertEqual(len(variants), 2)
        self.assertEqual(variants[0]["modelName"], "Alt alt-a")
        self.assertEqual(variants[1]["modelName"], "Alt alt-b")

    def test_rejects_user_message_index(self):
        with self.assertRaises(ValueError):
            self.state.add_message_variant(
                session_id=self.session["id"],
                message_index=0,
                model_ref="alt/model-7b",
                model_name="Alt 7B",
                canonical_repo=None,
                source="library",
                path=None,
                backend="auto",
                max_tokens=32,
                temperature=0.7,
            )

    def test_rejects_out_of_range_index(self):
        with self.assertRaises(ValueError):
            self.state.add_message_variant(
                session_id=self.session["id"],
                message_index=99,
                model_ref="alt/model-7b",
                model_name="Alt 7B",
                canonical_repo=None,
                source="library",
                path=None,
                backend="auto",
                max_tokens=32,
                temperature=0.7,
            )

    def test_rejects_unknown_session(self):
        with self.assertRaises(ValueError):
            self.state.add_message_variant(
                session_id="missing",
                message_index=1,
                model_ref="alt/model-7b",
                model_name="Alt 7B",
                canonical_repo=None,
                source="library",
                path=None,
                backend="auto",
                max_tokens=32,
                temperature=0.7,
            )

    def test_rejects_when_runtime_model_mismatches(self):
        # Runtime currently has alt/model-7b loaded; ask for a
        # different ref → should fail rather than auto-reload.
        with self.assertRaises(ValueError):
            self.state.add_message_variant(
                session_id=self.session["id"],
                message_index=1,
                model_ref="other/model-13b",
                model_name="Other 13B",
                canonical_repo=None,
                source="library",
                path=None,
                backend="auto",
                max_tokens=32,
                temperature=0.7,
            )

    def test_rejects_when_no_model_loaded(self):
        self.runtime.loaded_model = None
        with self.assertRaises(ValueError):
            self.state.add_message_variant(
                session_id=self.session["id"],
                message_index=1,
                model_ref="alt/model-7b",
                model_name="Alt 7B",
                canonical_repo=None,
                source="library",
                path=None,
                backend="auto",
                max_tokens=32,
                temperature=0.7,
            )


if __name__ == "__main__":
    unittest.main()
