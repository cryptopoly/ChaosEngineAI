"""Tests for the Phase 2.4 conversation-branching `fork_session` method.

Forking deep-copies messages [0..forkAtMessageIndex] from the source
thread into a fresh session and tags the new session with
`parentSessionId` + `forkedAtMessageIndex` so the sidebar can render
a relationship hint and future merge / diff features have the
linkage.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend_service.state import ChaosEngineState


def _fake_system_snapshot(capabilities=None):
    """Minimal snapshot — fork_session reads nothing from here."""
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


class _FakeRuntime:
    """Minimal stand-in for tests — exposes nothing fork_session uses."""

    runtime_note = None
    loaded_model = None

    def status(self, **_kwargs):
        return {"engineLabel": "test"}


def _make_state(tmp_path: Path) -> ChaosEngineState:
    state = ChaosEngineState(
        system_snapshot_provider=_fake_system_snapshot,
        library_provider=lambda: [],
        settings_path=tmp_path / "settings.json",
        benchmarks_path=tmp_path / "benchmarks.json",
        chat_sessions_path=tmp_path / "chat_sessions.json",
    )
    state.runtime = _FakeRuntime()
    return state


class ForkSessionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.state = _make_state(Path(self._tmp.name))
        self.source = self.state.create_session(title="Original")
        # Seed a few alternating user/assistant turns.
        self.source["messages"] = [
            {"role": "user", "text": "Hello"},
            {"role": "assistant", "text": "Hi there", "metrics": {"tokS": 5.0}},
            {"role": "user", "text": "Tell me about cats"},
            {"role": "assistant", "text": "Cats are great", "metrics": {"tokS": 7.0}},
        ]
        self.source["model"] = "Test/Model"
        self.source["modelRef"] = "test/model-7b"
        self.source["thinkingMode"] = "auto"
        self.state._persist_sessions()

    def tearDown(self):
        self._tmp.cleanup()

    def test_fork_copies_messages_up_to_index(self):
        fork = self.state.fork_session(self.source["id"], fork_at_message_index=1)
        # Index 1 = first assistant turn — fork should hold first
        # user + first assistant only.
        self.assertEqual(len(fork["messages"]), 2)
        self.assertEqual(fork["messages"][0]["text"], "Hello")
        self.assertEqual(fork["messages"][1]["text"], "Hi there")

    def test_fork_carries_parent_linkage(self):
        fork = self.state.fork_session(self.source["id"], fork_at_message_index=3)
        self.assertEqual(fork["parentSessionId"], self.source["id"])
        self.assertEqual(fork["forkedAtMessageIndex"], 3)

    def test_fork_carries_runtime_profile(self):
        fork = self.state.fork_session(self.source["id"], fork_at_message_index=1)
        self.assertEqual(fork["model"], "Test/Model")
        self.assertEqual(fork["modelRef"], "test/model-7b")
        self.assertEqual(fork["thinkingMode"], "auto")

    def test_fork_default_title(self):
        fork = self.state.fork_session(self.source["id"], fork_at_message_index=1)
        self.assertIn("Original", fork["title"])
        self.assertIn("fork", fork["title"].lower())

    def test_fork_custom_title(self):
        fork = self.state.fork_session(
            self.source["id"],
            fork_at_message_index=1,
            title="Cat tangent",
        )
        self.assertEqual(fork["title"], "Cat tangent")

    def test_fork_inserts_at_top_of_session_list(self):
        fork = self.state.fork_session(self.source["id"], fork_at_message_index=1)
        self.assertEqual(self.state.chat_sessions[0]["id"], fork["id"])

    def test_fork_messages_are_deep_copied(self):
        fork = self.state.fork_session(self.source["id"], fork_at_message_index=1)
        # Mutating the fork's messages must not bleed into the parent.
        fork["messages"][0]["text"] = "MUTATED"
        self.assertEqual(self.source["messages"][0]["text"], "Hello")

    def test_fork_unknown_session_raises(self):
        with self.assertRaises(ValueError):
            self.state.fork_session("nonexistent-id", fork_at_message_index=0)

    def test_fork_index_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self.state.fork_session(self.source["id"], fork_at_message_index=99)

    def test_fork_negative_index_raises(self):
        with self.assertRaises(ValueError):
            self.state.fork_session(self.source["id"], fork_at_message_index=-1)


if __name__ == "__main__":
    unittest.main()
