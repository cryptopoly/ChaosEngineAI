"""Tests for the shared `backend_service.runaway_guard` module.

Phase 2.0.5-F moved RunawayGuard out of `mlx_worker` so the llama.cpp
stream loop can use the same detector. These cases exercise the public
class directly and confirm the alias re-exported from `mlx_worker` is
the same symbol — both paths must see identical detection behaviour.
"""

import unittest

from backend_service.runaway_guard import RunawayGuard
from backend_service.mlx_worker import RunawayGuard as MlxAliasRunawayGuard


class SharedRunawayGuardTests(unittest.TestCase):
    def test_mlx_alias_is_same_class(self):
        # The mlx_worker shim must re-export the real class so existing
        # tests / callers don't see a divergent implementation.
        self.assertIs(RunawayGuard, MlxAliasRunawayGuard)

    def test_detects_repeated_lines(self):
        guard = RunawayGuard(min_line_length=20, max_repeats=3)
        with self.assertRaises(RuntimeError) as ctx:
            for _ in range(5):
                guard.feed("Wait, I will write 'Qwen3.5'. Let me try again.\n")
        self.assertIn("repeating itself", str(ctx.exception))

    def test_allows_normal_output(self):
        guard = RunawayGuard()
        guard.feed("Hello! How can I help you today?\n")
        guard.feed("I'm an AI assistant.\n")
        guard.flush()  # No raise = pass

    def test_detects_reasoning_loop(self):
        guard = RunawayGuard(max_reasoning_lines=10)
        with self.assertRaises(RuntimeError) as ctx:
            guard.feed("Wait, I should check the constraint again and verify.\n")
            guard.feed("Okay, I will just say 'Hello! How can I help?'\n")
            guard.feed("Actually, looking closer at the instruction again.\n")
            guard.feed("Wait, I need to check if I should explain more.\n")
            guard.feed("Let me re-read the constraint one more time now.\n")
            guard.feed("Wait, I should check the constraint once more time.\n")
        self.assertIn("reasoning loop", str(ctx.exception))

    def test_short_lines_dont_trip_repeat_check(self):
        # The repeat detector ignores lines below `min_line_length` so
        # short tokens like "OK." don't false-positive.
        guard = RunawayGuard(min_line_length=30, max_repeats=3)
        for _ in range(10):
            guard.feed("OK.\n")
        guard.flush()


if __name__ == "__main__":
    unittest.main()
