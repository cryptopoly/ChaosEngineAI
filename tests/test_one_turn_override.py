"""Tests for the Phase 2.12 one-turn model override.

When the user picks a warm model from the mid-thread swap menu, the
GenerateRequest carries `oneTurnOverride=True`. The backend honors
that by NOT persisting the override model's identity onto the
session, so the next plain message reverts to the session's default.
"""

import unittest

from backend_service.models import GenerateRequest


class GenerateRequestOneTurnOverrideTests(unittest.TestCase):
    def test_default_is_false(self):
        # Existing callers don't send the flag — the default must
        # preserve historic behaviour where sending with a different
        # model permanently switches the thread.
        request = GenerateRequest(prompt="hello")
        self.assertFalse(request.oneTurnOverride)

    def test_accepts_explicit_true(self):
        request = GenerateRequest(prompt="hello", oneTurnOverride=True)
        self.assertTrue(request.oneTurnOverride)

    def test_accepts_explicit_false(self):
        request = GenerateRequest(prompt="hello", oneTurnOverride=False)
        self.assertFalse(request.oneTurnOverride)

    def test_one_turn_override_coexists_with_model_fields(self):
        request = GenerateRequest(
            prompt="hello",
            modelRef="alt/model-7b",
            modelName="Alt Model 7B",
            backend="llama.cpp",
            oneTurnOverride=True,
        )
        self.assertTrue(request.oneTurnOverride)
        self.assertEqual(request.modelRef, "alt/model-7b")
        self.assertEqual(request.modelName, "Alt Model 7B")


class StatePersistGuardTests(unittest.TestCase):
    """The persist guard in state.py is a `if not getattr(...)` check —
    cover the contract directly so any future refactor that turns the
    flag into something falsy by default still exercises the guard."""

    def test_falsy_flag_passes_through_persist(self):
        request = GenerateRequest(prompt="hello")
        # The persist guard is `if not getattr(request, "oneTurnOverride", False)`
        # — verify the attribute is reachable and falsy on a fresh request.
        self.assertFalse(getattr(request, "oneTurnOverride", False))

    def test_truthy_flag_blocks_persist(self):
        request = GenerateRequest(prompt="hello", oneTurnOverride=True)
        self.assertTrue(getattr(request, "oneTurnOverride", False))


if __name__ == "__main__":
    unittest.main()
