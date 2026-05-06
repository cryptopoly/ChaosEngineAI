"""Phase 3.1 follow-up tests for DDTree accepted-span building.

The full DDTree generation loop pulls in MLX + dflash_mlx which can't
be exercised in CI; these tests exercise the run-length-encoding
logic in isolation by constructing the same shape of input the loop
produces and verifying the output.

Run-length encoding rules:
- Each per-token entry is (token_text, accepted: bool)
- Consecutive entries with the same `accepted` bool collapse into one
  span with `start` = char offset, `length` = char count, `accepted`
- First token is always verifier-decoded (False) — it's the prefill
  posterior decode
"""

from __future__ import annotations

import unittest


def build_spans(per_token_text: list[str], per_token_accepted: list[bool]) -> list[dict]:
    """Mirror of the inline RLE logic in ddtree.generate_ddtree_mlx.

    Extracted into a helper for testability — the production loop
    keeps the inline copy because it lives inside a hot path with
    other state to thread.
    """
    if not per_token_accepted or not per_token_text:
        return []
    limit = min(len(per_token_text), len(per_token_accepted))
    text = per_token_text[:limit]
    accepted = per_token_accepted[:limit]
    spans: list[dict] = []
    offset = 0
    run_start = 0
    run_kind = accepted[0]
    for idx, is_accepted in enumerate(accepted):
        if is_accepted != run_kind:
            spans.append({
                "start": run_start,
                "length": offset - run_start,
                "accepted": run_kind,
            })
            run_start = offset
            run_kind = is_accepted
        offset += len(text[idx])
    spans.append({
        "start": run_start,
        "length": offset - run_start,
        "accepted": run_kind,
    })
    return spans


class DDTreeSpanBuildTests(unittest.TestCase):
    def test_empty_input_returns_empty_spans(self):
        self.assertEqual(build_spans([], []), [])

    def test_single_verifier_token(self):
        spans = build_spans(["Hello"], [False])
        self.assertEqual(spans, [{"start": 0, "length": 5, "accepted": False}])

    def test_pure_draft_run(self):
        spans = build_spans(["a", "b", "c"], [True, True, True])
        self.assertEqual(spans, [{"start": 0, "length": 3, "accepted": True}])

    def test_alternating_runs(self):
        # Cycle pattern: verifier, then 2 draft, then verifier, then 1 draft.
        spans = build_spans(
            [" The", " quick", " brown", " fox", " jumps"],
            [False, True, True, False, True],
        )
        self.assertEqual(spans, [
            {"start": 0, "length": 4, "accepted": False},  # " The"
            {"start": 4, "length": 12, "accepted": True},  # " quick brown"
            {"start": 16, "length": 4, "accepted": False},  # " fox"
            {"start": 20, "length": 6, "accepted": True},  # " jumps"
        ])

    def test_typical_dflash_cycle(self):
        # Realistic cycle structure: prefill verifier, then a cycle of
        # 3 draft + 1 verifier, then another cycle of 2 draft + 1 verifier.
        spans = build_spans(
            ["Hi", " how", " are", " you", " today", "?", " I", " am", " well"],
            [False, True, True, True, False, True, True, False, False],
        )
        # Run breakdown:
        # idx 0: F                 → run F (Hi, len 2)
        # idx 1-3: T T T           → run T (" how are you", len 12)
        # idx 4: F                 → run F (" today", len 6)
        # idx 5-6: T T             → run T ("? I", len 3)
        # idx 7-8: F F             → run F (" am well", len 8)
        self.assertEqual(spans, [
            {"start": 0, "length": 2, "accepted": False},
            {"start": 2, "length": 12, "accepted": True},
            {"start": 14, "length": 6, "accepted": False},
            {"start": 20, "length": 3, "accepted": True},
            {"start": 23, "length": 8, "accepted": False},
        ])

    def test_handles_length_drift(self):
        # When per_token_text and per_token_accepted disagree on length
        # (defensive — shouldn't happen in production), align to the
        # shorter list.
        spans = build_spans(["a", "b", "c"], [True, True])
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["length"], 2)


class DDTreeSpanInvariantTests(unittest.TestCase):
    """Properties that should hold for any well-formed accepted span list."""

    def test_spans_cover_full_text(self):
        text_tokens = ["Lorem", " ipsum", " dolor"]
        accepted = [False, True, False]
        spans = build_spans(text_tokens, accepted)
        total_len = sum(s["length"] for s in spans)
        self.assertEqual(total_len, sum(len(t) for t in text_tokens))

    def test_spans_are_contiguous(self):
        text_tokens = ["foo", "bar", "baz", "qux"]
        accepted = [False, True, True, False]
        spans = build_spans(text_tokens, accepted)
        cursor = 0
        for span in spans:
            self.assertEqual(span["start"], cursor)
            cursor += span["length"]


if __name__ == "__main__":
    unittest.main()
