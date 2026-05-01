"""Runaway-generation detection shared across MLX worker and llama.cpp paths.

Phase 2.0.5-F: the MLX worker has had a `RunawayGuard` for a while that
catches three failure modes — repeated identical lines, near-duplicate
reasoning loops, and raw thinking-heading dumps. The llama.cpp streaming
path didn't have an equivalent, so a runaway on a GGUF model could fill the
context buffer and pin the host until the user noticed.

Moved here so both backends can import the same implementation. The
`mlx_worker` module re-exports it for backward compatibility with existing
imports.
"""

from __future__ import annotations

import re

from backend_service.reasoning_split import RAW_REASONING_HEADING_RE


_RAW_THINKING_HEADING_RE = RAW_REASONING_HEADING_RE

_REASONING_LINE_RE = re.compile(
    r"^\s*(?:"
    r"wait,|okay[,.]|actually[,.]|let me|i (?:need to|should|will|must|can)"
    r"|so (?:i |the )|hmm|looking|check(?:ing)?|(?:re)?evaluat"
    r"|draft(?:ing)?|refin(?:ing|e)|final (?:check|answer|decision|polish)"
    r")",
    re.IGNORECASE,
)


class RunawayGuard:
    """Detect and abort runaway generation loops in streamed output.

    Catches three failure modes:
    1. Repeated identical lines (e.g. "Wait, I will write 'Qwen3.5'." x100)
    2. Near-duplicate reasoning loops (lines starting with "Wait," / "Okay," etc.)
    3. Raw thinking-heading dumps (e.g. "Thinking Process:" at generation start)

    Raises ``RuntimeError`` when a runaway is detected.
    """

    def __init__(
        self,
        *,
        min_line_length: int = 30,
        max_repeats: int = 4,
        max_reasoning_lines: int = 20,
    ) -> None:
        self._min_line_length = min_line_length
        self._max_repeats = max_repeats
        self._max_reasoning_lines = max_reasoning_lines
        self._buffer = ""
        self._last_line: str | None = None
        self._repeat_count = 0
        self._reasoning_streak = 0
        self._total_chars = 0
        self._thinking_heading_seen = False

    def feed(self, text: str) -> None:
        """Feed a chunk of streamed text. Raises on detected runaway."""
        self._total_chars += len(text)
        self._buffer += text

        # Check for raw thinking heading at the start of generation
        if not self._thinking_heading_seen and self._total_chars < 200:
            if _RAW_THINKING_HEADING_RE.search(self._buffer):
                self._thinking_heading_seen = True

        # Check for repeated / reasoning lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._check_line(line)

    def flush(self) -> None:
        if self._buffer:
            self._check_line(self._buffer)
            self._buffer = ""

    @property
    def saw_thinking_heading(self) -> bool:
        return self._thinking_heading_seen

    def _check_line(self, line: str) -> None:
        normalized = " ".join(line.strip().lower().split())
        if len(normalized) < self._min_line_length:
            # Short lines still decay the reasoning streak so alternating
            # "Wait, ..." / "31536000 seconds." patterns get caught.
            self._reasoning_streak = max(0, self._reasoning_streak - 1)
            return

        # Exact-match repetition
        if normalized == self._last_line:
            self._repeat_count += 1
        else:
            self._last_line = normalized
            self._repeat_count = 1

        if self._repeat_count >= self._max_repeats:
            raise RuntimeError(
                "Stopped runaway generation: model is repeating itself."
            )

        # Near-duplicate reasoning loop detection
        # Lines like "Wait, I should...", "Okay, I'll...", "Actually, looking..."
        # Non-reasoning lines decay the streak by 1 instead of resetting,
        # so alternating "Wait, ..." / "31536000 seconds." still trips the guard.
        if _REASONING_LINE_RE.match(normalized):
            self._reasoning_streak += 2
        else:
            self._reasoning_streak = max(0, self._reasoning_streak - 1)

        if self._reasoning_streak >= self._max_reasoning_lines:
            raise RuntimeError(
                "Stopped runaway generation: model is stuck in a reasoning loop."
            )
