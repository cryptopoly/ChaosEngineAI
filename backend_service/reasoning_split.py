from __future__ import annotations

import re
from dataclasses import dataclass


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_TAIL_GUARD = len(_THINK_OPEN) - 1
_STARTUP_BUFFER_LIMIT = 500

# Per-model-family overrides for reasoning delimiters. Keyed by canonical
# repo or family prefix (case-insensitive prefix match). Models that do not
# match any entry use the default `<think>...</think>` tags. Add new entries
# here when adopting models that emit a non-standard reasoning marker.
# Values are (open_tag, close_tag) pairs.
_REASONING_DELIMITER_REGISTRY: dict[str, tuple[str, str]] = {
    # Default registry left empty — DeepSeek R1, Qwen3, GPT-OSS all emit
    # `<think>...</think>` and need no override. Populate per-family entries
    # here when a future model uses a different convention.
}


def reasoning_delimiters_for(model_ref: str | None) -> tuple[str, str]:
    """Resolve the reasoning open/close tag pair for a given model reference.

    Looks up `model_ref` against `_REASONING_DELIMITER_REGISTRY` using a
    case-insensitive prefix match (so `Qwen/Qwen3-8B-Instruct` matches a
    registry key of `qwen/qwen3`). Returns the default `<think>`/`</think>`
    pair when no match is found.
    """
    if not model_ref:
        return (_THINK_OPEN, _THINK_CLOSE)
    lower = model_ref.lower()
    for key, tags in _REASONING_DELIMITER_REGISTRY.items():
        if lower.startswith(key.lower()):
            return tags
    return (_THINK_OPEN, _THINK_CLOSE)

_RAW_REASONING_LABELS = (
    "thinking process",
    "chain of thought",
    "internal reasoning",
    "scratchpad",
    "reasoning steps",
    "reasoning process",
    "mental sandbox",
    "confidence score",
    "final check",
    "self-check",
    "sanity check",
    "verification",
    "analysis",
    "current draft",
    "word count",
    "final polish",
    "final verification",
)
_RAW_REASONING_PREFIXES = (
    "thinking",
    "chain",
    "internal",
    "scratch",
    "reason",
    "mental",
    "confidence",
    "final",
    "self",
    "sanity",
    "verif",
    "anal",
)
_META_REASONING_LABELS = (
    "confidence score",
    "mental sandbox",
    "scratchpad",
    "analysis",
    "reasoning",
    "verification",
    "assumptions",
    "constraints",
    "checklist",
    "self-check",
    "sanity check",
    "current draft",
    "word count",
    "final polish",
    "final verification",
)

_RAW_REASONING_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:>+\s*)?(?:[-*+]\s*)?(?:\d+\.\s+)?\*{0,2}"
    r"(?:Thinking Process|Chain of Thought|Internal Reasoning|Scratchpad|Reasoning Steps|"
    r"Reasoning Process|Mental Sandbox|Confidence Score|Final Check|Self-Check|Sanity Check|"
    r"Verification|Analysis|Current Draft|Word Count|Final Polish|Final Verification)"
    r"\*{0,2}\s*:?(?:\s+.*)?$",
    re.IGNORECASE,
)
_NUMBERED_OUTLINE_RE = re.compile(r"^\s*\d+\.\s+\S")
_BULLET_LINE_RE = re.compile(r"^\s{0,6}(?:[-*+]|•)\s+\S")
_INDENTED_BULLET_RE = re.compile(r"^\s{2,}(?:[-*+]|•|\d+\.)\s+\S")
_STEP_LINE_RE = re.compile(r"^\s*(?:step|phase|pass)\s*\d+\s*[:.-]\s+\S", re.IGNORECASE)
_REASONING_OPENER_RE = re.compile(
    r"^\s*(?:wait,|okay[,.]|actually[,.]|let me|let's|i (?:need to|should|will|must|can)"
    r"|so (?:i |the )|hmm|draft(?:ing)?|refin(?:e|ing)|double-check(?:ing)?|check(?:ing)?)",
    re.IGNORECASE,
)
_REASONING_ACTION_RE = re.compile(
    r"^\s*(?:analy[sz]e|evaluate|determine|draft(?:ing)?|refin(?:e|ing)|check|verify|review|inspect|"
    r"consider|compare|plan|brainstorm|outline|explore|test|validate|summari[sz]e)\b",
    re.IGNORECASE,
)
_META_REASONING_RE = re.compile(
    r"^\s*(?:[-*+]\s*)?(?:\d+\.\s+)?\*{0,2}"
    r"(?:Confidence Score|Mental Sandbox|Scratchpad|Analysis|Reasoning|Verification|"
    r"Assumptions|Constraints|Checklist|Self-Check|Sanity Check|Current Draft|"
    r"Word Count|Final Polish|Final Verification)"
    r"\*{0,2}\s*:\s*.*$",
    re.IGNORECASE,
)
RAW_REASONING_HEADING_RE = _RAW_REASONING_HEADING_RE


@dataclass
class ThinkingStreamResult:
    text: str = ""
    reasoning: str = ""
    reasoning_done: bool = False


def _find_tag(buffer: str, tag: str) -> int:
    return buffer.lower().find(tag)


def _looks_like_reasoning_start_prefix(buffer: str) -> bool:
    stripped = buffer.lstrip().lower()
    if not stripped:
        return False
    if any(label.startswith(stripped) for label in _RAW_REASONING_LABELS):
        return True
    if any(stripped.startswith(prefix) for prefix in _RAW_REASONING_PREFIXES):
        return True
    if stripped[:1] in {"-", "*", "+", "•"}:
        return True
    if re.match(r"^\d{1,3}(?:[.)]?\s*)?$", stripped):
        return True
    return False


def _is_reasoning_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _RAW_REASONING_HEADING_RE.match(line):
        return True
    if _META_REASONING_RE.match(line):
        return True
    if _NUMBERED_OUTLINE_RE.match(line):
        return True
    if _BULLET_LINE_RE.match(line) or _INDENTED_BULLET_RE.match(line):
        return True
    if _STEP_LINE_RE.match(line):
        return True
    if _REASONING_OPENER_RE.match(line):
        return True
    lowered = stripped.lower()
    return any(lowered.startswith(f"{label}:") for label in _META_REASONING_LABELS)


def _looks_like_raw_thinking_start(line: str) -> bool:
    return _is_reasoning_line(line)


def _looks_like_final_answer_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _is_reasoning_line(line):
        return False
    if _REASONING_ACTION_RE.match(line):
        return False
    if len(stripped) <= 5:
        return False
    if stripped.endswith((".", "!", "?")):
        return True
    lowered = stripped.lower()
    return lowered.startswith(
        (
            "answer",
            "final",
            "here is",
            "here's",
            "hello",
            "hi",
            "sure",
            "yes",
            "no",
            "the ",
            "this ",
            "that ",
            "it ",
            "you ",
            "i ",
        )
    )


def strip_thinking_tokens(text: str) -> str:
    splitter = ThinkingTokenFilter()
    result = splitter.feed(text)
    flushed = splitter.flush()
    return f"{result.text}{flushed.text}".strip()


class ThinkingTokenFilter:
    """Split streamed reasoning from final text for both XML and raw dumps.

    Parameters
    ----------
    detect_raw_reasoning:
        When ``True`` (default), headings like "Thinking Process:" and
        bullet-style reasoning are split into the reasoning field even
        without ``<think>`` XML tags.  Set to ``False`` when thinking
        mode is "off" so that models which emit thinking-like content
        as plain text still have their output shown to the user.
        XML ``<think>`` tags are always processed regardless.
    """

    def __init__(
        self,
        *,
        detect_raw_reasoning: bool = True,
        open_tag: str = _THINK_OPEN,
        close_tag: str = _THINK_CLOSE,
    ) -> None:
        # `open_tag` / `close_tag` let downstream callers override the XML
        # delimiters per model family — see `reasoning_delimiters_for()`.
        # Defaults match the `<think>...</think>` convention used by Qwen3,
        # DeepSeek R1, GPT-OSS, and most other reasoning models.
        if not open_tag or not close_tag:
            raise ValueError("ThinkingTokenFilter requires non-empty open/close tags.")
        self._inside_xml_think = False
        self._inside_raw_think = False
        self._startup_done = False
        self._buffer = ""
        self._pending_raw_final = ""
        self._total_fed = 0
        self._detect_raw = detect_raw_reasoning
        self._open_tag = open_tag
        self._close_tag = close_tag
        self._tail_guard = max(0, len(open_tag) - 1)

    def feed(self, text: str) -> ThinkingStreamResult:
        self._buffer += text
        self._total_fed += len(text)
        output = ThinkingStreamResult()

        while True:
            if not self._startup_done and not self._inside_xml_think and not self._inside_raw_think:
                think_idx = _find_tag(self._buffer, self._open_tag)
                if think_idx != -1:
                    output.text += self._buffer[:think_idx]
                    self._buffer = self._buffer[think_idx + len(self._open_tag):]
                    self._inside_xml_think = True
                    self._startup_done = True
                    continue

                if self._detect_raw and "\n" in self._buffer:
                    first_line = self._buffer.split("\n", 1)[0]
                    if _looks_like_raw_thinking_start(first_line):
                        self._inside_raw_think = True
                        self._startup_done = True
                        continue
                    self._startup_done = True
                    continue

                if self._detect_raw and self._total_fed < _STARTUP_BUFFER_LIMIT and _looks_like_reasoning_start_prefix(self._buffer):
                    break
                self._startup_done = True
                continue

            if self._inside_raw_think:
                while "\n" in self._buffer:
                    line, rest = self._buffer.split("\n", 1)
                    segment = line + "\n"
                    if self._pending_raw_final:
                        if not line.strip() or _looks_like_final_answer_line(line):
                            self._pending_raw_final += segment
                            self._buffer = rest
                            continue
                        output.reasoning += self._pending_raw_final + segment
                        self._pending_raw_final = ""
                        self._buffer = rest
                        continue
                    if line.strip() and _looks_like_final_answer_line(line):
                        self._pending_raw_final = segment
                        self._buffer = rest
                        continue
                    output.reasoning += segment
                    self._buffer = rest
                break

            if self._inside_xml_think:
                end_idx = _find_tag(self._buffer, self._close_tag)
                if end_idx == -1:
                    output.reasoning += self._buffer
                    self._buffer = ""
                    break
                output.reasoning += self._buffer[:end_idx]
                self._buffer = self._buffer[end_idx + len(self._close_tag):]
                self._inside_xml_think = False
                output.reasoning_done = True
                continue

            start_idx = _find_tag(self._buffer, self._open_tag)
            if start_idx != -1:
                output.text += self._buffer[:start_idx]
                self._buffer = self._buffer[start_idx + len(self._open_tag):]
                self._inside_xml_think = True
                continue

            if len(self._buffer) > self._tail_guard:
                if self._tail_guard == 0:
                    output.text += self._buffer
                    self._buffer = ""
                else:
                    output.text += self._buffer[:-self._tail_guard]
                    self._buffer = self._buffer[-self._tail_guard:]
            break

        return output

    def flush(self) -> ThinkingStreamResult:
        output = ThinkingStreamResult()
        if self._inside_xml_think:
            output.reasoning = self._buffer
            output.reasoning_done = True
            self._inside_xml_think = False
            self._startup_done = True
            self._buffer = ""
            return output

        if self._inside_raw_think:
            tail = self._buffer
            if self._pending_raw_final:
                if not tail or not tail.strip() or _looks_like_final_answer_line(tail):
                    output.text = self._pending_raw_final + tail
                else:
                    output.reasoning = self._pending_raw_final + tail
                self._pending_raw_final = ""
            elif tail.strip() and _looks_like_final_answer_line(tail):
                output.text = tail
            else:
                output.reasoning = tail
            output.reasoning_done = True
            self._inside_raw_think = False
            self._startup_done = True
            self._buffer = ""
            return output

        if not self._startup_done and _looks_like_raw_thinking_start(self._buffer):
            output.reasoning = self._buffer
            output.reasoning_done = True
            self._startup_done = True
            self._buffer = ""
            return output

        output.text = self._buffer
        self._buffer = ""
        self._startup_done = True
        return output
