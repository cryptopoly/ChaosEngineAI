"""Turn chat text into speakable sentences.

Read-aloud speaks a reply sentence by sentence so audio starts after the
first one is synthesized. Markdown, code and links are rewritten into what a
listener should hear first.
"""

from __future__ import annotations

import re

# Text one synthesis request may carry; longer replies are cut with a note.
MAX_SPEAK_CHARS = 20_000

_CODE_FENCE_RE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|$)", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_LIST_MARK_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_QUOTE_RE = re.compile(r"^\s*>\s?", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1")
_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{3,}.*$", re.MULTILINE)
_SPACES_RE = re.compile(r"[ \t]+")

# A sentence ends at . ! ? (or their CJK forms) followed by space, or at a
# paragraph break. Abbreviations like "e.g." stay attached to their sentence.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+|(?<=[。！？])\s*|\n{2,}")
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "Mr.", "Mrs.", "Ms.", "Dr.", "St.", "No.")


def clean_for_speech(text: str) -> str:
    """Rewrite markdown into plain spoken text."""
    text = _THINK_RE.sub(" ", text)
    text = _TOOL_CALL_RE.sub(" ", text)
    text = _CODE_FENCE_RE.sub("\n\n(code block omitted)\n\n", text)
    text = _IMAGE_RE.sub(lambda m: m.group(1) or "", text)
    text = _LINK_RE.sub(lambda m: m.group(1), text)
    text = _URL_RE.sub("link", text)
    text = _INLINE_CODE_RE.sub(lambda m: m.group(1), text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _TABLE_RULE_RE.sub("", text)
    text = text.replace("|", ", ")
    text = _HEADING_RE.sub("", text)
    text = _LIST_MARK_RE.sub("", text)
    text = _QUOTE_RE.sub("", text)
    for _ in range(2):  # nested emphasis
        text = _EMPHASIS_RE.sub(lambda m: m.group(2), text)
    # A list item or heading without end punctuation still ends a sentence.
    lines = []
    for line in text.split("\n"):
        stripped = _SPACES_RE.sub(" ", line).strip()
        if stripped and stripped[-1] not in ".!?:;,。！？":
            stripped += "."
        lines.append(stripped)
    return "\n".join(lines).strip()


def split_sentences(text: str, max_chars: int = 400) -> list[str]:
    """Split cleaned text into sentences no longer than ``max_chars``."""
    pieces: list[str] = []
    for chunk in _SENTENCE_END_RE.split(text):
        chunk = " ".join(chunk.split())
        if not chunk:
            continue
        if pieces and pieces[-1].endswith(_ABBREVIATIONS):
            pieces[-1] = f"{pieces[-1]} {chunk}"
            continue
        pieces.append(chunk)
    sentences: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            cut = piece.rfind(", ", 0, max_chars)
            if cut <= 0:
                cut = piece.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            sentences.append(piece[: cut + 1].strip())
            piece = piece[cut + 1 :].strip()
        if piece:
            sentences.append(piece)
    return sentences


def speech_sentences(text: str) -> list[str]:
    """Clean ``text`` and split it, capped at ``MAX_SPEAK_CHARS``."""
    cleaned = clean_for_speech(text[:MAX_SPEAK_CHARS])
    return split_sentences(cleaned)
