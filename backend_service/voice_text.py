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


# ---------------------------------------------------------------------------
# Sound tags (D-70): people write ``[laugh]``; each model gets what it understands
# ---------------------------------------------------------------------------

# Every tag name any speech model knows. Only these are treated as tags; other bracketed text ("[1]", "[sic]")
# is left to be read as it is.
KNOWN_TAGS = frozenset({
    "laugh", "chuckle", "sigh", "gasp", "cough", "clear throat", "sniff", "sniffle", "groan", "shush", "yawn",
    "angry", "fear", "surprised", "whispering", "advertisement", "dramatic", "narration", "crying", "happy",
    "sarcastic",
})
# Orpheus spells some tags its own way.
_ANGLE_NAMES = {"sniff": "sniffle"}
_TAG_RE = re.compile(r"\[\s*([A-Za-z][A-Za-z ]{0,24}?)\s*\]")
_TAG_ONLY_RE = re.compile(r"^(?:\s*\[\s*[A-Za-z][A-Za-z ]{0,24}?\s*\]\s*)+[.!?,;:]*\s*$")


def _tag_name(match: re.Match[str]) -> str:
    return " ".join(match.group(1).lower().split())


def apply_tags(sentence: str, allowed: frozenset[str] | set[str], style: str) -> tuple[str, bool]:
    """Rewrite the sound tags in one cleaned sentence for a model; returns ``(text, stripped_any)``.

    ``style`` "keep" leaves ``[laugh]`` for a model that reads it (Chatterbox-Turbo), "angle" writes ``<laugh>``
    (Orpheus; run after ``clean_for_speech`` so its HTML rule never sees them), "strip" removes them. A tag the
    model does not know (not in ``allowed``) is removed in every style, so no voice reads "laugh" aloud.
    """
    stripped = False

    def replace(match: re.Match[str]) -> str:
        nonlocal stripped
        name = _tag_name(match)
        if name not in KNOWN_TAGS:
            return match.group(0)
        if style == "angle":
            name = _ANGLE_NAMES.get(name, name)
        if style == "strip" or name not in allowed:
            stripped = True
            return " "
        return f"<{name}>" if style == "angle" else f"[{name}]"

    text = _TAG_RE.sub(replace, sentence)
    text = _SPACES_RE.sub(" ", text).strip()
    # "Hi [laugh] ." -> "Hi ." -> "Hi."
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    return text, stripped


def keep_tags_with_sentences(sentences: list[str]) -> list[str]:
    """A piece that is only tags ("[laugh].") joins the sentence before it (or the next when it is first)."""
    out: list[str] = []
    pending = ""
    for sentence in sentences:
        if _TAG_ONLY_RE.match(sentence) and any(_tag_name(m) in KNOWN_TAGS for m in _TAG_RE.finditer(sentence)):
            tags = sentence.rstrip(".!?,;: ")
            if out:
                out[-1] = f"{out[-1]} {tags}"
            else:
                pending = f"{pending} {tags}".strip()
            continue
        out.append(f"{pending} {sentence}".strip() if pending else sentence)
        pending = ""
    if pending:
        out.append(pending)
    return out


def guard_leading_paren(sentence: str) -> str:
    """VoxCPM2 reads text that starts with "(" as a voice description; drop that bracket pair so it is spoken."""
    stripped = sentence.lstrip()
    if not stripped.startswith("("):
        return sentence
    close = stripped.find(")")
    inner = stripped[1:close] if close > 0 else stripped[1:]
    rest = stripped[close + 1:] if close > 0 else ""
    return " ".join(f"{inner}{rest}".split())


def speech_sentences(text: str) -> list[str]:
    """Clean ``text`` and split it, capped at ``MAX_SPEAK_CHARS``."""
    cleaned = clean_for_speech(text[:MAX_SPEAK_CHARS])
    return split_sentences(cleaned)
