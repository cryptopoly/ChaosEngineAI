"""Japanese readings for Chatterbox Multilingual V3 on the NVIDIA lane, without a GPL package.

Chatterbox turns kanji into hiragana with ``pykakasi.kakasi().convert(text)`` before it tokenises Japanese. pykakasi is
GPL-3.0, so the speech GPU add-on leaves it out, and without it Chatterbox passed raw kanji to a tokenizer that cannot
read them: a Japanese sentence came out as unrelated syllables (RTX 4090 run, Whisper heard 三重は木の殻からする頃…).
This module answers the same call with ``fugashi`` (MIT) and the BSD-licensed ``unidic-lite`` dictionary instead.

``install_pykakasi_stand_in()`` registers it as the ``pykakasi`` module, only when the real one is not installed.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any, Callable


def katakana_to_hiragana(text: str) -> str:
    """Shift katakana (ァ-ヶ) to the matching hiragana; every other character stays."""
    return "".join(chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char for char in text)


class Kakasi:
    """The one method of pykakasi's ``kakasi()`` object that Chatterbox calls."""

    def __init__(self, tagger_factory: Callable[[], Any] | None = None) -> None:
        self._tagger_factory = tagger_factory
        self._tagger: Any = None

    def _tag(self, text: str) -> Any:
        if self._tagger is None:
            if self._tagger_factory is not None:
                self._tagger = self._tagger_factory()
            else:
                import fugashi  # noqa: PLC0415 - only the Japanese path needs it

                self._tagger = fugashi.Tagger()
        return self._tagger(text)

    def convert(self, text: str) -> list[dict[str, str]]:
        """``[{"orig", "hira", "kana", "hepburn"}, ...]`` per word, like pykakasi; unknown words keep their text."""
        phrases: list[dict[str, str]] = []
        for word in self._tag(text):
            surface = str(word.surface)
            feature = word.feature
            reading = getattr(feature, "kana", None) or getattr(feature, "pron", None)
            if not reading or reading == "*":
                reading = surface
            phrases.append({"orig": surface, "hira": katakana_to_hiragana(str(reading)),
                            "kana": str(reading), "hepburn": ""})
        return phrases


def kakasi() -> Kakasi:
    return Kakasi()


def install_pykakasi_stand_in() -> bool:
    """Register this module as ``pykakasi`` unless the real package is installed. True when the stand-in is in use."""
    if "pykakasi" in sys.modules:
        return getattr(sys.modules["pykakasi"], "__chaosengine_stand_in__", False)
    if importlib.util.find_spec("pykakasi") is not None:
        return False
    module = types.ModuleType("pykakasi")
    module.kakasi = kakasi  # type: ignore[attr-defined]
    module.__chaosengine_stand_in__ = True  # type: ignore[attr-defined]
    sys.modules["pykakasi"] = module
    return True
