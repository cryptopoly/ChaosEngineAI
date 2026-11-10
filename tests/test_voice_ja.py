"""Japanese readings for Chatterbox without pykakasi (GPL): backend_service/voice_ja.py."""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from backend_service import voice_ja


class _Feature:
    def __init__(self, kana: str | None = None, pron: str | None = None) -> None:
        self.kana, self.pron = kana, pron


class _Word:
    def __init__(self, surface: str, kana: str | None = None, pron: str | None = None) -> None:
        self.surface, self.feature = surface, _Feature(kana, pron)


def _tagger(words: list[_Word]):
    return lambda text: words


class KakasiTests(unittest.TestCase):
    def test_katakana_becomes_hiragana_and_other_text_stays(self) -> None:
        self.assertEqual(voice_ja.katakana_to_hiragana("ノボッタ カタカナ abc 灯台"), "のぼった かたかな abc 灯台")
        self.assertEqual(voice_ja.katakana_to_hiragana("ー"), "ー")  # the long-vowel mark is not shifted

    def test_each_word_gets_the_pykakasi_keys_and_a_hiragana_reading(self) -> None:
        words = [_Word("灯台", "トウダイ"), _Word("守", "モリ"), _Word("は", "ハ"), _Word("上っ", "ノボッ")]
        out = voice_ja.Kakasi(lambda: _tagger(words)).convert("灯台守は上っ")
        self.assertEqual([p["orig"] for p in out], ["灯台", "守", "は", "上っ"])
        self.assertEqual([p["hira"] for p in out], ["とうだい", "もり", "は", "のぼっ"])
        for phrase in out:
            self.assertEqual(set(phrase), {"orig", "hira", "kana", "hepburn"})

    def test_a_word_without_a_reading_keeps_its_text(self) -> None:
        out = voice_ja.Kakasi(lambda: _tagger([_Word("Wi-Fi", "*"), _Word("謎", None)])).convert("x")
        self.assertEqual([p["hira"] for p in out], ["Wi-Fi", "謎"])

    def test_the_pronunciation_field_is_the_fallback_reading(self) -> None:
        out = voice_ja.Kakasi(lambda: _tagger([_Word("今日", None, "キョー")])).convert("x")
        self.assertEqual(out[0]["hira"], "きょー")

    def test_the_tagger_is_built_once(self) -> None:
        factory = mock.Mock(return_value=_tagger([_Word("a", "ア")]))
        kakasi = voice_ja.Kakasi(factory)
        kakasi.convert("a")
        kakasi.convert("b")
        factory.assert_called_once()


class StandInTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = sys.modules.pop("pykakasi", None)

    def tearDown(self) -> None:
        sys.modules.pop("pykakasi", None)
        if self._saved is not None:
            sys.modules["pykakasi"] = self._saved

    def test_it_registers_as_pykakasi_when_the_real_package_is_missing(self) -> None:
        with mock.patch.object(voice_ja.importlib.util, "find_spec", return_value=None):
            self.assertTrue(voice_ja.install_pykakasi_stand_in())
        self.assertIsInstance(sys.modules["pykakasi"].kakasi(), voice_ja.Kakasi)
        with mock.patch.object(voice_ja.importlib.util, "find_spec", return_value=None):
            self.assertTrue(voice_ja.install_pykakasi_stand_in())  # a second call keeps it

    def test_the_real_package_is_left_alone(self) -> None:
        with mock.patch.object(voice_ja.importlib.util, "find_spec", return_value=object()):
            self.assertFalse(voice_ja.install_pykakasi_stand_in())
        self.assertNotIn("pykakasi", sys.modules)
        sys.modules["pykakasi"] = types.ModuleType("pykakasi")
        self.assertFalse(voice_ja.install_pykakasi_stand_in())

    def test_the_worker_installs_it_before_importing_chatterbox_v3(self) -> None:
        from pathlib import Path

        source = Path(voice_ja.__file__).with_name("voice_worker_torch.py").read_text(encoding="utf-8")
        self.assertLess(source.index("voice_ja.install_pykakasi_stand_in()"), source.index("from chatterbox.mtl_tts import"))


if __name__ == "__main__":
    unittest.main()
