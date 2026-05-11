"""FU-042 — unit tests for ``backend_service.i18n``.

Covers:

  * ``parse_accept_language`` — RFC-7231 parsing including q-values,
    quote-stripping, empty / malformed headers.
  * ``negotiate_locale`` — priority chain (override > accept-language
    > default), normalisation of variants (``zh-Hant`` → ``zh-TW``,
    bare ``zh`` → ``zh-CN``, ``pt`` → ``pt-BR``).
  * ``translator_for`` — returns a callable that defaults to the
    source string when the catalog is missing (matches the "ship en,
    follow-up fills" workflow from FU-042 §Q12).
"""

from __future__ import annotations

import unittest

from backend_service.i18n import (
    DEFAULT_LOCALE,
    SUPPORTED_LOCALES,
    negotiate_locale,
    ntranslator_for,
    parse_accept_language,
    translator_for,
)


class ParseAcceptLanguageTests(unittest.TestCase):
    def test_empty_header_returns_empty_list(self) -> None:
        self.assertEqual(parse_accept_language(None), [])
        self.assertEqual(parse_accept_language(""), [])

    def test_single_tag(self) -> None:
        self.assertEqual(parse_accept_language("en"), [("en", 1.0)])

    def test_multiple_tags_with_q_values(self) -> None:
        result = parse_accept_language("en-US,en;q=0.9,zh-CN;q=0.5")
        self.assertEqual(result[0], ("en-US", 1.0))
        self.assertAlmostEqual(result[1][1], 0.9)
        self.assertAlmostEqual(result[2][1], 0.5)

    def test_sorted_high_to_low_q(self) -> None:
        result = parse_accept_language("de;q=0.3,en;q=0.9,fr;q=0.6")
        qs = [q for _, q in result]
        self.assertEqual(qs, sorted(qs, reverse=True))

    def test_malformed_q_value_treated_as_zero(self) -> None:
        result = parse_accept_language("en;q=garbage,zh-CN;q=1.0")
        # zh-CN should sort first (q=1) since the malformed value falls to 0.
        self.assertEqual(result[0][0], "zh-CN")


class NegotiateLocaleTests(unittest.TestCase):
    def test_override_wins_over_header(self) -> None:
        self.assertEqual(negotiate_locale("zh-CN", override="ja"), "ja")

    def test_override_normalised(self) -> None:
        # ``zh-Hant`` should map to ``zh-TW`` via the normaliser.
        self.assertEqual(negotiate_locale(None, override="zh-Hant"), "zh-TW")
        self.assertEqual(negotiate_locale(None, override="zh"), "zh-CN")
        self.assertEqual(negotiate_locale(None, override="pt"), "pt-BR")
        self.assertEqual(negotiate_locale(None, override="pt-PT"), "pt-BR")

    def test_unknown_override_falls_through_to_header(self) -> None:
        self.assertEqual(negotiate_locale("ja", override="xx-XX"), "ja")

    def test_accept_language_first_supported_wins(self) -> None:
        # Tagalog isn't supported, German is — German wins.
        self.assertEqual(negotiate_locale("tl-PH,de;q=0.8"), "de")

    def test_default_when_nothing_matches(self) -> None:
        self.assertEqual(negotiate_locale(None), DEFAULT_LOCALE)
        self.assertEqual(negotiate_locale("xx-YY"), DEFAULT_LOCALE)

    def test_en_variants_normalize_to_en(self) -> None:
        for variant in ("en-US", "en-GB", "en-AU", "EN-IE"):
            self.assertEqual(negotiate_locale(variant), "en")

    def test_supported_locales_all_round_trip(self) -> None:
        # Every locale in our supported set normalises back to itself.
        for tag in SUPPORTED_LOCALES:
            self.assertEqual(negotiate_locale(None, override=tag), tag)


class TranslatorForTests(unittest.TestCase):
    def test_translator_returns_source_when_catalog_missing(self) -> None:
        # No `.mo` compiled for zh-CN yet — gettext call returns source.
        t = translator_for("zh-CN")
        self.assertEqual(t("Hello, world."), "Hello, world.")

    def test_translator_for_en_is_null_translations(self) -> None:
        # English path is the ``NullTranslations`` shortcut — explicit
        # contract that ``t(msg) == msg`` for en (the source language).
        t = translator_for("en")
        self.assertEqual(t("Save"), "Save")

    def test_ntranslator_returns_singular_or_plural(self) -> None:
        tn = ntranslator_for("en")
        self.assertEqual(tn("1 file", "%d files", 1), "1 file")
        self.assertEqual(tn("1 file", "%d files", 5), "%d files")


if __name__ == "__main__":
    unittest.main()
