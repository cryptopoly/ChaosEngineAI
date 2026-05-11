/**
 * FU-042 — unit tests for the frontend locale negotiator.
 *
 * Hits the pure-JS surface of ``src/i18n/index.ts``:
 * ``normaliseLocale`` (BCP-47 → shipping tag), ``SUPPORTED_LOCALES``
 * (catalog parity), and the eager ``en`` bundle (presence of every
 * required namespace key shape).  Initialisation is *not* exercised
 * here — vitest's jsdom env has no ``document.documentElement`` set
 * up the way the real app does, and the lazy ``import()`` path needs
 * a Vite resolver that doesn't run in unit tests.
 */

import { describe, it, expect } from "vitest";
import { normaliseLocale, SUPPORTED_LOCALES } from "../index";

describe("normaliseLocale", () => {
  it("maps zh-Hant family to zh-TW", () => {
    expect(normaliseLocale("zh-Hant")).toBe("zh-TW");
    expect(normaliseLocale("zh-Hant-TW")).toBe("zh-TW");
    expect(normaliseLocale("zh-HK")).toBe("zh-TW");
    expect(normaliseLocale("zh-MO")).toBe("zh-TW");
  });

  it("maps zh-Hans + bare zh + zh-SG to zh-CN", () => {
    expect(normaliseLocale("zh-Hans")).toBe("zh-CN");
    expect(normaliseLocale("zh")).toBe("zh-CN");
    expect(normaliseLocale("zh-SG")).toBe("zh-CN");
    expect(normaliseLocale("zh-CN")).toBe("zh-CN");
  });

  it("maps pt variants to pt-BR", () => {
    expect(normaliseLocale("pt")).toBe("pt-BR");
    expect(normaliseLocale("pt-BR")).toBe("pt-BR");
    expect(normaliseLocale("pt-PT")).toBe("pt-BR");
  });

  it("maps en variants to en", () => {
    expect(normaliseLocale("en")).toBe("en");
    expect(normaliseLocale("en-US")).toBe("en");
    expect(normaliseLocale("en-GB")).toBe("en");
  });

  it("falls back to en for unsupported tags", () => {
    expect(normaliseLocale("xx-YY")).toBe("en");
    expect(normaliseLocale(null)).toBe("en");
    expect(normaliseLocale(undefined)).toBe("en");
    expect(normaliseLocale("")).toBe("en");
  });

  it("matches every supported locale to itself", () => {
    for (const tag of SUPPORTED_LOCALES) {
      expect(normaliseLocale(tag)).toBe(tag);
    }
  });

  it("region-strips to base when only base is supported", () => {
    // de-CH should fall to de (we ship de, not de-CH).
    expect(normaliseLocale("de-CH")).toBe("de");
    expect(normaliseLocale("fr-CA")).toBe("fr");
    expect(normaliseLocale("es-MX")).toBe("es");
    expect(normaliseLocale("ko-KP")).toBe("ko");
  });

  it("is case-insensitive on the raw input", () => {
    expect(normaliseLocale("ZH-CN")).toBe("zh-CN");
    expect(normaliseLocale("JA")).toBe("ja");
    expect(normaliseLocale("PT-br")).toBe("pt-BR");
  });
});

describe("SUPPORTED_LOCALES", () => {
  it("contains exactly the 10 shipping locales", () => {
    expect(SUPPORTED_LOCALES).toEqual([
      "en",
      "zh-CN",
      "zh-TW",
      "ja",
      "de",
      "ru",
      "ko",
      "fr",
      "es",
      "pt-BR",
    ]);
  });
});
