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
import i18next from "i18next";
import IntlMessageFormat from "intl-messageformat";
import { EXPERIMENTAL_LOCALES, normaliseLocale, SUPPORTED_LOCALES } from "../index";

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
  it("contains the 10 shipping locales plus the experimental Klingon locale", () => {
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
      "tlh",
    ]);
  });

  it("marks tlh as experimental and nothing else", () => {
    expect([...EXPERIMENTAL_LOCALES]).toEqual(["tlh"]);
  });

  it("maps tlh and tlh-Latn to tlh", () => {
    expect(normaliseLocale("tlh")).toBe("tlh");
    expect(normaliseLocale("tlh-Latn")).toBe("tlh");
    expect(normaliseLocale("TLH")).toBe("tlh");
  });
});

describe("tlh locale (D-69)", () => {
  it("has no CLDR data in the runtime, so Intl falls back to English formatting", () => {
    // Klingon has no CLDR locale data; V8 resolves it to the default
    // (en-US).  Pinning this so a future runtime change that starts
    // returning root-locale formatting is noticed.
    const resolved = new Intl.NumberFormat("tlh").resolvedOptions().locale;
    expect(resolved.startsWith("en")).toBe(true);
    expect(new Intl.NumberFormat("tlh").format(1234.5)).toBe("1,234.5");
    expect(new Intl.PluralRules("tlh").select(1)).toBe("one");
    expect(new Intl.PluralRules("tlh").select(2)).toBe("other");
  });

  it("falls back to en for a key the tlh catalog does not have", async () => {
    const instance = i18next.createInstance();
    await instance.init({
      lng: "tlh",
      fallbackLng: "en",
      resources: {
        en: { common: { present: "Save", onlyInEnglish: "Only in English" } },
        tlh: { common: { present: "yIpol", onlyInEnglish: null as unknown as string } },
      },
      ns: ["common"],
      defaultNS: "common",
      returnEmptyString: false,
      returnNull: false,
      interpolation: { escapeValue: false },
    });
    expect(instance.t("present")).toBe("yIpol");
    expect(instance.t("onlyInEnglish")).toBe("Only in English");
  });

  it("every tlh catalog string compiles as an ICU message", async () => {
    const namespaces = ["common", "chat", "studio", "library", "setup", "diagnostics", "runtime", "errors", "meta", "dashboard"];
    const leaves = (node: unknown, prefix: string[] = []): Array<[string, string]> => {
      if (typeof node === "string") return [[prefix.join("."), node]];
      if (node && typeof node === "object" && !Array.isArray(node)) {
        return Object.entries(node).flatMap(([k, v]) => leaves(v, [...prefix, k]));
      }
      return [];
    };
    const failures: string[] = [];
    let checked = 0;
    for (const ns of namespaces) {
      const mod = await import(/* @vite-ignore */ `../../locales/tlh/${ns}.json`);
      for (const [key, value] of leaves(mod.default ?? mod)) {
        checked++;
        try {
          new IntlMessageFormat(value, "tlh", undefined, { ignoreTag: true });
        } catch (err) {
          failures.push(`${ns}.${key}: ${(err as Error).message}`);
        }
      }
    }
    expect(checked).toBeGreaterThan(100);
    expect(failures).toEqual([]);
  });

  it("every tlh plural has an 'other' branch and keeps the English placeholders", async () => {
    // Klingon nouns do not mark number, so a plural may collapse to a
    // single ``other`` branch; it must never lose a ``{name}`` the
    // call site passes in.
    const namespaces = ["common", "chat", "studio", "library", "setup", "diagnostics", "runtime", "errors", "dashboard"];
    const placeholders = (s: string) =>
      [...s.matchAll(/\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*[,}]/g)].map((m) => m[1]).sort();
    const leaves = (node: unknown, prefix: string[] = []): Array<[string, string]> => {
      if (typeof node === "string") return [[prefix.join("."), node]];
      if (node && typeof node === "object" && !Array.isArray(node)) {
        return Object.entries(node).flatMap(([k, v]) => leaves(v, [...prefix, k]));
      }
      return [];
    };
    const problems: string[] = [];
    for (const ns of namespaces) {
      const en = leaves((await import(/* @vite-ignore */ `../../locales/en/${ns}.json`)).default);
      const tlh = new Map(leaves((await import(/* @vite-ignore */ `../../locales/tlh/${ns}.json`)).default));
      for (const [key, enValue] of en) {
        const tlhValue = tlh.get(key);
        if (typeof tlhValue !== "string") continue;
        if (/plural/.test(tlhValue) && !/other\s*\{/.test(tlhValue)) problems.push(`${ns}.${key}: plural without other`);
        const missing = placeholders(enValue).filter((p) => !placeholders(tlhValue).includes(p));
        if (missing.length) problems.push(`${ns}.${key}: missing {${missing.join("}, {")}}`);
      }
    }
    expect(problems).toEqual([]);
  });
});
