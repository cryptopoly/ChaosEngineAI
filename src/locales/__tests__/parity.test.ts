/**
 * FU-042 — locale parity unit test.
 *
 * Loads every ``src/locales/<lang>/<ns>.json`` and asserts that the
 * leaf-key set matches the ``en`` source-of-truth catalog.  Catches
 * accidentally-deleted keys, typos in JSON paths, and translator
 * edits that drop a required key.
 *
 * Pulled into ``npm test`` automatically via vitest's glob discovery
 * — no separate CI hook needed (``i18n-validate.mjs`` covers the
 * extract-time gate; this is the runtime-import gate).
 */

import { describe, it, expect } from "vitest";

const NAMESPACES = [
  "common",
  "chat",
  "studio",
  "library",
  "setup",
  "diagnostics",
  "runtime",
  "errors",
  "meta",
  "dashboard",
] as const;

const LOCALES = [
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
] as const;

function leafPaths(node: unknown, prefix: string[] = []): string[] {
  if (typeof node === "string") return [prefix.join(".")];
  if (Array.isArray(node)) {
    // `meta.supportedLocales` is an array — treat as a single leaf.
    return prefix.length ? [prefix.join(".")] : [];
  }
  if (node && typeof node === "object") {
    const out: string[] = [];
    for (const [key, value] of Object.entries(node)) {
      out.push(...leafPaths(value, [...prefix, key]));
    }
    return out;
  }
  return [];
}

async function loadCatalog(locale: string, namespace: string): Promise<unknown> {
  const mod = await import(/* @vite-ignore */ `../${locale}/${namespace}.json`);
  return mod.default ?? mod;
}

describe("locale catalog parity", () => {
  for (const ns of NAMESPACES) {
    describe(`namespace: ${ns}`, () => {
      it(`every locale matches the en key set for ${ns}`, async () => {
        const enCatalog = await loadCatalog("en", ns);
        const enKeys = new Set(leafPaths(enCatalog));
        for (const locale of LOCALES) {
          if (locale === "en") continue;
          const catalog = await loadCatalog(locale, ns);
          const keys = new Set(leafPaths(catalog));
          const missing = [...enKeys].filter((k) => !keys.has(k));
          const extra = [...keys].filter((k) => !enKeys.has(k));
          expect({ locale, ns, missing, extra }).toEqual({
            locale,
            ns,
            missing: [],
            extra: [],
          });
        }
      });
    });
  }
});
