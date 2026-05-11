#!/usr/bin/env node
/**
 * FU-042 — i18n validator.
 *
 * Three independent checks, each emitting a non-zero exit code on
 * failure so CI can gate releases on the i18n surface:
 *
 *   1. **Parity** — every key in ``src/locales/en/<ns>.json`` exists
 *      in every other locale's ``<ns>.json``.  Per FU-042 §"Coverage
 *      Gate" the v1 policy is warn-only at <95 %; orchestrate via the
 *      ``--strict`` flag for the future block-on-merge upgrade.
 *
 *   2. **ICU syntax** — each leaf value is parsed by
 *      ``intl-messageformat`` so we catch malformed plurals like
 *      ``{n, plural, one {x} other}`` (missing brace) at build time
 *      rather than at runtime when a user actually hits the surface.
 *
 *   3. **Orphans** — keys present in *any* locale but missing from
 *      ``en/`` are flagged as definite mistakes (en is the source of
 *      truth; any other-locale extra is a translator error or a
 *      stale leftover).
 *
 * Exit codes:
 *   0   all checks pass
 *   1   structural error (malformed JSON, ICU parse fail, missing file)
 *   2   coverage below threshold (only when --strict is passed)
 */

import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const LOCALES = path.join(ROOT, "src", "locales");

const ARGS = new Set(process.argv.slice(2));
const STRICT = ARGS.has("--strict");
const MIN_COVERAGE = STRICT ? 95 : 0;

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
];

const LOCALES_LIST = [
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
];

// ``intl-messageformat`` is a transitive dep of ``i18next-icu``; load
// via createRequire so this CLI script works whether or not the user
// has run ``npm install`` yet (we fall back to a lighter regex check
// when the module isn't on disk).
const require = createRequire(import.meta.url);
let IntlMessageFormat = null;
try {
  ({ IntlMessageFormat } = require("intl-messageformat"));
} catch {
  // Fallback below.
}

function leafEntries(node, prefix = []) {
  if (typeof node === "string") return [[prefix.join("."), node]];
  if (!node || typeof node !== "object") return [];
  const acc = [];
  for (const [k, v] of Object.entries(node)) acc.push(...leafEntries(v, [...prefix, k]));
  return acc;
}

async function loadCatalog(locale) {
  const out = {};
  for (const ns of NAMESPACES) {
    const file = path.join(LOCALES, locale, `${ns}.json`);
    try {
      const raw = await fs.readFile(file, "utf-8");
      out[ns] = JSON.parse(raw);
    } catch (err) {
      throw new Error(`failed to read ${path.relative(ROOT, file)}: ${err.message}`);
    }
  }
  return out;
}

function validateIcu(value, locale, fullKey) {
  if (!/[{}]/.test(value)) return null;
  if (IntlMessageFormat) {
    try {
      new IntlMessageFormat(value, locale);
      return null;
    } catch (err) {
      return `ICU parse failed at ${locale}:${fullKey} — ${err.message}`;
    }
  }
  // Lightweight fallback when intl-messageformat isn't installed.
  // Balance-check the braces and reject obvious typos like a single
  // ``{`` with no closer.
  let depth = 0;
  for (const ch of value) {
    if (ch === "{") depth++;
    else if (ch === "}") depth--;
    if (depth < 0) return `Unbalanced '}' at ${locale}:${fullKey}`;
  }
  if (depth !== 0) return `Unbalanced '{' at ${locale}:${fullKey}`;
  return null;
}

async function main() {
  console.log(`[i18n-validate] mode = ${STRICT ? "strict (CI gate)" : "warn-only"}`);
  if (!IntlMessageFormat) {
    console.log(`[i18n-validate] intl-messageformat not installed — using regex fallback ICU check`);
  }

  const catalogs = {};
  for (const locale of LOCALES_LIST) {
    catalogs[locale] = await loadCatalog(locale);
  }

  const enEntries = {};
  for (const ns of NAMESPACES) {
    enEntries[ns] = new Map(leafEntries(catalogs.en[ns]));
  }

  const errors = [];
  const warnings = [];
  const coverage = {};

  // ICU + parity passes
  for (const locale of LOCALES_LIST) {
    let totalKeys = 0;
    let presentKeys = 0;
    for (const ns of NAMESPACES) {
      const localeMap = new Map(leafEntries(catalogs[locale][ns]));
      for (const [leaf, value] of localeMap) {
        const icuErr = validateIcu(value, locale, `${ns}.${leaf}`);
        if (icuErr) errors.push(icuErr);
      }
      // Orphans (key in locale but missing in en)
      if (locale !== "en") {
        for (const leaf of localeMap.keys()) {
          if (!enEntries[ns].has(leaf)) {
            warnings.push(`orphan ${locale}:${ns}.${leaf} — not present in en`);
          }
        }
      }
      // Coverage
      const enLeaves = enEntries[ns];
      totalKeys += enLeaves.size;
      for (const leaf of enLeaves.keys()) if (localeMap.has(leaf)) presentKeys++;
    }
    const pct = totalKeys === 0 ? 100 : Math.round((presentKeys / totalKeys) * 1000) / 10;
    coverage[locale] = { presentKeys, totalKeys, pct };
  }

  // Print coverage table
  console.log("[i18n-validate] coverage:");
  for (const locale of LOCALES_LIST) {
    const { presentKeys, totalKeys, pct } = coverage[locale];
    const bar = "█".repeat(Math.round(pct / 5));
    console.log(
      `  ${locale.padEnd(6)} ${pct.toString().padStart(5)}%  ${bar.padEnd(20)}  (${presentKeys}/${totalKeys})`,
    );
  }

  if (warnings.length) {
    console.log(`[i18n-validate] ${warnings.length} warning(s):`);
    for (const w of warnings) console.log(`  ! ${w}`);
  }

  if (errors.length) {
    console.error(`[i18n-validate] ${errors.length} error(s):`);
    for (const e of errors) console.error(`  ✗ ${e}`);
    process.exit(1);
  }

  // Coverage gate (strict only)
  const lowCoverage = Object.entries(coverage)
    .filter(([locale]) => locale !== "en")
    .filter(([, c]) => c.pct < MIN_COVERAGE);
  if (STRICT && lowCoverage.length) {
    console.error(`[i18n-validate] ${lowCoverage.length} locale(s) below ${MIN_COVERAGE}% threshold:`);
    for (const [locale, c] of lowCoverage) {
      console.error(`  ${locale}: ${c.pct}%`);
    }
    process.exit(2);
  }

  console.log("[i18n-validate] OK");
}

main().catch((err) => {
  console.error("[i18n-validate] failed:", err.message);
  process.exit(1);
});
