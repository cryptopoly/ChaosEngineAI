/**
 * D-69: the i18n validator's strict coverage gate skips experimental
 * (community) locales but still checks their structure.
 *
 * Runs ``scripts/i18n-validate.mjs`` against a throwaway fixture tree
 * through ``I18N_LOCALES_DIR`` so the real catalogs are not involved.
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { spawnSync } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const SCRIPT = path.resolve(__dirname, "../../../scripts/i18n-validate.mjs");
const NAMESPACES = ["common", "chat", "studio", "library", "setup", "diagnostics", "runtime", "errors", "meta", "dashboard"];
const LOCALES = ["en", "zh-CN", "zh-TW", "ja", "de", "ru", "ko", "fr", "es", "pt-BR", "tlh"];

let root: string;

function writeTree(halfCoverageFor: string[]) {
  for (const locale of LOCALES) {
    mkdirSync(path.join(root, locale), { recursive: true });
    for (const ns of NAMESPACES) {
      const full = { a: "one", b: "two {n}", c: "three", d: "four" };
      const half = { a: "one", b: "two {n}" };
      const body = halfCoverageFor.includes(locale) ? half : full;
      writeFileSync(path.join(root, locale, `${ns}.json`), JSON.stringify(body));
    }
  }
}

function run(args: string[]) {
  return spawnSync(process.execPath, [SCRIPT, ...args], {
    env: { ...process.env, I18N_LOCALES_DIR: root },
    encoding: "utf-8",
  });
}

beforeAll(() => {
  root = mkdtempSync(path.join(tmpdir(), "i18n-gate-"));
});

afterAll(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("i18n-validate strict gate", () => {
  it("passes when only the experimental locale is below 95 %", () => {
    writeTree(["tlh"]);
    const res = run(["--strict"]);
    expect(res.stdout).toContain("tlh     50%");
    expect(res.stdout).toContain("experimental, outside the strict gate");
    expect(res.status).toBe(0);
  });

  it("still fails when a shipping locale is below 95 %", () => {
    writeTree(["tlh", "de"]);
    const res = run(["--strict"]);
    expect(res.status).toBe(2);
    expect(res.stderr).toContain("de: 50%");
    expect(res.stderr).not.toContain("tlh: 50%");
  });

  it("still reports an ICU error in the experimental locale", () => {
    writeTree([]);
    writeFileSync(path.join(root, "tlh", "common.json"), JSON.stringify({ a: "one", b: "{n, plural, other", c: "three", d: "four" }));
    const res = run([]);
    expect(res.status).toBe(1);
    expect(res.stderr).toContain("tlh:common.b");
  });
});
