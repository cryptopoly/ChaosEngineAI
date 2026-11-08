/**
 * D-69: the i18n validator's strict coverage gate skips experimental
 * (community) locales but still checks their structure.
 *
 * Runs ``scripts/i18n-validate.mjs`` against a throwaway fixture tree
 * through ``I18N_LOCALES_DIR`` so the real catalogs are not involved.
 */

// The app tsconfig has no Node type definitions (the UI never touches
// Node APIs); this test drives a Node CLI script, so it declares the few
// Node bits it uses itself and loads the modules at run time.
declare const process: { execPath: string; env: Record<string, string | undefined> };
declare function require(id: string): any;

import { describe, it, expect, beforeAll, afterAll } from "vitest";

interface RunResult { status: number | null; stdout: string; stderr: string }
const { spawnSync } = require("node:child_process") as {
  spawnSync: (cmd: string, args: string[], opts: { env: Record<string, string | undefined>; encoding: string }) => RunResult;
};
const { mkdtempSync, mkdirSync, rmSync, writeFileSync } = require("node:fs") as {
  mkdtempSync: (prefix: string) => string;
  mkdirSync: (dir: string, opts: { recursive: boolean }) => void;
  rmSync: (dir: string, opts: { recursive: boolean; force: boolean }) => void;
  writeFileSync: (file: string, data: string) => void;
};
const { tmpdir } = require("node:os") as { tmpdir: () => string };
const path = require("node:path") as { resolve: (...parts: string[]) => string; join: (...parts: string[]) => string };

const SCRIPT = path.resolve(new URL(".", import.meta.url).pathname, "../../../scripts/i18n-validate.mjs");
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
    expect(res.stdout).toMatch(/tlh\s+50%/);
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
