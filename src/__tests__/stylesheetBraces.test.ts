/**
 * Guards the stylesheets against a merge that leaves a rule open. A missing "}" does not fail the build: the
 * browser reads every later rule as nested inside the open one, so whole features (take compare, the sample
 * card) silently lose their layout.
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

function cssFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return name === "node_modules" ? [] : cssFiles(path);
    return name.endsWith(".css") ? [path] : [];
  });
}

/** The first line where a top-level rule starts while another rule is still open, or null. */
function firstUnclosed(source: string): number | null {
  const text = source.replace(/\/\*[\s\S]*?\*\//g, (comment) => comment.replace(/[^\n]/g, " "));
  let depth = 0;
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    // A selector at column 0 opens a top-level rule; inside an at-rule block, rules are indented.
    if (depth > 0 && /^[.#a-zA-Z[:*@]/.test(line) && line.includes("{")) return i + 1;
    for (const ch of line) {
      if (ch === "{") depth += 1;
      else if (ch === "}") depth -= 1;
    }
  }
  return depth === 0 ? null : lines.length;
}

describe("stylesheets", () => {
  const files = cssFiles(join(__dirname, ".."));
  it("finds the stylesheets", () => {
    expect(files.some((f) => f.endsWith("styles.css"))).toBe(true);
  });
  it.each(files)("%s closes every rule before the next top-level rule", (file) => {
    expect(firstUnclosed(readFileSync(file, "utf8"))).toBeNull();
  });
});
