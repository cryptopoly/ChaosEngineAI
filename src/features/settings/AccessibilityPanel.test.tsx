import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { updateSettings } = vi.hoisted(() => ({ updateSettings: vi.fn(async (_p: unknown) => ({})) }));
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare function require(id: string): any;
const { readFileSync } = require("node:fs") as { readFileSync: (p: URL, enc: string) => string };
vi.mock("../../api", () => ({ updateSettings }));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_k: string, o?: { defaultValue?: string }) => o?.defaultValue ?? _k }),
}));

import { AccessibilityPanel, saveStrongBorders } from "./AccessibilityPanel";

describe("AccessibilityPanel", () => {
  beforeEach(() => updateSettings.mockClear());

  it("renders the switch on by default", () => {
    const html = renderToStaticMarkup(<AccessibilityPanel />);
    expect(html).toContain('role="switch"');
    expect(html).toContain('aria-checked="true"');
    expect(html).toContain("Stronger borders on fields and buttons");
  });

  it("toggling off saves strongControlBorders false, back on saves true", async () => {
    await saveStrongBorders(false);
    expect(updateSettings).toHaveBeenLastCalledWith({ strongControlBorders: false });
    await saveStrongBorders(true);
    expect(updateSettings).toHaveBeenLastCalledWith({ strongControlBorders: true });
  });
});

describe("--border-control token", () => {
  const css = readFileSync(new URL("../../styles.css", import.meta.url), "utf8");
  const root = css.slice(0, css.indexOf("\n}\n", css.indexOf("--app-bg")));
  const token = (name: string) => new RegExp(`${name}:\\s*(#[0-9a-fA-F]{6})`).exec(root)?.[1] as string;
  const lum = (hex: string) => {
    const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  };
  const ratio = (a: string, b: string) => {
    const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  };

  it.each(["--app-bg", "--surface", "--panel", "--panel-alt"])("keeps 3:1 against %s", (surface) => {
    expect(token("--border-control")).toBeTruthy();
    expect(ratio(token("--border-control"), token(surface))).toBeGreaterThanOrEqual(3);
  });

  it("has the off switch rule", () => {
    expect(css).toMatch(/:root\[data-strong-borders="off"\]\s*\{\s*--border-control:\s*var\(--border\)/);
  });
});
