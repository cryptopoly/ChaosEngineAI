import { describe, expect, it } from "vitest";
import { UI_SCALES, parseUiScale } from "../useUiScale";

describe("UI scale", () => {
  it("offers sizes above 100 % for a shared screen", () => {
    expect(UI_SCALES).toContain(1.1);
    expect(UI_SCALES).toContain(1.25);
  });

  it("keeps a stored large size and falls back to 90 % for anything unknown", () => {
    expect(parseUiScale("1.25")).toBe(1.25);
    expect(parseUiScale("1.1")).toBe(1.1);
    expect(parseUiScale("2")).toBe(0.9);
    expect(parseUiScale(null)).toBe(0.9);
  });
});
