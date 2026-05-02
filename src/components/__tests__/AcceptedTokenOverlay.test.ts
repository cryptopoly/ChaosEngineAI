import { describe, expect, it } from "vitest";
import { computeSpanStats } from "../AcceptedTokenOverlay";

describe("computeSpanStats", () => {
  it("returns zeros for null / empty input", () => {
    expect(computeSpanStats(null)).toEqual({
      totalChars: 0,
      acceptedChars: 0,
      acceptedRatio: 0,
      spanCount: 0,
    });
    expect(computeSpanStats([])).toEqual({
      totalChars: 0,
      acceptedChars: 0,
      acceptedRatio: 0,
      spanCount: 0,
    });
  });

  it("sums total + accepted chars across spans", () => {
    const stats = computeSpanStats([
      { start: 0, length: 10, accepted: false },
      { start: 10, length: 30, accepted: true },
      { start: 40, length: 10, accepted: false },
    ]);
    expect(stats.totalChars).toBe(50);
    expect(stats.acceptedChars).toBe(30);
    expect(stats.acceptedRatio).toBeCloseTo(0.6);
    expect(stats.spanCount).toBe(3);
  });

  it("handles all-accepted runs", () => {
    const stats = computeSpanStats([{ start: 0, length: 100, accepted: true }]);
    expect(stats.acceptedRatio).toBeCloseTo(1.0);
  });

  it("handles all-rejected runs", () => {
    const stats = computeSpanStats([{ start: 0, length: 100, accepted: false }]);
    expect(stats.acceptedRatio).toBeCloseTo(0);
  });
});
