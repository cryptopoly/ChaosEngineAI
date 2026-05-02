import { describe, expect, it } from "vitest";
import type { TokenLogprob } from "../../types";
import { computeStats, lowConfidenceEntries } from "../LogprobSummary";

function entry(token: string, logprob: number, alts: Array<[string, number]> = []): TokenLogprob {
  return {
    token,
    logprob,
    alternatives: alts.map(([t, lp]) => ({ token: t, logprob: lp })),
  };
}

describe("computeStats", () => {
  it("returns zeros for empty input", () => {
    expect(computeStats([])).toEqual({ count: 0, avgLogprob: 0, lowConfidenceCount: 0 });
  });

  it("computes average across valid logprobs", () => {
    const stats = computeStats([entry("a", -0.5), entry("b", -1.5)]);
    expect(stats.count).toBe(2);
    expect(stats.avgLogprob).toBeCloseTo(-1.0);
  });

  it("flags entries with logprob below -3 as low confidence", () => {
    const stats = computeStats([
      entry("a", -0.1),
      entry("b", -3.5),
      entry("c", -10.0),
    ]);
    expect(stats.lowConfidenceCount).toBe(2);
  });

  it("ignores invalid logprob values in average", () => {
    const stats = computeStats([
      entry("a", -1.0),
      { token: "b", logprob: null, alternatives: [] },
    ]);
    expect(stats.count).toBe(2);
    expect(stats.avgLogprob).toBeCloseTo(-1.0);
  });
});

describe("lowConfidenceEntries", () => {
  it("returns only entries below -3", () => {
    const flagged = lowConfidenceEntries([
      entry("a", -0.1),
      entry("b", -3.5),
      entry("c", -1.0),
      entry("d", -8.0),
    ]);
    expect(flagged.map((e) => e.token)).toEqual(["b", "d"]);
  });

  it("caps result at 12 entries", () => {
    const many = Array.from({ length: 30 }, (_, i) => entry(`t${i}`, -5));
    expect(lowConfidenceEntries(many)).toHaveLength(12);
  });

  it("returns empty for entries with no flagged values", () => {
    expect(lowConfidenceEntries([entry("a", -0.5), entry("b", -1.0)])).toEqual([]);
  });
});
