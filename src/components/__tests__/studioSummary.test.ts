import { describe, expect, it } from "vitest";
import { runEstimateBasis, summaryLine } from "../studioSummary";
import type { RecommendationRow } from "../recommendedSettings";

const row = (basis: "measured" | "estimated"): RecommendationRow => ({
  id: "steps",
  keys: ["steps"],
  values: { steps: 4 },
  reason: null,
  basis,
  inUse: false,
});

describe("summaryLine", () => {
  it("joins the parts that are set", () => {
    expect(summaryLine(["Steps 30", false, null, undefined, "", "Guidance 5"], "Model defaults")).toBe("Steps 30 · Guidance 5");
  });
  it("falls back when nothing is set", () => {
    expect(summaryLine([false, null], "Model defaults")).toBe("Model defaults");
    expect(summaryLine([])).toBeUndefined();
  });
});

describe("runEstimateBasis", () => {
  it("is measured when any row was timed here", () => {
    expect(runEstimateBasis([row("estimated"), row("measured")], false)).toBe("measured");
  });
  it("is estimated with rows or a time estimate", () => {
    expect(runEstimateBasis([row("estimated")], false)).toBe("estimated");
    expect(runEstimateBasis([], true)).toBe("estimated");
  });
  it("is none when nothing is known", () => {
    expect(runEstimateBasis([], false)).toBeNull();
  });
});
