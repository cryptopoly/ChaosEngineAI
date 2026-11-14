import { describe, expect, it } from "vitest";
import fixture from "../../../tests/fixtures/time_estimate_cases.json";
import { scaledSeconds, variantSeconds, type EstimateRef } from "../timeEstimate";

interface Case {
  name: string;
  kind: "image" | "video";
  engine?: string | null;
  catalog: number | null;
  ref: EstimateRef | null;
  request: { width: number; height: number; frames: number | null; steps: number | null };
  expected: number | null;
}

const cases = (fixture as { cases: Case[] }).cases;
const VREF: EstimateRef = { width: 768, height: 512, frames: 97, steps: 8 };

describe("timeEstimate shared cases (same file as tests/test_time_estimate.py)", () => {
  it("has the shared cases", () => {
    expect(cases.length).toBeGreaterThanOrEqual(10);
  });
  for (const c of cases) {
    it(c.name, () => {
      const got = variantSeconds(
        { estimatedGenerationSeconds: c.catalog, estimateRef: c.ref, engine: c.engine ?? null },
        c.kind,
        c.request,
      );
      if (c.expected === null) {
        expect(got).toBeNull();
      } else {
        expect(got).not.toBeNull();
        expect(Math.abs((got as number) - c.expected)).toBeLessThanOrEqual(1e-9 * Math.max(1, Math.abs(c.expected)));
      }
    });
  }
});

describe("scaledSeconds edges", () => {
  it("never throws on bad input and stays positive", () => {
    for (const args of [[0, 512], [512, 0], [null, null], ["a", "b"], [-5, 512]] as const) {
      const got = scaledSeconds(40, VREF, args[0], args[1], 97, 8);
      expect(got).toBe(40);
    }
    expect(scaledSeconds(Number.NaN, VREF, 512, 512)).toBeNull();
  });

  it("rises with size, frames and steps", () => {
    const series = [
      [256, 512, 768, 1024, 1536].map((s) => scaledSeconds(100, VREF, s, s, 97, 8) as number),
      [9, 25, 49, 97, 193, 257].map((f) => scaledSeconds(100, VREF, 768, 512, f, 8) as number),
      [1, 4, 8, 16, 50].map((s) => scaledSeconds(100, VREF, 768, 512, 97, s) as number),
    ];
    for (const values of series) {
      expect([...values].sort((a, b) => a - b)).toEqual(values);
      expect(values[0]).toBeLessThan(values[values.length - 1]);
    }
  });
});
