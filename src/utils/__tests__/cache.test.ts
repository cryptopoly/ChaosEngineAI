import { describe, expect, it } from "vitest";
import {
  parseContextK,
  estimateArchFromParams,
  estimateParamsBFromDisk,
  detectBitsPerWeight,
  compareOptionalNumber,
} from "../cache";

describe("parseContextK()", () => {
  it("parses K suffix", () => {
    expect(parseContextK("8K")).toBe(8);
    expect(parseContextK("32K")).toBe(32);
  });

  it("parses M suffix", () => {
    expect(parseContextK("1M")).toBe(1000);
    expect(parseContextK("0.5M")).toBe(500);
  });

  it("treats bare numbers as tokens and converts to K", () => {
    expect(parseContextK("8192")).toBeCloseTo(8);
    expect(parseContextK("32768")).toBeCloseTo(32);
  });

  it("returns 0 for null/undefined", () => {
    expect(parseContextK(null)).toBe(0);
    expect(parseContextK(undefined)).toBe(0);
    expect(parseContextK("")).toBe(0);
  });
});

describe("estimateArchFromParams()", () => {
  it("returns plausible arch for small models", () => {
    const arch = estimateArchFromParams(1);
    expect(arch.numLayers).toBe(22);
    expect(arch.hiddenSize).toBe(2048);
  });

  it("returns plausible arch for 8B models", () => {
    const arch = estimateArchFromParams(8);
    expect(arch.numLayers).toBe(32);
    expect(arch.hiddenSize).toBe(4096);
  });

  it("returns plausible arch for 70B+ models", () => {
    const arch = estimateArchFromParams(70);
    expect(arch.numLayers).toBe(80);
    expect(arch.hiddenSize).toBe(8192);
  });
});

describe("estimateParamsBFromDisk()", () => {
  it("estimates params from disk size and bits", () => {
    // 4GB at 4 bits/weight ≈ 8B params
    expect(estimateParamsBFromDisk(4, 4)).toBe(8);
  });

  it("returns 0 for missing inputs", () => {
    expect(estimateParamsBFromDisk(0, 4)).toBe(0);
    expect(estimateParamsBFromDisk(4, 0)).toBe(0);
  });
});

describe("detectBitsPerWeight()", () => {
  it("detects N-bit patterns", () => {
    expect(detectBitsPerWeight("Q4_K_M")).toBe(4.5);
    expect(detectBitsPerWeight("3-bit")).toBe(3.5);
    expect(detectBitsPerWeight("8bit")).toBe(8.5);
  });

  it("detects fp16/bf16", () => {
    expect(detectBitsPerWeight("bf16")).toBe(16);
    expect(detectBitsPerWeight("fp16")).toBe(16);
    expect(detectBitsPerWeight("float16")).toBe(16);
  });

  it("detects fp32", () => {
    expect(detectBitsPerWeight("fp32")).toBe(32);
    expect(detectBitsPerWeight("float32")).toBe(32);
  });

  it("defaults to 16 for unknown", () => {
    expect(detectBitsPerWeight("some-model-name")).toBe(16);
  });
});

describe("compareOptionalNumber()", () => {
  it("compares two known numbers ascending", () => {
    expect(compareOptionalNumber(1, 2, 1)).toBeLessThan(0);
    expect(compareOptionalNumber(2, 1, 1)).toBeGreaterThan(0);
    expect(compareOptionalNumber(5, 5, 1)).toBe(0);
  });

  it("compares two known numbers descending", () => {
    expect(compareOptionalNumber(1, 2, -1)).toBeGreaterThan(0);
    expect(compareOptionalNumber(2, 1, -1)).toBeLessThan(0);
  });

  it("sorts nulls after known values", () => {
    expect(compareOptionalNumber(5, null, 1)).toBe(-1);
    expect(compareOptionalNumber(null, 5, 1)).toBe(1);
  });

  it("treats two nulls as equal", () => {
    expect(compareOptionalNumber(null, null, 1)).toBe(0);
    expect(compareOptionalNumber(undefined, undefined, 1)).toBe(0);
  });
});
