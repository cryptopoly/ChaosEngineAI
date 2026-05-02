import { describe, expect, it } from "vitest";
import type { GenerationMetrics } from "../../types";
import { buildChips } from "../SubstrateRoutingBadge";

function makeMetrics(overrides: Partial<GenerationMetrics> = {}): GenerationMetrics {
  return {
    finishReason: "stop",
    promptTokens: 10,
    completionTokens: 20,
    totalTokens: 30,
    tokS: 42.0,
    runtimeNote: null,
    ...overrides,
  };
}

describe("SubstrateRoutingBadge buildChips", () => {
  it("returns empty when no relevant fields are set", () => {
    expect(buildChips(makeMetrics())).toEqual([]);
  });

  it("emits engine + cache chips when present", () => {
    const chips = buildChips(makeMetrics({
      engineLabel: "MLX",
      cacheLabel: "ChaosEngine bf16",
    }));
    const labels = chips.map((c) => c.label);
    expect(labels).toContain("MLX");
    expect(labels).toContain("ChaosEngine bf16");
  });

  it("falls back to backend when engineLabel missing", () => {
    const chips = buildChips(makeMetrics({ backend: "llama.cpp" }));
    expect(chips[0].label).toBe("llama.cpp");
  });

  it("synthesises a cache label from strategy + bits when cacheLabel missing", () => {
    const chips = buildChips(makeMetrics({ cacheStrategy: "TurboQuant", cacheBits: 4 }));
    expect(chips.find((c) => c.key === "cache")?.label).toBe("TurboQuant 4-bit");
  });

  it("emits speculative-decoding chip with tree budget when on", () => {
    const chips = buildChips(makeMetrics({
      speculativeDecoding: true,
      treeBudget: 128,
    }));
    expect(chips.find((c) => c.key === "spec")?.label).toBe("DDTree 128");
  });

  it("emits accepted-rate chip alongside DDTree when set", () => {
    const chips = buildChips(makeMetrics({
      speculativeDecoding: true,
      treeBudget: 64,
      dflashAcceptanceRate: 4.5,
    }));
    expect(chips.find((c) => c.key === "accept")?.label).toBe("4.5 avg accepted");
  });

  it("omits acceptance chip when speculative decoding is off", () => {
    const chips = buildChips(makeMetrics({
      speculativeDecoding: false,
      dflashAcceptanceRate: 4.5,
    }));
    expect(chips.find((c) => c.key === "accept")).toBeUndefined();
  });

  it("emits warn chip with truncated runtime note", () => {
    const chips = buildChips(makeMetrics({
      runtimeNote: "x".repeat(80),
    }));
    const note = chips.find((c) => c.key === "note");
    expect(note?.tone).toBe("warn");
    expect(note?.label.length).toBeLessThanOrEqual(48);
    expect(note?.title.length).toBe(80);
  });

  it("preserves short runtime notes verbatim", () => {
    const chips = buildChips(makeMetrics({ runtimeNote: "fell back to native" }));
    expect(chips.find((c) => c.key === "note")?.label).toBe("fell back to native");
  });
});
