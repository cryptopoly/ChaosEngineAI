import { describe, expect, it } from "vitest";
import type { GenerationMetrics } from "../../types";
import { buildChips, runtimeNoteIsWarning } from "../SubstrateRoutingBadge";

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

  it("truncates long runtime notes for the chip label but preserves full title", () => {
    const chips = buildChips(makeMetrics({
      runtimeNote: "Failed to load draft model: " + "x".repeat(80),
    }));
    const note = chips.find((c) => c.key === "note");
    expect(note?.tone).toBe("warn");
    expect(note?.label.length).toBeLessThanOrEqual(48);
    expect(note?.title.length).toBeGreaterThan(48);
  });

  it("preserves short runtime notes verbatim", () => {
    const chips = buildChips(makeMetrics({ runtimeNote: "fell back to native" }));
    expect(chips.find((c) => c.key === "note")?.label).toBe("fell back to native");
  });

  // FU-035: benign info notes ("Using python with MLX 0.31.x and mlx-lm
  // 0.31.y.") used to render with the orange warn tone. The tone now
  // reflects whether the note actually flags a problem, so operators
  // notice the orange chip on real warnings instead of every turn.
  it("uses default tone for benign version-info notes", () => {
    const chips = buildChips(makeMetrics({
      runtimeNote: "Using python with MLX 0.31.2 and mlx-lm 0.31.3.",
    }));
    expect(chips.find((c) => c.key === "note")?.tone).toBe("default");
  });

  it("uses warn tone when a benign prefix is followed by a fault clause", () => {
    const chips = buildChips(makeMetrics({
      runtimeNote: "Using python with MLX 0.31.2 and mlx-lm 0.31.3. DFLASH unavailable for 'foo/bar': no compatible draft model is registered.",
    }));
    expect(chips.find((c) => c.key === "note")?.tone).toBe("warn");
  });

  it("uses warn tone when the cache strategy fell back", () => {
    const chips = buildChips(makeMetrics({
      runtimeNote: "Using python with MLX 0.31.2 and mlx-lm 0.31.3. Cache strategy failed ('tuple'). Fell back to native f16 cache.",
    }));
    expect(chips.find((c) => c.key === "note")?.tone).toBe("warn");
  });
});

describe("runtimeNoteIsWarning", () => {
  it("returns false for plain version banner", () => {
    expect(runtimeNoteIsWarning("Using python with MLX 0.31.2 and mlx-lm 0.31.3.")).toBe(false);
  });

  it("returns true when 'unavailable' appears", () => {
    expect(runtimeNoteIsWarning("DFLASH unavailable for 'foo/bar'.")).toBe(true);
  });

  it("returns true when 'fell back' appears", () => {
    expect(runtimeNoteIsWarning("Cache strategy failed. Fell back to native f16 cache.")).toBe(true);
  });

  it("returns true when 'failed' appears", () => {
    expect(runtimeNoteIsWarning("Cache strategy failed.")).toBe(true);
  });

  it("returns true when 'error' appears", () => {
    expect(runtimeNoteIsWarning("error loading draft model")).toBe(true);
  });

  it("is case-insensitive", () => {
    expect(runtimeNoteIsWarning("WARNING: cache fallback")).toBe(true);
  });

  it("returns false for empty string", () => {
    expect(runtimeNoteIsWarning("")).toBe(false);
  });
});
