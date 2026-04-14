import { describe, expect, it } from "vitest";

import type { GenerationMetrics } from "../../types";
import {
  resolvedSpeculativeMode,
  runtimeOutcomeWarning,
} from "./runtimeDetails";

function makeMetrics(overrides: Partial<GenerationMetrics> = {}): GenerationMetrics {
  return {
    finishReason: "stop",
    promptTokens: 1,
    completionTokens: 1,
    totalTokens: 2,
    tokS: 1,
    runtimeNote: "Using python with MLX 0.31.1 and mlx-lm 0.31.2.",
    ...overrides,
  };
}

describe("resolvedSpeculativeMode()", () => {
  it("detects the current no-compatible-draft wording", () => {
    const metrics = makeMetrics({
      runtimeNote: "Using python with MLX 0.31.1 and mlx-lm 0.31.2. DFLASH unavailable for 'foo/bar': no compatible draft model is registered.",
      speculativeDecoding: false,
    });

    expect(resolvedSpeculativeMode(metrics)).toBe("Requested, no compatible draft");
  });
});

describe("runtimeOutcomeWarning()", () => {
  it("shows requested DDTree when no compatible draft exists", () => {
    const metrics = makeMetrics({
      runtimeNote: "Using python with MLX 0.31.1 and mlx-lm 0.31.2. DFLASH unavailable for 'foo/bar': no compatible draft model is registered.",
      speculativeDecoding: false,
      requestedSpeculativeDecoding: true,
      requestedTreeBudget: 64,
    });

    expect(runtimeOutcomeWarning(metrics)).toBe("DDTree (64) requested, no compatible draft");
  });

  it("shows requested cache when execution fell back to native", () => {
    const metrics = makeMetrics({
      cacheLabel: "TurboQ 3-bit 4+4",
      cacheStrategy: "turboquant",
      cacheBits: 3,
      fp16Layers: 4,
      runtimeNote: "Using python with MLX 0.31.1 and mlx-lm 0.31.2. Cache strategy failed ('tuple' object has no attribute 'swapaxes'). Fell back to native f16 cache.",
      requestedCacheLabel: "TurboQ 3-bit 4+4",
      requestedCacheStrategy: "turboquant",
      requestedCacheBits: 3,
      requestedFp16Layers: 4,
    });

    expect(runtimeOutcomeWarning(metrics)).toBe("TurboQ 3-bit 4+4 requested, ran Native f16 cache");
  });
});
