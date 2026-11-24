import { describe, expect, it } from "vitest";

import type { GenerationMetrics } from "../../types";
import {
  hasDecodeSpeed,
  modeCoveredByWarning,
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

describe("hasDecodeSpeed", () => {
  it("leaves out a speed the turn did not measure (tool-using turns report 0)", () => {
    expect(hasDecodeSpeed(0)).toBe(false);
    expect(hasDecodeSpeed(null)).toBe(false);
    expect(hasDecodeSpeed(undefined)).toBe(false);
    expect(hasDecodeSpeed(31.4)).toBe(true);
  });
});

describe("TensorFold start-up fallback", () => {
  const note =
    "Using python with MLX 0.32.4 and mlx-lm 0.32.2. DFLASH unavailable for 'Vontra/Qwen3.8-27B-MLX-4bit': no compatible draft model is registered. "
    + "TensorFold startup failed (--parallel 1 needs 34.1 GiB for its streams at a 262144-token context; 26.2 GiB remains); using standard MLX.";
  const metrics = makeMetrics({
    runtimeNote: note,
    speculativeDecoding: false,
    requestedSpeculativeDecoding: true,
    requestedTreeBudget: 0,
  });

  it("names TensorFold, not DFlash, as what did not run", () => {
    expect(resolvedSpeculativeMode(metrics)).toBe("TensorFold needs more memory at this window, ran standard MLX");
    expect(runtimeOutcomeWarning(metrics)).toBe("TensorFold needs more memory at this window, ran standard MLX");
  });

  it("says a start-up failure with no memory figure plainly", () => {
    const other = makeMetrics({
      runtimeNote: "TensorFold startup failed (the server exited); using standard MLX.",
      speculativeDecoding: false,
      requestedSpeculativeDecoding: true,
    });
    expect(runtimeOutcomeWarning(other)).toBe("TensorFold could not start, ran standard MLX");
  });

  it("does not print the same sentence twice in the summary line", () => {
    const mode = resolvedSpeculativeMode(metrics);
    const warning = runtimeOutcomeWarning(metrics);
    expect(modeCoveredByWarning(mode, warning)).toBe(true);
    const plain = makeMetrics({
      runtimeNote: "Using python with MLX 0.31.1. DFLASH unavailable for 'foo/bar': no compatible draft model is registered.",
      speculativeDecoding: false,
      requestedSpeculativeDecoding: true,
    });
    expect(modeCoveredByWarning(resolvedSpeculativeMode(plain), runtimeOutcomeWarning(plain))).toBe(true);
    expect(modeCoveredByWarning("DFlash", null)).toBe(false);
  });
});
