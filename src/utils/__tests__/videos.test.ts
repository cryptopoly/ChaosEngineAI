import { describe, expect, it } from "vitest";

import { assessVideoGenerationSafety } from "../videos";

// The safety heuristic now scales with device memory rather than a flat
// token threshold — a 64 GB M4 Max should tolerate far more frames than a
// 16 GB base M2. These tests pin both ends of that scale against the concrete
// bug report (Wan 2.1 T2V 1.3B at 832×480 × 96 frames detonating MPS) and
// the Studio defaults (832×480 × 33 frames staying safe).

describe("assessVideoGenerationSafety()", () => {
  describe("safe envelope on base hardware", () => {
    it("returns safe for the Studio defaults on a 16 GB M2", () => {
      // 832×480 × 33 frames ≈ 14k latent tokens, ~2.9 GB estimated peak.
      // On a 16 GB Mac the effective MPS budget is ~8 GB, so ratio ≈ 0.37
      // — comfortably below the 0.5 caution threshold.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: 16,
      });
      expect(result.riskLevel).toBe("safe");
      expect(result.reason).toBeNull();
      expect(result.suggestion).toBeNull();
      // The memory-estimate contract: always positive for a real request.
      expect(result.estimatedPeakGb).toBeGreaterThan(0);
      expect(result.deviceMemoryGb).toBe(16);
      expect(result.exceedsDevice).toBe(false);
    });

    it("returns safe for tiny clips even on a 16 GB Mac", () => {
      const result = assessVideoGenerationSafety({
        width: 480,
        height: 320,
        numFrames: 17,
        device: "mps",
        deviceMemoryGb: 16,
      });
      expect(result.riskLevel).toBe("safe");
    });

    it("uses the 16 GB MPS fallback when device memory is not reported", () => {
      // The whole point of the fallback is that over-warning a beefy machine
      // beats silently crashing a small one — so unknown-memory should behave
      // as if we're on a 16 GB MPS Mac.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: null,
      });
      expect(result.riskLevel).toBe("safe");
      expect(result.deviceMemoryGb).toBe(16);
    });

    it("ignores non-finite inputs (mid-edit NaN values)", () => {
      // The Studio inputs use NaN to represent "user is mid-edit / field is
      // empty". The safety helper must not flag a phantom warning at that
      // moment — the user hasn't committed to anything yet.
      const result = assessVideoGenerationSafety({
        width: Number.NaN,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: 16,
      });
      expect(result.riskLevel).toBe("safe");
      expect(result.suggestion).toBeNull();
      expect(result.estimatedPeakGb).toBe(0);
    });
  });

  describe("memory-scaled thresholds (Option A)", () => {
    it("a 64 GB M4 Max does NOT flag 832×480 × 50 as caution", () => {
      // This is the headline win of the memory-aware heuristic: a moderate
      // clip that the flat-threshold version would have flagged on every
      // Mac now comes back as safe on a 64 GB box — where it actually fits
      // in RAM with room to spare.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 50,
        device: "mps",
        deviceMemoryGb: 64,
      });
      expect(result.riskLevel).toBe("safe");
    });

    it("a 16 GB M2 DOES flag the same 832×480 × 50 as caution", () => {
      // Same config, smaller machine — it's close to the 8 GB MPS budget so
      // the user gets a heads-up that it might struggle.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 50,
        device: "mps",
        deviceMemoryGb: 16,
      });
      expect(result.riskLevel).toBe("caution");
      expect(result.reason).toMatch(/attention memory/);
    });

    it("a 128 GB M3 Ultra handles the observed-crash config safely", () => {
      // With 128 GB of unified memory and a 64 GB MPS budget, even 96 frames
      // at 832×480 (~16 GB peak) fits with lots of headroom.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 96,
        device: "mps",
        deviceMemoryGb: 128,
      });
      expect(result.riskLevel).toBe("safe");
    });
  });

  describe("danger band — the bug we're preventing", () => {
    it("flags danger for the observed-crash config on a 16 GB Mac", () => {
      // The exact configuration from the user's bug report, worst-case Mac.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 96,
        device: "mps",
        deviceMemoryGb: 16,
      });
      expect(result.riskLevel).toBe("danger");
      expect(result.reason).toMatch(/crash/);
      expect(result.suggestion).not.toBeNull();
      expect(result.suggestion!.numFrames).toBeLessThan(96);
      // Peak estimate should exceed the device budget — that's the whole
      // reason we're warning.
      expect(result.exceedsDevice).toBe(true);
    });

    it("suggestion lands back in the safe envelope", () => {
      // Whatever the helper hands back, applying it must not re-trigger the
      // warning — otherwise the user clicks "Use safer settings" and the
      // callout doesn't go away, which is worse than not having a button.
      const original = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 96,
        device: "mps",
        deviceMemoryGb: 16,
      });
      expect(original.suggestion).not.toBeNull();
      const suggestion = original.suggestion!;
      const recheck = assessVideoGenerationSafety({
        width: suggestion.width,
        height: suggestion.height,
        numFrames: suggestion.numFrames,
        device: "mps",
        deviceMemoryGb: 16,
      });
      expect(recheck.riskLevel).toBe("safe");
    });

    it("suggested frame count is Wan-compatible (n - 1) % 4 == 0", () => {
      // Wan / LTX pipelines reject any other frame count — handing back e.g.
      // 33 (valid) or 17 (valid) is fine; handing back 50 (invalid) breaks
      // the user's next generate.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 96,
        device: "mps",
        deviceMemoryGb: 16,
      });
      const frames = result.suggestion!.numFrames;
      expect((frames - 1) % 4).toBe(0);
    });
  });

  describe("CUDA gets more headroom than MPS at the same memory size", () => {
    it("24 GB CUDA verdicts a config that 24 GB MPS would flag danger", () => {
      // Same config (832×480 × 50 frames, ~4.6 GB peak), same total memory
      // (24 GB), but CUDA's larger effective budget (70% vs 50%) and looser
      // ratios (caution 0.7 vs 0.5) mean the same request is safe on a 4090
      // and only 'caution' on a 24 GB Mac. That's the asymmetry we want.
      const cuda = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 50,
        device: "cuda:0",
        deviceMemoryGb: 24,
      });
      const mps = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 50,
        device: "mps",
        deviceMemoryGb: 24,
      });
      expect(cuda.riskLevel).toBe("safe");
      expect(mps.riskLevel).toBe("caution");
    });

    it("still flags danger when the peak genuinely exceeds CUDA VRAM", () => {
      // A 4090 with 24 GB can't really handle 832×480 × 96 frames without
      // model offload (~20 GB attention peak vs 16.8 GB effective budget)
      // so the heuristic correctly stays at danger here. This test locks
      // that behaviour so we don't accidentally tune CUDA to be too loose.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 96,
        device: "cuda:0",
        deviceMemoryGb: 24,
      });
      expect(result.riskLevel).toBe("danger");
    });

    it("A100-class (40 GB) lands the observed-crash config at caution", () => {
      // With a larger dedicated VRAM pool, the same 96-frame clip is still
      // close to the limit (~20.9 GB peak vs 28 GB budget ≈ 75%) so the
      // user gets a heads-up without a hard block.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 96,
        device: "cuda:0",
        deviceMemoryGb: 40,
      });
      expect(result.riskLevel).toBe("caution");
    });

    it("the observed-crash config on CPU is danger", () => {
      // CPU uses fp32 (4 bytes/element) so the attention peak doubles, and
      // CPU inference is slow anyway — we warn harder.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 96,
        device: "cpu",
        deviceMemoryGb: 32,
      });
      expect(result.riskLevel).toBe("danger");
    });

    it("uses the CUDA fallback memory when not specified", () => {
      // Unknown CUDA device defaults to 12 GB — what most mid-range consumer
      // cards ship with. Overwarning a 4090 user in the "no telemetry" case
      // is still better than green-lighting a crash on a 3060.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "cuda",
        deviceMemoryGb: null,
      });
      expect(result.deviceMemoryGb).toBe(12);
    });
  });

  describe("consumer-facing fields", () => {
    it("returns a positive estimatedPeakGb for every valid request", () => {
      // The Studio uses this number in the always-on capacity line
      // ("this run ≈ 2.9 GB of attention memory"). It must always be
      // positive when inputs are valid — a zero here would read as "this
      // run wants 0 GB", which is nonsense UX.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: 16,
      });
      expect(result.estimatedPeakGb).toBeGreaterThan(0);
    });

    it("preserves the device memory it used in the reply", () => {
      // The Studio reads ``deviceMemoryGb`` from the reply to format the
      // capacity line — whatever we computed against must match what we
      // hand back, otherwise the user sees an inconsistent total.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: 96.5,
      });
      expect(result.deviceMemoryGb).toBe(96.5);
    });
  });
});
