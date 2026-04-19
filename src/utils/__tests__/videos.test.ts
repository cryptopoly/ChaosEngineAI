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
      // The message now says "peak memory" rather than "attention memory"
      // because the estimate can also include the model's resident footprint
      // when the caller passes ``baseModelFootprintGb``. Without it (this
      // case), the peak is just the attention term, but the wording is
      // consistent across both paths so the Studio UI doesn't need to
      // branch its copy.
      expect(result.reason).toMatch(/peak memory/);
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
      // ("this run ≈ 2.9 GB of peak memory"). It must always be positive
      // when inputs are valid — a zero here would read as "this run wants
      // 0 GB", which is nonsense UX.
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

    it("reports modelFootprintGb as 0 when no baseModelFootprintGb is passed", () => {
      // The Studio capacity line only shows a "model ≈ X GB" breakdown
      // when this is non-zero. The tests / attention-only path must leave
      // it at 0 so the UI falls back to the simple peak-memory framing.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: 64,
      });
      expect(result.modelFootprintGb).toBe(0);
    });
  });

  describe("model-footprint-aware estimate (the real Wan 2.1 crash case)", () => {
    // This block calibrates against the actual bug report: Wan 2.1 T2V
    // 1.3B (catalog sizeGb = 16.4) at 832×480 × 40 frames detonated MPS on
    // a 64 GB M4 Max because the model weights + UMT5-XXL text encoder
    // dominate memory — not the attention kernel. The Studio passes
    // ``selectedVariant.sizeGb`` as ``baseModelFootprintGb`` so the
    // warning reflects that reality.

    it("flags danger for Wan 2.1 1.3B at 40 frames on a 64 GB M4 Max", () => {
      // The exact config that crashed the user's backend. With the
      // resident-model term included (16.4 GB disk × 1.4 MPS fragmentation
      // ≈ 23 GB) the estimate now realistically lands in "danger".
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 40,
        device: "mps",
        deviceMemoryGb: 64,
        baseModelFootprintGb: 16.4,
      });
      expect(result.riskLevel).toBe("danger");
      // The resident term is the majority of the peak — the user needs to
      // see that it's the model itself, not just the attention kernel.
      expect(result.modelFootprintGb).toBeGreaterThan(result.estimatedPeakGb / 2);
      expect(result.reason).not.toBeNull();
    });

    it("hands back a null suggestion when the model alone doesn't fit", () => {
      // On a 64 GB M4 Max, Wan 2.1 1.3B's 23 GB resident footprint fills
      // most of the 32 GB MPS budget all by itself — no per-request tweak
      // (smaller resolution, fewer frames) can recover. The right answer
      // is "try a smaller model", which we signal by a null suggestion.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 40,
        device: "mps",
        deviceMemoryGb: 64,
        baseModelFootprintGb: 16.4,
      });
      expect(result.suggestion).toBeNull();
      expect(result.reason).toMatch(/model weights|text encoder/i);
    });

    it("flags danger for Wan 2.1 1.3B on a 16 GB M2 regardless of frame count", () => {
      // The model's resident footprint (~23 GB on MPS) is already three
      // times the 8 GB MPS budget — even a 9-frame 480×320 request is
      // doomed. Confirms the short-circuit triggers for small machines.
      const result = assessVideoGenerationSafety({
        width: 480,
        height: 320,
        numFrames: 9,
        device: "mps",
        deviceMemoryGb: 16,
        baseModelFootprintGb: 16.4,
      });
      expect(result.riskLevel).toBe("danger");
      expect(result.suggestion).toBeNull();
    });

    it("stays safe for Wan 2.1 1.3B on a 128 GB M3 Ultra", () => {
      // 64 GB MPS budget easily swallows 23 GB resident + ~3 GB attention.
      // The calibration target: a big machine actually gets to generate.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: 128,
        baseModelFootprintGb: 16.4,
      });
      expect(result.riskLevel).toBe("safe");
      expect(result.modelFootprintGb).toBeGreaterThan(0);
    });

    it("stays safe for LTX-Video (small model) on a 32 GB Mac", () => {
      // LTX ships at ~2 GB on disk so residency is ~2.8 GB on MPS — fits
      // with lots of headroom even on a 32 GB machine. Proves the
      // heuristic doesn't over-warn small models.
      const result = assessVideoGenerationSafety({
        width: 768,
        height: 512,
        numFrames: 41,
        device: "mps",
        deviceMemoryGb: 32,
        baseModelFootprintGb: 2.0,
      });
      expect(result.riskLevel).toBe("safe");
    });

    it("flags danger for Wan 2.1 14B on a 24 GB RTX 4090", () => {
      // 45 GB catalog size × 1.05 CUDA factor ≈ 47 GB resident. A 4090's
      // 16.8 GB effective VRAM can't hold the weights at all without
      // aggressive offload, so the heuristic correctly short-circuits
      // regardless of resolution / frames.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "cuda:0",
        deviceMemoryGb: 24,
        baseModelFootprintGb: 45,
      });
      expect(result.riskLevel).toBe("danger");
      expect(result.suggestion).toBeNull();
    });

    it("MPS fragmentation factor is larger than CUDA's (same model, same memory)", () => {
      // Apples-to-apples: 10 GB model on the same 32 GB total, MPS vs
      // CUDA. MPS should report a higher resident estimate because
      // unified-memory allocator fragmentation inflates the real footprint
      // more than CUDA's dedicated pool does. Locks that asymmetry so
      // future tweaks don't accidentally flip it.
      const mps = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: 32,
        baseModelFootprintGb: 10,
      });
      const cuda = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "cuda:0",
        deviceMemoryGb: 32,
        baseModelFootprintGb: 10,
      });
      expect(mps.modelFootprintGb).toBeGreaterThan(cuda.modelFootprintGb);
    });

    it("ignores non-positive or non-finite baseModelFootprintGb values", () => {
      // Guard: the caller should be able to pass through whatever the
      // catalog hands them (which occasionally ships a 0 sizeGb for
      // placeholder entries) without the heuristic blowing up or
      // silently applying a nonsense offset.
      const zero = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: 64,
        baseModelFootprintGb: 0,
      });
      const negative = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: 64,
        baseModelFootprintGb: -5,
      });
      const nan = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: 64,
        baseModelFootprintGb: Number.NaN,
      });
      expect(zero.modelFootprintGb).toBe(0);
      expect(negative.modelFootprintGb).toBe(0);
      expect(nan.modelFootprintGb).toBe(0);
    });
  });
});
