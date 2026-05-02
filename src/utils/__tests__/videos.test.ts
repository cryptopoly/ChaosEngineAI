import { afterEach, describe, expect, it } from "vitest";

import {
  assessVideoGenerationSafety,
  inferDeviceFromHostPlatform,
  videoDeleteLabelForRepo,
  videoDeleteRepoForVariant,
  videoDiscoverMemoryEstimate,
  videoDiscoverFamilyMatchesQuery,
  videoDiscoverVariantMatchesQuery,
  videoRuntimeErrorStatus,
} from "../videos";
import type { VideoModelFamily, VideoModelVariant } from "../../types";

// The safety heuristic now scales with device memory rather than a flat
// token threshold — a 64 GB M4 Max should tolerate far more frames than a
// 16 GB base M2. These tests pin both ends of that scale against the concrete
// bug report (Wan 2.1 T2V 1.3B at 832×480 × 96 frames detonating MPS) and
// the Studio defaults (832×480 × 33 frames staying safe).

function makeVideoVariant(overrides: Partial<VideoModelVariant>): VideoModelVariant {
  return {
    id: "variant",
    familyId: "family",
    name: "Variant",
    provider: "Provider",
    repo: "provider/repo",
    link: "https://huggingface.co/provider/repo",
    runtime: "runtime",
    styleTags: ["general"],
    taskSupport: ["txt2video"],
    sizeGb: 1,
    recommendedResolution: "512x512",
    defaultDurationSeconds: 4,
    note: "note",
    availableLocally: false,
    estimatedGenerationSeconds: 60,
    ...overrides,
  };
}

describe("video discover search helpers", () => {
  it("keeps variant-only terms from matching every variant in a family", () => {
    const family: VideoModelFamily = {
      id: "ltx-2",
      name: "LTX-2 (MLX)",
      provider: "Lightricks",
      headline: "Native Apple Silicon LTX-2 models",
      summary: "Pre-converted MLX weights.",
      updatedLabel: "Native MLX",
      badges: ["MLX Native"],
      defaultVariantId: "ltx-distilled",
      variants: [
        makeVideoVariant({
          id: "ltx-distilled",
          name: "LTX-2 distilled (MLX)",
          repo: "prince-canuma/LTX-2-distilled",
          styleTags: ["fast"],
        }),
        makeVideoVariant({
          id: "ltx-dev",
          name: "LTX-2 dev (MLX)",
          repo: "prince-canuma/LTX-2-dev",
          styleTags: ["quality"],
          note: "Dev pipeline for quality.",
        }),
      ],
    };

    expect(videoDiscoverFamilyMatchesQuery(family, "ltx-2")).toBe(true);
    expect(videoDiscoverFamilyMatchesQuery(family, "dev")).toBe(false);
    expect(videoDiscoverVariantMatchesQuery(family.variants[0], "dev")).toBe(false);
    expect(videoDiscoverVariantMatchesQuery(family.variants[1], "dev")).toBe(true);
  });
});

describe("video delete target helpers", () => {
  it("uses the active component download repo before the base repo", () => {
    const variant = makeVideoVariant({
      repo: "Lightricks/LTX-Video",
      ggufRepo: "city96/LTX-Video-gguf",
      ggufFile: "ltx-video-2b-v0.9-Q6_K.gguf",
    });

    expect(videoDeleteRepoForVariant(variant, { repo: "city96/LTX-Video-gguf" })).toBe("city96/LTX-Video-gguf");
  });

  it("uses the primary local repo for partial shared GGUF data", () => {
    const variant = makeVideoVariant({
      repo: "Lightricks/LTX-Video",
      ggufRepo: "city96/LTX-Video-gguf",
      ggufFile: "ltx-video-2b-v0.9-Q6_K.gguf",
      hasLocalData: true,
      localDataRepos: ["city96/LTX-Video-gguf"],
      primaryLocalRepo: "city96/LTX-Video-gguf",
    });

    const repo = videoDeleteRepoForVariant(variant);
    expect(repo).toBe("city96/LTX-Video-gguf");
    expect(videoDeleteLabelForRepo(variant, repo)).toBe("Delete shared GGUF download");
  });
});

describe("videoDiscoverMemoryEstimate()", () => {
  it("returns a GB label at the recommended resolution and default clip length", () => {
    const result = videoDiscoverMemoryEstimate(makeVideoVariant({
      sizeGb: 16.4,
      recommendedResolution: "832x480",
      defaultDurationSeconds: 4,
    }));

    expect(result).not.toBeNull();
    expect(result!.label).toMatch(/^~\d+ GB @ 832×480$/);
    expect(result!.frameCount).toBe(33);
    expect(result!.estimatedPeakGb).toBeGreaterThan(0);
  });

  it("returns null when no size or runtime footprint metadata is available", () => {
    const result = videoDiscoverMemoryEstimate(makeVideoVariant({
      sizeGb: 0,
      coreWeightsGb: null,
      repoSizeGb: null,
      runtimeFootprintGb: undefined,
    }));

    expect(result).toBeNull();
  });

  it("uses catalog runtime size instead of inflated local snapshot size", () => {
    const result = videoDiscoverMemoryEstimate(makeVideoVariant({
      name: "LTX-2.3 · dev (MLX)",
      repo: "prince-canuma/LTX-2.3-dev",
      sizeGb: 19,
      runtimeFootprintGb: 27,
      coreWeightsGb: 45.8,
      onDiskGb: 45.8,
      recommendedResolution: "768x512",
      defaultDurationSeconds: 4,
    }));

    expect(result).not.toBeNull();
    expect(result!.modelFootprintGb).toBeCloseTo(27, 1);
    expect(result!.estimatedPeakGb).toBeLessThan(35);
    expect(result!.title).toMatch(/local storage/i);
  });

  it("prefers host-specific runtime footprint metadata when present", () => {
    const mps = assessVideoGenerationSafety({
      width: 832,
      height: 480,
      numFrames: 33,
      device: "mps",
      deviceMemoryGb: 64,
      baseModelFootprintGb: 16.4,
      runtimeFootprintGb: 14,
      runtimeFootprintMpsGb: 23,
    });
    const cuda = assessVideoGenerationSafety({
      width: 832,
      height: 480,
      numFrames: 33,
      device: "cuda:0",
      deviceMemoryGb: 24,
      baseModelFootprintGb: 16.4,
      runtimeFootprintGb: 14,
      runtimeFootprintMpsGb: 23,
    });

    expect(mps.modelFootprintGb).toBe(23);
    expect(cuda.modelFootprintGb).toBe(14);
  });
});

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
    it("24 GB CUDA verdicts a config that 24 GB MPS would flag caution", () => {
      // Same config (832×480 × 65 frames), same total memory (24 GB).
      // MPS effective budget = 24*0.75 = 18 GB with a tighter caution
      // ratio (0.5); CUDA budget = 24*0.95 = 22.8 GB with a looser
      // caution ratio (0.7). Picked frame count to land in the band
      // where MPS trips caution but CUDA stays safe — this is the
      // asymmetry we surface to users so they understand why the same
      // request is "safe" on a 4090 and "caution" on a 24 GB Mac.
      const cuda = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 65,
        device: "cuda:0",
        deviceMemoryGb: 24,
      });
      const mps = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 65,
        device: "mps",
        deviceMemoryGb: 24,
      });
      expect(cuda.riskLevel).toBe("safe");
      expect(mps.riskLevel).toBe("caution");
    });

    it("flags caution when an attention-only CUDA estimate gets close to VRAM", () => {
      // A 4090 with 24 GB can't really handle 832×480 × 96 frames without
      // model offload (~20 GB attention peak vs 22.8 GB effective budget)
      // so the heuristic warns without hard-blocking.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 96,
        device: "cuda:0",
        deviceMemoryGb: 24,
      });
      expect(result.riskLevel).toBe("caution");
    });

    it("A100-class (40 GB) handles the attention-only observed-crash config", () => {
      // With a larger dedicated VRAM pool, the same 96-frame clip is still
      // comfortably below the limit (~20.9 GB peak vs 38 GB budget).
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 96,
        device: "cuda:0",
        deviceMemoryGb: 40,
      });
      expect(result.riskLevel).toBe("safe");
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

    it("flags caution for Wan 2.1 1.3B at 40 frames on a 64 GB M4 Max", () => {
      // The original observed-crash report. With the corrected MPS budget
      // (65% of unified memory, ~41.6 GB on 64 GB M4 Max) and the legacy
      // sizeGb × 1.4 fallback (16.4 × 1.4 ≈ 23 GB resident), the estimate
      // lands in "caution" — matches real-world reference behaviour where
      // this config runs successfully but is close to the comfortable
      // ceiling. The original "danger" verdict was over-strict.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 40,
        device: "mps",
        deviceMemoryGb: 64,
        baseModelFootprintGb: 16.4,
      });
      expect(result.riskLevel).toBe("caution");
      // The resident term is the majority of the peak — the user needs to
      // see that it's the model itself, not just the attention kernel.
      expect(result.modelFootprintGb).toBeGreaterThan(result.estimatedPeakGb / 2);
      expect(result.reason).not.toBeNull();
    });

    it("runtimeFootprintGb override beats the sizeGb × 1.4 heuristic", () => {
      // When the catalog supplies an explicit resident peak (Wan 2.2 5B
      // declares 22 GB), the estimator must use it directly rather than
      // multiplying disk size by 1.4. Same config without the override
      // would land in danger (24 × 1.4 = 33.6 GB resident); with it,
      // caution on a 64 GB Mac — matching the real Wan 2.2 5B footprint.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: 64,
        baseModelFootprintGb: 24.0,
        runtimeFootprintGb: 22.0,
      });
      expect(result.modelFootprintGb).toBe(22.0);
      // Wan 2.2 5B on 64 GB M4 Max is comfortable — should be safe or
      // caution depending on attention peak, never danger.
      expect(result.riskLevel).not.toBe("danger");
    });

    it("does not hard-block Wan 2.2 5B on a 24 GB RTX 4090", () => {
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "cuda:0",
        deviceMemoryGb: 24,
        baseModelFootprintGb: 24.0,
        runtimeFootprintGb: 22.0,
      });
      expect(result.modelFootprintGb).toBe(22.0);
      expect(result.estimatedPeakGb).toBeCloseTo(22.0, 1);
      expect(result.riskLevel).toBe("caution");
      expect(result.suggestion).toBeNull();
    });

    it("accounts for NF4 on standard Wan 2.2 5B CUDA runs", () => {
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "cuda:0",
        deviceMemoryGb: 24,
        baseModelFootprintGb: 24.0,
        runtimeFootprintGb: 22.0,
        repo: "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        useNf4: true,
      });
      expect(result.modelFootprintGb).toBe(14.5);
      expect(result.riskLevel).toBe("safe");
    });

    it("accounts for NF4 on standard Wan 2.1 14B CUDA runs", () => {
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "cuda:0",
        deviceMemoryGb: 24,
        baseModelFootprintGb: 45.0,
        runtimeFootprintGb: 39.0,
        repo: "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        useNf4: true,
      });
      expect(result.modelFootprintGb).toBe(18.0);
      expect(result.riskLevel).toBe("caution");
    });

    it("accounts for NF4 on HunyuanVideo CUDA runs", () => {
      const result = assessVideoGenerationSafety({
        width: 1280,
        height: 720,
        numFrames: 33,
        device: "cuda:0",
        deviceMemoryGb: 24,
        baseModelFootprintGb: 25.0,
        runtimeFootprintGb: 34.0,
        repo: "hunyuanvideo-community/HunyuanVideo",
        useNf4: true,
      });
      expect(result.modelFootprintGb).toBe(22.0);
      expect(result.riskLevel).not.toBe("danger");
    });

    it("still warns hard for very long Wan 2.2 5B clips on a 24 GB RTX 4090", () => {
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 96,
        device: "cuda:0",
        deviceMemoryGb: 24,
        baseModelFootprintGb: 24.0,
        runtimeFootprintGb: 22.0,
      });
      expect(result.riskLevel).toBe("danger");
      expect(result.suggestion).toBeNull();
    });

    it("hands back a null suggestion when the model alone doesn't fit", () => {
      // 24 GB Mac with Wan 2.1 1.3B's 23 GB resident footprint
      // (16.4 GB disk × 1.4 fallback). MPS budget = 18 GB; the model
      // alone exceeds the 9 GB caution threshold so no per-request
      // tweak (smaller resolution, fewer frames) can recover. Right
      // answer is "try a smaller model", signalled by a null
      // suggestion. (The 64 GB M4 Max no longer trips this path
      // since the bumped MPS budget gives Wan 2.1 1.3B real
      // headroom — matching upstream reference behaviour.)
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 40,
        device: "mps",
        deviceMemoryGb: 24,
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

    it("frames LTX-2 MLX on a 64 GB M4 Max as caution, not a hard no", () => {
      // LTX-2 MLX is large enough to deserve a heads-up, but the old copy
      // called the 50% comfort band the "safe usage" ceiling. That made a
      // 64 GB M4 Max look unsupported even though the run is below the
      // estimated Apple Silicon working set.
      const result = assessVideoGenerationSafety({
        width: 768,
        height: 512,
        numFrames: 24,
        device: "mps",
        deviceMemoryGb: 64,
        baseModelFootprintGb: 19.0,
      });
      expect(result.riskLevel).toBe("caution");
      expect(result.exceedsDevice).toBe(false);
      expect(result.reason).toMatch(/comfort target/i);
      expect(result.reason).toMatch(/working set/i);
      expect(result.reason).not.toMatch(/safe usage tops out/i);
    });

    it("flags danger for Wan 2.1 14B on a 24 GB RTX 4090", () => {
      // 45 GB catalog size × 1.05 CUDA factor ≈ 47 GB resident. A 4090's
      // 22.8 GB effective VRAM can't hold the weights at all without
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

    it("surfaces effectiveDevice and effectiveDeviceWasInferred for explicit MPS", () => {
      // When the backend told us the device, the safety result should echo
      // it back unchanged — and tag it as not inferred. The Studio uses
      // these two fields together to decide whether to mark the device
      // label as a guess in the always-on capacity line.
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "mps",
        deviceMemoryGb: 16,
      });
      expect(result.effectiveDevice).toBe("mps");
      expect(result.effectiveDeviceWasInferred).toBe(false);
    });

    it("surfaces effectiveDevice for explicit CUDA", () => {
      const result = assessVideoGenerationSafety({
        width: 832,
        height: 480,
        numFrames: 33,
        device: "cuda:0",
        deviceMemoryGb: 24,
      });
      expect(result.effectiveDevice).toBe("cuda");
      expect(result.effectiveDeviceWasInferred).toBe(false);
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

describe("host-platform device inference (Windows/Linux fallback)", () => {
  // When the backend probe hasn't come back (sidecar dead, Failed to fetch,
  // first-launch race), ``device`` is null. Before this behaviour landed we
  // defaulted to "mps" unconditionally — meaning a Windows RTX 4090 user saw
  // "close to the safe limit on Apple Silicon" even though there is no
  // Apple Silicon on their machine. These tests pin the new fallback: we
  // infer from the host OS and the label now matches the machine.

  const originalNavigator = globalThis.navigator;

  afterEach(() => {
    // Restore whatever Vitest's happy-dom handed us so we don't poison other tests.
    Object.defineProperty(globalThis, "navigator", {
      value: originalNavigator,
      configurable: true,
      writable: true,
    });
  });

  function stubNavigator(stub: Partial<Navigator> & { userAgentData?: { platform?: string } }): void {
    Object.defineProperty(globalThis, "navigator", {
      value: stub,
      configurable: true,
      writable: true,
    });
  }

  it("infers MPS on macOS via userAgentData", () => {
    stubNavigator({ userAgentData: { platform: "macOS" } });
    expect(inferDeviceFromHostPlatform()).toBe("mps");
  });

  it("infers CUDA on Windows via userAgentData", () => {
    stubNavigator({ userAgentData: { platform: "Windows" } });
    expect(inferDeviceFromHostPlatform()).toBe("cuda");
  });

  it("infers CUDA on Linux via userAgentData", () => {
    stubNavigator({ userAgentData: { platform: "Linux" } });
    expect(inferDeviceFromHostPlatform()).toBe("cuda");
  });

  it("falls back to legacy navigator.platform when userAgentData is missing (WKWebView)", () => {
    // macOS Tauri ships a WKWebView that doesn't expose the modern UA-CH API.
    // The legacy ``platform`` string is still "MacIntel" there — locking this
    // in means a user whose backend probe times out on first launch doesn't
    // see a Windows label on their M4 Mac.
    stubNavigator({ platform: "MacIntel", userAgent: "Mozilla/5.0 (Macintosh)" });
    expect(inferDeviceFromHostPlatform()).toBe("mps");
  });

  it("falls back to user agent substring for macOS when platform is generic", () => {
    // Some embedded WebViews report "" for platform but keep "Mac OS" in the UA.
    stubNavigator({ platform: "", userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" });
    expect(inferDeviceFromHostPlatform()).toBe("mps");
  });

  it("defaults to CUDA for an unknown non-macOS platform", () => {
    // Windows 10 with legacy fields only — the exact shape WebView2 has
    // shipped historically.
    stubNavigator({ platform: "Win32", userAgent: "Mozilla/5.0 (Windows NT 10.0)" });
    expect(inferDeviceFromHostPlatform()).toBe("cuda");
  });

  it("uses the inferred bucket in assessVideoGenerationSafety when device is null", () => {
    // The core of the Windows RTX 4090 bug: when the backend probe never
    // arrives, ``device`` is null. On Windows the safety helper must now
    // bucket into CUDA so the memory warning quotes "this GPU" instead of
    // "Apple Silicon (MPS)".
    stubNavigator({ userAgentData: { platform: "Windows" } });
    const result = assessVideoGenerationSafety({
      width: 832,
      height: 480,
      numFrames: 33,
      device: null,
      deviceMemoryGb: null,
    });
    expect(result.effectiveDevice).toBe("cuda");
    expect(result.effectiveDeviceWasInferred).toBe(true);
    // With the CUDA fallback we also use the CUDA default memory (12 GB) —
    // not the MPS default (16 GB). Locks the two paths together.
    expect(result.deviceMemoryGb).toBe(12);
  });

  it("still uses MPS fallback on macOS when device is null", () => {
    // The complementary case: the original behaviour was correct for
    // macOS and shouldn't regress. A macOS user with a dead probe still
    // sees MPS-strict defaults.
    stubNavigator({ userAgentData: { platform: "macOS" } });
    const result = assessVideoGenerationSafety({
      width: 832,
      height: 480,
      numFrames: 33,
      device: null,
      deviceMemoryGb: null,
    });
    expect(result.effectiveDevice).toBe("mps");
    expect(result.effectiveDeviceWasInferred).toBe(true);
    expect(result.deviceMemoryGb).toBe(16);
  });

  it("does not flag danger on a Windows RTX 4090 for Studio defaults with no backend probe", () => {
    // Regression guard for the exact user-reported symptom: Windows machine
    // with a 24 GB RTX 4090 seeing an Apple Silicon memory warning. With the
    // inferred CUDA bucket + caller-supplied 24 GB memory (which the Studio
    // passes when the backend has at least reported it once before going
    // stale), the Studio defaults stay comfortably safe.
    stubNavigator({ userAgentData: { platform: "Windows" } });
    const result = assessVideoGenerationSafety({
      width: 832,
      height: 480,
      numFrames: 33,
      device: null,
      deviceMemoryGb: 24,
    });
    expect(result.riskLevel).toBe("safe");
    expect(result.effectiveDevice).toBe("cuda");
    // The reason string shouldn't mention Apple Silicon when we're on a
    // Windows host — that was the cryptic bit that confused the user.
    expect(result.reason).toBeNull();
  });

  it("quotes 'this GPU' rather than 'Apple Silicon' in the warning on inferred-CUDA hosts", () => {
    // When the inferred path does fire a warning (e.g. too many frames),
    // the copy has to match the inferred bucket — otherwise we've only
    // half-fixed the bug.
    stubNavigator({ userAgentData: { platform: "Windows" } });
    const result = assessVideoGenerationSafety({
      width: 1920,
      height: 1080,
      numFrames: 97,
      device: null,
      deviceMemoryGb: 8, // deliberately tight so the warning fires
    });
    expect(result.riskLevel).not.toBe("safe");
    expect(result.reason).not.toBeNull();
    expect(result.reason).not.toMatch(/Apple Silicon/i);
    expect(result.reason).toMatch(/this GPU/i);
  });
});

describe("videoRuntimeErrorStatus()", () => {
  // The runtime status that Studio shows ("ENGINE: UNAVAILABLE / Fallback
  // active") is driven straight from this function when a runtime probe
  // fails. The important bit is the ``message`` we pass through — when the
  // Python sidecar dies (e.g. Wan 2.1 OOMs MPS), every fetch rejects with a
  // WebKit-specific string that reads to users as a Diffusers problem
  // rather than a backend transport problem. These tests lock the
  // translation in so the Studio surfaces actionable copy.

  it("translates WebKit's \"Load failed\" into an actionable message", () => {
    // The sidecar-crash bug report: after Wan 2.1 detonated the MPS
    // allocator, the runtime status showed the literal words "Load failed"
    // — WebKit's cryptic signal for "couldn't reach the server at all".
    // Users read that as a video-runtime problem rather than a transport
    // problem and didn't know to click Restart Backend.
    const status = videoRuntimeErrorStatus(new TypeError("Load failed"));
    expect(status.activeEngine).toBe("unavailable");
    expect(status.realGenerationAvailable).toBe(false);
    expect(status.message).toMatch(/Restart Backend/i);
    expect(status.message).not.toMatch(/^load failed$/i);
  });

  it("names the video runtime specifically, not the backend", () => {
    // The global BACKEND ONLINE pill is driven from the health probe and
    // can stay green while the video runtime probe fails (during restart,
    // or on the first boot-time probe while torch is importing). Saying
    // "Backend is not responding" in that state contradicts the pill and
    // confuses users. The message must name the video runtime instead.
    const status = videoRuntimeErrorStatus(new TypeError("Failed to fetch"));
    expect(status.message).toMatch(/video runtime/i);
    expect(status.message).not.toMatch(/^Backend is not responding/i);
  });

  it("translates Chromium's \"Failed to fetch\" the same way", () => {
    // Chromium-based runtimes (Linux Tauri via WebKitGTK, desktop Chrome
    // during dev, Windows WebView2) use a different canonical string for
    // the same condition. Both should route through the same translation.
    const status = videoRuntimeErrorStatus(new TypeError("Failed to fetch"));
    expect(status.message).toMatch(/Restart Backend/i);
  });

  it("is case-insensitive about the transport-error match", () => {
    // Defensive: different runtime versions sometimes capitalise
    // differently. We don't want a future Safari update that returns
    // "load failed" (lowercase) to slip through and resurface the cryptic
    // copy in the Studio.
    const status = videoRuntimeErrorStatus(new TypeError("LOAD FAILED"));
    expect(status.message).toMatch(/Restart Backend/i);
  });

  it("surfaces a dedicated message when the runtime probe times out", () => {
    // fetchJson re-throws AbortController-driven timeouts as "Request to
    // <path> timed out after Xs". That's different from Failed-to-fetch
    // (fetch rejected outright) — it means the backend accepted the
    // connection but didn't respond in time, which typically happens
    // during the first probe while torch is importing on Windows. We
    // translate both distinctly so users know whether to wait or to
    // Restart Backend.
    const status = videoRuntimeErrorStatus(
      new Error("Request to /api/video/runtime timed out after 30s"),
    );
    expect(status.message).toMatch(/video runtime/i);
    expect(status.message).toMatch(/timed out/i);
    expect(status.message).toMatch(/Restart Backend/i);
  });

  it("preserves real backend error messages unchanged", () => {
    // When the sidecar is alive and rejecting with a real message (e.g.
    // "Diffusers is not installed"), we want that to surface to the user
    // as-is — it's already actionable. The translation should ONLY catch
    // the opaque transport strings.
    const status = videoRuntimeErrorStatus(
      new Error("Diffusers is not installed — run pip install diffusers."),
    );
    expect(status.message).toMatch(/Diffusers is not installed/);
    expect(status.message).not.toMatch(/Restart Backend/i);
  });

  it("falls back to a generic message for unknown error shapes", () => {
    // Anything that isn't an Error instance (e.g. a rejected Promise with
    // a string, an object thrown by mistake) should still produce a
    // readable message rather than leaking "[object Object]" or "".
    const status = videoRuntimeErrorStatus("something weird");
    expect(status.message).toMatch(/unavailable/i);
    expect(status.activeEngine).toBe("unavailable");
  });
});
