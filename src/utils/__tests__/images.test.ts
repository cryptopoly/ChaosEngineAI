import { describe, expect, it } from "vitest";

import type { ImageModelVariant } from "../../types";
import { assessImageGenerationSafety, imageDiscoverMemoryEstimate } from "../images";
import { compareDiscoverVariants } from "../discoverSort";

describe("assessImageGenerationSafety()", () => {
  it("does not block standard FLUX 1024px generation on a 64 GB MPS Mac", () => {
    const result = assessImageGenerationSafety({
      width: 1024,
      height: 1024,
      device: "mps",
      deviceMemoryGb: 64,
      baseModelFootprintGb: 23.7,
      repo: "black-forest-labs/FLUX.1-schnell",
    });

    expect(result.riskLevel).toBe("safe");
    expect(result.effectiveDevice).toBe("mps");
    expect(result.modelFootprintGb).toBeCloseTo(17.775, 3);
  });

  it("still warns hard for standard FLUX on a 16 GB MPS Mac", () => {
    const result = assessImageGenerationSafety({
      width: 1024,
      height: 1024,
      device: "mps",
      deviceMemoryGb: 16,
      baseModelFootprintGb: 23.7,
      repo: "black-forest-labs/FLUX.1-schnell",
    });

    expect(result.riskLevel).toBe("danger");
    expect(result.reason).toMatch(/Apple Silicon \(MPS\)/);
    expect(result.suggestion).toBeNull();
  });

  it("flags 2K FLUX on 64 GB MPS because attention dominates", () => {
    const result = assessImageGenerationSafety({
      width: 2048,
      height: 2048,
      device: "mps",
      deviceMemoryGb: 64,
      baseModelFootprintGb: 23.7,
      repo: "black-forest-labs/FLUX.1-schnell",
    });

    expect(result.riskLevel).toBe("danger");
    expect(result.reason).toMatch(/peak memory/);
    expect(result.suggestion?.width).toBeLessThan(2048);
  });

  it("uses CUDA FLUX offload assumptions on dedicated GPUs", () => {
    const result = assessImageGenerationSafety({
      width: 1024,
      height: 1024,
      device: "cuda",
      deviceMemoryGb: 24,
      baseModelFootprintGb: 23.7,
      repo: "black-forest-labs/FLUX.1-dev",
    });

    expect(result.riskLevel).toBe("safe");
    expect(result.effectiveDevice).toBe("cuda");
    expect(result.reason).toBeNull();
  });

  it("does not apply FLUX runtime discounts to non-FLUX CUDA models", () => {
    const result = assessImageGenerationSafety({
      width: 1024,
      height: 1024,
      device: "cuda",
      deviceMemoryGb: 8,
      baseModelFootprintGb: 16.5,
      repo: "stabilityai/stable-diffusion-3.5-large-turbo",
    });

    expect(result.riskLevel).toBe("danger");
    expect(result.reason).toMatch(/this GPU/);
  });

  it("keeps CPU warnings labelled as CPU generation", () => {
    const result = assessImageGenerationSafety({
      width: 1024,
      height: 1024,
      device: "cpu",
      deviceMemoryGb: 16,
      baseModelFootprintGb: 13.1,
      repo: "stabilityai/stable-diffusion-xl-base-1.0",
    });

    expect(result.effectiveDevice).toBe("cpu");
    expect(result.riskLevel).not.toBe("safe");
    expect(result.reason).toMatch(/CPU generation/);
  });

  it("matches Qwen-Image's backend CPU route on Apple Silicon", () => {
    const result = assessImageGenerationSafety({
      width: 1024,
      height: 1024,
      device: "mps",
      deviceMemoryGb: 64,
      baseModelFootprintGb: 57.7,
      runtimeFootprintMpsGb: 60,
      runtimeFootprintCpuGb: 72,
      repo: "Qwen/Qwen-Image",
    });

    expect(result.effectiveDevice).toBe("cpu");
    expect(result.modelFootprintGb).toBe(72);
    expect(result.reason).toMatch(/CPU generation/);
  });

  it("falls back safely for invalid dimensions", () => {
    const result = assessImageGenerationSafety({
      width: 0,
      height: 1024,
      device: "mps",
      deviceMemoryGb: 64,
      baseModelFootprintGb: 23.7,
      repo: "black-forest-labs/FLUX.1-schnell",
    });

    expect(result.riskLevel).toBe("safe");
    expect(result.estimatedPeakGb).toBe(0);
  });
});

describe("imageDiscoverMemoryEstimate()", () => {
  const baseVariant: ImageModelVariant = {
    id: "black-forest-labs/FLUX.1-schnell",
    familyId: "flux-fast",
    name: "FLUX.1 Schnell",
    provider: "Black Forest Labs",
    repo: "black-forest-labs/FLUX.1-schnell",
    link: "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
    runtime: "Diffusers",
    styleTags: ["general"],
    taskSupport: ["txt2img"],
    sizeGb: 23.7,
    recommendedResolution: "1024x1024",
    note: "Fast local generation.",
    availableLocally: false,
    estimatedGenerationSeconds: null,
  };

  it("returns a GB label at the recommended resolution", () => {
    const result = imageDiscoverMemoryEstimate(baseVariant);

    expect(result).not.toBeNull();
    expect(result!.label).toMatch(/^~\d+ GB @ 1024×1024$/);
    expect(result!.estimatedPeakGb).toBeGreaterThan(0);
  });

  it("uses live core weight metadata ahead of stale catalog size", () => {
    const result = imageDiscoverMemoryEstimate({
      ...baseVariant,
      coreWeightsGb: 10,
      sizeGb: 23.7,
    });

    expect(result!.modelFootprintGb).toBeCloseTo(7.5, 2);
  });

  it("accounts for FLUX GGUF text encoders and MPS runtime overhead", () => {
    const result = imageDiscoverMemoryEstimate({
      ...baseVariant,
      id: "black-forest-labs/FLUX.1-dev-gguf-q4km",
      name: "FLUX.1 Dev · GGUF Q4_K_M",
      repo: "black-forest-labs/FLUX.1-dev",
      ggufRepo: "city96/FLUX.1-dev-gguf",
      ggufFile: "flux1-dev-Q4_K_M.gguf",
      sizeGb: 6.8,
      coreWeightsGb: null,
      onDiskGb: 53.9,
    });

    expect(result).not.toBeNull();
    expect(result!.modelFootprintGb).toBeCloseTo(41.3, 1);
    expect(result!.estimatedPeakGb).toBeGreaterThan(42);
    expect(result!.estimatedPeakGb).toBeLessThan(45);
    expect(result!.label).toMatch(/^~43 GB @ 1024×1024$/);
  });

  it("prefers host-specific runtime footprint metadata when present", () => {
    const result = imageDiscoverMemoryEstimate({
      ...baseVariant,
      id: "Tongyi-MAI/Z-Image-Turbo",
      name: "Z-Image-Turbo",
      repo: "Tongyi-MAI/Z-Image-Turbo",
      sizeGb: 30.58,
      runtimeFootprintGb: 16,
      runtimeFootprintMpsGb: 20,
    });

    expect(result).not.toBeNull();
    expect(result!.modelFootprintGb).toBe(20);
    expect(result!.label).toMatch(/^~22 GB @ 1024×1024$/);
  });

  it("returns null when model size is unknown", () => {
    const result = imageDiscoverMemoryEstimate({
      ...baseVariant,
      sizeGb: 0,
      coreWeightsGb: null,
      repoSizeGb: null,
    });

    expect(result).toBeNull();
  });
});

describe("compareDiscoverVariants()", () => {
  it("sorts Discover models by size and RAM metadata with unknowns last", () => {
    const small = { name: "small", sizeGb: 4, releaseDate: "2026-01" };
    const large = { name: "large", coreWeightsGb: 12, sizeGb: 8, releaseDate: "2025-01" };
    const unknown = { name: "unknown", releaseDate: "2027-01" };

    expect([small, unknown, large].sort((a, b) => compareDiscoverVariants("size", a, b)).map((item) => item.name))
      .toEqual(["large", "small", "unknown"]);
    expect([small, unknown, large].sort((a, b) => compareDiscoverVariants("ram", a, b)).map((item) => item.name))
      .toEqual(["large", "small", "unknown"]);
  });
});
