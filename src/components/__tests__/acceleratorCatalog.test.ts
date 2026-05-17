import { describe, expect, it } from "vitest";

import type { NativeBackendStatus } from "../../types/server";
import {
  ACCELERATOR_CATALOG,
  type AcceleratorMeta,
  getAccelerator,
  getApplicableAccelerators,
  isPlatformCompatible,
} from "../acceleratorCatalog";

/**
 * The catalog is the source of truth for "which accelerators exist".
 * Tests pin its shape so a typo in a pip-package name, a missing
 * capability field, or a stale entry can't ship silently — every
 * downstream surface (Phase 3-6) reads this registry verbatim.
 */

describe("ACCELERATOR_CATALOG", () => {
  it("ships exactly the six accelerators FU-056 Phase 1 wired probes for", () => {
    const ids = ACCELERATOR_CATALOG.map((entry) => entry.id).sort();
    expect(ids).toEqual([
      "dflash-cuda",
      "dflash-mlx",
      "kvpress",
      "nunchaku",
      "sageattention",
      "triattention",
    ]);
  });

  it.each(ACCELERATOR_CATALOG.map((entry) => [entry.id, entry]))(
    "%s catalog entry has all required fields",
    (_id, entry) => {
      expect(entry.label.length).toBeGreaterThan(0);
      expect(entry.shortLabel.length).toBeGreaterThan(0);
      expect(entry.pipPackage.length).toBeGreaterThan(0);
      expect(entry.capabilityField.length).toBeGreaterThan(0);
      expect(entry.versionField.length).toBeGreaterThan(0);
      expect(entry.speedupClaim.length).toBeGreaterThan(0);
      expect(entry.appliesTo.length).toBeGreaterThan(0);
      expect(entry.sizeOnDiskLabel.length).toBeGreaterThan(0);
      expect(["sync", "async"]).toContain(entry.installMode);
      expect(["cuda", "apple-silicon", "any"]).toContain(entry.platformGate);
      // FU row reference must look like "FU-NNN" (followUp string can
      // pair multiple FUs separated by "/", e.g. "FU-003 / FU-002").
      expect(entry.followUp).toMatch(/FU-\d{3}/);
    },
  );

  it("capability field names follow the Phase 1 ``*Available`` convention", () => {
    for (const entry of ACCELERATOR_CATALOG) {
      expect(entry.capabilityField).toMatch(/Available$/);
      expect(entry.versionField).toMatch(/Version$/);
    }
  });

  it("getAccelerator resolves known ids", () => {
    expect(getAccelerator("nunchaku")?.label).toBe("Nunchaku");
    expect(getAccelerator("sageattention")?.label).toBe("SageAttention");
    expect(getAccelerator("dflash-cuda")?.platformGate).toBe("cuda");
    expect(getAccelerator("dflash-mlx")?.platformGate).toBe("apple-silicon");
  });

  it("getAccelerator returns undefined for unknown ids", () => {
    expect(getAccelerator("flash-attn-3")).toBeUndefined();
    expect(getAccelerator("")).toBeUndefined();
  });
});

describe("getApplicableAccelerators", () => {
  it("returns empty for null / empty / unknown repos", () => {
    expect(getApplicableAccelerators(null)).toEqual([]);
    expect(getApplicableAccelerators(undefined)).toEqual([]);
    expect(getApplicableAccelerators("")).toEqual([]);
    expect(getApplicableAccelerators("some/random-thing")).toEqual([]);
  });

  it("returns empty for UNet pipelines (SDXL, SD1.5)", () => {
    expect(getApplicableAccelerators("stabilityai/stable-diffusion-xl-base-1.0")).toEqual([]);
    expect(getApplicableAccelerators("runwayml/stable-diffusion-v1-5")).toEqual([]);
    expect(getApplicableAccelerators("stabilityai/sdxl-turbo")).toEqual([]);
  });

  it("recommends nunchaku + sageattention for FLUX.1", () => {
    expect(getApplicableAccelerators("black-forest-labs/FLUX.1-dev")).toEqual([
      "nunchaku",
      "sageattention",
    ]);
    expect(getApplicableAccelerators("black-forest-labs/FLUX.1-schnell")).toEqual([
      "nunchaku",
      "sageattention",
    ]);
    expect(getApplicableAccelerators("black-forest-labs/FLUX.1-Kontext-dev")).toEqual([
      "nunchaku",
      "sageattention",
    ]);
  });

  it("recommends nunchaku + sageattention for SD3.5 / Qwen-Image / SANA / PixArt-Σ", () => {
    expect(getApplicableAccelerators("stabilityai/stable-diffusion-3.5-large")).toEqual([
      "nunchaku",
      "sageattention",
    ]);
    expect(getApplicableAccelerators("Qwen/Qwen-Image")).toEqual(["nunchaku", "sageattention"]);
    expect(getApplicableAccelerators("Qwen/Qwen-Image-2512")).toEqual([
      "nunchaku",
      "sageattention",
    ]);
    expect(getApplicableAccelerators("Efficient-Large-Model/SANA-1024px")).toEqual([
      "nunchaku",
      "sageattention",
    ]);
    expect(getApplicableAccelerators("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")).toEqual([
      "nunchaku",
      "sageattention",
    ]);
  });

  it("recommends sageattention (only) for video DiTs that nunchaku doesn't cover", () => {
    expect(getApplicableAccelerators("Wan-AI/Wan2.2-T2V-A14B-Diffusers")).toEqual([
      "sageattention",
    ]);
    expect(getApplicableAccelerators("tencent/HunyuanVideo")).toEqual(["sageattention"]);
    expect(getApplicableAccelerators("Lightricks/LTX-Video")).toEqual(["sageattention"]);
    expect(getApplicableAccelerators("THUDM/CogVideoX-5b")).toEqual(["sageattention"]);
    expect(getApplicableAccelerators("genmo/mochi-1-preview")).toEqual(["sageattention"]);
  });

  it("adds triattention for the specific Wan2.1 1.3B repo LongLive targets", () => {
    expect(getApplicableAccelerators("Wan-AI/Wan2.1-T2V-1.3B")).toEqual([
      "sageattention",
      "triattention",
    ]);
    expect(getApplicableAccelerators("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")).toEqual([
      "sageattention",
      "triattention",
    ]);
    // Other Wan sizes shouldn't surface triattention yet.
    expect(getApplicableAccelerators("Wan-AI/Wan2.1-T2V-14B-Diffusers")).toEqual([
      "sageattention",
    ]);
  });

  it("is case-insensitive", () => {
    expect(getApplicableAccelerators("BLACK-FOREST-LABS/flux.1-DEV")).toEqual([
      "nunchaku",
      "sageattention",
    ]);
  });
});

describe("isPlatformCompatible", () => {
  const cudaCaps = { mlxAvailable: false } as Pick<
    NativeBackendStatus,
    "mlxAvailable"
  >;
  const mlxCaps = { mlxAvailable: true } as Pick<
    NativeBackendStatus,
    "mlxAvailable"
  >;

  it("``any`` platform-gated entries are always compatible", () => {
    const fake: AcceleratorMeta = {
      ...ACCELERATOR_CATALOG[0],
      platformGate: "any",
    };
    expect(isPlatformCompatible(fake, cudaCaps)).toBe(true);
    expect(isPlatformCompatible(fake, mlxCaps)).toBe(true);
  });

  it("``cuda`` entries match when mlx is unavailable", () => {
    const nunchaku = ACCELERATOR_CATALOG.find((e) => e.id === "nunchaku")!;
    expect(isPlatformCompatible(nunchaku, cudaCaps)).toBe(true);
    expect(isPlatformCompatible(nunchaku, mlxCaps)).toBe(false);
  });

  it("``apple-silicon`` entries match when mlx is available", () => {
    const dflashMlx = ACCELERATOR_CATALOG.find((e) => e.id === "dflash-mlx")!;
    expect(isPlatformCompatible(dflashMlx, cudaCaps)).toBe(false);
    expect(isPlatformCompatible(dflashMlx, mlxCaps)).toBe(true);
  });
});
