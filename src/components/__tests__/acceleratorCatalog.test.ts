import { describe, expect, it } from "vitest";

import type { NativeBackendStatus } from "../../types/server";
import {
  ACCELERATOR_CATALOG,
  type AcceleratorMeta,
  getAccelerator,
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
