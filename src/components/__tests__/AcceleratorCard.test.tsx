import { describe, expect, it } from "vitest";

import type { NativeBackendStatus } from "../../types/server";
import {
  actionLabelFor,
  platformLabel,
  readInstalled,
  readVersion,
} from "../AcceleratorCard";
import { ACCELERATOR_CATALOG, getAccelerator } from "../acceleratorCatalog";

/**
 * No JSX render harness in the repo today (per
 * src/components/__tests__/ErrorBoundary.test.ts comment). We pin the
 * card's *pure-function* contract instead — the same helpers the
 * component body calls, exported for direct test reach.
 */

function makeCaps(overrides: Partial<NativeBackendStatus> = {}): NativeBackendStatus {
  return {
    pythonExecutable: "/x/python",
    mlxAvailable: false,
    mlxLmAvailable: false,
    mlxUsable: false,
    ggufAvailable: false,
    converterAvailable: false,
    ...overrides,
  };
}

describe("readInstalled", () => {
  const nunchaku = getAccelerator("nunchaku")!;

  it("returns false when capabilities is null", () => {
    expect(readInstalled(nunchaku, null)).toBe(false);
  });

  it("returns false when the field is missing (older backend)", () => {
    expect(readInstalled(nunchaku, makeCaps())).toBe(false);
  });

  it("returns true when the capability field is true", () => {
    expect(readInstalled(nunchaku, makeCaps({ nunchakuAvailable: true }))).toBe(true);
  });

  it("returns false when the capability field is explicitly false", () => {
    expect(readInstalled(nunchaku, makeCaps({ nunchakuAvailable: false }))).toBe(false);
  });
});

describe("readVersion", () => {
  const nunchaku = getAccelerator("nunchaku")!;

  it("returns null when capabilities is null", () => {
    expect(readVersion(nunchaku, null)).toBeNull();
  });

  it("returns null when the version field is missing or empty", () => {
    expect(readVersion(nunchaku, makeCaps())).toBeNull();
    expect(readVersion(nunchaku, makeCaps({ nunchakuVersion: "" }))).toBeNull();
  });

  it("returns the version string when present", () => {
    expect(readVersion(nunchaku, makeCaps({ nunchakuVersion: "1.2.1" }))).toBe("1.2.1");
  });

  it("returns null when the version is explicitly null", () => {
    expect(readVersion(nunchaku, makeCaps({ nunchakuVersion: null }))).toBeNull();
  });
});

describe("platformLabel", () => {
  it("maps every gate to a human-readable string", () => {
    expect(platformLabel("cuda")).toBe("CUDA only");
    expect(platformLabel("apple-silicon")).toBe("Apple Silicon only");
    expect(platformLabel("any")).toBe("Cross-platform");
  });

  it("covers every catalog platformGate value", () => {
    // Pins that every catalog entry uses a gate platformLabel knows
    // how to render — a new gate value would force this test to fail.
    for (const entry of ACCELERATOR_CATALOG) {
      const label = platformLabel(entry.platformGate);
      expect(label.length).toBeGreaterThan(0);
    }
  });
});

describe("actionLabelFor", () => {
  it("returns null when already installed (no button rendered)", () => {
    expect(
      actionLabelFor({ installed: true, installing: false, hasError: false, installMode: "sync" }),
    ).toBeNull();
  });

  it("returns ``Installing…`` mid-flight (overrides error)", () => {
    expect(
      actionLabelFor({ installed: false, installing: true, hasError: true, installMode: "sync" }),
    ).toBe("Installing…");
  });

  it("returns ``Retry`` after a failed attempt", () => {
    expect(
      actionLabelFor({ installed: false, installing: false, hasError: true, installMode: "sync" }),
    ).toBe("Retry");
  });

  it("returns ``Install`` for fresh sync installs", () => {
    expect(
      actionLabelFor({ installed: false, installing: false, hasError: false, installMode: "sync" }),
    ).toBe("Install");
  });

  it("returns ``Install (background)`` for async installs", () => {
    expect(
      actionLabelFor({ installed: false, installing: false, hasError: false, installMode: "async" }),
    ).toBe("Install (background)");
  });
});
