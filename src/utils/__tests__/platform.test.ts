import { describe, expect, it } from "vitest";

import { isAppleSiliconHost, isCudaHost, isIntelMac } from "../platform";

describe("isAppleSiliconHost", () => {
  it("returns true for Darwin + arm64", () => {
    expect(isAppleSiliconHost({ platform: "Darwin", arch: "arm64" })).toBe(true);
    expect(isAppleSiliconHost({ platform: "darwin", arch: "arm64" })).toBe(true);
  });

  it("treats aarch64 as Apple Silicon (some Linux probes report it)", () => {
    expect(isAppleSiliconHost({ platform: "darwin", arch: "aarch64" })).toBe(true);
  });

  it("returns false for Intel Mac", () => {
    expect(isAppleSiliconHost({ platform: "darwin", arch: "x86_64" })).toBe(false);
  });

  it("returns false for Windows + arm64 (still not Apple Silicon)", () => {
    expect(isAppleSiliconHost({ platform: "windows", arch: "arm64" })).toBe(false);
  });

  it("returns false for Linux", () => {
    expect(isAppleSiliconHost({ platform: "linux", arch: "x86_64" })).toBe(false);
  });

  it("returns false for null / undefined / partial system", () => {
    expect(isAppleSiliconHost(null)).toBe(false);
    expect(isAppleSiliconHost(undefined)).toBe(false);
    // @ts-expect-error — exercising the early-paint defensive branch
    expect(isAppleSiliconHost({ platform: "darwin" })).toBe(false);
  });

  it("is case-insensitive on both fields", () => {
    expect(isAppleSiliconHost({ platform: "DARWIN", arch: "ARM64" })).toBe(true);
  });
});

describe("isCudaHost", () => {
  it("returns true for Windows + x86_64", () => {
    expect(isCudaHost({ platform: "Windows", arch: "x86_64" })).toBe(true);
    expect(isCudaHost({ platform: "windows", arch: "AMD64" })).toBe(true);
  });

  it("returns true for Linux + x86_64", () => {
    expect(isCudaHost({ platform: "linux", arch: "x86_64" })).toBe(true);
  });

  it("returns false for Darwin (no CUDA on macOS)", () => {
    expect(isCudaHost({ platform: "darwin", arch: "x86_64" })).toBe(false);
    expect(isCudaHost({ platform: "darwin", arch: "arm64" })).toBe(false);
  });

  it("returns false for ARM Linux (not the CUDA-class hosts we ship for)", () => {
    expect(isCudaHost({ platform: "linux", arch: "arm64" })).toBe(false);
  });

  it("returns false for null / undefined system", () => {
    expect(isCudaHost(null)).toBe(false);
    expect(isCudaHost(undefined)).toBe(false);
  });
});

describe("isIntelMac", () => {
  it("returns true for Darwin + x86_64", () => {
    expect(isIntelMac({ platform: "darwin", arch: "x86_64" })).toBe(true);
    expect(isIntelMac({ platform: "darwin", arch: "amd64" })).toBe(true);
  });

  it("returns false for Apple Silicon", () => {
    expect(isIntelMac({ platform: "darwin", arch: "arm64" })).toBe(false);
  });

  it("returns false for Windows / Linux", () => {
    expect(isIntelMac({ platform: "windows", arch: "x86_64" })).toBe(false);
    expect(isIntelMac({ platform: "linux", arch: "x86_64" })).toBe(false);
  });
});
