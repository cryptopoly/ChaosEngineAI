import { describe, expect, it } from "vitest";

import { slowAlternativeName, slowPlatformKey, slowRowInfo } from "../slowRows";
import type { VideoModelVariant } from "../../types";
import type { VideoSlowOn } from "../../types/video";

const MAC = { platform: "Darwin", arch: "arm64", gpuVramTotalGb: null };
const INTEL_MAC = { platform: "Darwin", arch: "x86_64", gpuVramTotalGb: null };
const CUDA_PC = { platform: "Windows", arch: "x86_64", gpuVramTotalGb: 24 };
const CUDA_LINUX = { platform: "Linux", arch: "x86_64", gpuVramTotalGb: 48 };
const CPU_WINDOWS = { platform: "Windows", arch: "x86_64", gpuVramTotalGb: null };
const CPU_LINUX = { platform: "Linux", arch: "x86_64", gpuVramTotalGb: 0 };

const apple: VideoSlowOn = {
  platform: "apple-mps",
  basis: "measured",
  reason: "bf16-full-size",
  seconds: 1130,
  measuredOn: "M4 Max, 64 GB",
  measuredDate: "2026-10-21",
  size: "768x512",
  frames: 33,
  alternativeId: "dgrauet/ltx-2.5-mlx-q4",
};
const appleExpected: VideoSlowOn = { platform: "apple-mps", basis: "expected", reason: "pytorch-apple-gpu", alternativeId: "dgrauet/ltx-2.5-mlx-q4" };
const cpu: VideoSlowOn = { platform: "cpu-only", basis: "expected", reason: "no-gpu-cpu" };

describe("slowPlatformKey", () => {
  it("tells Apple Silicon, NVIDIA and CPU-only machines apart", () => {
    expect(slowPlatformKey(MAC)).toBe("apple-mps");
    expect(slowPlatformKey(CUDA_PC)).toBe("cuda");
    expect(slowPlatformKey(CUDA_LINUX)).toBe("cuda");
    expect(slowPlatformKey(CPU_WINDOWS)).toBe("cpu-only");
    expect(slowPlatformKey(CPU_LINUX)).toBe("cpu-only");
    expect(slowPlatformKey(INTEL_MAC)).toBe("cpu-only");
    expect(slowPlatformKey(null)).toBeNull();
    expect(slowPlatformKey(undefined)).toBeNull();
  });
});

describe("slowRowInfo", () => {
  it("returns the measured facts on Apple Silicon", () => {
    expect(slowRowInfo({ slowOn: [apple, cpu] }, MAC)).toEqual({
      platform: "apple-mps",
      kind: "measured",
      reason: "bf16-full-size",
      seconds: 1130,
      measuredOn: "M4 Max, 64 GB",
      measuredDate: "2026-10-21",
      size: "768x512",
      frames: 33,
      alternativeId: "dgrauet/ltx-2.5-mlx-q4",
    });
  });

  it("says expected, with no number, when the entry was not timed", () => {
    const info = slowRowInfo({ slowOn: [appleExpected, cpu] }, MAC);
    expect(info?.kind).toBe("expected");
    expect(info?.seconds).toBeNull();
    expect(info?.measuredOn).toBeNull();
    expect(info?.reason).toBe("pytorch-apple-gpu");
  });

  it("uses the CPU-only entry on a machine with no NVIDIA GPU", () => {
    for (const host of [CPU_WINDOWS, CPU_LINUX]) {
      const info = slowRowInfo({ slowOn: [appleExpected, cpu] }, host);
      expect(info?.platform).toBe("cpu-only");
      expect(info?.kind).toBe("expected");
      expect(info?.seconds).toBeNull();
      expect(info?.reason).toBe("no-gpu-cpu");
    }
  });

  it("never invents a number: a measured entry without time, place or date reads as expected", () => {
    expect(slowRowInfo({ slowOn: [{ ...apple, seconds: null }] }, MAC)?.kind).toBe("expected");
    expect(slowRowInfo({ slowOn: [{ ...apple, measuredOn: "" }] }, MAC)?.kind).toBe("expected");
    expect(slowRowInfo({ slowOn: [{ ...apple, measuredDate: null }] }, MAC)?.seconds).toBeNull();
  });

  it("shows nothing on NVIDIA CUDA unless an entry for it exists", () => {
    for (const host of [CUDA_PC, CUDA_LINUX]) {
      expect(slowRowInfo({ slowOn: [appleExpected, cpu] }, host)).toBeNull();
    }
    const measuredCuda: VideoSlowOn = { ...apple, platform: "cuda" };
    expect(slowRowInfo({ slowOn: [measuredCuda] }, CUDA_PC)?.platform).toBe("cuda");
  });

  it("shows nothing for rows without the field, an unknown host or no matching entry", () => {
    expect(slowRowInfo({}, MAC)).toBeNull();
    expect(slowRowInfo({ slowOn: null }, MAC)).toBeNull();
    expect(slowRowInfo({ slowOn: [] }, MAC)).toBeNull();
    expect(slowRowInfo(null, MAC)).toBeNull();
    expect(slowRowInfo({ slowOn: [apple] }, null)).toBeNull();
    expect(slowRowInfo({ slowOn: [cpu] }, MAC)).toBeNull();
  });
});

describe("slowAlternativeName", () => {
  const ltx25 = {
    id: "dgrauet/ltx-2.5-mlx-q4",
    name: "LTX-2.5 · distilled q4 (ltx-2-mlx)",
    repo: "dgrauet/ltx-2.5-mlx-q4",
    runtime: "ltx-2-mlx (MLX native)",
    styleTags: [],
  } as unknown as VideoModelVariant;

  it("names the faster row on a Mac", () => {
    expect(slowAlternativeName(slowRowInfo({ slowOn: [apple] }, MAC), [ltx25], MAC)).toBe(ltx25.name);
  });

  it("only points at a row that runs on this machine", () => {
    const info = slowRowInfo({ slowOn: [apple, cpu] }, MAC);
    expect(slowAlternativeName(info, [ltx25], CPU_WINDOWS)).toBeNull();
    expect(slowAlternativeName(info, [], MAC)).toBeNull();
    expect(slowAlternativeName(slowRowInfo({ slowOn: [cpu] }, CPU_WINDOWS), [ltx25], CPU_WINDOWS)).toBeNull();
    expect(slowAlternativeName(null, [ltx25], MAC)).toBeNull();
  });
});
