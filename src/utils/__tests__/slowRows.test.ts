import { describe, expect, it } from "vitest";

import { slowRowInfo } from "../slowRows";
import type { VideoSlowOn } from "../../types/video";

const MAC = { platform: "Darwin", arch: "arm64", gpuVramTotalGb: null };
const INTEL_MAC = { platform: "Darwin", arch: "x86_64", gpuVramTotalGb: null };
const CUDA_PC = { platform: "Windows", arch: "x86_64", gpuVramTotalGb: 24 };
const LINUX = { platform: "Linux", arch: "x86_64", gpuVramTotalGb: 48 };

const measured: VideoSlowOn = {
  platforms: ["apple-silicon"],
  basis: "measured",
  seconds: 1130,
  measuredOn: "M4 Max, 64 GB",
  measuredDate: "2026-10-21",
  size: "768x512",
  frames: 33,
  alternativeId: "dgrauet/ltx-2.5-mlx-q4",
};
const expected: VideoSlowOn = { platforms: ["apple-silicon"], basis: "expected", alternativeId: "dgrauet/ltx-2.5-mlx-q4" };

describe("slowRowInfo", () => {
  it("returns the measured facts on Apple Silicon", () => {
    expect(slowRowInfo({ slowOn: measured }, MAC)).toEqual({
      kind: "measured",
      seconds: 1130,
      measuredOn: "M4 Max, 64 GB",
      measuredDate: "2026-10-21",
      size: "768x512",
      frames: 33,
      alternativeId: "dgrauet/ltx-2.5-mlx-q4",
    });
  });

  it("says expected, with no number, when the row was not timed", () => {
    const info = slowRowInfo({ slowOn: expected }, MAC);
    expect(info?.kind).toBe("expected");
    expect(info?.seconds).toBeNull();
    expect(info?.measuredOn).toBeNull();
    expect(info?.alternativeId).toBe("dgrauet/ltx-2.5-mlx-q4");
  });

  it("never invents a number: a measured row without time, place or date reads as expected", () => {
    expect(slowRowInfo({ slowOn: { ...measured, seconds: null } }, MAC)?.kind).toBe("expected");
    expect(slowRowInfo({ slowOn: { ...measured, measuredOn: "" } }, MAC)?.kind).toBe("expected");
    expect(slowRowInfo({ slowOn: { ...measured, measuredDate: null } }, MAC)?.seconds).toBeNull();
  });

  it("shows nothing on CUDA, Linux, Intel Macs or an unknown host", () => {
    for (const host of [CUDA_PC, LINUX, INTEL_MAC, null, undefined]) {
      expect(slowRowInfo({ slowOn: measured }, host)).toBeNull();
      expect(slowRowInfo({ slowOn: expected }, host)).toBeNull();
    }
  });

  it("shows nothing for rows without the catalog field or for another platform list", () => {
    expect(slowRowInfo({}, MAC)).toBeNull();
    expect(slowRowInfo({ slowOn: null }, MAC)).toBeNull();
    expect(slowRowInfo(null, MAC)).toBeNull();
    expect(slowRowInfo({ slowOn: { ...expected, platforms: ["cuda"] } }, MAC)).toBeNull();
  });
});
