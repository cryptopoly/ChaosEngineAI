import { describe, expect, it } from "vitest";
import { NATIVE_KV_TYPES, nativeKvIndex } from "../NativeKvTypeField";

describe("NativeKvTypeField (FU-178)", () => {
  it("maps cacheBits to the llama.cpp cache types the backend emits", () => {
    expect(NATIVE_KV_TYPES.map((entry) => [entry.bits, entry.label])).toEqual([
      [0, "f16"],
      [8, "q8_0"],
      [4, "q4_0"],
    ]);
    expect(nativeKvIndex(0)).toBe(0);
    expect(nativeKvIndex(8)).toBe(1);
    expect(nativeKvIndex(4)).toBe(2);
  });

  it("shows f16 for a value native does not run on llama.cpp", () => {
    expect(nativeKvIndex(3)).toBe(0);
    expect(nativeKvIndex(16)).toBe(0);
  });
});
