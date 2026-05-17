import { describe, expect, it } from "vitest";

import {
  chatVariantPlatformGate,
  imageOrVideoVariantPlatformGate,
  isAppleSiliconHost,
  isCudaHost,
  isIntelMac,
  isVariantCompatibleWithHost,
} from "../platform";

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

describe("imageOrVideoVariantPlatformGate", () => {
  it("mflux runtime → apple-silicon", () => {
    expect(imageOrVideoVariantPlatformGate({ runtime: "mflux (MLX native)" })).toBe("apple-silicon");
  });

  it("mlx-video runtime → apple-silicon", () => {
    expect(imageOrVideoVariantPlatformGate({ runtime: "mlx-video (MLX native)" })).toBe("apple-silicon");
  });

  it("prince-canuma repos → apple-silicon (LTX-2 family)", () => {
    expect(imageOrVideoVariantPlatformGate({ repo: "prince-canuma/LTX-2-distilled", runtime: "" }))
      .toBe("apple-silicon");
  });

  it("apple-silicon styleTag → apple-silicon", () => {
    expect(imageOrVideoVariantPlatformGate({ runtime: "", styleTags: ["fast", "apple-silicon"] }))
      .toBe("apple-silicon");
  });

  it("nunchaku runtime → cuda", () => {
    expect(imageOrVideoVariantPlatformGate({ runtime: "diffusers + nunchaku SVDQuant (CUDA)" }))
      .toBe("cuda");
  });

  it("cuda styleTag → cuda", () => {
    expect(imageOrVideoVariantPlatformGate({ runtime: "", styleTags: ["cuda", "int4"] })).toBe("cuda");
  });

  it("diffusers / sd.cpp / GGUF rows → any", () => {
    expect(imageOrVideoVariantPlatformGate({ runtime: "diffusers LTXPipeline" })).toBe("any");
    expect(imageOrVideoVariantPlatformGate({ runtime: "stable-diffusion.cpp (subprocess)" })).toBe("any");
    expect(imageOrVideoVariantPlatformGate({ runtime: "Stub diffusion pipeline", styleTags: ["gguf"] }))
      .toBe("any");
  });

  it("empty / missing variant → any (safe default)", () => {
    expect(imageOrVideoVariantPlatformGate({})).toBe("any");
  });
});

describe("chatVariantPlatformGate", () => {
  it("mlx backend → apple-silicon", () => {
    expect(chatVariantPlatformGate({ backend: "mlx" })).toBe("apple-silicon");
    expect(chatVariantPlatformGate({ backend: "MLX" })).toBe("apple-silicon");
  });

  it("vllm backend → cuda", () => {
    expect(chatVariantPlatformGate({ backend: "vllm" })).toBe("cuda");
  });

  it("llama.cpp / gguf → any", () => {
    expect(chatVariantPlatformGate({ backend: "llama.cpp" })).toBe("any");
    expect(chatVariantPlatformGate({ backend: "gguf" })).toBe("any");
    expect(chatVariantPlatformGate({ backend: "auto" })).toBe("any");
  });

  it("missing backend → any", () => {
    expect(chatVariantPlatformGate({})).toBe("any");
  });
});

describe("isVariantCompatibleWithHost", () => {
  const win = { platform: "windows", arch: "x86_64" };
  const linux = { platform: "linux", arch: "x86_64" };
  const apple = { platform: "darwin", arch: "arm64" };
  const intelMac = { platform: "darwin", arch: "x86_64" };

  it("'any' gate passes every host", () => {
    expect(isVariantCompatibleWithHost("any", win)).toBe(true);
    expect(isVariantCompatibleWithHost("any", linux)).toBe(true);
    expect(isVariantCompatibleWithHost("any", apple)).toBe(true);
    expect(isVariantCompatibleWithHost("any", intelMac)).toBe(true);
  });

  it("'apple-silicon' gate only passes Apple Silicon", () => {
    expect(isVariantCompatibleWithHost("apple-silicon", apple)).toBe(true);
    expect(isVariantCompatibleWithHost("apple-silicon", win)).toBe(false);
    expect(isVariantCompatibleWithHost("apple-silicon", linux)).toBe(false);
    expect(isVariantCompatibleWithHost("apple-silicon", intelMac)).toBe(false);
  });

  it("'cuda' gate passes Win+Linux x86_64, not Mac", () => {
    expect(isVariantCompatibleWithHost("cuda", win)).toBe(true);
    expect(isVariantCompatibleWithHost("cuda", linux)).toBe(true);
    expect(isVariantCompatibleWithHost("cuda", apple)).toBe(false);
    expect(isVariantCompatibleWithHost("cuda", intelMac)).toBe(false);
  });

  it("null / undefined system → true (early-paint safety)", () => {
    expect(isVariantCompatibleWithHost("apple-silicon", null)).toBe(true);
    expect(isVariantCompatibleWithHost("cuda", undefined)).toBe(true);
  });
});
