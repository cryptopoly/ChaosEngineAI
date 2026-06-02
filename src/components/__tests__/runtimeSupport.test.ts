import { describe, expect, it } from "vitest";

import {
  dflashPackageFor,
  isMtpGgufRepo,
  isStrategyCompatible,
  resolveDflashSupport,
  sanitizeSpeculativeSelection,
  strategyIncompatReason,
} from "../runtimeSupport";

describe("dflashPackageFor()", () => {
  it("returns dflash-mlx for the MLX backend", () => {
    expect(dflashPackageFor("mlx")).toBe("dflash-mlx");
    expect(dflashPackageFor("MLX")).toBe("dflash-mlx");
  });

  it("returns dflash for the vLLM CUDA backend", () => {
    expect(dflashPackageFor("vllm")).toBe("dflash");
    expect(dflashPackageFor("VLLM")).toBe("dflash");
  });

  it("defaults to dflash-mlx for null / unknown backends", () => {
    expect(dflashPackageFor(null)).toBe("dflash-mlx");
    expect(dflashPackageFor(undefined)).toBe("dflash-mlx");
    expect(dflashPackageFor("")).toBe("dflash-mlx");
    expect(dflashPackageFor("auto")).toBe("dflash-mlx");
    expect(dflashPackageFor("gguf")).toBe("dflash-mlx");
  });
});

describe("resolveDflashSupport()", () => {
  const dflashInfo = {
    available: true,
    mlxAvailable: true,
    vllmAvailable: false,
    ddtreeAvailable: true,
    supportedModels: [
      "Qwen/Qwen3.5-35B-A3B",
      "Qwen/Qwen3.6-35B-A3B",
      "Qwen/Qwen3-Coder-30B-A3B",
    ],
  };

  it("matches supported community variants by canonical family", () => {
    const result = resolveDflashSupport({
      dflashInfo,
      selectedBackend: "mlx",
      canonicalRepo: "mlx-community/Qwen3.5-35B-A3B-4bit",
      modelName: "Qwen3.5-35B-A3B-MLX-4bit",
    });

    expect(result.enabled).toBe(true);
    expect(result.matchedModel).toBe("Qwen/Qwen3.5-35B-A3B");
    expect(result.reason).toBeNull();
  });

  it("rejects unsupported model families", () => {
    const result = resolveDflashSupport({
      dflashInfo,
      selectedBackend: "mlx",
      canonicalRepo: "Qwen/Qwen3-Coder-Next-32B-Instruct",
      modelName: "Qwen3-Coder-Next-MLX-4bit",
    });

    expect(result.enabled).toBe(false);
    expect(result.reason).toContain("No DFlash draft exists for this model");
  });

  it("rejects llama.cpp targets before checking model support", () => {
    const result = resolveDflashSupport({
      dflashInfo,
      selectedBackend: "gguf",
      canonicalRepo: "Qwen/Qwen3.5-35B-A3B",
    });

    expect(result.enabled).toBe(false);
    expect(result.reason).toContain("llama.cpp");
  });

  it("matches local MLX naming without a canonical repo", () => {
    const result = resolveDflashSupport({
      dflashInfo,
      selectedBackend: "mlx",
      modelRef: "Qwen3.5-35B-A3B-MLX-4bit",
      modelName: "Qwen3.5-35B-A3B-MLX-4bit",
    });

    expect(result.enabled).toBe(true);
    expect(result.matchedModel).toBe("Qwen/Qwen3.5-35B-A3B");
  });

  it("keeps model support when the DFlash runtime is not installed", () => {
    const result = resolveDflashSupport({
      dflashInfo: {
        ...dflashInfo,
        available: false,
        mlxAvailable: false,
      },
      selectedBackend: "mlx",
      canonicalRepo: "mlx-community/Qwen3.6-35B-A3B-4bit",
      modelName: "Qwen3.6-35B-A3B-4bit",
    });

    expect(result.enabled).toBe(false);
    expect(result.modelSupported).toBe(true);
    expect(result.matchedModel).toBe("Qwen/Qwen3.6-35B-A3B");
    expect(result.reason).toContain("Install dflash-mlx");
  });

  it("marks unsupported models even when the DFlash runtime is not installed", () => {
    const result = resolveDflashSupport({
      dflashInfo: {
        ...dflashInfo,
        available: false,
        mlxAvailable: false,
      },
      selectedBackend: "mlx",
      canonicalRepo: "some-org/UnknownModel-7B",
      modelName: "UnknownModel-7B-4bit",
    });

    expect(result.enabled).toBe(false);
    expect(result.modelSupported).toBe(false);
    expect(result.reason).toContain("No DFlash draft exists");
  });
});

describe("sanitizeSpeculativeSelection()", () => {
  const dflashInfo = {
    available: true,
    mlxAvailable: true,
    vllmAvailable: false,
    ddtreeAvailable: true,
    supportedModels: [
      "Qwen/Qwen3.5-35B-A3B",
    ],
  };

  it("forces unsupported models back to standard decoding", () => {
    const result = sanitizeSpeculativeSelection({
      dflashInfo,
      selectedBackend: "mlx",
      canonicalRepo: "Qwen/Qwen3-Coder-Next-32B-Instruct",
      modelName: "Qwen3-Coder-Next-MLX-4bit",
      speculativeDecoding: true,
      treeBudget: 64,
    });

    expect(result.speculativeDecoding).toBe(false);
    expect(result.treeBudget).toBe(0);
    expect(result.support.enabled).toBe(false);
  });
});

describe("strategy compatibility helpers", () => {
  it("flags TriAttention as incompatible with MLX", () => {
    expect(isStrategyCompatible("triattention", "mlx")).toBe(false);
    expect(strategyIncompatReason("triattention", "mlx")).toContain("vLLM");
  });

  it("FU-030: legacy chaosengine + rotorquant ids coerce to turboquant", () => {
    // Persisted user configs that still reference the dropped ids must
    // route through ``canonicalStrategyId`` so frontend filters treat
    // them as turboquant. Mirrors backend ``registry.resolve_legacy_id``.
    expect(isStrategyCompatible("chaosengine", "mlx")).toBe(true);
    expect(isStrategyCompatible("rotorquant", "mlx")).toBe(true);
    expect(strategyIncompatReason("chaosengine", "mlx")).toBeNull();
    expect(strategyIncompatReason("rotorquant", "mlx")).toBeNull();
  });
});

describe("isMtpGgufRepo()", () => {
  it("matches MTP-flavoured GGUF repos (drives the FU-074 GGUF-MTP toggle)", () => {
    expect(isMtpGgufRepo("ggml-org/Qwen3.6-27B-MTP-GGUF")).toBe(true);
    expect(isMtpGgufRepo("ggml-org/Qwen3.6-35B-A3B-MTP-GGUF")).toBe(true);
    expect(isMtpGgufRepo("am17an/Qwen3.6-27B-mtp-gguf-preview")).toBe(true);
  });

  it("rejects non-MTP GGUF and non-GGUF repos", () => {
    // Plain GGUF without MTP heads — no draft-mtp lane.
    expect(isMtpGgufRepo("ggml-org/Qwen3.6-27B-GGUF")).toBe(false);
    expect(isMtpGgufRepo("lmstudio-community/Qwen3.6-27B-GGUF")).toBe(false);
    // MLX repo (MTP via MTPLX, not the GGUF lane).
    expect(isMtpGgufRepo("Qwen/Qwen3.5-4B")).toBe(false);
    expect(isMtpGgufRepo("mlx-community/Qwen3.6-27B-4bit")).toBe(false);
  });

  it("handles null / empty input", () => {
    expect(isMtpGgufRepo(null)).toBe(false);
    expect(isMtpGgufRepo(undefined)).toBe(false);
    expect(isMtpGgufRepo("")).toBe(false);
  });
});
