import { describe, expect, it } from "vitest";

import {
  isStrategyCompatible,
  resolveDflashSupport,
  sanitizeSpeculativeSelection,
  strategyIncompatReason,
} from "../runtimeSupport";

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
  it("flags RotorQuant as incompatible with MLX", () => {
    expect(isStrategyCompatible("rotorquant", "mlx")).toBe(false);
    expect(strategyIncompatReason("rotorquant", "mlx")).toContain("llama.cpp or vLLM");
  });
});
