import { describe, expect, it } from "vitest";

import type { LaunchPreferences } from "../../../types";
import type { TensorfoldInfo } from "../../../components/tensorfoldSupport";
import { modelUsesTensorfold, summarizeLaunchSettings } from "../CompareView";

const FLASH = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP";
const DENSE = "Vontra/Qwen3.8-27B-MLX-4bit";

function settings(overrides: Partial<LaunchPreferences> = {}): LaunchPreferences {
  return {
    contextTokens: 8192,
    maxTokens: 2048,
    temperature: 0.7,
    cacheBits: 0,
    fp16Layers: 0,
    fusedAttention: false,
    cacheStrategy: "native",
    fitModelInMemory: true,
    speculativeDecoding: false,
    treeBudget: 0,
    kvBudget: 2048,
    ...overrides,
  };
}

function info(overrides: Partial<TensorfoldInfo> = {}): TensorfoldInfo {
  return {
    available: true,
    version: "0.5.0",
    supportedModels: [DENSE, FLASH],
    exclusiveModels: [FLASH],
    ...overrides,
  };
}

describe("modelUsesTensorfold()", () => {
  it("follows the backend rule for checkpoints stock MLX also loads", () => {
    const option = { canonicalRepo: DENSE, modelRef: "Qwen3.8-27B-MLX-4bit" };
    expect(modelUsesTensorfold(option, settings({ speculativeDecoding: true }), { info: info() })).toBe(true);
    expect(modelUsesTensorfold(option, settings({ speculativeDecoding: false }), { info: info() })).toBe(false);
  });

  it("always runs an exclusive checkpoint on TensorFold once installed", () => {
    const option = { canonicalRepo: FLASH, modelRef: FLASH };
    expect(modelUsesTensorfold(option, settings({ speculativeDecoding: false }), { info: info() })).toBe(true);
  });

  it("is false when TensorFold is not installed or the model is not one it serves", () => {
    const option = { canonicalRepo: FLASH, modelRef: FLASH };
    expect(modelUsesTensorfold(option, settings({ speculativeDecoding: true }), { info: info({ available: false }) })).toBe(false);
    expect(modelUsesTensorfold({ canonicalRepo: "mlx-community/Llama-3.2-3B-Instruct-4bit" }, settings({ speculativeDecoding: true }), { info: info() })).toBe(false);
    expect(modelUsesTensorfold(option, settings({ speculativeDecoding: true }), undefined)).toBe(false);
  });
});

describe("summarizeLaunchSettings()", () => {
  it("names TensorFold ahead of MTPLX and DFlash, as the backend prefers it", () => {
    const on = settings({ speculativeDecoding: true, treeBudget: 32 });
    expect(summarizeLaunchSettings(on, { usesTensorfold: true, usesMtplx: true })).toContain("TensorFold");
    expect(summarizeLaunchSettings(on, { usesMtplx: true })).toContain("MTPLX");
    expect(summarizeLaunchSettings(on)).toContain("DDTree 32");
  });

  it("shows no speculative label while the toggle is off", () => {
    const summary = summarizeLaunchSettings(settings(), { usesTensorfold: true });
    expect(summary).not.toContain("TensorFold");
  });
});
