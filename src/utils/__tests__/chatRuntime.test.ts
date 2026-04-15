import { describe, expect, it } from "vitest";

import { loadedModelMatchesRuntimeProfile, resolveChatRuntimeProfile } from "../chatRuntime";
import type { ChatSession, LaunchPreferences, LoadedModel } from "../../types";

const launchSettings: LaunchPreferences = {
  contextTokens: 8192,
  maxTokens: 4096,
  temperature: 0.7,
  cacheBits: 4,
  fp16Layers: 8,
  fusedAttention: false,
  cacheStrategy: "native",
  fitModelInMemory: true,
  speculativeDecoding: false,
  treeBudget: 0,
};

function makeSession(overrides: Partial<ChatSession> & { id: string }): ChatSession {
  const { id, ...rest } = overrides;
  return {
    id,
    title: "Thread",
    updatedAt: "2026-04-15 13:00:00",
    model: "Test model",
    cacheLabel: "Native f16",
    messages: [],
    ...rest,
  };
}

function makeLoadedModel(overrides: Partial<LoadedModel> = {}): LoadedModel {
  return {
    ref: "mlx-community/Qwen3-Coder-Next-MLX-4bit",
    name: "Qwen3-Coder-Next-MLX-4bit",
    canonicalRepo: "Qwen/Qwen3-Coder-Next",
    backend: "mlx",
    source: "library",
    engine: "mlx",
    cacheBits: 4,
    fp16Layers: 8,
    fusedAttention: false,
    cacheStrategy: "native",
    fitModelInMemory: true,
    contextTokens: 8192,
    loadedAt: "2026-04-15 13:00:00",
    path: "/tmp/qwen",
    runtimeTarget: "/tmp/qwen",
    runtimeNote: "Fake runtime",
    speculativeDecoding: false,
    dflashDraftModel: null,
    treeBudget: 0,
    ...overrides,
  };
}

describe("resolveChatRuntimeProfile()", () => {
  it("always uses launch settings regardless of session overrides", () => {
    // Session has different values — launch settings should win
    const session = makeSession({
      id: "thread-1",
      cacheStrategy: "turboquant",
      cacheBits: 1,
      fp16Layers: 0,
      speculativeDecoding: true,
      treeBudget: 64,
    });

    const profile = resolveChatRuntimeProfile(session, launchSettings);
    expect(profile.cacheStrategy).toBe("native"); // from launchSettings
    expect(profile.cacheBits).toBe(4); // from launchSettings
    expect(profile.speculativeDecoding).toBe(false); // from launchSettings
  });

  it("reflects launch settings changes immediately", () => {
    const session = makeSession({ id: "thread-2", speculativeDecoding: false });
    const updatedLaunch = { ...launchSettings, speculativeDecoding: true, treeBudget: 64 };

    const profile = resolveChatRuntimeProfile(session, updatedLaunch);
    expect(profile.speculativeDecoding).toBe(true);
    expect(profile.treeBudget).toBe(64);
  });
});

describe("loadedModelMatchesRuntimeProfile()", () => {
  it("matches when loaded model equals launch settings profile", () => {
    const profile = resolveChatRuntimeProfile(null, launchSettings);
    expect(loadedModelMatchesRuntimeProfile(makeLoadedModel(), profile)).toBe(true);
  });

  it("returns false when the loaded model differs from launch settings", () => {
    const updatedLaunch = { ...launchSettings, cacheStrategy: "turboquant", cacheBits: 3 };
    const profile = resolveChatRuntimeProfile(null, updatedLaunch);
    expect(loadedModelMatchesRuntimeProfile(makeLoadedModel(), profile)).toBe(false);
  });
});
