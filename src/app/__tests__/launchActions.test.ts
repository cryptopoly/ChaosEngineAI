import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../api", () => ({ checkModelMemory: vi.fn(async () => null) }));

import { performConfirmLaunch, type ConfirmLaunchDeps } from "../launchActions";
import { threadPatchFromOption } from "../../features/app/variantPayloads";
import type { ChatModelOption } from "../../types/chat";
import type { LaunchPreferences, ModelFamily, ModelVariant } from "../../types";

const launchSettings = {
  cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, fusedAttention: false, fitModelInMemory: true,
  contextTokens: 8192, speculativeDecoding: false, treeBudget: 0,
} as unknown as LaunchPreferences;

const local: ChatModelOption = {
  key: "library:/cache/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit",
  label: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
  detail: "MLX / 0.3 GB",
  group: "Local library",
  model: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
  modelRef: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
  canonicalRepo: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
  source: "library",
  path: "/cache/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit",
  backend: "mlx",
} as ChatModelOption;

function deps(extra: Partial<ConfirmLaunchDeps> = {}) {
  const calls: string[] = [];
  const all = {
    pendingLaunch: { action: "thread", preselectedKey: local.key },
    threadModelOptions: [local],
    launchSettings,
    previewControls: { paramsB: 0.5, numLayers: 24, numHeads: 14, numKvHeads: 2, hiddenSize: 896 },
    featuredModels: [] as ModelFamily[],
    activeChat: undefined,
    setPendingLaunch: vi.fn(),
    setError: vi.fn(),
    setActiveTab: vi.fn(),
    handleStartThreadWithVariant: vi.fn(async () => { calls.push("thread:variant"); }),
    handleStartThreadWithOption: vi.fn(async () => { calls.push("thread:option"); }),
    handleLoadModel: vi.fn(async () => { calls.push("load"); return { ok: true }; }),
    persistSessionChanges: vi.fn(async () => undefined),
    sanitizeSpeculativeForModel: () => ({ speculativeDecoding: false, treeBudget: 0 }),
    ...extra,
  } as unknown as ConfirmLaunchDeps;
  return { all, calls };
}

describe("performConfirmLaunch, Chat in a new thread", () => {
  beforeEach(() => vi.clearAllMocks());

  it("a local model that is not a catalogue row gets a new thread first, then loads", async () => {
    const { all, calls } = deps();
    await performConfirmLaunch(local.key, all);
    expect(calls).toEqual(["thread:option", "load"]);
    expect(all.handleStartThreadWithOption).toHaveBeenCalledWith(local);
    expect(all.handleLoadModel).toHaveBeenCalledWith(expect.objectContaining({
      modelRef: local.modelRef, canonicalRepo: local.canonicalRepo, path: local.path, backend: "mlx", source: "library",
    }));
    expect(all.setActiveTab).toHaveBeenCalledWith("chat");
  });

  it("a catalogue row still starts its thread from the variant", async () => {
    const variant = { id: local.modelRef, repo: local.modelRef, name: "Qwen2.5 0.5B", familyId: "qwen" } as unknown as ModelVariant;
    const family = { id: "qwen", name: "Qwen", variants: [variant] } as unknown as ModelFamily;
    const { all, calls } = deps({ featuredModels: [family] });
    await performConfirmLaunch(local.key, all);
    expect(calls).toEqual(["thread:variant"]);
  });
});

describe("threadPatchFromOption", () => {
  it("binds the thread to the exact local model the picker named", () => {
    const patch = threadPatchFromOption(local, {
      launchSettings,
      launchCacheLabel: "Native f16",
      sanitizeSpeculativeForModel: () => ({ speculativeDecoding: false, treeBudget: 0 }),
    });
    expect(patch).toMatchObject({
      model: local.model, modelRef: local.modelRef, canonicalRepo: local.canonicalRepo, modelSource: "library",
      modelPath: local.path, modelBackend: "mlx", cacheLabel: "Native f16", cacheStrategy: "native", contextTokens: 8192,
      dflashDraftModel: null,
    });
  });
});
