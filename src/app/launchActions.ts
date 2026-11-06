import { checkModelMemory } from "../api";
import type { ChatSession, ModelVariant } from "../types";
import type { ChatModelOption } from "../types/chat";
import type { TabRequest } from "../constants";
import type { ModelFamily, LaunchPreferences, LibraryItem, LoadModelActionResult, WorkspaceData } from "../types";
import { findLibraryItemForVariant, findVariantForReference } from "../utils";
import { sanitizeSpeculativeSelection } from "../components/runtimeSupport";
import type { LoadModelPayload } from "./context";

export type PendingLaunch = { action: "chat" | "server" | "thread"; preselectedKey?: string };

export interface ConfirmLaunchDeps {
  pendingLaunch: PendingLaunch | null;
  threadModelOptions: ChatModelOption[];
  launchSettings: LaunchPreferences;
  previewControls: {
    paramsB: number;
    numLayers: number;
    numHeads: number;
    numKvHeads: number;
    hiddenSize: number;
  };
  featuredModels: ModelFamily[];
  activeChat: ChatSession | undefined;
  setPendingLaunch: (value: PendingLaunch | null) => void;
  setError: (msg: string | null) => void;
  setActiveTab: (request: TabRequest) => void;
  handleStartThreadWithVariant: (variant: ModelVariant) => Promise<void>;
  handleLoadModel: (payload: LoadModelPayload) => Promise<LoadModelActionResult>;
  persistSessionChanges: (sessionId: string, patch: Partial<ChatSession>) => Promise<unknown>;
  sanitizeSpeculativeForModel: (params: {
    backend: string;
    modelRef: string;
    canonicalRepo?: string | null;
    modelName: string;
    speculativeDecoding: boolean;
    treeBudget: number;
  }) => { speculativeDecoding: boolean; treeBudget: number };
}

/** Confirm the Launch modal: advisory memory check, then chat / server / new-thread load. */
export async function performConfirmLaunch(selectedKey: string, deps: ConfirmLaunchDeps) {
  const {
    pendingLaunch, threadModelOptions, launchSettings, previewControls, featuredModels, activeChat,
    setPendingLaunch, setError, setActiveTab, handleStartThreadWithVariant, handleLoadModel,
    persistSessionChanges, sanitizeSpeculativeForModel,
  } = deps;
  if (!pendingLaunch) return;
  const option = threadModelOptions.find((o) => o.key === selectedKey);
  if (!option) return;
  // Advisory memory check: ask before loading a model that may not fit,
  // never refuse. A failed check never blocks the load.
  if (option.sizeGb && option.sizeGb > 0) {
    const contextTokens = option.maxContext
      ? Math.min(launchSettings.contextTokens, option.maxContext)
      : launchSettings.contextTokens;
    const check = await checkModelMemory({
      weightsGb: option.sizeGb,
      contextTokens,
      bits: launchSettings.cacheBits,
      fp16Layers: launchSettings.fp16Layers,
      strategy: launchSettings.cacheStrategy,
      paramsB: option.paramsB ?? previewControls.paramsB,
      numLayers: previewControls.numLayers,
      numHeads: previewControls.numHeads,
      numKvHeads: previewControls.numKvHeads,
      hiddenSize: previewControls.hiddenSize,
      modelPath: option.path ?? null,
      modelRepo: option.canonicalRepo ?? (option.modelRef.includes("/") ? option.modelRef : null),
    });
    if (check?.verdict === "may_not_fit" && check.message) {
      const proceed = window.confirm(`${check.message}\n\nLoad it anyway?`);
      if (!proceed) return;
    }
  }
  const { action } = pendingLaunch;
  setPendingLaunch(null);
  setError(null);
  if (action === "chat" || action === "thread") setActiveTab("chat");
  else if (action === "server") setActiveTab("server");
  if (action === "thread") {
    const variant = findVariantForReference(featuredModels, option.modelRef, option.model);
    if (variant) { await handleStartThreadWithVariant(variant); }
    else {
      await handleLoadModel({
        modelRef: option.modelRef,
        modelName: option.model,
        canonicalRepo: option.canonicalRepo,
        source: option.source,
        backend: option.backend,
        path: option.path ?? undefined,
      });
    }
  } else if (action === "chat") {
    if (activeChat) {
      const sanitizedSpeculative = sanitizeSpeculativeForModel({
        backend: option.backend,
        modelRef: option.modelRef,
        canonicalRepo: option.canonicalRepo ?? null,
        modelName: option.model,
        speculativeDecoding: activeChat.speculativeDecoding ?? launchSettings.speculativeDecoding,
        treeBudget: activeChat.treeBudget ?? launchSettings.treeBudget,
      });
      await persistSessionChanges(activeChat.id, {
        model: option.model,
        modelRef: option.modelRef,
        canonicalRepo: option.canonicalRepo ?? null,
        modelSource: option.source,
        modelPath: option.path ?? null,
        modelBackend: option.backend,
        speculativeDecoding: sanitizedSpeculative.speculativeDecoding,
        dflashDraftModel: null,
        treeBudget: sanitizedSpeculative.treeBudget,
        updatedAt: new Date().toLocaleString(),
      });
    }
    await handleLoadModel({
      modelRef: option.modelRef,
      modelName: option.model,
      canonicalRepo: option.canonicalRepo,
      source: option.source,
      backend: option.backend,
      path: option.path ?? undefined,
    });
  } else if (action === "server") {
    await handleLoadModel({
      modelRef: option.modelRef,
      modelName: option.model,
      canonicalRepo: option.canonicalRepo,
      source: option.source,
      backend: option.backend,
      path: option.path ?? undefined,
    });
  }
}

export interface SpeculativeParams {
  backend: string;
  modelRef: string;
  canonicalRepo?: string | null;
  modelName: string;
  speculativeDecoding: boolean;
  treeBudget: number;
}

/** Drop speculative decoding settings the host / model cannot run. */
export function sanitizeSpeculativeForSystem(system: WorkspaceData["system"], params: SpeculativeParams) {
  return sanitizeSpeculativeSelection({
    dflashInfo: system.dflash,
    tensorfoldInfo: system.tensorfold,
    selectedBackend: params.backend,
    modelRef: params.modelRef,
    canonicalRepo: params.canonicalRepo ?? null,
    modelName: params.modelName,
    speculativeDecoding: params.speculativeDecoding,
    treeBudget: params.treeBudget,
  });
}

/**
 * Turn the key a launch button passes into the key the picker can preselect:
 * a catalog key becomes the matching local library key, and a missing or stale
 * key falls back to the currently loaded model.
 */
export function normalizePreselectedKey(
  preselectedKey: string | undefined,
  {
    featuredModels,
    chatLibrary,
    threadModelOptions,
    loadedModelOption,
  }: {
    featuredModels: ModelFamily[];
    chatLibrary: LibraryItem[];
    threadModelOptions: ChatModelOption[];
    loadedModelOption: ChatModelOption | null;
  },
): string | undefined {
  let normalizedKey = preselectedKey;
  if (normalizedKey?.startsWith("catalog:")) {
    const modelRef = normalizedKey.slice("catalog:".length);
    const variant = findVariantForReference(featuredModels, modelRef, undefined);
    const localItem = variant ? findLibraryItemForVariant(chatLibrary, variant) : null;
    if (localItem) normalizedKey = `library:${localItem.path}`;
  }
  // If no key given, or the key references a model no longer in the options
  // (e.g. deleted/broken), fall back to the currently loaded model.
  if ((!normalizedKey || !threadModelOptions.some((o) => o.key === normalizedKey)) && loadedModelOption) {
    normalizedKey = loadedModelOption.key;
  }
  return normalizedKey;
}
