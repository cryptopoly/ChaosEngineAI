/**
 * Pure helpers that build load + thread payloads from a ``ModelVariant``.
 *
 * `loadPayloadFromVariant` — turn a catalog variant into the
 * ``handleLoadModel`` payload, preferring a downloaded library entry
 * over the catalog reference when one exists.
 * `threadPatchFromVariant` — turn a catalog variant into the chat-
 * session metadata patch (model fields + cache settings) used when
 * applying a variant to an existing thread or starting a new one.
 *
 * Pulled out of ``App.tsx`` as part of the v0.8.0 Phase 2c-9 refactor.
 * Both helpers were already pure given a few deps; extracting them
 * tightens the App composition root and makes the variant flow
 * unit-testable in isolation.
 */

import type { TabId } from "../../types";
import type {
  ChatSession,
  LaunchPreferences,
  LibraryItem,
  ModelVariant,
} from "../../types";
import {
  findLibraryItemForVariant,
  libraryItemBackend,
} from "../../utils";


type SanitizeSpeculativeFn = (params: {
  backend: string;
  modelRef: string;
  canonicalRepo: string | null | undefined;
  modelName: string;
  speculativeDecoding: boolean;
  treeBudget: number;
}) => { speculativeDecoding: boolean; treeBudget: number };


export function loadPayloadFromVariant(
  variant: ModelVariant,
  chatLibrary: LibraryItem[],
  nextTab?: TabId,
) {
  const localItem = findLibraryItemForVariant(chatLibrary, variant);
  if (localItem) {
    return {
      modelRef: localItem.name,
      modelName: localItem.name,
      canonicalRepo: variant.repo,
      source: "library",
      backend: libraryItemBackend(localItem),
      path: localItem.path,
      nextTab,
    };
  }
  return {
    modelRef: variant.id,
    modelName: variant.name,
    canonicalRepo: variant.repo,
    source: "catalog",
    backend: variant.backend,
    nextTab,
  };
}


export function threadPatchFromVariant(
  variant: ModelVariant,
  deps: {
    chatLibrary: LibraryItem[];
    launchSettings: LaunchPreferences;
    launchCacheLabel: string;
    sanitizeSpeculativeForModel: SanitizeSpeculativeFn;
  },
): Pick<
  ChatSession,
  "model" | "modelRef" | "canonicalRepo" | "modelSource" | "modelPath" | "modelBackend" | "cacheLabel" | "updatedAt"
  | "cacheStrategy" | "cacheBits" | "fp16Layers" | "fusedAttention" | "fitModelInMemory"
  | "contextTokens" | "speculativeDecoding" | "dflashDraftModel" | "treeBudget"
> {
  const { chatLibrary, launchSettings, launchCacheLabel, sanitizeSpeculativeForModel } = deps;
  const localItem = findLibraryItemForVariant(chatLibrary, variant);
  const modelRef = localItem?.name ?? variant.id;
  const modelName = localItem?.name ?? variant.name;
  const modelBackend = localItem ? libraryItemBackend(localItem, variant) : variant.backend;
  const sanitizedSpeculative = sanitizeSpeculativeForModel({
    backend: modelBackend,
    modelRef,
    canonicalRepo: variant.repo,
    modelName,
    speculativeDecoding: launchSettings.speculativeDecoding,
    treeBudget: launchSettings.treeBudget,
  });
  return {
    model: modelName,
    modelRef,
    canonicalRepo: variant.repo,
    modelSource: localItem ? "library" : "catalog",
    modelPath: localItem?.path ?? null,
    modelBackend,
    cacheLabel: launchCacheLabel,
    cacheStrategy: launchSettings.cacheStrategy,
    cacheBits: launchSettings.cacheBits,
    fp16Layers: launchSettings.fp16Layers,
    fusedAttention: launchSettings.fusedAttention,
    fitModelInMemory: launchSettings.fitModelInMemory,
    contextTokens: launchSettings.contextTokens,
    speculativeDecoding: sanitizedSpeculative.speculativeDecoding,
    dflashDraftModel: null,
    treeBudget: sanitizedSpeculative.treeBudget,
    updatedAt: new Date().toLocaleString(),
  };
}
