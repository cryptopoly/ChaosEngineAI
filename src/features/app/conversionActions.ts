/**
 * Model conversion flow handlers used by the App composition root.
 *
 * Pulls four handlers (prepareCatalogConversion, prepareLibraryConversion,
 * performConvertModel, pickConversionOutputDir) out of App.tsx so the
 * root file stays focused on top-level state wiring + render.
 *
 * Extracted as part of the v0.8.0 Phase 2c-10 refactor.
 */

import type { Dispatch, SetStateAction } from "react";

import { convertModel as convertModelApi } from "../../api";
import {
  findLibraryItemForVariant,
  libraryItemFormat,
  syncRuntime,
} from "../../utils";
import type {
  ConversionResult,
  LibraryItem,
  ModelVariant,
  TabId,
  WorkspaceData,
} from "../../types";


type ConversionDraft = {
  modelRef: string;
  path: string;
  hfRepo: string;
  outputPath: string;
  quantize: boolean;
  qBits: number;
  qGroupSize: number;
  dtype: string;
};

type SetLastConversion = Dispatch<SetStateAction<ConversionResult | null>>;


export function prepareCatalogConversion(
  model: ModelVariant,
  deps: {
    convertibleLibrary: LibraryItem[];
    setConversionDraft: (draft: ConversionDraft) => void;
    setLastConversion: SetLastConversion;
    setActiveTab: (tab: TabId) => void;
  },
): void {
  const matchingItem = findLibraryItemForVariant(deps.convertibleLibrary, model);
  if (matchingItem) {
    prepareLibraryConversion(matchingItem, undefined, {
      setConversionDraft: deps.setConversionDraft,
      setLastConversion: deps.setLastConversion,
      setActiveTab: deps.setActiveTab,
    });
    return;
  }
  deps.setActiveTab("conversion");
  deps.setLastConversion(null);
}


export function prepareLibraryConversion(
  item: LibraryItem,
  resolvedPath: string | undefined,
  deps: {
    setConversionDraft: (draft: ConversionDraft) => void;
    setLastConversion: SetLastConversion;
    setActiveTab: (tab: TabId) => void;
  },
): void {
  const isGguf = libraryItemFormat(item).toUpperCase() === "GGUF";
  deps.setConversionDraft({
    modelRef: item.name,
    path: resolvedPath ?? item.path,
    hfRepo: isGguf ? "" : item.name,
    outputPath: "",
    quantize: true,
    qBits: 4,
    qGroupSize: 64,
    dtype: "float16",
  });
  deps.setLastConversion(null);
  deps.setActiveTab("conversion");
}


export async function performConvertModel(deps: {
  conversionDraft: ConversionDraft;
  setError: (msg: string | null) => void;
  setBusyAction: (label: string | null) => void;
  setConversionStartedAt: (value: number | null) => void;
  setConversionError: (msg: string | null) => void;
  setShowConversionModal: (visible: boolean) => void;
  setLastConversion: SetLastConversion;
  setWorkspace: (updater: (current: WorkspaceData) => WorkspaceData) => void;
  refreshWorkspace: (preferredChatId?: string) => Promise<unknown>;
  activeChatId: string;
}): Promise<void> {
  const {
    conversionDraft,
    setError,
    setBusyAction,
    setConversionStartedAt,
    setConversionError,
    setShowConversionModal,
    setLastConversion,
    setWorkspace,
    refreshWorkspace,
    activeChatId,
  } = deps;
  const modelRef = conversionDraft.modelRef.trim();
  const path = conversionDraft.path.trim();
  const hfRepo = conversionDraft.hfRepo.trim();
  const outputPath = conversionDraft.outputPath.trim();
  if (!modelRef && !path) {
    setError("Enter a model reference or a local path before starting conversion.");
    return;
  }
  setBusyAction("Converting model...");
  setConversionStartedAt(Date.now());
  setConversionError(null);
  setShowConversionModal(true);
  try {
    const response = await convertModelApi({
      modelRef: modelRef || undefined,
      path: path || undefined,
      hfRepo: hfRepo || undefined,
      outputPath: outputPath || undefined,
      quantize: conversionDraft.quantize,
      qBits: conversionDraft.qBits,
      qGroupSize: conversionDraft.qGroupSize,
      dtype: conversionDraft.dtype,
    });
    setLastConversion(response.conversion);
    setWorkspace((current) =>
      syncRuntime({ ...current, library: response.library }, response.runtime),
    );
    await refreshWorkspace(activeChatId || undefined);
  } catch (actionError) {
    const message = actionError instanceof Error ? actionError.message : "Failed to convert model.";
    setError(message);
    setConversionError(message);
  } finally {
    setBusyAction(null);
    setConversionStartedAt(null);
  }
}


export async function pickConversionOutputDir(deps: {
  conversionSource: { name?: string } | null;
  setError: (msg: string | null) => void;
  updateConversionDraft: (field: "outputPath", value: string) => void;
}): Promise<void> {
  const { conversionSource, setError, updateConversionDraft } = deps;
  try {
    const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
    const picked = await tauriInvoke<string | null>("pick_directory");
    if (picked) {
      const suggested = conversionSource?.name
        ? `${picked.replace(/\/$/, "")}/${conversionSource.name.replace(/[^\w.-]/g, "-")}-mlx`
        : picked;
      updateConversionDraft("outputPath", suggested);
    }
  } catch (err) {
    setError(err instanceof Error ? err.message : "Could not open the directory picker.");
  }
}
